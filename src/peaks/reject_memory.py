"""Reject memory: what deleted rejects looked like, so the triage model keeps
learning "reject" after the files are gone.

Rejects are rated 1★ and then deleted with their files, so without this the
reject class would only ever hold the few scenes waiting to be deleted. Before
Peaks deletes a reject (and whenever it sees a 1★ scene that's embedded) it
stores that scene's triage features — visual summary + file quality, a few KB —
keyed by file fingerprint, plus who was in it (performer ids, studio) so a
deleted reject still counts against those track records. Entries stored before
that was kept have no performers/studio (None) and simply don't count there.

One file per embedding model (``reject_memory-<model>.npz``): features from a
different backbone aren't comparable, so switching models simply starts a new
memory instead of mixing the two.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


class RejectMemory:
    def __init__(self, folder: str | Path, model: str):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in model)
        self.path = Path(folder) / f"reject_memory-{safe}.npz"
        self._data: dict | None = None
        self._mtime: float | None = None

    def _load(self) -> dict:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            mtime = None
        if self._data is not None and mtime == self._mtime:
            return self._data
        data = {"fp": [], "vis": None, "qual": None, "ts": [], "who": []}
        if mtime is not None:
            try:
                with np.load(self.path, allow_pickle=False) as z:
                    fps = [str(x) for x in z["fp"]]
                    who = ([json.loads(x) if x else None for x in (str(v) for v in z["who"])]
                           if "who" in z.files else [None] * len(fps))
                    data = {"fp": fps, "vis": z["vis"], "qual": z["qual"],
                            "ts": [float(x) for x in z["ts"]], "who": who}
            except (OSError, ValueError, KeyError):
                pass            # unreadable → start over rather than break triage
        self._data, self._mtime = data, mtime
        return data

    def fingerprints(self) -> set[str]:
        return set(self._load()["fp"])

    def __len__(self) -> int:
        return len(self._load()["fp"])

    def add(self, items: dict[str, tuple]) -> int:
        """Remember {fingerprint: (visual, quality[, who])} for scenes not
        already stored — `who` is {performers: [ids], studio}. Feature sizes
        must match what's stored (same model). Returns how many were added."""
        data = self._load()
        have = set(data["fp"])
        new = [(fp, it[0], it[1], it[2] if len(it) > 2 else None)
               for fp, it in items.items() if fp and fp not in have]
        if not new:
            return 0
        vis = np.stack([np.asarray(n[1], np.float32) for n in new])
        qual = np.stack([np.asarray(n[2], np.float32) for n in new])
        if data["vis"] is not None and len(data["fp"]):
            if data["vis"].shape[1] != vis.shape[1] or data["qual"].shape[1] != qual.shape[1]:
                raise ValueError("reject memory feature size changed — different model?")
            vis = np.concatenate([data["vis"], vis])
            qual = np.concatenate([data["qual"], qual])
        fps = data["fp"] + [n[0] for n in new]
        ts = data["ts"] + [time.time()] * len(new)
        who = data["who"] + [n[3] for n in new]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, fp=np.array(fps), vis=vis, qual=qual, ts=np.array(ts),
                 who=np.array([json.dumps(w) if w is not None else "" for w in who]))
        os.replace(tmp, self.path)
        self._data = None
        return len(new)

    def rows(self, exclude: set[str] | None = None) -> tuple[list[str], np.ndarray | None, np.ndarray | None]:
        """(fingerprints, visual, quality) of remembered rejects, minus any in
        `exclude` (scenes still in the library, which train from live data)."""
        data = self._load()
        if not data["fp"]:
            return [], None, None
        keep = [i for i, fp in enumerate(data["fp"]) if not exclude or fp not in exclude]
        if not keep:
            return [], None, None
        return [data["fp"][i] for i in keep], data["vis"][keep], data["qual"][keep]

    def fill_who(self, items: dict[str, dict]) -> int:
        """Add performers/studio to remembered rejects stored without them
        (still in the library, so still knowable). Returns how many changed."""
        data = self._load()
        who = list(data["who"])
        n = 0
        for i, fp in enumerate(data["fp"]):
            if who[i] is None and items.get(fp) is not None:
                who[i] = items[fp]
                n += 1
        if n:
            tmp = self.path.with_name(self.path.name + ".tmp.npz")
            np.savez(tmp, fp=np.array(data["fp"]), vis=data["vis"], qual=data["qual"],
                     ts=np.array(data["ts"]),
                     who=np.array([json.dumps(w) if w is not None else "" for w in who]))
            os.replace(tmp, self.path)
            self._data = None
        return n

    def who(self, fps: list[str]) -> list[dict | None]:
        """Who was in each remembered reject (None when stored before that was kept)."""
        data = self._load()
        at = dict(zip(data["fp"], data["who"]))
        return [at.get(fp) for fp in fps]
