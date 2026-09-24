"""Reject memory: what deleted rejects looked like, so the triage model keeps
learning "reject" after the files are gone.

Rejects are rated 1★ and then deleted with their files, so without this the
reject class would only ever hold the few scenes waiting to be deleted. Before
Peaks deletes a reject (and whenever it sees a 1★ scene that's embedded) it
stores that scene's triage features — visual summary + file quality, a few KB —
keyed by file fingerprint.

One file per embedding model (``reject_memory-<model>.npz``): features from a
different backbone aren't comparable, so switching models simply starts a new
memory instead of mixing the two.
"""

from __future__ import annotations

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
        data = {"fp": [], "vis": None, "qual": None, "ts": []}
        if mtime is not None:
            try:
                with np.load(self.path, allow_pickle=False) as z:
                    data = {"fp": [str(x) for x in z["fp"]], "vis": z["vis"], "qual": z["qual"],
                            "ts": [float(x) for x in z["ts"]]}
            except (OSError, ValueError, KeyError):
                pass            # unreadable → start over rather than break triage
        self._data, self._mtime = data, mtime
        return data

    def fingerprints(self) -> set[str]:
        return set(self._load()["fp"])

    def __len__(self) -> int:
        return len(self._load()["fp"])

    def add(self, items: dict[str, tuple[np.ndarray, np.ndarray]]) -> int:
        """Remember {fingerprint: (visual, quality)} for scenes not already
        stored. Feature sizes must match what's stored (same model). Returns how
        many were added."""
        data = self._load()
        have = set(data["fp"])
        new = [(fp, v, q) for fp, (v, q) in items.items() if fp and fp not in have]
        if not new:
            return 0
        vis = np.stack([np.asarray(v, np.float32) for _, v, _ in new])
        qual = np.stack([np.asarray(q, np.float32) for _, _, q in new])
        if data["vis"] is not None and len(data["fp"]):
            if data["vis"].shape[1] != vis.shape[1] or data["qual"].shape[1] != qual.shape[1]:
                raise ValueError("reject memory feature size changed — different model?")
            vis = np.concatenate([data["vis"], vis])
            qual = np.concatenate([data["qual"], qual])
        fps = data["fp"] + [fp for fp, _, _ in new]
        ts = data["ts"] + [time.time()] * len(new)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, fp=np.array(fps), vis=vis, qual=qual, ts=np.array(ts))
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
