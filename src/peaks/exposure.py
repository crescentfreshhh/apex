"""Megaboard exposure: which scenes the board keeps showing you, and whether you
ever respond to them.

A *showing* is a clip that actually played (~5 s, board running, tab visible),
weighted by grid size — on a 4×4 board you can't respond to all 16 tiles, so a
showing there counts ¼. A *response* is anything that says "yes, this one":
saving a moment, 👍, "more like this"/scene/performer, pinning, enlarging, a
keeper grade. 👎 and Reject are *negative* responses — they don't clear
anything.

A scene shown a lot, over several days, that never drew a response is "often
passed over" — a reason (among others) to suggest letting it go. Grading a
scene starts its record over: your verdict is newer than the board's evidence.

One small JSON file (`models/board_exposure.json`), a row per scene seen:
{days: {YYYY-MM-DD: weighted showings}, pos, neg, last_seen, last_pos}.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

PASSED_SHOWINGS = 8.0   # weighted showings…
PASSED_DAYS = 3         # …spread over at least this many different days
MAX_DAYS = 120          # per scene, the most recent days kept


class ExposureStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._rows: dict[str, dict] | None = None
        self._mtime: float | None = None

    def _load(self) -> dict[str, dict]:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            mtime = None
        if self._rows is None or mtime != self._mtime:     # re-read if written elsewhere
            try:
                self._rows = json.loads(self.path.read_text()).get("scenes", {})
            except (OSError, ValueError, AttributeError):
                self._rows = {}
            self._mtime = mtime
        return self._rows

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"scenes": self._rows}, separators=(",", ":")))
        os.replace(tmp, self.path)
        self._mtime = self.path.stat().st_mtime

    def record(self, shows: dict | None = None, pos: dict | None = None, neg: dict | None = None,
               now: float | None = None) -> int:
        """Add a batch from the board: {scene_id: weighted showings}, and
        {scene_id: count} of positive / negative responses. Returns scenes touched."""
        now = time.time() if now is None else now
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        touched = set()
        with self._lock:
            rows = self._load()
            for sid, w in (shows or {}).items():
                w = float(w)
                if not (0 < w <= 50):          # a batch is ~30 s of one board
                    continue
                r = rows.setdefault(str(sid), {"days": {}, "pos": 0, "neg": 0})
                r["days"][day] = round(r["days"].get(day, 0.0) + w, 3)
                if len(r["days"]) > MAX_DAYS:
                    for d in sorted(r["days"])[: len(r["days"]) - MAX_DAYS]:
                        del r["days"][d]
                r["last_seen"] = now
                touched.add(str(sid))
            for key, src in (("pos", pos), ("neg", neg)):
                for sid, n in (src or {}).items():
                    n = int(n)
                    if n <= 0:
                        continue
                    r = rows.setdefault(str(sid), {"days": {}, "pos": 0, "neg": 0})
                    r[key] = r.get(key, 0) + n
                    if key == "pos":
                        r["last_pos"] = now
                    touched.add(str(sid))
            if touched:
                self._save()
        return len(touched)

    def summary(self, sid: str) -> dict:
        """{showings, days, pos, neg, passed} for one scene (zeros if never shown)."""
        r = self._load().get(str(sid))
        if not r:
            return {"showings": 0.0, "days": 0, "pos": 0, "neg": 0, "passed": False}
        showings = round(sum(r["days"].values()), 2)
        days = len(r["days"])
        pos = int(r.get("pos", 0))
        return {"showings": showings, "days": days, "pos": pos, "neg": int(r.get("neg", 0)),
                "passed": showings >= PASSED_SHOWINGS and days >= PASSED_DAYS and pos == 0}

    def all(self) -> dict[str, dict]:
        return {sid: self.summary(sid) for sid in list(self._load())}

    def reset(self, sid: str) -> dict | None:
        """A grade starts the record over; returns the old row (for undo)."""
        with self._lock:
            old = self._load().pop(str(sid), None)
            if old is not None:
                self._save()
        return old

    def restore(self, sid: str, row: dict | None) -> None:
        if row is None:
            return
        with self._lock:
            self._load()[str(sid)] = row
            self._save()

    def drop(self, sids) -> None:
        with self._lock:
            rows = self._load()
            gone = [s for s in map(str, sids) if s in rows]
            for s in gone:
                del rows[s]
            if gone:
                self._save()
