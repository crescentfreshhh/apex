"""The shrink trendline: how big the library is, over time, and what's been freed.

A snapshot a day ({date, scenes, bytes, tiers}) goes to state/library_trend.jsonl,
written the first time the library summary is read on a new day. Before the
first snapshot there's no record, so the past is reconstructed: on day D the
library held every scene still here that was added by D, plus every scene
deleted after D (from the action log, with its size). Reconstructed points are
marked as such. GB freed per month comes straight from the delete log.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path


class TrendMixin:
    def _trend_path(self) -> Path:
        return self._state_dir() / "library_trend.jsonl"

    def _trend_read(self) -> list[dict]:
        try:
            lines = self._trend_path().read_text().splitlines()
        except OSError:
            return []
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
        return out

    def _trend_snapshot(self, rows: list[dict], today: str | None = None) -> bool:
        """Append today's point once a day. True when one was written."""
        today = today or date.today().isoformat()
        if self.__dict__.get("_trend_day") == today:
            return False
        pts = self._trend_read()
        if pts and pts[-1].get("d") == today:
            self._trend_day = today
            return False
        tiers: dict[str, int] = {}
        for r in rows:
            tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
        p = {"d": today, "scenes": len(rows), "bytes": sum(int(r.get("size") or 0) for r in rows),
             "tiers": tiers}
        path = self._trend_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(p) + "\n")
        self._trend_day = today
        return True

    def _trend_deletes(self) -> list[dict]:
        """Every file deletion from the action log: {d, bytes}, oldest first."""
        out = []
        path = self.action_log().path
        try:
            fh = path.open(encoding="utf-8")
        except OSError:
            return out
        with fh:
            for ln in fh:
                if '"delete"' not in ln:
                    continue
                try:
                    e = json.loads(ln)
                except ValueError:
                    continue
                if e.get("action") == "delete" and e.get("file_deleted", True):
                    out.append({"d": str(e.get("ts", ""))[:10], "bytes": int(e.get("size") or 0)})
        return out

    def library_trend(self, days: int = 365) -> dict:
        rows = self._catalogue_all()
        self._trend_snapshot(rows)
        dels = self._trend_deletes()
        recorded = {p["d"]: p for p in self._trend_read()}
        today = date.today()
        start = today - timedelta(days=days - 1)
        first_rec = min(recorded) if recorded else today.isoformat()
        added = sorted(((r.get("created_at") or "")[:10], int(r.get("size") or 0)) for r in rows)
        pts = []
        for i in range(days):
            d = (start + timedelta(days=i)).isoformat()
            if d in recorded:
                p = recorded[d]
                pts.append({"d": d, "scenes": p["scenes"], "gb": round(p["bytes"] / 1e9, 1), "rec": False})
                continue
            if d > first_rec:
                continue                      # a gap after recording began: leave it out
            here = [b for c, b in added if c and c <= d]
            gone = [x["bytes"] for x in dels if x["d"] > d]
            if not here and not gone:
                continue
            pts.append({"d": d, "scenes": len(here) + len(gone),
                        "gb": round((sum(here) + sum(gone)) / 1e9, 1), "rec": True})
        months: dict[str, dict] = {}
        for x in dels:
            m = months.setdefault(x["d"][:7], {"month": x["d"][:7], "scenes": 0, "gb": 0.0})
            m["scenes"] += 1
            m["gb"] = round(m["gb"] + x["bytes"] / 1e9, 2)
        this = today.isoformat()[:7]
        return {"points": pts, "freed": sorted(months.values(), key=lambda m: m["month"]),
                "now": {"scenes": len(rows), "gb": round(sum(int(r.get("size") or 0) for r in rows) / 1e9, 1)},
                "month_freed_gb": (months.get(this) or {}).get("gb", 0.0),
                "recorded_since": first_rec if recorded else None, "at": time.time()}
