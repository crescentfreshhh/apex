"""Performer review: one performer at a time — find her (or let Peaks pick
someone due for a look), read a lead-in on her repertoire, then work through
her scenes best first and prune / re-grade.

Built on what Peaks already knows: the performer directory (library counts,
tiers, track record, taste rank), `who_of` (scene → Stash performer ids), the
suggestion lists and verdicts, and the scene signals. Each finished review is
remembered (`performer_reviews.json`) so the random pick favours performers
you haven't looked at in a while.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

MIN_SCENES = 3            # random only picks performers with at least this many scenes
RECENT_DAYS = 7           # …and not ones reviewed this recently (unless that's everyone)
NEVER_DAYS = 365          # "never reviewed" weighs like a year ago
TIER_ORDER = ("legendaire", "exceptionnelle", "merveilleuse", "upscale", "anomaly", "unreviewed", "rejected")
LOW_TIERS = ("unreviewed", "upscale")


class PerfReviewMixin:
    # --- memory of past reviews -------------------------------------------------

    def _pr_path(self) -> Path:
        return self._state_dir() / "performer_reviews.json"

    def _pr_state(self) -> dict:
        try:
            return json.loads(self._pr_path().read_text())
        except (OSError, ValueError):
            return {}

    def _pr_save(self, d: dict) -> None:
        p = self._pr_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, p)

    @staticmethod
    def _pr_days(last_at: float | None, now: float) -> int | None:
        return None if not last_at else max(0, int((now - float(last_at)) // 86400))

    # --- who's in the library ------------------------------------------------------

    def _pr_rows(self) -> dict[str, list[dict]]:
        """Performer id → her catalogue rows (all tiers)."""
        from ..tier_model import who_of

        out: dict[str, list[dict]] = {}
        for r in self._catalogue_all():
            for pid in who_of(r)["performers"]:
                out.setdefault(pid, []).append(r)
        return out

    def perf_review_list(self, q: str | None = None, limit: int = 60, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        rows = self._pr_rows()
        info = {p["id"]: p for p in self._stash_performers()}
        past = self._pr_state()
        needle = self._fold(q or "")
        out = []
        for pid, rs in rows.items():
            p = info.get(pid) or {}
            name = p.get("name") or next((n for r in rs for i, n in zip(r.get("performer_ids") or [], r.get("performers") or [])
                                          if str(i) == pid), "") or f"Performer {pid}"
            aliases = p.get("aliases") or []
            if needle and not any(needle in self._fold(x) for x in [name, *aliases]):
                continue
            tiers: dict[str, int] = {}
            for r in rs:
                tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
            last = (past.get(pid) or {}).get("last_at")
            out.append({"id": pid, "name": name, "aliases": aliases,
                        "scenes": sum(v for k, v in tiers.items() if k != "rejected"),
                        "rejected": tiers.get("rejected", 0), "tiers": tiers,
                        "reviewed_days": self._pr_days(last, now),
                        "photo": self.performer_photo_version(pid)})
        out.sort(key=lambda x: (-x["scenes"], self._fold(x["name"])))
        return {"performers": out[:limit], "total": len(out)}

    def perf_review_random(self, exclude: str | None = None, now: float | None = None,
                           rng: random.Random | None = None) -> dict | None:
        """Someone due for a review: ≥MIN_SCENES scenes, not reviewed in the last
        RECENT_DAYS (unless that's everyone), weighted by time since the last
        review × log(1 + scenes)."""
        now = time.time() if now is None else now
        rng = rng or random.Random()
        cands = [p for p in self.perf_review_list(limit=100_000, now=now)["performers"]
                 if p["scenes"] >= MIN_SCENES and p["id"] != str(exclude or "")]
        if not cands:
            return None
        due = [p for p in cands if p["reviewed_days"] is None or p["reviewed_days"] >= RECENT_DAYS]
        pool = due or cands

        def weight(p):
            days = NEVER_DAYS if p["reviewed_days"] is None else min(NEVER_DAYS, max(1, p["reviewed_days"]))
            return days * math.log1p(p["scenes"])

        return rng.choices(pool, weights=[weight(p) for p in pool], k=1)[0]

    # --- the lead-in --------------------------------------------------------------------

    def perf_review_brief(self, pid: str, now: float | None = None) -> dict:
        import datetime as _dt

        from ..words import age as age_words
        from ..words import share

        now = time.time() if now is None else now
        pid = str(pid)
        all_rows = self._catalogue_all()
        mine = self._pr_rows().get(pid) or []
        if not mine:
            raise LookupError(f"no scenes for performer {pid} in the library")
        live = [r for r in mine if r["tier"] != "rejected"]
        prof = self.performer_profile(pid)

        def counts(rs):
            c: dict[str, int] = {}
            for r in rs:
                c[r["tier"]] = c.get(r["tier"], 0) + 1
            return c

        her, lib = counts(live), counts([r for r in all_rows if r["tier"] != "rejected"])
        n_her, n_lib = max(1, len(live)), max(1, sum(lib.values()))
        shape = [{"tier": t, "n": her.get(t, 0), "pct": round(100 * her.get(t, 0) / n_her, 1),
                  "lib_pct": round(100 * lib.get(t, 0) / n_lib, 1)}
                 for t in TIER_ORDER if t != "rejected" and (her.get(t) or lib.get(t))]

        dates = sorted(r["date"] for r in live if r.get("date"))
        newest_age = None
        try:
            newest = max((r.get("created_at") or "") for r in live)
            t = _dt.datetime.fromisoformat(newest.replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=_dt.timezone.utc)
            newest_age = age_words(max(0, (_dt.datetime.fromtimestamp(now, _dt.timezone.utc) - t).days))
        except (ValueError, TypeError):
            pass

        # taste: saves and the best moment, from the scene signals
        signals = {}
        try:   # over the whole library: it also sets the library-wide percentiles suggestions use
            signals = self._scene_signals(all_rows)
        except Exception:  # noqa: BLE001 — no index / Stash down: skip the taste half
            signals = {}
        saves = sum((signals.get(r["scene_id"]) or {}).get("saves", 0) for r in live)
        rank = {t: i for i, t in enumerate(TIER_ORDER)}
        best = min(live, key=lambda r: (rank.get(r["tier"], 9), -(signals.get(r["scene_id"]) or {}).get("saves", 0),
                                        r.get("date") or ""), default=None)
        taste = None
        try:
            entry = next((p for p in self.performer_directory()["performers"] if p["id"] == pid), None)
            taste = (entry or {}).get("taste")
        except Exception:  # noqa: BLE001
            taste = None

        # open questions on her scenes (answered suggestions don't count)
        open_q = {"promote": 0, "second": 0, "trim": 0, "reject": 0, "unreviewed": her.get("unreviewed", 0)}
        try:
            from ..tier_model import quality_floor

            preds = self._tier_predictions(all_rows) or {}
            floor = quality_floor(all_rows)
            ids = {r["scene_id"] for r in live}
            for v in ("promote", "second", "trim"):
                open_q[v] = sum(1 for r in self._triage(v, live, preds, floor, signals) if r["scene_id"] in ids)
            open_q["reject"] = sum(1 for r in live if (self._recommend(r, preds.get(r["scene_id"]),
                                                                       signals.get(r["scene_id"])) or {}).get("grade") == "reject")
        except Exception:  # noqa: BLE001 — no model yet: just the unreviewed count
            pass

        gb = lambda rs: round(sum(int(r.get("size") or 0) for r in rs) / 1e9, 1)  # noqa: E731
        hi_res = sum(1 for r in live if (r.get("quality") or {}).get("res") in ("4K", "5K", "6K", "7K", "8K"))
        studios = (prof.get("studios") or [])[:3]
        last = self._pr_state().get(pid)
        return {
            "id": pid, "name": prof.get("name") or next((p["name"] for p in self.perf_review_list(limit=100_000)["performers"]
                                                          if p["id"] == pid), ""),
            "aliases": prof.get("aliases") or [], "fav": prof.get("fav", False), "photo": prof.get("photo"),
            "repertoire": {"scenes": len(live), "rejected": len(mine) - len(live),
                           "first": dates[0] if dates else None, "latest": dates[-1] if dates else None,
                           "newest_added": newest_age, "studios": studios},
            "record": prof.get("record"),
            "shape": shape,
            "taste": {"rank": taste, "saves": saves,
                      "best": {"scene_id": best["scene_id"], "title": best["title"], "tier": best["tier"]} if best else None},
            "open": open_q,
            "footprint": {"gb": gb(live), "low_gb": gb([r for r in live if r["tier"] in LOW_TIERS]),
                          "hi_res_share": share(hi_res / max(1, len(live))), "hi_res": hi_res},
            "last_review": ({**last, "days": self._pr_days(last.get("last_at"), now)} if last else None),
        }

    # --- a finished review ---------------------------------------------------------------

    def perf_review_done(self, pid: str, summary: dict | None = None, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        s = {k: int((summary or {}).get(k) or 0) for k in ("seen", "regraded", "rejected", "kept")}
        s["freed_bytes"] = int((summary or {}).get("freed_bytes") or 0)
        d = self._pr_state()
        prev = d.get(str(pid)) or {}
        d[str(pid)] = {**s, "last_at": now, "times": int(prev.get("times") or 0) + 1}
        self._pr_save(d)
        return d[str(pid)]
