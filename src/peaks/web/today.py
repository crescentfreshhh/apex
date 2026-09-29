"""For You as a daily curation page: one decision at a time, from everything
Peaks already knows how to suggest.

Jobs (each a queue built from the Catalogue's smart views / recommendations):
  saved    scenes you saved moments in → Légendaire            (quick confirm)
  reject   Peaks suggests letting it go (passed over, weakest)  (quick confirm)
  new      unreviewed scenes, likely keepers first — grade them
  promote  graded Merveilleuse the model rates higher
  second   graded Exceptionnelle/Légendaire the model rates lower
  anomaly  5★ with an unusual O-count
  trim     past the grace period, no saved moments — keep or let go
Upkeep (duplicates, tag conflicts, quality check, rejects awaiting delete) and
"teach Peaks" live on the task board / as check-ins in the browser.

Today's set is fixed for the day (models/today.json): a mix of quick
confirmations and real decisions, most useful first. A scene graded anywhere
else counts as done; skipping moves it to the back.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

JOBS = ("saved", "reject", "new", "promote", "second", "anomaly", "trim")
JOB_LABEL = {
    "saved": "Saved → Légendaire", "reject": "Suggested to let go", "new": "Grade new scenes",
    "promote": "Promotion candidates", "second": "Second look", "anomaly": "Anomalies",
    "trim": "No saved moments",
}
# quick one-key confirmations interleaved with real decisions, so it never drags
MIX = ("saved", "new", "reject", "new", "promote", "saved", "second", "reject", "new",
       "anomaly", "trim", "new")
DEFAULT_GOAL = 20


class TodayMixin:

    # --- candidates per job ------------------------------------------------------

    def _today_context(self) -> dict:
        from ..tier_model import quality_floor

        rows = self._catalogue_all()
        preds = self._tier_predictions(rows)
        sig = self._scene_signals(rows)
        return {"rows": rows, "by": {r["scene_id"]: r for r in rows}, "preds": preds, "sig": sig,
                "floor": quality_floor(rows), "names": self.tier_display_names()}

    def _job_queue(self, job: str, ctx: dict) -> list[dict]:
        """[{scene_id, pick, why, detail}] for one job, most valuable first."""
        rows, preds, sig, floor, names = ctx["rows"], ctx["preds"], ctx["sig"], ctx["floor"], ctx["names"]
        out = []
        if job == "saved":
            for r in self._triage("saved", rows, preds, floor, sig):
                rec = self._recommend(r, preds.get(r["scene_id"]), sig.get(r["scene_id"]), names) or {}
                out.append({"scene_id": r["scene_id"], "pick": "legendaire", **_why(rec)})
        elif job == "reject":
            cand = []
            for r in rows:
                if r["tier"] in ("legendaire", "rejected"):
                    continue
                s = sig.get(r["scene_id"]) or {}
                rec = self._recommend(r, preds.get(r["scene_id"]), s, names)
                if rec and rec.get("grade") == "reject":
                    cand.append((not s.get("passed"), -(s.get("showings") or 0), s.get("best") or 0, r, rec))
            cand.sort(key=lambda c: c[:3])
            out = [{"scene_id": c[3]["scene_id"], "pick": "reject", **_why(c[4])} for c in cand]
        elif job == "new":
            likely = self._triage("likely", rows, preds, floor, sig)
            seen = {r["scene_id"] for r in likely}
            rest = [r for r in rows if r["tier"] == "unreviewed" and r["scene_id"] not in seen]
            for r in likely + rest:
                p = preds.get(r["scene_id"])
                pick = p["tier"] if p else None
                if pick == "reject" and (sig.get(r["scene_id"]) or {}).get("new"):
                    pick = None                         # the grace period: never pushed toward Reject
                rec = self._recommend(r, p, sig.get(r["scene_id"]), names) or {}
                if rec.get("grade") and not (rec["grade"] == "reject" and (sig.get(r["scene_id"]) or {}).get("new")):
                    pick = rec["grade"]
                out.append({"scene_id": r["scene_id"], "pick": pick, **_why(rec)})
        elif job in ("promote", "second"):
            for r in self._triage(job, rows, preds, floor, sig):
                out.append({"scene_id": r["scene_id"], "pick": preds[r["scene_id"]]["tier"], "why": "", "detail": ""})
        elif job == "anomaly":
            for r in self._triage("anomaly", rows, preds, floor, sig):
                rec = self._suggest_for_anomaly(r, preds.get(r["scene_id"]), names) or {}
                out.append({"scene_id": r["scene_id"], "pick": rec.get("grade"), **_why(rec)})
        elif job == "trim":
            for r in self._triage("trim", rows, preds, floor, sig):
                rec = self._recommend(r, preds.get(r["scene_id"]), sig.get(r["scene_id"]), names) or {}
                out.append({"scene_id": r["scene_id"], "pick": rec.get("grade"), **_why(rec)})
        return out

    # --- today's set, fixed for the day ------------------------------------------------

    def _today_path(self) -> Path:
        return Path(self.cfg.modeling.dir) / "today.json"

    def today_goal(self) -> int:
        try:
            return max(1, int(self._settings().get("today_goal", DEFAULT_GOAL)))
        except (TypeError, ValueError):
            return DEFAULT_GOAL

    def _load_today(self) -> dict | None:
        try:
            d = json.loads(self._today_path().read_text())
        except (OSError, ValueError):
            return None
        return d if d.get("date") == time.strftime("%Y-%m-%d") else None

    def _save_today(self, d: dict) -> None:
        p = self._today_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(d))
        tmp.replace(p)

    def _mix(self, queues: dict[str, list[dict]], n: int, exclude: set[str]) -> list[dict]:
        """Take up to `n` items across the jobs in the MIX rhythm (a job with
        nothing left is skipped), never the same scene twice."""
        taken: list[dict] = []
        used = set(exclude)
        pos = {j: 0 for j in queues}
        idle = 0
        i = 0
        while len(taken) < n and idle < len(MIX):
            job = MIX[i % len(MIX)]
            i += 1
            q = queues.get(job) or []
            while pos[job] < len(q) and q[pos[job]]["scene_id"] in used:
                pos[job] += 1
            if pos[job] >= len(q):
                idle += 1
                continue
            idle = 0
            item = q[pos[job]]
            pos[job] += 1
            used.add(item["scene_id"])
            taken.append({"scene_id": item["scene_id"], "job": job})
        return taken

    def _today_state(self, ctx: dict, more: bool = False) -> dict:
        d = self._load_today()
        if d is None or more:
            queues = {j: self._job_queue(j, ctx) for j in JOBS}
            base = d or {"date": time.strftime("%Y-%m-%d"), "items": [], "done": [], "skipped": []}
            have = {it["scene_id"] for it in base["items"]}
            new = self._mix(queues, self.today_goal(), have)
            for it in new:
                it["tier"] = (ctx["by"].get(it["scene_id"]) or {}).get("tier")
            base["items"] += new
            d = base
            self._save_today(d)
        # graded anywhere else since it was picked → done
        changed = False
        for it in d["items"]:
            row = ctx["by"].get(it["scene_id"])
            if it["scene_id"] not in d["done"] and (row is None or row["tier"] != it.get("tier")):
                d["done"].append(it["scene_id"])
                changed = True
        if changed:
            self._save_today(d)
        return d

    def today_mark(self, scene_id: str, skip: bool = False, undo: bool = False) -> dict:
        """Done (answered here), skipped (to the back of today's set), or — after
        an undo — back to not done."""
        d = self._load_today() or {"date": time.strftime("%Y-%m-%d"), "items": [], "done": [], "skipped": []}
        sid = str(scene_id)
        if undo:
            d["done"] = [s for s in d["done"] if s != sid]
        elif skip:
            d["skipped"] = [s for s in d["skipped"] if s != sid] + [sid]
        elif sid not in d["done"]:
            d["done"].append(sid)
        self._save_today(d)
        return {"done": len(d["done"])}

    # --- the page ---------------------------------------------------------------------

    def curation_today(self, job: str | None = None, more: bool = False, n: int = 3) -> dict:
        """The next `n` decisions (full payloads, so the page can preload), today's
        progress, the task board's counts and this week's tally. With `job`, a
        stream of just that job (no daily cap)."""
        ctx = self._today_context()
        if job:
            if job not in JOBS:
                raise ValueError(f"unknown job: {job}")
            d = self._load_today() or {"done": [], "skipped": []}
            q = [it for it in self._job_queue(job, ctx) if it["scene_id"] not in set(d["done"])]
            skipped = set(d.get("skipped") or [])
            q.sort(key=lambda it: it["scene_id"] in skipped)       # skipped ones last
            nxt = [{**it, "job": job} for it in q[:n]]
            remaining, progress = len(q), None
        else:
            d = self._today_state(ctx, more=more)
            done = set(d["done"])
            skipped = d.get("skipped") or []
            order = [it for it in d["items"] if it["scene_id"] not in done]
            order.sort(key=lambda it: (it["scene_id"] in skipped,
                                       skipped.index(it["scene_id"]) if it["scene_id"] in skipped else 0))
            queue_items = {j: {q["scene_id"]: q for q in self._job_queue(j, ctx)}
                           for j in {it["job"] for it in order[:n]}}
            nxt = []
            for it in order[:n]:
                q = queue_items.get(it["job"], {}).get(it["scene_id"]) or {"pick": None, "why": "", "detail": ""}
                nxt.append({**q, **it})
            remaining = len(order)
            ids = {i["scene_id"] for i in d["items"]}
            progress = {"done": len(ids & done), "total": len(ids), "goal": self.today_goal()}
        return {"next": [self._decision_payload(it, ctx) for it in nxt], "remaining": remaining,
                "progress": progress, "job": job, "board": self._today_board(ctx), "week": self.week_tally(),
                "labels": JOB_LABEL}

    def _decision_payload(self, it: dict, ctx: dict) -> dict:
        sid = it["scene_id"]
        r = ctx["by"].get(sid) or {"scene_id": sid, "title": f"scene {sid}", "performers": [], "tier": "unreviewed"}
        moments = self._scene_moment_strips([sid], n=6).get(sid, [])
        return {
            **{k: r.get(k) for k in ("scene_id", "title", "performers", "performer_ids", "studio", "date",
                                     "duration", "tier", "rating100", "o_counter", "quality", "size", "path")},
            "job": it["job"], "pick": it.get("pick"), "why": it.get("why", ""), "detail": it.get("detail", ""),
            "stream": self.stream_url(sid, start=0), "moments": moments,
            "pred": ctx["preds"].get(sid), "signals": ctx["sig"].get(sid),
            "who": self._who_reasons(r, self._who_records(ctx["rows"]), ctx["names"]),
        }

    def _today_board(self, ctx: dict) -> dict:
        from ..tier_model import quality_flag

        rows, preds, sig, floor = ctx["rows"], ctx["preds"], ctx["sig"], ctx["floor"]
        counts = {j: len(self._job_queue(j, ctx)) for j in JOBS}
        rejected = [r for r in rows if r["tier"] == "rejected"]
        dupes = self.cached_duplicates()
        return {
            "jobs": counts,
            "likely": len(self._triage("likely", rows, preds, floor, sig)),
            "conflicts": sum(1 for r in rows if r["tag_state"]["conflict"]),
            "quality": sum(1 for r in rows if r["tier"] == "unreviewed" and quality_flag(r["quality"], floor)),
            "dupes": len(dupes["groups"]) if dupes else None,
            "rejected": {"count": len(rejected), "bytes": sum(int(r.get("size") or 0) for r in rejected)},
        }

    def week_tally(self) -> dict:
        """This week's decisions from the action log (every screen, not just this one)."""
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 7 * 86400))
        out = {"decisions": 0, "promoted": 0, "let_go": 0, "freed": 0}
        try:
            entries = self.action_log().tail(20000)
        except Exception:  # noqa: BLE001
            return out
        for e in entries:
            if (e.get("ts") or "") < cutoff:
                break                       # newest first
            a = e.get("action")
            if a == "grade":
                out["decisions"] += 1
                after = (e.get("after") or {}).get("tier")
                if after == "legendaire":
                    out["promoted"] += 1
                if after == "rejected" or e.get("grade") == "reject":
                    out["let_go"] += 1
            elif a == "delete":
                out["freed"] += int(e.get("size") or 0)
        return out


def _why(rec: dict) -> dict:
    return {"why": rec.get("why", ""), "detail": rec.get("detail", "")}
