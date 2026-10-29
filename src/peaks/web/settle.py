"""Settling the library: your grades, checked once and for all.

Only graded scenes take part — unreviewed scenes are left to Likely keepers and
Quality check (most of them end up rejected, so asking about them wastes time).
What's left are the questions:

  - disagree: every grade Peaks' honest verdict (from a model that never saw
    that grade) disagrees with — confident or just leaning, surest first —
    until you rule on it ("Keep mine") or grade it again;
  - drain: a scene in a tier being retired (Merveilleuse): which live tier?
    Asked most-to-learn first: uncertain and representative, spread out so a
    session isn't twenty look-alikes.

The two kinds are interleaved. The calibration (where Peaks is right 9 times in
10 on your grades) labels each disagreement "sure" or "leaning".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

TARGET = 0.90             # "right about 9 times in 10"
MIN_ABOVE = 20            # graded scenes above a threshold before it's trusted
FALLBACK = 0.90
NEIGHBOURS = 10           # for "how representative": mean similarity to the closest open questions
SPREAD = 0.6              # how hard a pick down-weights its look-alikes
QUEUE = 200               # questions ordered per request (the rest follow as they're answered)
FAST_RETRAIN = 10         # grades between retrains while you're settling
SETTLING_WINDOW = 1800    # seconds after the settle list was opened that count as "settling"


def calibrate(oof: dict) -> dict:
    """The lowest confidence at which the honest verdicts on your graded scenes
    are right at least TARGET of the time (with MIN_ABOVE scenes to show for it)."""
    rows = sorted(((float(v["conf"]), v["tier"] == v.get("graded")) for v in (oof or {}).values()
                   if v.get("graded")), reverse=True)
    best, right = None, 0
    for i, (c, ok) in enumerate(rows, 1):
        right += ok
        if i >= MIN_ABOVE and right / i >= TARGET:
            best = (c, right / i, i)
    if best is None:
        return {"t": FALLBACK, "measured": False, "n": len(rows)}
    return {"t": round(best[0], 3), "measured": True, "accuracy": round(best[1], 3), "above": best[2], "n": len(rows)}


def _margin(p: dict) -> float:
    v = sorted((p.get("probs") or {}).values(), reverse=True) + [0.0, 0.0]
    return float(v[0] - v[1])


def spread_order(value: np.ndarray, X: np.ndarray | None, k: int) -> list[int]:
    """Greedy picks by value, each pick down-weighting its look-alikes."""
    v = value.astype(np.float64).copy()
    out: list[int] = []
    taken = np.zeros(len(v), bool)
    for _ in range(min(k, len(v))):
        j = int(np.argmax(np.where(taken, -np.inf, v)))
        out.append(j)
        taken[j] = True
        if X is not None:
            sim = np.clip(X @ X[j], 0, 1)
            v *= 1.0 - SPREAD * sim ** 4
    return out


class SettleMixin:
    # --- state -----------------------------------------------------------------------------

    def _rulings_path(self) -> Path:
        return self._state_dir() / "rulings.json"

    def rulings(self) -> dict:
        try:
            return json.loads(self._rulings_path().read_text())
        except (OSError, ValueError):
            return {}

    def _rulings_save(self, d: dict) -> None:
        from .. import atomic

        self._rulings_path().parent.mkdir(parents=True, exist_ok=True)
        atomic.write_json(self._rulings_path(), d, indent=1)

    def settle_rule(self, sid: str) -> dict:
        """"Keep mine": your grade stands against Peaks' disagreement, for good —
        until you grade the scene again (a ruling is for the grade it was made on)."""
        row = next((r for r in self._catalogue_all() if r["scene_id"] == str(sid)), None)
        if row is None:
            raise KeyError(sid)
        d = self.rulings()
        d[str(sid)] = {"grade": row["tier"], "at": time.time()}
        self._rulings_save(d)
        self._settle_log(answer=True)
        return {"ruled": True, "scene_id": str(sid), "grade": row["tier"]}

    def settle_unrule(self, sid: str) -> dict:
        d = self.rulings()
        d.pop(str(sid), None)
        self._rulings_save(d)
        return {"ruled": False, "scene_id": str(sid)}

    def _settle_history_path(self) -> Path:
        return self._state_dir() / "settle_history.jsonl"

    def settle_history(self) -> list[dict]:
        try:
            return [json.loads(x) for x in self._settle_history_path().read_text().splitlines() if x.strip()]
        except (OSError, ValueError):
            return []

    def _settle_log(self, answer: bool = False, state: dict | None = None, force: bool = False) -> None:
        """One history point: the unsettled count and how many answers so far.
        Answers are counted every time; a point is written at most every 10 minutes
        (and after each retrain) so the file stays small."""
        self._settle_answers = getattr(self, "_settle_answers", None)
        if self._settle_answers is None:
            h = self.settle_history()
            self._settle_answers = h[-1]["answers"] if h else 0
        if answer:
            self._settle_answers += 1
        last = getattr(self, "_settle_logged_at", 0.0)
        if not force and time.time() - last < 600:
            return
        try:
            st = state or self.settle_state(log=False)
        except Exception:  # noqa: BLE001 — no model yet: nothing to log
            return
        p = self._settle_history_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps({"at": time.time(), "unsettled": st["unsettled"], "answers": self._settle_answers}) + "\n")
        self._settle_logged_at = time.time()

    # --- the questions ---------------------------------------------------------------------

    def _settle_classify(self, rows: list[dict], preds: dict, cal: dict) -> dict:
        from ..tier_model import train_class
        from ..tiers import draining

        rules = self.rulings()
        drain = set(draining())
        t = cal["t"]
        disagree, drains, agree = [], [], 0
        for r in rows:
            sid, p = r["scene_id"], preds.get(r["scene_id"])
            if p is None or p.get("from") == "who":
                continue                             # not embedded yet: nothing honest to say
            graded = train_class(r["tier"])
            if r.get("no_learn"):
                continue                             # a reject for your own reasons: never questioned
            if graded:                               # every grade the honest verdict disagrees with
                ru = rules.get(sid)
                if p["tier"] != graded and not (ru and ru.get("grade") == r["tier"]):
                    disagree.append(r)
                else:
                    agree += 1
            elif r["tier"] in drain:                 # a retiring tier: which live tier is it?
                drains.append(r)
            # unreviewed scenes are never questions here — most of them end up rejected
        return {"disagree": disagree, "drain": drains, "agree": agree, "t": t}

    def settle_state(self, log: bool = True) -> dict:
        """The counter and its context, for the Insights card and the sidebar."""
        st = self._tier_model_state()
        m = st["model"]
        if m is None:
            return {"ok": False, "reason": "Peaks needs a tier model first — grade a few scenes in two tiers or more"}
        rows = self._catalogue_all()
        preds = self._tier_predictions(rows)
        cal = calibrate(getattr(m, "oof", None) or {})
        c = self._settle_classify(rows, preds, cal)
        n = len(c["disagree"]) + len(c["drain"])
        self._settle_last = n
        out = {"ok": True, "unsettled": n, "disagree": len(c["disagree"]),
               "disagree_sure": sum(1 for r in c["disagree"] if preds[r["scene_id"]]["conf"] >= cal["t"]),
               "drain": len(c["drain"]), "agree": c["agree"], "threshold": cal, "scenes": len(rows),
               "unembedded": sum(1 for r in rows if r["scene_id"] not in preds or preds[r["scene_id"]].get("from") == "who"),
               "history": self.settle_history()[-400:], "projection": None}
        h = out["history"]
        if len(h) >= 3:
            recent = [x for x in h if x["at"] >= h[-1]["at"] - 14 * 86400] or h
            da = recent[-1]["answers"] - recent[0]["answers"]
            du = recent[0]["unsettled"] - recent[-1]["unsettled"]
            if da >= 20 and du > 0:
                per = du / da                         # questions settled per answer, lately
                out["projection"] = {"per_answer": round(per, 2), "answers": int(np.ceil(n / per))}
        if log:
            self._settle_log(state=out)               # at most one point per 10 minutes
        return out

    def _settle_vectors(self, rows: list[dict]) -> dict[str, np.ndarray]:
        feats = self._scene_features(rows)
        out = {}
        for sid, (vis, _) in feats.items():
            v = np.asarray(vis[: len(vis) // 2], np.float32)        # the scene's overall look
            out[sid] = v / max(float(np.linalg.norm(v)), 1e-9)
        return out

    def settle_queue(self, pool: list[dict], preds: dict) -> list[dict]:
        """The catalogue's `settle` view: questions, most to learn first."""
        st = self._tier_model_state()
        if st["model"] is None:
            return []
        self._settling_at = time.time()
        cal = calibrate(getattr(st["model"], "oof", None) or {})
        c = self._settle_classify(pool, preds, cal)
        names = self.tier_display_names()
        nm = lambda c_: names.get(c_, c_)  # noqa: E731
        info: dict[str, dict] = {}
        drains = c["drain"]
        order: list[dict] = []
        if drains:                                    # most to learn first, spread out
            vecs = self._settle_vectors(drains)
            have = [r for r in drains if r["scene_id"] in vecs]
            X = np.stack([vecs[r["scene_id"]] for r in have]) if have else None
            unc = np.array([1.0 - _margin(preds[r["scene_id"]]) for r in have])
            if X is not None and len(have) > 1:
                S = X @ X.T
                np.fill_diagonal(S, -1)
                k = min(NEIGHBOURS, len(have) - 1)
                rep = np.sort(S, axis=1)[:, -k:].mean(axis=1).clip(0, 1)
            else:
                rep = np.ones(len(have))
            value = unc * (0.25 + rep)
            order = [have[j] for j in spread_order(value, X, min(QUEUE, len(have)))]
            order += [r for r in drains if r["scene_id"] not in vecs][: max(0, QUEUE - len(order))]
        dis = sorted(c["disagree"], key=lambda r: -preds[r["scene_id"]]["conf"])
        # the two kinds interleaved in proportion, so neither waits for the other to run out
        out: list[dict] = []
        di = oi = 0
        while di < len(dis) or oi < len(order):
            if oi >= len(order) or (di < len(dis) and di * len(order) <= oi * len(dis)):
                out.append(dis[di])
                di += 1
            else:
                out.append(order[oi])
                oi += 1
        drain_ids = {r["scene_id"] for r in drains}
        for r in out:
            p = preds[r["scene_id"]]
            top = sorted((p.get("probs") or {}).items(), key=lambda kv: -kv[1])[:2]
            if r["scene_id"] in drain_ids:
                info[r["scene_id"]] = {"kind": "drain", "peaks": p["tier"], "conf": p["conf"],
                                       "top": [{"tier": k, "p": v} for k, v in top],
                                       "why": f"{nm(r['tier'])} is retiring · Peaks leans " + " / ".join(
                                           f"{nm(k)} {round(100 * v)}%" for k, v in top)}
            else:
                sure = p["conf"] >= c["t"]
                info[r["scene_id"]] = {"kind": "disagree", "peaks": p["tier"], "conf": p["conf"], "sure": sure,
                                       "yours": r["tier"],
                                       "why": f"You said {nm(r['tier'])} · Peaks says {nm(p['tier'])} "
                                              f"({round(100 * p['conf'])}%, {'sure' if sure else 'leaning'})"}
        self._settle_info = info
        return out

    def settle_count(self, pool: list[dict], preds: dict) -> int:
        st = self._tier_model_state()
        if st["model"] is None:
            return 0
        c = self._settle_classify(pool, preds, calibrate(getattr(st["model"], "oof", None) or {}))
        n = len(c["disagree"]) + len(c["drain"])
        if len(pool) >= len(self._catalogue_all()):          # the whole library: the floor shows it
            self._settle_last = n
        return n

    def settle_info(self, sid: str) -> dict | None:
        return (getattr(self, "_settle_info", None) or {}).get(str(sid))

    def settling(self) -> bool:
        return time.time() - getattr(self, "_settling_at", 0.0) < SETTLING_WINDOW
