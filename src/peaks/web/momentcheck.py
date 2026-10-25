"""Do Peaks' moment picks land where you would? Measured honestly.

A **hit**: a moment you saved lies within HIT_WINDOW seconds of one of Peaks'
top picks for that scene (the same well-spaced picking as the moment strips).

* **Live** (the number that matters): every save made in Peaks records — before
  the save joins your taste — whether Peaks' *current* picks already had it
  (state/moment_checks.jsonl).
* **Estimate** for saves made before that: each scene is scored only against
  loved moments from *other* scenes (its own saves left out), so Peaks isn't
  grading its own homework.
* **Pick approval**: your 👍/👎 on Peaks' picks while watching
  (state/pick_feedback.jsonl) — whether it picks your *kind* of moment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

HIT_WINDOW = 15.0        # seconds between a saved moment and a pick to count as found
TOP_K = 3
PICK_GAP = 20.0          # same spacing as the moment strips
EST_SCENES = 300         # most recent saved scenes the estimate looks at


class MomentCheckMixin:
    def _mc_path(self, name: str) -> Path:
        return self._state_dir() / name

    def _mc_append(self, name: str, rec: dict) -> None:
        p = self._mc_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _mc_read(self, name: str) -> list[dict]:
        try:
            lines = self._mc_path(name).read_text().splitlines()
        except OSError:
            return []
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
        return out

    @staticmethod
    def _mc_hit(picks: list[float], t: float) -> int | None:
        """Rank (1-based) of the pick that found `t`, or None."""
        for rank, p in enumerate(picks, 1):
            if abs(p - t) <= HIT_WINDOW:
                return rank
        return None

    def scene_picks(self, scene_id: str, n: int = 4) -> list[dict]:
        """Peaks' best moments in a scene, best first: [{t, score, thumb}]."""
        strip = self._scene_moment_strips([str(scene_id)], n=n, min_gap=PICK_GAP).get(str(scene_id)) or []
        return sorted(strip, key=lambda m: -(m.get("score") or 0))

    # --- live: every save is checked against the picks Peaks had before it -----------------

    def moment_check_on_save(self, scene_id: str, t: float) -> None:
        """Best-effort: never in the way of the save itself."""
        try:
            picks = self.scene_picks(scene_id, n=TOP_K)
            if not picks or picks[0].get("score") is None:
                return                              # no taste yet: those aren't picks, just spacing
            ts = [float(p["t"]) for p in picks]
            rank = self._mc_hit(ts, float(t))
            self._mc_append("moment_checks.jsonl", {
                "at": time.time(), "scene_id": str(scene_id), "t": round(float(t), 2),
                "picks": [round(x, 2) for x in ts], "rank": rank, "key": picks[0].get("key")})
        except Exception:  # noqa: BLE001
            pass

    def record_pick_feedback(self, scene_id: str, t: float, label: int) -> None:
        self._mc_append("pick_feedback.jsonl", {"at": time.time(), "scene_id": str(scene_id),
                                                "t": round(float(t), 2), "label": int(label)})

    # --- estimate: past saves, each scene scored without its own saves ----------------------

    def _mc_estimate(self) -> dict:
        model = self._model_name()
        idx = self.index(model)
        arr, sources = self._taste_sources(model)
        cached = self.__dict__.get("_mc_est_cache")
        sig = (id(idx), int(arr.shape[0]))
        if cached and cached[0] == sig:
            return cached[1]
        apex = [s for s in sources if s.get("kind") == "apex" and s.get("scene_id")]
        by_scene: dict[str, list[float]] = {}
        for s in apex:
            by_scene.setdefault(str(s["scene_id"]), []).append(float(s["time"]))
        sid_key = {str(m.get("scene_id")): k for k, m in idx.key_meta.items()}
        owner = np.array([str(s.get("scene_id") or "") for s in sources])
        out = {"scenes": 0, "saves": 0, "top1": 0, "top3": 0, "examples": []}
        if arr.shape[0] == 0:
            return out
        L = np.asarray(arr, np.float32)
        for sid in list(by_scene)[-EST_SCENES:]:
            span = idx._key_rows.get(sid_key.get(sid, ""))
            others = owner != sid
            if not span or not others.any():
                continue
            a, b = span
            frames = idx.rows(a, b).astype(np.float32)
            frames /= np.maximum(np.linalg.norm(frames, axis=1, keepdims=True), 1e-9)
            score = (frames @ L[others].T).max(axis=1)        # nearest loved moment elsewhere
            times = idx.times[a:b]
            picks: list[float] = []
            for j in np.argsort(-score):
                tt = float(times[j])
                if all(abs(tt - p) >= PICK_GAP for p in picks):
                    picks.append(tt)
                if len(picks) >= TOP_K:
                    break
            out["scenes"] += 1
            for t in by_scene[sid]:
                rank = self._mc_hit(picks, t)
                out["saves"] += 1
                out["top1"] += rank == 1
                out["top3"] += rank is not None
                if len(out["examples"]) < 400:
                    out["examples"].append({"scene_id": sid, "t": t, "picks": picks, "rank": rank,
                                            "key": sid_key.get(sid)})
        self._mc_est_cache = (sig, out)
        return out

    # --- the Insights view ---------------------------------------------------------------

    def moment_insights(self) -> dict:
        live = self._mc_read("moment_checks.jsonl")
        fb = self._mc_read("pick_feedback.jsonl")

        def rate(xs, f):
            return round(sum(1 for x in xs if f(x)) / len(xs), 3) if xs else None

        def by_month(xs, f):
            months: dict[str, list] = {}
            for x in xs:
                months.setdefault(time.strftime("%Y-%m", time.localtime(x["at"])), []).append(x)
            return [{"month": m, "n": len(v), "rate": rate(v, f)} for m, v in sorted(months.items())]

        try:
            est = self._mc_estimate()
        except Exception:  # noqa: BLE001 — no index / taste yet
            est = {"scenes": 0, "saves": 0, "top1": 0, "top3": 0, "examples": []}

        def thumb(key, t):
            return f"/api/frame?key={key}&t={t:g}" if key else None

        def example(x):
            near = min(x["picks"], key=lambda p: abs(p - x["t"])) if x.get("picks") else None
            return {"scene_id": x["scene_id"], "saved": x["t"], "pick": near, "rank": x.get("rank"),
                    "saved_thumb": thumb(x.get("key"), x["t"]),
                    "pick_thumb": thumb(x.get("key"), near) if near is not None else None}

        src = live if len(live) >= 10 else est["examples"]
        hits = [example(x) for x in reversed(src) if x.get("rank")][:6]
        misses = [example(x) for x in reversed(src) if not x.get("rank")][:6]
        return {
            "window": HIT_WINDOW, "top_k": TOP_K,
            "live": {"n": len(live), "top1": rate(live, lambda x: x.get("rank") == 1),
                     "top3": rate(live, lambda x: x.get("rank") is not None),
                     "by_month": by_month(live, lambda x: x.get("rank") is not None)},
            "estimate": {"scenes": est["scenes"], "n": est["saves"],
                         "top1": round(est["top1"] / est["saves"], 3) if est["saves"] else None,
                         "top3": round(est["top3"] / est["saves"], 3) if est["saves"] else None},
            "approval": {"n": len(fb), "rate": rate(fb, lambda x: x["label"] == 1),
                         "by_month": by_month(fb, lambda x: x["label"] == 1)},
            "hits": hits, "misses": misses,
            "examples_from": "live" if len(live) >= 10 else "estimate",
        }
