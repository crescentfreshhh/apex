"""Your criteria: why a grade is what it is, in your own five words.

You grade on body, face, outfit (slutwear), action (sex acts, aggression,
passion) and quality. You tell Peaks which of them make a scene better or worse
than it gave credit for — ▲/▼ chips on a scene — and rate a performer's face and
body once on her page. From those, Peaks learns one small model per criterion,
each from the inputs that fit it:

  face    — the scene's main face (ArcFace, from Faces): carries to look-alikes
  body    — how the scene looks (DINOv2 mean)
  outfit  — how it looks, plus your named concepts' scores (heels, lingerie…)
  action  — Motion and Hearing where present, plus how it looks
  quality — the picture-quality measures and the file's numbers

Each gives every scene a 0–1 score. A scene you labelled is scored by a model
that never saw its label (out of fold), so its own answer never leaks into it.
The five scores go to the tier model as one more group — "criteria" — kept only
when the honest held-out test says grading improves, like Hearing and Motion.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from .. import atomic

ASPECTS = ("body", "face", "outfit", "action", "quality")
WORDS = {"body": "Body", "face": "Face", "outfit": "Outfit", "action": "Action", "quality": "Quality"}
PERFORMER_ASPECTS = ("face", "body", "action")        # action: her enthusiasm, energy, passion
RATING_W = {-1: (0.0, 1.0), 1: (1.0, 1.0), 2: (1.0, 2.0)}     # her rating → (label, weight); 0 = ok: no label
MIN_LABELS = 15           # per criterion, before it gets a model
MIN_EACH = 3              # of each side
MIN_ACC = 0.6             # held-out accuracy before a criterion's score is used
FOLDS = 5


def band(x: float | None) -> str | None:
    if x is None or np.isnan(x):
        return None
    return "weak" if x < 0.35 else "ok" if x < 0.65 else "strong"


class CriteriaMixin:
    # --- what you said -------------------------------------------------------------------------

    def _crit_path(self) -> Path:
        return self._state_dir() / "criteria.json"

    def _pcrit_path(self) -> Path:
        return self._state_dir() / "performer_criteria.json"

    def criteria_reasons(self) -> dict:
        return atomic.read_json(self._crit_path(), default={}) or {}

    def performer_criteria(self) -> dict:
        return atomic.read_json(self._pcrit_path(), default={}) or {}

    def set_scene_criterion(self, scene_id: str, aspect: str, value: int) -> dict:
        """▲ (+1) / ▼ (−1) / clear (0) one criterion for a scene."""
        if aspect not in ASPECTS or int(value) not in (-1, 0, 1):
            raise ValueError("aspect is body/face/outfit/action/quality, value −1, 0 or 1")
        sid = str(scene_id)
        p = self._crit_path()
        with atomic.edit_lock(p):
            d = self.criteria_reasons()
            e = dict(d.get(sid) or {})
            if int(value):
                e[aspect] = int(value)
            else:
                e.pop(aspect, None)
            e["at"] = time.time()
            if any(k in e for k in ASPECTS):
                d[sid] = e
            else:
                d.pop(sid, None)
            p.parent.mkdir(parents=True, exist_ok=True)
            atomic.write_json(p, d, indent=1)
        self._crit_dirty()
        row = next((r for r in self._catalogue_all() if r["scene_id"] == sid), None)
        if row is not None and int(value):
            self._log_scene("criteria", row, detail=f"{'▲' if int(value) > 0 else '▼'} {WORDS[aspect]}")
        return {"scene_id": sid, "reasons": {k: v for k, v in (d.get(sid) or {}).items() if k in ASPECTS}}

    def set_performer_criterion(self, pid: str, aspect: str, value: int | None) -> dict:
        """Her face / body: −1 (no), 0 (ok), 1 (+), 2 (++); None clears."""
        if aspect not in PERFORMER_ASPECTS or (value is not None and int(value) not in (-1, 0, 1, 2)):
            raise ValueError("aspect is face, body or action, value −1, 0, 1 or 2")
        pid = str(pid)
        p = self._pcrit_path()
        with atomic.edit_lock(p):
            d = self.performer_criteria()
            e = dict(d.get(pid) or {})
            if value is None:
                e.pop(aspect, None)
            else:
                e[aspect] = int(value)
            e["at"] = time.time()
            d[pid] = e
            p.parent.mkdir(parents=True, exist_ok=True)
            atomic.write_json(p, d, indent=1)
        self._crit_dirty()
        return {"id": pid, "ratings": {k: e[k] for k in PERFORMER_ASPECTS if k in e}}

    def _crit_dirty(self) -> None:
        self.__dict__.pop("_crit_cache", None)          # _crit_last stays: the screen keeps it until the rebuild

    # --- what each criterion looks at ---------------------------------------------------------

    def _crit_inputs(self, sids: list[str], rows: dict) -> dict[str, dict[str, np.ndarray]]:
        """aspect → {scene_id: input vector} (scenes without that input left out)."""
        from ..tier_model import picture_features, quality_features

        means = self._scene_means()
        try:
            concepts = [self.concept_scene_scores(slug) for slug, _ in self.trained_concepts()]
        except Exception:  # noqa: BLE001
            concepts = []
        pics = self._picture_entries()
        out: dict[str, dict] = {a: {} for a in ASPECTS}
        for sid in sids:
            look = means.get(sid)
            if look is not None:
                out["body"][sid] = look
                cv = np.asarray([c.get(sid, 0.0) for c in concepts], np.float32)
                out["outfit"][sid] = np.r_[look, 3.0 * cv] if len(cv) else look
                parts, flags = [look], []
                for fn in (self.motion_scene_vector, self.sound_scene_vector):
                    try:
                        v = fn(sid)
                    except Exception:  # noqa: BLE001
                        v = None
                    flags.append(0.0 if v is not None else 1.0)
                    if v is not None:
                        parts.append(np.asarray(v, np.float32))
                out["action"][sid] = (parts, flags)             # padded below, once dims are known
            try:
                sf = self._scene_faces(sid)
            except Exception:  # noqa: BLE001
                sf = None
            if sf is not None and len(sf[1]):
                dom = self._dominant(sf[1])
                f = sf[1][dom].mean(axis=0)
                out["face"][sid] = f / max(float(np.linalg.norm(f)), 1e-9)
            r = rows.get(sid)
            if r is not None:
                out["quality"][sid] = np.r_[picture_features(pics.get(sid)), quality_features(r.get("quality") or {})]
        # action: the same width for every scene (missing senses as zeros + their flags)
        dims = {}
        for parts, flags in out["action"].values():
            i = 1
            for k, miss in enumerate(flags):
                if not miss:
                    dims[k] = len(parts[i])
                    i += 1
        act = {}
        for sid, (parts, flags) in out["action"].items():
            vec, i = [parts[0]], 1
            for k, miss in enumerate(flags):
                if k not in dims:
                    continue
                if miss:
                    vec.append(np.zeros(dims[k], np.float32))
                else:
                    vec.append(parts[i])
                    i += 1
            act[sid] = np.r_[np.concatenate(vec), np.asarray([flags[k] for k in dims], np.float32)]
        out["action"] = act
        return out

    def _crit_labels(self, rows: dict) -> dict[str, list[tuple[str, float, float, str]]]:
        """aspect → [(scene_id, label 0/1, weight, group)]. A group is what a fold
        holds out together: the scene, or the performer whose rating it came from."""
        out: dict[str, list] = {a: [] for a in ASPECTS}
        for sid, e in self.criteria_reasons().items():
            if sid not in rows:
                continue
            for a in ASPECTS:
                if e.get(a) in (1, -1):
                    out[a].append((sid, 1.0 if e[a] > 0 else 0.0, 1.0, "s" + sid))
        ratings = self.performer_criteria()
        if ratings:
            from ..tier_model import who_of

            her: dict[str, list[str]] = {}
            for r in rows.values():
                if r["tier"] == "rejected":
                    continue
                for pid in who_of(r)["performers"]:
                    her.setdefault(pid, []).append(r["scene_id"])
            for pid, e in ratings.items():
                scenes = her.get(str(pid), [])
                if not scenes:
                    continue
                for a in PERFORMER_ASPECTS:
                    lw = RATING_W.get(e.get(a))
                    if lw is None:
                        continue
                    w = lw[1] / np.sqrt(len(scenes))          # one prolific performer mustn't drown the rest
                    out[a] += [(sid, lw[0], w, "p" + str(pid)) for sid in scenes]
        return out

    # --- the models -----------------------------------------------------------------------------

    @staticmethod
    def _crit_fit(X, y, w):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)
                             ).fit(X, y, logisticregression__sample_weight=w)

    def criteria_models(self, wait: bool = True) -> dict:
        """Per criterion: {labels, yes, no, acc, used, scores: {scene_id: 0–1}} —
        every scene with that input scored; a labelled scene out of fold.

        wait=False (what the screen asks for): never block on a rebuild — hand back
        the last models (or none yet) and rebuild in the background, so a ▲/▼ tap
        answers at once even while an ingest keeps the machine busy."""
        if not wait:
            hit = self.__dict__.get("_crit_cache") or self.__dict__.get("_crit_last")
            if hit is not None and hit[0] == self._crit_stamp():
                return hit[1]
            self._crit_rebuild_soon()
            return hit[1] if hit is not None else {a: {"labels": 0, "yes": 0, "no": 0, "acc": None, "used": False,
                                                        "scores": {}} for a in ASPECTS}
        return self._crit_build()

    def _crit_stamp(self) -> tuple:
        return (self._crit_path().stat().st_mtime_ns if self._crit_path().exists() else 0,
                self._pcrit_path().stat().st_mtime_ns if self._pcrit_path().exists() else 0,
                getattr(self.index(self._model_name()), "gen", 0), len(self._catalogue_all()),
                len(self.__dict__.get("_face_all") or ()))

    def _crit_rebuild_soon(self) -> None:
        """One background rebuild at a time; another tap while it runs is picked up by the next."""
        lk = self.__dict__.setdefault("_crit_bg", threading.Lock())
        if not lk.acquire(blocking=False):
            return

        def run():
            try:
                self._crit_build()
            except Exception:  # noqa: BLE001
                pass
            finally:
                lk.release()
        threading.Thread(target=run, name="criteria-rebuild", daemon=True).start()

    def _crit_build(self) -> dict:
        rows = {r["scene_id"]: r for r in self._catalogue_all()}
        stamp = (self._crit_path().stat().st_mtime_ns if self._crit_path().exists() else 0,
                 self._pcrit_path().stat().st_mtime_ns if self._pcrit_path().exists() else 0,
                 getattr(self.index(self._model_name()), "gen", 0), len(rows),
                 len(self.__dict__.get("_face_all") or ()))
        hit = self.__dict__.get("_crit_cache")
        if hit is not None and hit[0] == stamp:
            return hit[1]
        labels = self._crit_labels(rows)
        need = sorted({sid for ls in labels.values() for sid, *_ in ls})
        out = {}
        inputs = None
        for a in ASPECTS:
            ls = labels[a]
            yes = sum(1 for _, y, *_ in ls if y > 0.5)
            info = {"labels": len(ls), "yes": yes, "no": len(ls) - yes, "acc": None, "used": False, "scores": {}}
            out[a] = info
            if len(ls) < MIN_LABELS or yes < MIN_EACH or len(ls) - yes < MIN_EACH:
                continue
            if inputs is None:                        # only built once some criterion has enough to learn
                live = [s for s, r in rows.items() if r["tier"] != "rejected"]
                inputs = self._crit_inputs(sorted(set(live) | set(need)), rows)
            xin = inputs[a]
            ls = [x for x in ls if x[0] in xin]
            if len(ls) < MIN_LABELS:
                continue
            X = np.stack([xin[s] for s, *_ in ls]).astype(np.float32)
            y = np.asarray([v for _, v, *_ in ls]) > 0.5
            w = np.asarray([v for *_, v, _ in ls], np.float32)
            groups = np.asarray([g for *_, g in ls])
            if y.all() or not y.any():
                continue
            # honest accuracy and out-of-fold scores: whole groups (a scene, or a performer) held out
            ug = np.unique(groups)
            rng = np.random.default_rng(0)
            fold_of_g = dict(zip(ug, rng.permutation(len(ug)) % FOLDS))
            fold = np.asarray([fold_of_g[g] for g in groups])
            oof = np.full(len(ls), np.nan)
            for f in range(FOLDS):
                te, tr = fold == f, fold != f
                if not te.any() or y[tr].all() or not y[tr].any():
                    continue
                oof[te] = self._crit_fit(X[tr], y[tr], w[tr]).predict_proba(X[te])[:, 1]
            ok = ~np.isnan(oof)
            acc = float(np.average((oof[ok] > 0.5) == y[ok], weights=w[ok])) if ok.any() else None
            base = float(max(np.average(y, weights=w), 1 - np.average(y, weights=w)))
            info["acc"] = None if acc is None else round(acc, 3)
            info["used"] = bool(acc is not None and acc >= MIN_ACC and acc >= base - 0.02)
            if not info["used"]:
                continue
            full = self._crit_fit(X, y, w)
            sids = list(xin)
            sc = full.predict_proba(np.stack([xin[s] for s in sids]))[:, 1]
            scores = dict(zip(sids, map(float, sc)))
            by_scene: dict[str, list] = {}
            for (sid, *_), p in zip(ls, oof):          # a labelled scene: never its own label's model
                if not np.isnan(p):
                    by_scene.setdefault(sid, []).append(float(p))
            for sid, ps in by_scene.items():
                scores[sid] = float(np.mean(ps))
            info["scores"] = scores
        self._crit_cache = self._crit_last = (stamp, out)
        return out

    def criteria_scene_vector(self, sid: str) -> np.ndarray | None:
        """The five scores for the tier model (0.5 where a criterion has nothing to
        say). None when no criterion has a model yet, or the scene has no score at all."""
        m = self.criteria_models(wait=False)        # never retrain inside a page request
        used = [a for a in ASPECTS if m[a]["used"]]
        if not used:
            return None
        v = [m[a]["scores"].get(str(sid)) if m[a]["used"] else None for a in ASPECTS]
        if all(x is None for x in v):
            return None
        return np.asarray([0.5 if x is None else x for x in v], np.float32)

    def criteria_read(self, sid: str) -> dict:
        """Peaks' read of a scene on each criterion: {aspect: {score, band}} (used ones only)."""
        m = self.criteria_models(wait=False)
        out = {}
        for a in ASPECTS:
            s = m[a]["scores"].get(str(sid)) if m[a]["used"] else None
            if s is not None:
                out[a] = {"score": round(s, 3), "band": band(s)}
        return out

    def criteria_for_scene(self, sid: str) -> dict:
        sid = str(sid)
        mine = {k: v for k, v in (self.criteria_reasons().get(sid) or {}).items() if k in ASPECTS}
        try:
            read = self.criteria_read(sid)
        except Exception:  # noqa: BLE001
            read = {}
        return {"scene_id": sid, "reasons": mine, "read": read, "aspects": list(ASPECTS), "words": WORDS}

    def criteria_teach(self, sid: str) -> str | None:
        """The one criterion worth asking about on this scene: the one Peaks is least
        sure of (score nearest the middle), preferring those with the fewest labels."""
        m = self.criteria_models(wait=False)
        mine = self.criteria_reasons().get(str(sid)) or {}
        best, key = None, None
        for a in ASPECTS:
            if a in mine:
                continue
            s = m[a]["scores"].get(str(sid)) if m[a]["used"] else None
            k = (abs((s if s is not None else 0.5) - 0.5), m[a]["labels"])
            if key is None or k < key:
                best, key = a, k
        return best

    # --- what matters to you ------------------------------------------------------------------------

    def criteria_summary(self) -> dict:
        """Per criterion: how many answers, how well its score predicts (held out),
        and how much it moves your grades (rank correlation with the grade)."""
        from ..tier_model import ORDINAL, train_class

        m = self.criteria_models()
        rows = [r for r in self._catalogue_all() if train_class(r["tier"])]
        rep = (self._tier_model_state().get("report") or {})
        out = []
        for a in ASPECTS:
            info = m[a]
            moves = None
            if info["used"]:
                pairs = [(info["scores"][r["scene_id"]], ORDINAL[train_class(r["tier"])]) for r in rows
                         if r["scene_id"] in info["scores"]]
                if len(pairs) >= 20:
                    s, g = (np.argsort(np.argsort(np.asarray(x))) for x in zip(*pairs))
                    moves = round(float(np.corrcoef(s, g)[0, 1]), 3)
            out.append({"aspect": a, "word": WORDS[a], "labels": info["labels"], "yes": info["yes"], "no": info["no"],
                        "acc": info["acc"], "used": info["used"], "moves": moves, "needs": max(0, MIN_LABELS - info["labels"])})
        return {"criteria": out, "in_grading": bool(rep.get("use_criteria")),
                "gain": rep.get("criteria_gain"), "performers_rated": len(self.performer_criteria())}
