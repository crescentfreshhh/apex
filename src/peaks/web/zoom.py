"""Moment zoom: picks become stretches of time, judged by how they unfold.

The main index samples a frame every few seconds and Peaks picks single frames. Here
each scene's candidate moments — its top taste picks, plus wherever you saved a
moment — get a dense look: a frame every ZOOM_STEP s over ±ZOOM_HALF s, decoded
in one ffmpeg run per window and embedded with the active model. They're kept in
the embedding cache under their own namespace ("<model>@zoom"), so the main
index is untouched. When the library itself is sampled at least that densely
(say every 2 s) the windows are simply read out of the main index instead.

Each window is described by its shape (peaks.moment_shape: build-up, hold,
movement, and your concepts), and a small model learns which shapes you save
and 👍. It's scored honestly — scenes held out, its picks against today's frame
picks, same hit rule as the Moments tab — and Peaks only ranks by shape when
it actually wins on your data. Either way a zoomed scene's picks become spans:
a start, a peak and an end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..moment_shape import FEATURES, WORDS, shape_features, span
from .hear import SOUND_WORDS

SENSE_WORDS = {**SOUND_WORDS}

ZOOM_HALF = 20.0          # seconds either side of a candidate
ZOOM_STEP = 2.0           # one frame every … seconds inside a window
ZOOM_CANDIDATES = 5       # taste picks per scene (plus your saved moments)
ZOOM_PER_RUN = 150        # scenes per background run
HIT = 15.0                # same as the Moments tab: a save within this of a pick
SWITCH_MIN_SCENES = 30
SWITCH_MARGIN = 0.02


class _Lazy:
    """Builds the embedder / sampler on first use — a dense library never needs them."""

    def __init__(self, make):
        self._make, self._obj = make, None

    def __getattr__(self, name):
        if self._obj is None:
            self._obj = self._make()
        return getattr(self._obj, name)


class ZoomMixin:
    # --- state ---------------------------------------------------------------------------

    def _zoom_ns(self) -> str:
        return f"{self._model_name()}@zoom"

    def _zoom_cache(self):
        from ..cache import EmbeddingCache

        return EmbeddingCache(self.cfg.embedding.cache_dir)

    def _zoom_path(self) -> Path:
        # one file per model: after a backbone switch (the Lab) zooming starts over
        # on the new index, and switching back finds the old work where it was
        model = self._model_name()
        legacy = self._state_dir() / "zoom.json"
        if legacy.exists() and not (self._state_dir() / f"zoom-{model}.json").exists():
            try:
                if json.loads(legacy.read_text()).get("ns", f"{model}@zoom") == f"{model}@zoom":
                    return legacy
            except (OSError, ValueError):
                pass
        return self._state_dir() / f"zoom-{model}.json"

    def _zoom_state(self) -> dict:
        try:
            d = json.loads(self._zoom_path().read_text())
        except (OSError, ValueError):
            d = {}
        d.setdefault("scenes", {})
        return d

    def _zoom_save(self, d: dict) -> None:
        from .. import atomic

        d["ns"] = self._zoom_ns()
        p = self._zoom_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic.write_json(p, d, indent=1)

    def zoom_settings(self) -> dict:
        z = self._settings().get("zoom") or {}
        return {"per_run": int(z.get("per_run") or ZOOM_PER_RUN), "on": bool(z.get("on", True))}

    # --- scoring a block of frames like the library is scored ----------------------------

    def _taste_block(self, vecs: np.ndarray) -> np.ndarray | None:
        """Taste for arbitrary frames: your trained classifier, else the nearest loved mode."""
        model, profile = self._model_name(), self.cfg.markers.tag_name
        clf = self._taste_model(profile, model)
        v = np.asarray(vecs, np.float32)
        if clf is not None:
            from ..classifier import with_context

            w = int(getattr(clf, "context", 0) or 0)
            return np.asarray(clf.predict_proba(with_context(v, w) if w else v), np.float32).reshape(-1)
        modes = self._taste_modes(model, profile=profile)
        if modes is None:
            return None
        return (v @ np.asarray(modes, np.float32).T).max(axis=1)

    def _zoom_hold_level(self) -> float:
        """Your library's 75th-percentile taste: 'holding' means staying above it."""
        cached = self.__dict__.get("_zoom_hold")
        idx = self.index(self._model_name())
        if cached and cached[0] == id(idx):
            return cached[1]
        try:
            scores, _ = self._taste_scores(self._model_name())
        except Exception:  # noqa: BLE001
            scores = None
        lvl = float(np.quantile(scores, 0.75)) if scores is not None and len(scores) else 0.5
        self._zoom_hold = (id(idx), lvl)
        return lvl

    # --- which scenes, which windows -------------------------------------------------------

    def _zoom_saved(self) -> dict[str, list[float]]:
        """scene_id → the times you saved moments at."""
        out: dict[str, list[float]] = {}
        try:
            for mk in self._saved_markers(self.cfg.markers.tag_name):
                if mk.get("scene_id"):
                    out.setdefault(str(mk["scene_id"]), []).append(float(mk.get("seconds") or 0.0))
        except Exception:  # noqa: BLE001 — Stash down: no saves this round
            pass
        return out

    def zoom_candidates(self, sid: str, saved: list[float] | None = None) -> list[dict]:
        """[{c: centre, kind: "pick"|"saved"}]: the top taste picks, plus your saves
        (a saved moment near a pick replaces nothing — both are kept, a pick is a pick)."""
        strip = self._scene_moment_strips([sid], n=ZOOM_CANDIDATES, min_gap=20.0).get(sid) or []
        picks = [{"c": float(m["t"]), "kind": "pick", "taste": m.get("score")} for m in strip]
        extra = [{"c": float(t), "kind": "saved"} for t in sorted(set(saved or []))
                 if all(abs(float(t) - p["c"]) > 3.0 for p in picks)]
        return picks + extra

    def zoom_todo(self) -> list[dict]:
        """Scenes to zoom, most useful first: ones you've saved in (they teach the
        model), then unreviewed, then keepers. Rejects never. A scene is redone
        when a new save isn't covered yet."""
        st = self._zoom_state()["scenes"]
        saved = self._zoom_saved()
        idx = self.index(self._model_name())
        have = {str(m.get("scene_id")) for m in idx.key_meta.values()}
        out = []
        for r in self._catalogue_all():
            sid = r["scene_id"]
            if r["tier"] == "rejected" or sid not in have or not r.get("path"):
                continue
            e = st.get(sid)
            covered = e is not None and all(any(abs(t - c) <= 3.0 for c in e.get("centers", []))
                                            for t in saved.get(sid, []))
            if e is not None and covered:
                continue
            rank = 0 if sid in saved else 1 if r["tier"] == "unreviewed" else 2
            out.append((rank, r.get("created_at") or "", r))
        out.sort(key=lambda x: x[1], reverse=True)            # newest first within…
        out.sort(key=lambda x: x[0])                          # …saved, unreviewed, keepers
        return [r for _, _, r in out]

    def zoom_due(self) -> bool:
        if not self.zoom_settings()["on"]:
            return False
        try:
            return bool(self.zoom_todo())
        except Exception:  # noqa: BLE001
            return False

    # --- the job -------------------------------------------------------------------------------

    def _zoom_sampler(self):
        from ..sampling import FrameSampler

        return FrameSampler(hwaccel=self.cfg.sampling.hwaccel)

    def zoom_scene(self, row: dict, saved: list[float] | None = None, embedder=None, sampler=None) -> dict:
        """Dense frames around each candidate of one scene, embedded and cached."""
        sid = row["scene_id"]
        idx = self.index(self._model_name())
        key = next((k for k, m in idx.key_meta.items() if str(m.get("scene_id")) == sid), None)
        if key is None:
            raise LookupError(f"scene {sid} isn't embedded")
        cands = self.zoom_candidates(sid, saved)
        times, vecs = [], []
        seen = set()
        dense = self._zoom_dense(key)
        if dense is not None:                                  # the index is already this dense: read it
            it, iv = dense
            inside = np.zeros(len(it), bool)
            for c in cands:
                inside |= (it >= c["c"] - ZOOM_HALF) & (it < c["c"] + ZOOM_HALF)
            times, vecs = list(it[inside]), list(iv[inside])
        todo = [] if dense is not None else cands
        embedder = embedder or (self._embedder() if todo else None)
        sampler = sampler or (self._zoom_sampler() if todo else None)
        for c in todo:
            frames = sampler.window_frames(row["path"], c["c"] - ZOOM_HALF, 2 * ZOOM_HALF, ZOOM_STEP)
            frames = [(t, im) for t, im in frames if round(t, 1) not in seen]
            if not frames:
                continue
            v = embedder.embed_images([im for _, im in frames])
            for (t, _), x in zip(frames, v):
                seen.add(round(t, 1))
                times.append(t)
                vecs.append(x)
        order = np.argsort(times)
        self._zoom_cache().save(key, self._zoom_ns(), np.asarray(times)[order],
                                np.asarray(vecs, np.float32)[order] if vecs else np.zeros((0, idx.dim), np.float32),
                                meta={"scene_id": sid, "candidates": cands})
        d = self._zoom_state()
        d["scenes"][sid] = {"key": key, "centers": [c["c"] for c in cands], "at": time.time(), "frames": len(times)}
        self._zoom_save(d)
        self.__dict__.pop("_zoom_rows_cache", None)
        return d["scenes"][sid]

    def _zoom_dense(self, key: str):
        """(times, vecs) straight from the main index when it was sampled at least
        as densely as a zoom (e.g. a 2 s library) — no decoding, no embedding."""
        from ..cache import EmbeddingCache

        cache, model = EmbeddingCache(self.cfg.embedding.cache_dir), self._model_name()
        try:
            iv = cache.peek_meta(key, model).get("interval")
            if iv is None or float(iv) > ZOOM_STEP + 1e-6:
                return None
            times, vecs, _ = cache.load(key, model)
        except (OSError, KeyError, ValueError, TypeError):
            return None
        return np.asarray(times, np.float32), vecs

    def _zoom_library_dense(self) -> bool:
        try:
            return float(self._active_sampling()[1] or 99) <= ZOOM_STEP + 1e-6
        except Exception:  # noqa: BLE001
            return False

    def run_zoom(self, job=None, limit: int | None = None) -> dict:
        log = job.log if job else print
        per_run = self.zoom_settings()["per_run"] * (40 if self._zoom_library_dense() else 1)   # reading is cheap
        todo = self.zoom_todo()[: limit or per_run]
        saved = self._zoom_saved()
        if not todo:
            return {"zoomed": 0}
        embedder, sampler = _Lazy(self._embedder), _Lazy(self._zoom_sampler)
        done = failed = 0
        t0 = time.time()
        for i, row in enumerate(todo):
            if job is not None and job.cancelled:
                log("stopped — the rest waits for the next run")
                break
            try:
                self.zoom_scene(row, saved.get(row["scene_id"]), embedder=embedder, sampler=sampler)
                done += 1
            except Exception as exc:  # noqa: BLE001 — one unreadable file mustn't stop the run
                failed += 1
                log(f"scene {row['scene_id']}: {exc}")
            if job is not None:
                rate = (time.time() - t0) / (i + 1)
                job.progress = {"total": len(todo), "done": i + 1, "eta_s": round(rate * (len(todo) - i - 1))}
        log(f"zoomed {done} scene(s){f', {failed} unreadable' if failed else ''}")
        del embedder
        try:
            rep = self.train_moment_model()
            log("moment model: " + (rep.get("verdict") or rep.get("reason") or "trained"))
        except Exception as exc:  # noqa: BLE001
            log(f"moment model: {exc}")
        return {"zoomed": done, "failed": failed}

    # --- windows, features, labels -----------------------------------------------------------

    def _zoom_windows(self, sid: str, entry: dict | None = None) -> list[dict]:
        """A zoomed scene's windows with their shape: [{c, kind, span, feats, x}]."""
        entry = entry or self._zoom_state()["scenes"].get(sid)
        if not entry:
            return []
        try:
            times, vecs, meta = self._zoom_cache().load(entry["key"], self._zoom_ns())
        except (OSError, KeyError, ValueError):
            return []
        if not len(times):
            return []
        s = self._taste_block(vecs)
        if s is None:
            return []
        level = self._zoom_hold_level()
        concepts = []
        try:
            for slug, _ in self.trained_concepts():
                w, b = self._cx_probe(slug)
                concepts.append((slug, 1.0 / (1.0 + np.exp(-(vecs @ w + b)))))
        except Exception:  # noqa: BLE001
            concepts = []
        out = []
        for c in meta.get("candidates") or []:
            m = np.abs(times - c["c"]) <= ZOOM_HALF + 0.01
            if m.sum() < 2:
                continue
            f = shape_features(times[m], s[m], vecs[m], level)
            for slug, p in concepts:
                f[f"concept:{slug}"] = float(p[m].max())
            sp = span(times[m], s[m])
            f.update(self._window_senses(sid, c["c"], sp[1]))
            out.append({"c": c["c"], "kind": c["kind"], "taste": c.get("taste"), "span": sp, "feats": f})
        return out

    def _window_senses(self, sid: str, centre: float, peak: float) -> dict:
        """Sound (and later motion) features of one window; {} when not heard/seen."""
        out = {}
        for fn in (self.sound_window_features,):
            try:
                f = fn(sid, centre, peak, ZOOM_HALF)
            except Exception:  # noqa: BLE001 — a sense is extra, never in the way
                f = None
            if f:
                out.update(f)
        return out

    def _zoom_feature_groups(self) -> dict[str, list[str]]:
        """Feature columns by group: the shape (+ your concepts), then each sense."""
        from .hear import SOUND_FEATURES

        shape = list(FEATURES)
        try:
            shape += [f"concept:{slug}" for slug, _ in self.trained_concepts()]
        except Exception:  # noqa: BLE001
            pass
        return {"shape": shape, "sound": list(SOUND_FEATURES)}

    def _zoom_feature_names(self) -> list[str]:
        return [n for cols in self._zoom_feature_groups().values() for n in cols]

    def _zoom_dataset(self):
        """Every zoomed window with a label where there is one: (rows, X, y, weight, scene)."""
        names = self._zoom_feature_names()
        saved = self._zoom_saved()
        fb = {}
        for x in self._mc_read("pick_feedback.jsonl"):
            fb.setdefault(x["scene_id"], []).append((float(x["t"]), int(x["label"])))
        rows, X, y, wt, grp = [], [], [], [], []
        for sid, entry in self._zoom_state()["scenes"].items():
            ws = self._zoom_windows(sid, entry)
            sv = saved.get(sid, [])
            for w in ws:
                peak = w["span"][1]
                lab, weight = None, 1.0
                if any(abs(t - peak) <= HIT or abs(t - w["c"]) <= 3.0 for t in sv):
                    lab = 1
                for t, l in fb.get(sid, []):
                    if abs(t - w["c"]) <= 3.0:
                        lab = l
                if lab is None and sv and w["kind"] == "pick":
                    lab, weight = 0, 0.5                # a candidate you didn't save, in a scene you did
                rows.append({"sid": sid, **w})
                X.append([w["feats"].get(n, 0.0) for n in names])
                y.append(-1 if lab is None else lab)
                wt.append(weight)
                grp.append(sid)
        return rows, np.asarray(X, np.float32).reshape(len(rows), len(names)), np.asarray(y), np.asarray(wt), grp, names

    # --- the model ------------------------------------------------------------------------------

    def train_moment_model(self) -> dict:
        """Fit the moment model and score it honestly against frame picks."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rows, X, y, wt, grp, names = self._zoom_dataset()
        lab = y >= 0
        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        rep = {"at": time.time(), "windows": len(rows), "scenes": len(set(grp)), "yes": pos, "no": neg}
        if pos < 5 or neg < 5:
            rep.update(trained=False, use_shape=False,
                       reason="needs saved moments or 👍/👎 on picks in zoomed scenes (at least 5 of each)")
            return self._zoom_store_model(None, rep)
        saved = self._zoom_saved()
        g = np.asarray(grp)
        scenes = sorted(set(g[lab]))
        groups = self._zoom_feature_groups()
        col = {n: i for i, n in enumerate(names)}
        # which senses have any data at all in these windows
        present = {k: v for k, v in groups.items()
                   if k == "shape" or (X[:, [col[n] for n in v]] != 0).any()}
        sets = {"shape": present["shape"]}
        for k in present:
            if k != "shape":
                sets[f"shape + {k}"] = present["shape"] + present[k]
        if len(present) > 2:
            sets["all"] = [n for k in present for n in present[k]]
        results, probs = {}, {}
        frame = None
        for label, cols in sets.items():
            idx = [col[n] for n in cols]
            prob = self._mm_heldout(X[:, idx], y, wt, g, lab, scenes)
            n, f1, f3, s1, s3 = self._mm_hits(rows, prob, scenes, saved)
            frame = frame or (n, f1, f3)
            results[label] = {"top1": s1, "top3": s3}
            probs[label] = prob
        n, f1, f3 = frame
        # the simplest set that isn't clearly beaten: a richer one must win by the margin
        chosen = "shape"
        for label in sets:
            if label != chosen and results[label]["top3"] is not None and results[chosen]["top3"] is not None \
                    and results[label]["top3"] >= results[chosen]["top3"] + SWITCH_MARGIN:
                chosen = label
        names = sets[chosen]
        Xc = X[:, [col[c] for c in names]]
        sc = StandardScaler().fit(Xc[lab])
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(Xc[lab]), y[lab],
                                                                                     sample_weight=wt[lab])
        best = results[chosen]
        use = bool(n >= SWITCH_MIN_SCENES and best["top3"] is not None and f3 is not None
                   and best["top3"] >= f3 + SWITCH_MARGIN)
        rep.update(trained=True, compared=n, frame={"top1": f1, "top3": f3},
                   shape=results["shape"], sets=results, uses=chosen, use_shape=use,
                   verdict=((f"shape picks win — Peaks now ranks moments by {chosen}" if chosen != "shape"
                             else "shape picks win — Peaks now ranks moments by shape") if use else
                            f"not better yet — Peaks keeps frame picks{'' if n >= SWITCH_MIN_SCENES else f' (compared on {n} of the {SWITCH_MIN_SCENES} scenes it needs)'}"),
                   weights=self._zoom_words(names, m.coef_[0]))
        return self._zoom_store_model((sc, m, names), rep)

    @staticmethod
    def _mm_heldout(X, y, wt, g, lab, scenes) -> np.ndarray:
        """Each window's probability from a model that never saw its scene."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler

        k = max(2, min(5, len(scenes)))
        prob = np.full(len(y), np.nan)
        for tr_s, te_s in GroupKFold(n_splits=k).split(scenes, groups=scenes):
            tr = lab & np.isin(g, [scenes[i] for i in tr_s])
            te = np.isin(g, [scenes[i] for i in te_s])
            if len(set(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(X[tr]), y[tr],
                                                                                         sample_weight=wt[tr])
            prob[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        return prob

    @staticmethod
    def _mm_hits(rows, prob, scenes, saved):
        """Held-out scenes with saves: (n, frame top1, frame top3, model top1, model top3)."""
        s1 = s3 = f1 = f3 = n = 0
        for sid in scenes:
            sv = saved.get(sid, [])
            ws = [(i, r) for i, r in enumerate(rows) if r["sid"] == sid and r["kind"] == "pick" and not np.isnan(prob[i])]
            if not sv or not ws:
                continue
            n += 1
            by_taste = [r["c"] for _, r in sorted(ws, key=lambda x: -(x[1]["taste"] or 0))]
            # the same point for both rankings — the candidate itself — so only the order differs
            by_shape = [r["c"] for i, r in sorted(ws, key=lambda x: -prob[x[0]])]

            def hit(picks, kk):
                return any(abs(p - t) <= HIT for p in picks[:kk] for t in sv)
            f1 += hit(by_taste, 1)
            f3 += hit(by_taste, 3)
            s1 += hit(by_shape, 1)
            s3 += hit(by_shape, 3)
        r3 = lambda a: round(a / n, 3) if n else None  # noqa: E731
        return n, r3(f1), r3(f3), r3(s1), r3(s3)

    @staticmethod
    def _zoom_words(names, coef) -> list[dict]:
        out = []
        for n, w in sorted(zip(names, coef), key=lambda x: -abs(x[1])):
            if abs(w) < 0.25:
                continue
            word = WORDS.get(n) or SENSE_WORDS.get(n) or (
                f"◇ {n.split(':', 1)[1].replace('-', ' ')}" if n.startswith("concept:") else n)
            out.append({"feature": n, "words": word, "weight": round(float(w), 2)})
        return out[:6]

    def _zoom_model_path(self) -> Path:
        return self._zoom_path().with_name(self._zoom_path().stem.replace("zoom", "moment_model", 1) + ".npz")

    def _zoom_store_model(self, fitted, rep: dict) -> dict:
        p = self._zoom_model_path()
        if fitted is not None:
            sc, m, names = fitted
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, mean=sc.mean_, scale=sc.scale_, coef=m.coef_[0], icpt=m.intercept_, names=np.array(names))
        d = self._zoom_state()
        d["model"] = rep
        self._zoom_save(d)
        self.__dict__.pop("_zoom_model_cache", None)
        return rep

    def _zoom_model(self):
        """(names, mean, scale, coef, intercept) when shape ranking is on, else None."""
        if not (self._zoom_state().get("model") or {}).get("use_shape"):
            return None
        p = self._zoom_model_path()
        hit = self.__dict__.get("_zoom_model_cache")
        if hit is not None and hit[0] == p:
            return hit[1]
        try:
            z = np.load(p)
            hit = ([str(x) for x in z["names"]], z["mean"], z["scale"], z["coef"], float(z["icpt"][0]))
        except (OSError, ValueError, KeyError):
            return None
        self._zoom_model_cache = (p, hit)
        return hit

    # --- spans for the players ------------------------------------------------------------------

    def zoom_picks(self, sid: str) -> list[dict] | None:
        """A zoomed scene's picks as spans, best first (by shape when it's switched
        on, else by taste). None when the scene isn't zoomed (frame picks then)."""
        ws = [w for w in self._zoom_windows(sid) if w["kind"] == "pick"]
        if not ws:
            return None
        mdl = self._zoom_model()
        for w in ws:
            if mdl is not None:
                names, mean, scale, coef, b = mdl
                x = (np.array([w["feats"].get(n, 0.0) for n in names]) - mean) / scale
                w["score"] = float(1 / (1 + np.exp(-(x @ coef + b))))
            else:
                w["score"] = float(w["feats"]["peak"])
        ws.sort(key=lambda w: -w["score"])
        key = (self._zoom_state()["scenes"].get(sid) or {}).get("key")
        return [{"t": w["span"][1], "start": w["span"][0], "end": w["span"][2], "score": round(w["score"], 4),
                 "shape": mdl is not None, "key": key,
                 "thumb": f"/api/frame?key={key}&t={w['span'][1]:g}" if key else None} for w in ws]

    def zoom_status(self) -> dict:
        d = self._zoom_state()
        try:
            idx = self.index(self._model_name())
            total = len({str(m.get("scene_id")) for m in idx.key_meta.values()})
        except Exception:  # noqa: BLE001
            total = None
        return {"zoomed": len(d["scenes"]), "embedded": total, "model": d.get("model"),
                "settings": self.zoom_settings()}
