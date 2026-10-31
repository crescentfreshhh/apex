"""The Lab: a new model has to beat the current one on *your* library first.

A trial embeds a sample of your graded scenes (stratified by tier, plus every
scene with saved moments) with the candidate, at the library's own sampling, in
the candidate's own cache namespace. Then both models are judged on exactly the
same scenes, the same way:

  - grading: a tier model from the visual features alone, each scene graded by
    a fold that never saw it — accuracy, within-one-tier, log-loss;
  - moments: each saved moment looked for in its scene by "nearest loved moment
    in other scenes" — first pick / top 3 within 15 s, and a tight 4 s hit.

The candidate wins only when it's clearly better — grading +2 pts, or moments
+5 pts — *and* the scenes it alone gets right outnumber the ones only the
current model gets by more than luck would (a sign test), with the other measure
not clearly worse. Small samples swing; this keeps a lucky draw from switching. Switching keeps the sample's
vectors — the re-embed starts that far in — and the old cache stays on disk, so
switching back is instant.
"""

from __future__ import annotations

import hashlib
import json
import time
from math import comb
from pathlib import Path

import numpy as np

CANDIDATES = ("dinov3_vitl16", "dinov3_vith16plus", "siglip2_so400m", "pe_core_l")
PER_TIER = 60             # graded scenes per tier in a trial
SAMPLE_CAP = 400          # scenes per trial, saved-moment scenes included
GRADE_WIN = 0.02          # grading accuracy margin for a win
MOMENT_WIN = 0.05         # … or this much better at finding your moments
MIN_SAVES = 10            # moments only count with at least this many saves
HIT, TIGHT, PICK_GAP, TOP_K = 15.0, 4.0, 20.0, 3


def _stable(sid: str) -> str:
    return hashlib.sha1(str(sid).encode()).hexdigest()


class LabMixin:
    # --- state ---------------------------------------------------------------------------

    def _lab_path(self) -> Path:
        return self._state_dir() / "lab.json"

    def _lab_state(self) -> dict:
        try:
            d = json.loads(self._lab_path().read_text())
        except (OSError, ValueError):
            d = {}
        d.setdefault("trials", {})
        return d

    def _lab_save(self, d: dict) -> None:
        from .. import atomic

        self._lab_path().parent.mkdir(parents=True, exist_ok=True)
        atomic.write_json(self._lab_path(), d, indent=1)

    def lab(self) -> dict:
        """Every candidate with its latest trial, and what's active now."""
        from ..embedding import dino_label

        active = self._active_dino_model()
        d = self._lab_state()
        out = []
        for c in CANDIDATES:
            t = d["trials"].get(c)
            out.append({"id": c, "label": dino_label(c), "kind": "dino", "active": c == active,
                        "licence": c.startswith("dinov3_"),        # Meta's licence + HF_TOKEN
                        "data_deleted": bool(t and t.get("data_deleted")),
                        "trial": t, "stale": bool(t and t.get("against") != active),
                        "error": (d.get("errors", {}).get(c) or {}).get("error"),
                        "sample": self.lab_sample_progress(c) if c != active else None})
        return {"active": active, "active_label": dino_label(active), "candidates": out,
                "sample_cap": SAMPLE_CAP, "senses": self.lab_senses()}

    def lab_senses(self) -> list[dict]:
        """The extra senses (hearing, motion): coverage, and what the honest tests
        say they add to finding moments and to grading."""
        mm = (self._zoom_view().get("model") or {})
        sets = mm.get("sets") or {}
        tier = (self._tier_model_state().get("report") or {})
        out = []
        for sid, label, status in (("sound", "Hearing", self.hear_status), ("motion", "Motion", self.motion_status)):
            try:
                st = status()
            except Exception as exc:  # noqa: BLE001
                st = {"error": str(exc)}
            key = f"shape + {sid}"
            out.append({
                "id": sid, "label": label, "status": st,
                "moments": {"without": sets.get("shape"), "with": sets.get(key),
                            "in_use": bool(mm.get("use_shape") and mm.get("uses") in (key, "all"))},
                "grading": {"without": ((tier.get(f"cv_without_{sid}") if tier.get(f"use_{sid}")
                                         else tier.get("cv")) or {}).get("exact"),
                            "with": (tier.get(f"cv_with_{sid}") or {}).get("exact"),
                            "gain": tier.get(f"{sid}_gain"), "in_use": bool(tier.get(f"use_{sid}"))},
            })
        return out

    # --- the sample ------------------------------------------------------------------------

    def lab_sample(self) -> list[dict]:
        """Graded scenes embedded with the current model: up to PER_TIER per tier
        (a fixed, stable choice so re-runs compare like with like) plus the
        scenes with saved moments, as many as fit — SAMPLE_CAP in all."""
        from ..tier_model import train_class

        idx = self.index(self._model_name())
        have = {str(m.get("scene_id")) for m in idx.key_meta.values()}
        saved = self._zoom_saved()
        rows = [r for r in self._catalogue_all() if r["scene_id"] in have]
        by_tier: dict[str, list[dict]] = {}
        for r in rows:
            c = train_class(r["tier"])
            if c:
                by_tier.setdefault(c, []).append(r)
        # every tier gets its share first (grading needs 2+ tiers to judge), then your
        # saved-moment scenes fill the room left — saves are mostly top-tier scenes,
        # and with more of them than the cap they'd otherwise crowd the others out
        per = min(PER_TIER, max(1, SAMPLE_CAP // max(1, len(by_tier))))
        pick: dict[str, dict] = {}
        for c, rs in by_tier.items():
            for r in sorted(rs, key=lambda r: _stable(r["scene_id"]))[:per]:
                pick[r["scene_id"]] = r
        for r in sorted((r for r in rows if r["scene_id"] in saved), key=lambda r: _stable(r["scene_id"])):
            if len(pick) >= SAMPLE_CAP:
                break
            pick.setdefault(r["scene_id"], r)
        return list(pick.values())

    # --- embedding the sample ----------------------------------------------------------------

    def _lab_embedder(self, cand: str):
        from ..embedding import make_dino

        return make_dino(cand, device=self.cfg.embedding.device or None)

    def _lab_sampler(self):
        from ..sampling import FrameSampler

        s = self.cfg.sampling
        mode, iv = self._active_sampling()
        return FrameSampler(interval_seconds=iv, mode=mode, hwaccel=s.hwaccel, pipeline=s.pipeline,
                            scene_timeout=s.scene_timeout)

    # --- trial data on disk -----------------------------------------------------------------

    def _lab_ns_dirs(self) -> dict[str, Path]:
        """Candidate model → its cache folder (only those on disk)."""
        from ..embedding import dino_cache_name

        root = Path(self.cfg.embedding.cache_dir)
        out = {}
        for c in CANDIDATES:
            d = root / dino_cache_name(c)
            if d.is_dir():
                out[c] = d
        return out

    @staticmethod
    def _dir_bytes(d: Path) -> tuple[int, int]:
        size = n = 0
        for f in d.rglob("*"):
            try:
                if f.is_file():
                    size += f.stat().st_size
                    n += f.suffix == ".npz"
            except OSError:
                pass
        return size, n

    def lab_trial_data(self, running: str | None = None, queued: str | None = None) -> dict:
        """Every candidate's trial embeddings on disk: size, scenes, and whether it's
        safe to delete (not the active model, no trial running or queued for it)."""
        from ..embedding import dino_label

        active = self._active_dino_model()
        rows = []
        for c, d in self._lab_ns_dirs().items():
            size, n = self._dir_bytes(d)
            busy = c in (running, queued)
            rows.append({"id": c, "label": dino_label(c), "bytes": size, "scenes": n, "active": c == active,
                         "busy": busy, "deletable": c != active and not busy})
        return {"items": rows, "bytes": sum(r["bytes"] for r in rows)}

    def lab_delete_data(self, cand: str, running: str | None = None, queued: str | None = None) -> dict:
        """Delete a candidate's trial embeddings (and any index snapshot of them).
        Its trial result stays; a re-run embeds from scratch."""
        import shutil

        from ..embedding import dino_cache_name

        if cand not in CANDIDATES:
            raise KeyError(cand)
        if cand == self._active_dino_model():
            raise ValueError("that's the model Peaks is using — its embeddings stay")
        if cand in (running, queued):
            raise ValueError("a trial is running or waiting for it — stop it first")
        root = Path(self.cfg.embedding.cache_dir)
        freed = 0
        for d in (root / dino_cache_name(cand), root / "_index" / dino_cache_name(cand)):
            if d.is_dir():
                freed += self._dir_bytes(d)[0]
                shutil.rmtree(d, ignore_errors=True)
        st = self._lab_state()
        if cand in st["trials"]:
            st["trials"][cand]["data_deleted"] = True
            self._lab_save(st)
        return {"deleted": cand, "freed_bytes": freed}

    @staticmethod
    def _lab_drop_broken(cache, ns: str, keys: list[str]) -> int:
        """Delete cached trial scenes whose vectors hold NaN / inf (DINOv3 in float16
        overflowed), so the run embeds them again. Returns how many."""
        n = 0
        for k in keys:
            if not cache.has(k, ns):
                continue
            try:
                _, v, _ = cache.load(k, ns)
                ok = bool(np.isfinite(np.asarray(v, np.float32)).all())
            except (OSError, ValueError, KeyError):
                ok = False
            if not ok:
                cache.delete(k, ns)
                n += 1
        return n

    def lab_sample_progress(self, cand: str) -> dict | None:
        """How much of the trial sample is already embedded with `cand` at the
        library's sampling — the work a next run carries on from."""
        from ..cache import EmbeddingCache
        from ..embedding import dino_cache_name

        try:
            rows = self.lab_sample()
            idx = self.index(self._model_name())
            key_of = {str(m.get("scene_id")): k for k, m in idx.key_meta.items()}
            sampler = self._lab_sampler()
            sig = getattr(sampler, "interval_signature", sampler.interval)
            cache = EmbeddingCache(self.cfg.embedding.cache_dir)
            ns = dino_cache_name(cand)
            done = sum(1 for r in rows if key_of.get(r["scene_id"]) and cache.has(key_of[r["scene_id"]], ns, interval=sig))
            return {"done": done, "total": len(rows)}
        except Exception:  # noqa: BLE001 — a progress line, never an error
            return None

    def _lab_embed(self, cand: str, rows: list[dict], job=None) -> dict:
        """Embed the sample with the candidate into its own namespace, at the
        library's sampling. Resumable: cached scenes are skipped. Returns
        {embedded, frames, seconds} for the speed estimate."""
        from ..cache import EmbeddingCache
        from ..embedding import dino_cache_name, dino_label
        from ..pipeline import embed_library, scene_key

        e = self.cfg.embedding
        sampler = self._lab_sampler()
        embedder = self._lab_embedder(cand)
        cache = EmbeddingCache(e.cache_dir)
        want = {r["scene_id"] for r in rows}
        scenes = [sc for sc in self.scenes() if str(sc.id) in want]
        log = job.log if job else print
        # what an earlier (interrupted) run already embedded is kept and skipped: count it,
        # so a resumed trial starts where it left off rather than looking like zero
        sig = getattr(sampler, "interval_signature", sampler.interval)
        ns = dino_cache_name(cand)
        idx = self.index(self._model_name())
        key_of = {str(m.get("scene_id")): k for k, m in idx.key_meta.items()}
        bad = self._lab_drop_broken(cache, ns, [key_of[str(sc.id)] for sc in scenes if str(sc.id) in key_of])
        if bad:
            log(f"{bad} scene(s) were embedded with broken (NaN) frames by an earlier version — embedding them again")
        already = sum(1 for sc in scenes
                      if (k := key_of.get(str(sc.id)) or (scene_key(sc) if getattr(sc, "fingerprint", None) else None))
                      and cache.has(k, ns, interval=sig))
        if already:
            log(f"resuming — {already} of {len(scenes)} already embedded with {dino_label(cand)}, carrying on")
        if job is not None:
            job.progress = {"total": len(scenes), "done": already, "stage": "embedding", "cand": cand}

        def _log(msg):
            log(msg)
            if job is not None and msg.lstrip().startswith("+"):
                job.progress["done"] = job.progress.get("done", 0) + 1

        t0 = time.time()
        stats = embed_library(scenes, sampler, embedder, cache, batch_size=e.batch_size, workers=e.workers,
                              total=len(scenes), log=_log,
                              should_stop=(lambda: job.cancelled) if job is not None else None)
        return {"embedded": int(stats.get("embedded", 0)), "frames": int(stats.get("frames", 0)),
                "seconds": round(time.time() - t0, 1)}

    def _lab_vectors(self, ns: str, rows: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """scene_id → (times, unit vectors) from one cache namespace."""
        from ..cache import EmbeddingCache

        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        idx = self.index(self._model_name())
        key_of = {str(m.get("scene_id")): k for k, m in idx.key_meta.items()}
        out = {}
        for r in rows:
            k = key_of.get(r["scene_id"])
            if not k or not cache.has(k, ns):
                continue
            t, v, _ = cache.load(k, ns)
            v = np.asarray(v, np.float32)
            ok = np.isfinite(v).all(axis=1)                 # a broken frame never reaches judging
            t, v = np.asarray(t)[ok], v[ok]
            if len(t):
                v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
                # float16 halves the sample held in memory; scoring widens one scene at a time
                out[r["scene_id"]] = (np.asarray(t, np.float32), v.astype(np.float16))
        return out

    # --- judging -------------------------------------------------------------------------------

    @staticmethod
    def _lab_grading(vecs: dict, rows: list[dict]) -> dict | None:
        """Honest grading from the visual features alone: each scene judged by a
        fold that never saw it."""
        from ..tier_model import TierModel, fit_folds, live_ordinal, train_class, usable_classes, visual_features

        rs = [r for r in rows if train_class(r["tier"]) and r["scene_id"] in vecs]
        y = [train_class(r["tier"]) for r in rs]
        ok, _ = usable_classes(y)
        keep = [i for i, c in enumerate(y) if c in ok]
        if len(ok) < 2 or len(keep) < 20:
            return None
        y = [y[i] for i in keep]
        Xv = np.stack([visual_features(vecs[rs[i]["scene_id"]][1]) for i in keep])
        Xq = np.zeros((len(keep), 1), np.float32)
        m = TierModel(use_quality=False, pca_dim=max(2, min(48, len(keep) // 4)))
        m.fit(Xv, Xq, y)
        _, proba, _ = fit_folds(m, Xv, Xq, y)
        pred = [m.classes_[int(j)] for j in proba.argmax(axis=1)]
        live = live_ordinal(set(y))
        p_true = np.array([proba[i, m.classes_.index(c)] for i, c in enumerate(y)])
        return {"n": len(y),
                "accuracy": round(float(np.mean([p == t for p, t in zip(pred, y)])), 4),
                "within_one": round(float(np.mean([abs(live[p] - live[t]) <= 1 for p, t in zip(pred, y)])), 4),
                "log_loss": round(float(-np.mean(np.log(np.clip(p_true, 1e-6, 1)))), 4),
                "right": sorted(rs[keep[i]]["scene_id"] for i, (p, t) in enumerate(zip(pred, y)) if p == t)}

    @staticmethod
    def _lab_moments(vecs: dict, saved: dict[str, list[float]], only: set | None = None) -> dict | None:
        """Find each saved moment by its nearest loved moment in *other* scenes."""
        bank, owner, items = [], [], []
        for sid, ts in saved.items():
            if sid not in vecs or (only is not None and sid not in only):
                continue
            t, v = vecs[sid]
            for s in ts:
                j = int(np.argmin(np.abs(t - s)))
                if abs(float(t[j]) - s) <= TIGHT:
                    bank.append(v[j])
                    owner.append(sid)
                    items.append((sid, s))
        if len(items) < MIN_SAVES or len(set(owner)) < 2:
            return None
        B, owner = np.stack(bank).astype(np.float32), np.array(owner)
        top1 = top3 = tight = 0
        found = []
        picks_of: dict[str, list[float]] = {}
        for sid, s in items:
            if sid not in picks_of:
                t, v = vecs[sid]
                others = owner != sid
                score = (v.astype(np.float32) @ B[others].T).max(axis=1) if others.any() else np.zeros(len(t))
                picks: list[float] = []
                for j in np.argsort(-score):
                    if all(abs(float(t[j]) - p) >= PICK_GAP for p in picks):
                        picks.append(float(t[j]))
                    if len(picks) >= TOP_K:
                        break
                picks_of[sid] = picks
            picks = picks_of[sid]
            top1 += bool(picks) and abs(picks[0] - s) <= HIT
            hit = any(abs(p - s) <= HIT for p in picks)
            top3 += hit
            if hit:
                found.append(f"{sid}@{s:g}")
            tight += any(abs(p - s) <= TIGHT for p in picks)
        n = len(items)
        return {"n": n, "scenes": len(picks_of), "top1": round(top1 / n, 4), "top3": round(top3 / n, 4),
                "tight": round(tight / n, 4), "right": sorted(found)}

    @staticmethod
    def _lab_paired(a: dict | None, b: dict | None) -> dict | None:
        """What only one side got right, and whether that's more than luck: an
        exact one-sided sign test on the disagreements (p ≤ 0.05)."""
        if not a or not b:
            return None
        ra, rb = set(a.pop("right", [])), set(b.pop("right", []))
        only_b, only_a = len(rb - ra), len(ra - rb)
        n = only_b + only_a

        def tail(k: int) -> float:                     # P(at least k of n by a coin flip)
            return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0
        p_better, p_worse = tail(only_b), tail(only_a)
        return {"only_candidate": only_b, "only_current": only_a, "p": round(min(p_better, p_worse), 4),
                "better": p_better <= 0.05, "worse": p_worse <= 0.05}

    @staticmethod
    def _lab_verdict(label: str, cur: dict, new: dict, pg: dict | None, pm: dict | None) -> dict:
        g0, g1 = cur.get("grading"), new.get("grading")
        m0, m1 = cur.get("moments"), new.get("moments")
        if not g0 or not g1:
            return {"win": False, "verdict": "not enough graded scenes to judge yet (needs 20+ in two tiers)"}
        dg = g1["accuracy"] - g0["accuracy"]
        dm = (m1["top3"] - m0["top3"]) if (m0 and m1) else None
        g_win = dg >= GRADE_WIN and bool(pg and pg["better"])
        m_win = dm is not None and dm >= MOMENT_WIN and bool(pm and pm["better"])
        win = (g_win and not (pm and pm["worse"])) or (m_win and not (pg and pg["worse"]))
        pts = lambda x: f"{'+' if x >= 0 else '−'}{abs(round(100 * x))} pts"  # noqa: E731
        bits = [f"grades {pts(dg)}"] + ([f"finds your moments {pts(dm)}"] if dm is not None else [])
        return {"win": bool(win), "d_grading": round(dg, 4), "d_moments": None if dm is None else round(dm, 4),
                "verdict": (f"{label} wins — " if win else f"{label} is no better yet — stay · ") + ", ".join(bits)}

    # --- the trial ---------------------------------------------------------------------------

    def run_lab(self, cand: str, job=None) -> dict:
        """One trial; a failure (no licence/token, …) is kept for the Lab card."""
        try:
            out = self._run_lab(cand, job)
        except Exception as exc:  # noqa: BLE001
            d = self._lab_state()
            d.setdefault("errors", {})[cand] = {"error": str(exc), "at": time.time()}
            self._lab_save(d)
            raise
        d = self._lab_state()
        if d.get("errors", {}).pop(cand, None) is not None:
            self._lab_save(d)
        return out

    def _run_lab(self, cand: str, job=None) -> dict:
        from ..embedding import dino_cache_name, dino_label

        if cand not in CANDIDATES:
            raise ValueError(f"unknown candidate: {cand}")
        active = self._active_dino_model()
        if cand == active:
            raise ValueError(f"{dino_label(cand)} is already the active model")
        log = job.log if job else print
        rows = self.lab_sample()
        log(f"lab: {dino_label(cand)} vs {dino_label(active)} on {len(rows)} scene(s)")
        speed = self._lab_embed(cand, rows, job)
        if job is not None and job.cancelled:
            log("stopped — the sample is kept, the next run carries on")
            return {"stopped": True}
        if job is not None:
            job.progress = {**(job.progress or {}), "stage": "judging", "cand": cand}
        cur_v = self._lab_vectors(self._model_name(), rows)
        new_v = self._lab_vectors(dino_cache_name(cand), rows)
        both = [r for r in rows if r["scene_id"] in cur_v and r["scene_id"] in new_v]
        saved = self._zoom_saved()
        common = set(cur_v) & set(new_v)
        side = {}
        for name, v in (("current", cur_v), ("candidate", new_v)):
            side[name] = {"grading": self._lab_grading(v, both), "moments": self._lab_moments(v, saved, common)}
        paired = self._lab_paired(side["current"]["grading"], side["candidate"]["grading"])
        paired_m = self._lab_paired(side["current"]["moments"], side["candidate"]["moments"])
        verdict = self._lab_verdict(dino_label(cand), side["current"], side["candidate"], paired, paired_m)
        sample_frames = sum(len(v[0]) for v in new_v.values())
        del cur_v, new_v                                # the sample's frames go back before anything else runs
        # the switch estimate: the candidate's measured speed over the whole library
        idx = self.index(self._model_name())
        fps = (speed["frames"] / speed["seconds"]) if speed["frames"] and speed["seconds"] else None
        est = (max(0, idx.size - sample_frames) / fps / 3600) if fps else None
        trial = {"candidate": cand, "against": active, "against_label": dino_label(active), "at": time.time(), "scenes": len(both),
                 "current": side["current"], "new": side["candidate"], "paired": paired, "paired_moments": paired_m,
                 **verdict,
                 "speed_fps": None if fps is None else round(fps, 1),
                 "switch_hours": None if est is None else round(est, 1),
                 "library_frames": int(idx.size), "sample_frames": int(sample_frames)}
        d = self._lab_state()
        d["trials"][cand] = trial
        self._lab_save(d)
        log("lab: " + verdict["verdict"])
        return trial

    def lab_switch(self, cand: str) -> dict:
        """Make a winning candidate the active model. The sample is already in its
        cache, so the re-embed starts that far in; the old cache stays."""
        t = self._lab_state()["trials"].get(cand)
        if not t or not t.get("win"):
            raise ValueError("only a candidate that won its trial can be switched to")
        if t.get("against") != self._active_dino_model():
            raise ValueError("the active model changed since that trial — run it again")
        return self.save_models(dino_model=cand)
