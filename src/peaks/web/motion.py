"""Motion: Peaks watches how your candidate moments move.

Single frames — even every 2 s — can't see movement itself: the pace, the
rhythm of it, a change of tempo. A video model can. For each zoomed scene, the
"motion" job decodes each candidate window (your saves, Peaks' top picks) at
4 frames a second and embeds overlapping 4-second clips with V-JEPA 2.1, a
self-supervised video encoder. The vectors live in the embedding cache under
their own name, keyed by file, so the visual index and its model are untouched.

The moment model gets motion features per window (moving like your saved
moments in *other* scenes, how much the movement changes), and the tier model a
motion group — each kept only where the honest held-out test says it helps.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from .hear import SenseRun

MOTION_NS = "vjepa21-l@motion"
MOTION_PER_RUN = 100
CLIP_FRAMES = 16          # frames per clip
CLIP_FPS = 4.0            # → 4-second clips
CLIP_EVERY = 4.0          # a clip centred every … seconds across a window
WINDOW_HALF = 20.0        # the zoom window: ±20 s around a candidate
MOTION_FEATURES: tuple[str, ...] = ("motion_taste", "motion_change_mean", "motion_change_max", "motion_like")
MOTION_WORDS = {"motion_taste": "moves like your moments", "motion_change_mean": "the movement keeps changing",
                "motion_change_max": "a change of pace", "motion_like": "moves like what you picked in the Lab"}
EXPLORE_WINDOWS = 2       # extra windows per scene, away from the picks, so the Lab's explorer sees more


def cut_clips(frames: list, start: float, fps: float = CLIP_FPS, n: int = CLIP_FRAMES,
              every: float = CLIP_EVERY) -> list[tuple[float, list]]:
    """[(centre, n frames)] from a window decoded at `fps` starting at `start`:
    clips of n frames centred every `every` seconds, whole clips only."""
    out = []
    half = n / fps / 2
    if len(frames) < n:
        return out
    end = start + len(frames) / fps
    c = start + half
    while c + half <= end + 1e-6:
        a = int(round((c - half - start) * fps))
        clip = frames[a: a + n]
        if len(clip) == n:
            out.append((round(c, 2), clip))
        c += every
    return out


class MotionMixin:
    # --- settings / state ----------------------------------------------------------------

    def motion_settings(self) -> dict:
        m = self._settings().get("motion") or {}
        return {"on": bool(m.get("on", True)), "per_run": int(m.get("per_run") or MOTION_PER_RUN),
                "explore_windows": int(m.get("explore_windows", EXPLORE_WINDOWS))}

    def save_motion_settings(self, on: bool | None = None, per_run: int | None = None) -> dict:
        s = dict(self._settings())
        m = dict(s.get("motion") or {})
        if on is not None:
            m["on"] = bool(on)
        if per_run is not None:
            m["per_run"] = max(1, min(2000, int(per_run)))
        s["motion"] = m
        self._save_settings_dict(s)
        return self.motion_settings()

    def _motion_cache(self):
        from ..cache import EmbeddingCache

        return EmbeddingCache(self.cfg.embedding.cache_dir)

    def motion_todo(self) -> list[dict]:
        """Zoomed scenes whose candidate windows aren't all watched yet — saves first."""
        cache = self._motion_cache()
        zoomed = self._zoom_state()["scenes"]
        saved = self._zoom_saved()
        rows = {r["scene_id"]: r for r in self._catalogue_all() if r["tier"] != "rejected" and r.get("path")}
        out = []
        for sid, e in zoomed.items():
            if sid not in rows:
                continue
            done = None
            if cache.has(e["key"], MOTION_NS):
                try:
                    done = cache.peek_meta(e["key"], MOTION_NS).get("centers")
                except Exception:  # noqa: BLE001
                    done = None
            want = self._motion_centers(sid, e, rows[sid])
            if done is None or sorted(round(c, 1) for c in done) != sorted(round(c, 1) for c in want):
                out.append(rows[sid])
        return sorted(out, key=lambda r: r["scene_id"] not in saved)

    def _motion_centers(self, sid: str, entry: dict, row: dict) -> list[float]:
        """The zoom's candidate windows plus a few explore windows elsewhere in the
        scene (the same ones every time, from the scene id) — so the Lab's motion
        explorer isn't limited to Peaks' own picks."""
        cs = [float(c) for c in entry.get("centers") or []]
        k = int(self.motion_settings().get("explore_windows", EXPLORE_WINDOWS))
        dur = float(row.get("duration") or 0)
        if k <= 0 or dur < 4 * WINDOW_HALF:
            return cs
        rng = np.random.default_rng(int(hashlib.sha1(str(sid).encode()).hexdigest()[:8], 16))
        extra: list[float] = []
        for _ in range(40):
            if len(extra) >= k:
                break
            c = round(float(rng.uniform(WINDOW_HALF, dur - WINDOW_HALF)), 1)
            if all(abs(c - x) >= 2 * WINDOW_HALF for x in cs + extra):
                extra.append(c)
        return cs + extra

    def motion_status(self) -> dict:
        try:
            zoomed = self._zoom_state()["scenes"]
            cache = self._motion_cache()
            seen = sum(1 for e in zoomed.values() if cache.has(e["key"], MOTION_NS))
            total = len(zoomed)
        except Exception:  # noqa: BLE001
            seen = total = None
        return {"seen": seen, "embedded": total, "settings": self.motion_settings(),
                "error": getattr(self, "_motion_error", None), "last_run": self.sense_last_run("motion")}

    def motion_due(self) -> bool:
        return self.motion_settings()["on"] and bool(self.motion_todo())

    # --- the job --------------------------------------------------------------------------

    def _motion_embedder(self):
        from ..embedding import MotionEmbedder

        return MotionEmbedder(device=self.cfg.embedding.device or None, log=getattr(self, "_motion_log", None))

    def _motion_sampler(self):
        from ..sampling import FrameSampler

        s = self.cfg.sampling
        return FrameSampler(interval_seconds=1.0, mode="interval", hwaccel=s.hwaccel, pipeline=s.pipeline,
                            scene_timeout=s.scene_timeout)

    def motion_scene(self, row: dict, embedder, sampler) -> dict:
        sid = row["scene_id"]
        e = self._zoom_state()["scenes"].get(sid)
        if not e:
            raise LookupError(f"scene {sid} isn't zoomed yet")
        times, vecs = [], []
        centers = self._motion_centers(sid, e, row)
        for c in centers:
            start = max(0.0, float(c) - WINDOW_HALF)
            frames = sampler.window_frames(row["path"], start, 2 * WINDOW_HALF, 1.0 / CLIP_FPS, size=embedder.size)
            clips = cut_clips([im for _, im in frames], start)
            clips = [(t, cl) for t, cl in clips if all(abs(t - x) > 1.0 for x in times)]
            if not clips:
                continue
            v = embedder.embed_clips([cl for _, cl in clips])
            times += [t for t, _ in clips]
            vecs.append(v)
        order = np.argsort(times)
        V = np.concatenate(vecs)[order] if vecs else np.zeros((0, embedder.dim), np.float32)
        self._motion_cache().save(e["key"], MOTION_NS, np.asarray(times, np.float32)[order], V,
                                  meta={"scene_id": sid, "centers": centers})
        self.__dict__.pop("_motion_bank_cache", None)
        self.__dict__.get("_motion_vec_memo", {}).pop(str(sid), None)
        return {"clips": len(times)}

    def run_motion(self, job=None, limit: int | None = None) -> dict:
        log = job.log if job else print
        todo = self.motion_todo()[: limit or self.motion_settings()["per_run"]]
        if not todo:
            return {"seen": 0}
        try:
            self._motion_log = log                        # a first run fetches the weights: say so
            embedder = self._motion_embedder()
        except Exception as exc:  # noqa: BLE001 — model missing: say so on the Lab card
            self._motion_error = str(exc)
            raise
        self._motion_error = None
        sampler = self._motion_sampler()
        done = 0
        run = SenseRun()
        t0 = time.time()
        for i, row in enumerate(todo):
            if job is not None and job.cancelled:
                log("stopped — the rest waits for the next run")
                run.stopped = "you stopped it"
                break
            try:
                self.motion_scene(row, embedder, sampler)
                done += 1
                run.ok()
            except Exception as exc:  # noqa: BLE001 — one unreadable file mustn't stop the run
                log(f"scene {row['scene_id']}: {exc}")
                if run.fail(row["scene_id"], exc):           # the same failure every time: stop, say so
                    self._motion_error = run.stopped
                    log(run.stopped)
                    break
            if job is not None:
                rate = (time.time() - t0) / (i + 1)
                job.progress = {"total": len(todo), "done": i + 1, "eta_s": round(rate * (len(todo) - i - 1))}
        log(f"watched {done} scene(s){f', {run.failed} unreadable' if run.failed else ''}")
        self._sense_record("motion", {"done": done, **run.summary()})
        del embedder
        if done:
            self._senses_retrain(log)
        return {"seen": done, "failed": run.failed}

    # --- what the moment model and the tier model get --------------------------------------

    def _motion_load(self, sid: str):
        e = self._zoom_state()["scenes"].get(str(sid))
        cache = self._motion_cache()
        if not e or not cache.has(e["key"], MOTION_NS):
            return None
        try:
            t, v, _ = cache.load(e["key"], MOTION_NS)
        except (OSError, KeyError, ValueError):
            return None
        if not len(t):
            return None
        return t, v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def _motion_bank(self):
        hit = self.__dict__.get("_motion_bank_cache")
        if hit is not None and time.time() - hit[0] < 300:
            return hit[1]
        vecs, owner = [], []
        for sid, ts in self._zoom_saved().items():
            m = self._motion_load(sid)
            if m is None:
                continue
            for t in ts:
                j = int(np.argmin(np.abs(m[0] - t)))
                if abs(float(m[0][j]) - t) <= CLIP_EVERY:
                    vecs.append(m[1][j])
                    owner.append(str(sid))
        bank = (np.stack(vecs), np.asarray(owner)) if vecs else None
        self._motion_bank_cache = (time.time(), bank)
        return bank

    def motion_window_features(self, sid: str, centre: float, peak: float, half: float) -> dict | None:
        m = self._motion_load(sid)
        if m is None:
            return None
        t, v = m
        w = np.abs(t - centre) <= half
        if w.sum() < 1:
            return None
        vw = v[w]
        change = 1.0 - (vw[1:] * vw[:-1]).sum(axis=1) if len(vw) > 1 else np.zeros(1)
        taste = 0.0
        bank = self._motion_bank()
        if bank is not None:
            others = bank[1] != str(sid)                       # never this scene's own saves
            if others.any():
                taste = float((vw @ bank[0][others].T).max())
        return {"motion_taste": taste, "motion_change_mean": float(change.mean()),
                "motion_change_max": float(change.max()), **self.sense_window_likes("motion", sid, centre, half)}

    def motion_scene_vector(self, sid: str) -> np.ndarray | None:
        memo = self.__dict__.setdefault("_motion_vec_memo", {})
        if str(sid) in memo:
            return memo[str(sid)]
        m = self._motion_load(sid)
        out = None
        if m is not None:
            mean = m[1].mean(axis=0)
            out = (mean / max(float(np.linalg.norm(mean)), 1e-9)).astype(np.float32)
        memo[str(sid)] = out
        return out
