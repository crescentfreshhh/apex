"""Hearing: Peaks listens to each scene, and learns whether sound helps.

A background job ("hear") decodes each scene's audio — scenes with saved
moments first, then graded ones, then the rest; rejects never — and keeps two
things in the embedding cache:

  - "clap-audio": a CLAP vector per 10 s block (every 5 s) — what it sounds
    like, in a space words live in too, so you can search sound with text;
  - "audio-env": every 2 s, how loud, how busy, how bright, how rhythmic.

The moment model gets sound features for each candidate window (getting louder,
a steady rhythm, sounding like your saved moments in *other* scenes) and the
tier model gets a sound group — each kept only where the honest held-out test
says it helps.
"""

from __future__ import annotations

import time

import numpy as np

from ..audio import ENV_FEATURES, decode_audio, envelope

CLAP_NS, ENV_NS = "clap-audio", "audio-env"
HEAR_PER_RUN = 300
SOUND_FEATURES: tuple[str, ...] = ("loud_peak", "loud_rise", "onset_mean", "rhythm", "sound_taste", "sound_like")
SOUND_WORDS = {"loud_peak": "loud", "loud_rise": "it gets louder", "onset_mean": "a lot happening",
               "rhythm": "a steady rhythm", "sound_taste": "sounds like your moments",
               "sound_like": "sounds like what you picked in the Lab"}
BATCH = 16


FAIL_FAST = 5            # this many failures in a row, all the same, and nothing done: stop the run


class SenseRun:
    """Counts a background run's failures and stops it early when every scene so
    far has failed the same way (a broken model, not a bad file)."""

    def __init__(self):
        self.failed, self.streak, self.errors, self.stopped = 0, 0, [], None
        self._msgs: set[str] = set()

    def ok(self) -> None:
        self.streak = 0

    def fail(self, sid, exc) -> bool:
        msg = f"{type(exc).__name__}: {exc}"[:300]
        self.failed += 1
        self.streak += 1
        self._msgs.add(msg)
        if len(self.errors) < 3:
            self.errors.append(f"scene {sid}: {msg}")
        if self.streak >= FAIL_FAST and len(self._msgs) == 1 and self.failed == self.streak:
            self.stopped = f"every scene failed the same way — {msg}"
            return True
        return False

    def summary(self) -> dict:
        return {"failed": self.failed, "first_errors": self.errors, "stopped": self.stopped}


class HearMixin:
    # --- settings / state ----------------------------------------------------------------

    def hear_settings(self) -> dict:
        h = self._settings().get("hearing") or {}
        return {"on": bool(h.get("on", True)), "per_run": int(h.get("per_run") or HEAR_PER_RUN)}

    def save_hear_settings(self, on: bool | None = None, per_run: int | None = None) -> dict:
        s = dict(self._settings())
        h = dict(s.get("hearing") or {})
        if on is not None:
            h["on"] = bool(on)
        if per_run is not None:
            h["per_run"] = max(1, min(5000, int(per_run)))
        s["hearing"] = h
        self._save_settings_dict(s)
        return self.hear_settings()

    def _hear_cache(self):
        from ..cache import EmbeddingCache

        return EmbeddingCache(self.cfg.embedding.cache_dir)

    def _hear_keys(self) -> dict[str, str]:
        """scene_id → cache key, for scenes in the visual index (the key is the file's)."""
        idx = self.index(self._model_name())
        hit = self.__dict__.get("_hear_keys_cache")
        if hit is None or hit[0] is not idx:
            hit = (idx, {str(m.get("scene_id")): k for k, m in idx.key_meta.items()})
            self._hear_keys_cache = hit
        return hit[1]

    def heard(self, sid: str) -> bool:
        k = self._hear_keys().get(str(sid))
        return bool(k) and self._hear_cache().has(k, ENV_NS)

    def hear_todo(self) -> list[dict]:
        from ..tier_model import train_class

        keys = self._hear_keys()
        cache = self._hear_cache()
        have = set(cache.keys(ENV_NS))
        saved = self._zoom_saved()
        rows = [r for r in self._catalogue_all() if r["tier"] != "rejected"
                and keys.get(r["scene_id"]) and keys[r["scene_id"]] not in have and r.get("path")]
        return sorted(rows, key=lambda r: (r["scene_id"] not in saved, not train_class(r["tier"]),
                                           r.get("created_at") or ""), reverse=False)

    def hear_status(self) -> dict:
        try:
            keys = self._hear_keys()
            have = set(self._hear_cache().keys(ENV_NS))
            heard = sum(1 for k in keys.values() if k in have)
            total = len(keys)
        except Exception:  # noqa: BLE001
            heard = total = None
        return {"heard": heard, "embedded": total, "settings": self.hear_settings(),
                "error": getattr(self, "_hear_error", None), "last_run": self.sense_last_run("hear")}

    def hear_due(self) -> bool:
        return self.hear_settings()["on"] and bool(self.hear_todo())

    # --- the job --------------------------------------------------------------------------

    def _hear_embedder(self):
        from ..embedding import ClapAudioEmbedder

        return ClapAudioEmbedder(device=self.cfg.embedding.device or None)

    def _hear_decode(self, path: str, sr: int):
        return decode_audio(path, sr)

    def hear_scene(self, row: dict, embedder) -> dict:
        sid = row["scene_id"]
        key = self._hear_keys().get(sid)
        if not key:
            raise LookupError(f"scene {sid} isn't embedded")
        cache = self._hear_cache()
        blocks = self._hear_decode(row["path"], embedder.sr)
        if blocks is None:                                    # no audio stream: remember that, move on
            cache.save(key, ENV_NS, np.zeros(0, np.float32), np.zeros((0, len(ENV_FEATURES)), np.float32),
                       meta={"scene_id": sid, "no_audio": True})
            cache.save(key, CLAP_NS, np.zeros(0, np.float32), np.zeros((0, embedder.dim), np.float32),
                       meta={"scene_id": sid, "no_audio": True})
            self.__dict__.get("_sound_vec_memo", {}).pop(str(sid), None)
            return {"no_audio": True}
        et, ef, ct, cv, pend, pend_t = [], [], [], [], [], []

        def flush():
            if pend:
                cv.append(embedder.embed_audio(pend))
                ct.extend(pend_t)
                pend.clear()
                pend_t.clear()
        hop = 5.0
        for start, x in blocks:
            t, f = envelope(x, embedder.sr, step=2.0, t0=start)
            keep = t < start + hop if len(x) >= 10 * embedder.sr else np.ones(len(t), bool)
            et.append(t[keep])
            ef.append(f[keep])
            pend.append(x)
            pend_t.append(start + min(len(x) / embedder.sr, 10.0) / 2)
            if len(pend) >= BATCH:
                flush()
        flush()
        times = np.concatenate(et) if et else np.zeros(0, np.float32)
        feats = np.concatenate(ef) if ef else np.zeros((0, len(ENV_FEATURES)), np.float32)
        cache.save(key, ENV_NS, times, feats, meta={"scene_id": sid})
        vecs = np.concatenate(cv) if cv else np.zeros((0, embedder.dim), np.float32)
        cache.save(key, CLAP_NS, np.asarray(ct, np.float32), vecs, meta={"scene_id": sid})
        self.__dict__.pop("_sound_bank_cache", None)
        self.__dict__.get("_sound_vec_memo", {}).pop(str(sid), None)
        return {"blocks": len(ct)}

    def run_hear(self, job=None, limit: int | None = None) -> dict:
        log = job.log if job else print
        todo = self.hear_todo()[: limit or self.hear_settings()["per_run"]]
        if not todo:
            return {"heard": 0}
        try:
            embedder = self._hear_embedder()
        except Exception as exc:  # noqa: BLE001 — model missing: say so on the Lab card
            self._hear_error = str(exc)
            raise
        self._hear_error = None
        done = silent = 0
        run = SenseRun()
        t0 = time.time()
        for i, row in enumerate(todo):
            if job is not None and job.cancelled:
                log("stopped — the rest waits for the next run")
                run.stopped = "you stopped it"
                break
            try:
                r = self.hear_scene(row, embedder)
                done += 1
                silent += bool(r.get("no_audio"))
                run.ok()
            except Exception as exc:  # noqa: BLE001 — one unreadable file mustn't stop the run
                log(f"scene {row['scene_id']}: {exc}")
                if run.fail(row["scene_id"], exc):           # the same failure every time: stop, say so
                    self._hear_error = run.stopped
                    log(run.stopped)
                    break
            if job is not None:
                rate = (time.time() - t0) / (i + 1)
                job.progress = {"total": len(todo), "done": i + 1, "eta_s": round(rate * (len(todo) - i - 1))}
        log(f"heard {done} scene(s){f', {silent} without sound' if silent else ''}{f', {run.failed} unreadable' if run.failed else ''}")
        self._sense_record("hear", {"done": done, "silent": silent, **run.summary()})
        del embedder
        if done > silent:
            self._senses_retrain(log)
        return {"heard": done, "silent": silent, "failed": run.failed}

    # --- what the last run did, for the Lab cards -----------------------------------------

    def _senses_runs_path(self):
        return self._state_dir() / "senses_runs.json"

    def _sense_record(self, kind: str, rec: dict) -> None:
        import json

        from .. import atomic

        p = self._senses_runs_path()
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            d = {}
        d[kind] = {"at": time.time(), **rec}
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic.write_json(p, d, indent=1)

    def sense_last_run(self, kind: str) -> dict | None:
        import json

        try:
            return json.loads(self._senses_runs_path().read_text()).get(kind)
        except (OSError, ValueError):
            return None

    def _senses_retrain(self, log) -> None:
        """New sense data: the moment model and the tier model re-decide what helps."""
        for name, fn in (("moment model", self.train_moment_model), ("tier model", self.train_tier_model)):
            try:
                rep = fn()
                log(f"{name}: " + str(rep.get("verdict") or rep.get("reason") or ("trained" if rep.get("trained") else "")))
            except Exception as exc:  # noqa: BLE001
                log(f"{name}: {exc}")

    # --- what the moment model and the tier model get --------------------------------------

    def _sound_load(self, sid: str):
        """(env_times, env, clap_times, clap) for a heard scene, else None."""
        k = self._hear_keys().get(str(sid))
        cache = self._hear_cache()
        if not k or not cache.has(k, ENV_NS):
            return None
        try:
            et, ef, meta = cache.load(k, ENV_NS)
            if meta.get("no_audio") or not len(et):
                return None
            ct, cv, _ = cache.load(k, CLAP_NS) if cache.has(k, CLAP_NS) else (np.zeros(0), np.zeros((0, 1)), {})
        except (OSError, KeyError, ValueError):
            return None
        if len(cv):
            cv = cv / np.maximum(np.linalg.norm(cv, axis=1, keepdims=True), 1e-9)
        return et, ef, ct, cv

    def _sound_bank(self):
        """CLAP vectors at your saved moments: (vectors, owning scene ids)."""
        hit = self.__dict__.get("_sound_bank_cache")
        if hit is not None and time.time() - hit[0] < 300:
            return hit[1]
        vecs, owner = [], []
        for sid, ts in self._zoom_saved().items():
            s = self._sound_load(sid)
            if s is None or not len(s[2]):
                continue
            for t in ts:
                j = int(np.argmin(np.abs(s[2] - t)))
                if abs(float(s[2][j]) - t) <= 5.0:
                    vecs.append(s[3][j])
                    owner.append(sid)
        bank = (np.stack(vecs), np.asarray(owner)) if vecs else None
        self._sound_bank_cache = (time.time(), bank)
        return bank

    def sound_window_features(self, sid: str, centre: float, peak: float, half: float) -> dict | None:
        """The sound features of one candidate window (None when the scene isn't heard)."""
        s = self._sound_load(sid)
        if s is None:
            return None
        et, ef, ct, cv = s
        li = ENV_FEATURES.index("loudness")
        m = np.abs(et - centre) <= half
        if m.sum() < 2:
            return None
        loud = ef[:, li]
        pre = (et >= peak - 10.0) & (et <= peak)
        rise = 0.0
        if pre.sum() >= 2:
            tc = et[pre] - et[pre].mean()
            rise = float((tc * (loud[pre] - loud[pre].mean())).sum() / max((tc * tc).sum(), 1e-9))
        taste = 0.0
        bank = self._sound_bank()
        cm = np.abs(ct - centre) <= half + 5.0
        if bank is not None and cm.any():
            others = bank[1] != str(sid)                       # never this scene's own saves
            if others.any():
                taste = float((cv[cm] @ bank[0][others].T).max())
        return {"loud_peak": float(loud[m].max() - np.median(loud)), "loud_rise": rise,
                "onset_mean": float(ef[m, ENV_FEATURES.index("onset")].mean()),
                "rhythm": float(ef[m, ENV_FEATURES.index("rhythm")].mean()), "sound_taste": taste,
                **self.sense_window_likes("sound", sid, centre, half)}

    def sound_scene_vector(self, sid: str) -> np.ndarray | None:
        """[unit mean of all CLAP blocks, unit mean of the loudest 10 %] — the tier model's sound group."""
        from ..tier_model import visual_features

        memo = self.__dict__.setdefault("_sound_vec_memo", {})
        if str(sid) in memo:
            return memo[str(sid)]
        s = self._sound_load(sid)
        out = None
        if s is not None and len(s[3]):
            et, ef, ct, cv = s
            loud = np.interp(ct, et, ef[:, ENV_FEATURES.index("loudness")]) if len(et) else None
            out = visual_features(cv, loud)
        memo[str(sid)] = out
        return out

    # --- search by sound -----------------------------------------------------------------------

    def sound_search(self, q: str, top_k: int = 60) -> list:
        """Moments whose sound matches words ("moaning", "music", "laughing"):
        CLAP text vs every heard block, best block per scene."""
        from ..search import Hit

        emb = self.__dict__.get("_clap_text")
        if emb is None:
            emb = self._hear_embedder()
            self._clap_text = emb
        w = emb.embed_text([q])[0]
        w = w / max(float(np.linalg.norm(w)), 1e-9)
        cache = self._hear_cache()
        keys = self._hear_keys()
        out = []
        for sid, k in keys.items():
            if not cache.has(k, CLAP_NS):
                continue
            ct, cv, meta = cache.load(k, CLAP_NS)
            if meta.get("no_audio") or not len(ct):
                continue
            s = (cv / np.maximum(np.linalg.norm(cv, axis=1, keepdims=True), 1e-9)) @ w
            j = int(np.argmax(s))
            out.append(Hit(scene_id=sid, key=k, time=float(max(0.0, ct[j] - 5.0)), score=float(s[j])))
        out.sort(key=lambda h: -h.score)
        return out[:top_k]
