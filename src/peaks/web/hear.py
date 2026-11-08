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

from ..atomic import settings_saver as _settings_saver


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
FAIL_WAIT_S = 7 * 86400  # a file that failed waits a week before it's tried again…
FAIL_PARK = 3            # …and after this many failures, until the file itself changes


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
    # --- files a sense couldn't read: remembered, so a run never starts on them again --------

    def _sense_fail_path(self):
        from pathlib import Path

        return Path(self.cfg.modeling.dir) / "sense_failures.json"

    def _sense_fail_all(self) -> dict:
        from .. import atomic

        return atomic.read_json(self._sense_fail_path(), default={}) or {}

    def sense_failures(self, sense: str) -> dict:
        """{scene_id: {n, at, msg, fp}} for one sense ("motion", "hear")."""
        return dict(self._sense_fail_all().get(sense) or {})

    def _sense_fail_edit(self, sense: str, fn) -> None:
        from .. import atomic

        p = self._sense_fail_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with atomic.edit_lock(p):
            d = self._sense_fail_all()
            cur = dict(d.get(sense) or {})
            fn(cur)
            if cur:
                d[sense] = cur
            else:
                d.pop(sense, None)
            atomic.write_json(p, d, indent=1)
        self.__dict__.pop("_sense_fail_memo", None)

    def _sense_failed(self, sense: str, row: dict, exc) -> None:
        sid, fp = str(row["scene_id"]), row.get("fingerprint")

        def add(cur):
            e = cur.get(sid) or {}
            n = int(e.get("n") or 0) + 1 if e.get("fp") == fp else 1     # a changed file starts over
            cur[sid] = {"n": n, "at": time.time(), "msg": f"{type(exc).__name__}: {exc}"[:240], "fp": fp}
        self._sense_fail_edit(sense, add)

    def _sense_unfail(self, sense: str, sids) -> None:
        sids = {str(s) for s in sids}
        if not sids or not (sids & set(self.sense_failures(sense))):
            return
        self._sense_fail_edit(sense, lambda cur: [cur.pop(s, None) for s in sids])

    def sense_retry_failed(self, sense: str) -> dict:
        n = len(self.sense_failures(sense))
        self._sense_fail_edit(sense, lambda cur: cur.clear())
        return {"cleared": n}

    def _sense_skip(self, fails: dict, row: dict, now: float | None = None) -> bool:
        """Leave this file out of the run? (it failed lately, or keeps failing)"""
        e = fails.get(str(row["scene_id"]))
        if not e or e.get("fp") != row.get("fingerprint"):
            return False                                   # never failed, or the file changed since
        if int(e.get("n") or 0) >= FAIL_PARK:
            return True
        return (now or time.time()) - float(e.get("at") or 0) < FAIL_WAIT_S

    def _sense_failure_count(self, sense: str) -> int:
        fails = self.sense_failures(sense)
        if not fails:
            return 0
        try:
            rows = {r["scene_id"]: r for r in self._catalogue_all()}
        except Exception:  # noqa: BLE001
            return len(fails)
        return sum(1 for sid in fails if sid in rows and self._sense_skip(fails, rows[sid]))

    # --- settings / state ----------------------------------------------------------------

    def hear_settings(self) -> dict:
        h = self._settings().get("hearing") or {}
        return {"on": bool(h.get("on", True)), "per_run": int(h.get("per_run") or HEAR_PER_RUN)}

    @_settings_saver
    def save_hear_settings(self, on: bool | None = None, per_run: int | None = None) -> dict:
        s = dict(self._settings())
        h = dict(s.get("hearing") or {})
        from .inputs import as_bool, as_int

        if on is not None:
            h["on"] = as_bool(on, "on")
        if per_run is not None:
            h["per_run"] = max(1, min(5000, as_int(per_run, "per_run")))
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
        if hit is None or hit[0] != idx.gen:          # by build number: never pins an old index in memory
            hit = (idx.gen, idx.sid_key)
            self._hear_keys_cache = hit
        return hit[1]

    def heard(self, sid: str) -> bool:
        k = self._hear_keys().get(str(sid))
        return bool(k) and self._hear_cache().has(k, ENV_NS)

    def hear_todo(self, first: bool = False) -> list[dict]:
        """Scenes not heard yet — saved moments first, then graded, then the rest;
        a file that failed lately is left out, one retried goes last. `first`:
        only whether there's any work (stops at the first)."""
        from ..tier_model import train_class

        keys = self._hear_keys()
        cache = self._hear_cache()
        have = set(cache.keys(ENV_NS))
        fails, now = self.sense_failures("hear"), time.time()
        rows = []
        for r in self._catalogue_all():
            if r["tier"] == "rejected" or not keys.get(r["scene_id"]) or keys[r["scene_id"]] in have or not r.get("path"):
                continue
            if self._sense_skip(fails, r, now):
                continue
            rows.append(r)
            if first:
                return rows
        saved = self._zoom_saved()
        return sorted(rows, key=lambda r: (r["scene_id"] in fails, r["scene_id"] not in saved,
                                           not train_class(r["tier"]), r.get("created_at") or ""))

    def hear_status(self) -> dict:
        try:
            keys = self._hear_keys()
            have = set(self._hear_cache().keys(ENV_NS))
            heard = sum(1 for k in keys.values() if k in have)
            total = len(keys)
        except Exception:  # noqa: BLE001
            heard = total = None
        return {"heard": heard, "embedded": total, "settings": self.hear_settings(),
                "unreadable": self._sense_failure_count("hear"),
                "error": getattr(self, "_hear_error", None), "last_run": self.sense_last_run("hear")}

    def hear_due(self) -> bool:
        return self.hear_settings()["on"] and bool(self.hear_todo(first=True))

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
            if self.__dict__.pop("_retrain_owed", False):
                self._senses_retrain(log)
            return {"heard": 0}
        try:
            embedder = self._hear_embedder()
        except Exception as exc:  # noqa: BLE001 — model missing: say so on the Lab card
            self._hear_error = str(exc)
            raise
        self._hear_error = None
        done = silent = 0
        ok_ids, failed_ids = [], []
        run = SenseRun()
        t0 = time.time()
        self._sense_record("hear", {"running": True, "done": 0, "silent": silent, **run.summary()})   # a new run replaces the last
        for i, row in enumerate(todo):
            if job is not None and job.cancelled:
                log("stopped — the rest waits for the next run")
                run.stopped = (f"paused for {job.paused_for}" if getattr(job, "paused_for", None) else "you stopped it")
                break
            try:
                r = self.hear_scene(row, embedder)
                done += 1
                silent += bool(r.get("no_audio"))
                run.ok()
                ok_ids.append(row["scene_id"])
            except Exception as exc:  # noqa: BLE001 — one unreadable file mustn't stop the run
                log(f"scene {row['scene_id']}: {exc}")
                self._sense_failed("hear", row, exc)
                failed_ids.append(row["scene_id"])
                if run.fail(row["scene_id"], exc):           # the same failure every time: stop, say so
                    self._hear_error = run.stopped
                    log(run.stopped)
                    self._sense_unfail("hear", failed_ids)   # the model is broken, not these files
                    break
            if (i + 1) % 25 == 0:
                self._sense_record("hear", {"running": True, "done": done, "silent": silent, **run.summary()})
            if job is not None:
                rate = (time.time() - t0) / (i + 1)
                job.progress = {"total": len(todo), "done": i + 1, "eta_s": round(rate * (len(todo) - i - 1))}
        self._sense_unfail("hear", ok_ids)
        log(f"heard {done} scene(s){f', {silent} without sound' if silent else ''}{f', {run.failed} unreadable' if run.failed else ''}")
        self._sense_record("hear", {"done": done, "silent": silent, **run.summary()})
        del embedder
        if getattr(job, "paused_for", None):
            log("retrain waits until it resumes")       # never rebuild the index during an ingest
        elif done > silent or self.__dict__.pop("_retrain_owed", False):
            self._senses_retrain(log)
        return {"heard": done, "silent": silent, "failed": run.failed}

    # --- what the last run did, for the Lab cards -----------------------------------------

    def _senses_runs_path(self):
        return self._state_dir() / "senses_runs.json"

    def _sense_record(self, kind: str, rec: dict) -> None:
        import json

        from .. import atomic

        p = self._senses_runs_path()
        with atomic.edit_lock(p):                     # Hearing and Motion both write here
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                d = {}
            d[kind] = {"at": time.time(), **rec}
            p.parent.mkdir(parents=True, exist_ok=True)
            atomic.write_json(p, d, indent=1)

    def sense_clear(self, kind: str) -> None:
        """Forget the last run's record and error: the card starts clean."""
        import json

        from .. import atomic

        setattr(self, f"_{kind}_error", None)
        p = self._senses_runs_path()
        with atomic.edit_lock(p):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                return
            if d.pop(kind, None) is not None:
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
        if out is not None:                  # a miss is never remembered: the scene may be heard any moment
            memo[str(sid)] = out
        return out

    # --- search by sound -----------------------------------------------------------------------

    def sound_search(self, q: str, top_k: int = 60) -> list:
        """Moments whose sound matches words ("moaning", "music", "laughing"):
        CLAP text vs every heard block, best block per scene."""
        from ..search import Hit

        import threading

        with self.__dict__.setdefault("_clap_lock", threading.Lock()):   # one CLAP, however many searches
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
