"""Picture quality in the app: measuring files (new ones after each ingest, the
whole library only when you start it), the "which looks better?" picks, the
learned score, and where it shows up (Catalogue, duplicates, Review).

On disk, under <models>/picture/:
  measures.jsonl   one record per measured file (later lines win), keyed by the
                   file's fingerprint; carries size+mtime so a replaced file is
                   measured again; failures are recorded too (not retried until
                   the file changes)
  pairs.jsonl      your picks: {a, b, ans: a|b|fine|bad, src, t}
  ratings.jsonl    your single-file ratings: {key, band 1–5 (Poor…Reference), t}; the last one wins
  model.json       the trained model
  state.json       whether the library backfill is on
  stills/          two small JPEGs per file for the comparison screen
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path

from .. import picture as pq

SKIP_TIERS = ("rejected",)


class PictureMixin:
    # --- files ------------------------------------------------------------------------

    def _pic_dir(self) -> Path:
        d = Path(self.cfg.modeling.dir) / "picture"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _pic_lock(self) -> threading.RLock:
        return self.__dict__.setdefault("_pic_lock_obj", threading.RLock())

    @staticmethod
    def _pic_key(row: dict) -> str:
        return str(row.get("fingerprint") or ("p:" + (row.get("path") or row["scene_id"])))

    @staticmethod
    def _pic_safe(key: str) -> str:
        return hashlib.sha1(key.encode()).hexdigest()[:20]

    @staticmethod
    def _pic_stat(path: str) -> list[int] | None:
        try:
            st = os.stat(path)
            return [int(st.st_size), int(st.st_mtime)]
        except OSError:
            return None

    def _pic_measures(self) -> dict[str, dict]:
        """key → its latest record (reloaded when the file changes on disk)."""
        p = self._pic_dir() / "measures.jsonl"
        try:
            mt = p.stat().st_mtime_ns
        except OSError:
            return {}
        c = self.__dict__.get("_pic_meas_cache")
        if c and c[0] == mt:
            return c[1]
        out: dict[str, dict] = {}
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    out[r["key"]] = r
                except (ValueError, KeyError):
                    continue
        self._pic_meas_cache = (mt, out)
        return out

    def _pic_append(self, name: str, rec: dict) -> None:
        with self._pic_lock(), open(self._pic_dir() / name, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _pic_rec(self, row: dict, check_file: bool = False) -> dict | None:
        """This file's measures, if current (and successful)."""
        r = self._pic_measures().get(self._pic_key(row))
        if not r or r.get("error") or r.get("v") != pq.VERSION:
            return None
        if check_file and row.get("path"):
            st = self._pic_stat(row["path"])
            if st and r.get("st") and st != r["st"]:
                return None
        return r

    def _pic_needs(self, row: dict) -> bool:
        if row.get("tier") in SKIP_TIERS or not row.get("path"):
            return False
        r = self._pic_measures().get(self._pic_key(row))
        if r is None or r.get("v") != pq.VERSION:
            return True
        st = self._pic_stat(row["path"])
        return bool(st and r.get("st") and st != r["st"])        # replaced since: measure again

    # --- measuring ---------------------------------------------------------------------

    def picture_measure_row(self, row: dict) -> dict:
        key, path = self._pic_key(row), row["path"]
        base = {"key": key, "scene_id": row["scene_id"], "st": self._pic_stat(path), "t": time.time()}
        try:
            rec = pq.measure(path)
        except Exception as exc:  # noqa: BLE001 — recorded; retried once the file changes
            rec = {**base, "v": pq.VERSION, "error": str(exc)[:300]}
            self._pic_append("measures.jsonl", rec)
            return rec
        rec.update(base)
        stills = self._pic_dir() / "stills"
        stills.mkdir(exist_ok=True)
        safe = self._pic_safe(key)
        try:
            pq.poster_still(path, rec, stills / f"{safe}_full.jpg", duration=rec.get("duration") or 0.0)
        except Exception:  # noqa: BLE001 — the measures still count without the pictures
            pass
        self._pic_append("measures.jsonl", rec)
        self._pic_dirty = True
        return rec

    def _pic_state(self) -> dict:
        try:
            return json.loads((self._pic_dir() / "state.json").read_text())
        except (OSError, ValueError):
            return {}

    def _pic_save_state(self, d: dict) -> None:
        from .. import atomic

        atomic.write_json(self._pic_dir() / "state.json", d)

    def picture_backfill(self, on: bool) -> dict:
        """Measure the whole library (when idle, resumable) — or stop."""
        st = self._pic_state()
        st["backfill"] = bool(on)
        if on:
            st.setdefault("started", time.time())
        self._pic_save_state(st)
        return self.picture_status()

    def _pic_todo(self) -> tuple[list[dict], list[dict]]:
        """(owed: new scenes from the last ingest, backfill: the rest — unreviewed
        first, then best tiers first) still needing a measure."""
        rows = self._catalogue_all()
        fresh = set(self.last_ingest().get("new") or [])
        owed = [r for r in rows if r["scene_id"] in fresh and self._pic_needs(r)]
        if not self._pic_state().get("backfill"):
            return owed, []
        order = {"unreviewed": 0, "legendaire": 1, "exceptionnelle": 2, "merveilleuse": 3, "anomaly": 4, "upscale": 5}
        rest = sorted((r for r in rows if r["scene_id"] not in fresh and self._pic_needs(r)),
                      key=lambda r: order.get(r["tier"], 9))
        return owed, rest

    def picture_due(self) -> bool:
        try:
            owed, rest = self._pic_todo()
        except Exception:  # noqa: BLE001 — Stash down: try again next minute
            return False
        return bool(owed or rest)

    def run_picture(self, job=None) -> dict:
        """One pass: new files first, then (if on) the library backfill. Stops on
        cancel; picks up where it left off next time."""
        log = job.log if job else print
        owed, rest = self._pic_todo()
        todo = owed + rest
        done = failed = 0
        t0 = time.time()
        if job is not None:
            job.progress = {"total": len(todo), "done": 0}
        for i, row in enumerate(todo):
            if job is not None and job.cancelled:
                log("stopped — the rest waits for the next pass")
                break
            rec = self.picture_measure_row(row)
            if rec.get("error"):
                failed += 1
            else:
                done += 1
            if job is not None:
                rate = (time.time() - t0) / (i + 1)
                job.progress = {"total": len(todo), "done": i + 1, "eta_s": round(rate * (len(todo) - i - 1))}
        if self._pic_state().get("backfill") and not (job is not None and job.cancelled) and not self._pic_todo()[1]:
            st = self._pic_state()
            st.update(backfill=False, finished=time.time())
            self._pic_save_state(st)
        log(f"measured {done} file(s){f', {failed} unreadable' if failed else ''}")
        return {"measured": done, "failed": failed}

    # --- your picks and the model ---------------------------------------------------------

    def _pic_picks(self) -> list[dict]:
        p = self._pic_dir() / "pairs.jsonl"
        out = []
        try:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        return out

    def _pic_ratings(self) -> dict[str, int]:
        out: dict[str, int] = {}
        try:
            with open(self._pic_dir() / "ratings.jsonl", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        out[str(r["key"])] = int(r["band"])
                    except (ValueError, KeyError, TypeError):
                        continue
        except OSError:
            pass
        return out

    def picture_rate(self, key: str, band: int) -> dict:
        """Your rating of one file's picture: 1 Poor … 5 Reference."""
        band = int(band)
        if not 1 <= band <= 5:
            raise ValueError("band is 1 (Poor) to 5 (Reference)")
        self._pic_append("ratings.jsonl", {"key": str(key), "band": band, "t": time.time()})
        self._pic_dirty = True
        return {"rated": len(self._pic_ratings())}

    @staticmethod
    def _pic_scored(rec: dict | None) -> bool:
        """Measured, current, and not VR (VR isn't rated or scored)."""
        return bool(rec) and not rec.get("error") and rec.get("v") == pq.VERSION and not rec.get("vr")

    def _pic_rows_by_key(self) -> dict[str, dict]:
        return {self._pic_key(r): r for r in self._catalogue_all()}

    def _pic_feats(self) -> dict[str, "pq.np.ndarray"]:
        by_key = self._pic_rows_by_key()
        return {k: pq.features(r, (by_key.get(k) or {}).get("quality"))
                for k, r in self._pic_measures().items() if self._pic_scored(r)}

    def _pic_model(self) -> pq.PictureModel:
        """The trained model — retrained when picks or measures changed."""
        with self._pic_lock():
            m = self.__dict__.get("_pic_model_obj")
            if m is not None and not self.__dict__.get("_pic_dirty"):
                return m
            if m is None and not self.__dict__.get("_pic_dirty"):
                try:
                    m = pq.PictureModel.from_state(json.loads((self._pic_dir() / "model.json").read_text()))
                    self._pic_model_obj = m
                    return m
                except (OSError, ValueError):
                    pass
            m = pq.PictureModel()
            m.train(self._pic_feats(), self._pic_picks(), self._pic_ratings())
            from .. import atomic

            atomic.write_json(self._pic_dir() / "model.json", m.state())
            self._pic_model_obj, self._pic_dirty = m, False
            self.__dict__.pop("_pic_scores_cache", None)
            return m

    def picture_answer(self, a: str, b: str, ans: str, src: str = "you") -> dict:
        if ans not in ("a", "b", "fine", "bad"):
            raise ValueError("answer is a, b, fine or bad")
        self._pic_append("pairs.jsonl", {"a": a, "b": b, "ans": ans, "src": src, "t": time.time()})
        self._pic_dirty = True
        return {"picks": len(self._pic_picks())}

    def picture_note_keep(self, keep: str, delete: list[str]) -> None:
        """You chose which duplicate to keep: if both copies were measured,
        that's a pick (kept one looks better)."""
        try:
            rows = {r["scene_id"]: r for r in self._catalogue_all()}
            k = rows.get(str(keep))
            if not k or not self._pic_rec(k):
                return
            for d in delete:
                r = rows.get(str(d))
                if r and self._pic_rec(r):
                    self.picture_answer(self._pic_key(k), self._pic_key(r), "a", src="duplicate")
        except Exception:  # noqa: BLE001 — a hint, never an error
            pass

    def picture_pair(self, rng: random.Random | None = None) -> dict | None:
        """The next "which looks better?" pair: two measured files with pictures,
        not asked before — similar content when possible (a shared performer or
        the same resolution), and once there's a model, the closest call."""
        rng = rng or random.Random()
        rows = [r for r in self._catalogue_all() if r.get("tier") not in SKIP_TIERS]
        cand = [(r, rec) for r in rows if r.get("path") for rec in [self._pic_rec(r)] if self._pic_scored(rec)]
        if len(cand) < 2:
            return None
        asked = {frozenset((p["a"], p["b"])) for p in self._pic_picks()}
        m = self._pic_model()
        # the files Peaks is least sure about come up more often
        conf = {sid: v.get("conf") for sid, v in self.picture_scores().items()} if m.trained else {}
        weight = [{"guess": 3, "fairly sure": 2, "sure": 1}.get(conf.get(r["scene_id"]), 2) for r, _ in cand]
        by_perf: dict[str, list[int]] = {}
        for i, (r, _) in enumerate(cand):
            for pid in r.get("performer_ids") or []:
                by_perf.setdefault(pid, []).append(i)
        for _ in range(60):
            i = rng.choices(range(len(cand)), weights=weight, k=1)[0]
            ra, ka = cand[i][0], self._pic_key(cand[i][0])
            pool = [j for pid in ra.get("performer_ids") or [] for j in by_perf.get(pid, []) if j != i]
            if not pool or rng.random() < 0.4:
                h = cand[i][1].get("h")
                pool = [j for j, (_, rec) in enumerate(cand) if j != i and pq.res_words(rec.get("h")) == pq.res_words(h)]
            if not pool:
                pool = [j for j in range(len(cand)) if j != i]
            pool = [j for j in pool if frozenset((ka, self._pic_key(cand[j][0]))) not in asked]
            if not pool:
                continue
            if m.trained:
                xa = pq.features(cand[i][1], ra.get("quality"))
                sa = m.raw(xa)
                sample = rng.sample(pool, min(40, len(pool)))
                j = min(sample, key=lambda j: abs(m.raw(pq.features(cand[j][1], cand[j][0].get("quality"))) - sa))
            else:
                j = rng.choice(pool)
            a, b = (cand[i], cand[j]) if rng.random() < 0.5 else (cand[j], cand[i])
            return {"a": self._pic_side(*a), "b": self._pic_side(*b)}
        return None

    def picture_one(self, rng: random.Random | None = None) -> dict | None:
        """The next file to rate on its own: unrated, not VR, the least sure
        first — and spread over the bands, so your ratings cover the whole scale."""
        rng = rng or random.Random()
        rated = self._pic_ratings()
        rows = [r for r in self._catalogue_all() if r.get("tier") not in SKIP_TIERS and r.get("path")]
        cand = [(r, rec) for r in rows for rec in [self._pic_rec(r)]
                if self._pic_scored(rec) and self._pic_key(r) not in rated]
        if not cand:
            return None
        m = self._pic_model()
        pics = self.picture_scores() if m.trained else {}
        if m.trained:
            # aim for the stretch of the scale with the fewest ratings so far (fifths of your library)
            def slot(r):
                p = pics.get(r["scene_id"]) or {}
                return min(4, int((p.get("rank") or 50) / 20))
            by_key = self._pic_rows_by_key()
            have = [0] * 5
            for k in rated:
                if k in by_key and by_key[k]["scene_id"] in pics:
                    have[slot(by_key[k])] += 1
            slots: dict[int, list] = {}
            for c in cand:
                slots.setdefault(slot(c[0]), []).append(c)
            target = min(slots, key=lambda s: (have[s], rng.random()))
            order = {"guess": 0, "fairly sure": 1, "sure": 2}
            pool = slots[target]
            least = min(order.get((pics.get(r["scene_id"]) or {}).get("conf"), 0) for r, _ in pool)
            pool = [c for c in pool if order.get((pics.get(c[0]["scene_id"]) or {}).get("conf"), 0) == least]
        else:
            pool = cand
        r, rec = rng.choice(pool)
        return {"one": self._pic_side(r, rec), "rated": len(rated)}

    def _pic_side(self, row: dict, rec: dict) -> dict:
        safe = self._pic_safe(self._pic_key(row))
        q = row.get("quality") or {}
        poster = self._pic_dir() / "stills" / f"{safe}_full.jpg"
        return {"key": self._pic_key(row), "scene_id": row["scene_id"], "title": row["title"],
                "res": q.get("res"), "mbps": q.get("mbps"), "codec": (q.get("codec") or "").upper() or None,
                "stream": self.stream_url(row["scene_id"], start=0),
                "duration": rec.get("duration") or row.get("duration") or 0,
                "poster": f"/api/picture/still/{safe}/full" if poster.exists() else None}

    def picture_still(self, safe: str, kind: str) -> Path | None:
        if kind != "full" or not safe.isalnum():
            return None
        p = self._pic_dir() / "stills" / f"{safe}_{kind}.jpg"
        return p if p.exists() else None

    # --- the score where it's used ------------------------------------------------------------

    _PIC_SCORES_FRESH = 60.0

    def picture_scores(self) -> dict[str, dict]:
        """scene_id → {score (0–100 on your scale, once calibrated), band,
        conf (how sure: you judged it / close to files you judged / a guess),
        rank (percentile, for ordering before calibration), ok (good enough),
        subs (five 0–10 marks), reasons, real (real-detail height), upscaled,
        vr (VR isn't scored)}."""
        import time as _t

        m = self._pic_model()
        c = self.__dict__.get("_pic_scores_cache")
        # while a measuring job appends to the file every few seconds, every
        # Catalogue click used to re-read it and re-score the library: within a
        # minute the last scores stand (a new rating/model still counts at once)
        if c and c[0][0] == id(m) and _t.monotonic() - c[2] < self._PIC_SCORES_FRESH:
            return c[1]
        meas = self._pic_measures()
        sig = (id(m), id(meas))
        if c and c[0] == sig:
            self._pic_scores_cache = (sig, c[1], _t.monotonic())
            return c[1]
        rows = []
        for r in self._catalogue_all():
            rec = meas.get(self._pic_key(r))
            if rec and not rec.get("error") and rec.get("v") == pq.VERSION:
                rows.append((r, rec))
        conf = self._pic_confidence(m, [(r, rec) for r, rec in rows if not rec.get("vr")]) if m.trained else {}
        out = {}
        for r, rec in rows:
            base = {"reasons": pq.reasons(rec), "real": rec.get("eff_h"), "subs": pq.subscores(rec),
                    "upscaled": (rec.get("eff_factor") or 1) >= 1.3, "vr": bool(rec.get("vr")),
                    "score": None, "band": None, "conf": None, "rank": None, "ok": None}
            if not rec.get("vr") and m.trained:
                x = pq.features(rec, r.get("quality"))
                sc = m.score(x)
                base.update(score=sc, band=pq.band_of(sc), rank=m.rank(x), ok=m.acceptable(x),
                            conf=conf.get(self._pic_key(r)), reasons=pq.reasons(rec, m, x))
            out[r["scene_id"]] = base
        self._pic_scores_cache = (sig, out, _t.monotonic())
        return out

    def _pic_confidence(self, m: pq.PictureModel, rows: list) -> dict[str, str]:
        """key → "sure" (you rated it, or it's been in 3+ of your picks),
        "fairly sure" (close to files you've judged — nearer than half the
        library is) or "guess"."""
        import numpy as np

        rated = self._pic_ratings()
        seen: dict[str, int] = {}
        for p in self._pic_picks():
            if p.get("ans") in ("a", "b", "fine", "bad"):
                for k in (p.get("a"), p.get("b")):
                    seen[k] = seen.get(k, 0) + 1
        keys = [self._pic_key(r) for r, _ in rows]
        if not keys:
            return {}
        Z = np.stack([m._z(pq.features(rec, r.get("quality"))) for r, rec in rows]).astype(np.float32)
        judged = [i for i, k in enumerate(keys) if k in rated or k in seen]
        out = {k: ("sure" if k in rated or seen.get(k, 0) >= 3 else "guess") for k in keys}
        if not judged:
            return out
        J = Z[judged]
        near = np.empty(len(keys), np.float32)
        for s0 in range(0, len(keys), 2048):                     # nearest judged file, in chunks
            d = ((Z[s0:s0 + 2048, None, :] - J[None, :, :]) ** 2).sum(axis=2)
            near[s0:s0 + 2048] = np.sqrt(d.min(axis=1))
        cut = float(np.median(near))
        for i, k in enumerate(keys):
            if out[k] == "guess" and near[i] <= cut:
                out[k] = "fairly sure"
        return out

    def picture_trusted(self) -> bool:
        try:
            return self._pic_model().trusted
        except Exception:  # noqa: BLE001
            return False

    def picture_status(self) -> dict:
        rows = [r for r in self._catalogue_all() if r.get("tier") not in SKIP_TIERS and r.get("path")]
        meas = self._pic_measures()
        measured = sum(1 for r in rows if (meas.get(self._pic_key(r)) or {}).get("v") == pq.VERSION
                       and not meas[self._pic_key(r)].get("error"))
        failed = sum(1 for r in rows if (meas.get(self._pic_key(r)) or {}).get("error"))
        picks = self._pic_picks()
        m = self._pic_model()
        st = self._pic_state()
        bands = {b: 0 for b in pq.BANDS}
        vr = 0
        for v in self.picture_scores().values():
            if v.get("vr"):
                vr += 1
            elif v.get("band"):
                bands[v["band"]] += 1
        return {"measured": measured, "total": len(rows), "failed": failed,
                "backfill": bool(st.get("backfill")), "finished": st.get("finished"),
                "picks": sum(1 for p in picks if p.get("src") == "you"),
                "from_duplicates": sum(1 for p in picks if p.get("src") == "duplicate"),
                "rated": len(self._pic_ratings()), "bands": bands, "vr": vr,
                "model": m.report}
