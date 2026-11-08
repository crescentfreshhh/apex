"""Faces: who is in a scene, recognised by face (InsightFace ArcFace).

The "faces" job looks at a dozen frames spread through each scene (decoded on
the CPU — one seek a frame — and run through the face models on the GPU) and
keeps every clear face it finds: where, when, and its 512-number ArcFace
embedding, in the embedding cache under FACE_NS.

Each performer gets a gallery of her faces:
  - her Stash photo (the largest face in it), and
  - faces from scenes she's credited in: the ones that match her photo, or —
    with no photo, in a scene she's the only one credited in — the face that
    recurs most.
A scene's faces are then compared with every gallery: a performer is suggested
when several of its faces match hers. Suggestions only — nothing is written
without a click (Orphans' Add, the performer page's "may also be in").
"""

from __future__ import annotations

import time

import numpy as np

from ..atomic import settings_saver as _settings_saver

FACE_NS = "arcface@faces"
REF_NS = "arcface@performers"
FACES_PER_RUN = 150
FRAMES = 12                # frames looked at per scene
MATCH = 0.42               # cosine: same person (ArcFace buffalo_l)
SURE = 0.52
MIN_FACES = 2              # matching faces in a scene before a name is suggested
GALLERY = 40               # faces kept per performer
SCENE_CLUSTER = 0.45


class FacesMixin:
    # --- settings / state ----------------------------------------------------------------

    def faces_settings(self) -> dict:
        f = self._settings().get("faces") or {}
        return {"on": bool(f.get("on", True)), "per_run": int(f.get("per_run") or FACES_PER_RUN)}

    @_settings_saver
    def save_faces_settings(self, on=None, per_run=None) -> dict:
        from .inputs import as_bool, as_int

        s = dict(self._settings())
        f = dict(s.get("faces") or {})
        if on is not None:
            f["on"] = as_bool(on, "on")
        if per_run is not None:
            f["per_run"] = max(1, min(5000, as_int(per_run, "per_run")))
        s["faces"] = f
        self._save_settings_dict(s)
        return self.faces_settings()

    def _face_cache(self):
        from ..cache import EmbeddingCache

        return EmbeddingCache(self.cfg.embedding.cache_dir)

    def _face_keys(self) -> dict[str, str]:
        """scene_id → cache key, for scenes in the visual index."""
        return self._hear_keys()

    def faces_todo(self) -> list[dict]:
        """Scenes not looked at yet: orphans first (they need names most), then keepers."""
        from .service import TIER_RANK_SORT

        keys = self._face_keys()
        cache = self._face_cache()
        rows = [r for r in self._catalogue_all() if r["tier"] != "rejected" and r["scene_id"] in keys
                and r.get("path") and not cache.has(keys[r["scene_id"]], FACE_NS)]
        excluded = self._excluded_performers()
        return sorted(rows, key=lambda r: (not self._is_orphan(r, excluded), -TIER_RANK_SORT.get(r["tier"], 0)))

    def faces_due(self) -> bool:
        return self.faces_settings()["on"] and self._due_memo("faces", lambda: bool(self.faces_todo()))

    def faces_status(self) -> dict:
        try:
            keys = self._face_keys()
            cache = self._face_cache()
            live = [keys[r["scene_id"]] for r in self._catalogue_all()       # rejects aren't looked at
                    if r["tier"] != "rejected" and r["scene_id"] in keys]
            seen = sum(1 for k in live if cache.has(k, FACE_NS))
            total = len(live)
        except Exception:  # noqa: BLE001
            seen = total = None
        g = self.__dict__.get("_face_gal")
        return {"seen": seen, "embedded": total, "left": (total - seen) if total is not None else None,
                "settings": self.faces_settings(),
                "performers": len(g[1]["owners_set"]) if g else None,
                "error": getattr(self, "_faces_error", None), "last_run": self.sense_last_run("faces")}

    # --- the job --------------------------------------------------------------------------

    def _face_embedder(self):
        from ..faces import FaceEmbedder

        return FaceEmbedder(log=getattr(self, "_faces_log", print))

    def _face_frame(self, path: str, t: float):
        from ..sampling import FrameSampler

        return FrameSampler(hwaccel=None).grab_frame(path, t, width=960)   # one seek: the CPU is quickest

    def face_scene(self, row: dict, embedder) -> dict:
        sid = row["scene_id"]
        key = self._face_keys().get(sid)
        if key is None:
            raise LookupError(f"scene {sid} isn't embedded")
        dur = float(row.get("duration") or 0)
        times = [dur * (i + 0.5) / FRAMES for i in range(FRAMES)] if dur > 0 else [5.0]
        ts, vecs, boxes, scores = [], [], [], []
        for t in times:
            try:
                img = self._face_frame(row["path"], t)
            except Exception:  # noqa: BLE001 — one bad seek: the other frames still count
                continue
            for f in embedder.faces(img):
                ts.append(round(t, 2))
                vecs.append(f.vec)
                boxes.append([round(x, 1) for x in f.box])
                scores.append(round(f.score, 3))
        V = np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, embedder.dim), np.float32)
        self._face_cache().save(key, FACE_NS, np.asarray(ts, np.float32), V,
                                meta={"scene_id": sid, "boxes": boxes, "scores": scores, "frames": len(times)})
        self.__dict__.pop("_face_gal", None)
        self.__dict__.pop("_face_all", None)
        return {"faces": len(ts)}

    def _face_refs(self, embedder, log) -> int:
        """Her Stash photo's face, for every performer with a photo not done yet."""
        cache = self._face_cache()
        done = 0
        for p in self._stash_performers():
            if not p.get("has_image") or cache.has(f"p{p['id']}", REF_NS):
                continue
            V = np.zeros((0, embedder.dim), np.float32)
            try:
                got = self._meta_client().performer_image(p["id"])
                if got:
                    from io import BytesIO

                    from PIL import Image

                    img = Image.open(BytesIO(got[0])).convert("RGB")
                    fs = sorted(embedder.faces(img), key=lambda f: -f.size)
                    if fs:
                        V = fs[0].vec[None].astype(np.float32)
            except Exception:  # noqa: BLE001
                pass
            cache.save(f"p{p['id']}", REF_NS, np.zeros(len(V), np.float32), V, meta={"pid": p["id"]})
            done += 1
        if done:
            log(f"faces: {done} performer photo(s) read")
            self._face_refs_n = self.__dict__.get("_face_refs_n", 0) + done
            self.__dict__.pop("_face_gal", None)
        return done

    def run_faces(self, job=None, limit: int | None = None) -> dict:
        from .hear import SenseRun

        log = job.log if job else print
        todo = self.faces_todo()[: limit or self.faces_settings()["per_run"]]
        try:
            self._faces_log = log
            embedder = self._face_embedder()
        except Exception as exc:  # noqa: BLE001 — model missing: say so on the card
            self._faces_error = str(exc)
            raise
        self._faces_error = None
        self._face_refs(embedder, log)
        if not todo:
            return {"seen": 0}
        run = SenseRun()
        done = found = 0
        t0 = time.time()
        self._sense_record("faces", {"running": True, "done": 0, **run.summary()})
        for i, row in enumerate(todo):
            if job is not None and job.cancelled:
                run.stopped = (f"paused for {job.paused_for}" if getattr(job, "paused_for", None) else "you stopped it")
                break
            try:
                found += self.face_scene(row, embedder)["faces"]
                done += 1
                run.ok()
            except Exception as exc:  # noqa: BLE001
                log(f"scene {row['scene_id']}: {exc}")
                if run.fail(row["scene_id"], exc):
                    self._faces_error = run.stopped
                    log(run.stopped)
                    break
            if (i + 1) % 25 == 0:
                self._sense_record("faces", {"running": True, "done": done, **run.summary()})
            if job is not None:
                rate = (time.time() - t0) / (i + 1)
                job.progress = {"total": len(todo), "done": i + 1, "eta_s": round(rate * (len(todo) - i - 1))}
        log(f"faces: looked at {done} scene(s), {found} face(s)")
        if done:
            try:                                       # her face, alone, where pictures were covers / frames
                n = self.upgrade_performer_photos(job=job)
                if n:
                    log(f"faces: {n} performer picture(s) now show just her face")
            except Exception:  # noqa: BLE001 — pictures are extra
                pass
        self._sense_record("faces", {"done": done, "faces": found, **run.summary()})
        return {"seen": done, "faces": found, "failed": run.failed}

    # --- galleries -----------------------------------------------------------------------------

    def _scene_faces(self, sid: str):
        key = self._face_keys().get(str(sid))
        cache = self._face_cache()
        if key is None or not cache.has(key, FACE_NS):
            return None
        try:
            t, v, meta = cache.load(key, FACE_NS)
        except (OSError, KeyError, ValueError):
            return None
        v = np.asarray(v, np.float32)
        return t, v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9), meta

    @staticmethod
    def _dominant(v: np.ndarray) -> np.ndarray:
        """Indices of the face that recurs most in a scene, and its look-alikes."""
        if not len(v):
            return np.zeros(0, int)
        s = v @ v.T
        hub = int((s >= SCENE_CLUSTER).sum(axis=1).argmax())
        return np.where(s[hub] >= SCENE_CLUSTER)[0]

    def _face_gallery(self):
        """(matrix, owners, info): every performer's gallery faces stacked, the
        performer of each row. Rebuilt when faces or credits change (per catalogue)."""
        from ..tier_model import who_of

        rows = self._catalogue_all()
        credits = hash(tuple((r["scene_id"], tuple(r.get("performer_ids") or ()), r["tier"] == "rejected")
                             for r in rows))       # who is credited where (a grade alone changes nothing here)
        stamp = (len(rows), credits, self.__dict__.get("_sense_runs", 0), self.__dict__.get("_face_refs_n", 0))
        hit = self.__dict__.get("_face_gal")
        if hit is not None and hit[0] == stamp:
            return hit[1]
        cache = self._face_cache()
        refs: dict[str, np.ndarray] = {}
        for k in cache.keys(REF_NS):
            try:
                _, v, meta = cache.load(k, REF_NS)
            except (OSError, KeyError, ValueError):
                continue
            if len(v):
                x = np.asarray(v[0], np.float32)
                refs[str(meta.get("pid") or k[1:])] = x / max(float(np.linalg.norm(x)), 1e-9)
        gal: dict[str, list[np.ndarray]] = {pid: [r] for pid, r in refs.items()}
        for r in rows:
            pids = list(who_of(r)["performers"])
            if not pids or r["tier"] == "rejected":
                continue
            sf = self._scene_faces(r["scene_id"])
            if sf is None or not len(sf[1]):
                continue
            v = sf[1]
            for pid in pids:
                if pid in refs:
                    mine = v[(v @ refs[pid]) >= MATCH]
                elif len(pids) == 1:
                    dom = self._dominant(v)
                    # only when one face clearly dominates: an uncredited co-star mustn't become her
                    mine = v[dom] if len(dom) >= 0.6 * len(v) else v[:0]
                else:
                    continue
                if len(mine):
                    gal.setdefault(pid, []).extend(mine[:6])
        mats, owners = [], []
        for pid, vs in gal.items():
            vs = vs[:1] + vs[1:][-(GALLERY - 1):] if len(vs) > GALLERY else vs
            mats.append(np.stack(vs))
            owners += [pid] * len(vs)
        out = {"M": np.concatenate(mats).astype(np.float32) if mats else np.zeros((0, 512), np.float32),
               "owners": np.asarray(owners), "owners_set": set(owners), "photo": set(refs)}
        self._face_gal = (stamp, out)
        return out

    # --- who is in this scene -------------------------------------------------------------------

    def face_matches(self, scene_id: str, k: int = 5) -> dict:
        """[{id, name, score, faces, strength, credited, t}] — performers whose
        faces appear in the scene, best first."""
        sid = str(scene_id)
        sf = self._scene_faces(sid)
        if sf is None:
            return {"scene_id": sid, "matches": [], "scanned": False}
        t, v, _ = sf
        g = self._face_gallery()
        if not len(v) or not len(g["M"]):
            return {"scene_id": sid, "matches": [], "scanned": True, "faces": int(len(v))}
        sims = v @ g["M"].T                                     # faces × gallery rows
        by: dict[str, np.ndarray] = {}
        for pid in g["owners_set"]:
            by[pid] = sims[:, g["owners"] == pid].max(axis=1)   # each face's best match to her
        row = next((r for r in self._catalogue_all() if r["scene_id"] == sid), {})
        credited = {str(p) for p in row.get("performer_ids") or []}
        no = set(self._orph_no().get(sid, []))
        names = {p["id"]: p["name"] for p in self._stash_performers()}
        out = []
        for pid, best in by.items():
            hits = best >= MATCH
            if hits.sum() < MIN_FACES or pid in no:
                continue
            top = np.sort(best[hits])[::-1][:3]
            score = float(top.mean())
            j = int(np.argmax(best))
            out.append({"id": pid, "name": names.get(pid, ""), "score": round(score, 4), "faces": int(hits.sum()),
                        "strength": "strong" if score >= SURE and hits.sum() >= 3 else "possible",
                        "credited": pid in credited, "t": float(t[j]),
                        "thumb": f"/api/frame?key={self._face_keys().get(sid)}&t={float(t[j]):g}"})
        out.sort(key=lambda x: (-x["faces"] * x["score"]))
        return {"scene_id": sid, "matches": out[:k], "scanned": True, "faces": int(len(v))}

    def _face_all(self):
        """Every scanned scene's faces in one matrix (float16) with its scene id —
        for "which scenes is she in?". Rebuilt after each face run."""
        hit = self.__dict__.get("_face_all")
        if hit is not None:
            return hit
        mats, owners, times = [], [], []
        for sid in self._face_keys():
            sf = self._scene_faces(sid)
            if sf is None or not len(sf[1]):
                continue
            mats.append(sf[1].astype(np.float16))
            owners += [sid] * len(sf[1])
            times += list(map(float, sf[0]))
        out = (np.concatenate(mats) if mats else np.zeros((0, 512), np.float16), np.asarray(owners), np.asarray(times))
        self._face_all = out
        return out

    def face_uncredited(self, pid: str, k: int = 30) -> dict:
        """Scenes she seems to be in (by face) that don't credit her — best first."""
        pid = str(pid)
        g = self._face_gallery()
        mine = g["M"][g["owners"] == pid] if len(g["M"]) else np.zeros((0, 512), np.float32)
        if not len(mine):
            return {"id": pid, "scenes": [], "reason": "no faces of hers yet — a Stash photo or a credited scene scanned"}
        A, owners, times = self._face_all()
        if not len(A):
            return {"id": pid, "scenes": [], "reason": "no scenes scanned for faces yet"}
        best = np.zeros(len(A), np.float32)
        for a in range(0, len(A), 20000):
            best[a: a + 20000] = (A[a: a + 20000].astype(np.float32) @ mine.T).max(axis=1)
        rows = {r["scene_id"]: r for r in self._catalogue_all()}
        out = []
        for sid in set(owners[best >= MATCH]):
            r = rows.get(sid)
            if r is None or r["tier"] == "rejected" or pid in {str(p) for p in r.get("performer_ids") or []}:
                continue
            m = owners == sid
            b = best[m]
            n = int((b >= MATCH).sum())
            if n < MIN_FACES or str(pid) in self._orph_no().get(sid, []):
                continue
            j = int(np.argmax(b))
            out.append({"scene_id": sid, "title": r.get("title") or f"scene {sid}", "faces": n,
                        "score": round(float(np.sort(b)[::-1][:3].mean()), 4), "tier": r["tier"],
                        "thumb": f"/api/frame?key={self._face_keys().get(sid)}&t={float(times[m][j]):g}"})
        out.sort(key=lambda x: (-x["faces"] * x["score"]))
        return {"id": pid, "scenes": out[:k]}
