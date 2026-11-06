"""The Performers directory: every performer in Stash, searchable, with a
proper picture — cached on disk so the page opens instantly.

Pictures (first that exists):
  1. the one you chose on her page (models/performer_photos.json)
  2. her Stash profile image (usually a portrait / full-body shot — skipped
     when Stash would serve its default silhouette)
  3. her face, alone: a frame from the Faces scan holding exactly one face — hers —
     cropped to a portrait around it (no new scanning: the scan kept every face's
     time, box and identity)
  4. the cover of her best-graded solo scene (only her in it)
  5. her best frame (the taste model's pick)
  6. the cover of any scene she's in (even a rejected one — it still shows her)
Automatic pictures from 4–6 are upgraded to 3 once the Faces scan has found her.
Each is resized once and kept in models/perf_photos/{id}.jpg; the browser
caches it too (the URL carries a version that changes with the picture).

The directory itself (Stash's performer list) is kept in memory and on disk:
opening Performers serves the last copy at once while a background refresh
reads Stash again.
"""

from __future__ import annotations

from .. import atomic

import gzip
import json
import threading
import time
from io import BytesIO
from pathlib import Path

FACE_MIN_SCORE = 0.7      # detection confidence of a portrait's face
FACE_MIN_PX = 80          # its size, in the scan's 960-px-wide frame
FACE_SCENES = 80          # her scenes looked at for portraits (best-graded first)
UPGRADABLE = ("cover", "frame", None)     # automatic pictures a face portrait replaces


def portrait_box(W: int, H: int, box) -> tuple[int, int, int, int]:
    """A 2:3 crop around a face: the face ~40% of its height, a little above the
    middle (room for shoulders), kept inside the frame."""
    x1, y1, x2, y2 = (float(v) for v in box)
    fh = max(y2 - y1, 1.0)
    ch = min(fh / 0.4, float(H))
    cw = ch * 2 / 3
    if cw > W:
        cw = float(W)
        ch = min(cw * 1.5, float(H))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(cx - cw / 2, 0.0), W - cw)
    top = min(max(cy - 0.38 * ch, 0.0), H - ch)
    return int(round(left)), int(round(top)), int(round(left + cw)), int(round(top + ch))


TIER_RANK = {"legendaire": 6, "exceptionnelle": 5, "merveilleuse": 4, "anomaly": 3, "unreviewed": 2, "rejected": 0}


class PerformersMixin:
    _PERF_DIR_TTL = 600.0          # re-read Stash's performer list after 10 min
    _PHOTO_BOX = (480, 720)        # max width × height of a cached picture

    # --- Stash's performer list, cached -----------------------------------------

    def _perf_dir_path(self) -> Path:
        return Path(self.cfg.modeling.dir) / "performer_directory.json.gz"

    def _stash_performers(self, refresh: bool = False) -> list[dict]:
        """Every Stash performer (peaks.stash_client.iter_performers). Memory,
        then disk, then Stash; a stale copy is served while a background read
        replaces it."""
        cached = self.__dict__.get("_perf_dir_cache")
        if cached is None and not refresh:
            try:
                with gzip.open(self._perf_dir_path(), "rt", encoding="utf-8") as fh:
                    d = json.load(fh)
                cached = (0.0, d["performers"])        # from disk: treat as stale
                self._perf_dir_cache = cached
            except (OSError, ValueError, KeyError, TypeError):
                cached = None
        if cached is None or refresh:
            return self._fetch_stash_performers()
        if time.monotonic() - cached[0] > self._PERF_DIR_TTL:
            self._refresh_perf_dir_bg()
        return cached[1]

    def _fetch_stash_performers(self) -> list[dict]:
        perfs = list(self._meta_client().iter_performers())
        self._perf_dir_cache = (time.monotonic(), perfs)
        self.__dict__.pop("_perf_by_id", None)
        try:
            p = self._perf_dir_path()
            from .. import atomic

            with atomic.temp_for(p) as tmp, gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump({"ts": time.time(), "performers": perfs}, fh)
        except OSError:
            pass
        return perfs

    def _refresh_perf_dir_bg(self) -> None:
        with self.__dict__.setdefault("_perf_dir_gate", threading.Lock()):
            if self.__dict__.get("_perf_dir_refreshing"):
                return
            self._perf_dir_refreshing = True

        def run():
            try:
                self._fetch_stash_performers()
            except Exception:  # noqa: BLE001 — Stash down: keep serving the last copy
                pass
            finally:
                self._perf_dir_refreshing = False
        threading.Thread(target=run, daemon=True, name="peaks-perf-dir").start()

    def _perf_info(self, pid: str) -> dict | None:
        by = self.__dict__.get("_perf_by_id")
        perfs = self._stash_performers()
        if by is None or by[0] is not perfs:
            by = (perfs, {p["id"]: p for p in perfs})
            self._perf_by_id = by
        return by[1].get(str(pid))

    # --- Peaks' per-performer figures, without blocking the page ------------------

    def _perf_stats_nowait(self) -> list[dict] | None:
        """The taste leaderboard (`performer_stats`) if it's built; otherwise
        start building it in the background and return None for now."""
        cached = self.__dict__.get("_perf_stats_cache")
        if cached is not None:
            return cached
        with self.__dict__.setdefault("_perf_stats_gate", threading.Lock()):
            if self.__dict__.get("_perf_stats_building"):
                return None
            self._perf_stats_building = True

        def run():
            try:
                self.performer_stats()
            except Exception:  # noqa: BLE001 — Stash down: the directory works without it
                pass
            finally:
                self._perf_stats_building = False
        threading.Thread(target=run, daemon=True, name="peaks-perf-stats").start()
        return None

    def performer_directory(self) -> dict:
        """Every performer, merged with what Peaks knows about her: moments and
        taste (once the leaderboard is built), her track record in your grades
        (in plain words), when the megaboard last showed her, and flags for the
        page's sections. Cheap to recompute: everything underneath is cached."""
        import datetime as _dt

        from ..tier_model import TIER_CLASS, who_of
        from ..words import rank, verdict

        perfs = self._stash_performers()
        stats = self._perf_stats_nowait()
        try:
            rows = self._catalogue_all()
        except Exception:  # noqa: BLE001 — Stash down: no records this time
            rows = []
        enc = self._who_records(rows)
        base = enc.base[0]
        try:
            seen = self.exposure().last_seen()
        except Exception:  # noqa: BLE001
            seen = {}
        try:
            fresh = set(self.last_ingest().get("new") or [])
        except Exception:  # noqa: BLE001
            fresh = set()
        grace = float(getattr(self.cfg.modeling, "trim_grace_days", 30) or 30)
        now = _dt.datetime.now(_dt.timezone.utc)

        per: dict[str, dict] = {}
        for r in rows:
            try:
                t = _dt.datetime.fromisoformat((r.get("created_at") or "").replace("Z", "+00:00"))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=_dt.timezone.utc)
                age = (now - t).days
            except ValueError:
                age = None
            for pid in who_of(r)["performers"]:
                d = per.setdefault(pid, {"lib": 0, "graded": 0, "recent": False, "seen": 0.0, "tiers": {}})
                d["lib"] += 1
                d["graded"] += r["tier"] in TIER_CLASS
                d["tiers"][r["tier"]] = d["tiers"].get(r["tier"], 0) + 1
                d["recent"] |= r["scene_id"] in fresh or (age is not None and age < grace)
                d["seen"] = max(d["seen"], seen.get(r["scene_id"], 0.0))

        st = {s["id"]: s for s in (stats or [])}
        affs = sorted((s["affinity"] for s in st.values() if s.get("affinity") is not None), reverse=True)
        out = []
        now_ts = time.time()
        for p in perfs:
            pid = p["id"]
            d = per.get(pid, {})
            s = st.get(pid, {})
            t = enc.perf.get(pid)
            rec = None
            if t:
                n, g, _, _ = enc.record(t)          # smoothed grade for the verdict…
                keep, top = t[2] / t[0], t[3] / t[0]  # …your actual shares for display
                word, tone = verdict(n, g, keep, base)
                rec = {"n": int(n), "verdict": word, "tone": tone, "keep": round(keep, 3), "top": round(top, 3)}
            aff = s.get("affinity")
            taste = None
            if aff is not None and affs:
                at = next((i for i, a in enumerate(affs) if a <= aff), len(affs))
                k, w = rank(at / len(affs))
                taste = {"key": k, "word": w, "affinity": round(aff, 4)}
            last = d.get("seen") or 0.0
            good = bool(rec and rec["tone"] == "good")
            out.append({
                "id": pid, "name": p["name"], "aliases": p.get("aliases") or [],
                "fav": p.get("favorite", False), "rating": p.get("rating100"),
                "scenes": p.get("scene_count", 0), "created": p.get("created_at") or "",
                "lib": d.get("lib", 0), "moments": s.get("moments"), "taste": taste, "record": rec,
                "img": bool(p.get("has_image")),
                "clip": ((s.get("top") or [{}])[0]).get("stream"),      # hover preview
                "tiers": d.get("tiers") or {},
                "last_seen_days": int((now_ts - last) / 86400) if last else None,
                "photo": self.performer_photo_version(pid),
                # the page's sections
                "new_to_you": bool(d.get("recent")) and not d.get("graded"),
                "stale": good and (not last or now_ts - last > 60 * 86400),
            })
        missing = sum(1 for e in out if not e["photo"] and e["lib"])
        return {"performers": out, "stats_ready": stats is not None, "photos_missing": missing,
                "base_grade": round(base, 2)}

    def performer_profile(self, pid: str) -> dict:
        """Her page's library facts: track record in words, tier breakdown, the
        studios she works with, and her scenes (best first)."""
        from ..tier_model import who_of
        from ..words import share, verdict

        pid = str(pid)
        rows = [r for r in self._catalogue_all() if pid in who_of(r)["performers"]]
        enc = self._who_records(self._catalogue_all())
        t = enc.perf.get(pid)
        rec = None
        if t:
            n, g, _, _ = enc.record(t)
            keep, top = t[2] / t[0], t[3] / t[0]
            word, tone = verdict(n, g, keep, enc.base[0])
            rec = {"n": int(n), "verdict": word, "tone": tone, "keep": round(keep, 3), "top": round(top, 3),
                   "keep_words": share(keep), "top_words": share(top)}
        tiers: dict[str, int] = {}
        studios: dict[str, int] = {}
        for r in rows:
            tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
            if r.get("studio"):
                studios[r["studio"]] = studios.get(r["studio"], 0) + 1
        scenes = sorted(rows, key=lambda r: r.get("date") or "", reverse=True)       # newest first…
        scenes.sort(key=lambda r: -TIER_RANK.get(r["tier"], 1))                        # …within best tier first
        info = self._perf_info(pid) or {}
        return {
            "id": pid, "name": info.get("name") or "", "aliases": info.get("aliases") or [],
            "fav": info.get("favorite", False), "record": rec, "tiers": tiers,
            "studios": sorted(({"name": k, "n": v} for k, v in studios.items()), key=lambda x: -x["n"])[:10],
            "scenes": [{"scene_id": r["scene_id"], "title": r["title"], "tier": r["tier"],
                        "date": r.get("date") or "", "studio": r.get("studio") or "",
                        "stream": self.stream_url(r["scene_id"], start=0)} for r in scenes[:600]],
            "photo": self.performer_photo_version(pid),
        }

    # --- pictures ---------------------------------------------------------------

    def _photo_dir(self) -> Path:
        return Path(self.cfg.modeling.dir) / "perf_photos"

    def _photo_path(self, pid: str) -> Path:
        safe = "".join(c for c in str(pid) if c.isalnum() or c in "-_")
        return self._photo_dir() / f"{safe}.jpg"

    def _photo_choices_path(self) -> Path:
        return Path(self.cfg.modeling.dir) / "performer_photos.json"

    def _photo_choices(self) -> dict:
        try:
            return json.loads(self._photo_choices_path().read_text())
        except (OSError, ValueError):
            return {}

    def performer_photo_version(self, pid: str) -> int:
        """Changes whenever her cached picture does (0 = not cached yet)."""
        try:
            return int(self._photo_path(pid).stat().st_mtime)
        except OSError:
            return 0

    def performer_photo(self, pid: str) -> bytes | None:
        """Her cached picture, built once on a miss (None: nothing to show)."""
        path = self._photo_path(pid)
        try:
            return path.read_bytes()
        except OSError:
            pass
        return path.read_bytes() if self._build_performer_photo(pid) else None

    def _her_scenes(self, pid: str) -> list[dict]:
        from ..tier_model import who_of

        try:
            rows = self._catalogue_all()
        except Exception:  # noqa: BLE001
            return []
        mine = [r for r in rows if str(pid) in who_of(r)["performers"]]
        return sorted(mine, key=lambda r: -TIER_RANK.get(r["tier"], 1))

    def _her_frames(self, pid: str, n: int = 8) -> list[dict]:
        stats = self.__dict__.get("_perf_stats_cache") or []
        s = next((x for x in stats if x.get("id") == str(pid)), None)
        return [t for t in (s or {}).get("top", [])[:n] if t.get("key")]

    # --- her face, alone ------------------------------------------------------------------

    def _face_portraits(self, pid: str, n: int = 6) -> list[dict]:
        """Frames where the Faces scan found exactly one face and it's hers, best
        first, one per scene: [{scene_id, t, box, sim, title}]."""
        import numpy as np

        from .faces import MATCH

        pid = str(pid)
        try:
            gal = self._face_gallery()
        except Exception:  # noqa: BLE001 — no faces yet
            return []
        M, owners = gal["M"], gal["owners"]
        mine = M[owners == pid] if len(M) else M
        if not len(mine):
            return []
        out = []
        for r in [x for x in self._her_scenes(pid) if x["tier"] != "rejected"][:FACE_SCENES]:
            sf = self._scene_faces(r["scene_id"])
            if sf is None or not len(sf[1]):
                continue
            t, v, meta = sf
            boxes, scores = meta.get("boxes") or [], meta.get("scores") or []
            t = np.round(np.asarray(t, np.float32), 2)
            solo = [str(p) for p in r.get("performer_ids") or []] == [pid]
            best = None
            for tt in np.unique(t):
                at = np.where(t == tt)[0]
                if len(at) != 1:                       # two people (or more) in the frame: never
                    continue
                i = int(at[0])
                if i >= len(boxes):
                    continue
                b = boxes[i]
                size = min(b[2] - b[0], b[3] - b[1])
                sc = float(scores[i]) if i < len(scores) else 0.0
                if sc < FACE_MIN_SCORE or size < FACE_MIN_PX:
                    continue
                sim = float((mine @ v[i]).max())
                if sim < MATCH:                        # one face, but not hers
                    continue
                rank = sim * min(size, 320) / 320 * sc * (1.15 if solo else 1.0) * (1 + TIER_RANK.get(r["tier"], 1) / 30)
                if best is None or rank > best["rank"]:
                    best = {"scene_id": r["scene_id"], "t": float(tt), "box": list(b), "sim": round(sim, 3),
                            "title": r.get("title") or f"scene {r['scene_id']}", "rank": rank}
            if best:
                out.append(best)
        out.sort(key=lambda x: -x["rank"])
        return out[:n]

    def _face_crop_jpeg(self, scene_id: str, t: float, box=None) -> bytes | None:
        """Her portrait from one frame: grabbed exactly as the scan did (so the box
        lines up) and cropped 2:3 around the face."""
        sid = str(scene_id)
        if box is None:                                 # a saved choice: find the face again
            sf = self._scene_faces(sid)
            if sf is None:
                return None
            ts, _, meta = sf
            boxes = meta.get("boxes") or []
            near = [i for i, x in enumerate(ts) if abs(float(x) - float(t)) < 0.02 and i < len(boxes)]
            if len(near) != 1:
                return None
            box = boxes[near[0]]
        row = next((r for r in self._catalogue_all() if r["scene_id"] == sid), None)
        if row is None or not row.get("path"):
            return None
        img = self._face_frame(row["path"], float(t)).convert("RGB")
        img = img.crop(portrait_box(img.width, img.height, box))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()

    def face_crop(self, pid: str, scene_id: str, t: float) -> bytes | None:
        """The preview of one "her face" option (only a frame that qualifies for her)."""
        hit = next((p for p in self._face_portraits(pid, 50)
                    if p["scene_id"] == str(scene_id) and abs(p["t"] - float(t)) < 0.02), None)
        return self._face_crop_jpeg(scene_id, t, hit["box"]) if hit else None

    # --- where each automatic picture came from ---------------------------------------------

    def _photo_src_path(self) -> Path:
        return self._photo_dir() / "sources.json"

    def _photo_srcs(self) -> dict:
        try:
            return json.loads(self._photo_src_path().read_text())
        except (OSError, ValueError):
            return {}

    def _note_photo_src(self, pid: str, kind: str | None) -> None:
        p = self._photo_src_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with atomic.edit_lock(p):
                d = self._photo_srcs()
                if kind is None:
                    d.pop(str(pid), None)
                else:
                    d[str(pid)] = kind
                atomic.write_text(p, json.dumps(d))
        except OSError:
            pass

    def upgrade_performer_photos(self, pids=None, job=None) -> int:
        """Automatic pictures from a cover or a frame → her face, alone, once the
        Faces scan has found her. Your chosen pictures and Stash photos stay."""
        chosen, srcs = self._photo_choices(), self._photo_srcs()
        if pids is None:
            pids = [f.stem for f in self._photo_dir().glob("*.jpg")]
        done = 0
        for pid in pids:
            pid = str(pid)
            if job is not None and getattr(job, "cancelled", False):
                break
            if pid in chosen or not self.performer_photo_version(pid):
                continue
            if srcs.get(pid) not in UPGRADABLE or (self._perf_info(pid) or {}).get("has_image"):
                continue
            if not self._face_portraits(pid, 1):
                continue
            if self._build_performer_photo(pid, only_face=True):    # a failed crop leaves what she had
                done += 1
        return done

    def _photo_bytes(self, choice: dict) -> bytes | None:
        """Raw image bytes for one picture source."""
        kind = choice.get("kind")
        try:
            if kind == "face":
                return self._face_crop_jpeg(choice["scene_id"], float(choice["t"]), choice.get("box"))
            if kind == "stash":
                got = self._meta_client().performer_image(choice["pid"])
                return got[0] if got else None
            if kind == "cover":
                got = self.scene_cover(choice["scene_id"])
                return got[0] if got else None
            if kind == "frame":
                path = self.path_for_key(choice["key"])
                return self.frame_jpeg(path, float(choice["t"]), size=720) if path else None
        except Exception:  # noqa: BLE001 — a missing source just falls through
            return None
        return None

    def _photo_sources(self, pid: str):
        pid = str(pid)
        chosen = self._photo_choices().get(pid)
        if chosen:
            yield {**chosen, "pid": pid}
        info = self._perf_info(pid) or {}
        if info.get("has_image"):
            yield {"kind": "stash", "pid": pid}
        for f in self._face_portraits(pid, 3):        # her face, alone (a few, in case a seek fails)
            yield {"kind": "face", "scene_id": f["scene_id"], "t": f["t"], "box": f["box"]}
        scenes = self._her_scenes(pid)
        solo = [r for r in scenes if (r.get("performer_ids") or []) == [pid] and r["tier"] != "rejected"]
        if solo:
            yield {"kind": "cover", "scene_id": solo[0]["scene_id"]}
        for f in self._her_frames(pid, 1):
            yield {"kind": "frame", "key": f["key"], "t": f["t"]}
        # last resort: any scene she's in (a rejected one still shows her)
        others = [r for r in scenes if r not in solo[:1]]
        if others:
            yield {"kind": "cover", "scene_id": others[0]["scene_id"]}

    def _build_performer_photo(self, pid: str, only_face: bool = False) -> bool:
        from PIL import Image

        chosen = self._photo_choices().get(str(pid))
        for src in self._photo_sources(pid):
            if only_face and src.get("kind") != "face":
                continue
            data = self._photo_bytes(src)
            if not data:
                continue
            try:
                img = Image.open(BytesIO(data))
                img = img.convert("RGB")
                img.thumbnail(self._PHOTO_BOX)
                out = self._photo_path(pid)
                from .. import atomic

                with atomic.temp_for(out) as tmp:
                    img.save(tmp, format="JPEG", quality=85)
                self._note_photo_src(pid, "chosen" if chosen and src.get("kind") == chosen.get("kind")
                                     and src.get("scene_id") == chosen.get("scene_id") else src.get("kind"))
                return True
            except Exception:  # noqa: BLE001 — unreadable image: try the next source
                continue
        return False

    def performer_photo_options(self, pid: str) -> dict:
        """What 'Choose picture' offers: her Stash image, covers of her scenes
        (best first) and her best frames — each with a preview URL."""
        from urllib.parse import quote

        pid = str(pid)
        chosen = self._photo_choices().get(pid)
        opts = []
        info = self._perf_info(pid) or {}
        if info.get("has_image"):
            opts.append({"kind": "stash", "label": "Stash profile picture",
                         "thumb": f"/api/performer/{quote(pid)}/image"})
        for f in self._face_portraits(pid, 6):
            opts.append({"kind": "face", "scene_id": f["scene_id"], "t": f["t"], "label": f["title"],
                         "thumb": f"/api/performer/{quote(pid)}/face-crop?scene_id={quote(f['scene_id'])}&t={f['t']:g}"})
        for r in self._her_scenes(pid)[:12]:
            if r["tier"] == "rejected":
                continue
            opts.append({"kind": "cover", "scene_id": r["scene_id"], "label": r["title"],
                         "solo": (r.get("performer_ids") or []) == [pid],
                         "thumb": f"/api/scene/{quote(r['scene_id'])}/cover"})
        for f in self._her_frames(pid):
            opts.append({"kind": "frame", "key": f["key"], "t": f["t"], "label": "a moment",
                         "thumb": f.get("thumb") or f"/api/frame?key={quote(f['key'])}&t={f['t']}"})
        for o in opts:
            o["chosen"] = bool(chosen) and all(chosen.get(k) == o.get(k) for k in ("kind", "scene_id", "key")) and (
                o["kind"] != "face" or abs(float(chosen.get("t") or 0) - float(o["t"])) < 0.02)
        return {"options": opts, "chosen": chosen}

    def choose_performer_photo(self, pid: str, choice: dict | None) -> dict:
        """Use this picture for her (or `None`: back to automatic). Rebuilds the
        cached file at once; returns the new version."""
        pid = str(pid)
        choices = self._photo_choices()
        if choice:
            kind = choice.get("kind")
            if kind not in ("stash", "cover", "frame", "face"):
                raise ValueError("unknown picture kind")
            keep = {"kind": kind}
            if kind in ("cover", "face"):
                keep["scene_id"] = str(choice["scene_id"])
            if kind == "face":
                keep["t"] = float(choice["t"])
            if kind == "frame":
                keep.update(key=str(choice["key"]), t=float(choice["t"]))
            choices[pid] = keep
        else:
            choices.pop(pid, None)
        p = self._photo_choices_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic.write_text(p, json.dumps(choices, indent=1))
        self._photo_path(pid).unlink(missing_ok=True)
        self._build_performer_photo(pid)
        return {"photo": self.performer_photo_version(pid)}

    def warm_performer_photos(self, job=None) -> dict:
        """Background job: cache a picture for every performer in your library
        who doesn't have one yet, a few at a time."""
        from concurrent.futures import ThreadPoolExecutor

        from ..tier_model import who_of

        try:
            ids = {pid for r in self._catalogue_all() for pid in who_of(r)["performers"]}
        except Exception:  # noqa: BLE001
            ids = set()
        try:
            self.upgrade_performer_photos(sorted(ids), job=job)      # her face, alone, where the scan found her
        except Exception:  # noqa: BLE001 — pictures are extra
            pass
        todo = sorted(pid for pid in ids if not self.performer_photo_version(pid))
        done = made = 0
        if job is not None:
            job.progress = {"done": 0, "total": len(todo)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for ok in pool.map(self._build_performer_photo, todo):
                done += 1
                made += bool(ok)
                if job is not None:
                    job.progress = {"done": done, "total": len(todo)}
                    if job.cancelled:
                        break
        return {"cached": made, "of": len(todo)}
