"""The Performers directory: every performer in Stash, searchable, with a
proper picture — cached on disk so the page opens instantly.

Pictures (first that exists):
  1. the one you chose on her page (models/performer_photos.json)
  2. her Stash profile image (usually a portrait / full-body shot — skipped
     when Stash would serve its default silhouette)
  3. the cover of her best-graded solo scene (only her in it)
  4. her best frame (the taste model's pick)
  5. the cover of any scene she's in (even a rejected one — it still shows her)
Each is resized once and kept in models/perf_photos/{id}.jpg; the browser
caches it too (the URL carries a version that changes with the picture).

The directory itself (Stash's performer list) is kept in memory and on disk:
opening Performers serves the last copy at once while a background refresh
reads Stash again.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from io import BytesIO
from pathlib import Path

TIER_RANK = {"legendaire": 6, "exceptionnelle": 5, "merveilleuse": 4, "upscale": 3,
             "anomaly": 3, "unreviewed": 2, "rejected": 0}


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
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump({"ts": time.time(), "performers": perfs}, fh)
            tmp.replace(p)
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
                        "date": r.get("date") or "", "studio": r.get("studio") or ""} for r in scenes[:200]],
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

    def _photo_bytes(self, choice: dict) -> bytes | None:
        """Raw image bytes for one picture source."""
        kind = choice.get("kind")
        try:
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

    def _build_performer_photo(self, pid: str) -> bool:
        from PIL import Image

        for src in self._photo_sources(pid):
            data = self._photo_bytes(src)
            if not data:
                continue
            try:
                img = Image.open(BytesIO(data))
                img = img.convert("RGB")
                img.thumbnail(self._PHOTO_BOX)
                out = self._photo_path(pid)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(out.name + ".tmp")
                img.save(tmp, format="JPEG", quality=85)
                tmp.replace(out)
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
            o["chosen"] = bool(chosen) and all(chosen.get(k) == o.get(k) for k in ("kind", "scene_id", "key"))
        return {"options": opts, "chosen": chosen}

    def choose_performer_photo(self, pid: str, choice: dict | None) -> dict:
        """Use this picture for her (or `None`: back to automatic). Rebuilds the
        cached file at once; returns the new version."""
        pid = str(pid)
        choices = self._photo_choices()
        if choice:
            kind = choice.get("kind")
            if kind not in ("stash", "cover", "frame"):
                raise ValueError("unknown picture kind")
            keep = {"kind": kind}
            if kind == "cover":
                keep["scene_id"] = str(choice["scene_id"])
            if kind == "frame":
                keep.update(key=str(choice["key"]), t=float(choice["t"]))
            choices[pid] = keep
        else:
            choices.pop(pid, None)
        p = self._photo_choices_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(choices, indent=1))
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
