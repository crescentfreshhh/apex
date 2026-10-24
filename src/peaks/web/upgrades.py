"""Upgrades: keepers that deserve a better copy, found through Prowlarr.

Wanted = Merveilleuse / Exceptionnelle / Légendaire scenes whose picture is
Poor or Watchable, that are upscales, or that are below 4K — minus VR and the
studios you excluded (some studios' "4K" is barely bigger than their 1080p
and looks worse). A daily pass searches a few of them (spaced out, so your
indexers aren't hammered) and keeps only releases that are this scene, are
healthy torrents and beat your copy. Nothing is downloaded until you press
Grab — then Prowlarr sends that one release to its download client.

The rest is what Peaks already does: the seedbox pull brings the file in,
ingest adds it, the duplicate check pairs it with your old copy and the
picture-aware recommendation (⧉ Compare copies) settles which to keep — the
Wanted entry then reads "arrived".

State: models/upgrades.json {scenes: {sid: {...}}, last_pass}.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .. import prowlarr as pw

KEEPERS = ("merveilleuse", "exceptionnelle", "legendaire")
TIER_FIRST = {"legendaire": 0, "exceptionnelle": 1, "merveilleuse": 2}
DEFAULTS = {"url": "", "api_key": "", "min_seeders": 5, "excluded_studios": [], "auto": False,
            "per_day": 40, "research_days": 14, "categories": []}
STALL_DAYS = 14
SEARCH_GAP = 3.0              # seconds between searches in the daily pass


class UpgradesMixin:
    # --- settings ------------------------------------------------------------------

    def upgrade_settings(self, reveal: bool = False) -> dict:
        s = {**DEFAULTS, **(self._settings().get("upgrades") or {})}
        s["url"] = os.environ.get("PEAKS_PROWLARR_URL") or s["url"]
        s["api_key"] = os.environ.get("PEAKS_PROWLARR_KEY") or s["api_key"]
        if not reveal:                          # the key never goes back to a browser
            k = s["api_key"]
            s["api_key"] = ("•" * 8 + k[-4:]) if k else ""
            s["has_key"] = bool(k)
        return s

    def save_upgrade_settings(self, body: dict) -> dict:
        from .. import atomic

        settings = dict(self._settings())
        cur = {**DEFAULTS, **(settings.get("upgrades") or {})}
        for k in ("url", "auto", "min_seeders", "per_day", "research_days", "excluded_studios", "categories"):
            if k in body and body[k] is not None:
                cur[k] = body[k]
        key = body.get("api_key")
        if isinstance(key, str) and key and not key.startswith("•"):      # a masked echo isn't a new key
            cur["api_key"] = key.strip()
        cur["url"] = str(cur["url"] or "").strip().rstrip("/")
        cur["min_seeders"] = max(1, int(cur["min_seeders"]))
        cur["per_day"] = max(1, min(500, int(cur["per_day"])))
        cur["research_days"] = max(1, int(cur["research_days"]))
        cur["auto"] = bool(cur["auto"])
        cur["excluded_studios"] = sorted({str(x).strip() for x in cur["excluded_studios"] or [] if str(x).strip()},
                                         key=str.lower)
        cur["categories"] = [int(c) for c in cur["categories"] or []]
        settings["upgrades"] = cur
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic.write_json(path, settings, indent=2, trailing_newline=True)
        self._settings_cache = settings
        return self.upgrade_settings()

    def exclude_upgrade_studio(self, studio: str) -> dict:
        s = self.upgrade_settings(reveal=True)
        return self.save_upgrade_settings({"excluded_studios": [*s["excluded_studios"], studio]})

    def _prowlarr(self) -> pw.ProwlarrClient:
        s = self.upgrade_settings(reveal=True)
        return pw.ProwlarrClient(s["url"], s["api_key"])

    def upgrade_test(self) -> dict:
        """Can Peaks reach Prowlarr, and which indexers does it have?"""
        try:
            c = self._prowlarr()
            health = c.health()
            idx = c.indexers()
        except pw.ProwlarrError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "indexers": [{"name": i.get("name"), "enabled": bool(i.get("enable", True))}
                                          for i in idx],
                "warnings": [h.get("message") for h in health if h.get("message")]}

    # --- state ---------------------------------------------------------------------

    def _upg_path(self) -> Path:
        return Path(self.cfg.modeling.dir) / "upgrades.json"

    def _upg_lock(self):
        return self.__dict__.setdefault("_upg_lock_obj", threading.RLock())

    def _upg_load(self) -> dict:
        from .. import atomic

        try:
            d = atomic.read_json(self._upg_path(), default=None)
        except atomic.DamagedFile:
            d = None
        d = d if isinstance(d, dict) else {}
        d.setdefault("scenes", {})
        return d

    def _upg_save(self, d: dict) -> None:
        from .. import atomic

        atomic.write_json(self._upg_path(), d, indent=1)

    # --- what's wanted -------------------------------------------------------------

    def _upg_reasons(self, row: dict, pic: dict | None) -> list[str]:
        out = []
        if pic and not pic.get("vr"):
            if pic.get("band") in ("Poor", "Watchable"):
                out.append(f"picture {pic['band']} {pic['score']}")
            if pic.get("upscaled"):
                real = pic.get("real")
                out.append(f"upscale — real detail of ~{int(real)}p" if real else "upscale")
        res = (row.get("quality") or {}).get("res")
        if res and pw.RES_HEIGHT.get(res, 0) < 2160:
            out.append(f"{res}")
        return out

    def upgrade_wanted(self) -> list[dict]:
        """Keepers that deserve a better copy, Légendaire first, then the ones
        you like most, then the worst pictures."""
        s = self.upgrade_settings(reveal=True)
        excluded = {x.lower() for x in s["excluded_studios"]}
        try:
            pics = self.picture_scores()
        except Exception:  # noqa: BLE001 — picture quality is extra
            pics = {}
        try:
            sig = self._scene_signals(self._catalogue_all())
        except Exception:  # noqa: BLE001
            sig = {}
        out = []
        for r in self._catalogue_all():
            if r["tier"] not in KEEPERS or (r.get("studio") or "").lower() in excluded or self._is_vr(r):
                continue
            pic = pics.get(r["scene_id"])
            reasons = self._upg_reasons(r, pic)
            if not reasons:
                continue
            best = (sig.get(r["scene_id"]) or {}).get("best")
            out.append({"row": r, "reasons": reasons, "pic": pic, "best": best})
        out.sort(key=lambda w: (TIER_FIRST[w["row"]["tier"]], -(w["best"] or 0),
                                (w["pic"] or {}).get("score") or 100))
        return out

    # --- searching -----------------------------------------------------------------

    def _upg_poor(self, pic: dict | None) -> bool:
        return bool(pic and (pic.get("upscaled") or pic.get("band") in ("Poor", "Watchable")))

    def upgrade_search(self, scene_id: str, client: pw.ProwlarrClient | None = None) -> dict:
        """Search Prowlarr for this scene; keep the healthy releases that are
        this scene and beat your copy, best first."""
        sid = str(scene_id)
        row = next((r for r in self._catalogue_all() if r["scene_id"] == sid), None)
        if row is None:
            raise LookupError(f"scene {sid} isn't in the library")
        s = self.upgrade_settings(reveal=True)
        client = client or self._prowlarr()
        try:
            pic = self.picture_scores().get(sid)
        except Exception:  # noqa: BLE001
            pic = None
        poor = self._upg_poor(pic)
        q = row.get("quality") or {}
        found, seen = [], set()
        for query in pw.queries_for(row):
            for rel in client.search(query, s["categories"] or None):
                key = (rel.get("guid"), rel.get("indexerId"))
                if key in seen or not rel.get("guid"):
                    continue
                seen.add(key)
                info = pw.parse_release(rel.get("title") or "")
                if info["vr"] or not pw.matches_scene(rel.get("title") or "", row):
                    continue
                if not pw.is_healthy(rel, s["min_seeders"]):
                    continue
                if not pw.is_upgrade(info["height"], rel.get("size"), q.get("res"), row.get("size"), poor):
                    continue
                found.append({"guid": rel["guid"], "indexer_id": rel.get("indexerId"),
                              "indexer": rel.get("indexer"), "title": rel.get("title"),
                              "size": rel.get("size"), "seeders": rel.get("seeders"),
                              "leechers": rel.get("leechers"), "height": info["height"],
                              "age": rel.get("age"), "info_url": rel.get("infoUrl")})
        found.sort(key=lambda c: (-(c["height"] or 0), -(c["seeders"] or 0), -(c["size"] or 0)))
        with self._upg_lock():
            d = self._upg_load()
            e = d["scenes"].get(sid, {})
            if e.get("state") in ("grabbed", "arrived", "dismissed"):
                e.update(searched_at=time.time(), found=found)     # keep the decision
            else:
                e.update(searched_at=time.time(), found=found, state="found" if found else "none")
            e.pop("error", None)
            d["scenes"][sid] = e
            self._upg_save(d)
        return {"scene_id": sid, "found": found}

    def upgrade_due(self) -> bool:
        s = self.upgrade_settings(reveal=True)
        if not (s["auto"] and s["url"] and s["api_key"]):
            return False
        return time.time() - (self._upg_load().get("last_pass") or 0) >= 20 * 3600

    def run_upgrade_pass(self, job=None, client: pw.ProwlarrClient | None = None,
                         sleep=time.sleep) -> dict:
        """The daily search: the top Wanted scenes not searched lately, a few
        seconds apart. Stops on cancel; a Prowlarr error ends the pass."""
        s = self.upgrade_settings(reveal=True)
        log = job.log if job is not None else (lambda *_: None)
        client = client or self._prowlarr()
        d = self._upg_load()
        cutoff = time.time() - s["research_days"] * 86400
        skip = {sid for sid, e in d["scenes"].items()
                if (e.get("searched_at") or 0) > cutoff or e.get("state") in ("grabbed", "dismissed")}
        todo = [w["row"] for w in self.upgrade_wanted() if w["row"]["scene_id"] not in skip][: s["per_day"]]
        searched = hits = 0
        if job is not None:
            job.progress = {"total": len(todo), "done": 0}
        for i, row in enumerate(todo):
            if job is not None and job.cancelled:
                break
            if i:
                sleep(SEARCH_GAP)
            try:
                r = self.upgrade_search(row["scene_id"], client=client)
            except pw.ProwlarrError as exc:
                log(f"Prowlarr: {exc} — stopping this pass")
                break
            searched += 1
            hits += bool(r["found"])
            if r["found"]:
                log(f"{row.get('title') or row['scene_id']}: {len(r['found'])} upgrade(s)")
            if job is not None:
                job.progress = {"total": len(todo), "done": i + 1}
        with self._upg_lock():
            d = self._upg_load()
            d["last_pass"] = time.time()
            self._upg_save(d)
        return {"searched": searched, "found": hits}

    # --- grabbing, and following the grab -------------------------------------------

    def upgrade_grab(self, scene_id: str, guid: str, client: pw.ProwlarrClient | None = None) -> dict:
        """Ask Prowlarr to send this release (one Peaks found for this scene) to
        its download client."""
        sid = str(scene_id)
        with self._upg_lock():
            e = self._upg_load()["scenes"].get(sid) or {}
        cand = next((c for c in e.get("found") or [] if c["guid"] == guid), None)
        if cand is None:
            raise LookupError("that release isn't one Peaks found for this scene — search again")
        (client or self._prowlarr()).grab(cand["guid"], cand["indexer_id"])
        row = next((r for r in self._catalogue_all() if r["scene_id"] == sid), None)
        with self._upg_lock():
            d = self._upg_load()
            e = d["scenes"].setdefault(sid, {})
            e.update(state="grabbed", grabbed={**cand, "at": time.time()},
                     have={"size": (row or {}).get("size"), "res": ((row or {}).get("quality") or {}).get("res")})
            self._upg_save(d)
        self._log_scene("upgrade-grab", row, detail=f"asked Prowlarr for {cand['title']} "
                        f"({cand.get('height') or '?'}p, {cand.get('seeders')} seeders, {cand.get('indexer')})")
        return e

    def upgrade_dismiss(self, scene_id: str) -> dict:
        with self._upg_lock():
            d = self._upg_load()
            e = d["scenes"].setdefault(str(scene_id), {})
            e["state"] = "dismissed"
            self._upg_save(d)
        return e

    def upgrade_retry(self, scene_id: str) -> dict:
        """A stalled grab: back to the found releases (minus the one that stalled)."""
        with self._upg_lock():
            d = self._upg_load()
            e = d["scenes"].setdefault(str(scene_id), {})
            bad = (e.get("grabbed") or {}).get("guid")
            e["found"] = [c for c in e.get("found") or [] if c["guid"] != bad]
            e["state"] = "found" if e["found"] else "none"
            e.pop("grabbed", None)
            self._upg_save(d)
        return e

    def _upg_follow(self, d: dict) -> bool:
        """Grabbed → arrived when the scene shows up in a duplicate group next to
        a bigger copy; → stalled after STALL_DAYS. Returns whether anything changed."""
        groups = (self.cached_duplicates() or {}).get("groups") or []
        changed = False
        now = time.time()
        for sid, e in d["scenes"].items():
            if e.get("state") != "grabbed":
                continue
            have = (e.get("have") or {}).get("size") or 0
            g = next((g for g in groups if sid in {r["scene_id"] for r in g["scenes"]}), None)
            if g and any(r["scene_id"] != sid and int(r.get("size") or 0) > have for r in g["scenes"]):
                e.update(state="arrived", arrived_at=now)
                changed = True
            elif now - (e.get("grabbed") or {}).get("at", now) > STALL_DAYS * 86400:
                e["state"] = "stalled"
                changed = True
        return changed

    def upgrades(self) -> dict:
        """Everything the Upgrades page shows."""
        with self._upg_lock():
            d = self._upg_load()
            if self._upg_follow(d):
                self._upg_save(d)
        wanted = self.upgrade_wanted()
        rows = {r["scene_id"]: r for r in self._catalogue_all()}
        try:
            pics = self.picture_scores()
        except Exception:  # noqa: BLE001
            pics = {}

        def card(sid, e, reasons=None):
            r = rows.get(sid)
            if r is None:
                return None
            return {"scene_id": sid, "title": r.get("title"), "studio": r.get("studio"), "date": r.get("date"),
                    "tier": r["tier"], "rating100": r.get("rating100"), "o_counter": r.get("o_counter"),
                    "quality": r.get("quality"), "size": r.get("size"), "picture": pics.get(sid),
                    "reasons": reasons if reasons is not None else self._upg_reasons(r, pics.get(sid)),
                    "found": e.get("found") or [], "state": e.get("state") or "wanted",
                    "searched_at": e.get("searched_at"), "grabbed": e.get("grabbed"),
                    "arrived_at": e.get("arrived_at")}

        out = {"found": [], "wanted": [], "grabbed": [], "arrived": []}
        listed = set()
        for w in wanted:
            sid = w["row"]["scene_id"]
            e = d["scenes"].get(sid, {})
            st = e.get("state")
            if st == "dismissed":
                continue
            c = card(sid, e, w["reasons"])
            listed.add(sid)
            if st == "found":
                out["found"].append(c)
            elif st in ("grabbed", "stalled"):
                out["grabbed"].append(c)
            elif st == "arrived":
                out["arrived"].append(c)
            else:
                out["wanted"].append(c)
        for sid, e in d["scenes"].items():         # grabbed / arrived scenes that left the Wanted list
            if sid in listed or e.get("state") not in ("grabbed", "stalled", "arrived"):
                continue
            c = card(sid, e)
            if c:
                out["arrived" if e["state"] == "arrived" else "grabbed"].append(c)
        s = self.upgrade_settings()
        return {**out, "settings": s, "configured": bool(s["url"] and s.get("has_key")),
                "last_pass": d.get("last_pass")}

    def upgrade_counts(self) -> dict:
        d = self._upg_load()
        n = sum(1 for e in d["scenes"].values() if e.get("state") == "found")
        return {"found": n}
