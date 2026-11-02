"""Discover: scenes you'd probably like but don't have.

Your favourites come from your grades — performers and studios whose scenes you
keep (Légendaire counts most, rejects count against). For each one linked to
StashDB (the stash-box set up in your Stash), Peaks lists their scenes there,
drops the ones you already own (matched by StashDB id), and ranks the rest by
how well their performers and studio fit your grades.

Nothing downloads on its own: "Find" searches Prowlarr for the scene (the same
matching and health rules as Upgrades) and "Grab" adds the release you pick to
your seedbox qBittorrent — the only writes being add_torrent, on your click.
"""

from __future__ import annotations

import time
from pathlib import Path

from .. import atomic
from .. import prowlarr as pw
from ..atomic import settings_saver as _settings_saver

TIER_W = {"legendaire": 3.0, "exceptionnelle": 2.0, "merveilleuse": 1.0, "rejected": -2.0}
MIN_KEEPS = 2              # keepers of theirs before a performer / studio counts as a favourite
TOP_PERFORMERS = 25
TOP_STUDIOS = 10
PER_SOURCE = 40            # newest scenes looked at per favourite
STUDIO_SHARE = 0.4         # a studio's weight in a scene's fit, next to its performers'


class DiscoverMixin:
    # --- settings / state ----------------------------------------------------------------

    def discover_settings(self) -> dict:
        d = self._settings().get("discover") or {}
        return {"performers": int(d.get("performers") or TOP_PERFORMERS),
                "studios": int(d.get("studios") or TOP_STUDIOS),
                "per_source": int(d.get("per_source") or PER_SOURCE)}

    @_settings_saver
    def save_discover_settings(self, body: dict) -> dict:
        from .inputs import as_int

        s = dict(self._settings())
        d = dict(s.get("discover") or {})
        for k, lo, hi in (("performers", 0, 200), ("studios", 0, 100), ("per_source", 5, 200)):
            if body.get(k) is not None:
                d[k] = as_int(body[k], k, lo, hi)
        s["discover"] = d
        self._save_settings_dict(s)
        return self.discover_settings()

    def _disc_path(self) -> Path:
        return self._state_dir() / "discover.json"

    def _disc_load(self) -> dict:
        d = atomic.read_json(self._disc_path(), default={}) or {}
        d.setdefault("items", {})
        d.setdefault("decided", {})          # stash-box id → {state: dismissed|grabbed, at, …}
        return d

    def _disc_edit(self, fn):
        with atomic.edit_lock(self._disc_path()):
            d = self._disc_load()
            out = fn(d)
            self._disc_path().parent.mkdir(parents=True, exist_ok=True)
            atomic.write_json(self._disc_path(), d, indent=1)
            return out

    # --- your favourites -----------------------------------------------------------------

    def discover_favourites(self) -> dict:
        """{performers: [{id, name, score, keeps}], studios: [{name, score, keeps}]} from your grades."""
        from ..tier_model import who_of

        perf: dict[str, list] = {}
        studio: dict[str, list] = {}
        for r in self._catalogue_all():
            w = TIER_W.get(r["tier"])
            if w is None:
                continue
            for pid in who_of(r)["performers"]:
                perf.setdefault(pid, []).append(w)
            if r.get("studio"):
                studio.setdefault(r["studio"], []).append(w)
        names = {p["id"]: p["name"] for p in self._stash_performers()}
        excluded = self._excluded_performers()

        def rank(d, top):
            out = []
            for k, ws in d.items():
                keeps = sum(1 for w in ws if w > 0)
                if keeps < MIN_KEEPS:
                    continue
                out.append((sum(ws) / (len(ws) + 2), keeps, k))      # shrunk toward 0: a few scenes prove little
            out.sort(reverse=True)
            return [x for x in out if x[0] > 0][:top]

        s = self.discover_settings()
        return {"performers": [{"id": k, "name": names.get(k, k), "score": round(sc, 3), "keeps": n}
                               for sc, n, k in rank(perf, s["performers"] * 2)
                               if names.get(k, "").strip().lower() not in excluded][: s["performers"]],
                "studios": [{"name": k, "score": round(sc, 3), "keeps": n} for sc, n, k in rank(studio, s["studios"])]}

    # --- what you already have, and their StashDB ids ---------------------------------------

    def _disc_local(self, endpoint: str) -> dict:
        """Your Stash's links to the stash-box: performer id → box id, studio name →
        box id, and every box scene id you own."""
        c = self._meta_client()

        def ids(entity: str, extra: str = "") -> list[dict]:
            q = (f"query {{ find{entity}(filter: {{per_page: -1}}) {{ {entity[0].lower() + entity[1:]} "
                 f"{{ id {extra} stash_ids {{ endpoint stash_id }} }} }} }}")
            d = c.execute(q)[f"find{entity}"]
            return d[entity[0].lower() + entity[1:]]

        def box(x):
            return next((s["stash_id"] for s in x.get("stash_ids") or [] if s.get("endpoint") == endpoint), None)
        perfs = {str(p["id"]): box(p) for p in ids("Performers")}
        studios = {s["name"]: box(s) for s in ids("Studios", "name")}
        owned = {box(s) for s in ids("Scenes")} - {None}
        return {"performers": perfs, "studios": studios, "owned": owned}

    # --- the job ------------------------------------------------------------------------------

    def _stashbox(self):
        from ..stashbox import StashBoxClient, pick_box, stash_boxes

        b = pick_box(stash_boxes(self._meta_client()))
        if b is None:
            raise RuntimeError("no stash-box (StashDB) set up in your Stash — Settings → Metadata Providers")
        return StashBoxClient(b["endpoint"], b["api_key"]), b

    def run_discover(self, job=None, box=None) -> dict:
        log = job.log if job else print
        client, b = box if box is not None else self._stashbox()
        log(f"discover: using {b.get('name') or b['endpoint']}")
        fav = self.discover_favourites()
        local = self._disc_local(b["endpoint"])
        s = self.discover_settings()
        pscore = {p["id"]: p["score"] for p in fav["performers"]}
        sscore = {x["name"]: x["score"] for x in fav["studios"]}
        box_perf = {local["performers"].get(p["id"]): p for p in fav["performers"]}
        box_perf.pop(None, None)
        sources = [("performer", local["performers"].get(p["id"]), p["name"]) for p in fav["performers"]]
        sources += [("studio", local["studios"].get(x["name"]), x["name"]) for x in fav["studios"]]
        unlinked = [name for _, sid, name in sources if not sid]
        sources = [x for x in sources if x[1]]
        found: dict[str, dict] = {}
        errors = []
        for i, (kind, sbid, name) in enumerate(sources):
            if job is not None and job.cancelled:
                break
            try:
                scenes = client.scenes(**{kind: sbid}, per_page=s["per_source"])
            except Exception as exc:  # noqa: BLE001 — one favourite failing doesn't stop the rest
                errors.append(f"{name}: {exc}")
                continue
            for sc in scenes:
                if sc["id"] in local["owned"]:
                    continue
                e = found.setdefault(sc["id"], {**sc, "because": []})
                if name not in e["because"]:
                    e["because"].append(name)
            if job is not None:
                job.progress = {"done": i + 1, "total": len(sources)}
        for e in found.values():          # fit: her grades (any favourite in it), plus the studio's
            ps = [box_perf[p]["score"] for p in e.get("performer_ids") or [] if p in box_perf]
            e["fit"] = round(sum(sorted(ps, reverse=True)[:2]) + STUDIO_SHARE * sscore.get(e["studio"], 0.0), 4)
            e["favourites"] = [box_perf[p]["name"] for p in e.get("performer_ids") or [] if p in box_perf]
        now = time.time()

        def save(d):
            d["items"] = found
            d["at"] = now
            d["unlinked"] = unlinked
            d["errors"] = errors[:20]
        self._disc_edit(save)
        log(f"discover: {len(found)} scene(s) you don't have, from {len(sources)} favourite(s)"
            + (f" · {len(unlinked)} favourite(s) not linked to StashDB in your Stash" if unlinked else ""))
        return {"found": len(found), "sources": len(sources), "unlinked": len(unlinked), "errors": len(errors)}

    # --- the page --------------------------------------------------------------------------------

    def discover(self, limit: int = 200) -> dict:
        d = self._disc_load()
        items = []
        for sid, e in d["items"].items():
            dec = d["decided"].get(sid) or {}
            if dec.get("state") == "dismissed":
                continue
            items.append({**e, "state": dec.get("state") or ("found" if e.get("releases") else "new"),
                          "releases": e.get("releases") or [], "searched_at": e.get("searched_at"),
                          "grabbed": dec.get("grabbed")})
        items.sort(key=lambda x: (x["state"] != "found", -x.get("fit", 0), x.get("date") or ""), reverse=False)
        return {"at": d.get("at"), "items": items[:limit], "total": len(items),
                "unlinked": d.get("unlinked") or [], "errors": d.get("errors") or [],
                "favourites": self.discover_favourites() if d.get("at") else None}

    def discover_search(self, box_id: str, client: pw.ProwlarrClient | None = None) -> dict:
        """Prowlarr releases that are this scene, healthy, flat (not VR) — best first."""
        d = self._disc_load()
        e = d["items"].get(box_id)
        if e is None:
            raise LookupError("that scene isn't in the Discover list — refresh it")
        s = self.upgrade_settings(reveal=True)
        client = client or self._prowlarr()
        found, seen = [], set()
        for query in pw.queries_for(e):
            for rel in client.search(query, s["categories"] or None):
                key = (rel.get("guid"), rel.get("indexerId"))
                if key in seen or not rel.get("guid"):
                    continue
                seen.add(key)
                info = pw.parse_release(rel.get("title") or "")
                if info["vr"] or not pw.matches_scene(rel.get("title") or "", e) or not pw.is_healthy(rel, s["min_seeders"]):
                    continue
                found.append({"guid": rel["guid"], "indexer_id": rel.get("indexerId"), "indexer": rel.get("indexer"),
                              "title": rel.get("title"), "size": rel.get("size"), "seeders": rel.get("seeders"),
                              "leechers": rel.get("leechers"), "height": info["height"], "age": rel.get("age"),
                              "info_url": rel.get("infoUrl"), "download_url": rel.get("downloadUrl"),
                              "magnet_url": rel.get("magnetUrl")})
        found.sort(key=lambda c: (-(c["height"] or 0), -(c["seeders"] or 0), -(c["size"] or 0)))

        def save(d):
            if box_id in d["items"]:
                d["items"][box_id].update(releases=found, searched_at=time.time())
        self._disc_edit(save)
        return {"id": box_id, "releases": found}

    def discover_grab(self, box_id: str, guid: str, client: pw.ProwlarrClient | None = None, qbt=None) -> dict:
        """Add the release you picked to your seedbox qBittorrent (the Upgrades category)."""
        e = self._disc_load()["items"].get(box_id) or {}
        cand = next((c for c in e.get("releases") or [] if c["guid"] == guid), None)
        if cand is None:
            raise LookupError("that release isn't one Peaks found for this scene — Find again")
        if qbt is None:
            c = self.cfg.seedbox
            if not (c.qbt_url and c.qbt_user and c.qbt_password):
                raise ValueError("grabs go to your seedbox qBittorrent, which isn't set up "
                                 "(PEAKS_QBT_URL / _USER / _PASSWORD on the container)")
            qbt = self._sj_qbt()
        client = client or self._prowlarr()
        kind, payload = client.fetch_torrent(cand.get("download_url"), cand.get("magnet_url"))
        category = self._upg_category()
        qbt.add_torrent(data=payload if kind == "torrent" else None,
                        magnet=payload if kind == "magnet" else None, category=category)
        grabbed = {**cand, "at": time.time(), "category": category}
        self._disc_edit(lambda d: d["decided"].__setitem__(box_id, {"state": "grabbed", "at": time.time(), "grabbed": grabbed}))
        self._log_scene("discover-grab", {"title": e.get("title")},
                        detail=f"added {cand['title']} to qBittorrent (category {category or 'none'}) — "
                               f"a StashDB scene you didn't have ({', '.join(e.get('because') or [])})")
        return {"state": "grabbed", "category": category}

    def discover_dismiss(self, box_id: str, undo: bool = False) -> dict:
        def f(d):
            if undo:
                d["decided"].pop(box_id, None)
            else:
                d["decided"][box_id] = {"state": "dismissed", "at": time.time()}
        self._disc_edit(f)
        return {"id": box_id, "dismissed": not undo}
