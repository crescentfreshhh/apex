"""Finding things in the library: the Catalogue's search words, the filters
that go beyond tiers (VR, saves, date added, Os, size, organised, several
performers / studios / tags at once), and the suggestions shown while typing
(performers incl. aliases, studios, tags, scenes) — used by the Catalogue
search box and the `/` palette.

Search words: every word must match somewhere in the scene — title, a
performer (or one of her aliases), the studio, a tag, the path or the
details text; accents and case don't matter. `"two words"` matches as a
phrase, `-word` leaves out scenes that have it.
"""

from __future__ import annotations

import re

VR_TAGS = ("vr", "virtual reality", "vr porn", "6dof", "180°", "180 vr", "360 vr")
_TERM = re.compile(r'(-?)"([^"]+)"|(\S+)')
TIER_RANK = {"legendaire": 6, "exceptionnelle": 5, "merveilleuse": 4, "anomaly": 2, "unreviewed": 1, "rejected": 0}


class FindMixin:
    # --- search words -----------------------------------------------------------------

    @classmethod
    def _search_terms(cls, q: str | None) -> list[tuple[bool, str]]:
        """'anna "red dress" -solo' → [(False, 'anna'), (False, 'red dress'), (True, 'solo')]."""
        out = []
        for m in _TERM.finditer(q or ""):
            if m.group(2) is not None:
                neg, text = bool(m.group(1)), m.group(2)
            else:
                tok = m.group(3)
                neg, text = tok.startswith("-") and len(tok) > 1, tok
                if neg:
                    text = tok[1:]
            text = cls._fold(text)
            if text:
                out.append((neg, text))
        return out

    def _perf_aliases(self) -> dict[str, list[str]]:
        """Stash performer id → aliases (from the cached performer directory)."""
        try:
            perfs = self._stash_performers()
        except Exception:  # noqa: BLE001 — Stash down and nothing cached: names only
            return {}
        sig = len(perfs)
        cached = self.__dict__.get("_alias_map")
        if cached and cached[0] == sig and cached[1] is perfs:
            return cached[2]
        amap = {str(p.get("id")): [a for a in (p.get("aliases") or []) if a] for p in perfs}
        self._alias_map = (sig, perfs, amap)
        return amap

    def _row_hay(self, r: dict, aliases: dict[str, list[str]]) -> str:
        """Everything a search word can match in this scene, folded once and reused."""
        cache = self.__dict__.setdefault("_hay_cache", {})
        als = [a for pid in (r.get("performer_ids") or []) for a in aliases.get(pid, ())]
        sig = (r["title"], r["path"], r["studio"], tuple(r["performers"]), tuple(r["tags"]),
               len(r.get("details") or ""), tuple(als))
        hit = cache.get(r["scene_id"])
        if hit and hit[0] == sig:
            return hit[1]
        hay = self._fold(" │ ".join([r["title"], r["studio"], r["path"], r.get("details") or "",
                                          *r["performers"], *als, *r["tags"]]))
        cache[r["scene_id"]] = (sig, hay)
        return hay

    # --- the facts filters use ------------------------------------------------------------

    @staticmethod
    def _is_vr(r: dict) -> bool:
        """VR: a VR tag, a side-by-side (2:1) frame, or a /vr/ folder."""
        for t in r.get("tags") or []:
            t = t.lower()
            if t in VR_TAGS or t.startswith("vr ") or t.endswith(" vr"):
                return True
        if "/vr/" in (r.get("path") or "").lower():
            return True
        q = r.get("quality") or {}
        w, h = q.get("w"), q.get("h")
        return bool(w and h and 1.95 <= w / h <= 2.05)

    def _saved_scene_ids(self) -> set[str]:
        try:
            return {str(m["scene_id"]) for m in self._saved_markers(self.cfg.markers.tag_name)
                    if m.get("scene_id")}
        except Exception:  # noqa: BLE001 — Stash down: no save information this time
            return set()

    # --- suggestions while typing ----------------------------------------------------------

    def catalogue_suggest(self, q: str, limit: int = 6) -> dict:
        """Performers (name or alias), studios, tags and scenes matching what's
        typed so far — starts-with first, then by how many scenes."""
        needle = self._fold(q)
        if not needle:
            return {"performers": [], "studios": [], "tags": [], "scenes": []}
        rows = self._catalogue_all()
        aliases = self._perf_aliases()
        perf: dict[str, list] = {}           # pid → [name, n, alias that matched]
        studios: dict[str, int] = {}
        tags: dict[str, int] = {}
        for r in rows:
            for pid, name in zip(r.get("performer_ids") or [], r["performers"]):
                e = perf.get(pid)
                if e is None:
                    perf[pid] = [name, 1]
                else:
                    e[1] += 1
            if r["studio"]:
                studios[r["studio"]] = studios.get(r["studio"], 0) + 1
            for t in r["tags"]:
                tags[t] = tags.get(t, 0) + 1

        def rank(text: str) -> int | None:
            f = self._fold(text)
            if f.startswith(needle):
                return 0
            if any(w.startswith(needle) for w in f.split()):
                return 1
            return 2 if needle in f else None

        def best(items):
            items = [x for x in items if x[0] is not None]
            items.sort(key=lambda x: (x[0], -x[1], self._fold(x[2])))
            return items[:limit]

        p_out = []
        for pid, (name, n) in perf.items():
            rk, alias = rank(name), None
            if rk is None:
                for a in aliases.get(pid, ()):
                    ra = rank(a)
                    if ra is not None:
                        rk, alias = ra + 1, a
                        break
            p_out.append((rk, n, name, pid, alias))
        performers = [{"id": pid, "name": name, "n": n, "alias": alias,
                       "photo": self.performer_photo_version(pid)}
                      for _, n, name, pid, alias in best(p_out)]
        studs = [{"name": s, "n": n} for _, n, s in best([(rank(s), n, s) for s, n in studios.items()])]
        tgs = [{"name": t, "n": n} for _, n, t in best([(rank(t), n, t) for t, n in tags.items()])]
        sc = []
        for r in rows:
            rk = rank(r["title"])
            if rk is not None:
                sc.append((rk, TIER_RANK.get(r["tier"], 1), r["title"], r))
        sc.sort(key=lambda x: (x[0], -x[1], self._fold(x[2])))
        scenes = [{"scene_id": r["scene_id"], "title": r["title"], "tier": r["tier"],
                   "studio": r["studio"], "date": r["date"], "performers": r["performers"][:3],
                   "stream": self.stream_url(r["scene_id"], start=0)}
                  for _, _, _, r in sc[:limit]]
        return {"performers": performers, "studios": studs, "tags": tgs, "scenes": scenes}

    # --- who she works with -------------------------------------------------------------------

    def performer_costars(self, pid: str, limit: int = 12) -> list[dict]:
        """The performers she shares the most scenes with."""
        pid = str(pid)
        out: dict[str, list] = {}
        for r in self._catalogue_all():
            ids = r.get("performer_ids") or []
            if pid not in ids:
                continue
            for oid, name in zip(ids, r["performers"]):
                if oid != pid:
                    e = out.setdefault(oid, [name, 0])
                    e[1] += 1
        top = sorted(out.items(), key=lambda kv: (-kv[1][1], self._fold(kv[1][0])))[:limit]
        return [{"id": oid, "name": name, "n": n, "photo": self.performer_photo_version(oid)}
                for oid, (name, n) in top]
