"""Studios: a browsable list (scene count, tier mix, track record) and a page
per studio — its tiers, the performers it uses most, its release span and
best scenes. Built from the catalogue rows and the tier model's studio
records (the same ones the performer pages use)."""

from __future__ import annotations

from .find import TIER_RANK

TIER_ORDER = ("legendaire", "exceptionnelle", "merveilleuse", "upscale", "anomaly", "unreviewed", "rejected")


class StudiosMixin:
    def _studio_record(self, enc, name: str) -> dict | None:
        from ..words import share, verdict

        t = enc.studio.get(name)
        if not t:
            return None
        n, g, _, _ = enc.record(t)
        keep, top = t[2] / t[0], t[3] / t[0]
        word, tone = verdict(n, g, keep, enc.base[0])
        return {"n": int(n), "verdict": word, "tone": tone, "keep": round(keep, 3), "top": round(top, 3),
                "keep_words": share(keep), "top_words": share(top)}

    def _by_studio(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for r in self._catalogue_all():
            if r["studio"]:
                out.setdefault(r["studio"], []).append(r)
        return out

    @staticmethod
    def _best_first(rows: list[dict]) -> list[dict]:
        rows = sorted(rows, key=lambda r: r.get("date") or "", reverse=True)       # newest first…
        rows.sort(key=lambda r: -TIER_RANK.get(r["tier"], 1))                        # …within best tier first
        return rows

    def studios_list(self, q: str | None = None, sort: str = "scenes") -> dict:
        """Every studio in the library. sort: scenes | best | name | recent."""
        groups = self._by_studio()
        enc = self._who_records(self._catalogue_all())
        needle = self._fold(q or "")
        out = []
        for name, rows in groups.items():
            if needle and needle not in self._fold(name):
                continue
            tiers: dict[str, int] = {}
            for r in rows:
                tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
            live = [r for r in rows if r["tier"] != "rejected"]
            cover = self._best_first(live or rows)[0]["scene_id"]
            dates = [r["date"] for r in rows if r.get("date")]
            out.append({"name": name, "scenes": len(live), "rejected": len(rows) - len(live), "tiers": tiers,
                        "top": tiers.get("legendaire", 0) + tiers.get("exceptionnelle", 0),
                        "record": self._studio_record(enc, name), "cover": cover,
                        "latest": max(dates) if dates else ""})
        if sort == "recent":                                   # latest release first, undated last
            out.sort(key=lambda s: s["latest"], reverse=True)
        else:
            keys = {"name": lambda s: self._fold(s["name"]),
                    "best": lambda s: (-s["top"], -s["scenes"], self._fold(s["name"])),
                    "scenes": lambda s: (-s["scenes"], self._fold(s["name"]))}
            out.sort(key=keys.get(sort, keys["scenes"]))
        return {"studios": out, "total": len(out)}

    def studio_profile(self, name: str) -> dict:
        rows = self._by_studio().get(name)
        if not rows:
            raise LookupError(f"no scenes from studio {name!r} in the library")
        enc = self._who_records(self._catalogue_all())
        tiers: dict[str, int] = {}
        perf: dict[str, list] = {}
        for r in rows:
            tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
            if r["tier"] == "rejected":
                continue
            for pid, pname in zip(r.get("performer_ids") or [], r["performers"]):
                e = perf.setdefault(pid, [pname, 0, 0])
                e[1] += 1
                e[2] += r["tier"] in ("legendaire", "exceptionnelle")
        live = [r for r in rows if r["tier"] != "rejected"]
        dates = sorted(r["date"] for r in live if r.get("date"))
        top_perf = sorted(perf.items(), key=lambda kv: (-kv[1][1], -kv[1][2], self._fold(kv[1][0])))[:16]
        best = self._best_first(live)
        return {
            "name": name, "scenes": len(live), "rejected": len(rows) - len(live),
            "tiers": {t: tiers[t] for t in TIER_ORDER if tiers.get(t)},
            "record": self._studio_record(enc, name),
            "first": dates[0] if dates else None, "latest": dates[-1] if dates else None,
            "gb": round(sum(int(r.get("size") or 0) for r in live) / 1e9, 1),
            "performers": [{"id": pid, "name": n, "n": c, "top": t, "photo": self.performer_photo_version(pid)}
                           for pid, (n, c, t) in top_perf],
            "best": [{"scene_id": r["scene_id"], "title": r["title"], "tier": r["tier"], "date": r.get("date") or "",
                      "performers": r["performers"][:3], "stream": self.stream_url(r["scene_id"], start=0)}
                     for r in best[:24]],
        }

    def studio_board(self, name: str, count: int = 3000, per_scene: int = 4) -> dict:
        """Megaboard supply for a studio: the best moments of each of its scenes
        (rejects left out), like the tier channels."""
        rows = [r for r in self._by_studio().get(name, []) if r["tier"] != "rejected"]
        if not rows:
            raise LookupError(f"no scenes from studio {name!r} in the library")
        sids = [r["scene_id"] for r in rows]
        return {"hits": self._moment_pool_for_scenes(sids, per_scene, count), "scenes": len(sids), "label": name}
