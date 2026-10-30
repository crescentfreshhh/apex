"""Orphans: scenes with no performer, and who they might be.

A scene is an orphan when Stash lists no performer for it — or only junk
"performers" that are really mis-applied tags (`_excluded_performers`, e.g.
"upscale"). Rejects are left out: nobody needs to name a scene on its way out.

Suggestions come from what the scene looks like, compared with the scenes you've
already credited: each performer is scored by her nearest credited scene (the
mean of its frame embeddings, cosine) blended with her overall centroid, plus a
small nudge when she has worked for the scene's studio. These are look-alike
embeddings, not face recognition — a hint to check against the scene shown next
to it, never written without a click. Adding a performer merges into the scene's
list in Stash (never replaces it), goes in the action log and can be undone;
"Not them" is remembered so the same name isn't offered again.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

NEAREST_W = 0.7          # nearest credited scene vs. the performer's overall centroid
STUDIO_NUDGE = 0.02      # she has scenes at this studio
SUGGEST_K = 3


class OrphansMixin:
    # --- which scenes ----------------------------------------------------------------

    def _is_orphan(self, row: dict, excluded: set[str] | None = None) -> bool:
        if row.get("tier") == "rejected":
            return False
        excluded = self._excluded_performers() if excluded is None else excluded
        return not [p for p in row.get("performers") or [] if p.strip().lower() not in excluded]

    def orphan_rows(self, rows: list[dict]) -> list[dict]:
        """Orphans among `rows`: keepers first (they're the ones worth naming), newest first within."""
        from .service import TIER_RANK_SORT

        ex = self._excluded_performers()
        out = [r for r in rows if self._is_orphan(r, ex)]
        out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        out.sort(key=lambda r: -TIER_RANK_SORT.get(r["tier"], 0))
        return out

    # --- "not them" memory -----------------------------------------------------------

    def _orph_path(self) -> Path:
        return self._state_dir() / "orphan_not_them.json"

    def _orph_lock(self) -> threading.Lock:
        return self.__dict__.setdefault("_orph_lk", threading.Lock())

    def _orph_no(self) -> dict[str, list[str]]:
        d = self.__dict__.get("_orph_mem")
        if d is None:
            try:
                d = json.loads(self._orph_path().read_text())
            except (OSError, ValueError):
                d = {}
            self._orph_mem = d
        return d

    def orphan_not_them(self, scene_id: str, performer_id: str) -> dict:
        from .. import atomic

        with self._orph_lock():
            d = self._orph_no()
            ids = d.setdefault(str(scene_id), [])
            if str(performer_id) not in ids:
                ids.append(str(performer_id))
            p = self._orph_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            atomic.write_json(p, d)
        return {"ok": True}

    # --- suggestions -----------------------------------------------------------------

    def _scene_means(self):
        """{scene_id: unit mean frame embedding} for every embedded scene, cached per index build."""
        idx = self.index(self._model_name())
        cached = self.__dict__.get("_scene_means_cache")
        if cached and cached[0] == id(idx):
            return cached[1]
        out = {}
        for key, (a, b) in idx._key_rows.items():
            sid = str((idx.key_meta.get(key) or {}).get("scene_id") or "")
            if sid and b > a:
                out[sid] = self._unit(idx.rows(a, b).mean(axis=0))
        self._scene_means_cache = (id(idx), out)
        return out

    def orphan_suggestions(self, scene_id: str, k: int = SUGGEST_K) -> dict:
        """Who this scene might be: [{id, name, score, strength, via}] best first.
        `via` is her nearest credited scene, to compare with your own eyes."""
        sid = str(scene_id)
        stats = self.performer_stats()
        cents = getattr(self, "_perf_centroids", {})
        by_id = getattr(self, "_perf_rows_by_id", {})
        scenes_of = getattr(self, "_perf_scenes_by_id", {})
        means = self._scene_means()
        me = means.get(sid)
        if me is None or not stats:
            return {"scene_id": sid, "suggestions": [],
                    "reason": "not embedded yet" if me is None else "no credited performers to compare with"}
        rows = {r["scene_id"]: r for r in self._catalogue_all()}
        studio = (rows.get(sid) or {}).get("studio") or ""
        studio_perfs = {pid for r in rows.values() if studio and r.get("studio") == studio
                        for pid in r.get("performer_ids") or []}
        no = set(self._orph_no().get(sid, []))
        scored = []
        for pid, cent in cents.items():
            if pid in no:
                continue
            best, via = -1.0, None
            for s in scenes_of.get(pid, ()):
                v = means.get(s)
                if v is None or s == sid:
                    continue
                c = float(me @ v)
                if c > best:
                    best, via = c, s
            if via is None:
                continue
            score = NEAREST_W * best + (1 - NEAREST_W) * float(me @ cent)
            if pid in studio_perfs:
                score += STUDIO_NUDGE
            scored.append((score, pid, via))
        scored.sort(reverse=True)
        top = scored[:k]
        out = []
        lead = top[0][0] - (scored[1][0] if len(scored) > 1 else top[0][0] - 0.1) if top else 0
        for i, (score, pid, via) in enumerate(top):
            # plain words, not a fake probability: what counts is a clear lead over the
            # other names (and the labels never rise down the list)
            if i == 0:
                strength = "strong" if lead >= 0.04 else "possible"
            else:
                strength = "possible" if top[0][0] - score < 0.03 else "weak"
            vr = rows.get(via) or {}
            out.append({"id": pid, "name": (by_id.get(pid) or {}).get("name", ""), "score": round(score, 4),
                        "strength": strength, "studio": pid in studio_perfs,
                        "via": {"scene_id": via, "title": vr.get("title") or f"scene {via}",
                                "cover": f"/api/scene/{via}/cover"}})
        return {"scene_id": sid, "suggestions": out}

    # --- writing it to Stash -------------------------------------------------------------

    def add_performer(self, scene_id: str, performer_id: str, source: str = "orphans") -> dict:
        """Add a performer to a scene in Stash (merged with any it has). Returns the
        row and the previous performer ids, for an undo."""
        sid = str(scene_id)
        cur = self._meta_client().scene_details([sid]).get(sid)
        if not cur:
            raise LookupError(f"scene {sid} not found in Stash")
        prev = [str(p.get("id")) for p in cur.get("performers_detail") or [] if p.get("id")]
        new = prev + ([str(performer_id)] if str(performer_id) not in prev else [])
        self.client().update_scene(sid, performer_ids=new)
        return self._performers_written(sid, prev, new, source)

    def orphan_create_performer(self, scene_id: str, name: str) -> dict:
        """The name typed isn't in Stash: create the performer, then add them to the
        scene (merged, logged, undoable like any add). An exact name match —
        ignoring case and spacing — is reused instead, so nobody is created twice."""
        import re

        clean = " ".join(str(name or "").split())
        if not clean:
            raise ValueError("a name is required")
        sid = str(scene_id)
        if not self._meta_client().scene_details([sid]).get(sid):
            raise LookupError(f"scene {sid} not found in Stash")
        pat = "(?i)^" + re.sub(r"\s+", r"\\s+", "".join("\\" + c if c in ".^$*+?()[]{}|\\" else c for c in clean)) + "$"
        same = [p for p in self.client().find_performers(pat, limit=5)
                if " ".join(p.get("name", "").split()).lower() == clean.lower()]
        if same:
            perf, created = {"id": str(same[0]["id"]), "name": same[0].get("name", clean)}, False
        else:
            perf, created = self.client().create_performer(clean), True
            self.__dict__.pop("_perf_dir_cache", None)          # the Performers page lists them
        out = self.add_performer(sid, perf["id"], source="orphans")
        if created:
            self._log_scene("performer-created", out["scene"], performer_id=perf["id"], name=perf["name"],
                            source="orphans")
        return {**out, "performer": perf, "created": created}

    def set_scene_performers(self, scene_id: str, performer_ids: list[str], source: str = "undo") -> dict:
        """Put a scene's performer list back exactly (the undo of `add_performer`)."""
        sid = str(scene_id)
        cur = self._meta_client().scene_details([sid]).get(sid) or {}
        prev = [str(p.get("id")) for p in cur.get("performers_detail") or [] if p.get("id")]
        ids = [str(p) for p in performer_ids]
        self.client().update_scene(sid, performer_ids=ids)
        return self._performers_written(sid, prev, ids, source)

    def _performers_written(self, sid: str, prev: list[str], new: list[str], source: str) -> dict:
        self.invalidate_meta(sid)
        row = self._cat_update_row(sid)
        self._perf_stats_cache = None              # her scene count changed
        self._log_scene("performers", row, source=source, before=prev, after=new)
        return {"scene": row, "previous": prev}
