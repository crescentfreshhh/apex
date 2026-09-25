"""Library management on top of Stash: the safety net (action log, grade
backups, capability check). Mixed into ``Service`` — kept apart from the
embedding/search code so the management surface stays readable."""

from __future__ import annotations

import time
from pathlib import Path


class LibraryMixin:
    # --- where library state lives (next to settings.json) -------------------

    def _state_dir(self) -> Path:
        return self._settings_path().parent

    def _backup_dir(self) -> Path:
        return self._state_dir() / "backups"

    def action_log(self):
        from ..ledger import ActionLog

        return ActionLog(self._state_dir() / "actions.jsonl")

    def _log_scene(self, action: str, row: dict | None, **fields) -> None:
        """One action-log line about a scene, identified every way that survives
        Stash changes (id, file fingerprint, path, title). Never raises: a full
        disk must not turn a successful Stash write into an error."""
        row = row or {}
        try:
            self.action_log().append(
                action, scene_id=row.get("scene_id"), fingerprint=row.get("fingerprint"),
                path=row.get("path"), title=row.get("title"), **fields)
        except Exception:  # noqa: BLE001
            pass

    def history(self, limit: int = 200) -> list[dict]:
        return self.action_log().tail(limit)

    # --- which optional Stash operations this server supports ----------------

    _CAPS_TTL = 600.0

    def capabilities(self, refresh: bool = False) -> dict:
        """{ok, reason, ops: {name: bool}}. Cached for a while; a failed check is
        not cached (Stash may just be restarting)."""
        cached = getattr(self, "_caps_cache", None)
        if cached and not refresh and time.monotonic() - cached[0] < self._CAPS_TTL:
            return cached[1]
        try:
            ops = self.client().capabilities()
        except Exception as exc:  # noqa: BLE001
            from ..stash_client import StashClient

            return {"ok": False, "reason": f"couldn't check Stash: {exc}",
                    "ops": {k: False for k in StashClient.CAPABILITY_FIELDS}}
        out = {"ok": True, "reason": "", "ops": ops}
        self._caps_cache = (time.monotonic(), out)
        return out

    def require_op(self, op: str) -> None:
        caps = self.capabilities()
        if not caps["ops"].get(op):
            raise RuntimeError(caps["reason"] or f"this Stash version has no {op} — update Stash to use it")

    # --- grade backups -------------------------------------------------------

    def backup_grades(self, rows: list[dict] | None = None, stamp: str | None = None) -> dict:
        from ..ledger import graded, write_backup

        rows = self._catalogue_all() if rows is None else rows
        path = write_backup(self._backup_dir(), rows, stamp=stamp)
        self._last_backup_day = time.strftime("%Y%m%d")
        return {"name": path.name,
                "count": sum(1 for r in rows if r.get("fingerprint") and graded(r))}

    def _maybe_daily_backup(self, rows: list[dict]) -> None:
        """First catalogue load of the day writes that day's backup (best-effort)."""
        today = time.strftime("%Y%m%d")
        if getattr(self, "_last_backup_day", None) == today:
            return
        self._last_backup_day = today
        if (self._backup_dir() / f"grades-backup-{today}.json").exists():
            return
        try:
            self.backup_grades(rows, stamp=today)
        except Exception:  # noqa: BLE001
            pass

    def list_grade_backups(self) -> list[dict]:
        from ..ledger import list_backups

        return list_backups(self._backup_dir())

    def backup_restore_preview(self, name: str) -> dict:
        from ..ledger import backup_diff, load_backup

        data = load_backup(self._backup_dir(), name)
        diff = backup_diff(data, self._catalogue_all(refresh=True))
        return {"name": name, "created": data.get("created"), **diff}

    def backup_restore_apply(self, job, name: str, confirm: bool = False) -> dict:
        """Re-apply a backup's grades to every scene whose grade differs (matched
        by fingerprint). Goes through the normal grade path, so tier tags and the
        action log behave exactly as for a hand-made grade."""
        from ..tiers import GRADES

        if not confirm:
            raise PermissionError("restoring a backup changes grades in Stash — confirm first")
        plan = self.backup_restore_preview(name)
        done, failed = 0, []
        for ch in plan["changes"]:
            if job is not None and job.cancelled:
                break
            to = ch["to"]
            grade = "reject" if to["tier"] == "rejected" else to["tier"]
            try:
                if grade in GRADES:
                    self.grade_scene(ch["scene_id"], grade, source=f"backup {name}")
                else:
                    self.restore_scene_grade(ch["scene_id"], to["rating100"], to["o_counter"],
                                             source=f"backup {name}")
                done += 1
                if job is not None:
                    job.progress = {"done": done, "total": len(plan["changes"])}
            except Exception as exc:  # noqa: BLE001
                failed.append({"scene_id": ch["scene_id"], "error": str(exc)})
                if job is not None:
                    job.log(f"scene {ch['scene_id']}: {exc}")
        return {"restored": done, "failed": failed, "missing": plan["missing"]}

    # --- tier tags (the renamer plugin files scenes by these) -----------------

    def tier_tag_names(self) -> dict[str, str]:
        from ..tiers import tier_tags

        return tier_tags(self._settings().get("tier_tags"))

    def save_tier_tags(self, tags: dict) -> dict[str, str]:
        import json

        from ..tiers import TIER_TAGS

        s = dict(self._settings())
        cur = dict(s.get("tier_tags") or {})
        for k, v in (tags or {}).items():
            if k in TIER_TAGS:
                if isinstance(v, str) and v.strip():
                    cur[k] = v.strip()[:80]
                else:
                    cur.pop(k, None)
        s["tier_tags"] = cur
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s, indent=2) + "\n")
        self._settings_cache = s
        self._tag_id_cache = {}
        self._cat_cache = None                 # rows carry tag states
        return self.tier_tag_names()

    def _tier_tag_ids(self) -> dict[str, str | None]:
        """{tier: Stash tag id or None if the tag doesn't exist yet}. Found ids
        are cached; missing ones are looked up again next time."""
        cache = getattr(self, "_tag_id_cache", None)
        if cache is None:
            cache = self._tag_id_cache = {}
        out = {}
        client = None
        for tier, name in self.tier_tag_names().items():
            key = name.lower()
            if key not in cache:
                client = client or self.client()
                t = client.find_tag_by_name(name)
                if t is not None:
                    cache[key] = str(t.id)
            out[tier] = cache.get(key)
        return out

    def _tier_tag_list(self, current: list[str], tier: str) -> list[str] | None:
        """The complete tag list a scene should carry in `tier`: its current tags
        minus every tier tag, plus this tier's (created if missing). None for a
        tier without a tag (reject etc. — tags are left alone)."""
        names = self.tier_tag_names()
        if tier not in names:
            return None
        ids = self._tier_tag_ids()
        this = ids.get(tier)
        if this is None:
            this = str(self.client().find_or_create_tag(names[tier]).id)
            self._tag_id_cache[names[tier].lower()] = this
        tier_ids = {i for i in ids.values() if i} | {this}
        return [str(t) for t in current if str(t) not in tier_ids] + [this]

    def tag_sync_preview(self) -> dict:
        """Graded scenes in a tagged tier whose tier tag / organized flag is not
        what a grade made in Peaks would give them. Nothing is changed."""
        rows = self._catalogue_all(refresh=True)
        todo = [r for r in rows if r["tag_state"]["needs_sync"]]
        moves = sum(1 for r in todo if r["tag_state"]["present"] != [r["tier"]])
        return {"count": len(todo), "moves": moves,
                "tags": self.tier_tag_names(),
                "items": [{k: r[k] for k in ("scene_id", "title", "path", "tier", "organized")}
                          | {"present": r["tag_state"]["present"]} for r in todo[:300]]}

    def tag_sync_apply(self, job, confirm: bool = False) -> dict:
        """Give every scene from the preview its tier tag (other tier tags
        removed) + organized, one sceneUpdate per scene, each logged."""
        if not confirm:
            raise PermissionError("syncing tier tags lets the renamer move files — confirm first")
        todo = [r for r in self._catalogue_all(refresh=True) if r["tag_state"]["needs_sync"]]
        done, failed = 0, []
        for r in todo:
            if job is not None and job.cancelled:
                break
            sid = r["scene_id"]
            try:
                cur = self._meta_client().scene_details([sid]).get(sid) or {}
                tags = self._tier_tag_list(cur.get("tag_ids") or [], r["tier"])
                self.client().update_scene(sid, tag_ids=tags, organized=True)
                self.invalidate_meta(sid)
                row = self._cat_update_row(sid)
                self._log_scene("tag-sync", row, after={"tier": row["tier"]},
                                detail=f"tag '{self.tier_tag_names()[row['tier']]}' + organized")
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed.append({"scene_id": sid, "error": str(exc)})
                if job is not None:
                    job.log(f"scene {sid}: {exc}")
            if job is not None:
                job.progress = {"done": done + len(failed), "total": len(todo)}
        return {"synced": done, "failed": failed}

    # --- bulk grading ------------------------------------------------------------

    def grade_bulk(self, job, scene_ids: list[str], grade: str) -> dict:
        """Grade scenes one at a time (one renamer run per scene), collecting each
        previous state so the whole batch can be undone together."""
        from ..tiers import GRADES

        if grade not in GRADES:
            raise ValueError(f"unknown grade: {grade}")
        ids = [str(s) for s in dict.fromkeys(scene_ids or [])]
        previous, failed = [], []
        for i, sid in enumerate(ids):
            if job is not None and job.cancelled:
                break
            try:
                out = self.grade_scene(sid, grade, source="bulk")
                previous.append({"scene_id": sid, **out["previous"]})
            except Exception as exc:  # noqa: BLE001
                failed.append({"scene_id": sid, "error": str(exc)})
                if job is not None:
                    job.log(f"scene {sid}: {exc}")
            if job is not None:
                job.progress = {"done": i + 1, "total": len(ids)}
        return {"graded": len(previous), "grade": grade, "previous": previous, "failed": failed}

    def restore_bulk(self, job, items: list[dict]) -> dict:
        done, failed = 0, []
        for i, it in enumerate(items or []):
            if job is not None and job.cancelled:
                break
            try:
                self.restore_scene_grade(it["scene_id"], it.get("rating100"), it.get("o_counter") or 0,
                                         source="bulk undo", tag_ids=it.get("tag_ids"),
                                         organized=it.get("organized"))
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed.append({"scene_id": it.get("scene_id"), "error": str(exc)})
            if job is not None:
                job.progress = {"done": i + 1, "total": len(items)}
        return {"restored": done, "failed": failed}

    # --- deleting rejects (and remembering what they looked like) --------------

    def reject_memory(self):
        from ..reject_memory import RejectMemory

        return RejectMemory(self.cfg.modeling.dir, self._model_name())

    def _remember_rejects(self, rows: list[dict], feats: dict | None = None) -> int:
        """Store triage features of 1★ scenes that are embedded and not yet
        remembered. Cheap when there's nothing new."""
        mem = self.reject_memory()
        have = mem.fingerprints()
        todo = [r for r in rows if r.get("tier") == "rejected" and r.get("fingerprint")
                and r["fingerprint"] not in have]
        if not todo:
            return 0
        feats = self._scene_features(todo) if feats is None else feats
        return mem.add({r["fingerprint"]: feats[r["scene_id"]] for r in todo
                        if r["scene_id"] in feats})

    def delete_preview(self, scene_ids: list[str] | None = None) -> dict:
        """The rejected (1★) scenes a delete would remove — all of them, or the
        selected ones — with their total size. Nothing is changed."""
        rows = [r for r in self._catalogue_all(refresh=True) if r["tier"] == "rejected"]
        if scene_ids is not None:
            want = {str(s) for s in scene_ids}
            rows = [r for r in rows if r["scene_id"] in want]
        return {"count": len(rows), "bytes": sum(int(r.get("size") or 0) for r in rows),
                "ids": [r["scene_id"] for r in rows],       # exactly what a confirm deletes
                "capable": bool(self.capabilities()["ops"].get("scenesDestroy")),
                "items": [{k: r.get(k) for k in ("scene_id", "title", "path", "size")}
                          for r in rows[:300]]}

    def _drop_local(self, rows: list[dict]) -> None:
        """Forget deleted scenes everywhere in Peaks: embedding cache (every
        model), hidden set, display metadata and the catalogue listing."""
        from ..cache import EmbeddingCache, path_key

        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        models = cache.models()
        for r in rows:
            key = r.get("fingerprint") or path_key(r.get("path") or r["scene_id"])
            for m in models:
                cache.delete(key, m)
            self.set_scene_hidden(r["scene_id"], False)
            self.invalidate_meta(r["scene_id"])
        for m in models:
            self.invalidate_index(m)
        gone = {r["scene_id"] for r in rows}
        cached = getattr(self, "_cat_cache", None)
        if cached:
            cached[1][:] = [r for r in cached[1] if r["scene_id"] not in gone]

    def delete_scenes(self, job, scene_ids: list[str], confirm: bool = False,
                      reason: str = "reject", keep_ids: set[str] | None = None) -> dict:
        """Delete scenes AND their files from Stash, a few at a time.

        reason="reject": each scene is re-read right before deletion and refused
        unless it is still rated 1★; its features go to the reject memory first.
        reason="duplicate": the caller has already chosen a keeper (never in
        `keep_ids`); the copies are not remembered as rejects.
        Every deleted file is logged."""
        if not confirm:
            raise PermissionError("deleting removes the files from disk — confirm first")
        if reason not in ("reject", "duplicate"):
            raise ValueError(f"unknown delete reason: {reason}")
        self.require_op("scenesDestroy")
        from ..tiers import tier_of

        ids = [str(s) for s in dict.fromkeys(scene_ids or []) if str(s) not in (keep_ids or set())]
        client = self.client()
        deleted, refused, failed, freed = [], [], [], 0
        CHUNK = 10
        for i in range(0, len(ids), CHUNK):
            if job is not None and job.cancelled:
                break
            part = ids[i:i + CHUNK]
            fresh = client.scene_details(part)          # the state right now, not the listing's
            ok_rows = []
            for sid in part:
                m = fresh.get(sid)
                if m is None:
                    refused.append({"scene_id": sid, "why": "no longer in Stash"})
                    continue
                row = self._cat_row(sid, m)
                if reason == "reject" and tier_of(m.get("rating100"), m.get("o_counter")) != "rejected":
                    refused.append({"scene_id": sid, "title": row["title"],
                                    "why": "not rated 1★ any more"})
                    self._cat_put_row(sid, m)                 # the listing was stale
                    self.set_scene_hidden(sid, False)
                    continue
                ok_rows.append(row)
            if not ok_rows:
                continue
            if reason == "reject":
                try:
                    self._remember_rejects(ok_rows)
                except Exception as exc:  # noqa: BLE001 — never block a delete on memory
                    if job is not None:
                        job.log(f"reject memory: {exc}")
            try:
                client.destroy_scenes([r["scene_id"] for r in ok_rows],
                                      delete_file=True, delete_generated=True)
            except Exception as exc:  # noqa: BLE001
                failed += [{"scene_id": r["scene_id"], "error": str(exc)} for r in ok_rows]
                if job is not None:
                    job.log(f"delete failed: {exc}")
                continue
            for r in ok_rows:
                freed += int(r.get("size") or 0)
                deleted.append(r["scene_id"])
                self._log_scene("delete", r, reason=reason, size=r.get("size"),
                                before={"rating100": r["rating100"], "o_counter": r["o_counter"],
                                        "tier": r["tier"]},
                                detail=f"file deleted ({reason})")
                if job is not None:
                    job.log(f"deleted {r['path']}")
            self._drop_local(ok_rows)
            if job is not None:
                job.progress = {"done": min(i + CHUNK, len(ids)), "total": len(ids)}
        return {"deleted": len(deleted), "freed_bytes": freed, "refused": refused,
                "failed": failed, "ids": deleted}

    # --- duplicates (Stash's phash groups, judged on file quality) --------------

    def _dupe_ignore_path(self) -> Path:
        return self._state_dir() / "duplicates_ignored.json"

    def _dupe_ignored(self) -> set[frozenset]:
        import json

        try:
            return {frozenset(map(str, g)) for g in json.loads(self._dupe_ignore_path().read_text())}
        except (OSError, ValueError):
            return set()

    def ignore_duplicate_group(self, scene_ids: list[str]) -> int:
        """'Not duplicates': this exact group stops showing (a new copy joining
        it makes it a different group, which shows again)."""
        import json

        ids = frozenset(str(s) for s in scene_ids)
        if len(ids) < 2:
            raise ValueError("a duplicate group needs at least two scenes")
        groups = self._dupe_ignored() | {ids}
        p = self._dupe_ignore_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sorted(sorted(g) for g in groups)))
        cached = getattr(self, "_dupe_cache", None)
        if cached:
            cached["groups"] = [g for g in cached["groups"]
                                if frozenset(s["scene_id"] for s in g["scenes"]) != ids]
        return len(groups)

    @staticmethod
    def dupe_keeper(rows: list[dict]) -> str:
        """The copy to keep: highest resolution, then bitrate, then file size;
        an existing keeper grade breaks remaining ties."""
        from ..tiers import RES_CLASSES

        rank = {c: i for i, c in enumerate(RES_CLASSES)}
        grade = {"legendaire": 4, "exceptionnelle": 3, "merveilleuse": 2, "upscale": 1}
        return max(rows, key=lambda r: (rank.get(r["quality"]["res"] or "", -1),
                                         r["quality"]["mbps"] or 0, int(r.get("size") or 0),
                                         grade.get(r["tier"], 0)))["scene_id"]

    @staticmethod
    def dupe_best_grade(rows: list[dict]) -> str | None:
        for t in ("legendaire", "exceptionnelle", "merveilleuse", "upscale"):
            if any(r["tier"] == t for r in rows):
                return t
        return None

    def find_duplicates(self, job=None, accuracy: str = "exact", duration_diff: float = -1.0,
                        only_ids: set[str] | None = None) -> dict:
        """Ask Stash for its phash duplicate groups and lay each out with the
        facts to judge them (resolution, bitrate, size, tier, path, added) plus a
        recommended keeper. `only_ids` keeps groups containing one of those
        scenes (the ingest check). The result is cached for the Catalogue."""
        import time as _t

        from ..stash_client import StashClient

        self.require_op("findDuplicateScenes")
        distance = StashClient.DUPLICATE_ACCURACY.get(accuracy)
        if distance is None:
            raise ValueError(f"unknown accuracy: {accuracy}")
        # the phash comparison can take minutes on a big library at low accuracy
        client = self.client()
        client.timeout = 900
        if job is not None:
            dur = "any duration" if duration_diff < 0 else f"duration ±{duration_diff:g}s"
            job.log(f"asking Stash for duplicates ({accuracy}, {dur})…")
        groups = client.duplicate_groups(distance, duration_diff)
        ignored = self._dupe_ignored()
        groups = [g for g in groups if frozenset(g) not in ignored]
        if only_ids is not None:
            groups = [g for g in groups if set(g) & {str(i) for i in only_ids}]
        ids = [s for g in groups for s in g]
        known = {r["scene_id"]: r for r in self._catalogue_all()}
        missing = [s for s in ids if s not in known]
        if missing:                       # e.g. copies outside the library folder
            for sid, m in client.scene_details(missing).items():
                known[sid] = self._cat_row(sid, m)
        out = []
        for g in groups:
            rows = [known[s] for s in g if s in known]
            if len(rows) < 2:
                continue
            keep = self.dupe_keeper(rows)
            size = {r["scene_id"]: int(r.get("size") or 0) for r in rows}
            out.append({"scenes": rows, "keep": keep, "best_grade": self.dupe_best_grade(rows),
                        "reclaim": sum(size.values()) - size[keep]})
        out.sort(key=lambda g: -g["reclaim"])
        result = {"groups": out, "accuracy": accuracy, "duration_diff": duration_diff,
                  "checked_at": _t.strftime("%Y-%m-%d %H:%M"), "ignored": len(ignored),
                  "reclaim": sum(g["reclaim"] for g in out)}
        if only_ids is None:
            self._dupe_cache = result
        return result

    def cached_duplicates(self) -> dict | None:
        return getattr(self, "_dupe_cache", None)

    def resolve_duplicate(self, job, keep_id: str, delete_ids: list[str], confirm: bool = False) -> dict:
        """Keep one copy, delete the others (files included). If any copy
        carries a better keeper grade than the kept one, that grade moves to the
        kept copy FIRST (normal grade path → tier tag + organized), so a grade
        is never lost with a deleted copy. Deleted copies aren't rejects."""
        from ..tiers import tier_of

        if not confirm:
            raise PermissionError("deleting removes the files from disk — confirm first")
        keep_id = str(keep_id)
        delete_ids = [str(s) for s in delete_ids if str(s) != keep_id]
        if not delete_ids:
            raise ValueError("nothing to delete")
        self.require_op("scenesDestroy")
        fresh = self.client().scene_details([keep_id, *delete_ids])
        if keep_id not in fresh:
            raise LookupError(f"the copy to keep (scene {keep_id}) is no longer in Stash")
        rows = [self._cat_row(s, m) for s, m in fresh.items()]
        best = self.dupe_best_grade(rows)
        order = ["upscale", "merveilleuse", "exceptionnelle", "legendaire"]
        kept_tier = tier_of(fresh[keep_id].get("rating100"), fresh[keep_id].get("o_counter"))
        carried = None
        if best and (kept_tier not in order or order.index(kept_tier) < order.index(best)):
            self.grade_scene(keep_id, best, source="duplicate")
            carried = best
        res = self.delete_scenes(job, delete_ids, confirm=True, reason="duplicate",
                                 keep_ids={keep_id})
        keep_row = self._cat_update_row(keep_id)
        self._log_scene("duplicate", keep_row, detail=(
            f"kept this copy, deleted {res['deleted']}" + (f", carried grade {best}" if carried else "")))
        cached = getattr(self, "_dupe_cache", None)
        if cached:
            cached["groups"] = [g for g in cached["groups"]
                                if keep_id not in {s["scene_id"] for s in g["scenes"]}]
        return {**res, "kept": keep_id, "carried_grade": carried}

    # --- ingest: new files end to end (the user's Stash routine, then Peaks) ----
    # scan (phashes on) → identify → auto tag, each with Stash's saved task
    # defaults and waited on; then Peaks embeds the new scenes and checks them
    # for duplicates. Identify and auto tag are limited to the new scenes.

    INGEST_POLL = 2.0
    _DONE = {"FINISHED", "CANCELLED", "FAILED"}

    def _ingest_path(self) -> Path:
        return self._state_dir() / "ingest.json"

    def last_ingest(self) -> dict:
        import json

        try:
            return json.loads(self._ingest_path().read_text())
        except (OSError, ValueError):
            return {}

    def _wait_stash_job(self, client, stash_job: str, label: str, job) -> dict:
        """Poll a Stash job until it ends. Stopping the Peaks job stops it."""
        last = None
        while True:
            if job is not None and job.cancelled:
                try:
                    client.stop_job(stash_job)
                except Exception:  # noqa: BLE001
                    pass
                raise InterruptedError(f"{label} stopped")
            info = client.find_job(stash_job)
            if info is None:                       # finished and already forgotten
                return {"status": "FINISHED"}
            status = info.get("status")
            if job is not None:
                pct = info.get("progress")
                msg = f"{label}: {status.lower() if status else '?'}" + (
                    f" {round(100 * pct)}%" if isinstance(pct, (int, float)) and pct >= 0 else "")
                if msg != last:
                    job.log(msg)
                    last = msg
                job.progress = {"stage": label, "pct": pct}
            if status in self._DONE:
                if status != "FINISHED":
                    raise RuntimeError(f"{label} {status.lower()}: {info.get('error') or 'see Stash logs'}")
                return info
            time.sleep(self.INGEST_POLL)

    # Stash scan options → (label, default). Defaults are the user's Stash scan
    # settings: covers + video phashes, nothing else.
    INGEST_SCAN_FIELDS = {
        "scanGenerateCovers": ("covers", True),
        "scanGeneratePreviews": ("previews", False),
        "scanGenerateImagePreviews": ("animated image previews", False),
        "scanGenerateSprites": ("scrubber sprites", False),
        "scanGeneratePhashes": ("video phashes", True),
        "scanGenerateThumbnails": ("image thumbnails", False),
        "scanGenerateImagePhashes": ("image phashes", False),
        "scanGenerateClipPreviews": ("image clip previews", False),
        "rescan": ("rescan files", False),
    }

    def ingest_scan_options(self) -> dict[str, bool]:
        saved = self._settings().get("ingest_scan") or {}
        opts = {k: bool(saved.get(k, d)) for k, (_, d) in self.INGEST_SCAN_FIELDS.items()}
        opts["scanGeneratePhashes"] = True                   # duplicates need them
        if not opts["scanGeneratePreviews"]:
            opts["scanGenerateImagePreviews"] = False        # a sub-option of previews
        return opts

    def save_ingest_scan(self, opts: dict) -> dict[str, bool]:
        import json

        s = dict(self._settings())
        s["ingest_scan"] = {k: bool(v) for k, v in (opts or {}).items()
                            if k in self.INGEST_SCAN_FIELDS and isinstance(v, bool)}
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s, indent=2) + "\n")
        self._settings_cache = s
        return self.ingest_scan_options()

    def _write_ingest(self, record: dict) -> None:
        import json

        try:
            self._ingest_path().parent.mkdir(parents=True, exist_ok=True)
            tmp = self._ingest_path().with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=1))
            tmp.replace(self._ingest_path())
        except OSError:
            pass

    def _ingest_refresh(self, new: list[str]) -> None:
        """Stash just changed the new scenes (titles, performers, tags…): drop
        their cached metadata so the Review queue / Catalogue show it."""
        for sid in new:
            self.invalidate_meta(sid)
        self._cat_cache = None

    def _repair_new_tier_tags(self, new: list[str], log) -> int:
        """Scenes graded while the ingest runs can lose their tier tag to a later
        Stash stage (e.g. identify overwriting tags). Re-tag any new scene whose
        keeper grade is missing its one tier tag or organized flag."""
        fixed = 0
        try:
            fresh = self._meta_client().scene_details(new)
        except Exception:  # noqa: BLE001 — best-effort; the final pass retries
            return 0
        for sid, m in fresh.items():
            row = self._cat_put_row(sid, m)
            if not row["tag_state"]["needs_sync"]:
                continue
            try:
                tags = self._tier_tag_list(m.get("tag_ids") or [], row["tier"])
                self.client().update_scene(sid, tag_ids=tags, organized=True)
                self.invalidate_meta(sid)
                row = self._cat_update_row(sid)
                self._log_scene("tag-sync", row, source="ingest", after={"tier": row["tier"]},
                                detail=f"re-tagged '{self.tier_tag_names()[row['tier']]}' after a Stash stage")
                fixed += 1
            except Exception as exc:  # noqa: BLE001
                log(f"couldn't re-tag scene {sid}: {exc}")
        if fixed:
            log(f"re-applied the tier tag on {fixed} scene(s) graded during the ingest")
        return fixed

    def run_ingest(self, job=None, embed_busy=None) -> dict:
        log = job.log if job is not None else print
        for op in ("metadataScan", "findJob"):
            self.require_op(op)
        client = self.client()
        client.timeout = 120
        stages: dict[str, str] = {}
        record = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "running": True,
                  "stage": "scan", "new": self.last_ingest().get("new") or [], "stages": stages}
        before = client.all_scene_ids()
        try:
            defaults = client.config_defaults()
        except Exception as exc:  # noqa: BLE001
            log(f"couldn't read Stash's saved task defaults ({exc}) — using Stash's own defaults")
            defaults = {"scan": None, "identify": None, "autoTag": None}

        new: list[str] = []
        dupes = None

        def stage(name: str) -> None:
            record["stage"] = name
            self._write_ingest(record)

        try:
            # 1. scan — Peaks' own scan options (Settings → Ingest), video phashes
            # always on; only fields this Stash version's scan input actually has
            opts = self.ingest_scan_options()
            scan_in = {k: v for k, v in opts.items() if client.input_has("ScanMetadataInput", k)}
            on = [label for k, (label, _) in self.INGEST_SCAN_FIELDS.items() if scan_in.get(k)]
            log("1/5 scan: " + (", ".join(on) or "nothing extra generated"))
            self._wait_stash_job(client, client.metadata_scan(scan_in), "scan", job)
            new = sorted(client.all_scene_ids() - before, key=lambda x: int(x) if x.isdigit() else 0)
            stages["scan"] = f"{len(new)} new scene(s)"
            log(f"scan done: {len(new)} new scene(s)")
            record["new"] = new                    # (a failed scan keeps the last list)
            if new:
                # publish now: the new scenes are reviewable while the rest runs
                self._cat_cache = None
                log("new scenes are in the Review queue now — the rest keeps running")

            if new:
                # 2. identify — saved sources/options, only the new scenes
                stage("identify")
                ident = defaults.get("identify") or {}
                if self.capabilities()["ops"].get("metadataIdentify") and ident.get("sources"):
                    ident_in = client.fit_input(ident, "IdentifyMetadataInput")
                    ident_in.pop("paths", None)
                    ident_in["sceneIDs"] = new
                    log(f"2/5 identify: {len(ident_in.get('sources') or [])} source(s), {len(new)} scene(s)")
                    self._wait_stash_job(client, client.metadata_identify(ident_in), "identify", job)
                    stages["identify"] = "done"
                    self._ingest_refresh(new)
                    self._repair_new_tier_tags(new, log)
                else:
                    stages["identify"] = "skipped — no identify sources saved in Stash's Tasks page"
                    log("2/5 identify: " + stages["identify"])

                # 3. auto tag — only the new files, at their paths NOW (a scene
                # graded meanwhile may have been moved by the renamer)
                stage("auto tag")
                if self.capabilities()["ops"].get("metadataAutoTag"):
                    paths = [m["path"] for m in client.scene_details(new).values() if m.get("path")]
                    at = defaults.get("autoTag") or {}
                    picked = {k: at.get(k) for k in ("performers", "studios", "tags") if at.get(k)}
                    at_in = picked or {"performers": ["*"], "studios": ["*"], "tags": ["*"]}
                    at_in = client.fit_input({**at_in, "paths": paths}, "AutoTagMetadataInput")
                    log(f"3/5 auto tag: {', '.join(k for k in ('performers', 'studios', 'tags') if k in at_in)}")
                    self._wait_stash_job(client, client.metadata_auto_tag(at_in), "auto tag", job)
                    stages["auto tag"] = "done"
                    self._ingest_refresh(new)
                    self._repair_new_tier_tags(new, log)
                else:
                    stages["auto tag"] = "skipped — not supported by this Stash"

                # 4. Peaks embed — only the new scenes, in id order (the order the
                # Review queue shows them), never the whole backlog
                stage("embed")
                if job is not None:
                    job.progress = {"stage": "embed"}
                if embed_busy and embed_busy():
                    stages["embed"] = "skipped — an embed pass is already running; the next pass picks these up"
                    log("4/5 embed: " + stages["embed"])
                else:
                    log(f"4/5 embed: {len(new)} new scene(s)")
                    st = self.run_embed(job, scene_ids=set(new))
                    stages["embed"] = f"{st.get('embedded', 0)} embedded, {st.get('failed', 0)} failed"

                # 5. duplicates among the new scenes
                stage("duplicates")
                if self.capabilities()["ops"].get("findDuplicateScenes"):
                    if job is not None:
                        job.progress = {"stage": "duplicates"}
                    log("5/5 duplicates: checking the new scenes against the library")
                    dupes = self.find_duplicates(job, only_ids=set(new))
                    stages["duplicates"] = f"{len(dupes['groups'])} group(s)"
                    self._merge_dupes(dupes)
                else:
                    stages["duplicates"] = "skipped — not supported by this Stash"
                self._repair_new_tier_tags(new, log)
            else:
                for k in ("identify", "auto tag", "embed", "duplicates"):
                    stages[k] = "nothing new"
        finally:
            # always leave a finished record, even when a stage failed
            record.update(running=False, stage="done", finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                          dupe_ids=sorted({r["scene_id"] for g in (dupes or {}).get("groups", [])
                                           for r in g["scenes"]}))
            self._write_ingest(record)
            self._cat_cache = None                 # new scenes → re-read the listing

        try:
            self.action_log().append("ingest", detail=f"{len(new)} new scene(s)", stages=stages)
        except Exception:  # noqa: BLE001
            pass
        if job is not None:
            job.progress = {"stage": "done"}
        log("done — " + " · ".join(f"{k}: {v}" for k, v in stages.items()))
        return {"new": len(new), "stages": stages,
                "duplicates": len((dupes or {}).get("groups", []))}

    def _merge_dupes(self, found: dict) -> None:
        """Show groups found by an ingest in the Duplicates view straight away."""
        if not found.get("groups"):
            return
        cached = getattr(self, "_dupe_cache", None)
        if cached is None:
            self._dupe_cache = dict(found)
            return
        have = {frozenset(r["scene_id"] for r in g["scenes"]) for g in cached["groups"]}
        extra = [g for g in found["groups"]
                 if frozenset(r["scene_id"] for r in g["scenes"]) not in have]
        cached["groups"] = extra + cached["groups"]
        cached["reclaim"] = sum(g["reclaim"] for g in cached["groups"])

    # --- browsing: facets, saved views, storage -----------------------------------

    @staticmethod
    def _storage(rows: list[dict]) -> dict:
        """Space and count per tier (sizes as Stash reports them)."""
        out: dict[str, dict] = {}
        for r in rows:
            t = out.setdefault(r["tier"], {"count": 0, "bytes": 0})
            t["count"] += 1
            t["bytes"] += int(r.get("size") or 0)
        return out

    LOW_TIERS = ("rejected", "unreviewed", "anomaly", "upscale")

    def storage(self, largest: int = 20) -> dict:
        rows = self._catalogue_all()
        low = sorted((r for r in rows if r["tier"] in self.LOW_TIERS),
                     key=lambda r: -int(r.get("size") or 0))[:largest]
        return {"tiers": self._storage(rows),
                "total": {"count": len(rows), "bytes": sum(int(r.get("size") or 0) for r in rows)},
                "largest_low": [{k: r.get(k) for k in ("scene_id", "title", "path", "size", "tier",
                                                        "quality")} for r in low]}

    def catalogue_facets(self, limit: int = 3000) -> dict:
        """Performers / studios / tags present in the library, most used first
        (feeds the Catalogue filter typeahead)."""
        from collections import Counter

        rows = self._catalogue_all()
        perf, stud, tags = Counter(), Counter(), Counter()
        for r in rows:
            perf.update(set(r["performers"]))
            if r["studio"]:
                stud[r["studio"]] += 1
            tags.update(set(r["tags"]))
        return {k: [[n, c] for n, c in cnt.most_common(limit)]
                for k, cnt in (("performers", perf), ("studios", stud), ("tags", tags))}

    SAVED_VIEW_KEYS = ("tier", "view", "new", "q", "res", "min_mbps", "sort", "performer",
                       "studio", "tag", "date_from", "date_to", "dur_min", "dur_max")

    def saved_views(self) -> list[dict]:
        return list(self._settings().get("saved_views") or [])

    def save_view(self, name: str, params: dict) -> list[dict]:
        name = (name or "").strip()[:60]
        if not name:
            raise ValueError("a saved view needs a name")
        clean = {k: v for k, v in (params or {}).items()
                 if k in self.SAVED_VIEW_KEYS and v not in (None, "", False)}
        views = [v for v in self.saved_views() if v.get("name") != name]
        views.append({"name": name, "params": clean})
        return self._write_views(views)

    def delete_view(self, name: str) -> list[dict]:
        return self._write_views([v for v in self.saved_views() if v.get("name") != name])

    def _write_views(self, views: list[dict]) -> list[dict]:
        import json

        s = dict(self._settings())
        s["saved_views"] = views
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s, indent=2) + "\n")
        self._settings_cache = s
        return views
