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
