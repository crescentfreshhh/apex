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
