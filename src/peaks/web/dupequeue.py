"""Duplicates in batches: decisions are queued, deletions run in the background.

"Keep this" (or "keep the recommended copy in every group") puts the decision
in a queue and returns at once; one worker — the single "library" job, so the
renamer still sees one scene at a time — resolves the queued groups in order
with `resolve_duplicate` (the group's highest tier is carried to the keeper
first, copies aren't rejects, everything is logged).

Each item waits UNDO_SECONDS before it starts, so a misclick can be undone. A
failed item is listed (Retry / Dismiss) and never stops the rest. The queue is
a file in the state dir: it survives a reload or a restart, and an item that
was mid-delete is simply run again (copies already gone are refused harmlessly
as "no longer in Stash").
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

UNDO_SECONDS = 5.0
KEEP_FINISHED = 50


class DupeQueueMixin:
    # --- state -------------------------------------------------------------------

    def _dq_lock(self) -> threading.RLock:
        return self.__dict__.setdefault("_dq_lk", threading.RLock())

    def _dq_path(self) -> Path:
        return self._state_dir() / "dupe_queue.json"

    def _dq(self) -> dict:
        st = self.__dict__.get("_dq_mem")
        if st is None:
            try:
                st = json.loads(self._dq_path().read_text())
            except (OSError, ValueError):
                st = {}
            st.setdefault("items", [])
            st.setdefault("paused", False)
            for it in st["items"]:                 # a restart mid-delete: run it again
                if it["status"] == "running":
                    it["status"] = "queued"
            self._dq_mem = st
        return st

    def _dq_save(self) -> None:
        p = self._dq_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._dq()))
        os.replace(tmp, p)

    def _dq_open_ids(self) -> set[str]:
        """Scene ids in queued / running items (keepers and copies)."""
        return {s for it in self._dq()["items"] if it["status"] in ("queued", "running")
                for s in (it["keep"], *it["delete"])}

    # --- the duplicates view --------------------------------------------------------

    def _dq_take_group(self, keep: str, delete: list[str]) -> dict | None:
        """Remove the group these scenes belong to from the cached view (it
        comes back on undo / clear)."""
        cached = getattr(self, "_dupe_cache", None)
        if not cached:
            return None
        ids = {keep, *delete}
        for g in cached["groups"]:
            if ids & {r["scene_id"] for r in g["scenes"]}:
                cached["groups"].remove(g)
                cached["reclaim"] = sum(x["reclaim"] for x in cached["groups"])
                return g
        return None

    def _dq_put_group(self, g: dict | None) -> None:
        cached = getattr(self, "_dupe_cache", None)
        if not cached or not g:
            return
        if any({r["scene_id"] for r in x["scenes"]} == {r["scene_id"] for r in g["scenes"]}
               for x in cached["groups"]):
            return
        cached["groups"].append(g)
        cached["groups"].sort(key=lambda x: -x["reclaim"])
        cached["reclaim"] = sum(x["reclaim"] for x in cached["groups"])

    def dupe_queue_hide(self, groups: list[dict]) -> list[dict]:
        """A fresh scan shouldn't show groups already waiting in the queue."""
        busy = self._dq_open_ids()
        return [g for g in groups if not busy & {r["scene_id"] for r in g["scenes"]}]

    # --- queue actions ----------------------------------------------------------------

    def dupe_queue_add(self, keep: str, delete: list[str], delete_file: bool = True,
                       now: float | None = None) -> dict:
        keep = str(keep)
        delete = [str(s) for s in dict.fromkeys(delete or []) if str(s) != keep]
        if not delete:
            raise ValueError("nothing to delete")
        with self._dq_lock():
            if self._dq_open_ids() & {keep, *delete}:
                raise ValueError("these copies overlap a group that's already queued")
            g = self._dq_take_group(keep, delete)
            rows = {r["scene_id"]: r for r in (g or {}).get("scenes", [])}
            it = {"id": uuid.uuid4().hex[:10], "keep": keep, "delete": delete,
                  "delete_file": bool(delete_file), "status": "queued",
                  "queued_at": time.time() if now is None else now,
                  "not_before": (time.time() if now is None else now) + UNDO_SECONDS,
                  "title": (rows.get(keep) or {}).get("title") or f"Scene {keep}",
                  "bytes": sum(int(rows.get(s, {}).get("size") or 0) for s in delete),
                  "group": g}
            self._dq()["items"].append(it)
            self._dq_save()
        return self._dq_public(it)

    def dupe_queue_add_recommended(self, delete_file: bool = True) -> dict:
        cached = getattr(self, "_dupe_cache", None) or {}
        added, skipped = 0, 0
        for g in list(cached.get("groups") or []):
            keep = g["keep"]
            others = [r["scene_id"] for r in g["scenes"] if r["scene_id"] != keep]
            try:
                self.dupe_queue_add(keep, others, delete_file=delete_file)
                added += 1
            except ValueError:
                skipped += 1
        return {"added": added, "skipped": skipped, **self.dupe_queue_status()}

    def _dq_find(self, item_id: str) -> dict:
        it = next((x for x in self._dq()["items"] if x["id"] == item_id), None)
        if it is None:
            raise LookupError("no such queued group")
        return it

    def dupe_queue_undo(self, item_id: str) -> dict:
        """Take back a group that hasn't started — it returns to the view."""
        with self._dq_lock():
            it = self._dq_find(item_id)
            if it["status"] != "queued":
                raise ValueError("already " + {"running": "deleting", "done": "done", "failed": "failed"}[it["status"]])
            self._dq()["items"].remove(it)
            self._dq_put_group(it.get("group"))
            self._dq_save()
        return self.dupe_queue_status()

    def dupe_queue_retry(self, item_id: str) -> dict:
        with self._dq_lock():
            it = self._dq_find(item_id)
            if it["status"] != "failed":
                raise ValueError("only a failed group can be retried")
            it.update(status="queued", error=None, not_before=0.0)
            self._dq_save()
        return self.dupe_queue_status()

    def dupe_queue_dismiss(self, item_id: str) -> dict:
        with self._dq_lock():
            it = self._dq_find(item_id)
            if it["status"] in ("queued", "running"):
                raise ValueError("still in the queue — use Undo")
            self._dq()["items"].remove(it)
            self._dq_save()
        return self.dupe_queue_status()

    def dupe_queue_clear(self) -> dict:
        """Drop everything that hasn't started; those groups come back."""
        with self._dq_lock():
            st = self._dq()
            for it in [x for x in st["items"] if x["status"] == "queued"]:
                st["items"].remove(it)
                self._dq_put_group(it.get("group"))
            self._dq_save()
        return self.dupe_queue_status()

    def dupe_queue_pause(self, paused: bool) -> dict:
        with self._dq_lock():
            self._dq()["paused"] = bool(paused)
            self._dq_save()
        return self.dupe_queue_status()

    def dupe_queue_pending(self) -> bool:
        st = self._dq()
        return not st["paused"] and any(it["status"] == "queued" for it in st["items"])

    # --- the worker ---------------------------------------------------------------------

    def drain_dupe_queue(self, job=None, sleep=time.sleep) -> dict:
        """Resolve queued groups in order until none are left (or Stop)."""
        done = failed = 0
        while True:
            if job is not None and job.cancelled:
                self.dupe_queue_pause(True)
                break
            with self._dq_lock():
                st = self._dq()
                if st["paused"]:
                    break
                queued = [it for it in st["items"] if it["status"] == "queued"]
                if not queued:
                    break
                it = queued[0]
                wait = it["not_before"] - time.time()
                if wait <= 0:
                    it.update(status="running", started_at=time.time())
                    self._dq_save()
            if wait > 0:                           # the undo window
                sleep(min(wait, 1.0))
                continue
            if job is not None:
                left = sum(1 for x in self._dq()["items"] if x["status"] == "queued")
                job.log(f"keeping {it['keep']} ({it['title']}), deleting {', '.join(it['delete'])} · {left} more queued")
            try:
                res = self.resolve_duplicate(job, it["keep"], it["delete"], confirm=True,
                                             delete_file=it["delete_file"])
                upd = {"status": "done", "result": {k: res.get(k) for k in (
                    "deleted", "freed_bytes", "carried_grade", "files_deleted", "refused")}}
                done += 1
            except Exception as exc:  # noqa: BLE001 — one group's trouble mustn't stop the rest
                upd = {"status": "failed", "error": str(exc)}
                failed += 1
                if job is not None:
                    job.log(f"  ! {it['title']}: {exc}")
            with self._dq_lock():
                it.update(finished_at=time.time(), **upd)
                if it["status"] == "done":
                    it.pop("group", None)
                fin = [x for x in self._dq()["items"] if x["status"] == "done"]
                for old in fin[:-KEEP_FINISHED]:
                    self._dq()["items"].remove(old)
                self._dq_save()
        return {"resolved": done, "failed": failed}

    # --- what the page shows ----------------------------------------------------------------

    @staticmethod
    def _dq_public(it: dict) -> dict:
        return {k: v for k, v in it.items() if k != "group"}

    def dupe_queue_status(self) -> dict:
        st = self._dq()
        items = st["items"]
        by = {s: [self._dq_public(x) for x in items if x["status"] == s]
              for s in ("queued", "running", "done", "failed")}
        return {"paused": st["paused"], **by,
                "freed": sum(int((x.get("result") or {}).get("freed_bytes") or 0) for x in by["done"]),
                "queued_bytes": sum(x["bytes"] for x in by["queued"] + by["running"])}
