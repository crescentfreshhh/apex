"""Library management on top of Stash: the safety net (action log, grade
backups, capability check). Mixed into ``Service`` — kept apart from the
embedding/search code so the management surface stays readable."""

from __future__ import annotations

import re
import threading
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
                self._watch_path(sid, row.get("path"))      # the renamer will move it
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

    # --- following your renamer: a tier change moves the file; follow it -----------
    # After a grade your renamer (a Stash plugin) renames/moves the file and Stash
    # records the new path. Peaks' cache still has the old one until a Sync — so
    # each changed scene is watched for a few minutes and fixed the moment Stash
    # reports the move: every model's cache entry, any loaded index, the
    # catalogue row. Nothing is re-embedded.
    _WATCH_STEPS = (3.0, 10.0, 30.0, 90.0, 180.0)   # seconds after the change
    _WATCH_TICK = 2.0

    def follow_renamer_on(self) -> bool:
        return bool(self._settings().get("follow_renamer", True))

    def _watch_path(self, sid: str, old_path: str | None) -> None:
        """Start following one scene after a tier change (`old_path`: where the
        file was when the change was made)."""
        if not self.follow_renamer_on():
            return
        now = time.monotonic()
        with self.__dict__.setdefault("_watch_lock", threading.Lock()):
            self.__dict__.setdefault("_path_watch", {})[str(sid)] = {
                "t0": now, "step": 0, "due": now + self._WATCH_STEPS[0], "old": old_path}
            if self.__dict__.get("_watch_running"):
                return
            self._watch_running = True
        threading.Thread(target=self._path_worker, daemon=True, name="peaks-follow-renamer").start()

    def _path_worker(self) -> None:
        while True:
            time.sleep(self._WATCH_TICK)
            try:
                self._path_watch_tick()
            except Exception:  # noqa: BLE001 — never let the follower die on a blip
                pass
            with self._watch_lock:
                if not self._path_watch:
                    self._watch_running = False
                    return

    def _path_watch_tick(self, now: float | None = None) -> int:
        """Check the scenes that are due: one batched Stash read per 400. Returns
        how many moves were picked up."""
        import os

        now = time.monotonic() if now is None else now
        with self._watch_lock:
            due = [sid for sid, e in self._path_watch.items() if e["due"] <= now]
        if not due:
            return 0
        model = self._model_name()
        try:
            idx = self.index(model)
            sid_key = {str(m.get("scene_id")): k for k, m in idx.key_meta.items()}
        except Exception:  # noqa: BLE001 — no index: only the catalogue rows follow
            idx, sid_key = None, {}
        fixed = 0
        for i in range(0, len(due), 400):
            batch = due[i: i + 400]
            try:
                details = self._meta_client().scene_details(batch)
            except Exception:  # noqa: BLE001 — Stash down: try again at the next step
                details = None
            for sid in batch:
                with self._watch_lock:
                    e = self._path_watch.get(sid)
                if e is None:
                    continue
                cur = (details or {}).get(sid) or {}
                new = cur.get("path")
                key = sid_key.get(sid)
                cached = ((idx.key_meta.get(key) or {}).get("path") if idx is not None and key else None)
                if details is not None and new and (new != e["old"] or (cached and new != cached)):
                    if key and cached != new:
                        self._apply_moved_path(key, new)
                    row = self._cat_put_row(sid, cur)
                    self._log_scene("moved", row, before={"path": e["old"]}, after={"path": new},
                                    detail="followed your renamer — path updated, nothing re-embedded")
                    self.__dict__.setdefault("_moves", []).append(
                        {"at": time.time(), "scene_id": sid, "title": row.get("title"), "path": new})
                    del self._moves[:-200]
                    self._unresolved_moves().pop(sid, None)
                    with self._watch_lock:
                        self._path_watch.pop(sid, None)
                    fixed += 1
                    continue
                step = e["step"] + 1
                if step >= len(self._WATCH_STEPS):
                    with self._watch_lock:
                        self._path_watch.pop(sid, None)
                    # Stash still has the old path but the file isn't there: moved
                    # behind Stash's back — a Stash scan (then Sync) is needed
                    gone = new or e["old"]
                    # (only when Peaks can see that folder at all — no false alarms
                    # on a setup where the library isn't mounted here)
                    if gone and os.path.isdir(os.path.dirname(gone)) and not os.path.exists(gone):
                        self._unresolved_moves()[sid] = {"at": time.time(), "scene_id": sid,
                                                         "title": cur.get("title") or "", "path": gone}
                    continue
                e["step"], e["due"] = step, e["t0"] + self._WATCH_STEPS[step]
        return fixed

    def _unresolved_moves(self) -> dict:
        return self.__dict__.setdefault("_moves_unresolved", {})

    def renamer_follow_status(self) -> dict:
        """For the Maintenance card: moves picked up automatically today, the
        most recent ones, scenes still being watched, and any the renamer moved
        where Stash can't see them yet."""
        day = time.strftime("%Y-%m-%d")
        moves = self.__dict__.get("_moves", [])
        today = [m for m in moves if time.strftime("%Y-%m-%d", time.localtime(m["at"])) == day]
        return {"on": self.follow_renamer_on(), "today": len(today), "recent": moves[-5:][::-1],
                "watching": len(self.__dict__.get("_path_watch", {})),
                "unresolved": list(self._unresolved_moves().values())[-20:]}

    # --- cleaning the library root: zips and empty folders (on Sync) ---------------
    # Your image-set zips and the empty folders a renamer leaves behind. Only
    # *.zip files are deleted and only folders with nothing left in them are
    # removed (os.rmdir — it refuses anything that isn't empty). The root itself
    # is never removed; a missing, empty (unmounted?) or read-only root is left
    # alone and the reason reported. Peaks' backup folder (and any folder holding
    # a .peaks-keep marker) is never entered. The first run only lists what it would do;
    # after you approve it once, every Sync cleans up automatically.

    def _cleanup_root(self) -> Path:
        import os

        return Path(os.environ.get("PEAKS_CLEANUP_ROOT", "/data") or "/data")

    def cleanup_settings(self) -> dict:
        s = self._settings()
        return {"on": bool(s.get("cleanup_on_sync", True)), "approved": bool(s.get("cleanup_approved", False)),
                "root": str(self._cleanup_root())}

    def save_cleanup_settings(self, on: bool | None = None, approved: bool | None = None) -> dict:
        import json

        s = dict(self._settings())
        if on is not None:
            s["cleanup_on_sync"] = bool(on)
        if approved is not None:
            s["cleanup_approved"] = bool(approved)
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s, indent=2) + "\n")
        self._settings_cache = s
        return self.cleanup_settings()

    def cleanup_library_root(self, apply: bool = False, log=None) -> dict:
        """Find (and with `apply`, delete) every *.zip under the root, then every
        folder left with nothing in it, deepest first."""
        import os

        log = log or (lambda *_: None)
        root = self._cleanup_root()
        out = {"root": str(root), "applied": False, "zips": 0, "folders": 0, "bytes": 0,
               "sample": [], "errors": [], "skipped": None, "at": time.time()}
        if not root.is_absolute() or not root.is_dir():
            out["skipped"] = f"{root} isn't there"
        else:
            try:
                has_any = any(root.iterdir())
            except OSError as exc:
                has_any, out["skipped"] = False, f"can't read {root}: {exc}"
            if not has_any and not out["skipped"]:
                out["skipped"] = f"{root} is empty — is the library mounted?"
        if out["skipped"]:
            self._last_cleanup = out
            return out
        zips: list[Path] = []
        empties: set[Path] = set()
        # never enter Peaks' own backup folder, or any folder marked .peaks-keep
        protected = set()
        try:
            protected.add(Path(self.backup_root()).resolve())
        except Exception:  # noqa: BLE001
            pass
        walked = []
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            here = Path(dirpath)
            every = list(dirnames)
            dirnames[:] = [d for d in dirnames
                           if (here / d).resolve() not in protected and not (here / d / ".peaks-keep").exists()]
            walked.append((here, every, filenames))
        for here, every, filenames in reversed(walked):          # children before parents
            kept = [f for f in filenames if not f.lower().endswith(".zip")]
            for f in filenames:
                if f.lower().endswith(".zip"):
                    zips.append(here / f)
            # empty once its zips go: no other file, and every subfolder is empty too
            # (a protected subfolder is never "empty", so its parent stays)
            if here != root and not kept and all((here / d) in empties for d in every):
                empties.add(here)
        for z in zips:
            try:
                out["bytes"] += z.stat().st_size
            except OSError:
                pass
        out["zips"], out["folders"] = len(zips), len(empties)
        out["sample"] = ([{"kind": "zip", "path": str(z)} for z in zips[:40]]
                         + [{"kind": "folder", "path": str(d)} for d in sorted(empties)[:40]])
        if apply and (zips or empties):
            if not os.access(root, os.W_OK):
                out["skipped"] = (f"Peaks can't write to {root} — set the Media path to read-write "
                                  "in the container settings")
                self._last_cleanup = out
                return out
            done_z = done_d = freed = 0
            for z in zips:
                try:
                    size = z.stat().st_size
                    z.unlink()
                    done_z += 1
                    freed += size
                    self.action_log().append("cleanup", path=str(z), detail="deleted zip (image set)", size=size)
                    log(f"  - deleted zip {z}")
                except OSError as exc:
                    out["errors"].append({"path": str(z), "error": str(exc)})
            for d in sorted(empties, key=lambda p: len(p.parts), reverse=True):
                try:
                    d.rmdir()                       # refuses unless truly empty
                    done_d += 1
                    self.action_log().append("cleanup", path=str(d), detail="removed empty folder")
                    log(f"  - removed empty folder {d}")
                except OSError as exc:
                    out["errors"].append({"path": str(d), "error": str(exc)})
            out.update(applied=True, zips=done_z, folders=done_d, bytes=freed)
        self._last_cleanup = out
        return out

    def cleanup_status(self) -> dict:
        return {**self.cleanup_settings(), "last": self.__dict__.get("_last_cleanup")}

    # --- same-file copies: one scene, the same download attached twice -----------
    # NOT the Duplicates tool (different scenes that look alike). Here Stash has
    # already put 2+ files on ONE scene ("File count > 1"). Only files with the
    # exact same size AND the same content hash (oshash, else md5) count as
    # copies; one of each set is kept — in the scene's tier folder, with the
    # cleaner name, the primary, the oldest — and only the extra FILES are
    # deleted. The scene (grade, markers, tags, Peaks' data) is never touched,
    # and never loses its last file. Same size but a different/missing hash, or
    # different sizes, are only listed. First run lists; one approval makes it
    # automatic on every Sync and Ingest.
    _COLLISION_NAME = re.compile(r"^(none(_\d+)?|copy of .+|.+_\d+|.+ \(\d+\))$", re.I)

    def copies_settings(self) -> dict:
        s = self._settings()
        return {"on": bool(s.get("copies_on", True)), "approved": bool(s.get("copies_approved", False))}

    def save_copies_settings(self, on: bool | None = None, approved: bool | None = None) -> dict:
        import json

        s = dict(self._settings())
        if on is not None:
            s["copies_on"] = bool(on)
        if approved is not None:
            s["copies_approved"] = bool(approved)
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s, indent=2) + "\n")
        self._settings_cache = s
        return self.copies_settings()

    def find_file_copies(self) -> dict:
        """What would go: per scene, the file kept and the identical copies to
        delete (with why), plus what's left for you to look at."""
        import unicodedata

        from ..tiers import tier_of

        def fold(t: str) -> str:
            return "".join(c for c in unicodedata.normalize("NFD", str(t or "")) if not unicodedata.combining(c)).lower().strip()

        tags, names = self.tier_tag_names(), self.tier_display_names()
        plans, needs_look, versions = [], [], []
        for sc in self.client().multi_file_scenes():
            tier = tier_of(sc.get("rating100"), sc.get("o_counter"))
            homes = {fold(x) for x in (tags.get(tier), names.get(tier), tier) if x} if tier in tags else set()
            primary = sc["files"][0]["id"]

            def in_home(f):
                return bool(homes) and any(fold(seg) in homes for seg in Path(f["path"]).parent.parts)

            def rank(f):
                stem = Path(f["basename"] or f["path"]).stem
                return (not in_home(f), bool(self._COLLISION_NAME.match(stem)), f["id"] != primary, f["mod_time"] or "")

            by_size: dict[int, list] = {}
            for f in sc["files"]:
                by_size.setdefault(f["size"], []).append(f)
            if len(by_size) > 1:
                versions.append({"scene_id": sc["id"], "title": sc["title"],
                                 "files": [{"path": f["path"], "size": f["size"]} for f in sc["files"]]})
            for size, fs in by_size.items():
                if len(fs) < 2:
                    continue
                by_hash: dict[str, list] = {}
                for f in fs:
                    h = f["fingerprints"].get("oshash") or f["fingerprints"].get("md5")
                    by_hash.setdefault(h or f"?{f['id']}", []).append(f)
                for h, group in by_hash.items():
                    if h.startswith("?") or len(group) < 2:
                        continue
                    keep = min(group, key=rank)
                    drop = [f for f in group if f is not keep]
                    why = ("kept the copy in the " + (names.get(tier) or tier) + " folder") if in_home(keep) and any(
                        not in_home(f) for f in drop) else ("kept the cleaner name" if any(
                            self._COLLISION_NAME.match(Path(f["basename"] or f["path"]).stem) for f in drop)
                            and not self._COLLISION_NAME.match(Path(keep["basename"] or keep["path"]).stem)
                        else "kept Stash's primary file" if keep["id"] == primary else "kept the oldest copy")
                    plans.append({"scene_id": sc["id"], "title": sc["title"], "tier": tier, "size": size,
                                  "keep": {"id": keep["id"], "path": keep["path"], "primary": keep["id"] == primary},
                                  "delete": [{"id": f["id"], "path": f["path"]} for f in drop], "why": why})
                left = [f for h, g in by_hash.items() if h.startswith("?") or len(g) < 2 for f in g]
                if len(left) >= 2 or (left and len(by_hash) > 1):
                    needs_look.append({"scene_id": sc["id"], "title": sc["title"], "size": size,
                                       "files": [f["path"] for f in fs],
                                       "reason": "same size, but the content hashes differ or are missing"})
        return {"plans": plans, "needs_look": needs_look, "versions": versions,
                "files": sum(len(p["delete"]) for p in plans),
                "bytes": sum(p["size"] * len(p["delete"]) for p in plans)}

    def remove_file_copies(self, apply: bool = False, log=None) -> dict:
        """Find (and with `apply`, delete) the extra identical files."""
        import os

        log = log or (lambda *_: None)
        found = self.find_file_copies()
        out = {**found, "applied": False, "removed": 0, "freed": 0, "errors": [], "method": None, "at": time.time()}
        if apply and found["plans"]:
            client = self.client()
            via_stash = bool(self.capabilities()["ops"].get("deleteFiles"))
            out["method"] = "stash" if via_stash else "direct"
            root = Path(self._cleanup_root()).resolve()
            for p in found["plans"]:
                try:
                    if not p["keep"]["primary"]:           # never leave the scene pointing at a deleted file
                        client.set_primary_file(p["scene_id"], p["keep"]["id"])
                    if via_stash:
                        client.delete_files([d["id"] for d in p["delete"]])
                    else:
                        for d in p["delete"]:
                            path = Path(d["path"]).resolve()
                            if root in path.parents and path.is_file():
                                os.remove(path)
                            else:
                                raise RuntimeError(f"{d['path']} isn't under {root} — left for Stash")
                    for d in p["delete"]:
                        self.action_log().append("file-copy", scene_id=p["scene_id"], title=p["title"],
                                                 path=d["path"], kept=p["keep"]["path"], size=p["size"],
                                                 detail=f"removed an identical copy ({p['why']})")
                        log(f"  - removed copy {d['path']} (kept {p['keep']['path']})")
                    out["removed"] += len(p["delete"])
                    out["freed"] += p["size"] * len(p["delete"])
                    self.invalidate_meta(p["scene_id"])
                except Exception as exc:  # noqa: BLE001 — one scene's trouble mustn't stop the rest
                    out["errors"].append({"scene_id": p["scene_id"], "error": str(exc)})
            out["applied"] = True
        self._last_copies = out
        return out

    def file_copies_step(self, log=None) -> dict | None:
        """The Sync / Ingest step: preview until approved, then remove."""
        st = self.copies_settings()
        if not st["on"]:
            return None
        r = self.remove_file_copies(apply=st["approved"], log=log)
        if log:
            if r["applied"]:
                log(f"same-file copies: removed {r['removed']} file(s)" + (f", {len(r['errors'])} failed" if r["errors"] else ""))
            elif r["files"]:
                log(f"same-file copies: found {r['files']} identical extra file(s) — approve once in "
                    "Activity → Maintenance to remove them on every Sync / Ingest")
        return {k: r[k] for k in ("applied", "files", "removed", "freed", "bytes")}

    def copies_status(self) -> dict:
        return {**self.copies_settings(), "last": self.__dict__.get("_last_copies"),
                "stash": str(self.cfg.stash.url or "").rstrip("/")}

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
        """Store triage features (and performers/studio) of 1★ scenes that are
        embedded and not yet remembered. Cheap when there's nothing new."""
        from ..tier_model import who_of

        mem = self.reject_memory()
        have = mem.fingerprints()
        rejects = [r for r in rows if r.get("tier") == "rejected" and r.get("fingerprint")]
        todo = [r for r in rejects if r["fingerprint"] not in have]
        # remembered before performers/studio were kept: fill them in while knowable
        stale = [r for r in rejects if r["fingerprint"] in have]
        if stale and None in mem.who([r["fingerprint"] for r in stale]):
            mem.fill_who({r["fingerprint"]: who_of(r) for r in stale})
        if not todo:
            return 0
        feats = self._scene_features(todo) if feats is None else feats
        return mem.add({r["fingerprint"]: (*feats[r["scene_id"]][:2], who_of(r)) for r in todo
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
        try:                                        # deleted scenes can't fail to embed any more
            from ..failures import failure_log_for

            failure_log_for(self.cfg).resolve_many(
                r.get("fingerprint") or path_key(r.get("path") or r["scene_id"]) for r in rows)
        except Exception:  # noqa: BLE001 — best-effort
            pass
        try:
            self.exposure().drop(gone)              # the board's record of them, too
        except Exception:  # noqa: BLE001 — best-effort
            pass
        cached = getattr(self, "_cat_cache", None)
        if cached:
            cached[1][:] = [r for r in cached[1] if r["scene_id"] not in gone]

    def delete_scenes(self, job, scene_ids: list[str], confirm: bool = False,
                      reason: str = "reject", keep_ids: set[str] | None = None,
                      delete_file: bool = True) -> dict:
        """Delete scenes from Stash, a few at a time — with their files unless
        `delete_file` is False (then the files stay on disk).

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
                                      delete_file=bool(delete_file), delete_generated=True)
            except Exception as exc:  # noqa: BLE001
                failed += [{"scene_id": r["scene_id"], "error": str(exc)} for r in ok_rows]
                if job is not None:
                    job.log(f"delete failed: {exc}")
                continue
            for r in ok_rows:
                if delete_file:
                    freed += int(r.get("size") or 0)
                deleted.append(r["scene_id"])
                self._log_scene("delete", r, reason=reason, size=r.get("size"), file_deleted=bool(delete_file),
                                before={"rating100": r["rating100"], "o_counter": r["o_counter"],
                                        "tier": r["tier"]},
                                detail=(f"file deleted ({reason})" if delete_file
                                        else f"removed from Stash, file kept ({reason})"))
                if job is not None:
                    job.log(("deleted " if delete_file else "removed from Stash (file kept) ") + str(r["path"]))
            self._drop_local(ok_rows)
            if job is not None:
                job.progress = {"done": min(i + CHUNK, len(ids)), "total": len(ids)}
        return {"deleted": len(deleted), "freed_bytes": freed, "refused": refused,
                "failed": failed, "ids": deleted, "files_deleted": bool(delete_file)}

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
        """The copy to keep: most pixels (8K VR beats 4K), then resolution class
        when dimensions are unknown, then bitrate, then file size; an existing
        keeper grade breaks remaining ties."""
        from ..tiers import RES_CLASSES

        rank = {c: i for i, c in enumerate(RES_CLASSES)}
        grade = {"legendaire": 4, "exceptionnelle": 3, "merveilleuse": 2, "upscale": 1}

        def key(r):
            q = r["quality"]
            pixels = (q.get("w") or 0) * (q.get("h") or 0)
            return (pixels, rank.get(q.get("res") or "", -1), q.get("mbps") or 0,
                    int(r.get("size") or 0), grade.get(r["tier"], 0))
        return max(rows, key=key)["scene_id"]

    @staticmethod
    def dupe_best_grade(rows: list[dict]) -> str | None:
        return LibraryMixin.dupe_best_tier(rows)[0]

    @staticmethod
    def dupe_best_tier(rows: list[dict]) -> tuple[str | None, str | None]:
        """The highest keeper tier anywhere in a duplicate group — from a copy's
        grade OR any tier tag it carries — and where it came from ('grade' /
        'tag'). Rejected / unreviewed never count."""
        for t in ("legendaire", "exceptionnelle", "merveilleuse", "upscale"):
            if any(r.get("tier") == t for r in rows):
                return t, "grade"
            if any(t in ((r.get("tag_state") or {}).get("present") or []) for r in rows):
                return t, "tag"
        return None, None

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
        out = self.dupe_queue_hide(out)            # already waiting in the delete queue
        result = {"groups": out, "accuracy": accuracy, "duration_diff": duration_diff,
                  "checked_at": _t.strftime("%Y-%m-%d %H:%M"), "ignored": len(ignored),
                  "reclaim": sum(g["reclaim"] for g in out)}
        if only_ids is None:
            self._dupe_cache = result
        return result

    def cached_duplicates(self) -> dict | None:
        return getattr(self, "_dupe_cache", None)

    def resolve_duplicate(self, job, keep_id: str, delete_ids: list[str], confirm: bool = False,
                          delete_file: bool = True) -> dict:
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
        rows = {s: self._cat_row(s, m) for s, m in fresh.items()}
        best, best_from = self.dupe_best_tier(list(rows.values()))
        order = ["upscale", "merveilleuse", "exceptionnelle", "legendaire"]
        kept_tier = tier_of(fresh[keep_id].get("rating100"), fresh[keep_id].get("o_counter"))
        kept_tags = (rows[keep_id].get("tag_state") or {}).get("present") or []
        carried = None
        # the winner takes the group's highest tier — from a grade or a tier tag —
        # and exactly that one tier tag (grade_scene: rating/O, tag, organized)
        if best and (kept_tier not in order or order.index(kept_tier) < order.index(best)
                     or kept_tags != [best]):
            self.grade_scene(keep_id, best, source="duplicate")
            carried = best
        res = self.delete_scenes(job, delete_ids, confirm=True, reason="duplicate",
                                 keep_ids={keep_id}, delete_file=delete_file)
        keep_row = self._cat_update_row(keep_id)
        self._log_scene("duplicate", keep_row, detail=(
            f"kept this copy, deleted {res['deleted']}"
            + (f", made it {best} (highest {best_from} in the group)" if carried else "")))
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
                self._watch_path(sid, row.get("path"))      # the renamer will move it
                self._log_scene("tag-sync", row, source="ingest", after={"tier": row["tier"]},
                                detail=f"re-tagged '{self.tier_tag_names()[row['tier']]}' after a Stash stage")
                fixed += 1
            except Exception as exc:  # noqa: BLE001
                log(f"couldn't re-tag scene {sid}: {exc}")
        if fixed:
            log(f"re-applied the tier tag on {fixed} scene(s) graded during the ingest")
        return fixed

    def _release_for_stash(self, embed_busy, log) -> None:
        """Free Peaks' big memory (caches, and the search index unless an embed
        pass is using it) before Stash starts decoding files."""
        from . import memwatch

        try:
            before = memwatch.rss_bytes()
            if not (embed_busy and embed_busy()):
                self.invalidate_index()
            self.shed_memory(drop_indexes=True)
            freed = max(0, before - memwatch.rss_bytes())
            if freed >= 64 * 1048576:
                log(f"released {freed / 1073741824:.1f} GB of Peaks memory for Stash's scan")
        except Exception:  # noqa: BLE001 — never block an ingest on housekeeping
            pass

    def run_ingest(self, job=None, embed_busy=None, paths: list[str] | None = None,
                   trigger: str = "manual") -> dict:
        log = job.log if job is not None else print
        for op in ("metadataScan", "findJob"):
            self.require_op(op)
        client = self.client()
        client.timeout = 120
        stages: dict[str, str] = {}
        record = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "running": True, "trigger": trigger,
                  "stage": "scan", "new": self.last_ingest().get("new") or [], "stages": stages}
        if trigger == "watch":
            log("started by the folder watch: new files in " + ", ".join(paths or []))
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

        # Stash's scan/identify/auto tag are the memory-heavy part (an ffmpeg per
        # file). Give back what Peaks holds and don't reload the index until our
        # own embed stage needs it.
        self._ingest_stash_busy = True
        self._release_for_stash(embed_busy, log)
        try:
            # 1. scan — Peaks' own scan options (Settings → Ingest), video phashes
            # always on; only fields this Stash version's scan input actually has
            opts = self.ingest_scan_options()
            scan_in = {k: v for k, v in opts.items() if client.input_has("ScanMetadataInput", k)}
            on = [label for k, (label, _) in self.INGEST_SCAN_FIELDS.items() if scan_in.get(k)]
            where = ""
            if paths and client.input_has("ScanMetadataInput", "paths"):
                scan_in["paths"] = list(paths)     # only the watched folders — quick
                where = " · only " + ", ".join(paths)
            log("1/5 scan: " + (", ".join(on) or "nothing extra generated") + where)
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
                self._ingest_stash_busy = False
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
            self._ingest_stash_busy = False
            # always leave a finished record, even when a stage failed
            record.update(running=False, stage="done", finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                          dupe_ids=sorted({r["scene_id"] for g in (dupes or {}).get("groups", [])
                                           for r in g["scenes"]}))
            self._write_ingest(record)
            self._cat_cache = None                 # new scenes → re-read the listing

        try:                                       # the same download attached twice
            c = self.file_copies_step(log)
            if c is not None:
                stages["same-file copies"] = (f"removed {c['removed']}" if c["applied"]
                                              else f"{c['files']} found — awaiting approval" if c["files"] else "none")
        except Exception as exc:  # noqa: BLE001 — never fail an ingest on this
            stages["same-file copies"] = f"skipped: {exc}"
        try:
            self.action_log().append("ingest", detail=f"{len(new)} new scene(s)", stages=stages)
        except Exception:  # noqa: BLE001
            pass
        if job is not None:
            job.progress = {"stage": "done"}
        log("done — " + " · ".join(f"{k}: {v}" for k, v in stages.items()))
        return {"new": len(new), "stages": stages, "trigger": trigger,
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
