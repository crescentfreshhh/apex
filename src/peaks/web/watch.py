"""Auto-ingest: watch the download folder(s) and ingest new files on their own.

Once a minute the scheduler lists the watched folders (video files only — temp
and hidden files are skipped). A file counts once its size and modified time
have stayed put for `watch_settle` seconds; when there are settled new files and
the folder has been quiet (nothing new, nothing growing) for `watch_quiet`
seconds, one Ingest runs for the lot, scanning only the watched folders.

Polling, not file-system events: on Unraid, writes through /mnt/user from
another container or the host often never reach a bind-mounted container's
inotify. One folder listing a minute is cheap and wakes only that folder's disk.

State (`models/watch.json`): the files already handled, the ones being watched
settle, and the last auto-ingest. Turning the watch on (or changing folders)
records what's already there as handled — only what arrives afterwards counts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".m4v", ".webm", ".ts", ".flv", ".mpg", ".mpeg"}
TEMP_EXT = {".part", ".!qb", ".tmp", ".crdownload", ".partial"}
RETRY_SECONDS = 600                                # a failed auto-ingest tries again after this
DEFAULTS = {"watch_on": False, "watch_paths": [], "watch_settle": 120, "watch_quiet": 180}


def _is_candidate(name: str) -> bool:
    if name.startswith("."):                       # hidden, rsync's .name.XXXXXX temp files
        return False
    low = name.lower()
    if any(low.endswith(x) for x in TEMP_EXT):
        return False
    return os.path.splitext(low)[1] in VIDEO_EXT


class WatchMixin:
    # --- settings -------------------------------------------------------------

    def watch_settings(self) -> dict:
        s = self._settings()
        out = {k: s.get(k, v) for k, v in DEFAULTS.items()}
        out["watch_paths"] = [str(p) for p in (out["watch_paths"] or []) if str(p).strip()]
        return out

    def save_watch_settings(self, on: bool | None = None, paths: list[str] | None = None,
                            settle: int | None = None, quiet: int | None = None) -> dict:
        s = dict(self._settings())
        if on is not None:
            if on and not s.get("watch_on"):      # switched on: start from what's there now
                with self._watch_lock():
                    self._watch_state().pop("roots", None)
                    self._watch_save()
            s["watch_on"] = bool(on)
        if paths is not None:
            clean = []
            for p in paths:
                p = str(p).strip().rstrip("/") or "/"
                if p and p not in clean:
                    clean.append(p)
            s["watch_paths"] = clean
        if settle is not None:
            s["watch_settle"] = max(30, min(3600, int(settle)))
        if quiet is not None:
            s["watch_quiet"] = max(0, min(3600, int(quiet)))
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s, indent=2) + "\n")
        self._settings_cache = s
        self.watch_tick()                          # turned on / new folders: baseline right now
        return self.watch_status()

    # --- state ------------------------------------------------------------------

    def _watch_file(self) -> Path:
        return Path(self.cfg.modeling.dir) / "watch.json"

    def _watch_state(self) -> dict:
        st = self.__dict__.get("_watch_mem")
        if st is None:
            try:
                st = json.loads(self._watch_file().read_text())
            except (OSError, ValueError):
                st = {}
            st.setdefault("done", [])
            st.setdefault("seen", {})
            st.setdefault("last_change", 0.0)
            self._watch_mem = st
        return st

    def _watch_save(self) -> None:
        p = self._watch_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._watch_state()))
        os.replace(tmp, p)

    # --- the minute tick ----------------------------------------------------------

    def _watch_list(self, roots: list[str]) -> tuple[dict[str, tuple[int, float]], list[str]]:
        """{path: (size, mtime)} of candidate videos under `roots`, and the
        roots that couldn't be read."""
        files: dict[str, tuple[int, float]] = {}
        bad = []
        for root in roots:
            if not os.path.isdir(root):
                bad.append(root)
                continue
            for d, dirs, names in os.walk(root, onerror=lambda e: None):
                dirs[:] = [x for x in dirs if not x.startswith(".")]
                for n in names:
                    if not _is_candidate(n):
                        continue
                    p = os.path.join(d, n)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    files[p] = (st.st_size, st.st_mtime)
        return files, bad

    def _watch_lock(self):
        import threading

        return self.__dict__.setdefault("_watch_lk", threading.RLock())

    def watch_tick(self, now: float | None = None, force: bool = False) -> dict | None:
        """Look at the folders once. Returns {paths, files} when an Ingest should
        start now (`force` skips the quiet wait — never the settle), else None."""
        with self._watch_lock():
            return self._watch_tick(now, force)

    def _watch_tick(self, now: float | None, force: bool) -> dict | None:
        now = time.time() if now is None else now
        cfg = self.watch_settings()
        roots = cfg["watch_paths"]
        if not cfg["watch_on"] or not roots:
            return None
        st = self._watch_state()
        files, bad = self._watch_list(roots)
        st["checked"], st["unreadable"] = now, bad
        if st.get("roots") != roots:               # first look (or new folders): baseline
            st.update(roots=roots, done=sorted(files), seen={}, last_change=0.0, baseline_at=now)
            self._watch_save()
            return None
        done = set(st["done"]) & set(files)        # forget what moved away / was deleted
        seen = {p: v for p, v in st["seen"].items() if p in files and p not in done}
        for p, (size, mtime) in files.items():
            if p in done:
                continue
            v = seen.get(p)
            if v is None or v["size"] != size or v["mtime"] != mtime:
                seen[p] = {"size": size, "mtime": mtime, "since": now}
                st["last_change"] = now
        settled = sorted(p for p, v in seen.items() if now - v["since"] >= cfg["watch_settle"])
        settling = len(seen) - len(settled)
        st.update(done=sorted(done), seen=seen, settled=len(settled), settling=settling)
        go = bool(settled) and settling == 0 and (force or now - st["last_change"] >= cfg["watch_quiet"])
        if go and not force and now < float(st.get("retry_after") or 0):   # a failed batch waits
            go = False
        self._watch_save()
        return {"paths": roots, "files": settled} if go else None

    def watch_started(self, files: list[str], now: float | None = None) -> None:
        """An Ingest took these files: they're handled."""
        with self._watch_lock():
            self._watch_took(files, now)

    def _watch_took(self, files: list[str], now: float | None) -> None:
        st = self._watch_state()
        st["done"] = sorted(set(st["done"]) | set(files))
        st["seen"] = {p: v for p, v in st["seen"].items() if p not in set(files)}
        st["settled"] = 0
        st["last_run"] = {"at": time.time() if now is None else now, "files": len(files),
                          "sample": [os.path.basename(f) for f in files[:5]]}
        self._watch_save()

    def watch_failed(self, files: list[str], error: str, now: float | None = None) -> None:
        """The ingest for these files failed: they go back to waiting (already
        settled) and the batch is retried after RETRY_SECONDS."""
        now = time.time() if now is None else now
        with self._watch_lock():
            st = self._watch_state()
            st["done"] = sorted(set(st["done"]) - set(files))
            for f in files:
                try:
                    size, mtime = os.path.getsize(f), os.path.getmtime(f)
                except OSError:
                    continue                           # gone meanwhile: nothing to retry
                st["seen"][f] = {"size": size, "mtime": mtime, "since": now - 10 * 365 * 86400}
            st.update(last_error={"at": now, "error": str(error)[:300], "files": len(files)},
                      retry_after=now + RETRY_SECONDS)
            self._watch_save()

    def watch_succeeded(self) -> None:
        with self._watch_lock():
            st = self._watch_state()
            if "last_error" in st or "retry_after" in st:
                st.pop("last_error", None)
                st.pop("retry_after", None)
                self._watch_save()

    # --- embeds owed: new scenes an ingest couldn't embed -------------------------------

    def _followup_path(self) -> Path:
        return self._state_dir() / "embed_followup.json"

    def embed_followup(self) -> list[str]:
        try:
            return sorted(set(json.loads(self._followup_path().read_text())), key=lambda x: int(x) if x.isdigit() else 0)
        except (OSError, ValueError):
            return []

    def _save_followup(self, ids) -> None:
        p = self._followup_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(set(ids))))
        os.replace(tmp, p)

    def add_embed_followup(self, ids) -> None:
        with self._watch_lock():
            self._save_followup(set(self.embed_followup()) | {str(i) for i in ids})

    def run_embed_followup(self, job=None) -> dict:
        """Embed the scenes an ingest had to leave (DINO, then CLIP when you use
        text search). They're cleared whether or not each embeds — failures go
        to the failure log as with any embed, so nothing loops."""
        ids = self.embed_followup()
        if not ids:
            return {"embedded": 0}
        if job is not None:
            job.log(f"embedding {len(ids)} new scene(s) an ingest left for later")
        out = self.embed_new_scenes(job, ids)
        with self._watch_lock():
            self._save_followup(set(self.embed_followup()) - set(ids))
        return out

    def _clip_too(self) -> bool:
        """Also embed for text search? Only when you have a CLIP index and the
        default model isn't CLIP already."""
        from ..embedding import canonical_name

        try:
            return canonical_name(self.cfg.embedding.model) != "clip" and self.has_clip_index()
        except Exception:  # noqa: BLE001
            return False

    def embed_new_scenes(self, job, ids) -> dict:
        """The default model, then CLIP (text search) when it's in use — only
        these scenes. Returns run stats plus a one-line summary."""
        ids = {str(i) for i in ids}
        st = self.run_embed(job, scene_ids=ids)
        line = f"{st.get('embedded', 0)} embedded, {st.get('failed', 0)} failed"
        if self._clip_too() and not (job is not None and job.cancelled):
            if job is not None:
                job.log("  + text search (CLIP) for the same scenes")
            c = self.run_embed(job, scene_ids=ids, model="clip")
            st["clip"] = c
            line += f" · text search: {c.get('embedded', 0)} embedded"
        st["summary"] = line
        return st

    # --- what the UI shows -----------------------------------------------------------

    def watch_status(self) -> dict:
        cfg = self.watch_settings()
        st = self._watch_state()
        last = self.last_ingest() if hasattr(self, "last_ingest") else {}
        auto = last if (last or {}).get("trigger") == "watch" else None
        return {**cfg, "checked": st.get("checked"), "settling": st.get("settling", 0),
                "settled": st.get("settled", 0), "unreadable": st.get("unreadable", []),
                "baseline_at": st.get("baseline_at"), "last_run": st.get("last_run"),
                "last_ingest": auto, "last_error": st.get("last_error"),
                "retry_after": st.get("retry_after"), "embed_owed": len(self.embed_followup()),
                "ingest_warnings": self.ingest_warnings(auto),
                "warnings": self.watch_warnings(cfg["watch_paths"])}

    @staticmethod
    def ingest_warnings(rec: dict | None) -> list[str]:
        """Steps the last auto-ingest skipped that you'd want to know about."""
        stages = (rec or {}).get("stages") or {}
        out = []
        ident = str(stages.get("identify") or "")
        if ident.startswith("skipped") and "identify sources" in ident:
            out.append("identify was skipped — no identify sources are saved on Stash's Tasks page")
        emb = str(stages.get("embed") or "")
        if "in scope" in emb or emb.startswith("deferred"):
            out.append("embed: " + emb)
        return out

    def watch_warnings(self, paths: list[str]) -> list[str]:
        out = []
        libs = None
        try:
            libs = self.client().library_paths()
        except Exception:  # noqa: BLE001 — Stash down / old: skip the coverage check
            libs = None
        for p in paths:
            if not os.path.isdir(p):
                out.append(f"Peaks can't see {p} — check the folder name and the /data mapping")
            elif not os.access(p, os.R_OK):
                out.append(f"Peaks can't read {p}")
            elif libs and not any(p == lib or p.startswith(lib.rstrip("/") + "/") for lib in libs):
                out.append(f"{p} isn't inside any Stash library folder — Stash's scan won't pick it up")
        return out

    def watch_dirs(self, under: str | None = None) -> dict:
        """Sub-folders of `under` (default: the library root) for the picker."""
        base = Path(under or self._cleanup_root())
        try:
            dirs = sorted((str(d) for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")),
                          key=str.lower)
        except OSError:
            dirs = []
        return {"under": str(base), "parent": str(base.parent) if base.parent != base else None, "dirs": dirs[:500]}
