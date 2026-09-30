"""Backup & restore: everything Peaks knows, as weekly snapshot folders.

Each snapshot (``peaks-YYYY-MM-DD-HHMM/``) is complete on its own — settings and
the action log, taste labels, trained models and memories, playlists, the
failure log and the whole embedding cache — but a file that hasn't changed
since the previous snapshot is a *hard link* to it, not a copy (the
``rsync --link-dest`` idea). So a week usually costs only what's new, and
deleting an old snapshot frees only what no newer one shares.

Snapshots live in ``/data/.peaks-backups`` (your library share — on the array,
not the appdata cache drive; ``PEAKS_BACKUP_DIR`` overrides). The folder holds a
``.peaks-keep`` marker and the library cleanup never enters it.

A snapshot is written as ``<name>.partial`` and renamed only when complete; a
restore takes a safety snapshot of the current state first, puts every file
back and reloads Peaks — no restart.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

KEEP_MARKER = ".peaks-keep"
SNAP_RE = re.compile(r"^peaks-[A-Za-z0-9-]+$")
PARTS = ("state", "labels", "models", "collections", "failures", "embeddings")
SMALL_PARTS = tuple(p for p in PARTS if p != "embeddings")
# rebuildable caches in the models folder — not worth keeping
_MODEL_SKIP = re.compile(r"(\.scores\.(npy|json)$|^performer_directory\.json\.gz$|\.tmp(\.npz)?$|\.partial$)")
DEFAULTS = {"backup_on": True, "backup_day": 6, "backup_hour": 4, "backup_minute": 10, "backup_keep": 4}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class BackupMixin:

    # --- where things are ------------------------------------------------------------

    def backup_root(self) -> Path:
        env = os.environ.get("PEAKS_BACKUP_DIR")
        return Path(env) if env else self._cleanup_root() / ".peaks-backups"

    def backup_settings(self) -> dict:
        s = self._settings()
        return {k: s.get(k, v) for k, v in DEFAULTS.items()}

    def save_backup_settings(self, **kw) -> dict:
        s = dict(self._settings())
        for k, v in kw.items():
            if k in DEFAULTS and v is not None:
                s[k] = type(DEFAULTS[k])(v)
        s["backup_keep"] = max(1, int(s.get("backup_keep", 4)))
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s, indent=2) + "\n")
        self._settings_cache = s
        return self.backup_status()

    def _part_sources(self) -> dict[str, tuple[Path, str]]:
        """part -> (source path, 'file' | 'dir' | 'state')."""
        from ..failures import failure_log_for

        return {
            "state": (self._state_dir(), "state"),
            "labels": (Path(self.cfg.modeling.labels_path), "file"),
            "models": (Path(self.cfg.modeling.dir), "dir"),
            "collections": (self._collections_dir(), "dir"),
            "failures": (failure_log_for(self.cfg).path, "file"),
            "embeddings": (Path(self.cfg.embedding.cache_dir), "dir"),
        }

    def _part_files(self, part: str) -> list[tuple[Path, str]]:
        """[(absolute source file, relative path inside the part)]."""
        src, kind = self._part_sources()[part]
        out: list[tuple[Path, str]] = []
        if kind == "file":
            if src.is_file():
                out.append((src, src.name))
        elif kind == "state":
            # the settings folder's own small files (+ grade snapshots) — its
            # subfolders (cache, models…) are other parts
            if src.is_dir():
                # (labels.json / failures.json often live here too — they're their own parts)
                others = {Path(p).resolve() for part_, (p, k) in self._part_sources().items() if k == "file"}
                for f in sorted(src.iterdir()):
                    if f.is_file() and not f.name.endswith(".tmp") and f.resolve() not in others:
                        out.append((f, f.name))
                gb = src / "backups"
                if gb.is_dir():
                    for f in sorted(gb.rglob("*")):
                        if f.is_file():
                            out.append((f, str(f.relative_to(src))))
        elif src.is_dir():
            for f in sorted(src.rglob("*")):
                if not f.is_file() or f.name.endswith(".tmp"):
                    continue
                if part == "models" and _MODEL_SKIP.search(f.name):
                    continue
                out.append((f, str(f.relative_to(src))))
        return out

    # --- status ---------------------------------------------------------------------

    def _backup_writable(self) -> tuple[bool, str | None]:
        root = self.backup_root()
        base = root if root.exists() else root.parent
        if not base.is_dir():
            return False, f"{base} isn't there"
        if not os.access(base, os.W_OK):
            return False, f"Peaks can't write to {base} — set the Media path (/data) to read-write"
        return True, None

    def list_snapshots(self) -> list[dict]:
        root = self.backup_root()
        out = []
        if not root.is_dir():
            return out
        for d in root.iterdir():
            if not d.is_dir() or not SNAP_RE.match(d.name):
                continue
            try:
                man = json.loads((d / "manifest.json").read_text())
            except (OSError, ValueError):
                continue
            out.append({"name": d.name, "created": man.get("created"), "bytes": man.get("bytes", 0),
                        "new_bytes": man.get("new_bytes", 0), "files": man.get("files", 0),
                        "parts": man.get("parts", {}), "links_ok": man.get("links_ok", True),
                        "label": man.get("label", "")})
        out.sort(key=lambda x: float(x.get("created") or 0), reverse=True)     # newest first
        return out

    def backup_status(self) -> dict:
        ok, why = self._backup_writable()
        root = self.backup_root()
        free = None
        try:
            free = shutil.disk_usage(root if root.exists() else root.parent).free
        except OSError:
            pass
        snaps = self.list_snapshots()
        return {**self.backup_settings(), "root": str(root), "writable": ok, "reason": why, "free": free,
                "snapshots": snaps, "last": self.__dict__.get("_last_backup"),
                "links_ok": snaps[0]["links_ok"] if snaps else None, "due": self.backup_due()}

    def backup_due(self, now: float | None = None) -> bool:
        """Weekly: on the set day, from the set time, if no snapshot in 6 days."""
        s = self.backup_settings()
        if not s["backup_on"] or not self._backup_writable()[0]:
            return False
        t = time.localtime(now if now is not None else time.time())
        if t.tm_wday != int(s["backup_day"]) or (t.tm_hour, t.tm_min) < (int(s["backup_hour"]), int(s["backup_minute"])):
            return False
        snaps = [x for x in self.list_snapshots() if x.get("label") != "before restore"]
        if snaps and snaps[0].get("created") and (now or time.time()) - float(snaps[0]["created"]) < 6 * 86400:
            return False
        return True

    # --- making a snapshot ------------------------------------------------------------

    def run_backup(self, job=None, label: str = "", prune: bool = True) -> dict:
        """One snapshot of everything; unchanged files hard-linked to the last one."""
        log = job.log if job else (lambda *_: None)
        ok, why = self._backup_writable()
        if not ok:
            raise RuntimeError(why)
        root = self.backup_root()
        root.mkdir(parents=True, exist_ok=True)
        (root / KEEP_MARKER).touch(exist_ok=True)
        for stale in root.glob("*.partial"):          # an interrupted run
            shutil.rmtree(stale, ignore_errors=True)
        prev = next((root / s["name"] for s in self.list_snapshots()), None)
        name = "peaks-" + time.strftime("%Y-%m-%d-%H%M")
        while (root / name).exists():
            name += "b"
        work = root / (name + ".partial")
        work.mkdir()
        files = {p: self._part_files(p) for p in PARTS}
        total = sum(len(v) for v in files.values())
        done = linked = copied = new_bytes = all_bytes = 0
        links_ok = True
        parts_meta: dict[str, dict] = {}
        sums: dict[str, str] = {}
        if job is not None:
            job.progress = {"done": 0, "total": total}
        for part, items in files.items():
            pb = 0
            for src, rel in items:
                if job is not None and job.cancelled:
                    shutil.rmtree(work, ignore_errors=True)
                    raise RuntimeError("cancelled")
                dest = work / part / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    st = src.stat()
                except OSError:
                    continue                           # vanished mid-run (a cache being rewritten)
                old = prev / part / rel if prev else None
                did_link = False
                if links_ok and old is not None:
                    try:
                        ost = old.stat()
                        if ost.st_size == st.st_size and int(ost.st_mtime) == int(st.st_mtime):
                            os.link(old, dest)
                            did_link = True
                    except FileNotFoundError:
                        pass
                    except OSError:
                        links_ok = False           # this filesystem won't hard-link: copy from now on
                if did_link:
                    linked += 1
                else:
                    tmp = dest.with_name(dest.name + ".tmp")
                    shutil.copy2(src, tmp)
                    os.replace(tmp, dest)
                    copied += 1
                    new_bytes += st.st_size
                pb += st.st_size
                if part in SMALL_PARTS:
                    sums[f"{part}/{rel}"] = _sha(dest)
                done += 1
                if job is not None and done % 200 == 0:
                    job.progress = {"done": done, "total": total}
            parts_meta[part] = {"files": len(items), "bytes": pb, "src": str(self._part_sources()[part][0])}
            all_bytes += pb
            log(f"{part}: {len(items):,} files")
        from .. import __version__

        man = {"created": time.time(), "version": __version__, "label": label, "model": self._model_name(),
               "parts": parts_meta, "files": total, "bytes": all_bytes, "new_bytes": new_bytes,
               "linked": linked, "copied": copied, "links_ok": links_ok, "sha256": sums}
        (work / "manifest.json").write_text(json.dumps(man, indent=1))
        work.rename(root / name)
        if prune:
            self.prune_backups()
        self._last_backup = {"name": name, "at": time.time(), "new_bytes": new_bytes, "linked": linked,
                             "copied": copied, "links_ok": links_ok}
        log(f"snapshot {name}: {total:,} files, {copied:,} new/changed, {linked:,} unchanged (linked)")
        return {**self._last_backup, "files": total, "bytes": all_bytes}

    def prune_backups(self) -> int:
        keep = int(self.backup_settings()["backup_keep"])
        snaps = self.list_snapshots()
        gone = 0
        for s in snaps[keep:]:
            d = self.backup_root() / s["name"]
            if SNAP_RE.match(d.name) and d.parent == self.backup_root():
                shutil.rmtree(d, ignore_errors=True)
                gone += 1
        return gone

    # --- restoring --------------------------------------------------------------------

    def _snapshot_dir(self, name: str) -> Path:
        if not SNAP_RE.match(name or ""):
            raise ValueError("not a snapshot name")
        d = self.backup_root() / name
        if not (d / "manifest.json").is_file():
            raise FileNotFoundError(f"no snapshot {name}")
        return d

    def backup_restore_snapshot_preview(self, name: str) -> dict:
        d = self._snapshot_dir(name)
        man = json.loads((d / "manifest.json").read_text())
        parts = {}
        for p in PARTS:
            parts[p] = {"snapshot": (man.get("parts", {}).get(p) or {}).get("files", 0),
                        "now": len(self._part_files(p))}
        return {"name": name, "created": man.get("created"), "parts": parts, "bytes": man.get("bytes", 0)}

    def backup_restore(self, job, name: str, confirm: bool = False, grades: bool = False) -> dict:
        """Put a snapshot back: safety snapshot of the current state first, then
        every part copied into place (checksums verified), then Peaks reloads."""
        if not confirm:
            raise PermissionError("restoring replaces what Peaks knows now — confirm first")
        d = self._snapshot_dir(name)
        man = json.loads((d / "manifest.json").read_text())
        log = job.log if job else (lambda *_: None)
        log("safety snapshot of the current state first…")
        safety = self.run_backup(job=None, label="before restore", prune=False)["name"]
        sources = self._part_sources()
        restored = 0
        for part in PARTS:
            pdir = d / part
            if not pdir.is_dir():
                continue
            src, kind = sources[part]
            base = src.parent if kind == "file" else src
            have = {rel for _, rel in self._part_files(part)}
            for f in sorted(pdir.rglob("*")):
                if not f.is_file():
                    continue
                rel = str(f.relative_to(pdir))
                dest = base / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".restore-tmp")
                shutil.copy2(f, tmp)
                want = man.get("sha256", {}).get(f"{part}/{rel}")
                if want and _sha(tmp) != want:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"checksum mismatch for {part}/{rel} — the snapshot is damaged")
                os.replace(tmp, dest)
                have.discard(rel)
                restored += 1
            # small parts come back exactly (files the snapshot didn't have go);
            # extra embeddings (newer scenes) are simply kept
            if part in ("models", "collections"):
                for rel in have:
                    (base / rel).unlink(missing_ok=True)
        log(f"restored {restored:,} files from {name}")
        self._reload_after_restore()
        out = {"restored": restored, "safety": safety, "name": name}
        if grades:
            snaps = self.list_grade_backups()
            if snaps:
                out["grades"] = self.backup_restore_apply(job, snaps[0]["name"], confirm=True)
        return out

    def _reload_after_restore(self) -> None:
        """Forget everything read from disk so the restored files take effect."""
        self._settings_cache = None
        try:
            self.invalidate_index()
        except Exception:  # noqa: BLE001
            pass
        self._invalidate_taste_caches()
        self._board_score_cache.clear()
        self.__dict__.setdefault("_score_meta", {}).clear()
        with self._taste_lock:
            self._taste.clear()
        for attr in ("_tier_state", "_tier_preds", "_cat_cache", "_who_cache", "_perf_dir_cache",
                     "_perf_by_id", "_exposure", "_scale_cache", "_vis_feat_cache"):
            self.__dict__.pop(attr, None)
        self._perf_stats_cache = None

    # --- the small part as one file (off-box copies) -------------------------------------

    def export_small_backup(self, name: str) -> Path:
        """A .tar.gz of a snapshot without its embeddings (MB-sized)."""
        import tarfile
        import tempfile

        d = self._snapshot_dir(name)
        out = Path(tempfile.gettempdir()) / f"{name}-small.tar.gz"
        with tarfile.open(out, "w:gz") as tar:
            tar.add(d / "manifest.json", arcname=f"{name}/manifest.json")
            for part in SMALL_PARTS:
                if (d / part).is_dir():
                    tar.add(d / part, arcname=f"{name}/{part}")
        return out

    def import_small_backup(self, data: bytes) -> dict:
        """Unpack an uploaded small backup as a new snapshot (validated: one
        snapshot folder, a manifest, only known parts, no absolute or ../ paths)."""
        import io
        import tarfile

        ok, why = self._backup_writable()
        if not ok:
            raise RuntimeError(why)
        try:
            tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        except tarfile.TarError as exc:
            raise ValueError(f"not a Peaks backup: {exc}")
        members = tar.getmembers()
        tops = {m.name.split("/", 1)[0] for m in members}
        if len(tops) != 1:
            raise ValueError("not a Peaks backup (expected one snapshot folder)")
        top = tops.pop()
        if not SNAP_RE.match(top):
            raise ValueError("not a Peaks backup (unexpected folder name)")
        names = {m.name for m in members}
        if f"{top}/manifest.json" not in names:
            raise ValueError("not a Peaks backup (no manifest)")
        for m in members:
            parts = m.name.split("/")
            if m.name.startswith("/") or ".." in parts or not (m.isfile() or m.isdir()):
                raise ValueError(f"refused: unsafe entry {m.name!r}")
            if len(parts) > 1 and parts[1] not in (*PARTS, "manifest.json"):
                raise ValueError(f"refused: unknown part {parts[1]!r}")
        root = self.backup_root()
        root.mkdir(parents=True, exist_ok=True)
        (root / KEEP_MARKER).touch(exist_ok=True)
        name = "peaks-upload-" + time.strftime("%Y-%m-%d-%H%M")
        while (root / name).exists():
            name += "b"
        work = root / (name + ".partial")
        work.mkdir()
        for m in members:
            if not m.isfile():
                continue
            rel = m.name.split("/", 1)[1]
            dest = work / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        work.rename(root / name)
        return {"name": name}
