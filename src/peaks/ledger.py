"""Safety net for library management: an append-only action log and daily grade
backups keyed by file fingerprint.

Everything Peaks changes in Stash (grades, restores, deletes, tag syncs, ingest
runs) is appended to ``actions.jsonl`` — one JSON object per line, never
rewritten. Grade backups snapshot every graded scene by *file fingerprint*, not
Stash scene id, so they survive a rescan, a renamer move or even a lost Stash
database: restoring matches files by fingerprint and re-applies their grades.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from pathlib import Path

from .tiers import tier_of


class ActionLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, action: str, **fields) -> dict:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "action": action, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def tail(self, n: int = 200) -> list[dict]:
        """The latest `n` entries, newest first. Unreadable lines are skipped."""
        if not self.path.is_file():
            return []
        last: deque[str] = deque(maxlen=max(1, int(n)))
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last.append(line)
        out = []
        for line in reversed(last):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


# --- grade backups -------------------------------------------------------------

BACKUP_KEEP = 14
_BACKUP_RE = re.compile(r"^grades-backup-(\d{8})(?:-(\d{6}))?\.json$")


def backup_entry(row: dict) -> dict:
    return {
        "rating100": row.get("rating100"),
        "o_counter": int(row.get("o_counter") or 0),
        "tier": row.get("tier") or tier_of(row.get("rating100"), row.get("o_counter")),
        "organized": bool(row.get("organized")),
        "tags": list(row.get("tags") or []),
        "title": row.get("title") or "",
        "path": row.get("path") or "",
        "scene_id": row.get("scene_id"),
    }


def graded(row: dict) -> bool:
    """Worth backing up: any rating or O-count at all."""
    return bool((row.get("rating100") or 0) > 0 or (row.get("o_counter") or 0) > 0)


def write_backup(folder: str | Path, rows: list[dict], stamp: str | None = None,
                 keep: int = BACKUP_KEEP) -> Path:
    """Snapshot every graded row with a fingerprint → grades-backup-<stamp>.json
    (atomic), then prune to the newest `keep` files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d")
    scenes = {r["fingerprint"]: backup_entry(r) for r in rows
              if r.get("fingerprint") and graded(r)}
    path = folder / f"grades-backup-{stamp}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": len(scenes), "scenes": scenes,
    }, ensure_ascii=False, indent=1))
    os.replace(tmp, path)
    for old in list_backups(folder)[keep:]:
        try:
            (folder / old["name"]).unlink()
        except OSError:
            pass
    return path


def list_backups(folder: str | Path) -> list[dict]:
    """Backups newest first: [{name, created, count, bytes}]."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    out = []
    for p in folder.iterdir():
        if not _BACKUP_RE.match(p.name):
            continue
        try:
            head = json.loads(p.read_text())
            out.append({"name": p.name, "created": head.get("created", ""),
                        "count": head.get("count", 0), "bytes": p.stat().st_size})
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda b: (b["created"], b["name"]), reverse=True)


def load_backup(folder: str | Path, name: str) -> dict:
    if not _BACKUP_RE.match(name or ""):
        raise ValueError(f"not a backup name: {name!r}")
    path = Path(folder) / name
    if not path.is_file():
        raise LookupError(f"backup not found: {name}")
    return json.loads(path.read_text())


def backup_diff(backup: dict, rows: list[dict]) -> dict:
    """What restoring `backup` would change. Rows are matched by fingerprint, so
    moved/rescanned files still line up. Returns
    {changes: [{scene_id, title, path, from:{…}, to:{…}}], missing, same}."""
    by_fp = {r["fingerprint"]: r for r in rows if r.get("fingerprint")}
    changes, missing, same = [], 0, 0
    for fp, want in (backup.get("scenes") or {}).items():
        row = by_fp.get(fp)
        if row is None:
            missing += 1
            continue
        cur_r = row.get("rating100") or None
        want_r = want.get("rating100") or None
        if cur_r == want_r and int(row.get("o_counter") or 0) == int(want.get("o_counter") or 0):
            same += 1
            continue
        changes.append({
            "scene_id": row["scene_id"], "fingerprint": fp,
            "title": row.get("title") or want.get("title") or "",
            "path": row.get("path") or "",
            "from": {"rating100": cur_r, "o_counter": int(row.get("o_counter") or 0),
                     "tier": row.get("tier")},
            "to": {"rating100": want_r, "o_counter": int(want.get("o_counter") or 0),
                   "tier": tier_of(want_r, want.get("o_counter"))},
        })
    return {"changes": changes, "missing": missing, "same": same}
