"""One small SQLite file for Peaks' own state.

The per-scene records (verdicts, rulings, criteria, performer criteria, the
megaboard's exposure, sense failures) and the append-only histories (the
action log, settle history, moment checks, pick feedback) used to live in JSON
files that were rewritten whole, fsync'd, on every change and parsed whole on
every read. Here each is a table row: a grade upserts one row, a timeline asks
for one scene's lines by index, and nothing is parsed that isn't asked for.

Two tables carry everything:

    maps(name, key, value)            a dict store: one row per key
    logs(id, name, at, scene_id, value)  an append-only log: one row per line

The first time a store is opened, its old JSON/JSONL file (if any) is imported
and renamed ``<file>.migrated`` — that's the backup. ``python -m peaks.store
export <state dir>`` writes every store back out as the JSON files it came
from, for a rollback to an older Peaks.

WAL mode with ``synchronous=NORMAL``: a commit is a sequential append to the
WAL, durable across a process crash; only a power loss can lose the last
moments, which is the same promise the fsync'd rewrites gave in practice.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_OPEN: dict[str, "Store"] = {}
_OPEN_LOCK = threading.Lock()


def open_store(path: str | Path) -> "Store":
    """One Store per database file per process (several services may share one)."""
    key = str(Path(path).resolve())
    with _OPEN_LOCK:
        st = _OPEN.get(key)
        if st is None:
            st = _OPEN[key] = Store(path)
        return st


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._vers: dict[str, int] = {}         # name → bumps since open (memo keys)
        self._map_cache: dict[str, dict] = {}    # name → the whole map (readers share it)
        self._done: set[str] = set()             # migrations checked this process

    # --- plumbing ---------------------------------------------------------------------------

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            c = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30, isolation_level=None)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=30000")
            c.executescript(
                "CREATE TABLE IF NOT EXISTS maps(name TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,"
                " updated REAL NOT NULL, PRIMARY KEY(name, key)) WITHOUT ROWID;"
                "CREATE TABLE IF NOT EXISTS logs(id INTEGER PRIMARY KEY, name TEXT NOT NULL, at REAL NOT NULL,"
                " scene_id TEXT, value TEXT NOT NULL);"
                "CREATE INDEX IF NOT EXISTS logs_name ON logs(name, id);"
                "CREATE INDEX IF NOT EXISTS logs_scene ON logs(name, scene_id);"
                "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            self._conn = c
        return self._conn

    def _tx(self):
        return _Tx(self)

    def version(self, name: str) -> int:
        """Changes whenever `name` is written through this process (memo keys)."""
        return self._vers.get(name, 0)

    def _bump(self, name: str) -> None:
        self._vers[name] = self._vers.get(name, 0) + 1

    def close(self) -> None:
        """Close (the WAL is folded into the file); the next call reopens."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._map_cache.clear()
            self._done.clear()

    def snapshot_to(self, dest: str | Path) -> None:
        """A consistent copy of the database (a live WAL-mode file can't be
        copied byte for byte): SQLite's own online backup."""
        with self._lock:
            out = sqlite3.connect(str(dest))
            try:
                self._db().backup(out)
            finally:
                out.close()

    # --- maps -------------------------------------------------------------------------------

    def map_all(self, name: str) -> dict:
        """The whole map. Shared with other readers: treat it as read-only."""
        with self._lock:
            d = self._map_cache.get(name)
            if d is None:
                rows = self._db().execute("SELECT key, value FROM maps WHERE name=?", (name,)).fetchall()
                d = self._map_cache[name] = {k: json.loads(v) for k, v in rows}
            return d

    def map_get(self, name: str, key: str, default=None):
        return self.map_all(name).get(str(key), default)

    def map_set(self, name: str, key: str, value) -> None:
        self.map_set_many(name, {key: value})

    def map_set_many(self, name: str, items: dict, delete=()) -> None:
        """Upsert `items` and delete `delete`, in one transaction."""
        now = time.time()
        with self._lock, self._tx():
            db = self._db()
            if items:
                db.executemany("INSERT INTO maps(name, key, value, updated) VALUES(?,?,?,?)"
                               " ON CONFLICT(name, key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
                               [(name, str(k), json.dumps(v, separators=(",", ":")), now) for k, v in items.items()])
            gone = [str(k) for k in delete]
            if gone:
                db.executemany("DELETE FROM maps WHERE name=? AND key=?", [(name, k) for k in gone])
            d = self._map_cache.get(name)
            if d is not None:
                for k, v in items.items():
                    d[str(k)] = v
                for k in gone:
                    d.pop(k, None)
            self._bump(name)

    def map_delete(self, name: str, *keys) -> None:
        self.map_set_many(name, {}, delete=keys)

    def map_replace(self, name: str, items: dict) -> None:
        """The whole map becomes `items` (a wholesale edit, or a migration)."""
        now = time.time()
        with self._lock, self._tx():
            db = self._db()
            db.execute("DELETE FROM maps WHERE name=?", (name,))
            db.executemany("INSERT INTO maps(name, key, value, updated) VALUES(?,?,?,?)",
                           [(name, str(k), json.dumps(v, separators=(",", ":")), now) for k, v in items.items()])
            self._map_cache[name] = {str(k): v for k, v in items.items()}
            self._bump(name)

    def map_edit(self, name: str):
        """``with store.map_edit("rulings") as d:`` — edit a private copy; the
        rows that changed are written when the block ends."""
        return _MapEdit(self, name)

    # --- logs -------------------------------------------------------------------------------

    def log_append(self, name: str, rec: dict, at: float | None = None, scene_id=None) -> dict:
        with self._lock, self._tx():
            self._db().execute("INSERT INTO logs(name, at, scene_id, value) VALUES(?,?,?,?)",
                               (name, float(at if at is not None else rec.get("at") or time.time()),
                                None if scene_id is None else str(scene_id),
                                json.dumps(rec, ensure_ascii=False, separators=(",", ":"))))
            self._bump(name)
        return rec

    def log_read(self, name: str, limit: int | None = None, newest_first: bool = False,
                 scene_id=None, contains: str | None = None) -> list[dict]:
        """Records of one log; `scene_id` uses the index, `contains` is a cheap
        text prefilter on the stored JSON (e.g. '"action":"delete"')."""
        q, args = "SELECT value FROM logs WHERE name=?", [name]
        if scene_id is not None:
            q += " AND scene_id=?"
            args.append(str(scene_id))
        if contains:
            q += " AND instr(value, ?) > 0"
            args.append(contains)
        q += " ORDER BY id " + ("DESC" if newest_first else "ASC")
        if limit is not None:
            q += " LIMIT ?"
            args.append(int(limit))
        with self._lock:
            rows = self._db().execute(q, args).fetchall()
        return [json.loads(v) for (v,) in rows]

    def log_tail(self, name: str, n: int) -> list[dict]:
        """The latest `n`, newest first."""
        return self.log_read(name, limit=n, newest_first=True)

    def log_count(self, name: str) -> int:
        with self._lock:
            return int(self._db().execute("SELECT COUNT(*) FROM logs WHERE name=?", (name,)).fetchone()[0])

    def log_clear(self, name: str) -> None:
        with self._lock, self._tx():
            self._db().execute("DELETE FROM logs WHERE name=?", (name,))
            self._bump(name)

    def log_iter_lines(self, name: str):
        """Every record as a JSON line, oldest first (a download)."""
        with self._lock:
            rows = self._db().execute("SELECT value FROM logs WHERE name=? ORDER BY id", (name,)).fetchall()
        for (v,) in rows:
            yield v + "\n"

    # --- migration from the JSON files ---------------------------------------------------------

    def _migrated(self, name: str) -> bool:
        if name in self._done:
            return True
        with self._lock:
            r = self._db().execute("SELECT value FROM meta WHERE key=?", ("migrated:" + name,)).fetchone()
        if r is not None:
            self._done.add(name)
        return r is not None

    def _mark(self, name: str, source: Path | None) -> None:
        with self._lock:
            self._db().execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)",
                               ("migrated:" + name, json.dumps({"from": str(source) if source else None, "at": time.time()})))
            self._done.add(name)

    def migrate_map(self, name: str, path: str | Path, unwrap: str | None = None) -> bool:
        """Import a ``{key: value}`` JSON file (or its `unwrap` member) once, then
        rename it ``.migrated``. Returns whether anything was imported."""
        if self._migrated(name):
            return False
        p = Path(path)
        got = False
        try:
            d = json.loads(p.read_text())
            if unwrap:
                d = d.get(unwrap) or {}
            if isinstance(d, dict):
                self.map_replace(name, d)
                got = True
        except (OSError, ValueError, AttributeError):
            pass
        self._mark(name, p if got else None)
        if got:
            _set_aside(p)
        return got

    def migrate_log(self, name: str, path: str | Path, scene_key: str = "scene_id", at_key: str = "at") -> bool:
        if self._migrated(name):
            return False
        p = Path(path)
        recs = []
        try:
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    try:
                        recs.append(json.loads(ln))
                    except ValueError:
                        continue
        except OSError:
            recs = None
        if recs:
            with self._lock, self._tx():
                self._db().executemany(
                    "INSERT INTO logs(name, at, scene_id, value) VALUES(?,?,?,?)",
                    [(name, _epoch(r.get(at_key)), None if r.get(scene_key) is None else str(r.get(scene_key)),
                      json.dumps(r, ensure_ascii=False, separators=(",", ":"))) for r in recs])
                self._bump(name)
        self._mark(name, p if recs else None)
        if recs:
            _set_aside(p)
        return bool(recs)

    # --- export (rollback) ----------------------------------------------------------------------

    def export(self, folder: str | Path) -> list[str]:
        """Write every store back as the JSON file it came from, under `folder`."""
        from . import atomic

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for name, fname, wrap in MAP_FILES:
            d = self.map_all(name)
            if d:
                atomic.write_json(folder / fname, {wrap: d} if wrap else d, indent=1)
                out.append(fname)
        for name, fname in LOG_FILES:
            lines = list(self.log_iter_lines(name))
            if lines:
                atomic.write_text(folder / fname, "".join(lines))
                out.append(fname)
        return out


# name → the file it replaces (relative to the state dir, or the models dir for the two below)
MAP_FILES = (
    ("verdicts", "verdicts.json", None),
    ("rulings", "rulings.json", None),
    ("criteria", "criteria.json", None),
    ("performer_criteria", "performer_criteria.json", None),
    ("sense_failures", "sense_failures.json", None),
    ("board_exposure", "board_exposure.json", "scenes"),
)
LOG_FILES = (
    ("actions", "actions.jsonl"),
    ("settle_history", "settle_history.jsonl"),
    ("moment_checks", "moment_checks.jsonl"),
    ("pick_feedback", "pick_feedback.jsonl"),
)


class _Tx:
    def __init__(self, st: Store):
        self.st = st

    def __enter__(self):
        self.st._db().execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, et, ev, tb):
        self.st._db().execute("ROLLBACK" if et else "COMMIT")
        return False


class _MapEdit:
    def __init__(self, st: Store, name: str):
        self.st, self.name = st, name

    def __enter__(self) -> dict:
        self.st._lock.acquire()
        self.before = self.st.map_all(self.name)
        self.d = {k: v for k, v in self.before.items()}
        return self.d

    def __exit__(self, et, ev, tb):
        try:
            if et is None:
                changed = {k: v for k, v in self.d.items() if k not in self.before or self.before[k] != v}
                gone = [k for k in self.before if k not in self.d]
                if changed or gone:
                    self.st.map_set_many(self.name, changed, delete=gone)
        finally:
            self.st._lock.release()
        return False


def _epoch(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):                     # the action log's "YYYY-MM-DDTHH:MM:SS"
        try:
            return time.mktime(time.strptime(v[:19], "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            pass
    return time.time()


def _set_aside(p: Path) -> None:
    try:
        p.rename(p.with_name(p.name + ".migrated"))
    except OSError:
        pass


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Peaks state store")
    ap.add_argument("cmd", choices=["export"])
    ap.add_argument("state_dir", help="the folder that holds peaks.db")
    ap.add_argument("--to", default=None, help="where to write the JSON files (default: the state dir)")
    a = ap.parse_args(argv)
    st = Store(Path(a.state_dir) / "peaks.db")
    for f in st.export(a.to or a.state_dir):
        print("wrote", f)


if __name__ == "__main__":
    main()
