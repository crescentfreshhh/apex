"""Saving state files safely, when several requests save at once.

The old pattern — write `<file>.tmp`, then rename it over `<file>` — survives a
crash, but every save used the *same* temp name. Two saves that overlapped
(a rating while the megaboard records what you watched, two tabs…) both
opened that one temp file, the shorter write landed over the start of the
longer one, and the result — valid JSON followed by the other save's tail —
was renamed into place: "JSONDecodeError: Extra data".

Here every save gets its own temp file, is flushed to disk, then replaces
the file in one step; saves of the same file inside Peaks also wait for each
other. `edit_lock` holds a file across load → change → save so two edits
can't drop each other's change. `read_json` repairs the overlap's signature
(a complete document followed by leftovers) and refuses anything else rather
than pretending a damaged file is empty.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

_LOCKS: dict[str, threading.RLock] = {}
_GUARD = threading.Lock()
_REPAIRS: list[dict] = []          # what read_json fixed, for Activity


def _lock(path) -> threading.RLock:
    key = os.path.abspath(str(path))
    with _GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


@contextlib.contextmanager
def edit_lock(path):
    """Hold `path` for a whole load → change → save (other saves wait)."""
    with _lock(path):
        yield


@contextlib.contextmanager
def temp_for(path, suffix: str = ".tmp"):
    """A unique temp path next to `path`; on a clean exit it replaces `path`
    in one step, on an error it's removed. For writers that need a filename
    (numpy, pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock(path):
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=suffix, dir=path.parent)
        os.close(fd)
        try:
            yield Path(tmp)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


def write_bytes(path, data: bytes, fsync: bool = True) -> Path:
    path = Path(path)
    with temp_for(path) as tmp:
        with open(tmp, "wb") as fh:
            fh.write(data)
            if fsync:
                fh.flush()
                os.fsync(fh.fileno())
    return path


def write_text(path, text: str, fsync: bool = True) -> Path:
    return write_bytes(path, text.encode("utf-8"), fsync=fsync)


def write_json(path, obj, indent: int | None = None, fsync: bool = True, trailing_newline: bool = False) -> Path:
    text = json.dumps(obj, indent=indent)
    return write_text(path, text + ("\n" if trailing_newline else ""), fsync=fsync)


class DamagedFile(ValueError):
    """A state file Peaks can't read — kept as it is, never overwritten."""


def read_json(path, default=None, *, repair: bool = True, set_aside: bool = False):
    """The JSON in `path` (`default` if it doesn't exist).

    A file holding a complete document followed by leftovers — what two
    overlapping saves used to leave — is repaired: the first document is a
    whole, valid snapshot, so it's kept, the damaged file is set aside as
    `<name>.damaged-<time>`, and a clean file is written. Any other damage
    raises DamagedFile: a damaged file is never treated as empty, because the
    next save would then replace your data with nothing. With `set_aside`
    (for files Peaks can't run without, like settings) the damaged file is
    moved aside as `<name>.damaged-<time>` — kept, reported — and `default`
    is returned instead."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    try:
        return json.loads(text)
    except ValueError as exc:
        if not repair or "Extra data" not in str(exc):
            if set_aside:
                with _lock(path):
                    damaged = path.with_name(f"{path.name}.damaged-{time.strftime('%Y%m%d-%H%M%S')}")
                    with contextlib.suppress(OSError):
                        os.replace(path, damaged)
                _REPAIRS.append({"file": path.name, "kept": None, "set_aside": True, "error": str(exc)[:200],
                                 "damaged_copy": damaged.name, "at": time.time()})
                return default
            raise DamagedFile(f"{path.name} is damaged and can't be read ({exc}) — it's left untouched") from exc
    with _lock(path):
        obj, end = json.JSONDecoder().raw_decode(text.lstrip())
        stamp = time.strftime("%Y%m%d-%H%M%S")
        damaged = path.with_name(f"{path.name}.damaged-{stamp}")
        with contextlib.suppress(OSError):
            os.replace(path, damaged)
        write_json(path, obj, indent=_indent_of(text))
    info = {"file": path.name, "kept": len(obj) if isinstance(obj, (list, dict)) else None,
            "dropped_bytes": len(text) - end, "damaged_copy": damaged.name, "at": time.time()}
    _REPAIRS.append(info)
    return obj


def _indent_of(text: str) -> int | None:
    """Keep the file's own layout (pretty-printed or compact)."""
    for line in text.splitlines()[1:3]:
        stripped = line.lstrip(" ")
        if stripped and len(stripped) < len(line):
            return len(line) - len(stripped)
    return None


def repairs() -> list[dict]:
    return list(_REPAIRS)


def clear_repairs() -> None:
    _REPAIRS.clear()
