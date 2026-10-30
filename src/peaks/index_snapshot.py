"""A saved copy of the built search index, so (re)loading it takes seconds.

Building the index from the cache means opening every per-scene `.npz` (twice:
count, then load) and converting float16 → float32 → bfloat16 — about a minute
for a 2 s-sampled library, paid at every start and every time an ingest frees
memory for Stash. Almost nothing changes between two loads, so after a build
the finished matrix is written once as:

    <cache>/_index/<model>/matrix-<gen>.npy   rows in their final dtype
    <cache>/_index/<model>/times-<gen>.npy
    <cache>/_index/<model>/manifest.json      per scene: file size + mtime, row range,
                                              scene id, path; dtype, dim, generation

Loading compares the scene files' size+mtime with the manifest:
- nothing changed → the matrix is memory-mapped: usable at once, and after an
  ingest freed memory the pages are usually still in the OS file cache;
- a few scenes added / changed / removed → unchanged rows come from the
  snapshot, only those files are read, and a fresh snapshot is written behind;
- no snapshot / dtype or dim mismatch / anything odd → None (full build).

Files carry a generation number so a rewrite never overwrites a matrix that's
still mapped; the manifest is written last (it's the commit marker) and older
generations are removed afterwards. PEAKS_INDEX_SNAPSHOT=0 turns it all off.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def enabled() -> bool:
    return os.environ.get("PEAKS_INDEX_SNAPSHOT", "1").strip().lower() not in ("0", "false", "no", "off")


def snap_dir(cache, model: str) -> Path:
    return Path(cache.root) / "_index" / model


def _lock(d: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(d), threading.Lock())


def file_stats(cache, model: str, keys: list[str]) -> dict[str, list[int]]:
    """{key: [size, mtime_ns]} for the scene files that exist."""
    out = {}
    for k in keys:
        try:
            st = os.stat(cache._file(k, model))
        except OSError:
            continue
        out[k] = [int(st.st_size), int(st.st_mtime_ns)]
    return out


def read(cache, model: str) -> dict | None:
    """The saved snapshot (matrix memory-mapped), or None."""
    d = snap_dir(cache, model)
    try:
        man = json.loads((d / "manifest.json").read_text())
        gen = man["gen"]
        try:
            matrix = np.load(d / f"matrix-{gen}.npy", mmap_mode="r")
        except (ValueError, OSError):          # a filesystem without mmap: read it whole
            matrix = np.load(d / f"matrix-{gen}.npy")
        times = np.load(d / f"times-{gen}.npy")
    except (OSError, ValueError, KeyError):
        return None
    total = man.get("total")
    if not isinstance(total, int) or matrix.shape[0] < total or times.shape[0] != total:
        return None
    if matrix.shape[0] > total:                # built a little big (the library shrank mid-build)
        matrix = matrix[:total]
    return {"man": man, "matrix": matrix, "times": times}


_FLUSH_EVERY = 256 << 20


def _save_flushed(path: Path, arr: np.ndarray) -> None:
    """np.save, in chunks that are forced to disk and dropped from the page
    cache as they go: a ~10 GB snapshot never sits in memory unwritten, where
    it would count against the container's limit."""
    arr = np.ascontiguousarray(arr)
    flat = arr.reshape(-1).view(np.uint8) if arr.size else np.empty(0, np.uint8)
    with open(path, "wb") as fh:
        np.lib.format.write_array_header_1_0(fh, np.lib.format.header_data_from_array_1_0(arr))
        fd = fh.fileno()
        start = fh.tell()
        for off in range(0, flat.size, _FLUSH_EVERY):
            fh.write(memoryview(flat[off:off + _FLUSH_EVERY]))
            fh.flush()
            os.fsync(fd)
            if hasattr(os, "posix_fadvise"):
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
        assert fh.tell() == start + flat.size


def write(index, cache, model: str, stats: dict[str, list[int]]) -> bool:
    """Save `index` (a built SearchIndex) as the snapshot. Safe to call from a
    background thread; a concurrent write for the same model is skipped."""
    d = snap_dir(cache, model)
    lk = _lock(d)
    if not lk.acquire(blocking=False):
        return False
    try:
        d.mkdir(parents=True, exist_ok=True)
        gen = f"{int(time.time() * 1000):x}"
        scenes = {}
        for key, (a, b) in index._key_rows.items():
            if key not in stats:
                continue
            m = index.key_meta.get(key) or {}
            scenes[key] = {"st": stats[key], "rows": [int(a), int(b)],
                           "scene_id": m.get("scene_id"), "path": m.get("path")}
        man = {"gen": gen, "model": model, "total": int(index.size), "dim": int(index.dim),
               "bf16": bool(index._bf16), "scenes": scenes, "written": time.time()}
        for name, arr in (("matrix", index.matrix), ("times", index.times)):
            tmp = d / f"{name}-{gen}.npy.tmp"
            _save_flushed(tmp, np.asarray(arr))
            os.replace(tmp, d / f"{name}-{gen}.npy")
        tmp = d / "manifest.json.tmp"
        tmp.write_text(json.dumps(man))
        os.replace(tmp, d / "manifest.json")      # the commit: readers switch to `gen` now
        for f in d.iterdir():                     # older generations (a mapped one stays valid)
            if f.suffix in (".npy", ".tmp") and gen not in f.name:
                try:
                    f.unlink()
                except OSError:
                    pass
        return True
    except Exception:  # noqa: BLE001 — a snapshot is an optimisation, never an error
        return False
    finally:
        lk.release()


def _drop_pages(arr) -> None:
    """Write a writable map's dirty pages to disk and let them go from memory
    (they're re-read from the file when touched)."""
    import mmap

    try:
        arr.flush()
    except (AttributeError, OSError, ValueError):
        pass
    mm = getattr(arr, "_mmap", None)
    if mm is not None and hasattr(mm, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
        try:
            mm.madvise(mmap.MADV_DONTNEED)
        except (OSError, ValueError):
            pass


class Writer:
    """Builds the index matrix straight into its snapshot file on disk (the
    cache SSD), so a build or a patch never holds the matrix in process memory:
    rows are written through a map, flushed and dropped as they go, and the
    finished file is then mapped read-only — the kernel keeps its pages cached
    while RAM is free and drops them when it isn't.

    `Writer.open(...)` returns None where that isn't possible (snapshots off, a
    filesystem without mmap); callers then build in memory as before."""

    def __init__(self, d: Path, total: int, dim: int, bf16: bool):
        self.d, self.bf16 = d, bf16
        self.lock = _lock(d)
        self.lock.acquire()
        try:
            self.gen = f"{int(time.time() * 1000):x}"
            self.tmp = d / f"matrix-{self.gen}.npy.tmp"
            self.matrix = np.lib.format.open_memmap(self.tmp, mode="w+", dtype=np.uint16 if bf16 else np.float32,
                                                    shape=(max(1, int(total)), int(dim)))
        except BaseException:
            self.lock.release()
            raise
        self.since = 0
        self.closed = False

    @classmethod
    def open(cls, cache, model: str, total: int, dim: int, bf16: bool) -> "Writer | None":
        if not enabled() or total <= 0 or dim <= 0:
            return None
        d = snap_dir(cache, model)
        try:
            d.mkdir(parents=True, exist_ok=True)
            return cls(d, total, dim, bf16)
        except (OSError, ValueError):
            return None

    def wrote(self, nbytes: int) -> None:
        self.since += int(nbytes)
        if self.since >= _FLUSH_EVERY:
            _drop_pages(self.matrix)
            self.since = 0

    def commit(self, index, stats: dict[str, list[int]], used: int):
        """Finish: times + manifest written, older generations removed; returns
        the matrix mapped read-only (`used` rows)."""
        try:
            _drop_pages(self.matrix)
            del self.matrix
            name = self.d / f"matrix-{self.gen}.npy"
            os.replace(self.tmp, name)
            tt = self.d / f"times-{self.gen}.npy.tmp"
            _save_flushed(tt, np.asarray(index.times[:used], dtype=np.float32))
            os.replace(tt, self.d / f"times-{self.gen}.npy")
            scenes = {}
            for key, (a, b) in index._key_rows.items():
                if key not in stats:
                    continue
                m = index.key_meta.get(key) or {}
                scenes[key] = {"st": stats[key], "rows": [int(a), int(b)],
                               "scene_id": m.get("scene_id"), "path": m.get("path")}
            man = {"gen": self.gen, "model": self.d.name, "total": int(used), "dim": int(index.dim),
                   "bf16": bool(self.bf16), "scenes": scenes, "written": time.time()}
            tmp = self.d / "manifest.json.tmp"
            tmp.write_text(json.dumps(man))
            os.replace(tmp, self.d / "manifest.json")   # the commit
            for f in self.d.iterdir():                   # older generations (a mapped one stays valid)
                if f.suffix in (".npy", ".tmp") and self.gen not in f.name:
                    try:
                        f.unlink()
                    except OSError:
                        pass
            mapped = np.load(name, mmap_mode="r")
            return mapped[:used] if mapped.shape[0] > used else mapped
        finally:
            self.closed = True
            self.lock.release()

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            del self.matrix
        except AttributeError:
            pass
        try:
            self.tmp.unlink()
        except OSError:
            pass
        self.lock.release()


def write_behind(index, cache, model: str, stats: dict[str, list[int]]) -> None:
    threading.Thread(target=write, args=(index, cache, model, stats), daemon=True,
                     name="peaks-index-snapshot").start()
