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
    if matrix.shape[0] != man.get("total") or times.shape[0] != man.get("total"):
        return None
    return {"man": man, "matrix": matrix, "times": times}


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
            with open(tmp, "wb") as fh:
                np.save(fh, np.asarray(arr))
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


def write_behind(index, cache, model: str, stats: dict[str, list[int]]) -> None:
    threading.Thread(target=write, args=(index, cache, model, stats), daemon=True,
                     name="peaks-index-snapshot").start()
