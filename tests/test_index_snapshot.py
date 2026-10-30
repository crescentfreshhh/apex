"""The search-index snapshot: built once, then loaded (memory-mapped) without
reading the per-scene cache; a few changed scenes are patched in by reading only
those files; anything odd falls back to the full build."""

import os
import time

import numpy as np
import pytest

import peaks.index_snapshot as snap
from peaks.cache import EmbeddingCache
from peaks.search import SearchIndex

MODEL = "dinov2"
DIM = 16


def _vecs(rng, n):
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(snap, "write_behind", lambda *a: snap.write(*a))   # synchronous in tests
    c = EmbeddingCache(tmp_path / "cache")
    rng = np.random.default_rng(1)
    for i in range(12):
        n = 5 + i
        c.save(f"k{i:02d}", MODEL, np.arange(n, dtype=np.float32) * 2, _vecs(rng, n),
               meta={"scene_id": str(100 + i), "path": f"/data/{i}.mp4"})
    return c


class Counting:
    def __init__(self, cache):
        self.c, self.loads, self.peeks = cache, 0, 0
        self.root = cache.root

    def __getattr__(self, name):
        return getattr(self.c, name)

    def load(self, *a):
        self.loads += 1
        return self.c.load(*a)

    def peek_count(self, *a):
        self.peeks += 1
        return self.c.peek_count(*a)


def _search(idx, q):
    return [(h.key, round(h.time, 3), round(h.score, 5)) for h in idx.search(q, top_k=20)]


def _bump(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_second_load_reads_no_scene_files_and_matches(cache, monkeypatch, dtype):
    monkeypatch.setenv("PEAKS_INDEX_DTYPE", dtype)
    q = _vecs(np.random.default_rng(9), 1)[0]
    full = SearchIndex(cache, MODEL).build()
    assert (snap.snap_dir(cache, MODEL) / "manifest.json").exists()
    cnt = Counting(cache)
    fast = SearchIndex(cnt, MODEL).build()
    assert cnt.loads == 0 and cnt.peeks == 0
    assert isinstance(fast.matrix, np.memmap) or fast.matrix.base is not None
    assert _search(fast, q) == _search(full, q)
    assert fast.key_meta["k03"] == {"scene_id": "103", "path": "/data/3.mp4"}
    assert fast.vector_at("k05", 4.0) is not None


def test_changed_added_removed_read_only_those(cache):
    SearchIndex(cache, MODEL).build()
    rng = np.random.default_rng(5)
    cache.save("k02", MODEL, np.arange(7, dtype=np.float32), _vecs(rng, 7), meta={"scene_id": "102", "path": "/x/2.mp4"})
    _bump(cache._file("k02", MODEL))
    cache.save("k99", MODEL, np.arange(4, dtype=np.float32), _vecs(rng, 4), meta={"scene_id": "199", "path": "/data/99.mp4"})
    cache.delete("k07", MODEL)
    cnt = Counting(cache)
    patched = SearchIndex(cnt, MODEL).build()
    assert cnt.loads == 2 and cnt.peeks == 0                       # k02 (changed) + k99 (new)
    fresh = SearchIndex(cache, MODEL)._build_full(cache.keys(MODEL))
    q = _vecs(np.random.default_rng(3), 1)[0]
    assert sorted(_search(patched, q)) == sorted(_search(fresh, q))
    assert "k07" not in patched._key_rows and patched.key_meta["k02"]["path"] == "/x/2.mp4"
    cnt = Counting(cache)                                           # the snapshot was refreshed
    SearchIndex(cnt, MODEL).build()
    assert cnt.loads == 0


def test_corrupt_or_missing_snapshot_falls_back(cache):
    q = _vecs(np.random.default_rng(2), 1)[0]
    want = _search(SearchIndex(cache, MODEL).build(), q)
    d = snap.snap_dir(cache, MODEL)
    (d / "manifest.json").write_text("{not json")
    cnt = Counting(cache)
    assert _search(SearchIndex(cnt, MODEL).build(), q) == want and cnt.loads > 0
    for f in d.iterdir():
        f.unlink()
    assert _search(SearchIndex(cache, MODEL).build(), q) == want


def test_switch_off(cache, monkeypatch):
    monkeypatch.setenv("PEAKS_INDEX_SNAPSHOT", "0")
    SearchIndex(cache, MODEL).build()
    assert not (snap.snap_dir(cache, MODEL) / "manifest.json").exists()


def test_snapshot_folder_is_not_a_model(cache):
    SearchIndex(cache, MODEL).build()
    assert cache.models() == [MODEL]


def test_rewrite_keeps_a_mapped_matrix_valid(cache):
    a = SearchIndex(cache, MODEL).build()
    mapped = SearchIndex(cache, MODEL).build()                      # mapped from generation 1
    time.sleep(0.01)
    snap.write(a, cache, MODEL, snap.file_stats(cache, MODEL, cache.keys(MODEL)))   # generation 2
    assert float(np.asarray(mapped.matrix[0], dtype=np.float32)[0]) == float(np.asarray(a.matrix[0], dtype=np.float32)[0])


# --- the matrix lives on the SSD: built there, patched there, never copied into memory ---

def _gens(cache):
    return sorted(f.name for f in snap.snap_dir(cache, MODEL).iterdir() if f.suffix in (".npy", ".tmp"))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_full_build_and_patch_are_mapped_from_disk_and_match(cache, monkeypatch, dtype):
    monkeypatch.setenv("PEAKS_INDEX_DTYPE", dtype)
    q = _vecs(np.random.default_rng(4), 1)[0]
    built = SearchIndex(cache, MODEL).build()
    assert built.on_disk and len(_gens(cache)) == 2 and not any(g.endswith(".tmp") for g in _gens(cache))
    monkeypatch.setenv("PEAKS_INDEX_SNAPSHOT", "0")
    in_ram = SearchIndex(cache, MODEL).build()
    assert not in_ram.on_disk and _search(built, q) == _search(in_ram, q)
    monkeypatch.delenv("PEAKS_INDEX_SNAPSHOT")

    old_gens = _gens(cache)
    rng = np.random.default_rng(8)
    cache.save("k50", MODEL, np.arange(6, dtype=np.float32), _vecs(rng, 6), meta={"scene_id": "150", "path": "/d/50.mp4"})
    time.sleep(0.01)
    patched = SearchIndex(cache, MODEL).build()
    assert patched.on_disk and "k50" in patched._key_rows
    assert set(_gens(cache)).isdisjoint(old_gens)                       # the old generation is gone…
    assert _search(built, q)                                            # …yet the old map still reads
    monkeypatch.setenv("PEAKS_INDEX_SNAPSHOT", "0")
    assert sorted(_search(patched, q)) == sorted(_search(SearchIndex(cache, MODEL).build(), q))


def test_a_build_that_shrank_midway_still_loads(cache, monkeypatch):
    real = cache.load
    calls = {"n": 0}

    def flaky(key, model):
        calls["n"] += 1
        if key == "k11":
            raise OSError("gone")                                       # counted, then unreadable
        return real(key, model)
    monkeypatch.setattr(cache, "load", flaky)
    idx = SearchIndex(cache, MODEL).build()
    assert idx.on_disk and "k11" not in idx._key_rows
    monkeypatch.setattr(cache, "load", real)
    cnt = Counting(cache)
    again = SearchIndex(cnt, MODEL).build()                            # k11 read back in; the rest from disk
    assert cnt.loads == 1 and again.size == idx.size + 16


def test_a_failed_build_releases_the_snapshot_and_leaves_no_temp(cache):
    import threading

    def boom(done, total):
        raise RuntimeError("stop")                                      # inside the build, writer open
    with pytest.raises(RuntimeError):
        SearchIndex(cache, MODEL)._build_full(cache.keys(MODEL), on_progress=boom, stats={})
    assert not any(g.endswith(".tmp") for g in _gens(cache)) if snap.snap_dir(cache, MODEL).exists() else True
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("idx", SearchIndex(cache, MODEL).build()))
    t.start(); t.join(10)
    assert "idx" in out and out["idx"].on_disk                          # no lock left held


def test_mapped_pages_are_not_counted_as_real_memory(tmp_path):
    from peaks.web import memwatch

    if memwatch._status_kb("RssAnon") is None:
        pytest.skip("no /proc/self/status")
    arr = np.lib.format.open_memmap(tmp_path / "big.npy", mode="w+", dtype=np.uint8, shape=(256 << 20,))
    arr[:] = 1
    arr.flush()
    del arr
    m = np.load(tmp_path / "big.npy", mmap_mode="r")
    anon0, file0 = memwatch.rss_bytes(), memwatch.file_rss_bytes()
    assert int(m[:: 4096].sum()) == (256 << 20) // 4096                 # touch every page
    assert memwatch.file_rss_bytes() - file0 > 200 << 20
    assert memwatch.rss_bytes() - anon0 < 32 << 20
