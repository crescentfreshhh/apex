import numpy as np
import pytest

from peaks.cache import EmbeddingCache
from peaks.search import Hit, SearchIndex


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _seed(cache, key, scene_id, vecs, times):
    cache.save(
        key, "dino",
        np.asarray(times, dtype=np.float32),
        np.asarray(vecs, dtype=np.float32),
        meta={"scene_id": scene_id},
    )


def test_build_stacks_all_scenes(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0]), _unit([0, 1, 0])], [0.0, 8.0])
    _seed(cache, "k2", "2", [_unit([0, 0, 1])], [0.0])
    idx = SearchIndex(cache, "dino").build()
    assert idx.size == 3 and idx.dim == 3
    assert set(idx.scene_ids) == {"1", "2"}


def test_vector_around_pools_within_window(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0]), _unit([0, 1, 0])], [0.0, 8.0])
    idx = SearchIndex(cache, "dino").build(["k1"])

    # window excludes the far frame → just the near one (== vector_at)
    v0 = idx.vector_around("k1", 0.0, window=6.0)
    assert np.allclose(v0, _unit([1, 0, 0]))
    # window covers both frames → unit mean of the two
    vmid = idx.vector_around("k1", 4.0, window=6.0)
    assert np.allclose(vmid, _unit([1, 1, 0]))
    # window<=0 degrades to the single nearest frame
    assert np.allclose(idx.vector_around("k1", 8.0, window=0.0), _unit([0, 1, 0]))
    # a window that lands between samples still returns the nearest (never None)
    assert idx.vector_around("k1", 3.0, window=0.5) is not None


def test_build_preallocates_matches_manual_stack(tmp_path, monkeypatch):
    monkeypatch.delenv("PEAKS_INDEX_DTYPE", raising=False)   # auto → float32 when small
    # the preallocated build must produce the exact same matrix/times/rows as a
    # naive per-scene stack (parity guard for the memory-lean rewrite).
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0]), _unit([0, 1, 0])], [0.0, 8.0])
    _seed(cache, "k2", "2", [_unit([0, 0, 1])], [4.0])
    _seed(cache, "k3", "3", [_unit([1, 1, 0]), _unit([0, 1, 1])], [1.0, 2.0])
    keys = ["k1", "k2", "k3"]
    idx = SearchIndex(cache, "dino").build(keys)

    expected = np.concatenate([cache.load(k, "dino")[1] for k in keys], axis=0)
    # small index → float32 storage (auto); the float32 accessor matches exactly
    assert idx.storage == "float32"
    assert np.array_equal(idx.rows(0, idx.size), expected)
    assert idx.times.tolist() == [0.0, 8.0, 4.0, 1.0, 2.0]
    assert idx._key_rows == {"k1": (0, 2), "k2": (2, 3), "k3": (3, 5)}
    assert idx.keys == ["k1", "k1", "k2", "k3", "k3"]


def test_peek_count_reads_frame_count(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0]), _unit([0, 1, 0])], [0.0, 8.0])
    assert cache.peek_count("k1", "dino") == 2


def test_build_tolerates_count_growth_between_passes(tmp_path, monkeypatch):
    # a scene may gain frames between the count pass and the load pass during a
    # live embed; the extra rows are clamped, never an out-of-bounds write.
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])], [0, 1, 2])
    real_peek = cache.peek_count
    monkeypatch.setattr(cache, "peek_count", lambda k, m: 1 if k == "k1" else real_peek(k, m))
    idx = SearchIndex(cache, "dino").build(["k1"])
    assert idx.size == 1 and idx.dim == 3  # sized to the undercount, clamped safely


def test_search_min_score_returns_all_above_floor(tmp_path):
    cache = EmbeddingCache(tmp_path)
    # 5 scenes at varying cosine to the query axis [1,0,0]
    for i, v in enumerate([[1, 0, 0], [0.9, 0.1, 0], [0.8, 0.2, 0], [0.2, 1, 0], [0, 1, 0]]):
        _seed(cache, f"k{i}", str(i), [_unit(v)], [0.0])
    idx = SearchIndex(cache, "dino").build()
    q = _unit([1, 0, 0])

    # unbounded threshold search returns EVERY frame with cosine >= floor (not top 60).
    # note vectors are unit-normalized, so [0.8,0.2,0] scores ~0.97, not 0.8.
    hits = idx.search(q, top_k=None, min_score=0.7)
    assert len(hits) == 3  # scenes 0,1,2 clear 0.7; 3 (~0.20) and 4 (0.0) don't
    assert all(h.score >= 0.7 for h in hits)
    assert [h.scene_id for h in hits] == ["0", "1", "2"]  # ordered best-first
    # a looser floor returns more; a stricter one fewer
    assert len(idx.search(q, top_k=None, min_score=0.0)) == 5
    assert len(idx.search(q, top_k=None, min_score=0.999)) == 1  # only the exact match


def test_search_min_score_respects_per_scene(tmp_path):
    cache = EmbeddingCache(tmp_path)
    # one scene with 4 near-identical high-scoring frames
    _seed(cache, "k1", "1", [_unit([1, 0, 0])] * 4, [0, 1, 2, 3])
    _seed(cache, "k2", "2", [_unit([0.95, 0.05, 0])], [0.0])
    idx = SearchIndex(cache, "dino").build()
    hits = idx.search(_unit([1, 0, 0]), top_k=None, per_scene=1, min_score=0.5)
    # per_scene=1 caps scene 1 to a single moment even though 4 qualify
    assert sum(1 for h in hits if h.scene_id == "1") == 1
    assert {h.scene_id for h in hits} == {"1", "2"}


def test_search_ranks_by_cosine(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0])], [0.0])
    _seed(cache, "k2", "2", [_unit([0.9, 0.1, 0])], [0.0])
    _seed(cache, "k3", "3", [_unit([0, 1, 0])], [0.0])
    idx = SearchIndex(cache, "dino").build()

    hits = idx.search(_unit([1, 0, 0]), top_k=3)
    assert [h.scene_id for h in hits] == ["1", "2", "3"]  # by descending cosine
    assert hits[0].score > hits[1].score > hits[2].score
    assert isinstance(hits[0], Hit)


def test_search_empty_index(tmp_path):
    idx = SearchIndex(EmbeddingCache(tmp_path), "dino").build()
    assert idx.size == 0
    assert idx.search(np.array([1.0, 0, 0])) == []


def test_per_scene_cap_spreads_results(tmp_path):
    cache = EmbeddingCache(tmp_path)
    # scene 1 has 5 near-identical strong matches; scene 2 has one slightly worse
    _seed(cache, "k1", "1", [_unit([1, 0, 0.01 * i]) for i in range(5)],
          [i * 8.0 for i in range(5)])
    _seed(cache, "k2", "2", [_unit([0.8, 0.2, 0])], [0.0])
    idx = SearchIndex(cache, "dino").build()

    capped = idx.search(_unit([1, 0, 0]), top_k=10, per_scene=2)
    from_scene1 = sum(1 for h in capped if h.scene_id == "1")
    assert from_scene1 == 2  # capped
    assert any(h.scene_id == "2" for h in capped)  # scene 2 still surfaces


def test_search_by_frame_excludes_own_scene(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0]), _unit([0.99, 0.01, 0])], [0.0, 8.0])
    _seed(cache, "k2", "2", [_unit([0.95, 0.05, 0])], [0.0])
    idx = SearchIndex(cache, "dino").build()

    hits = idx.search_by_frame("k1", 0.0, top_k=5)
    assert all(h.key != "k1" for h in hits)  # own scene excluded
    assert hits and hits[0].scene_id == "2"


def test_vector_at_picks_nearest_time(tmp_path):
    cache = EmbeddingCache(tmp_path)
    a, b = _unit([1, 0, 0]), _unit([0, 1, 0])
    _seed(cache, "k1", "1", [a, b], [0.0, 10.0])
    idx = SearchIndex(cache, "dino").build()

    np.testing.assert_allclose(idx.vector_at("k1", 1.0), a, atol=2e-3)  # nearest 0.0
    np.testing.assert_allclose(idx.vector_at("k1", 9.0), b, atol=2e-3)  # nearest 10.0
    assert idx.vector_at("missing", 0.0) is None


def test_clip_span_tail_uses_scene_own_step(tmp_path):
    """The tail after the last similar frame is the scene's OWN sampling step, not
    the configured interval — a 2s-embedded scene must not get an 8s tail (the
    library may be half-converted, and per-run overrides never touch config)."""
    cache = EmbeddingCache(tmp_path)
    same, cut = _unit([1, 0, 0]), _unit([0, 1, 0])
    _seed(cache, "k1", "1", [same, same, cut], [10.0, 12.0, 14.0])   # 2s grid
    idx = SearchIndex(cache, "dino").build(["k1"])
    s, e = idx.clip_span("k1", 10.0, similarity=0.5, min_dur=1.0, max_dur=30.0, interval=8.0)
    assert (s, e) == (10.0, 14.0)      # last similar frame 12.0 + 2s own step (not +8)


def test_cache_meta_reads_skip_vectors(tmp_path, monkeypatch):
    """Resume checks read only meta — never the (multi-GB in aggregate) vectors."""
    cache = EmbeddingCache(tmp_path)
    cache.save("k1", "dino", np.array([0.0], dtype=np.float32),
               np.stack([_unit([1, 0, 0])]), meta={"interval": -102.0})
    cache.save("k2", "dino", np.array([0.0], dtype=np.float32),
               np.stack([_unit([0, 1, 0])]), meta={})
    monkeypatch.setattr(EmbeddingCache, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded vectors")))
    assert cache.has("k1", "dino", interval=-102.0)
    assert not cache.has("k1", "dino", interval=-108.0)
    assert cache.signatures("dino") == {"k1": -102.0, "k2": None}
    assert cache.signatures("dino") == {"k1": -102.0, "k2": None}   # memoized path


@pytest.mark.parametrize("storage", ["float32", "bfloat16"])
def test_index_math_is_dtype_safe(tmp_path, monkeypatch, storage):
    """Every accessor returns float32 matching exact float32 math, for both the
    float32 and the RAM-lean bfloat16 storage."""
    monkeypatch.setenv("PEAKS_INDEX_DTYPE", storage)
    cache = EmbeddingCache(tmp_path)
    rng = np.random.default_rng(0)
    vecs = [_unit(rng.standard_normal(16)) for _ in range(7)]
    _seed(cache, "k1", "1", vecs[:4], [0.0, 2.0, 4.0, 6.0])
    _seed(cache, "k2", "2", vecs[4:], [0.0, 2.0, 4.0])
    idx = SearchIndex(cache, "dino").build(["k1", "k2"])
    assert idx.storage == storage
    exact = np.concatenate([cache.load(k, "dino")[1] for k in ("k1", "k2")]).astype(np.float32)
    q = _unit(rng.standard_normal(16))
    tol = 1e-6 if storage == "float32" else 1e-2    # bf16: ~3 significant digits
    got = idx.dot(q)
    assert got.dtype == np.float32 and np.allclose(got, exact @ q, atol=tol)
    for chunk in (1, 3, 100):                       # chunk boundaries don't matter
        assert np.allclose(idx.apply(lambda b: b @ q, chunk=chunk), exact @ q, atol=tol)
    assert np.allclose(idx.rows(4, 7), exact[4:7], atol=tol)
    assert np.allclose(idx.take(np.array([6, 0])), exact[[6, 0]], atol=tol)
    v = idx.vector_at("k2", 2.0)
    assert v.dtype == np.float32 and np.allclose(v, exact[5], atol=tol)
    hits = idx.search(exact[2], top_k=1)
    assert hits[0].key == "k1" and hits[0].time == 4.0   # nearest is itself


def test_bf16_encoding_is_near_lossless():
    from peaks.search import _bf16_into_f32, _f32_to_bf16
    rng = np.random.default_rng(1)
    a = rng.standard_normal((50, 32)).astype(np.float32)
    back = _bf16_into_f32(_f32_to_bf16(a), np.zeros(a.shape, np.float32))
    rel = np.abs(back - a) / np.maximum(np.abs(a), 1e-30)
    assert rel.max() <= 2 ** -8                     # round-to-nearest: ≤ half an ulp of 8 bits


def test_clip_span_holds_until_shot_change(tmp_path):
    cache = EmbeddingCache(tmp_path)
    same = _unit([1, 0, 0])
    cut = _unit([0, 1, 0])                       # a drastically different frame (a cut)
    # frames every 2s: three similar (the moment's shot), then a cut, then more
    _seed(cache, "k1", "1",
          [same, same, same, cut, cut],
          [0.0, 2.0, 4.0, 6.0, 8.0])
    idx = SearchIndex(cache, "dino").build(["k1"])

    # from t=0 the clip holds through the three similar frames and stops at the cut
    start, end = idx.clip_span("k1", 0.0, similarity=0.5, min_dur=1.0, max_dur=30.0, interval=2.0)
    assert start == 0.0
    assert end == 6.0          # last similar frame at 4.0 + one 2.0s interval

    # min_duration floors a very short hold (anchor at the last frame → no room to grow)
    s2, e2 = idx.clip_span("k1", 8.0, similarity=0.5, min_dur=5.0, max_dur=30.0, interval=2.0)
    assert e2 - s2 == 5.0      # natural hold (2s) floored up to min_dur

    # max_duration caps a long hold
    s3, e3 = idx.clip_span("k1", 0.0, similarity=0.5, min_dur=1.0, max_dur=3.0, interval=2.0)
    assert e3 - s3 == 3.0

    # missing scene / disabled → fixed fallback window, start at the moment
    s4, e4 = idx.clip_span("missing", 12.0, similarity=0.5, min_dur=3.0, max_dur=30.0, interval=2.0)
    assert (s4, e4) == (12.0, 32.0)     # min(max_dur, 20) = 20s fallback
    s5, e5 = idx.clip_span("k1", 0.0, similarity=0.0, min_dur=3.0, max_dur=30.0, interval=2.0)
    assert (s5, e5) == (0.0, 20.0)      # similarity<=0 disables detection


def test_build_only_requested_keys(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _seed(cache, "k1", "1", [_unit([1, 0, 0])], [0.0])
    _seed(cache, "k2", "2", [_unit([0, 1, 0])], [0.0])
    idx = SearchIndex(cache, "dino").build(keys=["k1"])
    assert idx.size == 1 and idx.scene_ids == ["1"]
