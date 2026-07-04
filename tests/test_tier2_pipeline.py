"""Tier-2 orchestration: training-set assembly, candidates, train->score loop."""

import numpy as np

from peaks.cache import EmbeddingCache
from peaks.config import ScoringConfig
from peaks.labels import LabelStore
from peaks.models import Scene
from peaks.pipeline import (
    build_training_set,
    gather_candidates,
    score_library,
    train_profile,
)


def _scene(id_, key):
    return Scene.from_dict(
        {
            "id": id_,
            "title": "",
            "files": [{"path": f"/m/{id_}.mp4", "fingerprints": [
                {"type": "oshash", "value": key}]}],
            "scene_markers": [],
        }
    )


def _cache_scene(cache, key, scene_id, n=20, dim=16, seed=0):
    """Cache a scene: first half 'positive-ish' (+1), second half 'negative' (-1)."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(1.0, 0.2, size=(n // 2, dim)).astype(np.float32)
    neg = rng.normal(-1.0, 0.2, size=(n // 2, dim)).astype(np.float32)
    vecs = np.vstack([pos, neg])
    times = np.arange(n, dtype=np.float32) * 2.0
    cache.save(key, "fake", times, vecs, meta={"scene_id": scene_id, "path": f"/m/{scene_id}.mp4"})
    return times, vecs


def test_build_training_set_picks_nearest_frames(tmp_path):
    cache = EmbeddingCache(tmp_path)
    times, vecs = _cache_scene(cache, "k1", "1", n=10)
    store = LabelStore(tmp_path / "labels.json")
    store.add("k1", 0.0, 1, "apex")    # nearest frame idx 0
    store.add("k1", 18.0, 0, "apex")   # nearest frame idx 9
    store.add("k2", 5.0, 1, "apex")    # no cache -> skipped

    X, y = build_training_set(store, cache, "fake", "apex")
    assert X.shape == (2, 16)
    np.testing.assert_allclose(X[0], vecs[0], atol=2e-3)  # cache stores float16
    np.testing.assert_allclose(X[1], vecs[9], atol=2e-3)
    assert list(y) == [1, 0]


def test_build_training_set_empty(tmp_path):
    cache = EmbeddingCache(tmp_path)
    store = LabelStore(tmp_path / "labels.json")
    X, y = build_training_set(store, cache, "fake", "apex")
    assert X.shape[0] == 0 and y.shape[0] == 0


def test_gather_candidates_prioritizes_high_scores(tmp_path):
    cache = EmbeddingCache(tmp_path)
    _cache_scene(cache, "k1", "1", n=10)

    # scorer: high for the +1 cluster (first half), low for the rest
    def score_frames(vecs):
        return (vecs.mean(axis=1) > 0).astype(np.float32)

    cands = gather_candidates(
        cache, "fake", score_frames, top_per_scene=3, random_per_scene=0, seed=1
    )
    assert len(cands) == 3
    assert all(c.score == 1.0 for c in cands)  # all from the positive cluster
    assert cands[0].scene_id == "1" and cands[0].path == "/m/1.mp4"


def test_train_then_score_finds_positive_region(tmp_path):
    """Full Tier-2 loop: label -> train -> score -> segment lands on positives."""
    cache = EmbeddingCache(tmp_path)
    # contiguous positive block at frames 0-9 (times 0..18), negatives after
    times, vecs = _cache_scene(cache, "k1", "1", n=20, dim=16, seed=3)
    store = LabelStore(tmp_path / "labels.json")
    # label a few frames from each region
    for t in (0.0, 2.0, 4.0):
        store.add("k1", t, 1, "apex")
    for t in (30.0, 34.0, 38.0):
        store.add("k1", t, 0, "apex")

    clf, stats = train_profile(store, cache, "fake", "apex", kind="logreg")
    assert stats["samples"] == 6 and stats["positives"] == 3

    scoring = ScoringConfig(
        high=0.5, low=0.4, min_duration=2.0, merge_gap=4.0, smooth_window=1, pad=0.0
    )
    captured = []

    class _Client:
        def find_or_create_tag(self, name):
            return type("T", (), {"id": "1", "name": name})()

        def create_scene_marker(self, *, scene_id, seconds, primary_tag_id, title, end_seconds):
            captured.append((seconds, end_seconds))

    score_library(
        [_scene("1", "k1")], cache, "fake", clf.predict_proba, scoring,
        client=_Client(), tag_name="apex", write=True, log=lambda *_: None,
    )
    # at least one segment, and it should start within the positive region (early)
    assert captured
    assert min(s for s, _ in captured) < 18.0
