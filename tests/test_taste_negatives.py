"""Taste model needs negative contrast.

A taste set is almost all positives (every 👍 and ⭐-save is a positive label),
so a binary classifier trained on it alone saturates — it calls everything
in-taste. `train_profile(background_ratio>0)` adds a random library background as
implicit negatives; these tests show that restores a discriminative, spread-out
score instead of ~1.0 everywhere.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from peaks.cache import EmbeddingCache  # noqa: E402
from peaks.pipeline import train_profile  # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


class _Lab:
    def __init__(self, key, time, label, ts=0.0):
        self.key, self.time, self.label, self.ts = key, time, int(label), ts


class _Store:
    def __init__(self, labs):
        self._labs = labs

    def for_profile(self, profile):
        return self._labs

    def counts(self, profile):
        pos = sum(1 for x in self._labs if x.label == 1)
        return pos, len(self._labs) - pos


def _seed(tmp_path):
    """A library: 6 'taste' scenes clustered on axis 0, 40 random background scenes."""
    dim = 16
    cache = EmbeddingCache(str(tmp_path / "cache"))
    rng = np.random.default_rng(0)
    taste_dir = np.zeros(dim, dtype=np.float32)
    taste_dir[0] = 1.0
    pos_labels = []
    for i in range(6):
        key = f"pos{i}"
        v = _unit(taste_dir + rng.normal(0, 0.03, dim))
        cache.save(key, "dinov2", np.array([0.0], dtype=np.float32),
                   np.stack([v]), meta={"scene_id": f"p{i}"})
        pos_labels.append(_Lab(key, 0.0, 1))
    for i in range(40):
        vecs = np.stack([_unit(rng.normal(0, 1, dim)) for _ in range(3)])
        cache.save(f"bg{i}", "dinov2", np.array([0.0, 1.0, 2.0], dtype=np.float32),
                   vecs, meta={"scene_id": f"b{i}"})
    return cache, pos_labels, dim, taste_dir


def test_positives_only_cannot_train(tmp_path):
    cache, pos_labels, _, _ = _seed(tmp_path)
    # no background, no explicit negatives → only one class → cannot train
    with pytest.raises(ValueError, match="both"):
        train_profile(_Store(pos_labels), cache, "dinov2", "apex", background_ratio=0.0)


def test_background_negatives_restore_discrimination(tmp_path):
    cache, pos_labels, dim, taste_dir = _seed(tmp_path)
    clf, stats = train_profile(
        _Store(pos_labels), cache, "dinov2", "apex", background_ratio=1.0,
    )
    assert stats["kind"] == "logreg"
    assert stats["positives"] == 6 and stats["background"] > 0

    # a taste-like frame scores high, a random one low — real separation
    hi = clf.predict_proba(_unit(taste_dir).reshape(1, -1))[0]
    lo = clf.predict_proba(_unit(np.array([0] + [1] * (dim - 1), dtype=np.float32)).reshape(1, -1))[0]
    assert hi > 0.6 and lo < 0.4 and hi - lo > 0.3

    # over the whole library the scores SPREAD (the bug was: all ≈ 1.0)
    allv = []
    for k in cache.keys("dinov2"):
        _t, vecs, _m = cache.load(k, "dinov2")
        allv.append(vecs)
    proba = clf.predict_proba(np.vstack(allv))
    assert proba.max() <= 1.0 and proba.min() >= 0.0
    assert float(proba.mean()) < 0.8      # not saturated at the top
    assert float(proba.std()) > 0.05      # meaningfully spread (was ~0 when saturated)
