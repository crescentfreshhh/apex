"""Galaxy map: scene-point projection + clustering, and the Service build/cache."""

import numpy as np
import pytest

from peaks.cache import EmbeddingCache
from peaks.galaxy import cluster, project, scene_points
from peaks.search import SearchIndex


def _index_with_scenes(tmp_path, n_scenes=8, frames=4, dim=8):
    """Two visually distinct blobs of scenes in a real SearchIndex."""
    cache = EmbeddingCache(tmp_path / "cache")
    rng = np.random.default_rng(0)
    for s in range(n_scenes):
        base = np.zeros(dim, dtype="float32")
        base[0 if s % 2 == 0 else 1] = 1.0  # blob A on axis 0, blob B on axis 1
        vecs = base + rng.normal(0, 0.05, size=(frames, dim)).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        times = np.arange(frames, dtype="float32") * 5.0
        cache.save(f"k{s}", "dinov2", times, vecs, meta={"scene_id": str(s + 1)})
    return SearchIndex(cache, "dinov2").build()


def test_scene_points_one_per_scene(tmp_path):
    idx = _index_with_scenes(tmp_path, n_scenes=6, frames=4, dim=8)
    pts = scene_points(idx)
    assert len(pts["keys"]) == 6
    assert pts["centroids"].shape == (6, 8)
    assert pts["rep_t"].shape == (6,)
    # centroids are unit-normalized; rep_t is one of the scene's real timestamps
    np.testing.assert_allclose(np.linalg.norm(pts["centroids"], axis=1), 1.0, atol=1e-5)
    assert set(pts["scene_ids"]) == {str(i) for i in range(1, 7)}
    assert all(0.0 <= t <= 15.0 for t in pts["rep_t"])


def test_project_and_cluster_shapes(tmp_path):
    idx = _index_with_scenes(tmp_path, n_scenes=10, frames=3, dim=8)
    pts = scene_points(idx)
    coords = project(pts["centroids"], method="pca")  # pca: fast + deterministic
    assert coords.shape == (10, 2)
    assert np.isfinite(coords).all()
    assert coords.min() >= 0.0 and coords.max() <= 1.0  # normalized to [0,1]
    labels = cluster(coords)
    assert labels.shape == (10,)


def test_project_empty_and_tiny():
    assert project(np.zeros((0, 8), dtype="float32")).shape == (0, 2)
    assert project(np.ones((2, 8), dtype="float32"), method="pca").shape == (2, 2)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("umap") is None, reason="umap not installed"
)
def test_project_umap_smoke(tmp_path):
    idx = _index_with_scenes(tmp_path, n_scenes=12, frames=3, dim=8)
    coords = project(scene_points(idx)["centroids"], method="umap")
    assert coords.shape == (12, 2) and np.isfinite(coords).all()
