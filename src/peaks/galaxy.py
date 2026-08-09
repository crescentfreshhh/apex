"""Galaxy map: project the library into 2D so similar scenes fall near each other.

Pure numpy / sklearn / umap — no web or Stash deps, so it's cheap to import and
easy to test. Given a built `SearchIndex` (one row per cached frame), we collapse
each scene to a single centroid vector, project all centroids to 2D, and cluster
them. The web layer turns the result into a pannable star-field.

UMAP is imported lazily inside `project()` so this module imports fine without it
(PCA is the automatic fallback), keeping dev/tests off the numba dependency.
"""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def scene_points(index) -> dict:
    """Collapse each scene in `index` to one point.

    Returns a dict with aligned arrays:
      keys       list[str]        cache key per scene
      scene_ids  list[str|None]   Stash scene id per scene
      rep_t      np.ndarray (n,)  timestamp of the most-representative frame
      centroids  np.ndarray (n,d) unit mean vector per scene

    A scene's centroid is the (L2-normalized) mean of its frame vectors; its
    representative frame is the one most similar to that centroid — the best
    single thumbnail for the scene.
    """
    keys, scene_ids, rep_t, cents = [], [], [], []
    for key, (start, end) in index._key_rows.items():
        block = index.matrix[start:end]
        if block.shape[0] == 0:
            continue
        c = _unit(block.mean(axis=0).astype(np.float32))
        sims = block @ c
        rep = int(np.argmax(sims))
        keys.append(key)
        scene_ids.append(index.scene_ids[start] if start < len(index.scene_ids) else None)
        rep_t.append(float(index.times[start + rep]))
        cents.append(c)
    centroids = (
        np.asarray(cents, dtype=np.float32)
        if cents else np.zeros((0, index.dim or 1), dtype=np.float32)
    )
    return {
        "keys": keys,
        "scene_ids": scene_ids,
        "rep_t": np.asarray(rep_t, dtype=np.float32),
        "centroids": centroids,
    }


def project(centroids: np.ndarray, method: str = "umap", seed: int = 42) -> np.ndarray:
    """Project (n, d) scene centroids to (n, 2), normalized to [0, 1] per axis.

    `method="umap"` (cosine metric) gives the tightest, most navigable clusters;
    `method="pca"` is an instant, dependency-light fallback (also used when UMAP
    isn't installed). Fewer than 3 points skip straight to a trivial layout.
    """
    n = centroids.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n < 3:
        return _normalize01(np.arange(n, dtype=np.float32)[:, None].repeat(2, axis=1))

    coords = None
    if method == "umap":
        try:
            import umap  # lazy: keeps the module importable without numba

            reducer = umap.UMAP(
                n_components=2, metric="cosine",
                n_neighbors=min(15, n - 1), min_dist=0.1, random_state=seed,
            )
            coords = reducer.fit_transform(centroids)
        except Exception:  # noqa: BLE001 — no umap / fit failure → PCA fallback
            coords = None
    if coords is None:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=seed).fit_transform(centroids)
    return _normalize01(np.asarray(coords, dtype=np.float32))


def _normalize01(coords: np.ndarray) -> np.ndarray:
    lo = coords.min(axis=0, keepdims=True)
    span = coords.max(axis=0, keepdims=True) - lo
    span[span == 0] = 1.0
    return ((coords - lo) / span).astype(np.float32)


def cluster(coords: np.ndarray) -> np.ndarray:
    """Assign each 2D point a cluster id (-1 = noise/unclustered).

    HDBSCAN finds natural blobs (no fixed k), which matches how UMAP lays things
    out; KMeans is the fallback. Returns an (n,) int array.
    """
    n = coords.shape[0]
    if n < 6:
        return np.zeros((n,), dtype=int)
    min_cluster = max(10, n // 200)
    try:
        from sklearn.cluster import HDBSCAN

        # copy=True: never mutate the caller's coords (we reuse them after this)
        labels = HDBSCAN(min_cluster_size=min_cluster, copy=True).fit_predict(coords)
        if (labels >= 0).any():
            return labels.astype(int)
    except Exception:  # noqa: BLE001 — fall back to KMeans
        pass
    from sklearn.cluster import KMeans

    k = int(min(12, max(2, n // 50)))
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(coords).astype(int)
