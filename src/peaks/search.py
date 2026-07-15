"""In-memory nearest-neighbour search over the embedding cache.

The cache is a per-scene set of L2-normalized frame vectors. This module
stacks them into one matrix so any query vector (a frame you pick, an uploaded
image, or a CLIP text prompt) can be matched against every moment in the
library by cosine similarity — the engine behind "find more like this" and
text search.

Brute-force numpy is plenty for libraries up to a few million frames (a query
is one matmul, single-digit milliseconds). A faiss backend can slot in later
behind the same interface if the library outgrows RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import EmbeddingCache


@dataclass
class Hit:
    scene_id: str | None
    key: str
    time: float
    score: float


class SearchIndex:
    def __init__(self, cache: EmbeddingCache, model_name: str):
        self.cache = cache
        self.model_name = model_name
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.keys: list[str] = []
        self.scene_ids: list[str | None] = []
        self.times: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._key_rows: dict[str, tuple[int, int]] = {}  # key -> (start, end)
        self.key_meta: dict[str, dict] = {}  # key -> {scene_id, path}

    @property
    def size(self) -> int:
        return self.matrix.shape[0]

    @property
    def dim(self) -> int:
        return self.matrix.shape[1] if self.matrix.ndim == 2 else 0

    def build(self, keys: list[str] | None = None) -> "SearchIndex":
        """Load every cached scene for this model into one matrix.

        Vectors are L2-normalized at write time, so cosine similarity is just a
        dot product. Stored float32 for fast matmul (the cache is float16).
        """
        keys = keys if keys is not None else self.cache.keys(self.model_name)
        mats, all_keys, all_scenes, all_times = [], [], [], []
        for key in keys:
            try:
                times, vecs, meta = self.cache.load(key, self.model_name)
            except Exception:
                continue
            if vecs.shape[0] == 0:
                continue
            start = sum(m.shape[0] for m in mats)
            mats.append(vecs.astype(np.float32))
            self._key_rows[key] = (start, start + vecs.shape[0])
            all_keys.extend([key] * vecs.shape[0])
            sid = meta.get("scene_id")
            all_scenes.extend([sid] * vecs.shape[0])
            all_times.append(times.astype(np.float32))
            self.key_meta[key] = {"scene_id": sid, "path": meta.get("path")}
        if mats:
            self.matrix = np.concatenate(mats, axis=0)
            self.times = np.concatenate(all_times, axis=0)
        else:
            self.matrix = np.zeros((0, 0), dtype=np.float32)
            self.times = np.zeros((0,), dtype=np.float32)
        self.keys = all_keys
        self.scene_ids = all_scenes
        return self

    # --- querying ------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        top_k: int = 60,
        *,
        per_scene: int | None = None,
        exclude_key: str | None = None,
    ) -> list[Hit]:
        """Top-`top_k` moments most similar to `query` (a 1D vector or (1,d)).

        `per_scene` caps how many hits any single scene contributes, so results
        span the library instead of clustering in one video. `exclude_key`
        drops a scene from the results (e.g. the one you searched *from*).
        """
        if self.size == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        scores = self.matrix @ q  # cosine (rows are unit vectors)

        # take a generous slice, then apply per-scene capping + exclusions
        pool = min(self.size, max(top_k * 8, top_k + 50))
        idx = np.argpartition(-scores, pool - 1)[:pool]
        idx = idx[np.argsort(-scores[idx])]

        hits: list[Hit] = []
        per_scene_counts: dict[str | None, int] = {}
        for i in idx:
            key = self.keys[i]
            if exclude_key is not None and key == exclude_key:
                continue
            sid = self.scene_ids[i]
            if per_scene is not None:
                c = per_scene_counts.get(sid, 0)
                if c >= per_scene:
                    continue
                per_scene_counts[sid] = c + 1
            hits.append(Hit(scene_id=sid, key=key, time=float(self.times[i]), score=float(scores[i])))
            if len(hits) >= top_k:
                break
        return hits

    def vector_at(self, key: str, time: float) -> np.ndarray | None:
        """The stored vector for the frame nearest `time` in scene `key`."""
        rows = self._key_rows.get(key)
        if rows is None:
            return None
        start, end = rows
        seg_times = self.times[start:end]
        if seg_times.size == 0:
            return None
        i = start + int(np.argmin(np.abs(seg_times - time)))
        return self.matrix[i]

    def search_by_frame(
        self, key: str, time: float, top_k: int = 60, *, per_scene: int | None = 3
    ) -> list[Hit]:
        """Find moments like the frame at (key, time). Excludes its own scene."""
        v = self.vector_at(key, time)
        if v is None:
            return []
        return self.search(v, top_k=top_k, per_scene=per_scene, exclude_key=key)
