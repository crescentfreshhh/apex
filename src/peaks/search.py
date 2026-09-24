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

import os
import sys
from dataclasses import dataclass

import numpy as np

from .cache import EmbeddingCache

# In-RAM storage for the stacked matrix.
#
# A 2s-sampled library is millions of frames: DINOv2-L at ~4M frames is ~16 GB
# as float32 — too much for a 24 GB container once torch/CLIP/etc. are resident.
# Large indexes are therefore stored as **bfloat16** (2 bytes/value, stored as the
# upper 16 bits of each float32 in a uint16 array): half the RAM, and on measured
# data the cosine error is ≤3e-4 with 99.9% top-1000 overlap vs float32.
# bfloat16 (not float16) because numpy widens it to float32 with a cheap
# bit-placement copy (~2.5x faster than numpy's float16 cast on the same box).
# Search on a bf16 index is still slower than a single float32 BLAS matvec
# (conversion-bound), so small indexes stay float32.
#
# PEAKS_INDEX_DTYPE = auto (default) | float32 | bfloat16
#   auto → float32 unless the float32 matrix would exceed PEAKS_INDEX_F32_MAX_MB
#          (default 6144), then bfloat16.
# All math goes through `dot` / `apply` / `rows` / `take` / `vector_at`, which
# hand back float32 in bounded chunks — never touch `matrix` numerically.
_CHUNK = 8192  # rows per float32 working chunk (32 MB at 1024-dim)
_LITTLE = sys.byteorder == "little"


def _index_dtype_for(total: int, dim: int) -> str:
    want = os.environ.get("PEAKS_INDEX_DTYPE", "auto").lower()
    if want in ("float32", "f32"):
        return "float32"
    if want in ("bfloat16", "bf16", "float16", "f16", "half"):
        return "bfloat16"
    try:
        cap_mb = float(os.environ.get("PEAKS_INDEX_F32_MAX_MB", "6144"))
    except ValueError:
        cap_mb = 6144.0
    return "bfloat16" if total * dim * 4 > cap_mb * 1024 * 1024 else "float32"


def _f32_to_bf16(a: np.ndarray) -> np.ndarray:
    """float32 → bfloat16 bits (uint16), round-to-nearest-even."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_into_f32(bits: np.ndarray, buf: np.ndarray) -> np.ndarray:
    """Widen bf16 bits (n, d) into the float32 buffer `buf` (≥ n rows, low halves
    zero) and return the float32 view of the first n rows."""
    n = bits.shape[0]
    b = buf[:n]
    if _LITTLE:   # write each bf16 into the HIGH half of its float32 slot
        b.view(np.uint16).reshape(n, bits.shape[1], 2)[..., 1] = bits
    else:
        b.view(np.uint32)[...] = bits.astype(np.uint32) << 16
    return b


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
        self._bf16 = False  # storage: bfloat16 bits (uint16) vs float32

    @property
    def size(self) -> int:
        return self.matrix.shape[0]

    @property
    def dim(self) -> int:
        return self.matrix.shape[1] if self.matrix.ndim == 2 else 0

    # --- dtype-safe access ---------------------------------------------------
    # `matrix` is raw storage: float32, or bfloat16 bits in a uint16 array (see
    # module docstring). Never do math on it directly — use these, which always
    # return float32.

    @property
    def storage(self) -> str:
        return "bfloat16" if self._bf16 else "float32"

    def _f32(self, block: np.ndarray) -> np.ndarray:
        if not self._bf16:
            return np.asarray(block, dtype=np.float32)
        buf = np.zeros(block.shape, dtype=np.float32)
        return _bf16_into_f32(block, buf)

    def rows(self, start: int, end: int) -> np.ndarray:
        """float32 rows [start, end) — for per-scene blocks."""
        return self._f32(self.matrix[start:end])

    def take(self, idx) -> np.ndarray:
        """float32 copy of the given row indices."""
        return self._f32(self.matrix[idx])

    def apply(self, fn, chunk: int = _CHUNK) -> np.ndarray:
        """Apply row-wise `fn` to the whole matrix in bounded float32 chunks and
        concatenate the results along axis 0. A float32 index is passed through
        whole (no copy); a bf16 index is widened chunk by chunk into one reused
        buffer, so no whole-matrix float32/float64 copy ever exists."""
        n = self.size
        if n == 0:
            return np.zeros((0,), dtype=np.float32)
        if not self._bf16:
            return np.asarray(fn(self.matrix))
        buf = np.zeros((min(chunk, n), self.dim), dtype=np.float32)
        parts = []
        for s in range(0, n, chunk):
            blk = _bf16_into_f32(self.matrix[s : s + chunk], buf)
            parts.append(np.array(fn(blk)))   # copy: buf is reused next chunk
        return np.concatenate(parts, axis=0)

    def dot(self, q: np.ndarray) -> np.ndarray:
        """`matrix @ q` as float32 (q: (d,) → (n,), or (d,k) → (n,k))."""
        q = np.asarray(q, dtype=np.float32)
        return self.apply(lambda b: b @ q).astype(np.float32, copy=False)

    def build(self, keys: list[str] | None = None) -> "SearchIndex":
        """Load every cached scene for this model into one matrix.

        Vectors are L2-normalized at write time, so cosine similarity is just a
        dot product. Stored float32, or bfloat16 for large indexes (see
        `_index_dtype_for`); use `dot`/`apply`/`rows` for math — always float32.

        The matrix is **preallocated** from a cheap first pass (frame counts via
        `peek_count`, no vectors) and filled in place, so building a
        whole-library index needs one copy of the data — not the ~2x transient a
        per-scene `mats` list plus `np.concatenate` would hold. For a large
        library (millions of frames) that halves the peak RAM of a rebuild.
        """
        keys = keys if keys is not None else self.cache.keys(self.model_name)

        # pass 1: cheap frame counts (reads only the tiny `times` arrays) to
        # size the matrix without a transient second copy.
        kept: list[str] = []
        counts: list[int] = []
        for key in keys:
            try:
                n = self.cache.peek_count(key, self.model_name)
            except Exception:
                continue
            if n > 0:
                kept.append(key)
                counts.append(n)

        total = sum(counts)
        if total == 0:
            self.matrix = np.zeros((0, 0), dtype=np.float32)
            self.times = np.zeros((0,), dtype=np.float32)
            self.keys = []
            self.scene_ids = []
            return self

        # dim from the first kept scene, then fill the preallocated matrix.
        _, first_vecs, _ = self.cache.load(kept[0], self.model_name)
        dim = int(first_vecs.shape[1])
        self._bf16 = _index_dtype_for(total, dim) == "bfloat16"
        matrix = np.empty((total, dim), dtype=np.uint16 if self._bf16 else np.float32)
        times_all = np.empty((total,), dtype=np.float32)
        all_keys: list[str] = []
        all_scenes: list[str | None] = []
        pos = 0
        for key in kept:
            try:
                times, vecs, meta = self.cache.load(key, self.model_name)
            except Exception:
                continue
            n = vecs.shape[0]
            if n == 0:
                continue
            if pos + n > total:  # count grew between passes (mid-embed write)
                n = total - pos
                vecs, times = vecs[:n], times[:n]
            matrix[pos : pos + n] = _f32_to_bf16(vecs) if self._bf16 else vecs
            times_all[pos : pos + n] = times.astype(np.float32)
            self._key_rows[key] = (pos, pos + n)
            all_keys.extend([key] * n)
            sid = meta.get("scene_id")
            all_scenes.extend([sid] * n)
            self.key_meta[key] = {"scene_id": sid, "path": meta.get("path")}
            pos += n
            if pos >= total:
                break

        # trim if the live cache shrank between passes (any count drift)
        self.matrix = matrix[:pos]
        self.times = times_all[:pos]
        self.keys = all_keys
        self.scene_ids = all_scenes
        return self

    # --- querying ------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        top_k: int | None = 60,
        *,
        per_scene: int | None = None,
        exclude_key: str | None = None,
        min_score: float | None = None,
    ) -> list[Hit]:
        """Moments most similar to `query` (a 1D vector or (1,d)).

        `top_k` caps how many hits come back; pass `None` for *unbounded* (every
        qualifying moment). `min_score` returns **all** frames whose cosine is at
        least that — the "match strength" floor for Explore — instead of a fixed
        top slice. `per_scene` caps how many hits any single scene contributes, so
        results span the library instead of clustering in one video. `exclude_key`
        drops a scene from the results (e.g. the one you searched *from*).
        """
        if self.size == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        scores = self.dot(q)  # cosine (rows are unit vectors); chunked float32

        if min_score is not None:
            # every qualifying frame, ordered best-first (all scores already computed)
            idx = np.where(scores >= min_score)[0]
            idx = idx[np.argsort(-scores[idx])]
        else:
            # bounded top-k: partition a generous pool, then order just that slice
            pool = min(self.size, max((top_k or 60) * 8, (top_k or 60) + 50))
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
            if top_k is not None and len(hits) >= top_k:
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
        return self.rows(i, i + 1)[0]

    def vector_around(self, key: str, time: float, window: float) -> np.ndarray | None:
        """A 'peak-level' vector for the moment at (key, time): the *mean* of the
        frames within ±`window` seconds, re-normalized to unit length — so a peak
        is matched by its whole gist, not one (possibly fluky) midpoint frame.
        Always includes the nearest frame, so a sparse/edge moment still returns a
        vector; `window`<=0 or a single frame degrades to `vector_at`."""
        rows = self._key_rows.get(key)
        if rows is None:
            return None
        start, end = rows
        seg_times = self.times[start:end]
        if seg_times.size == 0:
            return None
        if window <= 0:
            return self.vector_at(key, time)
        mask = np.abs(seg_times - time) <= window
        if not mask.any():  # window fell between samples → nearest frame
            mask[int(np.argmin(np.abs(seg_times - time)))] = True
        block = self.rows(start, end)[mask]
        v = block.mean(axis=0).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def clip_span(
        self, key: str, time: float, *,
        similarity: float = 0.5, min_dur: float = 3.0, max_dur: float = 30.0,
        interval: float = 2.0,
    ) -> tuple[float, float]:
        """Smart clip bounds for the moment at (key, time): start AT the moment and
        grow the end forward while each following sampled frame still resembles the
        moment (cosine to the anchor frame ≥ `similarity`), stopping at the first
        drastically-different frame (a shot change / drift). The length is clamped
        to [`min_dur`, `max_dur`]. The tail added after the last similar frame is
        that scene's OWN sampling step (median gap of its timestamps), so a 2s scene
        gets a 2s tail even in a library half-converted from 8s; `interval` is only
        the fallback for single-frame scenes. Falls back to a fixed
        min(max_dur, 20)s clip when the scene isn't embedded or `similarity`<=0.
        Returns (start, end) seconds; start is the moment itself so the stream
        offset is unchanged."""
        fixed = min(max_dur, 20.0) if max_dur else 20.0
        rows = self._key_rows.get(key)
        if similarity <= 0 or rows is None:
            return (round(float(time), 3), round(float(time) + fixed, 3))
        start_i, end_i = rows
        seg_times = self.times[start_i:end_i]
        n = int(seg_times.size)
        if n == 0:
            return (round(float(time), 3), round(float(time) + fixed, 3))
        step = float(np.median(np.diff(seg_times))) if n > 1 else float(interval)
        if not step > 0:
            step = float(interval)
        block = self.rows(start_i, end_i)          # float32; unit rows → dot = cosine
        ai = int(np.argmin(np.abs(seg_times - time)))
        sims = block @ block[ai]                  # cosine of every frame to the anchor
        j = ai
        while j + 1 < n and float(sims[j + 1]) >= similarity:
            j += 1
        end = float(seg_times[j]) + step           # cover the last similar frame
        dur = end - float(time)
        dur = max(min_dur, min(dur, max_dur)) if max_dur else max(min_dur, dur)
        return (round(float(time), 3), round(float(time) + dur, 3))

    def search_by_frame(
        self, key: str, time: float, top_k: int | None = 60, *,
        per_scene: int | None = 3, min_score: float | None = None,
    ) -> list[Hit]:
        """Find moments like the frame at (key, time). Excludes its own scene."""
        v = self.vector_at(key, time)
        if v is None:
            return []
        return self.search(
            v, top_k=top_k, per_scene=per_scene, exclude_key=key, min_score=min_score
        )
