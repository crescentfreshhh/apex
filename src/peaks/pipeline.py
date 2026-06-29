"""Orchestration: the two passes that tie the pieces together.

  embed_library  — Stash scenes → sampled frames → embeddings → on-disk cache.
                   The GPU-heavy, resumable, one-time pass.

  score_library  — cached embeddings → similarity vs your references → segments
                   → Stash `apex` markers (or a dry-run preview).

These need ffmpeg + the `[ml]` extra + a live Stash to actually run; the
building blocks they call (sampler, embedder, cache, scorer) are unit-tested
offline. Kept deliberately thin so the logic lives in the tested modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .cache import EmbeddingCache, path_key
from .embedding import Embedder
from .models import Scene
from .sampling import FrameSampler
from .scoring import Segment, extract_segments, similarity_scores, smooth

Logger = Callable[[str], None]


def scene_key(scene: Scene) -> str:
    """Cache key for a scene: prefer the file fingerprint, else hash the path."""
    return scene.fingerprint or path_key(scene.path or scene.id)


def _embed_in_batches(
    embedder: Embedder, images: list, batch_size: int
) -> np.ndarray:
    if not images:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    chunks = []
    for i in range(0, len(images), batch_size):
        chunks.append(embedder.embed_images(images[i : i + batch_size]))
    return np.concatenate(chunks, axis=0)


def embed_library(
    scenes: Iterable[Scene],
    sampler: FrameSampler,
    embedder: Embedder,
    cache: EmbeddingCache,
    *,
    batch_size: int = 64,
    log: Logger = print,
) -> dict:
    """Embed every scene not already cached. Resumable + idempotent."""
    stats = {"embedded": 0, "skipped": 0, "failed": 0, "frames": 0}
    for scene in scenes:
        key = scene_key(scene)
        if cache.has(key, embedder.name):
            stats["skipped"] += 1
            continue
        if not scene.path:
            log(f"  ! scene {scene.id} has no file; skipping")
            stats["failed"] += 1
            continue
        try:
            times, frames = [], []
            for ts, img in sampler.iter_frames(scene.path):
                times.append(ts)
                frames.append(img.copy())  # detach from the temp-file context
            vecs = _embed_in_batches(embedder, frames, batch_size)
            cache.save(
                key,
                embedder.name,
                np.asarray(times, dtype=np.float32),
                vecs,
                meta={
                    "scene_id": scene.id,
                    "path": scene.path,
                    "interval": sampler.interval,
                    "model": embedder.name,
                    "dim": embedder.dim,
                    "n_frames": len(times),
                },
            )
            stats["embedded"] += 1
            stats["frames"] += len(times)
            log(f"  + scene {scene.id}: {len(times)} frames -> cache")
        except Exception as exc:  # keep the batch going; log the casualty
            log(f"  ! scene {scene.id} failed: {exc}")
            stats["failed"] += 1
    return stats


def load_references(embedder: Embedder, references_dir: str | Path) -> np.ndarray:
    """Embed every image in a directory into reference vectors (m, dim)."""
    from PIL import Image as PILImage  # lazy

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = sorted(
        p for p in Path(references_dir).glob("**/*") if p.suffix.lower() in exts
    )
    if not files:
        raise FileNotFoundError(f"no reference images found in {references_dir}")
    images = [PILImage.open(p).convert("RGB") for p in files]
    return embedder.embed_images(images)


def score_scene(
    times: np.ndarray,
    vecs: np.ndarray,
    references: np.ndarray,
    scoring,
) -> list[Segment]:
    """Pure scoring for one scene's cached embeddings (no I/O)."""
    scores = similarity_scores(vecs, references, reduce=scoring.reduce)
    scores = smooth(scores, scoring.smooth_window)
    return extract_segments(
        scores,
        times,
        high=scoring.high,
        low=scoring.low,
        min_duration=scoring.min_duration,
        merge_gap=scoring.merge_gap,
        max_duration=scoring.max_duration or None,
        pad=scoring.pad,
    )


def score_library(
    scenes: Iterable[Scene],
    cache: EmbeddingCache,
    embedder_name: str,
    references: np.ndarray,
    scoring,
    *,
    client=None,
    tag_name: str = "apex",
    write: bool = False,
    log: Logger = print,
) -> dict:
    """Score cached scenes into segments; optionally write Stash markers.

    write=False is a dry run: it logs the segments it *would* create, perfect
    for tuning thresholds before touching Stash.
    """
    stats = {"scenes": 0, "segments": 0, "skipped": 0}
    tag = None
    if write:
        if client is None:
            raise ValueError("write=True requires a client")
        tag = client.find_or_create_tag(tag_name)

    for scene in scenes:
        key = scene_key(scene)
        if not cache.has(key, embedder_name):
            stats["skipped"] += 1
            continue
        times, vecs, _ = cache.load(key, embedder_name)
        segs = score_scene(times, vecs, references, scoring)
        stats["scenes"] += 1
        stats["segments"] += len(segs)
        for s in segs:
            if write:
                client.create_scene_marker(
                    scene_id=scene.id,
                    seconds=s.start,
                    primary_tag_id=tag.id,
                    title=tag_name,
                    end_seconds=s.end,
                )
            else:
                log(
                    f"  ~ scene {scene.id}: {s.start:7.1f}-{s.end:7.1f}s "
                    f"peak={s.peak_score:.3f}"
                )
    return stats
