"""High-level operations shared by the CLI, the web API, and the scheduler.

Keeps orchestration in one place: running an (incremental) embed pass, scoring,
building/caching the search index, CLIP text queries, and on-demand frame
thumbnails. The web layer stays thin glue over this.
"""

from __future__ import annotations

import threading
from io import BytesIO

import numpy as np

from ..cache import EmbeddingCache
from ..config import Config
from ..embedding import canonical_name
from ..search import Hit, SearchIndex


class Service:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.load()
        self._index: dict[str, SearchIndex] = {}
        self._index_lock = threading.Lock()
        self._clip = None  # lazily-loaded CLIP embedder for text queries
        self._clip_lock = threading.Lock()
        self._meta: dict[str, dict] = {}  # scene_id -> Stash display metadata
        self._meta_lock = threading.Lock()

    # --- library / scenes ----------------------------------------------------

    def client(self):
        from ..stash_client import StashClient

        return StashClient.from_config(self.cfg)

    def scenes(self, limit: int = 0):
        prefix = self.cfg.library.path
        it = self.client().iter_scenes(path_prefix=prefix)
        out = []
        for s in it:
            out.append(s)
            if limit and len(out) >= limit:
                break
        return out

    def stats(self) -> dict:
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        model = canonical_name(self.cfg.embedding.model)
        cached = len(cache.keys(model))
        return {
            "library_path": self.cfg.library.path or "(whole library)",
            "model": model,
            "cached_scenes": cached,
            "device": self.cfg.embedding.device or "auto",
            "interval": self.cfg.sampling.interval_seconds,
            "mode": self.cfg.sampling.mode,
        }

    # --- embed / score (job targets) ----------------------------------------

    def run_embed(self, job=None, limit: int = 0) -> dict:
        """One incremental embed pass (skips already-cached scenes)."""
        from ..pipeline import embed_library
        from ..sampling import FrameSampler

        log = (job.log if job else print)
        sampler = FrameSampler(
            interval_seconds=self.cfg.sampling.interval_seconds,
            mode=self.cfg.sampling.mode,
            hwaccel=self.cfg.sampling.hwaccel,
            pipeline=self.cfg.sampling.pipeline,
            scene_timeout=self.cfg.sampling.scene_timeout,
        )
        from ..embedding import get_embedder

        kwargs = {"device": self.cfg.embedding.device} if self.cfg.embedding.device else {}
        embedder = get_embedder(self.cfg.embedding.model, **kwargs)
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        scenes = self.scenes(limit=limit)
        total = len(scenes)
        if job:
            job.progress = {"total": total, "done": 0}

        def _log(msg):
            log(msg)
            if job:
                job.progress["done"] = (
                    job.progress.get("done", 0) + (1 if msg.lstrip().startswith("+") else 0)
                )

        stats = embed_library(
            scenes, sampler, embedder, cache,
            batch_size=self.cfg.embedding.batch_size,
            workers=self.cfg.embedding.workers,
            total=total, log=_log,
        )
        self.invalidate_index(embedder.name)
        return stats

    def run_score(self, job=None, tag: str | None = None, write: bool = False) -> dict:
        from ..pipeline import (
            load_references,
            resolve_references_dir,
            score_library,
        )
        from ..scoring import make_similarity_scorer

        log = (job.log if job else print)
        tag = tag or self.cfg.markers.tag_name
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        model = canonical_name(self.cfg.embedding.model)
        # model file first, else references
        from pathlib import Path

        from ..classifier import TasteClassifier
        from ..pipeline import safe_tag

        model_path = Path(self.cfg.modeling.dir) / f"{safe_tag(tag)}.pkl"
        if model_path.exists():
            clf = TasteClassifier.load(model_path)
            score_frames = clf.predict_proba
            log(f"scoring with model {model_path.name}")
        else:
            refs_dir = resolve_references_dir(self.cfg.scoring.references_dir, tag)
            embedder = get_embedder_for_references(self.cfg)
            refs = load_references(embedder, refs_dir)
            score_frames = make_similarity_scorer(refs, self.cfg.scoring.reduce)
            log(f"scoring with {refs.shape[0]} references from {refs_dir}")

        client = self.client() if write else None
        return score_library(
            self.scenes(), cache, model, score_frames, self.cfg.scoring,
            client=client, tag_name=tag, write=write, log=log,
        )

    # --- search index --------------------------------------------------------

    def index(self, model: str | None = None) -> SearchIndex:
        model = model or canonical_name(self.cfg.embedding.model)
        with self._index_lock:
            if model not in self._index:
                cache = EmbeddingCache(self.cfg.embedding.cache_dir)
                self._index[model] = SearchIndex(cache, model).build()
            return self._index[model]

    def invalidate_index(self, model: str | None = None) -> None:
        with self._index_lock:
            if model is None:
                self._index.clear()
            else:
                self._index.pop(model, None)

    def search_by_frame(self, key: str, time: float, top_k: int = 60) -> list[Hit]:
        return self.index().search_by_frame(key, time, top_k=top_k)

    def search_text(self, text: str, top_k: int = 60) -> list[Hit]:
        """CLIP text -> nearest moments in the CLIP index."""
        vec = self._clip_text_vector(text)
        return self.index("clip").search(vec, top_k=top_k, per_scene=3)

    def has_clip_index(self) -> bool:
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        return len(cache.keys("clip")) > 0

    def _clip_text_vector(self, text: str) -> np.ndarray:
        with self._clip_lock:
            if self._clip is None:
                from ..embedding import ClipEmbedder

                self._clip = ClipEmbedder(
                    device=self.cfg.embedding.device or None
                )
            return self._clip.embed_text([text])[0]

    # --- scene metadata (titles, performers, studio, tags) -------------------

    def scene_meta(self, scene_ids: list[str]) -> dict[str, dict]:
        """Display metadata from Stash, cached per scene id. Network failures
        degrade gracefully to {} so the UI still renders thumbnails."""
        want = [s for s in {str(i) for i in scene_ids if i}]
        missing = [s for s in want if s not in self._meta]
        if missing:
            try:
                fetched = self.client().scene_details(missing)
            except Exception:
                fetched = {}
            with self._meta_lock:
                for sid in missing:
                    self._meta[sid] = fetched.get(sid, {})
        return {s: self._meta.get(s, {}) for s in want}

    def invalidate_meta(self) -> None:
        with self._meta_lock:
            self._meta.clear()

    # --- thumbnails ----------------------------------------------------------

    def frame_jpeg(self, path: str, time: float, size: int = 320) -> bytes:
        """A single JPEG thumbnail at (path, time), decoded on demand."""
        from PIL import Image  # lazy

        from ..sampling import FrameSampler

        sampler = FrameSampler(hwaccel=self.cfg.sampling.hwaccel)
        img: Image.Image = sampler.grab_frame(path, time)
        img.thumbnail((size, size))
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=82)
        return buf.getvalue()

    def path_for_key(self, key: str) -> str | None:
        meta = self.index().key_meta.get(key)
        return meta.get("path") if meta else None

    def stream_url(self, scene_id: str, start: float | None = None) -> str:
        return self.client().stream_url(scene_id, start=start)


def get_embedder_for_references(cfg: Config):
    from ..embedding import get_embedder

    kwargs = {"device": cfg.embedding.device} if cfg.embedding.device else {}
    return get_embedder(cfg.embedding.model, **kwargs)
