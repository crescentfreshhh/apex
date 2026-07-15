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

    def _meta_client(self):
        """A short-timeout, no-retry client for NON-critical reads (metadata).
        If Stash is slow/down, these fail fast and the UI degrades to blank
        rather than blocking a search response behind retry backoff."""
        from ..stash_client import StashClient

        c = StashClient(
            url=self.cfg.stash.url, api_key=self.cfg.stash.api_key, timeout=5
        )
        c.RETRY_SLEEPS = ()  # no retries for cosmetic data
        return c

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
        from ..failures import failure_log_for

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
            "failures": len(failure_log_for(self.cfg)),
        }

    def _embedder(self, model: str | None = None):
        from ..embedding import get_embedder

        kwargs = {"device": self.cfg.embedding.device} if self.cfg.embedding.device else {}
        return get_embedder(model or self.cfg.embedding.model, **kwargs)

    # --- embed / score (job targets) ----------------------------------------

    def run_embed(
        self,
        job=None,
        limit: int = 0,
        *,
        model: str | None = None,
        mode: str | None = None,
        interval: float | None = None,
        hwaccel: str | None = None,
        pipeline: str | None = None,
        workers: int | None = None,
        scene_timeout: float | None = None,
    ) -> dict:
        """One incremental embed pass (skips already-cached scenes).

        Every keyword overrides the corresponding config value for this run
        only — so the web UI can pick the model (e.g. a CLIP pass) or tweak
        sampling without touching container env vars. `None` means "use the
        configured default". An empty-string `hwaccel` explicitly forces CPU
        decode (distinct from `None`)."""
        from ..failures import failure_log_for
        from ..pipeline import embed_library
        from ..sampling import FrameSampler

        s, e = self.cfg.sampling, self.cfg.embedding
        log = (job.log if job else print)
        sampler = FrameSampler(
            interval_seconds=(s.interval_seconds if interval is None else interval),
            mode=(s.mode if mode is None else mode),
            hwaccel=(s.hwaccel if hwaccel is None else hwaccel),
            pipeline=(s.pipeline if pipeline is None else pipeline),
            scene_timeout=(s.scene_timeout if scene_timeout is None else scene_timeout),
        )
        embedder = self._embedder(model)
        n_workers = e.workers if workers is None else workers
        cache = EmbeddingCache(e.cache_dir)
        scenes = self.scenes(limit=limit)
        total = len(scenes)
        if job:
            job.progress = {"total": total, "done": 0}
            log(
                f"embed: {total} scene(s) · model={embedder.name} · mode={sampler.mode} "
                f"· interval={sampler.interval:g}s · hwaccel={sampler.hwaccel or 'off'} "
                f"· workers={n_workers}"
            )

        def _log(msg):
            log(msg)
            if job:
                job.progress["done"] = (
                    job.progress.get("done", 0) + (1 if msg.lstrip().startswith("+") else 0)
                )

        stats = embed_library(
            scenes, sampler, embedder, cache,
            batch_size=e.batch_size,
            workers=n_workers,
            total=total, log=_log,
            failure_log=failure_log_for(self.cfg),
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

    def run_sync(self, job=None, prune: bool = True, all_models: bool = True) -> dict:
        """Reconcile the cache with Stash: refresh moved scenes' stored paths
        and (optionally) prune entries for scenes deleted from Stash.

        Fetches the WHOLE library (unscoped) so a scene that moved out of the
        embed scope isn't mistaken for a deletion. A safety guard refuses to
        prune when Stash returns nothing but the cache is non-empty (an
        unreachable/empty response must never wipe the cache)."""
        from ..pipeline import sync_cache

        log = (job.log if job else print)
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        models = cache.models() if all_models else [canonical_name(self.cfg.embedding.model)]
        models = [m for m in models if m]
        # one scene fetch, reused across models
        scenes = list(self.client().iter_scenes())
        total = {"cached": 0, "moved": 0, "orphaned": 0, "pruned": 0}
        safe_prune = prune
        if prune and not scenes and any(cache.keys(m) for m in models):
            log("  ! Stash returned no scenes — skipping prune (cache left intact)")
            safe_prune = False
        for model in models:
            log(f"sync: model {model} ({len(scenes)} live scenes)")
            s = sync_cache(scenes, cache, model, prune=safe_prune, log=log)
            for k in total:
                total[k] += s.get(k, 0)
            self.invalidate_index(model)
        self.invalidate_meta()
        total["models"] = len(models)
        return total

    def run_fix(self, job=None, limit: int = 0, dry_run: bool = False) -> dict:
        """Retry scenes recorded in the failure log through a fallback ladder.

        Most sparse-mode casualties are seek/NVDEC quirks, not broken files, so
        we re-attempt each: first sparse with NVDEC off, then a full LINEAR
        decode (the path VLC/Stash use) which tolerates awkward seek tables.
        A scene that embeds under any strategy is cleared from the log; one that
        exhausts them stays, its entry updated with the last error."""
        from ..failures import failure_log_for
        from ..pipeline import embed_library
        from ..sampling import FrameSampler

        log = (job.log if job else print)
        flog = failure_log_for(self.cfg)
        entries = flog.entries()
        if limit:
            entries = entries[:limit]
        result = {"fixed": 0, "failed": 0, "total": len(entries)}
        if not entries:
            log("no recorded failures — nothing to fix")
            return result
        if job:
            job.progress = {"total": len(entries), "done": 0}

        # (mode, hwaccel, pipeline): distinct decode strategies, cheap → tolerant
        ladder = [
            ("sparse", "", "raw"),      # sparse seek, no NVDEC
            ("interval", "", "jpeg"),   # full linear decode, most forgiving
        ]
        if dry_run:
            for e in entries:
                log(f"  · would retry scene {e.get('scene_id')} ({e.get('path')})")
            return result

        import os

        embedder = self._embedder()
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        iv = self.cfg.sampling.interval_seconds
        to = self.cfg.sampling.scene_timeout
        for e in entries:
            key = e["key"]
            scene = _scene_from_entry(e)
            if not scene.path or not os.path.exists(scene.path):
                log(f"  ? scene {e.get('scene_id')}: file missing at {scene.path} "
                    "(moved/deleted? run sync) — skipped")
                result["failed"] += 1
            else:
                ok = False
                last = ""
                for mode, hw, pipe in ladder:
                    sampler = FrameSampler(
                        interval_seconds=iv, mode=mode, hwaccel=hw,
                        pipeline=pipe, scene_timeout=to,
                    )
                    lines: list[str] = []
                    try:
                        st = embed_library(
                            [scene], sampler, embedder, cache,
                            batch_size=self.cfg.embedding.batch_size,
                            total=1, log=lines.append,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st = {"embedded": 0}
                        lines.append(f"failed: {exc}")
                    if st.get("embedded") == 1:
                        ok = True
                        log(f"  ✓ scene {scene.id}: fixed via "
                            f"{mode}/hwaccel={hw or 'off'}/{pipe}")
                        break
                    last = next((ln for ln in lines if "fail" in ln.lower()), "")
                    log(f"    · {mode}/{hw or 'off'}/{pipe} didn't work")
                if ok:
                    flog.resolve(key)
                    result["fixed"] += 1
                    self.invalidate_index(embedder.name)
                else:
                    flog.record(
                        key, e.get("scene_id"), scene.path,
                        error=last or "all fallback strategies failed",
                        mode="fix-exhausted", model=embedder.name,
                    )
                    result["failed"] += 1
            if job:
                job.progress["done"] = result["fixed"] + result["failed"]
        return result

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
                fetched = self._meta_client().scene_details(missing)
            except Exception:
                fetched = {}
            with self._meta_lock:
                for sid in missing:
                    self._meta[sid] = fetched.get(sid, {})
        return {s: self._meta.get(s, {}) for s in want}

    def invalidate_meta(self, scene_id: str | None = None) -> None:
        with self._meta_lock:
            if scene_id is None:
                self._meta.clear()
            else:
                self._meta.pop(str(scene_id), None)

    def update_scene(self, scene_id: str, **fields) -> dict:
        """Write editable fields to Stash, then return fresh metadata."""
        self.client().update_scene(scene_id, **fields)
        self.invalidate_meta(scene_id)
        return self.scene_meta([scene_id]).get(str(scene_id), {})

    def add_o(self, scene_id: str) -> int:
        count = self.client().scene_add_o(scene_id)
        self.invalidate_meta(scene_id)
        return count

    def remove_o(self, scene_id: str) -> int:
        count = self.client().scene_delete_o(scene_id)
        self.invalidate_meta(scene_id)
        return count

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


def _scene_from_entry(entry: dict):
    """Rebuild a minimal Scene from a failure-log record so it can be re-fed to
    embed_library. The cache key is the file fingerprint (unless it was a
    path-derived fallback key), so reconstructing it here keeps the retry
    writing to the same cache entry."""
    from ..models import Scene

    key = entry.get("key", "")
    fps = [] if key.startswith("path-") else [{"type": "oshash", "value": key}]
    return Scene.from_dict(
        {
            "id": entry.get("scene_id") or "",
            "title": "",
            "files": [{"path": entry.get("path") or "", "fingerprints": fps}],
            "scene_markers": [],
        }
    )


def get_embedder_for_references(cfg: Config):
    from ..embedding import get_embedder

    kwargs = {"device": cfg.embedding.device} if cfg.embedding.device else {}
    return get_embedder(cfg.embedding.model, **kwargs)
