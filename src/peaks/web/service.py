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
        self._taste: dict[str, object] = {}  # taste classifiers, keyed by file
        self._taste_lock = threading.Lock()
        self._vocab_cache = None  # (labels, CLIP-text matrix) for classification

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
            should_stop=(lambda: job.cancelled) if job else None,
        )
        self.invalidate_index(embedder.name)
        return stats

    def run_score(
        self,
        job=None,
        tag: str | None = None,
        write: bool = False,
        *,
        high: float | None = None,
        low: float | None = None,
        reduce: str | None = None,
        max_duration: float | None = None,
        normalize: str | None = None,
    ) -> dict:
        """Score cached scenes into apex segments; write markers when asked.

        `high`/`low`/`reduce` override the configured scoring for this run only
        (so thresholds can be tuned from the GUI). A dry run (write=False) also
        logs a calibration read-out — the actual frame-score distribution — so
        you can pick thresholds that match your library instead of guessing. On
        a write run the megaboard playlist is rebuilt automatically."""
        from dataclasses import replace
        from pathlib import Path

        from ..classifier import TasteClassifier
        from ..pipeline import (
            load_references,
            resolve_references_dir,
            safe_tag,
            score_library,
        )
        from ..scoring import make_similarity_scorer

        log = (job.log if job else print)
        tag = tag or self.cfg.markers.tag_name
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        model = canonical_name(self.cfg.embedding.model)

        overrides = {
            k: v
            for k, v in {
                "high": high, "low": low,
                "max_duration": max_duration, "normalize": normalize,
            }.items()
            if v is not None
        }
        scoring = replace(self.cfg.scoring, **overrides) if overrides else self.cfg.scoring

        model_path = Path(self.cfg.modeling.dir) / f"{safe_tag(tag)}.pkl"
        if model_path.exists():
            clf = TasteClassifier.load(model_path)
            score_frames = clf.predict_proba
            log(f"scoring with model {model_path.name}")
        else:
            refs_dir = resolve_references_dir(self.cfg.scoring.references_dir, tag)
            embedder = get_embedder_for_references(self.cfg)
            refs = load_references(embedder, refs_dir)
            score_frames = make_similarity_scorer(refs, reduce or self.cfg.scoring.reduce)
            log(f"scoring with {refs.shape[0]} references from {refs_dir}")

        if not write:
            self._log_score_calibration(cache, model, score_frames, scoring, log)

        client = self.client() if write else None
        stats = score_library(
            self.scenes(), cache, model, score_frames, scoring,
            client=client, tag_name=tag, write=write, log=log,
            should_stop=(lambda: job.cancelled) if job else None,
        )
        if write:
            pl = self.run_playlist(tags=[tag], log=log)
            stats["playlist"] = pl["count"]
        return stats

    def _log_score_calibration(
        self, cache, model, score_frames, scoring, log, sample: int = 300
    ) -> None:
        """Report the real frame-score distribution so thresholds aren't a
        guess. This is the antidote to a silent "0 segments": it shows whether
        anything reaches `high`, and suggests values that would."""
        import random

        import numpy as np

        from ..scoring import normalize_scores, smooth

        keys = cache.keys(model)
        if not keys:
            log(f"  (no cached embeddings for model '{model}' — embed first)")
            return
        pick = keys if len(keys) <= sample else random.Random(0).sample(keys, sample)
        chunks = []
        for k in pick:
            try:
                _, vecs, _ = cache.load(k, model)
            except Exception:
                continue
            if vecs.shape[0] == 0:
                continue
            s = normalize_scores(score_frames(vecs), getattr(scoring, "normalize", "none"))
            chunks.append(smooth(np.asarray(s, dtype=np.float32), scoring.smooth_window))
        if not chunks:
            return
        arr = np.concatenate(chunks)

        def p(q):
            return float(np.percentile(arr, q))

        over = float((arr >= scoring.high).mean())
        sug_high, sug_low = round(p(99.0), 3), round(p(97.0), 3)
        log(
            f"  calibration · {len(pick)} scenes / {arr.size} frames "
            f"(normalize={getattr(scoring, 'normalize', 'none')}):"
        )
        log(
            f"    frame score  p50={p(50):.3f}  p90={p(90):.3f}  "
            f"p99={p(99):.3f}  max={float(arr.max()):.3f}"
        )
        log(
            f"    current high={scoring.high} low={scoring.low} "
            f"→ {over * 100:.2f}% of frames qualify"
        )
        if over == 0:
            log(
                f"    ⚠ nothing reaches high={scoring.high}. Try high≈{sug_high} "
                f"low≈{sug_low} in Score → Advanced (or lower further)."
            )
        else:
            log(f"    (to tighten/loosen, try high≈{sug_high} low≈{sug_low})")

    def export_reel(
        self, job=None, tag: str | None = None, limit: int = 0, name: str | None = None
    ) -> dict:
        """Concatenate a tag's apex clips into one video (fast stream-copy).

        Reads the scene files directly off the mounted (read-only) library and
        copies each [start,end] segment without re-encoding, then concats them.
        Stream-copy is fast but needs codec-compatible sources; clips that can't
        be copied are skipped and reported. Output lands in the exports dir."""
        import os
        import subprocess
        import tempfile
        import time as _t
        from pathlib import Path

        log = (job.log if job else print)
        tag = tag or self.cfg.markers.tag_name
        client = self.client()
        apexes = [m for m in client.iter_markers_by_tag(tag) if m["scene_id"]]
        if limit:
            apexes = apexes[:limit]
        if not apexes:
            log(f"no '{tag}' markers to export")
            return {"clips": 0}

        details = client.scene_details(sorted({a["scene_id"] for a in apexes}))
        exports = Path(os.environ.get("PEAKS_EXPORT_DIR", "/config/exports"))
        exports.mkdir(parents=True, exist_ok=True)
        name = _safe_reel_name(name or f"reel-{tag}-{_t.strftime('%Y%m%d-%H%M%S')}") + ".mp4"
        out = exports / name
        if job:
            job.progress = {"total": len(apexes), "done": 0}

        with tempfile.TemporaryDirectory() as td:
            segs: list[str] = []
            for i, a in enumerate(apexes):
                if job and job.cancelled:
                    log(f"  ⏹ stop requested — halting after {len(segs)} clips")
                    break
                d = details.get(str(a["scene_id"])) or {}
                path = d.get("path")
                if not path or not os.path.exists(path):
                    log(f"  ! scene {a['scene_id']}: file missing — skipped")
                    continue
                start = float(a["seconds"])
                end = float(a["end_seconds"]) if a.get("end_seconds") else start + 15.0
                seg = os.path.join(td, f"seg{i:04d}.ts")
                r = subprocess.run(
                    ["ffmpeg", "-y", "-ss", f"{start:g}", "-to", f"{end:g}", "-i", path,
                     "-c", "copy", "-f", "mpegts", seg],
                    capture_output=True,
                )
                if r.returncode == 0 and os.path.exists(seg) and os.path.getsize(seg) > 0:
                    segs.append(seg)
                    if job:
                        job.progress["done"] = len(segs)
                    log(f"  + clip {len(segs)}: scene {a['scene_id']} {start:.0f}-{end:.0f}s")
                else:
                    log(f"  ! scene {a['scene_id']} clip failed (codec mismatch?) — skipped")
            if not segs:
                log("no clips extracted")
                return {"clips": 0}
            listf = os.path.join(td, "list.txt")
            with open(listf, "w") as f:
                for s in segs:
                    f.write(f"file '{s}'\n")
            cc = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listf, "-c", "copy", str(out)],
                capture_output=True,
            )
            if cc.returncode != 0:
                raise RuntimeError("concat failed: " + cc.stderr.decode("replace")[-300:])
        size = out.stat().st_size if out.exists() else 0
        log(f"reel: {len(segs)} clips → {out} ({size // 1_000_000} MB)")
        return {"clips": len(segs), "name": name, "path": str(out), "bytes": size}

    def reels(self) -> list[dict]:
        """List exported reels (newest first)."""
        import os
        from pathlib import Path

        exports = Path(os.environ.get("PEAKS_EXPORT_DIR", "/config/exports"))
        if not exports.is_dir():
            return []
        files = [p for p in exports.glob("*.mp4") if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"name": p.name, "bytes": p.stat().st_size} for p in files]

    def reel_path(self, name: str) -> str | None:
        """Absolute path of an export, or None if the name escapes the dir."""
        import os
        from pathlib import Path

        exports = Path(os.environ.get("PEAKS_EXPORT_DIR", "/config/exports")).resolve()
        p = (exports / _safe_reel_name(Path(name).stem)).with_suffix(".mp4")
        return str(p) if p.exists() and exports in p.resolve().parents else None

    # --- saved collections (named boards of moments) -------------------------

    def _collections_dir(self):
        import os
        from pathlib import Path

        return Path(os.environ.get("PEAKS_COLLECTIONS_DIR", "/config/collections"))

    def save_collection(self, name: str, apexes: list) -> dict:
        import json

        d = self._collections_dir()
        d.mkdir(parents=True, exist_ok=True)
        safe = _safe_reel_name(name)
        data = {"name": name, "count": len(apexes), "apexes": apexes}
        (d / f"{safe}.json").write_text(json.dumps(data))
        return {"name": name, "safe": safe, "count": len(apexes)}

    def list_collections(self) -> list[dict]:
        import json

        d = self._collections_dir()
        if not d.is_dir():
            return []
        out = []
        for p in sorted(d.glob("*.json")):
            try:
                j = json.loads(p.read_text())
            except Exception:
                continue
            out.append({"name": j.get("name", p.stem), "safe": p.stem,
                        "count": j.get("count", len(j.get("apexes", [])))})
        return out

    def load_collection(self, name: str):
        import json
        from pathlib import Path

        p = self._collections_dir() / f"{_safe_reel_name(Path(name).stem)}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def run_playlist(self, job=None, tags=None, log=None) -> dict:
        """(Re)build the megaboard playlist from Stash markers → the mounted
        webapp dir, so the board updates with one click (or automatically after
        a scoring run)."""
        import os
        from pathlib import Path

        from ..playlist import build_playlist, write_playlist

        log = log or (job.log if job else print)
        tags = tags or [self.cfg.markers.tag_name]
        pl = build_playlist(self.client(), tags, limit=None)
        out = Path(os.environ.get("PEAKS_WEBAPP_DIR", "webapp")) / "playlist.json"
        write_playlist(pl, out)
        log(f"megaboard: {pl['count']} apex(es) for '{pl['tag']}' → {out}")
        return {"tag": pl["tag"], "count": pl["count"], "out": str(out)}

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
            if job and job.cancelled:
                log(f"  ⏹ stop requested — halting after {result['fixed']} fixed")
                break
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

    def search_by_frame(
        self, key: str, time: float, top_k: int = 60, taste: bool = False
    ) -> list[Hit]:
        hits = self.index().search_by_frame(key, time, top_k=top_k)
        if taste:
            hits = self._rerank_by_taste(hits, canonical_name(self.cfg.embedding.model))
        return hits

    def find_duplicates(
        self, key: str, time: float, threshold: float = 0.9, top_k: int = 40,
        model: str | None = None,
    ) -> list[Hit]:
        """Near-identical moments in OTHER scenes (re-encodes, re-uploads).
        Uses DINOv2 by default — structural identity, the right space for visual
        duplicates — and keeps only the strongest match per other scene above
        `threshold`."""
        model = model or canonical_name(self.cfg.embedding.model)
        idx = self.index(model)
        v = idx.vector_at(key, time)
        if v is None:
            return []
        hits = idx.search(v, top_k=top_k, per_scene=1, exclude_key=key)
        return [h for h in hits if h.score >= threshold]

    def scene_timeline(
        self,
        key: str,
        *,
        model: str | None = None,
        text: str | None = None,
        ref_key: str | None = None,
        ref_t: float | None = None,
    ) -> dict:
        """Per-frame relevance across ONE scene — the data behind the heatmap.

        Scores every cached frame of scene `key` against a query: a CLIP text
        prompt (`text`, forces the clip model), or another frame (`ref_key` +
        `ref_t`, the "find similar" source). Returns points sorted by time; the
        UI maps them onto the video's own duration."""
        model = "clip" if text else (model or canonical_name(self.cfg.embedding.model))
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        try:
            times, vecs, meta = cache.load(key, model)
        except Exception:
            return {"points": [], "model": model}
        if vecs.shape[0] == 0:
            return {"points": [], "model": model, "scene_id": meta.get("scene_id")}
        if text:
            q = self._clip_text_vector(text)
        elif ref_key is not None and ref_t is not None:
            q = self.index(model).vector_at(ref_key, ref_t)
        else:
            q = None
        if q is None:
            return {"points": [], "model": model, "scene_id": meta.get("scene_id")}
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        scores = (vecs.astype(np.float32) @ q)
        points = [[round(float(t), 2), round(float(s), 4)] for t, s in zip(times, scores)]
        return {"points": points, "model": model, "scene_id": meta.get("scene_id")}

    def create_apex(
        self, scene_id: str, start: float, end: float | None = None, tag: str | None = None
    ) -> dict:
        """Write a marker at `start` (a moment you saved while watching). Shows
        up in Stash and on the next megaboard build."""
        tag = tag or self.cfg.markers.tag_name
        client = self.client()
        t = client.find_or_create_tag(tag)
        if end is None or end <= start:
            end = start + 15.0
        marker = client.create_scene_marker(
            scene_id=str(scene_id), seconds=float(start), primary_tag_id=t.id,
            title=f"{tag} (saved)", end_seconds=float(end),
        )
        return marker

    def search_text(self, text: str, top_k: int = 60, taste: bool = False) -> list[Hit]:
        """CLIP text -> nearest moments. Supports blended queries: words
        prefixed with '-' are pushed AWAY from ("beach -crowd -text"), so you
        can steer results without re-typing the whole prompt. With `taste`, the
        results are re-ranked by your trained preference model."""
        vec = self._clip_query_vector(text)
        hits = self.index("clip").search(vec, top_k=top_k, per_scene=3)
        return self._rerank_by_taste(hits, "clip") if taste else hits

    # --- taste model (explicit thumbs → personalized ranking) ----------------

    def _label_store(self):
        from ..labels import LabelStore

        return LabelStore(self.cfg.modeling.labels_path)

    def add_label(
        self, key: str, time: float, label: int,
        profile: str | None = None, scene_id: str | None = None,
    ) -> dict:
        profile = profile or self.cfg.markers.tag_name
        store = self._label_store()
        store.add(key, float(time), int(label), profile, scene_id=scene_id)
        store.save()
        pos, neg = store.counts(profile)
        return {"profile": profile, "positive": pos, "negative": neg}

    def label_counts(self, profile: str | None = None) -> dict:
        profile = profile or self.cfg.markers.tag_name
        pos, neg = self._label_store().counts(profile)
        return {"profile": profile, "positive": pos, "negative": neg}

    def _taste_path(self, profile: str, model: str):
        from pathlib import Path

        from ..pipeline import safe_tag

        return Path(self.cfg.modeling.dir) / "taste" / f"{safe_tag(profile)}__{model}.pkl"

    def train_taste(self, profile: str | None = None, model: str | None = None) -> dict:
        """Fit a preference classifier from your thumbs, in one embedding space
        (kept separate from scoring's models so the two never collide)."""
        from ..pipeline import train_profile

        profile = profile or self.cfg.markers.tag_name
        model = model or canonical_name(self.cfg.embedding.model)
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        clf, stats = train_profile(
            self._label_store(), cache, model, profile, kind=self.cfg.modeling.classifier
        )
        out = self._taste_path(profile, model)
        out.parent.mkdir(parents=True, exist_ok=True)
        clf.save(out)
        with self._taste_lock:
            self._taste.pop(str(out), None)
        return {"model": model, "profile": profile, **stats}

    def _taste_model(self, profile: str, model: str):
        from ..classifier import TasteClassifier

        p = self._taste_path(profile, model)
        if not p.exists():
            return None
        with self._taste_lock:
            if str(p) not in self._taste:
                self._taste[str(p)] = TasteClassifier.load(p)
            return self._taste[str(p)]

    def has_taste(self, profile: str | None = None, model: str | None = None) -> bool:
        profile = profile or self.cfg.markers.tag_name
        model = model or canonical_name(self.cfg.embedding.model)
        return self._taste_path(profile, model).exists() or self._taste_path(profile, "clip").exists()

    def _rerank_by_taste(
        self, hits: list[Hit], model: str, profile: str | None = None,
        relevance_weight: float = 0.2,
    ) -> list[Hit]:
        """Re-order the (already query-relevant) results by your taste model,
        with the original relevance kept as a gentle tiebreak. Taste is primary
        because this only runs when you've explicitly asked for "my taste".
        No-op if there's no model for this space."""
        profile = profile or self.cfg.markers.tag_name
        clf = self._taste_model(profile, model)
        if clf is None or not hits:
            return hits
        idx = self.index(model)
        vecs = []
        for h in hits:
            v = idx.vector_at(h.key, h.time)
            vecs.append(v if v is not None else np.zeros(idx.dim or 1, dtype=np.float32))
        taste = np.asarray(clf.predict_proba(np.stack(vecs)), dtype=np.float32)
        ss = np.array([h.score for h in hits], dtype=np.float32)
        span = float(ss.max() - ss.min()) or 1.0
        snorm = (ss - float(ss.min())) / span
        final = (1.0 - relevance_weight) * taste + relevance_weight * snorm
        return [hits[i] for i in np.argsort(-final)]

    def has_clip_index(self) -> bool:
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        return len(cache.keys("clip")) > 0

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def _clip_query_vector(self, text: str, neg_weight: float = 0.5) -> np.ndarray:
        """Build a query vector from a prompt with optional '-negative' terms.
        Positive phrase minus the negatives' direction, renormalized — standard
        CLIP embedding arithmetic."""
        pos, neg = [], []
        for tok in text.split():
            if len(tok) > 1 and tok.startswith("-"):
                neg.append(tok[1:])
            else:
                pos.append(tok[1:] if (len(tok) > 1 and tok.startswith("+")) else tok)
        pos_phrase = " ".join(pos) or text  # all-negative: fall back to literal
        q = self._unit(self._clip_text_vector(pos_phrase))
        if neg:
            q = self._unit(q - neg_weight * self._unit(self._clip_text_vector(" ".join(neg))))
        return q

    # --- "what CLIP sees" — zero-shot moment classification ------------------

    def _vocab(self) -> list[str]:
        """Classification prompts: a user-supplied /config/vocab.txt (one per
        line) if present, else the built-in default list."""
        import os
        from pathlib import Path

        from ..vocab import DEFAULT_VOCAB

        path = Path(os.environ.get("PEAKS_VOCAB", "/config/vocab.txt"))
        if path.is_file():
            lines = [ln.strip() for ln in path.read_text().splitlines()]
            terms = [ln for ln in lines if ln and not ln.startswith("#")]
            if terms:
                return terms
        return DEFAULT_VOCAB

    def _vocab_matrix(self):
        """(labels, matrix) for the vocabulary, CLIP-text-embedded once and
        cached (unit rows, so scoring a frame is one matmul)."""
        with self._clip_lock:
            cached = getattr(self, "_vocab_cache", None)
        if cached is not None:
            return cached
        labels = self._vocab()
        mat = np.stack([self._unit(self._clip_text_vector(t)) for t in labels])
        with self._clip_lock:
            self._vocab_cache = (labels, mat)
        return labels, mat

    def auto_tag(
        self, job=None, top: int = 5, min_score: float = 0.0, limit: int = 0
    ) -> dict:
        """Zero-shot tag the library: for each CLIP-embedded scene, take the
        vocabulary labels that best match any of its moments (max over frames)
        and write them to Stash. Writes are batched one bulk update per tag
        (ADD mode, so existing tags are preserved). Cancellable."""
        from collections import defaultdict

        log = (job.log if job else print)
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        keys = cache.keys("clip")
        if not keys:
            log("no CLIP cache — run a CLIP embed pass first")
            return {"scenes": 0, "tags": 0}
        if limit:
            keys = keys[:limit]
        labels, mat = self._vocab_matrix()
        client = self.client()
        tag_id: dict[str, str] = {}
        assign: dict[str, list[str]] = defaultdict(list)
        scored = 0
        if job:
            job.progress = {"total": len(keys), "done": 0}
        for k in keys:
            if job and job.cancelled:
                log(f"  ⏹ stop requested — halting after {scored} scenes")
                break
            try:
                _, vecs, meta = cache.load(k, "clip")
            except Exception:
                continue
            sid = meta.get("scene_id")
            if not sid or vecs.shape[0] == 0:
                continue
            per_label = (vecs.astype(np.float32) @ mat.T).max(axis=0)  # (V,)
            for i in np.argsort(-per_label)[:top]:
                if float(per_label[i]) < min_score:
                    continue
                lab = labels[i]
                if lab not in tag_id:
                    tag_id[lab] = client.find_or_create_tag(lab).id
                assign[tag_id[lab]].append(str(sid))
            scored += 1
            if job:
                job.progress["done"] = scored
            if scored % 100 == 0:
                log(f"  scored {scored}/{len(keys)} scenes")

        written = 0
        for tid, sids in assign.items():
            if job and job.cancelled:
                break
            client.add_scene_tags(sids, [tid])
            written += 1
        log(f"auto-tag: {scored} scenes → {written} tags applied")
        return {"scenes": scored, "tags": written}

    def classify_frame(self, key: str, time: float, top_k: int = 6) -> dict:
        """Top vocabulary matches for one frame — what CLIP thinks it is."""
        v = self.index("clip").vector_at(key, time)
        if v is None:
            return {"labels": []}
        labels, mat = self._vocab_matrix()
        scores = mat @ self._unit(v)
        order = np.argsort(-scores)[:top_k]
        return {"labels": [[labels[i], round(float(scores[i]), 3)] for i in order]}

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


def _safe_reel_name(name: str) -> str:
    """Filesystem-safe basename (no path separators or surprises)."""
    keep = "".join(c if c.isalnum() or c in "-_." else "-" for c in name)
    return (keep.strip("-.") or "reel")[:120]


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
