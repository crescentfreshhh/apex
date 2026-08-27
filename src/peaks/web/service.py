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
        self._pool = None  # cached scene pool for the shuffle board
        self._settings_cache = None  # GUI-saved active models (lazily read)
        self._taste_src_cache = {}  # model -> (stacked loved-moment vecs, sources)
        self._taste_modes_cache = {}  # model -> K×dim unit "mode" centroids of the loved set
        self._board_score_cache = {}  # model -> (per-moment taste score array, scored_by)
        self._board_universe_cache = {}  # (model, per_scene) -> ordered per-scene-capped Hits
        self._labels_since_train = 0  # new ratings since the last (auto)train

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
        model = self._model_name()
        cached = len(cache.keys(model))
        return {
            "library_path": self.cfg.library.path or "(whole library)",
            "model": model,
            "cached_scenes": cached,
            "dino_model": self._active_dino_model(),
            "clip_model": self._active_clip_model(),
            "clip_cached": len(cache.keys(self._clip_name())),
            "device": self.cfg.embedding.device or "auto",
            "interval": self.cfg.sampling.interval_seconds,
            "mode": self.cfg.sampling.mode,
            "failures": len(failure_log_for(self.cfg)),
        }

    # --- active backbone/variant (GUI-saved, overriding config) -------------

    def _settings_path(self):
        import os
        from pathlib import Path

        return Path(os.environ.get("PEAKS_SETTINGS", "/config/settings.json"))

    def _settings(self) -> dict:
        """GUI-saved settings (currently the active DINOv2 backbone + CLIP
        variant). Cached; save_models() clears the cache."""
        if self._settings_cache is None:
            import json

            path = self._settings_path()
            try:
                self._settings_cache = json.loads(path.read_text()) if path.is_file() else {}
            except (OSError, ValueError):
                self._settings_cache = {}
        return self._settings_cache

    def _active_dino_model(self) -> str:
        return self._settings().get("dino_model") or self.cfg.embedding.dino_model

    def _active_clip_model(self) -> str:
        return self._settings().get("clip_model") or self.cfg.embedding.clip_model

    def _active_clip_pretrained(self) -> str:
        """Checkpoint paired to the active CLIP variant: the saved override maps
        to that variant's default weights; otherwise the configured pair."""
        from ..embedding import default_pretrained

        saved = self._settings().get("clip_model")
        if saved and saved != self.cfg.embedding.clip_model:
            return default_pretrained(saved)
        return self.cfg.embedding.clip_pretrained

    def get_models(self) -> dict:
        """Active models + the option lists, so the GUI can render the pickers
        and show whether the choice is a saved override or the container default."""
        s = self._settings()
        return {
            "dino_model": self._active_dino_model(),
            "clip_model": self._active_clip_model(),
            # curated to the top-tier backbones/variants the GUI offers; the
            # smaller models still validate (for env overrides) but aren't shown.
            "dino_options": ["dinov2_vitl14", "dinov2_vitg14"],
            "clip_options": ["ViT-H-14"],
            "dino_saved": bool(s.get("dino_model")),
            "clip_saved": bool(s.get("clip_model")),
            "dino_default": self.cfg.embedding.dino_model,
            "clip_default": self.cfg.embedding.clip_model,
        }

    def save_models(self, dino_model: str | None = None, clip_model: str | None = None) -> dict:
        """Persist the active DINOv2 backbone and/or CLIP variant to
        /config/settings.json so the *whole* pipeline (embed, score, search,
        megaboard, 'CLIP sees') uses them — no container restart, no env vars.
        Validates against the known model lists; a blank value clears the
        override back to the container default."""
        import json

        from ..embedding import DINO_BACKBONES

        clip_opts = {"ViT-B-32", "ViT-L-14", "ViT-H-14"}
        s = dict(self._settings())
        if dino_model is not None:
            if dino_model and dino_model not in DINO_BACKBONES:
                raise ValueError(f"unknown DINOv2 backbone: {dino_model}")
            if dino_model:
                s["dino_model"] = dino_model
            else:
                s.pop("dino_model", None)
        if clip_model is not None:
            if clip_model and clip_model not in clip_opts:
                raise ValueError(f"unknown CLIP variant: {clip_model}")
            if clip_model:
                s["clip_model"] = clip_model
            else:
                s.pop("clip_model", None)
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s, indent=2) + "\n")
        self._settings_cache = s
        # a changed backbone means a different cache namespace / vector space,
        # so drop any built search indexes to force a clean rebuild.
        self._index = {}
        return self.get_models()

    def schedule_settings(self) -> dict:
        """Current recurring-embed settings (settings.json overriding config).
        The web scheduler reads this live, so a UI change applies without a
        restart. `embed_hours` 0 = off."""
        s = self._settings()
        sc = self.cfg.schedule
        hours = float(s.get("embed_hours", sc.embed_hours) or 0.0)
        return {
            "embed_hours": hours,
            "embed_seconds": hours * 3600.0,
            "sync": bool(s.get("sync", sc.sync)),
            "prune": bool(s.get("prune", sc.prune)),
        }

    def save_schedule(self, embed_hours=None, sync=None, prune=None) -> dict:
        """Persist recurring-embed settings to /config/settings.json."""
        import json

        s = dict(self._settings())
        if embed_hours is not None:
            s["embed_hours"] = max(0.0, float(embed_hours))
        if sync is not None:
            s["sync"] = bool(sync)
        if prune is not None:
            s["prune"] = bool(prune)
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s, indent=2) + "\n")
        self._settings_cache = s
        return self.schedule_settings()

    def embed_status(self) -> dict:
        """How much of the Stash library is embedded — {embedded, total, pending}
        — the 'N new scenes not yet embedded' number. `total`/`pending` are None
        if Stash is unreachable."""
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        embedded = len(cache.keys(self._model_name()))
        try:
            total = self.client().scene_count()
        except Exception:  # noqa: BLE001 — Stash down
            total = None
        pending = max(0, total - embedded) if total is not None else None
        return {"embedded": embedded, "total": total, "pending": pending}

    # --- taste profiles (each = its own labels + Stash tag + model + feed) -----

    def list_profiles(self) -> list[str]:
        """All taste profiles — the default tag, any registered in settings.json,
        and any that already have labels — default first."""
        default = self.cfg.markers.tag_name
        reg = self._settings().get("profiles") or []
        profs = {default} | set(reg) | set(self._label_store().profiles())
        return sorted(profs, key=lambda p: (p != default, p.lower()))

    def _write_profile_registry(self, names: list[str]) -> None:
        import json

        s = dict(self._settings())
        s["profiles"] = names
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s, indent=2) + "\n")
        self._settings_cache = s

    def create_profile(self, name: str) -> list[str]:
        name = (name or "").strip()
        if not name:
            raise ValueError("empty profile name")
        reg = list(self._settings().get("profiles") or [])
        if name not in reg:
            reg.append(name)
            self._write_profile_registry(reg)
        return self.list_profiles()

    def delete_profile(self, name: str, purge_apexes: bool = False) -> dict:
        """Delete a profile: drop it from the registry and erase its taste
        (labels + trained model; its Stash markers too when `purge_apexes`)."""
        if name == self.cfg.markers.tag_name:
            raise ValueError("can't delete the default profile")
        reg = [p for p in (self._settings().get("profiles") or []) if p != name]
        self._write_profile_registry(reg)
        self.delete_taste(profile=name, purge_apexes=purge_apexes)
        self._invalidate_taste_caches()
        return {"profiles": self.list_profiles()}

    def _clip_name(self) -> str:
        """Cache/index name for the active CLIP variant (namespaced so a bigger
        model doesn't collide with an existing ViT-B-32 'clip' cache)."""
        from ..embedding import clip_cache_name

        return clip_cache_name(self._active_clip_model())

    def _model_name(self, alias: str | None = None) -> str:
        """Backbone-aware cache name for a channel, resolved against the *active*
        (GUI-saved) DINOv2 backbone / CLIP variant."""
        from ..embedding import canonical_name, clip_cache_name, dino_cache_name

        key = (alias or self.cfg.embedding.model).lower()
        if key in ("dino", "dinov2"):
            return dino_cache_name(self._active_dino_model())
        if key == "clip":
            return clip_cache_name(self._active_clip_model())
        return canonical_name(alias or self.cfg.embedding.model)

    def _embedder(self, model: str | None = None):
        """Build the embedder for `model`, using the *active* backbone/variant
        (a GUI-saved choice in /config/settings.json overriding config). The
        embedder namespaces its own cache from the variant, so the whole
        pipeline stays self-consistent with the active choice."""
        from ..embedding import get_embedder

        name = model or self.cfg.embedding.model
        kwargs = {"device": self.cfg.embedding.device} if self.cfg.embedding.device else {}
        canon = canonical_name(name)
        if canon == "clip":
            kwargs["model_name"] = self._active_clip_model()
            kwargs["pretrained"] = self._active_clip_pretrained()
        elif canon == "dinov2":
            kwargs["model_name"] = self._active_dino_model()
        return get_embedder(name, **kwargs)

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
        model = self._model_name()

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

    def export_collection(
        self, job=None, name: str = "", limit: int = 200
    ) -> dict:
        """Concatenate a saved collection's clips into one downloadable video —
        the collection-flavoured sibling of export_reel. Highest-scoring moments
        first, capped at `limit` so a giant collection can't kick off a
        multi-hour render unasked. Stream-copy (fast), skips codec mismatches."""
        import os
        import subprocess
        import tempfile
        from pathlib import Path

        log = (job.log if job else print)
        coll = self.load_collection(name)
        if not coll or not coll.get("apexes"):
            log(f"no collection '{name}' to export")
            return {"clips": 0}
        apexes = sorted(coll["apexes"], key=lambda a: a.get("score", 0), reverse=True)
        if limit:
            apexes = apexes[:limit]
        apexes = [a for a in apexes if a.get("scene_id")]

        details = self.client().scene_details(sorted({str(a["scene_id"]) for a in apexes}))
        exports = Path(os.environ.get("PEAKS_EXPORT_DIR", "/config/exports"))
        exports.mkdir(parents=True, exist_ok=True)
        safe = _safe_reel_name(Path(name).stem) + ".mp4"
        out = exports / safe
        if job:
            job.progress = {"total": len(apexes), "done": 0}

        ff = getattr(self.cfg.sampling, "ffmpeg", "ffmpeg") if hasattr(self.cfg, "sampling") else "ffmpeg"
        with tempfile.TemporaryDirectory() as td:
            segs: list[str] = []
            for i, a in enumerate(apexes):
                if job and job.cancelled:
                    log(f"  ⏹ stop requested — halting after {len(segs)} clips")
                    break
                path = (details.get(str(a["scene_id"])) or {}).get("path")
                if not path or not os.path.exists(path):
                    log(f"  ! scene {a['scene_id']}: file missing — skipped")
                    continue
                start = float(a.get("start") or 0)
                end = float(a["end"]) if a.get("end") else start + float(a.get("duration") or 20)
                seg = os.path.join(td, f"seg{i:04d}.ts")
                r = subprocess.run(
                    [ff, "-y", "-ss", f"{start:g}", "-to", f"{end:g}", "-i", path,
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
                [ff, "-y", "-f", "concat", "-safe", "0", "-i", listf, "-c", "copy", str(out)],
                capture_output=True,
            )
            if cc.returncode != 0:
                raise RuntimeError("concat failed: " + cc.stderr.decode("replace")[-300:])
        size = out.stat().st_size if out.exists() else 0
        log(f"export: {len(segs)} clips → {out} ({size // 1_000_000} MB)")
        return {"clips": len(segs), "name": safe, "path": str(out), "bytes": size}

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

    def save_collection(
        self, name: str, apexes: list, query: str | None = None,
        params: dict | None = None, live: bool = False, source: dict | None = None,
    ) -> dict:
        import json

        d = self._collections_dir()
        d.mkdir(parents=True, exist_ok=True)
        safe = _safe_reel_name(name)
        data = {"name": name, "count": len(apexes), "apexes": apexes}
        if query:
            data["query"] = query          # remembered for a future "refresh"
        if params:
            data["params"] = params
        if live and source:                # a live playlist re-derives from its source
            data["live"] = True
            data["source"] = source
        (d / f"{safe}.json").write_text(json.dumps(data))
        return {"name": name, "safe": safe, "count": len(apexes), "live": bool(live and source)}

    def derive_collection(self, name: str):
        """Re-run a live playlist's saved source spec → fresh Hits (newest taste /
        library). None when the playlist isn't live or doesn't exist."""
        c = self.load_collection(name)
        if not c or not c.get("live"):
            return None
        s = c.get("source") or {}
        kind = s.get("kind")
        try:
            if kind == "performer":
                q = s.get("query") or None
                r = self.performer_best(
                    performer_id=s.get("id"), name=s.get("name"), scene_id=s.get("scene_id"),
                    count=1500, per_scene=40, query=q, spread=(q is None),
                )
                return r["hits"]
            if kind == "search":
                return self.search_text(
                    s.get("q", ""), top_k=int(s.get("top_k") or 300),
                    taste=bool(s.get("taste")), per_scene=int(s.get("per") or 3),
                    min_score=(s.get("min") or None), neg_weight=float(s.get("neg") or 0.5),
                )
            if kind == "foryou":
                return self.board_pool(count=int(s.get("count") or 1500))["hits"]
            if kind == "stat":
                return self.stats_board(s.get("metric", "fresh"), id=s.get("id"))["hits"]
            if kind == "scene":
                return self.scene_moments(str(s.get("scene_id")))["hits"]
            if kind == "performer_moment":
                return self.performer_moment_matches(str(s.get("scene_id")), float(s.get("t") or 0.0))["hits"]
            if kind == "similar":
                key = self._key_for_scene(str(s.get("scene_id")), self._model_name())
                return self.search_by_frame(key, float(s.get("t") or 0.0), top_k=300, per_scene=3) if key else []
        except Exception:  # noqa: BLE001 — a bad spec shouldn't 500 the board
            return []
        return []

    def list_collections(self) -> list[dict]:
        import json

        d = self._collections_dir()
        if not d.is_dir():
            return []
        out = []
        model = self._model_name()
        for p in sorted(d.glob("*.json")):
            try:
                j = json.loads(p.read_text())
            except Exception:
                continue
            apexes = j.get("apexes") or []
            thumb = None
            for a in apexes[:4]:   # first embedded moment → a cover frame
                sid = a.get("scene_id")
                if sid is None:
                    continue
                try:
                    key = self._key_for_scene(str(sid), model)
                except Exception:  # noqa: BLE001
                    key = None
                if key:
                    t = a.get("start", a.get("time", 0)) or 0
                    thumb = f"/api/frame?key={key}&t={float(t):g}"
                    break
            out.append({"name": j.get("name", p.stem), "safe": p.stem,
                        "count": j.get("count", len(apexes)), "thumb": thumb,
                        "live": bool(j.get("live"))})
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

    def _collection_path(self, name: str):
        from pathlib import Path

        return self._collections_dir() / f"{_safe_reel_name(Path(name).stem)}.json"

    def delete_collection(self, name: str) -> dict:
        """Remove a saved collection's file (path-safe, no traversal)."""
        p = self._collection_path(name)
        removed = False
        try:
            if p.exists():
                p.unlink()
                removed = True
        except OSError:
            pass
        return {"removed": removed, "safe": p.stem}

    def rename_collection(self, name: str, new_name: str) -> dict:
        """Change a collection's DISPLAY name in place — the file/`safe` stem
        (which the board options and ?collection= URLs key off) stays put, so
        nothing that references it breaks."""
        import json

        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("new name is empty")
        p = self._collection_path(name)
        if not p.exists():
            raise FileNotFoundError(name)
        data = json.loads(p.read_text())
        data["name"] = new_name
        p.write_text(json.dumps(data))
        return {"name": new_name, "safe": p.stem, "count": data.get("count", 0)}

    # --- megaboard sources (shuffle-all / apex tag / collection) -------------

    def scene_pool(self, refresh: bool = False) -> list[dict]:
        """Every scene (within library scope) as {scene_id, duration, url} — the
        pool the shuffle board draws random moments from. Cached, since it's a
        full Stash pass; call with refresh=True to rebuild."""
        if self._pool is None or refresh:
            client = self.client()
            pool = []
            for s in client.iter_scenes(path_prefix=self.cfg.library.path):
                if not s.path:
                    continue
                pool.append({
                    "scene_id": s.id,
                    "duration": s.duration or 0,
                    "url": client.stream_url(s.id, start=0),
                })
            self._pool = pool
        return self._pool

    def board_apexes(self, tag: str | None = None) -> dict:
        """Live apex playlist for a tag (no on-disk write) so the board's source
        picker can load any tag on demand."""
        from ..playlist import build_playlist

        return build_playlist(self.client(), [tag or self.cfg.markers.tag_name], limit=None)

    def board_sources(self) -> dict:
        return {"tag": self.cfg.markers.tag_name, "collections": self.list_collections()}

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

    def run_embed_multi(self, job=None, models=None, limit: int = 0, **overrides) -> dict:
        """Run several embed passes back-to-back in one job (e.g. DINOv2 then
        CLIP), so you can queue both and walk away. Stops between passes if
        cancelled; totals are aggregated and each pass is reported."""
        log = (job.log if job else print)
        models = list(models) if models else [self.cfg.embedding.model]
        total = {"embedded": 0, "skipped": 0, "failed": 0, "frames": 0}
        passes: dict[str, dict] = {}
        for i, m in enumerate(models):
            if job and job.cancelled:
                log("  ⏹ stop requested — halting the queue")
                break
            log(f"=== embed pass {i + 1}/{len(models)}: model={m} ===")
            st = self.run_embed(job, limit=limit, model=m, **overrides)
            passes[m] = st
            for k in total:
                total[k] += st.get(k, 0)
        total["passes"] = passes
        return total

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
        models = cache.models() if all_models else [self._model_name()]
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
        result = {"fixed": 0, "failed": 0, "pruned": 0, "total": len(entries)}
        if not entries:
            log("no recorded failures — nothing to fix")
            return result
        if job:
            job.progress = {"total": len(entries), "done": 0}

        # Scenes deleted from the library will fail forever (there's nothing left
        # to decode), so the counter never drops. Ask Stash which of these scene
        # ids still exist; the rest are stale and get pruned from the log instead
        # of retried. If Stash is unreachable we skip pruning (retry everything)
        # rather than risk dropping a scene that's only transiently missing.
        alive: set[str] | None = None
        want = [e.get("scene_id") for e in entries if e.get("scene_id")]
        if want:
            try:
                alive = self.client().existing_scene_ids(want)
            except Exception as exc:  # noqa: BLE001 — Stash down: don't prune
                log(f"  (couldn't check Stash for deleted scenes: {exc} — retrying all)")
                alive = None

        # (mode, hwaccel, pipeline): distinct decode strategies, cheap → tolerant
        ladder = [
            ("sparse", "", "raw"),      # sparse seek, no NVDEC
            ("interval", "", "jpeg"),   # full linear decode, most forgiving
        ]
        def _is_stale(e) -> bool:
            # deleted from Stash → nothing to retry, prune it
            sid = e.get("scene_id")
            return alive is not None and sid is not None and str(sid) not in alive

        if dry_run:
            for e in entries:
                if _is_stale(e):
                    log(f"  · would prune scene {e.get('scene_id')} — gone from Stash "
                        f"({e.get('path')})")
                else:
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
            if _is_stale(e):
                flog.resolve(key)
                log(f"  🗑 scene {e.get('scene_id')}: deleted from Stash — pruned from the log")
                result["pruned"] += 1
                if job:
                    job.progress["done"] = result["fixed"] + result["failed"] + result["pruned"]
                continue
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
                job.progress["done"] = result["fixed"] + result["failed"] + result["pruned"]
        log(f"done — fixed {result['fixed']}, pruned {result['pruned']} deleted, "
            f"{result['failed']} still failing of {result['total']}")
        return result

    # --- search index --------------------------------------------------------

    def index(self, model: str | None = None, refresh: bool = False) -> SearchIndex:
        """The in-memory search index for `model`, built lazily and cached.

        A cached index is normally reused until an embed pass finishes (which
        calls invalidate_index). Two exceptions keep live views honest during an
        in-progress embed: an *empty* cached index is always rebuilt (so the
        first embedded frames show up), and with `refresh=True` the index is
        rebuilt whenever the cache has grown since it was built (so the swipe
        trainer's candidate pool keeps expanding as scenes embed)."""
        model = model or self._model_name()
        with self._index_lock:
            idx = self._index.get(model)
            rebuild = idx is None or idx.size == 0
            if refresh and not rebuild:
                cache = EmbeddingCache(self.cfg.embedding.cache_dir)
                if len(cache.keys(model)) != getattr(idx, "source_key_count", -1):
                    rebuild = True
            if rebuild:
                # drop the old index first so its matrix is freed before the new
                # one is allocated — a whole-library float32 matrix is GBs, and
                # holding both at once is what spiked RSS during an embed.
                self._index.pop(model, None)
                idx = None
                cache = EmbeddingCache(self.cfg.embedding.cache_dir)
                keys = cache.keys(model)
                idx = SearchIndex(cache, model).build(keys)
                idx.source_key_count = len(keys)
                self._index[model] = idx
            return idx

    def invalidate_index(self, model: str | None = None) -> None:
        with self._index_lock:
            if model is None:
                self._index.clear()
            else:
                self._index.pop(model, None)
        from . import memwatch  # freeing a whole-library matrix → return it to the OS

        memwatch.malloc_trim()

    # --- memory self-policing ------------------------------------------------

    def memory_status(self) -> dict:
        """RSS vs the watchdog's soft limit, for the /api/memory readout."""
        from . import memwatch

        rss = memwatch.rss_bytes()
        limit = memwatch.soft_limit_bytes()
        with self._index_lock:
            resident = sorted(self._index.keys())
        return {
            "rss_mb": round(rss / 1048576, 1),
            "limit_mb": round(limit / 1048576, 1) if limit else None,
            "pct": round(100 * rss / limit, 1) if limit else None,
            "indexes": resident,
        }

    def shed_memory(self, drop_indexes: bool = True) -> dict:
        """Release derived caches (and, if asked, idle model indexes) and hand the
        freed pages back to the OS. The watchdog calls this under memory pressure;
        everything shed here rebuilds lazily on next use. Returns a small report."""
        import gc

        from . import memwatch

        before = memwatch.rss_bytes()
        self._invalidate_taste_caches()          # taste + board score/universe caches
        with self._meta_lock:
            self._meta = {}
        self._pool = None
        self._vocab_cache = None
        self._perf_stats_cache = None
        self._perf_centroids = {}
        dropped: list[str] = []
        if drop_indexes:
            keep = self._model_name()            # the model the board/For You needs
            with self._index_lock:
                for m in [m for m in self._index if m != keep]:
                    self._index.pop(m, None)
                    dropped.append(m)
        gc.collect()
        memwatch.malloc_trim()
        after = memwatch.rss_bytes()
        return {
            "freed_mb": round(max(0, before - after) / 1048576, 1),
            "rss_mb": round(after / 1048576, 1),
            "dropped_indexes": dropped,
        }

    def search_by_frame(
        self, key: str, time: float, top_k: int | None = 60, taste: bool = False,
        per_scene: int | None = 3, min_score: float | None = None,
    ) -> list[Hit]:
        hits = self.index().search_by_frame(
            key, time, top_k=top_k, per_scene=per_scene, min_score=min_score
        )
        if taste:
            hits = self._rerank_by_taste(hits, self._model_name())
        return hits

    def find_duplicates(
        self, key: str, time: float, threshold: float = 0.9, top_k: int = 40,
        model: str | None = None,
    ) -> list[Hit]:
        """Near-identical moments in OTHER scenes (re-encodes, re-uploads).
        Uses DINOv2 by default — structural identity, the right space for visual
        duplicates — and keeps only the strongest match per other scene above
        `threshold`."""
        model = model or self._model_name()
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
        model = self._clip_name() if text else (model or self._model_name())
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
        up in Stash and on the next megaboard build. A manual save is also a
        deliberate "I love this", so it's folded into your taste model as a
        positive (feeds the reranker, not just the retrieval centroid)."""
        tag = tag or self.cfg.markers.tag_name
        client = self.client()
        t = client.find_or_create_tag(tag)
        if end is None or end <= start:
            end = start + 15.0
        marker = client.create_scene_marker(
            scene_id=str(scene_id), seconds=float(start), primary_tag_id=t.id,
            title=f"{tag} (saved)", end_seconds=float(end),
        )
        self._label_apex_as_taste(scene_id, float(start), tag)
        return marker

    def _label_apex_as_taste(self, scene_id: str, time: float, profile: str) -> None:
        """Record a saved apex as a positive taste label (best-effort). Only
        works once the scene is embedded (we need its cache key); never blocks
        the save if it can't."""
        try:
            key = self._key_for_scene(str(scene_id), self._model_name())
            if key:
                self.add_label(key, time, 1, profile=profile, scene_id=str(scene_id))
        except Exception:  # noqa: BLE001 — taste labeling must never break a save
            pass

    def find_apex(
        self, scene_id: str, time: float, tag: str | None = None, tol: float = 2.0
    ) -> dict | None:
        """The apex-tagged marker on `scene_id` nearest `time` (within `tol`
        seconds), or None. Lets the board offer 'remove as apex' only for
        moments that really are apexes."""
        tag = tag or self.cfg.markers.tag_name
        try:
            markers = self.client().markers_for_scene(str(scene_id))
        except Exception:  # noqa: BLE001 — Stash down / unknown scene
            return None
        best, best_d = None, float(tol)
        for m in markers:
            if m.get("primary_tag") != tag:
                continue
            d = abs(float(m["seconds"]) - float(time))
            if d <= best_d:
                best, best_d = m, d
        return best

    def remove_apex(
        self, scene_id: str, time: float, tag: str | None = None, tol: float = 2.0
    ) -> dict:
        """Delete the apex marker on `scene_id` nearest `time` — the inverse of
        create_apex. Returns {removed, marker_id}. The taste 👍 that the original
        save recorded is left in place (undo it with 👎 if you want)."""
        m = self.find_apex(scene_id, time, tag=tag, tol=tol)
        if not m:
            return {"removed": 0, "marker_id": None}
        self.client().destroy_scene_markers([m["marker_id"]])
        return {"removed": 1, "marker_id": m["marker_id"]}

    def search_text(
        self, text: str, top_k: int | None = 60, taste: bool = False,
        per_scene: int | None = 3, min_score: float | None = None,
        neg_weight: float = 0.5,
    ) -> list[Hit]:
        """CLIP text -> nearest moments. Supports blended queries: words
        prefixed with '-' are pushed AWAY from ("beach -crowd -text"), so you
        can steer results without re-typing the whole prompt (`neg_weight` sets
        how hard). `min_score` returns *all* moments at least that close (the
        Explore match-strength floor); `top_k=None` is unbounded. With `taste`,
        the results are re-ranked by your trained preference model."""
        vec = self._clip_query_vector(text, neg_weight=neg_weight)
        clip = self._clip_name()
        hits = self.index(clip).search(
            vec, top_k=top_k, per_scene=per_scene, min_score=min_score
        )
        return self._rerank_by_taste(hits, clip) if taste else hits

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
        self._labels_since_train += 1
        pos, neg = store.counts(profile)
        return {"profile": profile, "positive": pos, "negative": neg}

    def delete_taste(
        self,
        profile: str | None = None,
        within_minutes: float | None = None,
        purge_apexes: bool = False,
    ) -> dict:
        """Erase learned taste: all 👍/👎 ratings for a profile, or only those
        from the last `within_minutes` (an undo). Retrains from what remains if
        both classes survive, else drops the trained model. Saved apex markers in
        Stash are NOT touched — they're separate curated favourites — unless
        `purge_apexes` is set on a full wipe, which also deletes every apex marker
        for the profile (a true scorched-earth reset)."""
        profile = profile or self.cfg.markers.tag_name
        store = self._label_store()
        if within_minutes and within_minutes > 0:
            from time import time as _now

            removed = store.remove(profile, newer_than=_now() - within_minutes * 60.0)
        else:
            removed = store.remove(profile)  # everything for this profile
        store.save()
        self._invalidate_taste_caches()
        self.reset_labels_since_train()
        pos, neg = store.counts(profile)

        retrained = model_deleted = False
        if within_minutes and pos >= 1 and neg >= 1:
            # enough remains → refit so the model reflects the reduced history
            try:
                self.train_taste(profile=profile)
                retrained = True
            except Exception:  # noqa: BLE001 — fall through to dropping the model
                model_deleted = self._delete_taste_models(profile)
        else:
            # full wipe, or not enough labels left to train → remove the model(s)
            model_deleted = self._delete_taste_models(profile)

        out = {
            "profile": profile, "removed": removed, "positive": pos, "negative": neg,
            "retrained": retrained, "model_deleted": model_deleted,
        }
        # scorched-earth: on a full wipe, also delete the apex markers in Stash so
        # the taste centroid has nothing left to rebuild from (For You goes empty).
        full_wipe = not (within_minutes and within_minutes > 0)
        if purge_apexes and full_wipe:
            try:
                # apex markers live under the profile's marker tag — the same
                # enumeration the taste centroid reads (the default profile's tag
                # is the configured marker tag; others tag by profile name)
                ids = [
                    mk["marker_id"]
                    for mk in self.client().iter_markers_by_tag(profile)
                    if mk.get("marker_id")
                ]
                self.client().destroy_scene_markers(ids)
                out["apexes_removed"] = len(ids)
                self._invalidate_taste_caches()  # centroid now rebuilds empty
            except Exception as e:  # noqa: BLE001 — labels/model are already gone
                out["apexes_removed"] = 0
                out["apex_error"] = str(e)
        return out

    def _delete_taste_models(self, profile: str) -> bool:
        """Delete every trained taste model for `profile` (across embedding
        spaces) and drop any in-memory copies. Returns True if anything went."""
        from pathlib import Path

        from ..pipeline import safe_tag

        prefix = safe_tag(profile) + "__"
        taste_dir = Path(self.cfg.modeling.dir) / "taste"
        removed = False
        if taste_dir.is_dir():
            for p in taste_dir.glob(prefix + "*.pkl"):
                try:
                    p.unlink()
                    removed = True
                except OSError:
                    pass
        with self._taste_lock:
            for k in [k for k in self._taste if Path(k).name.startswith(prefix)]:
                self._taste.pop(k, None)
        if removed:
            self._invalidate_taste_caches()  # board falls back to centroid scoring
        return removed

    def autotrain_due(self, profile: str | None = None) -> bool:
        """True when enough new ratings have piled up to retrain in the
        background — and there's at least one 👍 and one 👎 (a classifier needs
        both classes). 0 in config disables auto-training."""
        n = self.cfg.modeling.autotrain_every
        if n <= 0 or self._labels_since_train < n:
            return False
        pos, neg = self._label_store().counts(profile or self.cfg.markers.tag_name)
        return pos >= 1 and neg >= 1

    def reset_labels_since_train(self) -> None:
        self._labels_since_train = 0

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
        model = model or self._model_name()
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        clf, stats = train_profile(
            self._label_store(), cache, model, profile,
            kind=self.cfg.modeling.taste_classifier,
            recency_halflife_days=self.cfg.modeling.recency_halflife_days,
        )
        out = self._taste_path(profile, model)
        out.parent.mkdir(parents=True, exist_ok=True)
        clf.save(out)
        with self._taste_lock:
            self._taste.pop(str(out), None)
        self._invalidate_taste_caches()  # board must re-score with the new model
        return {"model": model, "profile": profile, **stats}

    def _taste_model(self, profile: str, model: str):
        from ..classifier import TasteClassifier

        p = self._taste_path(profile, model)
        if not p.exists():
            return None
        with self._taste_lock:
            if str(p) not in self._taste:
                try:
                    self._taste[str(p)] = TasteClassifier.load(p)
                except Exception:  # noqa: BLE001 — tolerate a mid-write retrain
                    return None
            return self._taste.get(str(p))

    def has_taste(self, profile: str | None = None, model: str | None = None) -> bool:
        profile = profile or self.cfg.markers.tag_name
        model = model or self._model_name()
        return self._taste_path(profile, model).exists() or self._taste_path(profile, self._clip_name()).exists()

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

    # --- "For You": taste centroid, recommendations, active learning ---------

    def _taste_sources(self, model: str, rebuild: bool = False, profile: str | None = None):
        """Unit vectors of your loved moments in `model` space — apex markers
        plus thumbs-up labels — stacked oldest→newest, with a parallel list of
        their {key,time,scene_id,kind}. Scoped to a taste `profile` (its Stash
        marker tag + its label set); cached per (model, profile); `rebuild`
        re-reads the Stash markers and the label store."""
        profile = profile or self.cfg.markers.tag_name
        ckey = (model, profile)
        if not rebuild and ckey in self._taste_src_cache:
            return self._taste_src_cache[ckey]
        idx = self.index(model)
        sid_key: dict[str, str] = {}
        for k, m in idx.key_meta.items():
            sid = m.get("scene_id")
            if sid is not None:
                sid_key.setdefault(str(sid), k)
        vecs, sources, seen = [], [], set()

        def _add(key, t, scene_id, kind):
            sig = (key, round(float(t), 1))
            if key is None or sig in seen:
                return
            v = idx.vector_at(key, float(t))
            if v is None:
                return
            vecs.append(self._unit(v))
            sources.append(
                {"key": key, "time": round(float(t), 2), "scene_id": scene_id, "kind": kind}
            )
            seen.add(sig)

        # apex markers — the moments you explicitly saved
        try:
            for mk in self.client().iter_markers_by_tag(profile):
                sid = mk.get("scene_id")
                if sid:
                    _add(sid_key.get(str(sid)), mk.get("seconds") or 0.0, str(sid), "apex")
        except Exception:  # noqa: BLE001 — Stash down: fall back to labels only
            pass
        # thumbs-up labels — from the swipe trainer / viewer
        try:
            for lab in self._label_store().for_profile(profile):
                if lab.label == 1:
                    _add(lab.key, lab.time, lab.scene_id, "thumb")
        except Exception:  # noqa: BLE001
            pass

        dim = idx.dim or 1
        arr = np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, dim), dtype=np.float32)
        self._taste_src_cache[ckey] = (arr, sources)
        return arr, sources

    def _taste_centroid(self, model: str, recent: int = 0, rebuild: bool = False, profile: str | None = None):
        """(unit centroid, count, sources). `recent`>0 averages only your latest
        N loved moments — the 'what you're into lately' view. `profile` scopes it
        to one taste profile (default = the configured tag)."""
        arr, sources = self._taste_sources(model, rebuild=rebuild, profile=profile)
        if arr.shape[0] == 0:
            return None, 0, sources
        a = arr[-recent:] if (recent and recent < arr.shape[0]) else arr
        return self._unit(a.mean(axis=0)), int(a.shape[0]), sources

    def _taste_modes(self, model: str, profile: str | None = None):
        """Your loved set as K unit "mode" centroids, not one averaged point — so
        scoring can reward a moment near *any* of your distinct interests instead
        of only the bland midpoint between them. With few loved moments each is its
        own mode (≈ nearest-neighbour); above the cap they're k-means clusters.
        Returns a K×dim unit matrix, or None when there's no taste yet."""
        profile = profile or self.cfg.markers.tag_name
        ckey = (model, profile)
        cached = self._taste_modes_cache.get(ckey)
        if cached is not None:
            return cached
        arr, _ = self._taste_sources(model, profile=profile)
        L = int(arr.shape[0])
        if L == 0:
            return None
        k = max(1, min(L, int(self.cfg.modeling.taste_modes)))
        if L <= k:
            modes = arr  # each loved moment is its own mode
        else:
            try:
                from sklearn.cluster import KMeans

                km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(arr)
                modes = km.cluster_centers_.astype(np.float32)
            except Exception:  # noqa: BLE001 — sklearn hiccup → fall back to the mean
                modes = self._unit(arr.mean(axis=0))[None, :]
        # unit-normalise rows so a dot with unit frame vectors is cosine similarity
        norms = np.linalg.norm(modes, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        modes = (modes / norms).astype(np.float32)
        self._taste_modes_cache[ckey] = modes
        return modes

    def _hidden_path(self):
        from pathlib import Path

        return Path(self.cfg.modeling.dir) / "hidden_scenes.json"

    def hidden_scene_ids(self) -> set[str]:
        """Scenes you 1★'d ('mark for deletion') — hidden from every peaks feed,
        board, and pivot. A local set (no per-request Stash call), populated by
        peaks' own mark-for-deletion; loaded once and cached in memory."""
        cached = getattr(self, "_hidden_set", None)
        if cached is not None:
            return cached
        import json

        ids: set[str] = set()
        p = self._hidden_path()
        try:
            if p.exists():
                ids = {str(x) for x in json.loads(p.read_text())}
        except Exception:  # noqa: BLE001 — a bad/missing file just means nothing hidden
            ids = set()
        self._hidden_set = ids
        return ids

    def set_scene_hidden(self, scene_id: str, hidden: bool) -> None:
        """Add/remove a scene from the hidden set (persisted). Driven by the
        rating: 1★ ('mark for deletion') hides it, a higher/cleared rating unhides."""
        import json

        ids = set(self.hidden_scene_ids())
        sid = str(scene_id)
        if hidden:
            ids.add(sid)
        else:
            ids.discard(sid)
        self._hidden_set = ids
        p = self._hidden_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(sorted(ids)))
        except Exception:  # noqa: BLE001 — persistence best-effort
            pass

    def recommend(
        self, top_k: int = 60, model: str | None = None,
        per_scene: int = 2, recent: int = 0, rebuild: bool = False,
        exclude: set | None = None, shuffle: bool | None = None,
        seed: int | None = None, min_score: float | None = None,
        profile: str | None = None,
    ) -> dict:
        """Moments across the whole library ranked for you — retrieve-then-rerank:
        the taste centroid pulls a generous candidate pool (fast, spans the
        library), then your trained taste classifier reranks that pool (so your
        👎 passes and "Train now" shape the feed, not just your 👍). Falls back
        to pure centroid ranking when there's no trained model yet. `exclude` is
        a set of scene_ids to drop (Taste Radio uses it to keep the stream
        endless — never replay what you've already seen). `shuffle` reorders the
        candidate pool with a rank-weighted random shuffle so each rebuild
        surfaces a fresh mix (strong matches stay likely near the top); it
        defaults to `rebuild`, so an explicit "Rebuild from my taste" varies
        while background refreshes stay stable."""
        model = model or self._model_name()
        if shuffle is None:
            shuffle = rebuild
        if rebuild:
            self._invalidate_taste_caches()
        c, n, sources = self._taste_centroid(model, recent=recent, rebuild=rebuild, profile=profile)
        if c is None:
            return {"hits": [], "sources": 0, "total": 0, "model": model, "reranked": False}
        # retrieve a pool larger than we'll show, so the reranker has room to
        # surface moments the centroid alone ranked lower (and exclusion still fills).
        pool = self.index(model).search(c, top_k=max(top_k * 6, 300), per_scene=per_scene)
        if exclude:
            exclude = {str(s) for s in exclude}
            pool = [h for h in pool if str(h.scene_id) not in exclude]
        if min_score is not None:  # taste floor: only moments this close to your taste
            pool = [h for h in pool if h.score >= min_score]
        reranked = self._taste_model(profile or self.cfg.markers.tag_name, model) is not None
        ranked = self._rerank_by_taste(pool, model, profile=profile) if reranked else pool
        if shuffle:
            ranked = self._shuffle_ranked(ranked, top_k, seed)
        diversity = self.cfg.modeling.feed_diversity
        hits = self._diversify(ranked, model, top_k, diversity)
        return {
            "hits": hits, "sources": n, "total": len(sources),
            "model": model, "reranked": reranked, "diversified": diversity > 0,
        }

    def _shuffle_ranked(
        self, ranked: list[Hit], top_k: int, seed: int | None = None
    ) -> list[Hit]:
        """Rank-weighted random reorder of the candidate pool: sample without
        replacement with weights that decay by rank, so highly-ranked moments
        stay probable near the front but every call yields a different mix. The
        temperature is tied to `top_k` so the shuffle spans roughly the moments
        that could actually make the feed, not the whole long tail."""
        n = len(ranked)
        if n <= 1:
            return ranked
        rng = np.random.default_rng(seed)
        temp = float(max(top_k, 20))
        w = np.exp(-np.arange(n) / temp)
        p = w / w.sum()
        order = rng.choice(n, size=n, replace=False, p=p)
        return [ranked[int(i)] for i in order]

    def _invalidate_taste_caches(self) -> None:
        """Drop the cached taste centroid, per-moment scores and board universe so
        the next feed/board reflects new ratings or a freshly trained model."""
        self._taste_src_cache.clear()
        self._taste_modes_cache.clear()
        self._board_score_cache.clear()
        self._board_universe_cache.clear()
        self._peak_index_cache = None   # peaks depend on the taste scorer
        self._scene_seg_cache = None    # so do the per-scene segment spans

    def _taste_scores(self, model: str, profile: str | None = None):
        """One taste score per indexed moment, for the whole library. Uses your
        trained classifier's probability when a model exists (a learned,
        multi-modal boundary that recognises your *diverse* taste), otherwise
        scores each moment by its similarity to your *nearest loved mode* — so a
        moment close to any of your distinct interests scores high, not just those
        near the average of everything. Returns (scores[idx.size], scored_by) or
        (None, None) with no taste yet. Cached per (model, profile)."""
        profile = profile or self.cfg.markers.tag_name
        ckey = (model, profile)
        cached = self._board_score_cache.get(ckey)
        if cached is not None:
            return cached
        idx = self.index(model)
        if idx.size == 0:
            return None, None
        clf = self._taste_model(profile, model)
        if clf is not None:
            scores = np.asarray(clf.predict_proba(idx.matrix), dtype=np.float32).reshape(-1)
            scored_by = "classifier"
        else:
            modes = self._taste_modes(model, profile=profile)
            if modes is None:
                return None, None
            # cosine to every mode, keep each moment's best → spans your whole taste
            scores = (idx.matrix @ modes.T).max(axis=1).astype(np.float32)
            scored_by = "modes"
        out = (scores, scored_by)
        self._board_score_cache[ckey] = out
        return out

    def _board_universe(self, model: str, per_scene: int, profile: str | None = None):
        """Every scene's best taste-moment (plus up to `per_scene-1` more), ordered
        best-first across the whole library — so the board is guaranteed ≥1 moment
        for *every* scene you've indexed, not just the few nearest your taste's
        centre. (scored_by, list[Hit]) cached per (model, per_scene, profile)."""
        profile = profile or self.cfg.markers.tag_name
        key = (model, per_scene, profile)
        cached = self._board_universe_cache.get(key)
        if cached is not None:
            return cached
        scores, scored_by = self._taste_scores(model, profile=profile)
        if scores is None:
            return None, []
        idx = self.index(model)
        # top-`per_scene` moments per scene, then globally best-first — vectorised
        # so a million-frame library builds in one pass, not a Python loop.
        sid_arr = np.asarray([str(s) if s is not None else "" for s in idx.scene_ids])
        _, inv = np.unique(sid_arr, return_inverse=True)
        grouped = np.lexsort((-scores, inv))  # by scene, best-first within scene
        inv_s = inv[grouped]
        pos = np.arange(inv_s.size)
        new_grp = np.empty(inv_s.size, dtype=bool)
        new_grp[0] = True
        new_grp[1:] = inv_s[1:] != inv_s[:-1]
        rank_in_scene = pos - np.maximum.accumulate(np.where(new_grp, pos, 0))
        chosen = grouped[rank_in_scene < per_scene]
        chosen = chosen[np.argsort(-scores[chosen])]  # global best-first
        hits = [
            Hit(scene_id=idx.scene_ids[i], key=idx.keys[i],
                time=float(idx.times[i]), score=float(scores[i]))
            for i in chosen.tolist()
        ]
        out = (scored_by, hits)
        self._board_universe_cache[key] = out
        return out

    def board_pool(
        self, count: int = 400, model: str | None = None,
        per_scene: int = 4, exclude: set | None = None,
        min_score: float = 0.0, seed: int | None = None, profile: str | None = None,
    ) -> dict:
        """The endless For You megaboard's supply of moments — full-library
        coverage, one peak per scene. Every scene contributes its best
        taste-moment (plus its next-best few for depth), scored by your trained
        model when you have one; `min_score` is then an *optional tightener*
        (0 = off → every scene's peak; raise it → only your strongest scenes).
        `exclude` (scene_ids already shown) marches the stream forward across
        batches. `count` moments come back as a rank-weighted random sample so
        strong matches surface first but the whole library stays reachable."""
        model = model or self._model_name()
        scored_by, uni = self._board_universe(model, per_scene, profile=profile)
        if not uni:
            return {"hits": [], "scenes": 0, "moments": 0, "model": model, "scored_by": None}
        pool = [h for h in uni if h.score >= min_score] if min_score > 0 else uni
        scenes_total = len({str(h.scene_id) for h in pool})
        moments_total = len(pool)
        if exclude:
            exclude = {str(s) for s in exclude}
            pool = [h for h in pool if str(h.scene_id) not in exclude]
        diversity = self.cfg.modeling.feed_diversity
        if min_score > 0:
            # The floor already gates quality → draw a *uniform-random* sample within
            # it, so the board is genuinely different every load instead of replaying
            # the same top-scored moments. (per-scene cap + a huge pool keep it varied.)
            hits = self._sample_uniform(pool, count, seed)
        elif diversity > 0 and len(pool) > count:
            # No floor → surface your strongest: a broad rank-weighted candidate
            # sample, then MMR-select so it spreads across your taste's modes.
            cand = self._sample_ranked(pool, min(len(pool), 3 * count), seed)
            hits = self._diversify(cand, model, count, diversity)
        else:
            hits = self._sample_ranked(pool, count, seed)
        return {
            "hits": hits, "scenes": scenes_total, "moments": moments_total,
            "model": model, "scored_by": scored_by,
        }

    def _sample_uniform(
        self, pool: list[Hit], count: int, seed: int | None = None
    ) -> list[Hit]:
        """Unbiased random sample (without replacement) of `count` moments in
        random order — every moment in `pool` equally likely, no rank weighting.
        Shuffles even when the pool is ≤ `count`. Used when a taste floor is set:
        the floor is the quality gate, so within it the board should be truly
        random, not a replay of the top-scored moments."""
        n = len(pool)
        if n == 0:
            return []
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(count, n), replace=False)
        return [pool[int(i)] for i in idx]

    def _sample_ranked(
        self, ranked: list[Hit], count: int, seed: int | None = None
    ) -> list[Hit]:
        """Rank-weighted random sample (without replacement) of `count` moments
        from a best-first pool: weights decay by rank so strong matches are more
        likely, but a broad temperature keeps the whole tail reachable — the
        board never fixates on the same few hundred. Returns fewer than `count`
        only when the pool is smaller."""
        n = len(ranked)
        if n <= count:
            return ranked
        rng = np.random.default_rng(seed)
        temp = float(max(n / 4.0, count, 200.0))  # broad: spread across the pool
        w = np.exp(-np.arange(n) / temp)
        p = w / w.sum()
        order = rng.choice(n, size=count, replace=False, p=p)
        return [ranked[int(i)] for i in order]

    def _diversify(self, hits: list[Hit], model: str, k: int, diversity: float) -> list[Hit]:
        """Maximal Marginal Relevance: pick a top-`k` that balances taste-rank
        against variety, so the feed spans your taste instead of collapsing into
        near-duplicates of your single favourite. `diversity` in [0,1]: 0 keeps
        the pure ranking, higher trades relevance for spread."""
        if diversity <= 0 or len(hits) <= k:
            return hits[:k]
        idx = self.index(model)
        dim = idx.dim or 1
        rows = []
        for h in hits:
            v = idx.vector_at(h.key, h.time)
            rows.append(v if v is not None else np.zeros(dim, dtype=np.float32))
        c = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(c, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        c = c / norms  # unit rows → dot = cosine
        n = len(hits)
        rel = np.linspace(1.0, 0.0, n, dtype=np.float32)  # rank-based (best first)
        lam = 1.0 - float(np.clip(diversity, 0.0, 1.0))
        chosen = [0]
        max_sim = c @ c[0]  # similarity of every candidate to the first pick
        picked = {0}
        while len(chosen) < min(k, n):
            mmr = lam * rel - (1.0 - lam) * max_sim
            mmr[list(picked)] = -np.inf
            j = int(np.argmax(mmr))
            chosen.append(j)
            picked.add(j)
            max_sim = np.maximum(max_sim, c @ c[j])
        return [hits[i] for i in chosen]

    def next_uncertain(self, model: str | None = None, pool: int = 800,
                       profile: str | None = None) -> dict | None:
        """The unlabeled frame the taste model is least sure about — active
        learning: rating the ambiguous ones teaches it fastest. Falls back to
        centroid-ambiguity, then random, when there's no model/centroid yet."""
        model = model or self._model_name()
        idx = self.index(model, refresh=True)  # pick up frames from an in-progress embed
        if idx.size == 0:
            return None
        profile = profile or self.cfg.markers.tag_name
        labeled = self._label_store().labeled_ids(profile)
        rng = np.random.default_rng()
        rows = rng.choice(idx.size, size=min(pool, idx.size), replace=False)
        clf = self._taste_model(profile, model)
        if clf is not None:
            p = np.asarray(clf.predict_proba(idx.matrix[rows]), dtype=np.float32)
            unc = -np.abs(p - 0.5)  # nearest the 0.5 decision boundary
        else:
            # use the centroid only if it's already cached — never trigger a
            # (network) marker rebuild from the swipe loop.
            cached = self._taste_src_cache.get((model, profile))
            if cached is not None and cached[0].shape[0] > 0:
                c = self._unit(cached[0].mean(axis=0))
                sims = idx.matrix[rows] @ c
                unc = -np.abs(sims - float(np.median(sims)))  # mid-similarity = ambiguous
            else:
                unc = rng.random(rows.shape[0])  # cold start: anything
        for j in np.argsort(-unc):
            i = int(rows[j])
            key, t = idx.keys[i], float(idx.times[i])
            if (key, round(t, 2)) in labeled:
                continue
            return {"key": key, "time": round(t, 2), "scene_id": idx.scene_ids[i], "score": 0.0}
        return None

    def sample_frames(
        self, count: int = 10, model: str | None = None, seed: int | None = None
    ) -> list[dict]:
        """A batch of `count` uniformly-random distinct frames from the whole
        library — the Taste Picker collage. No taste model, no uncertainty, no
        labeled-set filtering: deliberately just random (re-picking an already
        loved frame is a harmless upsert on /api/label)."""
        model = model or self._model_name()
        idx = self.index(model, refresh=True)  # pick up frames from an in-progress embed
        if idx.size == 0:
            return []
        rng = np.random.default_rng(seed)
        rows = rng.choice(idx.size, size=min(count, idx.size), replace=False)
        return [
            {
                "key": idx.keys[i],
                "time": round(float(idx.times[i]), 2),
                "scene_id": idx.scene_ids[i],
            }
            for i in (int(r) for r in rows)
        ]

    def _moments_for_scenes(
        self, scene_ids, per_scene: int = 6, model: str | None = None
    ) -> list["Hit"]:
        """Up to `per_scene` frames spread across each given scene, as Hits —
        the moment pool behind 'more from this actress' (plays her scenes)."""
        from ..search import Hit

        model = model or self._model_name()
        idx = self.index(model)
        sid_key = {str(m.get("scene_id")): k for k, m in idx.key_meta.items()}
        hits: list[Hit] = []
        for sid in scene_ids:
            key = sid_key.get(str(sid))
            if key is None:
                continue
            rows = idx._key_rows.get(key)
            if not rows:
                continue
            start, end = rows
            n = end - start
            step = max(1, n // per_scene)
            for i in range(start, end, step):
                hits.append(Hit(
                    scene_id=idx.scene_ids[i], key=idx.keys[i],
                    time=float(idx.times[i]), score=0.0,
                ))
        return hits

    def performer_board(
        self, scene_id: str, count: int = 300, per_scene: int = 6
    ) -> dict:
        """'More from this actress': resolve this scene's lead performer, pull
        her other scenes from Stash, keep the embedded ones, and return a
        shuffled pool of moments from them. `{performer, hits}` (empty hits when
        she has no other embedded scenes / Stash is unreachable)."""
        empty = {"performer": None, "hits": []}
        try:
            details = self.client().scene_details([str(scene_id)])
        except Exception:  # noqa: BLE001 — Stash down
            return empty
        perfs = (details.get(str(scene_id)) or {}).get("performers_detail") or []
        perfs = [p for p in perfs if p.get("id")]
        if not perfs:
            return empty
        model = self._model_name()
        idx = self.index(model)
        embedded = {str(m.get("scene_id")) for m in idx.key_meta.values()}

        # pick the performer with the most *embedded* scenes to play from
        best_name, best_scenes = None, []
        for p in perfs:
            try:
                scenes = self.client().scenes_for_performer(p["id"])
            except Exception:  # noqa: BLE001
                scenes = []
            keep = [s for s in scenes if str(s) in embedded]
            if len(keep) > len(best_scenes):
                best_name, best_scenes = p.get("name") or "this performer", keep
        if not best_scenes:
            return empty

        hits = self._moments_for_scenes(best_scenes, per_scene=per_scene, model=model)
        rng = np.random.default_rng()
        rng.shuffle(hits)
        return {"performer": best_name, "hits": hits[:count]}

    def performer_moment_matches(
        self, scene_id: str, t: float, count: int = 300, per_scene: int = 6
    ) -> dict:
        """'More of THIS moment, same actress': moments across this scene's lead
        performer's embedded scenes, ranked by visual similarity to the moment at
        (scene_id, t). Falls back to a spread of her scenes if the moment vector
        can't be located. `{performer, hits}`."""
        model = self._model_name()
        _pid, pname, scenes = self._resolve_performer(scene_id=scene_id)
        if not scenes:
            return {"performer": pname, "hits": []}
        key = self._key_for_scene(scene_id, model)
        v = self.index(model).vector_at(key, float(t)) if key else None
        if v is None:  # can't locate the moment → her spread instead
            hits = self._moments_for_scenes(scenes, per_scene=per_scene, model=model)
        else:
            hits = self._ranked_moments_for_scenes(scenes, v, per_scene=per_scene, model=model)
        return {"performer": pname, "hits": hits[:count]}

    def scene_moments(
        self, scene_id: str, count: int = 300, per_scene: int = 60
    ) -> dict:
        """'More moments in THIS scene': a diverse, time-spread pool of moments
        drawn from a single scene. `{hits}`."""
        model = self._model_name()
        hits = self._moments_for_scenes([str(scene_id)], per_scene=per_scene, model=model)
        return {"hits": hits[:count]}

    def _ranked_moments_for_scenes(
        self, scene_ids, query_vec, per_scene: int = 6, model: str | None = None,
    ) -> list["Hit"]:
        """Top `per_scene` frames of each scene by cosine to `query_vec` (a unit
        vector — taste centroid or a CLIP text query), as Hits with the real
        score. The 'best moments' engine (vs `_moments_for_scenes`, which is
        time-spread with score 0)."""
        from ..search import Hit

        model = model or self._model_name()
        idx = self.index(model)
        if idx.size == 0 or query_vec is None:
            return []
        q = self._unit(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        sid_key = {str(m.get("scene_id")): k for k, m in idx.key_meta.items()}
        hits: list[Hit] = []
        for sid in scene_ids:
            key = sid_key.get(str(sid))
            if key is None:
                continue
            rows = idx._key_rows.get(key)
            if not rows:
                continue
            start, end = rows
            scores = idx.matrix[start:end] @ q
            order = np.argsort(-scores)[:per_scene]
            for j in order:
                i = start + int(j)
                hits.append(Hit(
                    scene_id=idx.scene_ids[i], key=idx.keys[i],
                    time=float(idx.times[i]), score=float(scores[int(j)]),
                ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def _resolve_performer(self, name=None, performer_id=None, scene_id=None):
        """(performer_id, name, embedded_scene_ids) for a name / id / scene, keeping
        only her scenes that are embedded. Picks the best-embedded match on a name."""
        idx = self.index(self._model_name())
        embedded = {str(m.get("scene_id")) for m in idx.key_meta.values()}
        candidates = []
        if scene_id is not None:
            try:
                d = self.client().scene_details([str(scene_id)])
                candidates = [p for p in (d.get(str(scene_id)) or {}).get("performers_detail") or [] if p.get("id")]
            except Exception:  # noqa: BLE001
                candidates = []
        elif performer_id is not None:
            candidates = [{"id": str(performer_id), "name": name or ""}]
        elif name:
            try:
                candidates = self.client().find_performers(name)
            except Exception:  # noqa: BLE001
                candidates = []
        excluded = self._excluded_performers()
        candidates = [p for p in candidates if (p.get("name", "").strip().lower() not in excluded)]
        best = None
        for p in candidates:
            try:
                scenes = [s for s in self.client().scenes_for_performer(p["id"]) if str(s) in embedded]
            except Exception:  # noqa: BLE001
                scenes = []
            if best is None or len(scenes) > len(best[2]):
                best = (str(p["id"]), p.get("name") or name or "this performer", scenes)
        return best or (None, None, [])

    def performer_best(
        self, name=None, performer_id=None, scene_id=None, count: int = 200,
        per_scene: int = 6, query: str | None = None,
        min_score: float = 0.0, spread: bool = False,
    ) -> dict:
        """A performer's moments — by default her BEST, her embedded scenes' frames
        ranked by your taste centroid (or by a CLIP text `query`, 'her best
        lingerie'). `spread=True` instead returns a diverse, time-spread coverage of
        her scenes (the megaboard's 0% floor); `min_score>0` keeps only taste hits at
        or above that floor (tightening toward your taste). Returns `{performer,
        hits}` ready to save as a collection."""
        pid, pname, scenes = self._resolve_performer(name, performer_id, scene_id)
        if not scenes:
            return {"performer": pname, "hits": []}
        if query:  # focus on an attribute → rank in CLIP space
            model = self._clip_name()
            qvec = self._clip_query_vector(query)
            hits = self._ranked_moments_for_scenes(scenes, qvec, per_scene=per_scene, model=model)
        elif spread:  # diverse span of her scenes, no taste ranking (floor at 0%)
            hits = self._moments_for_scenes(scenes, per_scene=per_scene)
        else:      # rank by taste closeness in the DINO space
            model = self._model_name()
            qvec, _, _ = self._taste_centroid(model)
            if qvec is None:  # no taste yet → fall back to spread coverage
                hits = self._moments_for_scenes(scenes, per_scene=per_scene)
            else:
                hits = self._ranked_moments_for_scenes(scenes, qvec, per_scene=per_scene, model=model)
                if min_score > 0:  # tighten toward your taste
                    hits = [h for h in hits if h.score >= min_score]
        return {"performer": pname, "hits": hits[:count]}

    def _excluded_performers(self) -> set[str]:
        """Names to hide from the Performers tab — junk 'performers' that are
        really mis-applied tags (e.g. 'upscale'). Never touches Stash. Configurable
        via env PEAKS_PERFORMER_EXCLUDE (comma-separated); defaults to 'upscale'."""
        import os

        raw = os.environ.get("PEAKS_PERFORMER_EXCLUDE", "upscale")
        return {n.strip().lower() for n in raw.split(",") if n.strip()}

    def performer_stats(self, rebuild: bool = False) -> list[dict]:
        """Leaderboard of the library's performers by moments generated: for each
        performer appearing on an embedded scene, {id, name, scenes, moments, and
        taste_best/mean when a centroid exists}. Cached (Stash-heavy to build)."""
        if not rebuild and getattr(self, "_perf_stats_cache", None) is not None:
            return self._perf_stats_cache
        model = self._model_name()
        idx = self.index(model)
        # scene_id -> (key, n_frames)
        sid_key = {str(m.get("scene_id")): k for k, m in idx.key_meta.items()}
        embedded = list(sid_key)
        if not embedded:
            self._perf_stats_cache = []
            return []
        try:
            details = self.client().scene_details(embedded)
        except Exception:  # noqa: BLE001 — Stash down
            return getattr(self, "_perf_stats_cache", None) or []

        c, _, _ = self._taste_centroid(model)
        cu = self._unit(c) if c is not None else None
        dim = idx.dim or 1
        excluded = self._excluded_performers()
        # performer id -> {name, scenes:set, o, rating_sum, rating_n}
        perf: dict[str, dict] = {}
        for sid, meta in details.items():
            for p in meta.get("performers_detail") or []:
                if not p.get("id") or (p.get("name", "").strip().lower() in excluded):
                    continue
                e = perf.setdefault(str(p["id"]), {
                    "name": p.get("name", ""), "scenes": set(),
                    "o": 0, "rating_sum": 0.0, "rating_n": 0,
                })
                e["scenes"].add(str(sid))
                e["o"] += int(meta.get("o_counter") or 0)
                if meta.get("rating100") is not None:
                    e["rating_sum"] += float(meta["rating100"])
                    e["rating_n"] += 1

        rows = []
        centroids: dict[str, np.ndarray] = {}
        for pid, e in perf.items():
            moments = 0
            best = mean_sum = 0.0
            vsum = np.zeros(dim, dtype=np.float32)
            ranked: list[tuple[float, int]] = []  # (score, matrix_row) for her top moments
            for sid in e["scenes"]:
                rows_span = idx._key_rows.get(sid_key.get(sid, ""))
                if not rows_span:
                    continue
                start, end = rows_span
                moments += end - start
                vsum += idx.matrix[start:end].sum(axis=0)
                if cu is not None:
                    s = idx.matrix[start:end] @ cu
                    best = max(best, float(s.max()))
                    mean_sum += float(s.sum())
                    j = int(np.argmax(s))
                    ranked.append((float(s[j]), start + j))  # this scene's best frame
                else:
                    ranked.append((0.0, start))  # no taste yet: a representative frame
            if moments:
                centroids[pid] = self._unit(vsum)
            ranked.sort(reverse=True)

            def _stream(sid, t):
                try:
                    return self.stream_url(sid, start=t) if sid else None
                except Exception:  # noqa: BLE001 — a bad stream url shouldn't kill the board
                    return None

            top = [{
                "key": idx.keys[i], "t": round(float(idx.times[i]), 2),
                "scene_id": idx.scene_ids[i],
                "thumb": f"/api/frame?key={idx.keys[i]}&t={idx.times[i]:g}",
                "stream": _stream(idx.scene_ids[i], idx.times[i]),
            } for _, i in ranked[:6]]
            taste_mean = round(mean_sum / moments, 4) if (cu is not None and moments) else None
            rows.append({
                "id": pid, "name": e["name"], "scenes": len(e["scenes"]),
                "moments": int(moments),
                "taste_best": round(best, 4) if cu is not None else None,
                "taste_mean": taste_mean, "affinity": taste_mean,
                "o_counter": e["o"],
                "rating": round(e["rating_sum"] / e["rating_n"], 1) if e["rating_n"] else None,
                "top": top,
            })
        rows.sort(key=lambda r: r["moments"], reverse=True)
        self._perf_stats_cache = rows
        self._perf_centroids = centroids
        self._perf_rows_by_id = {r["id"]: r for r in rows}
        # her embedded scene ids, for the Statistics tab's per-performer peak counts
        self._perf_scenes_by_id = {pid: set(e["scenes"]) for pid, e in perf.items()}
        return rows

    # --- Statistics tab: peaks over the whole library --------------------------

    def _peak_score_fn(self, model: str):
        """(score_frames, source_label) for the peaks pipeline, or (None, reason).
        Mirrors `score`'s scorer selection, then falls back to the For You taste
        model and finally the taste centroid, so peaks are computable whenever any
        taste exists."""
        from pathlib import Path

        from ..classifier import TasteClassifier
        from ..pipeline import safe_tag

        tag = self.cfg.markers.tag_name
        p = Path(self.cfg.modeling.dir) / f"{safe_tag(tag)}.pkl"
        if p.exists():
            try:
                return TasteClassifier.load(p).predict_proba, "score model"
            except Exception:  # noqa: BLE001 — fall through to taste-based scorers
                pass
        clf = self._taste_model(tag, model)
        if clf is not None:
            return clf.predict_proba, "taste model"
        c, _, _ = self._taste_centroid(model)
        if c is not None:
            cu = self._unit(c)
            return (lambda vecs: np.asarray(vecs, dtype=np.float32) @ cu), "taste centroid"
        return None, "no taste yet"

    def _scene_segments(self, model: str | None = None, rebuild: bool = False):
        """`({scene_id: [Segment, ...]}, source)` — the apex segments (peaks) the
        scorer extracts from every embedded scene, via the same `score_scene`
        pipeline `peaks score` uses. Cached, keyed on the index size so new embeds
        recompute it. When `normalize=none` the hysteresis thresholds are drawn
        from the library score distribution (p90/p75) so the result is meaningful
        whichever scorer is active. Shared by the peak-count stats (`_peak_index`)
        and the reclamation report."""
        from dataclasses import replace

        from ..pipeline import score_scene
        from ..scoring import extract_segments, smooth

        model = model or self._model_name()
        idx = self.index(model)
        cached = getattr(self, "_scene_seg_cache", None)
        if not rebuild and cached is not None and cached[0] == idx.size:
            return cached[1]

        fn, source = self._peak_score_fn(model)
        if fn is None or idx.size == 0:
            out = ({}, source)
            self._scene_seg_cache = (idx.size, out)
            return out

        sc = self.cfg.scoring
        segments: dict[str, list] = {}
        if sc.normalize in ("", "none"):
            flat = np.asarray(fn(idx.matrix), dtype=np.float32).reshape(-1)
            high = float(np.percentile(flat, 90.0)) if flat.size else sc.high
            low = float(np.percentile(flat, 75.0)) if flat.size else sc.low
            for key, (start, end) in idx._key_rows.items():
                sid = str((idx.key_meta.get(key) or {}).get("scene_id"))
                if not sid or sid == "None":
                    continue
                series = smooth(flat[start:end], sc.smooth_window)
                segs = extract_segments(
                    series, idx.times[start:end], high=high, low=low,
                    min_duration=sc.min_duration, merge_gap=sc.merge_gap,
                    max_duration=sc.max_duration or None, pad=sc.pad,
                )
                if segs:
                    segments[sid] = segs
        else:  # scene-z: thresholds are std-devs; honour the configured scoring
            scoring = replace(sc)
            for key, (start, end) in idx._key_rows.items():
                sid = str((idx.key_meta.get(key) or {}).get("scene_id"))
                if not sid or sid == "None":
                    continue
                segs = score_scene(idx.times[start:end], idx.matrix[start:end], fn, scoring)
                if segs:
                    segments[sid] = segs

        out = (segments, source)
        self._scene_seg_cache = (idx.size, out)
        return out

    def _peak_index(self, model: str | None = None, rebuild: bool = False) -> dict:
        """`{peaks: {scene_id: [{t, score}, ...]}, total, scenes, source}` — the
        per-scene peaks as midpoint+score points (for the Statistics tab), derived
        from the shared `_scene_segments` pass. Cached on index size."""
        model = model or self._model_name()
        idx = self.index(model)
        cached = getattr(self, "_peak_index_cache", None)
        if not rebuild and cached is not None and cached[0] == idx.size:
            return cached[1]
        segments, source = self._scene_segments(model, rebuild=rebuild)
        peaks = {
            sid: [
                {"t": round(float(s.midpoint), 2), "score": round(float(s.peak_score), 4)}
                for s in segs
            ]
            for sid, segs in segments.items()
        }
        total = sum(len(v) for v in peaks.values())
        out = {"peaks": peaks, "total": total, "scenes": len(peaks), "source": source}
        self._peak_index_cache = (idx.size, out)
        return out

    def reclamation_report(
        self,
        floor: float | None = None,
        waste_ratio: float = 0.4,
        min_scene_secs: float = 0.0,
        limit: int = 0,
        model: str | None = None,
    ) -> dict:
        """Where the disk is going: for every embedded scene, compare its total
        *peak* footage (segments whose peak_score ≥ `floor`) against the whole
        file, and bucket the dead weight.

          • no_peak — zero qualifying peaks → the whole file is reclaimable.
          • sparse  — has peaks but they cover less than `waste_ratio` of the
                      file → a per-scene peak reel would keep only ~kept_frac of
                      the bytes; the rest is reclaimable.

        READ-ONLY: nothing is written, marked, or deleted. Sizes are read from
        disk (`os.path.getsize`); durations from Stash. Scenes already queued
        (`hidden_scene_ids`) and files that can't be resolved are excluded from
        the actionable lists (the latter counted as `unresolved`). Rows are
        sorted by reclaimable bytes, biggest first; `limit`>0 caps each list
        (totals always span the whole library)."""
        import os

        model = model or self._model_name()
        segments, source = self._scene_segments(model)
        idx = self.index(model)
        empty = {
            "floor": floor, "waste_ratio": waste_ratio, "source": source, "model": model,
            "no_peak": [], "sparse": [],
            "totals": {"no_peak_scenes": 0, "no_peak_bytes": 0, "sparse_scenes": 0,
                       "sparse_bytes": 0, "total_bytes": 0, "considered": 0, "unresolved": 0},
        }
        if idx.size == 0:
            return empty

        sids = sorted({str((m or {}).get("scene_id")) for m in idx.key_meta.values()}
                      - {"None", ""})
        hidden = self.hidden_scene_ids()
        sids = [s for s in sids if s not in hidden]
        if not sids:
            return empty
        try:
            details = self.client().scene_details(sids)
        except Exception:  # noqa: BLE001 — Stash down: no sizes/durations to report
            return empty

        no_peak: list[dict] = []
        sparse: list[dict] = []
        unresolved = considered = 0
        for sid in sids:
            d = details.get(sid) or {}
            path = d.get("path")
            if not path or not os.path.exists(path):
                unresolved += 1
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                unresolved += 1
                continue
            duration = float(d.get("duration") or 0.0)
            segs = segments.get(sid, [])
            if floor is not None:
                segs = [s for s in segs if s.peak_score >= floor]
            if duration < min_scene_secs:
                continue
            considered += 1
            kept_secs = float(sum(s.duration for s in segs))
            peak_score = max((float(s.peak_score) for s in segs), default=0.0)
            title = d.get("title") or ""
            base = {
                "scene_id": sid, "title": title, "path": path,
                "size": size, "duration": round(duration, 1),
                "kept_secs": round(kept_secs, 1), "peak_score": round(peak_score, 4),
            }
            if not segs:
                no_peak.append({**base, "kept_frac": 0.0, "reclaim_bytes": size})
                continue
            kept_frac = (kept_secs / duration) if duration > 0 else 1.0
            if kept_frac < waste_ratio:
                est_bytes = int(size * kept_frac)
                sparse.append({
                    **base, "kept_frac": round(kept_frac, 3),
                    "est_bytes": est_bytes, "reclaim_bytes": size - est_bytes,
                })
            # else: peaky enough — keep the whole file, not reclaimable

        no_peak.sort(key=lambda r: r["reclaim_bytes"], reverse=True)
        sparse.sort(key=lambda r: r["reclaim_bytes"], reverse=True)
        totals = {
            "no_peak_scenes": len(no_peak),
            "no_peak_bytes": sum(r["reclaim_bytes"] for r in no_peak),
            "sparse_scenes": len(sparse),
            "sparse_bytes": sum(r["reclaim_bytes"] for r in sparse),
            "considered": considered, "unresolved": unresolved,
        }
        totals["total_bytes"] = totals["no_peak_bytes"] + totals["sparse_bytes"]
        return {
            "floor": floor, "waste_ratio": waste_ratio, "source": source, "model": model,
            "no_peak": no_peak[:limit] if limit else no_peak,
            "sparse": sparse[:limit] if limit else sparse,
            "totals": totals,
        }

    def reel_scene(
        self,
        scene_id: str,
        floor: float | None = None,
        dry_run: bool = True,
        job=None,
        model: str | None = None,
    ) -> dict:
        """Plan (or, when `dry_run=False`, produce) a per-scene *peak reel*: a new
        file holding only this scene's qualifying peak segments, in chronological
        order, losslessly stream-copied (`-c copy`, no re-encode). Peaks NEVER
        touches the original — the reel is a separate file in the exports dir.

        The POC default is `dry_run=True`: it returns the plan (the segments it
        would keep, the estimated output bytes, the path it would write) and runs
        no ffmpeg / writes nothing. `dry_run=False` (a deliberately-enabled later
        step) actually cuts the reel via `_reel_segments`."""
        import os
        from pathlib import Path

        log = (job.log if job else (lambda *_: None))
        model = model or self._model_name()
        sid = str(scene_id)
        segments, _ = self._scene_segments(model)
        segs = segments.get(sid, [])
        if floor is not None:
            segs = [s for s in segs if s.peak_score >= floor]
        segs = sorted(segs, key=lambda s: s.start)  # chronological
        d = (self.client().scene_details([sid]) or {}).get(sid) or {}
        path = d.get("path")
        duration = float(d.get("duration") or 0.0)
        size = os.path.getsize(path) if (path and os.path.exists(path)) else 0
        kept_secs = float(sum(s.duration for s in segs))
        kept_frac = (kept_secs / duration) if duration > 0 else 0.0
        est_bytes = int(size * kept_frac)
        exports = Path(os.environ.get("PEAKS_EXPORT_DIR", "/config/exports")) / "reels"
        out_path = str(exports / (_safe_reel_name(f"peaks-scene-{sid}") + ".mp4"))
        plan = {
            "scene_id": sid, "path": path, "out_path": out_path,
            "segments": [{"start": round(s.start, 2), "end": round(s.end, 2),
                          "duration": round(s.duration, 2), "peak_score": round(float(s.peak_score), 4)}
                         for s in segs],
            "kept_secs": round(kept_secs, 1), "kept_frac": round(kept_frac, 3),
            "size": size, "est_bytes": est_bytes, "dry_run": dry_run,
        }
        if dry_run:
            log(f"[dry-run] scene {sid}: would reel {len(segs)} segment(s) "
                f"(~{kept_secs:.0f}s, ~{est_bytes // 1_000_000} MB) → {out_path}")
            return plan
        if not segs:
            raise ValueError(f"scene {sid}: no qualifying peaks to reel")
        if not path or not os.path.exists(path):
            raise ValueError(f"scene {sid}: source file missing")
        exports.mkdir(parents=True, exist_ok=True)
        written = self._reel_segments(path, segs, out_path, log=log)
        plan["bytes"] = os.path.getsize(written) if os.path.exists(written) else 0
        plan["out_path"] = written
        return plan

    def _reel_segments(self, path: str, segments, out: str, log=None) -> str:
        """Lossless stream-copy of each [start,end] segment from `path`, concatenated
        into `out` (`-ss/-to -c copy` then a concat pass). Shared by the peak-reel
        and the collection/reel exports. Returns the output path."""
        import os
        import subprocess
        import tempfile

        log = log or (lambda *_: None)
        ff = getattr(self.cfg.sampling, "ffmpeg", "ffmpeg") if hasattr(self.cfg, "sampling") else "ffmpeg"
        with tempfile.TemporaryDirectory() as td:
            segfiles: list[str] = []
            for i, s in enumerate(segments):
                start, end = float(s.start), float(s.end)
                seg = os.path.join(td, f"seg{i:04d}.ts")
                r = subprocess.run(
                    [ff, "-y", "-ss", f"{start:g}", "-to", f"{end:g}", "-i", path,
                     "-c", "copy", "-f", "mpegts", seg],
                    capture_output=True,
                )
                if r.returncode == 0 and os.path.exists(seg) and os.path.getsize(seg) > 0:
                    segfiles.append(seg)
                else:
                    log(f"  ! segment {start:.0f}-{end:.0f}s failed (codec mismatch?) — skipped")
            if not segfiles:
                raise RuntimeError("no segments could be cut (codec mismatch?)")
            listf = os.path.join(td, "list.txt")
            with open(listf, "w") as f:
                for sf in segfiles:
                    f.write(f"file '{sf}'\n")
            cc = subprocess.run(
                [ff, "-y", "-f", "concat", "-safe", "0", "-i", listf, "-c", "copy", out],
                capture_output=True,
            )
            if cc.returncode != 0:
                raise RuntimeError("concat failed: " + cc.stderr.decode("replace")[-300:])
        return out

    def _scene_mtimes(self, model: str) -> dict[str, float]:
        """scene_id -> when Peaks last wrote its embedding (the 'analyzed at'
        signal for the freshness stats)."""
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        idx = self.index(model)
        key_sid = {k: str((m or {}).get("scene_id")) for k, m in idx.key_meta.items()}
        out: dict[str, float] = {}
        for key, t in cache.mtimes(model).items():
            sid = key_sid.get(key)
            if sid and sid != "None":
                out[sid] = max(out.get(sid, 0.0), t)
        return out

    def statistics(self) -> dict:
        """Everything the Statistics tab shows: build health/backlog, a freshness
        timeline proving Peaks keeps ingesting new scenes, the peak leaderboards,
        and the standout on-taste scene."""
        import time

        model = self._model_name()
        idx = self.index(model)
        pk = self._peak_index(model)
        peaks = pk["peaks"]
        now = time.time()

        embedded = {str(m.get("scene_id")) for m in idx.key_meta.values() if m.get("scene_id")}
        try:
            total_library = self.client().scene_count()
        except Exception:  # noqa: BLE001 — Stash down
            total_library = None
        from ..failures import failure_log_for

        rows = self.performer_stats()
        perf_scenes = getattr(self, "_perf_scenes_by_id", {})
        rows_by_id = getattr(self, "_perf_rows_by_id", {})

        # freshness — scenes analyzed into the build over time (cache mtimes)
        scene_mtime = self._scene_mtimes(model)

        def window(days: float) -> dict:
            cutoff = now - days * 86400
            sids = [s for s, t in scene_mtime.items() if t >= cutoff]
            return {"scenes": len(sids), "peaks": sum(len(peaks.get(s, ())) for s in sids)}

        weeks = 12
        buckets = [0] * weeks
        for t in scene_mtime.values():
            w = int((now - t) // (7 * 86400))
            if 0 <= w < weeks:
                buckets[weeks - 1 - w] += 1  # oldest → newest
        last_sid = max(scene_mtime, key=scene_mtime.get) if scene_mtime else None

        # performer peak leaderboard
        peak_rows = []
        for pid, scenes in perf_scenes.items():
            p = sum(len(peaks.get(s, ())) for s in scenes)
            if p <= 0:
                continue
            r = rows_by_id.get(pid, {})
            peak_rows.append({
                "id": pid, "name": r.get("name", ""), "peaks": p,
                "scenes": len(scenes), "taste": r.get("taste_mean"),
            })
        peak_rows.sort(key=lambda r: r["peaks"], reverse=True)
        taste_rows = [r for r in rows if r.get("taste_mean") is not None]
        top_taste = max(taste_rows, key=lambda r: r["taste_mean"], default=None)

        # most on-taste scene — the single highest peak in the library
        best_sid, best_score = None, -1.0
        for sid, lst in peaks.items():
            for seg in lst:
                if seg["score"] > best_score:
                    best_score, best_sid = seg["score"], sid

        titles = {}
        want = [s for s in (best_sid, last_sid) if s]
        if want:
            try:
                titles = self.client().scene_details(want)
            except Exception:  # noqa: BLE001
                titles = {}

        def _scene_card(sid, extra=None):
            if not sid:
                return None
            m = titles.get(sid) or {}
            perfs = ", ".join(p.get("name", "") for p in (m.get("performers_detail") or []) if p.get("name"))
            card = {"scene_id": sid, "title": m.get("title") or f"scene {sid}", "performers": perfs}
            if extra:
                card.update(extra)
            return card

        return {
            "peak_term": "peaks",  # a "peak" = a scored apex segment
            "peak_source": pk["source"],
            "build": {
                "embedded_scenes": len(embedded),
                "library_scenes": total_library,
                "backlog": (total_library - len(embedded)) if total_library is not None else None,
                "frames": idx.size,
                "performers": len(perf_scenes),
                "total_peaks": pk["total"],
                "failures": len(failure_log_for(self.cfg)),
            },
            "freshness": {
                "timeline_weeks": buckets,
                "last_24h": window(1),
                "last_7d": window(7),
                "last_30d": window(30),
                "last_analyzed": _scene_card(
                    last_sid, {"at": scene_mtime.get(last_sid)} if last_sid else None
                ),
            },
            "top_actress_by_peaks": peak_rows[0] if peak_rows else None,
            "leaderboard": peak_rows[:10],
            "top_actress_by_taste": (
                {"id": top_taste["id"], "name": top_taste["name"], "taste": top_taste["taste_mean"]}
                if top_taste else None
            ),
            "most_ontaste_scene": _scene_card(
                best_sid, {"score": round(best_score, 4)} if best_sid else None
            ),
        }

    def stats_board(self, metric: str, id: str | None = None, count: int = 1500) -> dict:
        """Moments for a Statistics-tab card, so each stat is playable on the
        megaboard as its own distinct playlist. `{performer?, hits}`."""
        from ..search import Hit

        model = self._model_name()
        idx = self.index(model)

        if metric in ("actress", "most_peaks_actress", "most_ontaste_actress"):
            pid = id
            if metric != "actress":
                st = self.statistics()
                pick = st["top_actress_by_peaks"] if metric == "most_peaks_actress" else st["top_actress_by_taste"]
                pid = (pick or {}).get("id")
            if not pid:
                return {"hits": []}
            return self.performer_best(
                performer_id=pid, count=count, per_scene=40, min_score=0.0, spread=True
            )

        if metric == "most_ontaste_scene":
            sid = id
            if not sid:
                sid = (self.statistics().get("most_ontaste_scene") or {}).get("scene_id")
            if not sid:
                return {"hits": []}
            return {"hits": self.scene_moments(str(sid), count=count)["hits"]}

        if metric == "fresh":
            pk = self._peak_index(model)
            peaks = pk["peaks"]
            scene_mtime = self._scene_mtimes(model)
            sid_key = {str(m.get("scene_id")): k for k, m in idx.key_meta.items()}
            order = sorted(scene_mtime, key=scene_mtime.get, reverse=True)
            hits: list[Hit] = []
            for sid in order:
                key = sid_key.get(sid)
                if key is None:
                    continue
                for seg in peaks.get(sid, ()):  # this scene's peaks, newest scenes first
                    hits.append(Hit(scene_id=sid, key=key, time=float(seg["t"]), score=float(seg["score"])))
                    if len(hits) >= count:
                        return {"hits": hits}
            return {"hits": hits}

        return {"hits": []}

    def similar_performers(self, performer_id: str, k: int = 8) -> list[dict]:
        """Performers whose moments sit closest to hers in embedding space —
        'if you like her, try these'. Cosine of unit performer centroids."""
        self.performer_stats()  # ensure centroids/rows are built
        cents = getattr(self, "_perf_centroids", {})
        rows = getattr(self, "_perf_rows_by_id", {})
        me = cents.get(str(performer_id))
        if me is None:
            return []
        out = []
        for pid, v in cents.items():
            if pid == str(performer_id):
                continue
            r = rows.get(pid, {})
            out.append({"id": pid, "name": r.get("name", ""),
                        "score": round(float(me @ v), 4), "top": r.get("top", [])})
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:k]

    def performer_fingerprint(self, scene_ids, top_k: int = 10) -> list[list]:
        """'What she's known for': her CLIP frames averaged and matched against
        the vocabulary → top attribute terms. [] when CLIP isn't embedded."""
        if not self.has_clip_index():
            return []
        clip = self._clip_name()
        idx = self.index(clip)
        sid_key = {str(m.get("scene_id")): k for k, m in idx.key_meta.items()}
        vsum, n = None, 0
        for sid in scene_ids:
            key = sid_key.get(str(sid))
            rows_span = idx._key_rows.get(key) if key else None
            if not rows_span:
                continue
            start, end = rows_span
            block = idx.matrix[start:end]
            vsum = block.sum(axis=0) if vsum is None else vsum + block.sum(axis=0)
            n += end - start
        if not n:
            return []
        labels, mat = self._vocab_matrix()
        scores = mat @ self._unit(vsum / n)
        order = np.argsort(-scores)[:top_k]
        return [[labels[i], round(float(scores[i]), 3)] for i in order]

    def performer_detail(
        self, name=None, performer_id=None, count: int = 300, per_scene: int = 6,
    ) -> dict:
        """Everything the detail page needs: her best moments (hero + strip), her
        leaderboard stats, taste distribution, known-for fingerprint, similar."""
        pid, pname, scenes = self._resolve_performer(name, performer_id, None)
        self.performer_stats()  # ensure the leaderboard rows/names are available
        row = getattr(self, "_perf_rows_by_id", {}).get(str(pid))
        if row and row.get("name"):  # prefer the real name over a generic fallback
            pname = row["name"]
        if not scenes:
            return {"performer": pname, "id": pid, "hits": [], "stats": row,
                    "distribution": None, "fingerprint": [], "similar": []}
        model = self._model_name()
        c, _, _ = self._taste_centroid(model)
        hits = self.performer_best(performer_id=pid, name=pname, count=count, per_scene=per_scene)["hits"]

        # taste distribution over ALL her frames (not just the top)
        distribution = None
        if c is not None:
            idx = self.index(model)
            cu = self._unit(c)
            sid_key = {str(m.get("scene_id")): kk for kk, m in idx.key_meta.items()}
            parts = []
            for sid in scenes:
                rs = idx._key_rows.get(sid_key.get(str(sid), ""))
                if rs:
                    parts.append(idx.matrix[rs[0]:rs[1]] @ cu)
            if parts:
                s = np.concatenate(parts)
                counts, edges = np.histogram(s, bins=20)
                distribution = {
                    "median": round(float(np.median(s)), 4),
                    "best": round(float(s.max()), 4),
                    "counts": [int(x) for x in counts],
                    "edges": [round(float(x), 4) for x in edges],
                }

        return {
            "performer": pname, "id": pid,
            "hits": hits,
            "stats": row,
            "distribution": distribution,
            "fingerprint": self.performer_fingerprint(scenes),
            "similar": self.similar_performers(pid),
        }

    def performer_roulette(self, min_moments: int = 20) -> dict:
        """A random performer, weighted toward your taste — 'surprise me'."""
        rows = [r for r in self.performer_stats() if r["moments"] >= min_moments]
        if not rows:
            rows = self.performer_stats()
        if not rows:
            return {"id": None, "name": None}
        w = np.array([max(0.01, (r.get("affinity") or 0.1)) for r in rows], dtype=np.float64)
        r = rows[int(np.random.default_rng().choice(len(rows), p=w / w.sum()))]
        return {"id": r["id"], "name": r["name"]}

    def hall_of_fame(self, top_n: int = 10, per_scene: int = 6, count: int = 300) -> dict:
        """Auto-generate a best-of collection for each of your top performers by
        taste affinity. Returns the collections created."""
        rows = sorted(self.performer_stats(), key=lambda r: (r.get("affinity") or 0), reverse=True)
        created = []
        for r in rows[:top_n]:
            best = self.performer_best(performer_id=r["id"], name=r["name"], count=count, per_scene=per_scene)
            apexes = [{
                "scene_id": h.scene_id, "start": round(h.time, 2), "end": round(h.time + 20, 2),
                "duration": 20, "url": self.stream_url(h.scene_id, start=h.time),
                "score": round(h.score, 4), "title": r["name"],
            } for h in best["hits"] if h.scene_id]
            if apexes:
                saved = self.save_collection(f"{r['name']} — best of", apexes)
                created.append(saved)
        return {"created": created}

    def taste_visual(self, per_mode: int = 6, model: str | None = None, profile: str | None = None) -> dict:
        """Your taste shown as *frames*, not CLIP words: for each taste "mode"
        (a distinct cluster of what you like), the nearest frames in the library.
        `{modes: [{frames:[{key,time,scene_id,score,thumb}]}], sources}`."""
        model = model or self._model_name()
        idx = self.index(model)
        modes = self._taste_modes(model, profile=profile)
        if modes is None or idx.size == 0:
            return {"modes": [], "sources": 0}
        out, seen = [], set()
        for m in modes:
            frames = []
            for h in idx.search(self._unit(m), top_k=per_mode * 3, per_scene=1):
                sig = (h.key, round(h.time, 1))
                if sig in seen:
                    continue
                seen.add(sig)
                frames.append({
                    "key": h.key, "time": round(float(h.time), 2), "scene_id": h.scene_id,
                    "score": round(float(h.score), 3),
                    "thumb": f"/api/frame?key={h.key}&t={h.time:g}",
                })
                if len(frames) >= per_mode:
                    break
            if frames:
                out.append({"frames": frames})
        arr, _ = self._taste_sources(model, profile=profile)
        return {"modes": out, "sources": int(arr.shape[0])}

    def list_labels(
        self, profile: str | None = None, limit: int | None = None, offset: int = 0
    ) -> dict:
        """Your taste labels as editable frames — newest first, paginated so the
        For You editor loads on demand instead of decoding every thumbnail at once.
        `{labels:[{key,time,scene_id,label,thumb}], total, positive, negative,
        has_model}`."""
        profile = profile or self.cfg.markers.tag_name
        labs = sorted(self._label_store().for_profile(profile), key=lambda x: (x.ts or 0.0), reverse=True)
        pos = sum(1 for lab in labs if lab.label == 1)
        page = labs[offset:] if limit is None else labs[offset:offset + max(0, int(limit))]
        items = [{
            "key": lab.key, "time": round(float(lab.time), 2), "scene_id": lab.scene_id,
            "label": int(lab.label), "thumb": f"/api/frame?key={lab.key}&t={lab.time:g}",
        } for lab in page]
        return {"labels": items, "total": len(labs), "positive": pos,
                "negative": len(labs) - pos, "has_model": self.has_taste(profile)}

    def remove_label(self, key: str, time: float, profile: str | None = None) -> dict:
        """Delete one taste label (the editor's ×). Invalidates the taste caches
        so the centroid/visual refresh; the model updates on the next retrain."""
        profile = profile or self.cfg.markers.tag_name
        store = self._label_store()
        removed = store.remove_one(key, float(time), profile)
        if removed:
            store.save()
            self._invalidate_taste_caches()
        pos, neg = store.counts(profile)
        return {"removed": bool(removed), "positive": pos, "negative": neg}

    def taste_words(self, top_k: int = 8, recent: int = 0) -> dict:
        """Your taste centroid described in vocabulary terms (CLIP space) — the
        'what you're into' readout. `recent` limits to your latest N loved
        moments so you can compare lately-vs-all-time (taste drift)."""
        if not self.has_clip_index():
            return {"labels": [], "sources": 0}
        c, n, _ = self._taste_centroid(self._clip_name(), recent=recent)
        if c is None:
            return {"labels": [], "sources": 0}
        labels, mat = self._vocab_matrix()
        scores = mat @ self._unit(c)
        order = np.argsort(-scores)[:top_k]
        return {
            "labels": [[labels[i], round(float(scores[i]), 3)] for i in order],
            "sources": n,
        }

    def taste_metrics(
        self, model: str | None = None, threshold: float | None = None
    ) -> dict:
        """Make the taste score legible: score every embedded frame against your
        taste centroid (one matmul) and describe the distribution.

        The For You % is a cosine to your taste centroid — there is no absolute
        pass/fail, so 'good' is only meaningful *relative to your own library*.
        This returns the percentile bands that make a raw % interpretable (top
        1/5/10/25%), counts of moments **and distinct scenes** in each band (the
        real 'how many meet my taste' number), an optional absolute-threshold
        count, a histogram for a sparkline, and the label/source health counts.
        """
        model = model or self._model_name()
        counts = self.label_counts()
        base = {
            "has_taste": False,
            "model": model,
            "labels": {
                "positive": counts["positive"],
                "negative": counts["negative"],
                "has_model": self.has_taste(),
            },
        }
        # rebuild the centroid so the panel reflects labels/markers added since
        # the feed was last built (the src cache is otherwise only refreshed on a
        # feed rebuild); cheap, and only runs on a metrics load, not per slider tick.
        c, n, sources = self._taste_centroid(model, rebuild=True)
        idx = self.index(model)
        if c is None or idx.size == 0:
            return base

        scores = (idx.matrix @ c).astype(np.float32)
        scene_ids = np.asarray([s if s is not None else "" for s in idx.scene_ids])
        sorted_scores = np.sort(scores)
        # per-scene best moment: "scenes on-taste at t" = scenes whose max ≥ t.
        uniq, inv = np.unique(scene_ids, return_inverse=True)
        scene_max = np.full(uniq.shape[0], -np.inf, dtype=np.float32)
        np.maximum.at(scene_max, inv, scores)

        def moments_ge(t: float) -> int:
            return int(scores.size - np.searchsorted(sorted_scores, t, side="left"))

        def scenes_ge(t: float) -> int:
            return int((scene_max >= t).sum())

        pcts = {q: float(np.percentile(scores, q)) for q in (50, 75, 90, 95, 99)}
        bands = [
            {"label": lbl, "pct": q, "cutoff": round(pcts[q], 4),
             "moments": moments_ge(pcts[q]), "scenes": scenes_ge(pcts[q])}
            for lbl, q in (("Top 1%", 99), ("Top 5%", 95), ("Top 10%", 90), ("Top 25%", 75))
        ]

        # CDF over the score range so the threshold slider counts client-side
        # (no per-tick server call): moments/scenes with score ≥ each step.
        lo, hi = float(sorted_scores[0]), float(sorted_scores[-1])
        steps = np.linspace(lo, hi, 101) if hi > lo else np.array([lo])
        m_ge = (scores.size - np.searchsorted(sorted_scores, steps, side="left")).astype(int)
        s_ge = (scene_max[:, None] >= steps[None, :]).sum(axis=0).astype(int)

        n_apex = sum(1 for s in sources if s.get("kind") == "apex")
        n_thumb = sum(1 for s in sources if s.get("kind") == "thumb")
        hist_counts, hist_edges = np.histogram(scores, bins=24)

        out = {
            **base,
            "has_taste": True,
            "indexed": {"frames": int(idx.size), "scenes": int(uniq.size)},
            "sources": {
                "total": len(sources), "apex": n_apex, "thumbs_up": n_thumb,
                "used_in_centroid": n,
            },
            "distribution": {
                "min": round(lo, 4), "mean": round(float(scores.mean()), 4),
                "p50": round(pcts[50], 4), "p75": round(pcts[75], 4),
                "p90": round(pcts[90], 4), "p95": round(pcts[95], 4),
                "p99": round(pcts[99], 4), "max": round(hi, 4),
            },
            "bands": bands,
            "histogram": {
                "edges": [round(float(e), 4) for e in hist_edges],
                "counts": [int(x) for x in hist_counts],
            },
            "cdf": {
                "thresholds": [round(float(t), 4) for t in steps],
                "moments_ge": [int(x) for x in m_ge],
                "scenes_ge": [int(x) for x in s_ge],
            },
        }
        if threshold is not None:
            t = float(threshold)
            out["threshold"] = {
                "value": round(t, 4),
                "moments": moments_ge(t),
                "scenes": scenes_ge(t),
                "percentile": round(float(np.searchsorted(sorted_scores, t, side="left")) / scores.size * 100, 1),
            }
        return out

    # --- galaxy map: 2D projection of the whole library ----------------------

    def _galaxy_path(self, space: str):
        import os
        from pathlib import Path

        return Path(os.environ.get("PEAKS_GALAXY_DIR", "/config/galaxy")) / f"{space}.json"

    def get_galaxy(self, space: str = "dino") -> dict:
        """The cached 2D map for a space, or {"built": False} if not built yet."""
        import json

        space = "clip" if space == "clip" else "dino"
        p = self._galaxy_path(space)
        if not p.is_file():
            return {"built": False, "space": space}
        try:
            data = json.loads(p.read_text())
            data["built"] = True
            return data
        except (OSError, ValueError):
            return {"built": False, "space": space}

    def build_galaxy(self, job=None, space: str = "dino", method: str = "umap") -> dict:
        """Project every embedded scene to 2D (one point per scene), cluster the
        points, colour them by your taste, and cache it. Runs as a background job
        because UMAP on a few thousand scenes takes a beat."""
        import json
        from time import time as _now

        from ..galaxy import cluster, project, scene_points

        log = (job.log if job else print)
        space = "clip" if space == "clip" else "dino"
        model = self._clip_name() if space == "clip" else self._model_name()
        idx = self.index(model, refresh=True)
        if idx.size == 0:
            raise RuntimeError(f"nothing embedded for {space} yet — run an embed pass first")

        pts = scene_points(idx)
        n = len(pts["keys"])
        if n == 0:
            raise RuntimeError("no scenes to map")
        log(f"galaxy[{space}]: projecting {n} scenes to 2D…")
        coords = project(pts["centroids"], method=method)
        labels = cluster(coords)
        taste = self._galaxy_taste()
        cluster_meta = self._galaxy_cluster_labels(coords, labels, pts["scene_ids"])

        records = []
        for i, key in enumerate(pts["keys"]):
            sid = pts["scene_ids"][i]
            t = round(float(pts["rep_t"][i]), 2)
            records.append({
                "scene_id": sid, "key": key, "t": t,
                "x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4),
                "c": int(labels[i]),
                "taste": round(float(taste.get(str(sid), 0.0)), 3) if sid else 0.0,
                "url": self.stream_url(sid, start=t) if sid else None,
            })
        out = {
            "space": space, "model": model, "scenes": records,
            "clusters": cluster_meta, "has_taste": bool(taste), "built_at": _now(),
        }
        p = self._galaxy_path(space)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out))
        log(f"galaxy[{space}]: {n} scenes, {len(cluster_meta)} clusters cached")
        return {"space": space, "scenes": n, "clusters": len(cluster_meta)}

    def _galaxy_taste(self) -> dict:
        """scene_id -> taste probability, from the DINO taste model applied to
        each scene's (unit) DINO centroid. Batched; {} when no model exists."""
        dino = self._model_name()
        clf = self._taste_model(self.cfg.markers.tag_name, dino)
        if clf is None:
            return {}
        idx = self.index(dino)
        sids, cents = [], []
        for _key, (start, end) in idx._key_rows.items():
            block = idx.matrix[start:end]
            if block.shape[0] == 0:
                continue
            sid = idx.scene_ids[start] if start < len(idx.scene_ids) else None
            if sid is None:
                continue
            sids.append(str(sid))
            cents.append(self._unit(block.mean(axis=0)))
        if not cents:
            return {}
        try:
            probs = clf.predict_proba(np.asarray(cents, dtype=np.float32))
        except Exception:  # noqa: BLE001 — dim mismatch etc.; skip taste colour
            return {}
        return {s: float(p) for s, p in zip(sids, probs)}

    def _galaxy_cluster_labels(self, coords, labels, scene_ids) -> list[dict]:
        """Per cluster: its 2D centre, size, and top CLIP vocab terms (if CLIP is
        embedded). Noise (-1) is skipped."""
        clip_terms = self._cluster_clip_terms(labels, scene_ids) if self.has_clip_index() else {}
        meta = []
        for c in sorted({int(x) for x in labels}):
            if c < 0:
                continue
            mask = labels == c
            meta.append({
                "id": c, "n": int(mask.sum()),
                "cx": round(float(coords[mask, 0].mean()), 4),
                "cy": round(float(coords[mask, 1].mean()), 4),
                "label": clip_terms.get(c, ""),
            })
        return meta

    def _cluster_clip_terms(self, labels, scene_ids, top_k: int = 3) -> dict:
        """Top vocab terms per cluster, from the mean CLIP centroid of its scenes.
        Best-effort — {} if anything's missing (labels are cosmetic)."""
        try:
            vocab, mat = self._vocab_matrix()
            cidx = self.index(self._clip_name())
            sid_cent = {}
            for _key, (start, end) in cidx._key_rows.items():
                block = cidx.matrix[start:end]
                if block.shape[0] == 0:
                    continue
                sid = cidx.scene_ids[start] if start < len(cidx.scene_ids) else None
                if sid is not None:
                    sid_cent[str(sid)] = self._unit(block.mean(axis=0))
            out = {}
            for c in sorted({int(x) for x in labels}):
                if c < 0:
                    continue
                cents = [
                    sid_cent[str(scene_ids[i])]
                    for i in range(len(scene_ids))
                    if int(labels[i]) == c and str(scene_ids[i]) in sid_cent
                ]
                if not cents:
                    continue
                scores = mat @ self._unit(np.mean(cents, axis=0))
                order = np.argsort(-scores)[:top_k]
                out[c] = ", ".join(vocab[j] for j in order)
            return out
        except Exception:  # noqa: BLE001
            return {}

    def has_clip_index(self) -> bool:
        cache = EmbeddingCache(self.cfg.embedding.cache_dir)
        return len(cache.keys(self._clip_name())) > 0

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

    def _vocab_path(self):
        import os
        from pathlib import Path

        return Path(os.environ.get("PEAKS_VOCAB", "/config/vocab.txt"))

    def get_vocab(self) -> dict:
        """Current classification vocabulary as editable text (one term/line)."""
        path = self._vocab_path()
        return {
            "vocab": "\n".join(self._vocab()),
            "count": len(self._vocab()),
            "from_file": path.is_file(),
            "path": str(path),
        }

    def save_vocab(self, text: str) -> dict:
        """Write the vocabulary file and drop the cached matrix so the next
        classification rebuilds against the new terms."""
        path = self._vocab_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [ln.rstrip() for ln in text.splitlines()]
        path.write_text("\n".join(lines) + "\n")
        with self._clip_lock:
            self._vocab_cache = None
        return {"count": len([ln for ln in lines if ln.strip() and not ln.startswith("#")])}

    def _vocab_matrix(self):
        """(labels, matrix) for the vocabulary, CLIP-text-embedded once and
        cached (unit rows, so scoring a frame is one matmul)."""
        with self._clip_lock:
            cached = getattr(self, "_vocab_cache", None)
        if cached is not None:
            return cached
        labels = self._vocab()
        # one batched CLIP-text forward pass for the whole list (was one call
        # per term) so a few-hundred-term vocabulary builds quickly.
        vecs = self._clip_text_batch(labels)
        mat = np.stack([self._unit(v) for v in vecs])
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
        clip = self._clip_name()
        keys = cache.keys(clip)
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
                _, vecs, meta = cache.load(k, clip)
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

    def _key_for_scene(self, scene_id: str, model: str) -> str | None:
        """Reverse-lookup a scene's cache key (the megaboard knows scene_id, not
        the fingerprint key)."""
        idx = self.index(model)
        for k, m in idx.key_meta.items():
            if str(m.get("scene_id")) == str(scene_id):
                return k
        return None

    def classify_frame(
        self, key: str | None = None, time: float = 0.0,
        scene_id: str | None = None, top_k: int = 6,
    ) -> dict:
        """Top vocabulary matches for one frame — what CLIP thinks it is.
        Accepts a cache key (Explore) or a scene_id (megaboard tiles)."""
        clip = self._clip_name()
        if key is None and scene_id is not None:
            key = self._key_for_scene(scene_id, clip)
        if key is None:
            return {"labels": []}
        v = self.index(clip).vector_at(key, time)
        if v is None:
            return {"labels": []}
        labels, mat = self._vocab_matrix()
        scores = mat @ self._unit(v)
        order = np.argsort(-scores)[:top_k]
        return {"labels": [[labels[i], round(float(scores[i]), 3)] for i in order]}

    def _ensure_clip(self):
        """Lazily build (once) the CLIP embedder used for text vectors."""
        with self._clip_lock:
            if self._clip is None:
                from ..embedding import ClipEmbedder

                self._clip = ClipEmbedder(
                    model_name=self.cfg.embedding.clip_model,
                    pretrained=self.cfg.embedding.clip_pretrained,
                    device=self.cfg.embedding.device or None,
                )
        return self._clip

    def _clip_text_vector(self, text: str) -> np.ndarray:
        return self._ensure_clip().embed_text([text])[0]

    def _clip_text_batch(self, labels: list[str]) -> np.ndarray:
        """CLIP-embed many prompts in one forward pass (the vocab-matrix seam)."""
        return np.asarray(self._ensure_clip().embed_text(list(labels)), dtype=np.float32)

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
        if "rating100" in fields:
            r = int(fields.get("rating100") or 0)
            self.set_scene_hidden(scene_id, 0 < r <= 20)  # 1★ = mark for deletion
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
