"""Configuration loading.

Resolution order (highest priority first):
    1. Environment variables (STASH_URL, STASH_API_KEY, PEAKS_DEVICE,
       PEAKS_HWACCEL, PEAKS_LIBRARY_PATH, PEAKS_SAMPLING_MODE,
       PEAKS_INTERVAL_SECONDS)
    2. A TOML file (default: ./config.toml)
    3. Built-in defaults

The TOML file is gitignored so your API key never gets committed.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.toml")


@dataclass
class StashConfig:
    url: str = "http://192.168.1.2:6969"
    api_key: str = ""
    timeout: int = 30


@dataclass
class SamplingConfig:
    interval_seconds: float = 2.0
    mode: str = "interval"  # "interval" | "keyframes" (throughput fallback)
    hwaccel: str = ""  # "" | "cuda" | "auto" — GPU-assisted decode
    pipeline: str = "raw"  # "raw" (fast: frames straight to GPU) | "jpeg"


@dataclass
class MarkersConfig:
    tag_name: str = "apex"


@dataclass
class EmbeddingConfig:
    model: str = "dino"  # "dino" | "clip" | "fake"
    cache_dir: str = "cache/embeddings"
    device: str = ""  # "" = auto (cuda if available)
    batch_size: int = 64


@dataclass
class ScoringConfig:
    reduce: str = "max"  # "max" | "mean"
    normalize: str = "none"  # "none" | "scene-z" (thresholds become std-devs)
    smooth_window: int = 3
    high: float = 0.45  # tune after Tier-1 validation
    low: float = 0.35
    min_duration: float = 3.0
    merge_gap: float = 2.0
    max_duration: float = 30.0
    pad: float = 0.5
    references_dir: str = "references"


@dataclass
class ModelingConfig:
    dir: str = "models"  # where trained classifiers are saved (gitignored)
    classifier: str = "logreg"  # "logreg" | "mlp"
    labels_path: str = "labels.json"


@dataclass
class LibraryConfig:
    # Only work on scenes whose file path starts with this (empty = whole
    # library). Point it at a folder to exclude everything else — e.g. skip VR.
    path: str = ""


@dataclass
class Config:
    stash: StashConfig = field(default_factory=StashConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    markers: MarkersConfig = field(default_factory=MarkersConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    library: LibraryConfig = field(default_factory=LibraryConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        """Load config from a TOML file (if present) then apply env overrides."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        raw: dict = {}
        if path.exists():
            with path.open("rb") as fh:
                raw = tomllib.load(fh)

        stash_raw = raw.get("stash", {})
        sampling_raw = raw.get("sampling", {})
        markers_raw = raw.get("markers", {})
        embedding_raw = raw.get("embedding", {})
        scoring_raw = raw.get("scoring", {})
        modeling_raw = raw.get("modeling", {})
        library_raw = raw.get("library", {})

        stash = StashConfig(
            url=os.environ.get("STASH_URL", stash_raw.get("url", StashConfig.url)),
            api_key=os.environ.get(
                "STASH_API_KEY", stash_raw.get("api_key", StashConfig.api_key)
            ),
            timeout=int(stash_raw.get("timeout", StashConfig.timeout)),
        )
        sampling = SamplingConfig(
            interval_seconds=float(
                os.environ.get(
                    "PEAKS_INTERVAL_SECONDS",
                    sampling_raw.get(
                        "interval_seconds", SamplingConfig.interval_seconds
                    ),
                )
            ),
            mode=os.environ.get(
                "PEAKS_SAMPLING_MODE", sampling_raw.get("mode", SamplingConfig.mode)
            ),
            hwaccel=os.environ.get(
                "PEAKS_HWACCEL", sampling_raw.get("hwaccel", SamplingConfig.hwaccel)
            ),
            pipeline=os.environ.get(
                "PEAKS_PIPELINE",
                sampling_raw.get("pipeline", SamplingConfig.pipeline),
            ),
        )
        markers = MarkersConfig(
            tag_name=markers_raw.get("tag_name", MarkersConfig.tag_name)
        )
        embedding = EmbeddingConfig(
            model=embedding_raw.get("model", EmbeddingConfig.model),
            cache_dir=embedding_raw.get("cache_dir", EmbeddingConfig.cache_dir),
            device=os.environ.get(
                "PEAKS_DEVICE", embedding_raw.get("device", EmbeddingConfig.device)
            ),
            batch_size=int(embedding_raw.get("batch_size", EmbeddingConfig.batch_size)),
        )
        scoring = ScoringConfig(
            reduce=scoring_raw.get("reduce", ScoringConfig.reduce),
            normalize=scoring_raw.get("normalize", ScoringConfig.normalize),
            smooth_window=int(
                scoring_raw.get("smooth_window", ScoringConfig.smooth_window)
            ),
            high=float(scoring_raw.get("high", ScoringConfig.high)),
            low=float(scoring_raw.get("low", ScoringConfig.low)),
            min_duration=float(
                scoring_raw.get("min_duration", ScoringConfig.min_duration)
            ),
            merge_gap=float(scoring_raw.get("merge_gap", ScoringConfig.merge_gap)),
            max_duration=float(
                scoring_raw.get("max_duration", ScoringConfig.max_duration)
            ),
            pad=float(scoring_raw.get("pad", ScoringConfig.pad)),
            references_dir=scoring_raw.get(
                "references_dir", ScoringConfig.references_dir
            ),
        )
        modeling = ModelingConfig(
            dir=modeling_raw.get("dir", ModelingConfig.dir),
            classifier=modeling_raw.get("classifier", ModelingConfig.classifier),
            labels_path=modeling_raw.get("labels_path", ModelingConfig.labels_path),
        )
        library = LibraryConfig(
            path=os.environ.get(
                "PEAKS_LIBRARY_PATH", library_raw.get("path", LibraryConfig.path)
            ),
        )
        return cls(
            stash=stash,
            sampling=sampling,
            markers=markers,
            embedding=embedding,
            scoring=scoring,
            modeling=modeling,
            library=library,
        )
