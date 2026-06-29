"""Configuration loading.

Resolution order (highest priority first):
    1. Environment variables (STASH_URL, STASH_API_KEY)
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


@dataclass
class MarkersConfig:
    tag_name: str = "apex"


@dataclass
class Config:
    stash: StashConfig = field(default_factory=StashConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    markers: MarkersConfig = field(default_factory=MarkersConfig)

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

        stash = StashConfig(
            url=os.environ.get("STASH_URL", stash_raw.get("url", StashConfig.url)),
            api_key=os.environ.get(
                "STASH_API_KEY", stash_raw.get("api_key", StashConfig.api_key)
            ),
            timeout=int(stash_raw.get("timeout", StashConfig.timeout)),
        )
        sampling = SamplingConfig(
            interval_seconds=float(
                sampling_raw.get("interval_seconds", SamplingConfig.interval_seconds)
            )
        )
        markers = MarkersConfig(
            tag_name=markers_raw.get("tag_name", MarkersConfig.tag_name)
        )
        return cls(stash=stash, sampling=sampling, markers=markers)
