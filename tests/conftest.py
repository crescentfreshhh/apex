"""Shared test isolation.

The web service persists GUI settings (models, sampling, clip length, export
quality) to PEAKS_SETTINGS (default /config/settings.json) and writes reels to
PEAKS_EXPORT_DIR (default /config/exports). A test that saves a setting without
pointing those at a temp dir would leak into the real /config and change the
behaviour of every later test that builds a Service — so pin both to a
per-test temp dir for every test.
"""

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("PEAKS_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("PEAKS_WARM_ON_START", "0")   # no background start-up load in tests


@pytest.fixture(autouse=True)
def _isolate_measure_state(tmp_path, monkeypatch):
    """Ratings and grades bump the taste-measure counter under the configured
    models dir; tests that build a default Config (relative "models") would
    otherwise write it into the repo."""
    try:
        import peaks.web.service as svc_mod
    except ImportError:  # web extras not installed
        return
    monkeypatch.setattr(svc_mod.Service, "_measure_state_path",
                        lambda self: tmp_path / "taste-measure" / "measure_state.json")
    # …and the on-disk scene list, when a test builds a default Config
    real = svc_mod.Service._catalogue_disk_path
    monkeypatch.setattr(svc_mod.Service, "_catalogue_disk_path",
                        lambda self: real(self) if Path(self.cfg.modeling.dir).is_absolute()
                        else tmp_path / "catalogue_cache.json.gz")
