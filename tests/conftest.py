"""Shared test isolation.

The web service persists GUI settings (models, sampling, clip length, export
quality) to PEAKS_SETTINGS (default /config/settings.json) and writes reels to
PEAKS_EXPORT_DIR (default /config/exports). A test that saves a setting without
pointing those at a temp dir would leak into the real /config and change the
behaviour of every later test that builds a Service — so pin both to a
per-test temp dir for every test.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("PEAKS_EXPORT_DIR", str(tmp_path / "exports"))
