"""Service.run_fix orchestration, with embed_library stubbed (no ffmpeg)."""

import pytest

pytest.importorskip("fastapi")

from peaks.config import Config  # noqa: E402
from peaks.failures import failure_log_for  # noqa: E402
from peaks.web.service import Service  # noqa: E402


class _FakeEmbedder:
    name = "dinov2"


def _service(tmp_path):
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache" / "embeddings")
    cfg.embedding.model = "dino"
    svc = Service(cfg)
    svc._embedder = lambda: _FakeEmbedder()  # no torch
    return svc, cfg


def test_run_fix_clears_fixed_and_skips_missing(tmp_path, monkeypatch):
    real = tmp_path / "real.mp4"
    real.write_bytes(b"x")
    svc, cfg = _service(tmp_path)
    flog = failure_log_for(cfg)
    flog.record("fp_ok", "1", str(real), error="seek fail")
    flog.record("fp_gone", "2", "/nope/missing.mp4", error="seek fail")

    # run_fix imports embed_library from pipeline; first strategy succeeds
    import peaks.pipeline as pl
    monkeypatch.setattr(pl, "embed_library", lambda *a, **k: {"embedded": 1})

    result = svc.run_fix()
    assert result["fixed"] == 1 and result["failed"] == 1 and result["total"] == 2
    assert flog.keys() == {"fp_gone"}  # the fixed one cleared, missing kept


def test_run_fix_records_exhaustion(tmp_path, monkeypatch):
    real = tmp_path / "real.mp4"
    real.write_bytes(b"x")
    svc, cfg = _service(tmp_path)
    flog = failure_log_for(cfg)
    flog.record("fp_bad", "3", str(real), error="orig")

    import peaks.pipeline as pl
    monkeypatch.setattr(
        pl, "embed_library",
        lambda *a, **k: (k.get("log", lambda *_: None)("failed: still broken") or {"embedded": 0}),
    )

    result = svc.run_fix()
    assert result["fixed"] == 0 and result["failed"] == 1
    entry = {e["key"]: e for e in flog.entries()}["fp_bad"]
    assert entry["mode"] == "fix-exhausted"  # updated, still logged


def test_run_fix_empty_log(tmp_path):
    svc, _ = _service(tmp_path)
    assert svc.run_fix() == {"fixed": 0, "failed": 0, "total": 0}


def test_run_fix_dry_run_touches_nothing(tmp_path):
    svc, cfg = _service(tmp_path)
    flog = failure_log_for(cfg)
    flog.record("fp1", "1", "/data/a.mp4", error="e")
    result = svc.run_fix(dry_run=True)
    assert result["total"] == 1 and result["fixed"] == 0
    assert flog.keys() == {"fp1"}  # dry run leaves the log intact
