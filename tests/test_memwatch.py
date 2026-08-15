"""Memory self-policing: RSS/limit probes and the Service.shed_memory action."""

import numpy as np
import pytest

pytest.importorskip("fastapi")

from peaks.config import Config  # noqa: E402
import peaks.web.service as svc_mod  # noqa: E402
from peaks.web import memwatch  # noqa: E402


def test_rss_and_trim_are_safe():
    assert memwatch.rss_bytes() > 0          # Linux CI: resident memory readable
    memwatch.malloc_trim()                   # must never raise


def test_soft_limit_from_env(monkeypatch):
    monkeypatch.setenv("PEAKS_MEM_LIMIT_MB", "1234")
    assert memwatch.soft_limit_bytes() == 1234 * 1024 * 1024


def test_soft_limit_none_when_no_limit(monkeypatch):
    monkeypatch.delenv("PEAKS_MEM_LIMIT_MB", raising=False)
    monkeypatch.setattr(memwatch, "cgroup_limit_bytes", lambda: None)
    assert memwatch.soft_limit_bytes() is None


def test_shed_memory_clears_caches_and_idle_indexes(tmp_path, monkeypatch):
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    svc = svc_mod.Service(cfg)
    monkeypatch.setattr(svc, "_model_name", lambda: "dinov2")

    # derived caches + a primary (dinov2) and an idle (clip) index
    svc._board_score_cache["dinov2"] = (np.zeros(3, dtype="float32"), "modes")
    svc._taste_src_cache["dinov2"] = "x"
    svc._board_universe_cache[("dinov2", 4)] = []
    svc._perf_stats_cache = [{"id": "1"}]
    svc._index["dinov2"] = object()
    svc._index["clip"] = object()

    r = svc.shed_memory(drop_indexes=True)

    assert svc._board_score_cache == {} and svc._taste_src_cache == {}
    assert svc._board_universe_cache == {}
    assert svc._perf_stats_cache is None
    # the board model's index is kept; the idle one is dropped
    assert "dinov2" in svc._index and "clip" not in svc._index
    assert r["dropped_indexes"] == ["clip"]

    # drop_indexes=False keeps all indexes, still clears caches
    svc._board_score_cache["dinov2"] = (np.zeros(1, dtype="float32"), "modes")
    r2 = svc.shed_memory(drop_indexes=False)
    assert svc._board_score_cache == {} and r2["dropped_indexes"] == []
    assert "dinov2" in svc._index
