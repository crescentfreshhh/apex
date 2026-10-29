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


def test_a_soft_purge_keeps_what_pages_read(tmp_path, monkeypatch):
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    svc = svc_mod.Service(cfg)
    monkeypatch.setattr(svc, "_model_name", lambda: "dinov2")
    svc._meta = {"1": {"title": "x"}}
    svc._board_universe_cache[("dinov2", 4)] = []
    svc._snapshots = {"lab": (1, {})}
    svc._vocab_cache = "v"
    svc._index["dinov2"], svc._index["clip"] = object(), object()
    r = svc.shed_memory(drop_indexes=True, keep_hot=True)
    assert svc._meta and svc._board_universe_cache and svc._snapshots       # the page path stays warm
    assert svc._vocab_cache is None and r["dropped_indexes"] == ["clip"]
    svc.shed_memory(drop_indexes=True)                                     # the hard limit: everything
    assert not svc._meta and not svc._board_universe_cache and "_snapshots" not in svc.__dict__


def test_the_policy_never_loops_and_still_guards_the_hard_limit():
    now = [0.0]
    G = 1024 ** 3
    p = memwatch.PurgePolicy(10 * G, clock=lambda: now[0])
    assert p.decide(8 * G) is None                         # under 90%: nothing
    assert p.decide(9.5 * G) == "soft"
    p.record("soft", int(9.5 * G), int(9.0 * G))           # freed 512 MB
    now[0] += 20
    assert p.decide(9.5 * G) is None                       # cooldown: not every 20 s
    now[0] += 300
    assert p.decide(9.5 * G) == "soft"
    p.record("soft", int(9.2 * G), int(9.17 * G))          # freed ~30 MB: futile
    now[0] += 400
    assert p.decide(9.2 * G) is None                       # …so not again until memory grows
    assert p.decide(9.17 * G + 600 * memwatch.MB) == "soft"
    assert p.decide(10.2 * G) == "hard"                    # the OOM guard
    p.record("hard", int(10.2 * G), int(10.15 * G))        # freed ~50 MB
    now[0] += 120
    assert p.decide(10.2 * G) is None                      # futile + little growth: even hard waits
    p.last_freed = 2 * G
    assert p.decide(10.2 * G) == "hard"
    p.record("hard", int(10.2 * G), int(8 * G), ["x"])
    now[0] += 30
    assert p.decide(10.5 * G) is None                      # a minute between hard purges
    now[0] += 31
    assert p.decide(10.5 * G) == "hard"
    assert memwatch.PURGES[-1]["level"] == "hard" and memwatch.PURGES[-1]["freed_mb"] > 2200


def test_one_watchdog_check_purges_softly_and_logs_it(tmp_path, monkeypatch):
    from peaks.web.app import watch_once

    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    svc = svc_mod.Service(cfg)
    monkeypatch.setattr(svc, "_model_name", lambda: "dinov2")
    svc._meta = {"1": {}}
    rss = iter([95, 95, 80])
    monkeypatch.setattr(memwatch, "rss_bytes", lambda: next(rss) * memwatch.MB)
    e = watch_once(svc, memwatch.PurgePolicy(100 * memwatch.MB))
    assert e["level"] == "soft" and e["freed_mb"] == 15 and svc._meta
