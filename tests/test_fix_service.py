"""Service.run_fix orchestration, with embed_library stubbed (no ffmpeg)."""

import pytest

pytest.importorskip("fastapi")

from peaks.config import Config  # noqa: E402
from peaks.failures import failure_log_for  # noqa: E402
from peaks.web.service import Service  # noqa: E402


class _FakeEmbedder:
    name = "dinov2"


class _FakeClient:
    """Stands in for the Stash client so run_fix's deleted-scene check is offline.
    `gone` is the set of scene ids to report as no longer in the library."""

    def __init__(self, gone=frozenset()):
        self.gone = set(gone)

    def existing_scene_ids(self, ids):
        return {str(i) for i in ids if str(i) not in self.gone}


def _service(tmp_path, gone=frozenset()):
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache" / "embeddings")
    cfg.embedding.model = "dino"
    svc = Service(cfg)
    svc._embedder = lambda model=None: _FakeEmbedder()  # no torch
    svc.client = lambda: _FakeClient(gone)  # offline Stash existence check
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


def test_run_embed_applies_overrides(tmp_path, monkeypatch):
    import peaks.pipeline as pl
    import peaks.sampling as smp
    import peaks.web.service as svc_mod

    svc, _ = _service(tmp_path)
    cap = {}

    def fake_embedder(model=None):
        cap["model"] = model
        return _FakeEmbedder()

    class FakeSampler:
        def __init__(self, **kw):
            cap["sampler"] = kw
            self.mode = kw.get("mode")
            self.interval = kw.get("interval_seconds")
            self.hwaccel = kw.get("hwaccel")

    def fake_lib(*a, **k):
        cap["workers"] = k.get("workers")
        return {"embedded": 0}

    svc._embedder = fake_embedder  # instance override that captures the model
    monkeypatch.setattr(svc_mod.Service, "scenes", lambda self, limit=0: [])
    monkeypatch.setattr(smp, "FrameSampler", FakeSampler)
    monkeypatch.setattr(pl, "embed_library", fake_lib)

    svc.run_embed(
        model="clip", mode="interval", interval=4.0,
        hwaccel="cuda", workers=2, scene_timeout=600.0,
    )
    assert cap["model"] == "clip"
    assert cap["sampler"]["mode"] == "interval"
    assert cap["sampler"]["interval_seconds"] == 4.0
    assert cap["sampler"]["scene_timeout"] == 600.0
    assert cap["workers"] == 2


def test_run_embed_defaults_when_no_overrides(tmp_path, monkeypatch):
    import peaks.pipeline as pl
    import peaks.sampling as smp
    import peaks.web.service as svc_mod

    svc, cfg = _service(tmp_path)
    cap = {}

    class FakeSampler:
        def __init__(self, **kw):
            cap["sampler"] = kw
            self.mode = kw.get("mode")
            self.interval = kw.get("interval_seconds")
            self.hwaccel = kw.get("hwaccel")

    monkeypatch.setattr(svc_mod.Service, "scenes", lambda self, limit=0: [])
    monkeypatch.setattr(smp, "FrameSampler", FakeSampler)
    monkeypatch.setattr(pl, "embed_library", lambda *a, **k: {"embedded": 0})

    svc.run_embed()
    # falls back to configured sampling defaults
    assert cap["sampler"]["mode"] == cfg.sampling.mode
    assert cap["sampler"]["interval_seconds"] == cfg.sampling.interval_seconds


def test_run_embed_multi_runs_each_model(tmp_path, monkeypatch):
    import peaks.web.service as svc_mod

    svc, _ = _service(tmp_path)
    calls = []

    def fake_embed(self, job=None, limit=0, **kw):
        calls.append(kw.get("model"))
        return {"embedded": 1, "skipped": 0, "failed": 0, "frames": 10}

    monkeypatch.setattr(svc_mod.Service, "run_embed", fake_embed)
    res = svc.run_embed_multi(models=["dino", "clip"])
    assert calls == ["dino", "clip"]  # back-to-back, in order
    assert res["embedded"] == 2 and res["frames"] == 20
    assert set(res["passes"]) == {"dino", "clip"}


def test_run_fix_empty_log(tmp_path):
    svc, _ = _service(tmp_path)
    assert svc.run_fix() == {"fixed": 0, "failed": 0, "pruned": 0, "total": 0}


def test_run_fix_dry_run_touches_nothing(tmp_path):
    svc, cfg = _service(tmp_path)
    flog = failure_log_for(cfg)
    flog.record("fp1", "1", "/data/a.mp4", error="e")
    result = svc.run_fix(dry_run=True)
    assert result["total"] == 1 and result["fixed"] == 0
    assert flog.keys() == {"fp1"}  # dry run leaves the log intact


def test_run_fix_prunes_scenes_deleted_from_stash(tmp_path, monkeypatch):
    real = tmp_path / "real.mp4"
    real.write_bytes(b"x")
    # scene "2" no longer exists in Stash → should be pruned, not retried
    svc, cfg = _service(tmp_path, gone={"2"})
    flog = failure_log_for(cfg)
    flog.record("fp_ok", "1", str(real), error="seek fail")
    flog.record("fp_deleted", "2", str(real), error="seek fail")

    import peaks.pipeline as pl
    monkeypatch.setattr(pl, "embed_library", lambda *a, **k: {"embedded": 1})

    result = svc.run_fix()
    assert result["fixed"] == 1 and result["pruned"] == 1 and result["failed"] == 0
    assert flog.keys() == set()  # fixed one cleared, deleted one pruned


def test_run_fix_keeps_all_when_stash_unreachable(tmp_path, monkeypatch):
    # if the Stash existence check fails, prune nothing (a scene may only be
    # transiently missing) — retry as before.
    svc, cfg = _service(tmp_path)

    def _boom():
        raise RuntimeError("stash down")

    svc.client = _boom
    flog = failure_log_for(cfg)
    flog.record("fp_bad", "9", "/nope/missing.mp4", error="e")

    result = svc.run_fix()
    assert result["pruned"] == 0 and result["failed"] == 1
    assert flog.keys() == {"fp_bad"}  # nothing pruned


# --- Phase 6: uncached-only embed + auto-prune deleted-from-Stash failures ----

class _FakeScene:
    def __init__(self, sid):
        self.id = sid
        self.fingerprint = sid          # scene_key() prefers this
        self.path = f"/v/{sid}.mp4"
        self.markers = []


def test_run_embed_only_embeds_uncached(tmp_path, monkeypatch):
    import peaks.pipeline as pl
    import peaks.sampling as smp
    import peaks.web.service as svc_mod

    svc, _ = _service(tmp_path)
    scenes = [_FakeScene("1"), _FakeScene("2"), _FakeScene("3")]
    monkeypatch.setattr(svc_mod.Service, "scenes", lambda self, limit=0: scenes)

    class FakeCache:
        def __init__(self, *a, **k):
            pass
        def has(self, key, model, interval=None):
            return key in {"1", "2"}       # 1 & 2 already embedded
        def keys(self, model):
            return ["1", "2"]

    monkeypatch.setattr(svc_mod, "EmbeddingCache", FakeCache)

    class FakeSampler:
        def __init__(self, **kw):
            self.mode = kw.get("mode")
            self.interval = kw.get("interval_seconds")
            self.hwaccel = kw.get("hwaccel")

    monkeypatch.setattr(smp, "FrameSampler", FakeSampler)

    cap = {}

    def fake_lib(passed, *a, **k):
        cap["ids"] = [s.id for s in passed]
        cap["total"] = k.get("total")
        return {"embedded": 0}

    monkeypatch.setattr(pl, "embed_library", fake_lib)

    svc.run_embed()
    assert cap["ids"] == ["3"]     # only the uncached scene is worked
    assert cap["total"] == 1       # progress denominator = real pending, not the scan


def test_prune_dead_failures(tmp_path):
    svc, cfg = _service(tmp_path, gone={"2"})
    flog = failure_log_for(cfg)
    flog.record("k1", "1", "/v/1.mp4", error="worker exit 1")
    flog.record("k2", "2", "/v/2.mp4", error="worker exit 1")   # deleted from Stash
    assert svc.prune_dead_failures() == 1
    assert flog.keys() == {"k1"}   # the deleted one is gone, the live one stays


def test_prune_dead_failures_skips_on_outage(tmp_path):
    svc, cfg = _service(tmp_path)

    def _boom():
        raise RuntimeError("stash down")

    svc.client = _boom
    flog = failure_log_for(cfg)
    flog.record("k9", "9", "/v/9.mp4", error="worker exit 1")
    assert svc.prune_dead_failures() == 0   # never prune when Stash can't be reached
    assert flog.keys() == {"k9"}


def test_run_embed_prunes_dead_failures_at_end(tmp_path, monkeypatch):
    import peaks.pipeline as pl
    import peaks.sampling as smp
    import peaks.web.service as svc_mod

    svc, cfg = _service(tmp_path, gone={"5"})
    flog = failure_log_for(cfg)
    flog.record("k5", "5", "/v/5.mp4", error="worker exit 1")   # deleted scene
    monkeypatch.setattr(svc_mod.Service, "scenes", lambda self, limit=0: [])

    class FakeCache:
        def __init__(self, *a, **k):
            pass
        def has(self, key, model, interval=None):
            return False
        def keys(self, model):
            return []

    monkeypatch.setattr(svc_mod, "EmbeddingCache", FakeCache)

    class FakeSampler:
        def __init__(self, **kw):
            self.mode = kw.get("mode")
            self.interval = kw.get("interval_seconds")
            self.hwaccel = kw.get("hwaccel")

    monkeypatch.setattr(smp, "FrameSampler", FakeSampler)
    monkeypatch.setattr(pl, "embed_library", lambda *a, **k: {"embedded": 0})

    svc.run_embed()
    assert flog.keys() == set()    # deleted-scene failure auto-pruned at end of embed
