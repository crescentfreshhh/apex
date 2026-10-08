"""The embeddings index (~a minute to load) stays warm: an embed or sync that
changed nothing keeps it; anything that drops it gets it reloaded in the
background; pages can ask whether it's loading."""

import types

import pytest

pytest.importorskip("fastapi")

from peaks.config import Config  # noqa: E402
from peaks.web.service import Service  # noqa: E402

SENTINEL = types.SimpleNamespace(size=10, key_meta={"k1": {"path": "/old/a.mp4", "scene_id": "1"}})


class _Emb:
    name = "dinov2"


def _svc(tmp_path):
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.embedding.model = "dino"
    cfg.modeling.dir = str(tmp_path / "models")
    svc = Service(cfg)
    svc._embedder = lambda model=None: _Emb()
    return svc


@pytest.fixture
def fake_embed_env(monkeypatch):
    import peaks.sampling as smp
    import peaks.web.service as svc_mod

    class FakeCache:
        def __init__(self, *a, **k):
            pass

        def has(self, key, model, interval=None):
            return False

        def keys(self, model):
            return []

        def models(self):
            return ["dinov2"]

    class FakeSampler:
        def __init__(self, **kw):
            self.mode, self.interval, self.hwaccel = kw.get("mode"), kw.get("interval_seconds"), kw.get("hwaccel")

    monkeypatch.setattr(svc_mod, "EmbeddingCache", FakeCache)
    monkeypatch.setattr(smp, "FrameSampler", FakeSampler)
    monkeypatch.setattr(svc_mod.Service, "scenes", lambda self, limit=0: [])
    monkeypatch.setattr(svc_mod.Service, "prune_dead_failures", lambda self: 0, raising=False)


@pytest.mark.parametrize("embedded, kept", [(0, True), (3, False)])
def test_embed_keeps_the_index_unless_it_added_scenes(tmp_path, monkeypatch, fake_embed_env, embedded, kept):
    import peaks.pipeline as pl

    svc = _svc(tmp_path)
    svc._index["dinov2"] = SENTINEL
    monkeypatch.setattr(pl, "embed_library", lambda *a, **k: {"embedded": embedded})
    svc.run_embed()
    assert ("dinov2" in svc._index) is kept


def _sync(svc, monkeypatch, result, moved=None):
    import peaks.pipeline as pl

    def fake_sync(scenes, cache, model, prune=False, log=print, moved_out=None):
        if moved_out is not None and moved:
            moved_out.update(moved)
        return dict(result)

    monkeypatch.setattr(pl, "sync_cache", fake_sync)
    svc.client = lambda: types.SimpleNamespace(iter_scenes=lambda: iter([types.SimpleNamespace(id="1")]))
    return svc.run_sync(prune=True)


def test_sync_with_nothing_changed_keeps_the_index(tmp_path, monkeypatch, fake_embed_env):
    svc = _svc(tmp_path)
    svc._index["dinov2"] = SENTINEL
    _sync(svc, monkeypatch, {"cached": 5, "moved": 0, "orphaned": 0, "pruned": 0})
    assert svc._index.get("dinov2") is SENTINEL


def test_sync_path_moves_are_patched_in_place(tmp_path, monkeypatch, fake_embed_env):
    svc = _svc(tmp_path)
    idx = types.SimpleNamespace(size=10, key_meta={"k1": {"path": "/old/a.mp4", "scene_id": "1"}})
    svc._index["dinov2"] = idx
    _sync(svc, monkeypatch, {"cached": 5, "moved": 1, "orphaned": 0, "pruned": 0},
          moved={"k1": ("/data/Légendaire/a.mp4", "1", False)})
    assert svc._index.get("dinov2") is idx and idx.key_meta["k1"]["path"] == "/data/Légendaire/a.mp4"


def test_sync_prunes_or_id_changes_rebuild(tmp_path, monkeypatch, fake_embed_env):
    svc = _svc(tmp_path)
    svc._index["dinov2"] = SENTINEL
    _sync(svc, monkeypatch, {"cached": 5, "moved": 0, "orphaned": 1, "pruned": 1})
    assert "dinov2" not in svc._index
    svc._index["dinov2"] = SENTINEL
    _sync(svc, monkeypatch, {"cached": 5, "moved": 1, "orphaned": 0, "pruned": 0},
          moved={"k1": ("/x.mp4", "9", True)})
    assert "dinov2" not in svc._index


class _Jobs:
    def __init__(self, running=()):
        self.live, self.started = set(running), []

    def running(self, k):
        return object() if k in self.live else None

    def start(self, kind, fn):
        self.started.append(kind)
        return object()


def test_rewarm_reloads_a_cold_index_when_idle(tmp_path, monkeypatch):
    import peaks.web.app as app_mod

    monkeypatch.setenv("PEAKS_WARM_ON_START", "1")
    svc = _svc(tmp_path)
    for busy in ("ingest", "embed", "warmup", "library"):
        jobs = _Jobs(running={busy})
        assert app_mod._rewarm(svc, jobs) is None and jobs.started == []
    jobs = _Jobs()
    assert app_mod._rewarm(svc, jobs) is not None and jobs.started == ["warmup"]
    svc._index[svc._model_name()] = SENTINEL                  # warm already: nothing to do
    jobs = _Jobs()
    assert app_mod._rewarm(svc, jobs) is None and jobs.started == []
    del svc._index[svc._model_name()]
    monkeypatch.setenv("PEAKS_WARM_ON_START", "0")           # switched off: respected
    assert app_mod._rewarm(svc, _Jobs()) is None


def test_warm_endpoint_reports_loading(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc = _svc(tmp_path)
    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(app_mod.create_app(svc.cfg))
    assert api.get("/api/warm").json()["loading"] is True
    svc._index[svc._model_name()] = SENTINEL
    st = api.get("/api/warm").json()
    assert st["loading"] is False and st["loaded"] is True
