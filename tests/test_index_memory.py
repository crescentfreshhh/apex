"""The out-of-memory guard around the ~10 GB search index: deletes mark it stale
instead of dropping it (one rebuild later, when idle); a rebuild never
allocates a second copy past the memory limit; the snapshot copy lets go of
mapped pages as it goes; warm-up and embeds never overlap."""

import numpy as np
import pytest

pytest.importorskip("fastapi")

import peaks.index_snapshot as snap  # noqa: E402
import peaks.search as search_mod  # noqa: E402
from peaks.cache import EmbeddingCache, path_key  # noqa: E402
from peaks.config import Config  # noqa: E402
from peaks.search import SearchIndex  # noqa: E402
from peaks.web import memwatch  # noqa: E402
from peaks.web.service import IndexMemoryError, Service  # noqa: E402

MODEL = "dinov2"
DIM = 16
GB = 1 << 30


def _vecs(rng, n):
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class _Emb:
    name = MODEL


@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.setattr(snap, "write_behind", lambda *a: snap.write(*a))
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.embedding.model = "dino"
    cfg.modeling.dir = str(tmp_path / "models")
    c = EmbeddingCache(cfg.embedding.cache_dir)
    rng = np.random.default_rng(1)
    for i in range(8):
        c.save(path_key(f"/data/{i}.mp4"), MODEL, np.arange(6, dtype=np.float32), _vecs(rng, 6),
               meta={"scene_id": str(i), "path": f"/data/{i}.mp4"})
    s = Service(cfg)
    s._embedder = lambda model=None: _Emb()
    s._model_name = lambda: MODEL
    return s


def _limit(monkeypatch, limit, rss):
    monkeypatch.setattr(memwatch, "soft_limit_bytes", lambda: limit)
    monkeypatch.setattr(memwatch, "rss_bytes", lambda: rss)


def test_deleting_scenes_keeps_the_index_and_marks_it_stale(svc):
    idx = svc.index()
    svc._drop_local([{"scene_id": "3", "path": "/data/3.mp4"}])
    assert svc._index[MODEL] is idx and svc.index() is idx      # still serving, no rebuild
    assert svc.index_state() == "stale"
    svc._drop_local([{"scene_id": "4", "path": "/data/4.mp4"}])  # a queue of deletes: still the one index
    assert svc.index() is idx


def test_stale_rebuild_builds_beside_then_swaps(svc):
    idx = svc.index()
    svc._drop_local([{"scene_id": "3", "path": "/data/3.mp4"}])
    out = svc.rebuild_stale_index()
    assert out["rebuilt"] and out["beside"]
    new = svc.index()
    assert new is not idx and new.size == idx.size - 6 and svc.index_state() == "warm"
    assert svc.rebuild_stale_index() == {"rebuilt": False}      # nothing stale: nothing to do


def test_no_room_keeps_the_old_index(svc, monkeypatch):
    idx = svc.index()
    svc._drop_local([{"scene_id": "3", "path": "/data/3.mp4"}])
    _limit(monkeypatch, 20 * GB, 19.5 * GB)                     # not even room once the old one goes
    built = []
    monkeypatch.setattr(SearchIndex, "build", lambda self, *a, **k: built.append(1) or self)
    assert svc.rebuild_stale_index() == {"rebuilt": False, "reason": "memory"}
    assert built == [] and svc.index() is idx and svc.index_state() == "stale"


def test_no_room_and_no_index_says_so(svc, monkeypatch):
    _limit(monkeypatch, 20 * GB, 19.5 * GB)
    with pytest.raises(IndexMemoryError, match="Not enough memory"):
        svc.index()
    _limit(monkeypatch, 20 * GB, 4 * GB)                        # memory freed up: it loads
    assert svc.index().size == 48


def test_tight_room_swaps_instead_of_doubling(svc, monkeypatch):
    idx = svc.index()
    svc._drop_local([{"scene_id": "3", "path": "/data/3.mp4"}])
    need = svc._index_need_bytes(EmbeddingCache(svc.cfg.embedding.cache_dir), MODEL, [])
    # room for one copy (counting the old one's release), not for two side by side
    _limit(monkeypatch, 20 * GB, 19 * GB - need + idx.matrix.nbytes - 1)
    out = svc.rebuild_stale_index()
    assert out == {"rebuilt": True, "beside": False} and svc.index().size == 42


def test_memory_error_is_a_503(svc, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    app = app_mod.create_app(svc.cfg)

    def boom():
        raise IndexMemoryError("Not enough memory to load the library right now")

    app.add_api_route("/api/_boom", boom)
    r = TestClient(app).get("/api/_boom")
    assert r.status_code == 503 and "Not enough memory" in r.json()["detail"]


def test_snapshot_patch_releases_mapped_pages_and_matches(svc, monkeypatch):
    cache = EmbeddingCache(svc.cfg.embedding.cache_dir)
    SearchIndex(cache, MODEL).build()
    cache.delete(path_key("/data/2.mp4"), MODEL)
    calls = []
    real = search_mod._release_mapped
    monkeypatch.setattr(search_mod, "_RELEASE_EVERY", 1)
    monkeypatch.setattr(search_mod, "_release_mapped", lambda a: calls.append(1) or real(a))
    patched = SearchIndex(cache, MODEL).build()
    assert len(calls) >= 7                                      # after every scene, then once more
    fresh = SearchIndex(cache, MODEL)._build_full(cache.keys(MODEL))
    q = _vecs(np.random.default_rng(3), 1)[0]
    hits = lambda i: sorted((h.key, round(h.time, 3), round(h.score, 5)) for h in i.search(q, top_k=20))  # noqa: E731
    assert hits(patched) == hits(fresh)


def test_snapshot_writer_flushes_in_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(snap, "_FLUSH_EVERY", 7)
    for a in (np.arange(30, dtype=np.float32).reshape(5, 6), np.arange(12, dtype=np.uint16).reshape(3, 4)):
        snap._save_flushed(tmp_path / "x.npy", a)
        b = np.load(tmp_path / "x.npy", mmap_mode="r")
        assert b.dtype == a.dtype and b.shape == a.shape and (b == a).all()


class _Jobs:
    def __init__(self, running=()):
        self.live, self.started = set(running), []

    def running(self, k):
        return object() if k in self.live else None

    def start(self, kind, fn):
        self.started.append((kind, fn))
        return object()


def test_no_embed_during_warm_up(svc):
    import peaks.web.app as app_mod

    jobs = _Jobs(running={"warmup"})
    assert app_mod._scheduled_embeds(svc, jobs, {"last": 0.0}, 3600, "run") is None and jobs.started == []


def test_rewarm_rebuilds_a_stale_index_only_when_idle(svc, monkeypatch):
    import peaks.web.app as app_mod

    monkeypatch.setenv("PEAKS_WARM_ON_START", "1")
    svc.index()
    jobs = _Jobs()
    assert app_mod._rewarm(svc, jobs) is None                   # warm: nothing to do
    svc._drop_local([{"scene_id": "3", "path": "/data/3.mp4"}])
    for busy in ("library", "embed", "ingest"):
        jobs = _Jobs(running={busy})
        assert app_mod._rewarm(svc, jobs) is None and jobs.started == []
    jobs = _Jobs()
    assert app_mod._rewarm(svc, jobs) is not None
    assert jobs.started == [("warmup", svc.rebuild_stale_index)]


def test_health_line_shows_the_index(svc):
    from peaks.web import forensics

    forensics.set_index_probe(svc.index_health)
    try:
        assert "index=cold" in forensics.health_line()
        svc.index()
        assert "index=warm" in forensics.health_line()
    finally:
        forensics.set_index_probe(None)
