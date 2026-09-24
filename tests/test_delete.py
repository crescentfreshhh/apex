"""Deleting rejects: files go only after confirmation, each scene is re-checked
as still 1★ right before deletion, every file is logged, and the reject's
features are remembered first so the triage model keeps a reject class."""

import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from peaks.reject_memory import RejectMemory  # noqa: E402
from test_catalogue import _triage_svc  # noqa: E402


def _api(svc, monkeypatch):
    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    return TestClient(app_mod.create_app(svc.cfg))


def _wait(client, job):
    for _ in range(300):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """The triage library plus 12 embedded rejects (look like upscales, low bitrate)."""
    from peaks.cache import EmbeddingCache

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    rng = np.random.default_rng(1)
    look = rng.standard_normal(24)
    look /= np.linalg.norm(look)
    cache = EmbeddingCache(svc.cfg.embedding.cache_dir)
    for i in range(12):
        sid = f"r{i}"
        stash.s[sid] = {"rating100": 20, "o_counter": 0, "width": 1280, "height": 720,
                        "bit_rate": 2e6, "frame_rate": 30, "video_codec": "h264",
                        "date": "2019-01-01", "size": 500_000_000, "tag_ids": [],
                        "organized": False, "fingerprint": f"fpr{i}"}
        v = np.stack([look + 0.3 * rng.standard_normal(24) for _ in range(10)]).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        cache.save(f"fpr{i}", "dinov2", np.arange(10, dtype=np.float32) * 30, v,
                   meta={"scene_id": sid})
    return svc, stash


def test_preview_lists_rejects_with_their_size(lib, monkeypatch):
    svc, stash = lib
    d = _api(svc, monkeypatch).get("/api/catalogue/delete-preview").json()
    assert d["count"] == 12 and d["bytes"] == 12 * 500_000_000 and d["capable"]
    assert stash.calls == [] or all(c[0] != "destroy" for c in stash.calls)
    sel = _api(svc, monkeypatch).get("/api/catalogue/delete-preview", params={"ids": "r1,r2,0"}).json()
    assert sel["count"] == 2                                  # scene 0 isn't a reject


def test_delete_needs_confirm_and_capability(lib, monkeypatch):
    svc, stash = lib
    client = _api(svc, monkeypatch)
    assert client.post("/api/catalogue/delete", json={}).status_code == 409
    stash.caps["scenesDestroy"] = False
    svc._caps_cache = None
    assert client.post("/api/catalogue/delete", json={"confirm": True}).status_code == 501
    assert "r0" in stash.s


def test_delete_rechecks_rating_remembers_then_destroys(lib, monkeypatch):
    svc, stash = lib
    client = _api(svc, monkeypatch)
    client.get("/api/catalogue")                              # listing sees 12 rejects
    stash.s["r3"]["rating100"] = 100                          # rescued in Stash since then
    stash.s["r3"]["o_counter"] = 16
    mem = svc.reject_memory()
    seen_at_destroy = []
    real = stash.destroy_scenes
    def destroy(ids, **kw):
        seen_at_destroy.append(set(RejectMemory(svc.cfg.modeling.dir, svc._model_name()).fingerprints()))
        return real(ids, **kw)
    monkeypatch.setattr(stash, "destroy_scenes", destroy)

    ids = [f"r{i}" for i in range(12)]                       # selected before r3 was rescued
    j = _wait(client, client.post("/api/catalogue/delete", json={"scene_ids": ids, "confirm": True}).json())
    r = j["result"]
    assert r["deleted"] == 11 and r["freed_bytes"] == 11 * 500_000_000, j
    assert [x["scene_id"] for x in r["refused"]] == ["r3"]
    assert "r3" in stash.s and "r0" not in stash.s
    # features were stored BEFORE the files went
    assert {f"fpr{i}" for i in range(12) if i != 3} <= set().union(*seen_at_destroy)
    assert len(mem) >= 11
    # every file logged; the cache entries and hidden set are cleaned up
    logged = [e for e in svc.history() if e["action"] == "delete"]
    assert len(logged) == 11 and all(e["path"] and e["fingerprint"] for e in logged)
    from peaks.cache import EmbeddingCache
    assert not EmbeddingCache(svc.cfg.embedding.cache_dir).has("fpr0", "dinov2")
    assert "r0" not in svc.hidden_scene_ids()
    assert client.get("/api/catalogue").json()["counts"]["rejected"] == 0


def test_training_keeps_the_reject_class_after_deletion(lib, monkeypatch):
    pytest.importorskip("sklearn")
    svc, stash = lib
    client = _api(svc, monkeypatch)
    _wait(client, client.post("/api/catalogue/delete", json={"confirm": True}).json())
    assert not [s for s, v in stash.s.items() if v["rating100"] == 20]      # all gone
    rep = svc.train_tier_model()
    assert rep["trained"] and rep["remembered_rejects"] == 12
    assert "reject" in rep["classes"] and rep["counts"]["reject"] == 12

    # a different embedding model starts its own memory — nothing mixed in
    other = RejectMemory(svc.cfg.modeling.dir, "some-other-model")
    assert len(other) == 0


def test_memory_refuses_mismatched_feature_sizes(tmp_path):
    mem = RejectMemory(tmp_path, "m")
    assert mem.add({"a": (np.ones(8), np.ones(3))}) == 1
    assert mem.add({"a": (np.ones(8), np.ones(3))}) == 0            # already remembered
    with pytest.raises(ValueError):
        mem.add({"b": (np.ones(9), np.ones(3))})
    fps, v, q = mem.rows(exclude={"a"})
    assert fps == [] and v is None
