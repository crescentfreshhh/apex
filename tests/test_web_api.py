"""Web API tests via FastAPI TestClient — a fake embedding cache, no torch,
no Stash, no ffmpeg (frame decoding is monkeypatched)."""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from peaks.cache import EmbeddingCache  # noqa: E402
from peaks.config import Config  # noqa: E402
from peaks.web.app import create_app  # noqa: E402
from peaks.web.jobs import Job, JobManager  # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.embedding.cache_dir = str(tmp_path / "cache")
    c.embedding.model = "dino"  # canonical -> dinov2
    cache = EmbeddingCache(c.embedding.cache_dir)
    cache.save(
        "k1", "dinov2",
        np.array([0.0, 8.0], dtype=np.float32),
        np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])]),
        meta={"scene_id": "1", "path": "/data/Rando/a.mp4"},
    )
    cache.save(
        "k2", "dinov2",
        np.array([0.0], dtype=np.float32),
        np.stack([_unit([0.9, 0.1, 0])]),
        meta={"scene_id": "2", "path": "/data/Rando/b.mp4"},
    )
    return c


@pytest.fixture
def client(cfg):
    return TestClient(create_app(cfg))


def test_stats(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["cached_scenes"] == 2 and body["model"] == "dinov2"


def test_capabilities_reports_index(client):
    r = client.get("/api/capabilities")
    body = r.json()
    assert body["indexed_frames"] == 3
    assert body["has_clip"] is False  # no clip cache seeded


def test_similarity_search_returns_hits_with_thumb_urls(client, monkeypatch):
    # avoid hitting Stash for the stream URL / metadata (no network in tests)
    from peaks.web import service as svc

    monkeypatch.setattr(
        svc.Service, "stream_url", lambda self, sid, start=None: f"stream/{sid}@{start}"
    )
    monkeypatch.setattr(svc.Service, "scene_meta", lambda self, ids: {})
    r = client.get("/api/search/similar", params={"key": "k1", "t": 0.0, "top_k": 5})
    assert r.status_code == 200
    hits = r.json()
    assert hits and hits[0]["scene_id"] == "2"  # own scene k1 excluded
    assert hits[0]["thumb"].startswith("/api/frame?key=k2")
    assert "stream" in hits[0]


def test_text_search_without_clip_index_is_400(client):
    r = client.get("/api/search/text", params={"q": "red couch"})
    assert r.status_code == 400
    assert "CLIP" in r.json()["detail"]


def test_frame_endpoint_decodes_via_service(client, monkeypatch):
    from peaks.web import service as svc

    monkeypatch.setattr(
        svc.Service, "frame_jpeg", lambda self, path, t, size=320: b"\xff\xd8jpeg\xff\xd9"
    )
    r = client.get("/api/frame", params={"key": "k1", "t": 0.0})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content.startswith(b"\xff\xd8")


def test_frame_unknown_key_404(client):
    r = client.get("/api/frame", params={"key": "nope", "t": 0.0})
    assert r.status_code == 404


def test_embed_job_lifecycle(client, monkeypatch):
    from peaks.web import service as svc

    def fake_embed(self, job=None, limit=0):
        job.log("+ scene 1")
        job.progress = {"total": 1, "done": 1}
        return {"embedded": 1, "skipped": 0}

    monkeypatch.setattr(svc.Service, "run_embed", fake_embed)
    r = client.post("/api/embed")
    assert r.status_code == 200
    jid = r.json()["id"]
    for _ in range(50):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running":
            break
        time.sleep(0.02)
    assert j["status"] == "done"
    assert j["result"] == {"embedded": 1, "skipped": 0}


def test_sync_job_lifecycle(client, monkeypatch):
    from peaks.web import service as svc

    seen = {}

    def fake_sync(self, job=None, prune=True, all_models=True):
        seen["prune"] = prune
        job.log("- pruned k9")
        return {"cached": 2, "moved": 1, "orphaned": 1, "pruned": 1, "models": 1}

    monkeypatch.setattr(svc.Service, "run_sync", fake_sync)
    r = client.post("/api/sync", params={"prune": "false"})
    assert r.status_code == 200
    jid = r.json()["id"]
    for _ in range(50):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running":
            break
        time.sleep(0.02)
    assert j["status"] == "done"
    assert j["result"]["pruned"] == 1
    assert seen["prune"] is False  # query param threaded through


def test_failures_endpoint_lists_log(cfg):
    from peaks.failures import failure_log_for

    failure_log_for(cfg).record(
        "fp9", "9", "/data/Rando/x.mp4", error="Invalid NAL unit size",
        mode="sparse", hwaccel="cuda", pipeline="raw", model="dinov2",
    )
    c = TestClient(create_app(cfg))
    body = c.get("/api/failures").json()
    assert len(body["failures"]) == 1
    assert body["failures"][0]["scene_id"] == "9"
    assert c.get("/api/stats").json()["failures"] == 1


def test_fix_job_lifecycle(client, monkeypatch):
    from peaks.web import service as svc

    def fake_fix(self, job=None, limit=0, dry_run=False):
        job.log("  ✓ scene 9: fixed via interval/off/jpeg")
        return {"fixed": 1, "failed": 0, "total": 1}

    monkeypatch.setattr(svc.Service, "run_fix", fake_fix)
    jid = client.post("/api/fix").json()["id"]
    for _ in range(50):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running":
            break
        time.sleep(0.02)
    assert j["status"] == "done" and j["result"]["fixed"] == 1


def test_defaults_endpoint(client):
    d = client.get("/api/defaults").json()
    assert d["model"] == "dino"
    assert "interval" in d and "workers" in d and "mode" in d
    assert "high" in d and "low" in d and d["tag"] == "apex"
    assert "max_duration" in d and "normalize" in d


def test_score_forwards_thresholds(client, monkeypatch):
    from peaks.web import service as svc

    seen = {}

    def fake_score(self, job=None, tag=None, write=False, **kw):
        seen.update(kw)
        seen["tag"], seen["write"] = tag, write
        return {"segments": 0}

    monkeypatch.setattr(svc.Service, "run_score", fake_score)
    jid = client.post(
        "/api/score",
        params={"tag": "apex", "write": "true", "high": 0.35, "low": 0.28,
                "reduce": "mean", "max_duration": 30, "normalize": "scene-z"},
    ).json()["id"]
    for _ in range(50):
        if client.get(f"/api/jobs/{jid}").json()["status"] != "running":
            break
        time.sleep(0.02)
    assert seen["high"] == 0.35 and seen["low"] == 0.28 and seen["reduce"] == "mean"
    assert seen["max_duration"] == 30 and seen["normalize"] == "scene-z"
    assert seen["tag"] == "apex" and seen["write"] is True


def test_playlist_job(client, monkeypatch):
    from peaks.web import service as svc

    monkeypatch.setattr(
        svc.Service, "run_playlist",
        lambda self, job=None, tags=None: {"tag": "apex", "count": 4, "out": "x"},
    )
    jid = client.post("/api/playlist").json()["id"]
    for _ in range(50):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running":
            break
        time.sleep(0.02)
    assert j["status"] == "done" and j["result"]["count"] == 4


def test_embed_forwards_advanced_overrides(client, monkeypatch):
    from peaks.web import service as svc

    seen = {}

    def fake_embed(self, job=None, limit=0, **kw):
        seen.update(kw)
        seen["limit"] = limit
        return {"embedded": 0}

    monkeypatch.setattr(svc.Service, "run_embed", fake_embed)
    jid = client.post(
        "/api/embed",
        params={"model": "clip", "mode": "interval", "interval": 4,
                "hwaccel": "", "workers": 2, "timeout": 600},
    ).json()["id"]
    for _ in range(50):
        if client.get(f"/api/jobs/{jid}").json()["status"] != "running":
            break
        time.sleep(0.02)
    assert seen["model"] == "clip" and seen["mode"] == "interval"
    assert seen["interval"] == 4 and seen["workers"] == 2
    assert seen["scene_timeout"] == 600
    assert seen["hwaccel"] == ""  # empty string forwarded (force CPU), not dropped


def test_embed_queues_multiple_models(client, monkeypatch):
    from peaks.web import service as svc

    seen = {}

    def fake_multi(self, job=None, models=None, limit=0, **kw):
        seen["models"] = models
        return {"embedded": 0}

    monkeypatch.setattr(svc.Service, "run_embed_multi", fake_multi)
    jid = client.post("/api/embed", params={"model": "dino,clip"}).json()["id"]
    for _ in range(50):
        if client.get(f"/api/jobs/{jid}").json()["status"] != "running":
            break
        time.sleep(0.02)
    assert seen["models"] == ["dino", "clip"]


def test_embed_without_overrides_stays_bare(client, monkeypatch):
    from peaks.web import service as svc

    seen = {}

    def fake_embed(self, job=None, limit=0, **kw):
        seen["kw"] = kw
        return {"embedded": 0}

    monkeypatch.setattr(svc.Service, "run_embed", fake_embed)
    jid = client.post("/api/embed").json()["id"]
    for _ in range(50):
        if client.get(f"/api/jobs/{jid}").json()["status"] != "running":
            break
        time.sleep(0.02)
    assert seen["kw"] == {}  # nothing forwarded → run_embed uses config defaults


def test_scene_edit_endpoints(client, monkeypatch):
    from peaks.web import service as svc

    calls = {}

    def fake_update(self, sid, **f):
        calls["update"] = (sid, f)
        return {"rating100": f.get("rating100"), "organized": f.get("organized")}

    monkeypatch.setattr(svc.Service, "update_scene", fake_update)
    monkeypatch.setattr(svc.Service, "add_o", lambda self, sid: 5)
    monkeypatch.setattr(svc.Service, "remove_o", lambda self, sid: 4)

    r = client.patch("/api/scene/7", json={"rating100": 80, "organized": True})
    assert r.status_code == 200 and r.json()["rating100"] == 80
    assert calls["update"] == ("7", {"rating100": 80, "organized": True})

    assert client.post("/api/scene/7/o").json() == {"o_counter": 5}
    assert client.delete("/api/scene/7/o").json() == {"o_counter": 4}


def test_label_and_train_endpoints(client, monkeypatch):
    from peaks.web import service as svc

    seen = {}
    monkeypatch.setattr(svc.Service, "add_label",
                        lambda self, key, t, label, profile=None, scene_id=None:
                        seen.update(key=key, t=t, label=label, scene_id=scene_id) or {"positive": 1, "negative": 0})
    monkeypatch.setattr(svc.Service, "train_taste",
                        lambda self, profile=None, model=None: {"samples": 4, "positives": 2, "cv_auc": 0.9})

    r = client.post("/api/label", params={"key": "k1", "t": 3.0, "label": 1, "scene_id": "7"})
    assert r.status_code == 200 and r.json()["positive"] == 1
    assert seen["key"] == "k1" and seen["label"] == 1 and seen["scene_id"] == "7"

    r2 = client.post("/api/train")
    assert r2.status_code == 200 and r2.json()["cv_auc"] == 0.9


def test_search_forwards_taste_flag(client, monkeypatch):
    from peaks.web import service as svc

    seen = {}
    monkeypatch.setattr(svc.Service, "has_clip_index", lambda self: True)
    monkeypatch.setattr(svc.Service, "scene_meta", lambda self, ids: {})
    monkeypatch.setattr(svc.Service, "stream_url", lambda self, sid, start=None: "s")

    def fake_text(self, q, top_k=60, taste=False):
        seen["taste"] = taste
        return []

    monkeypatch.setattr(svc.Service, "search_text", fake_text)
    client.get("/api/search/text", params={"q": "x", "taste": "true"})
    assert seen["taste"] is True


def test_timeline_endpoint(client, monkeypatch):
    from peaks.web import service as svc

    monkeypatch.setattr(
        svc.Service, "scene_timeline",
        lambda self, key, **kw: {"points": [[0, 0.5]], "model": "clip", "kw": list(kw)},
    )
    r = client.get("/api/timeline", params={"key": "k1", "q": "red couch"})
    assert r.status_code == 200 and r.json()["points"] == [[0, 0.5]]


def test_save_apex_endpoint(client, monkeypatch):
    from peaks.web import service as svc

    seen = {}

    def fake(self, scene_id, start, end=None, tag=None):
        seen.update(scene_id=scene_id, start=start, end=end, tag=tag)
        return {"id": "m1", "seconds": start}

    monkeypatch.setattr(svc.Service, "create_apex", fake)
    r = client.post("/api/scene/5/apex", params={"t": 42})
    assert r.status_code == 200 and r.json()["marker"]["id"] == "m1"
    assert seen["scene_id"] == "5" and seen["start"] == 42


def test_scene_edit_empty_body_400(client):
    assert client.patch("/api/scene/7", json={}).status_code == 400


def test_hits_include_editable_metadata(client, monkeypatch):
    from peaks.web import service as svc

    monkeypatch.setattr(svc.Service, "stream_url", lambda self, sid, start=None: "s")
    monkeypatch.setattr(
        svc.Service, "scene_meta",
        lambda self, ids: {i: {"title": "T", "rating100": 40, "o_counter": 2, "organized": True} for i in map(str, ids)},
    )
    hits = client.get("/api/search/similar", params={"key": "k1", "t": 0.0}).json()
    assert hits[0]["rating100"] == 40 and hits[0]["o_counter"] == 2
    assert hits[0]["organized"] is True


# --- JobManager unit tests ----------------------------------------------------


def test_jobmanager_one_per_kind():
    jm = JobManager()
    started = []

    def slow(job: Job):
        started.append(job.id)
        time.sleep(0.2)
        return {}

    jm.start("embed", slow)
    with pytest.raises(RuntimeError, match="already running"):
        jm.start("embed", slow)


def test_jobmanager_cancel_marks_cancelled():
    jm = JobManager()
    started = threading.Event()

    def loop(job: Job):
        started.set()
        while not job.cancelled:
            time.sleep(0.01)
        return {"stopped": True}

    job = jm.start("x", loop)
    assert started.wait(1)
    job.request_cancel()
    for _ in range(100):
        if job.status != "running":
            break
        time.sleep(0.01)
    assert job.status == "cancelled" and job.result == {"stopped": True}


def test_jobmanager_captures_errors():
    jm = JobManager()

    def boom(job: Job):
        raise ValueError("kaboom")

    job = jm.start("x", boom)
    for _ in range(50):
        if job.status != "running":
            break
        time.sleep(0.02)
    assert job.status == "error" and "kaboom" in job.error
