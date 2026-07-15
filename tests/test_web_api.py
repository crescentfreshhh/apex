"""Web API tests via FastAPI TestClient — a fake embedding cache, no torch,
no Stash, no ffmpeg (frame decoding is monkeypatched)."""

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
    # avoid hitting Stash for the stream URL
    from peaks.web import service as svc

    monkeypatch.setattr(
        svc.Service, "stream_url", lambda self, sid, start=None: f"stream/{sid}@{start}"
    )
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
