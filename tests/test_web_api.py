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
    c.embedding.dino_model = "dinov2_vits14"  # legacy "dinov2" namespace (tests seed it)
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


def test_radio_endpoint(cfg, tmp_path, monkeypatch):
    import peaks.web.service as svc_mod

    cfg.modeling.labels_path = str(tmp_path / "labels.json")

    class _C:  # fast, offline: no markers, no network
        def iter_markers_by_tag(self, tag, page_size=200):
            return iter(())

        def stream_url(self, sid, start=None):
            return f"http://s/{sid}?t={start}"

    monkeypatch.setattr(svc_mod.Service, "client", lambda self: _C())
    monkeypatch.setattr(svc_mod.Service, "scene_meta", lambda self, ids: {})
    client = TestClient(create_app(cfg))
    # a thumbs-up gives the taste centroid something to build the queue from
    client.post("/api/label", params={"key": "k1", "t": 0.0, "label": 1, "scene_id": "1"})

    d = client.get("/api/radio?count=5").json()
    assert d["items"] and all(it["scene_id"] and it["stream"] for it in d["items"])
    sids = ",".join({it["scene_id"] for it in d["items"]})
    assert client.get("/api/radio?count=5&exclude=" + sids).json()["items"] == []


def test_sample_endpoint_returns_random_frames(cfg, monkeypatch):
    import peaks.web.service as svc_mod

    monkeypatch.setattr(svc_mod.Service, "scene_meta", lambda self, ids: {})
    monkeypatch.setattr(svc_mod.Service, "stream_url", lambda self, sid, start=None: f"http://s/{sid}?t={start}")
    client = TestClient(create_app(cfg))

    # library has 3 seeded frames (k1 x2, k2 x1) — asking for more is clamped.
    d = client.get("/api/foryou/sample?count=10").json()
    items = d["items"]
    assert 0 < len(items) <= 3
    assert all(it["thumb"].startswith("/api/frame?key=") for it in items)
    assert all(it["stream"] for it in items)
    # distinct (key, time) rows, no repeats within a batch
    pairs = {(it["key"], it["time"]) for it in items}
    assert len(pairs) == len(items)

    # count caps the batch size
    small = client.get("/api/foryou/sample?count=2").json()["items"]
    assert len(small) == 2


def test_taste_metrics_endpoint(cfg, tmp_path, monkeypatch):
    import peaks.web.service as svc_mod

    cfg.modeling.labels_path = str(tmp_path / "labels.json")

    class _C:
        def iter_markers_by_tag(self, tag, page_size=200):
            return iter(())

    monkeypatch.setattr(svc_mod.Service, "client", lambda self: _C())
    client = TestClient(create_app(cfg))

    # no taste yet -> has_taste False, still returns label health
    assert client.get("/api/taste/metrics").json()["has_taste"] is False

    client.post("/api/label", params={"key": "k1", "t": 0.0, "label": 1, "scene_id": "1"})
    m = client.get("/api/taste/metrics").json()
    assert m["has_taste"] is True
    assert len(m["bands"]) == 4 and m["indexed"]["frames"] == 3
    assert m["sources"]["thumbs_up"] >= 1
    assert "threshold" not in m  # only present when asked

    mt = client.get("/api/taste/metrics?threshold=0.5").json()["threshold"]
    assert mt["value"] == 0.5 and 0 <= mt["percentile"] <= 100


def test_foryou_board_endpoint(cfg, tmp_path, monkeypatch):
    import peaks.web.service as svc_mod

    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    monkeypatch.setattr(svc_mod.Service, "scene_meta", lambda self, ids: {})
    monkeypatch.setattr(svc_mod.Service, "stream_url", lambda self, sid, start=None: f"http://s/{sid}?t={start}")

    class _C:
        def iter_markers_by_tag(self, tag, page_size=200):
            return iter(())

    monkeypatch.setattr(svc_mod.Service, "client", lambda self: _C())
    client = TestClient(create_app(cfg))
    client.post("/api/label", params={"key": "k1", "t": 0.0, "label": 1, "scene_id": "1"})

    board = client.get("/api/foryou/board?count=6").json()
    items = board["items"]
    assert items and all(it["scene_id"] and it["stream"] for it in items)
    # coverage numbers come back so the board can show "how many match my taste"
    assert board["scenes"] >= 1 and board["moments"] >= board["scenes"]
    assert board["scored_by"] in ("classifier", "modes", "centroid")
    sids = ",".join({it["scene_id"] for it in items})
    assert client.get("/api/foryou/board?count=6&exclude=" + sids).json()["items"] == []
    # taste floor is an optional tightener: an impossibly high floor filters all out
    assert client.get("/api/foryou/board?count=6&min_score=2.0").json()["items"] == []


def test_profiles_are_isolated(cfg, tmp_path, monkeypatch):
    """Each taste profile keeps its own labels + feed; the default is untouched
    when a second profile is trained, and deleting a profile erases its taste."""
    import peaks.web.service as svc_mod

    cfg.modeling.labels_path = str(tmp_path / "labels.json")

    class _C:  # offline: no markers for any tag
        def iter_markers_by_tag(self, tag, page_size=200):
            return iter(())

        def stream_url(self, sid, start=None):
            return f"http://s/{sid}?t={start}"

        def destroy_scene_markers(self, ids):
            return len(ids)

    monkeypatch.setattr(svc_mod.Service, "client", lambda self: _C())
    monkeypatch.setattr(svc_mod.Service, "scene_meta", lambda self, ids: {})
    client = TestClient(create_app(cfg))

    default = client.get("/api/profiles").json()["default"]
    assert client.get("/api/profiles").json()["profiles"] == [default]

    # a new profile shows up and is selectable
    made = client.post("/api/profiles", params={"name": "kinks"}).json()["profiles"]
    assert "kinks" in made

    # a 👍 filed under "kinks" trains only that profile — the default stays empty
    client.post("/api/label", params={"key": "k1", "t": 0.0, "label": 1,
                                      "scene_id": "1", "profile": "kinks"})
    assert client.get("/api/labels", params={"profile": "kinks"}).json()["positive"] == 1
    assert client.get("/api/labels").json()["positive"] == 0
    # and only "kinks" has a feed
    assert client.get("/api/foryou", params={"profile": "kinks"}).json()["items"]
    assert client.get("/api/foryou").json()["items"] == []

    # deleting drops it from the registry and erases its labels
    client.request("DELETE", "/api/profiles", params={"name": "kinks"})
    assert "kinks" not in client.get("/api/profiles").json()["profiles"]
    assert client.get("/api/labels", params={"profile": "kinks"}).json()["positive"] == 0

    # the default profile can't be deleted
    assert client.request("DELETE", "/api/profiles",
                          params={"name": default}).status_code == 400


def test_search_similar_accepts_scene_id(cfg, monkeypatch):
    import peaks.web.service as svc_mod

    monkeypatch.setattr(svc_mod.Service, "scene_meta", lambda self, ids: {})
    monkeypatch.setattr(svc_mod.Service, "stream_url", lambda self, sid, start=None: f"http://s/{sid}?t={start}")
    client = TestClient(create_app(cfg))

    # scene "1" is embedded (fixture seeds k1 -> scene_id "1"); resolves to its key
    d = client.get("/api/search/similar?scene_id=1&t=0&top_k=5").json()
    hits = d["items"]
    assert d["total"] == len(hits) and hits
    assert all(h["thumb"].startswith("/api/frame?key=") for h in hits)
    # unknown scene -> empty (not an error)
    assert client.get("/api/search/similar?scene_id=nope&t=0").json() == {"total": 0, "items": []}


def test_search_similar_threshold_payload(cfg, monkeypatch):
    import peaks.web.service as svc_mod

    # seed extra scenes so a similarity search returns several matches
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    for sid in ("3", "4", "5"):
        cache.save(f"k{sid}", "dinov2", np.array([0.0], dtype=np.float32),
                   np.stack([_unit([0.9, 0.1, 0])]), meta={"scene_id": sid, "path": f"/d/{sid}.mp4"})
    monkeypatch.setattr(svc_mod.Service, "scene_meta",
                        lambda self, ids: {i: {"title": f"scene {i}", "performers": ["X"]} for i in ids})
    monkeypatch.setattr(svc_mod.Service, "stream_url", lambda self, sid, start=None: f"http://s/{sid}?t={start}")
    client = TestClient(create_app(cfg))

    # similar to scene 1 (k1@0), low floor → several matches as {total, items};
    # only the first `enrich`=1 is enriched, the rest are lightweight.
    d = client.get("/api/search/similar?scene_id=1&t=0&min_score=0.0&per_scene=0&enrich=1").json()
    assert d["total"] == len(d["items"]) and d["total"] > 1
    assert d["items"][0]["title"].startswith("scene")   # preview enriched
    assert d["items"][1]["title"] == ""                 # tail is lightweight
    assert d["items"][1]["scene_id"] and d["items"][1]["stream"]  # still playable
    # a strict floor returns fewer than a loose one
    strict = client.get("/api/search/similar?scene_id=1&t=0&min_score=0.99").json()["total"]
    loose = client.get("/api/search/similar?scene_id=1&t=0&min_score=0.0").json()["total"]
    assert strict <= loose


def test_collection_rename_delete_endpoints(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKS_COLLECTIONS_DIR", str(tmp_path / "coll"))
    client = TestClient(create_app(cfg))
    client.post("/api/collection", json={"name": "Faves", "apexes": [{"scene_id": "1", "start": 0, "url": "u"}]})
    safe = client.get("/api/collections").json()["collections"][0]["safe"]

    client.post("/api/collection/rename", json={"name": "Faves", "new_name": "Top Tier"})
    got = client.get("/api/collections").json()["collections"][0]
    assert got["name"] == "Top Tier" and got["safe"] == safe   # link stays stable

    assert client.delete("/api/collection", params={"name": safe}).status_code == 200
    assert client.get("/api/collections").json()["collections"] == []
    assert client.delete("/api/collection", params={"name": safe}).status_code == 404


def _perf_client_cls():
    class _C:
        def scene_details(self, ids):
            return {str(i): {"performers_detail": [{"id": "p1", "name": "Ava"}]} for i in ids}

        def scenes_for_performer(self, pid, limit=500):
            return ["1", "2"]

        def find_performers(self, name=None, limit=500):
            return [{"id": "p1", "name": "Ava", "image": "/i", "scene_count": 2}]
    return _C


def test_performer_best_and_leaderboard_endpoints(cfg, tmp_path, monkeypatch):
    import peaks.web.service as svc_mod

    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: _perf_client_cls()())
    monkeypatch.setattr(svc_mod.Service, "scene_meta", lambda self, ids: {})
    monkeypatch.setattr(svc_mod.Service, "stream_url", lambda self, sid, start=None: f"http://s/{sid}?t={start}")
    client = TestClient(create_app(cfg))
    client.post("/api/label", params={"key": "k1", "t": 0.0, "label": 1, "scene_id": "1"})

    best = client.get("/api/performer/best?name=Ava&per_scene=2").json()
    assert best["performer"] == "Ava" and best["items"]
    assert all(it["scene_id"] in {"1", "2"} for it in best["items"])
    scores = [it["score"] for it in best["items"]]
    assert scores == sorted(scores, reverse=True)   # ranked

    board = client.get("/api/performers?sort=moments").json()["performers"]
    assert board and board[0]["name"] == "Ava" and board[0]["scenes"] == 2
    assert "affinity" in board[0] and "top" in board[0]

    detail = client.get("/api/performer/detail?id=p1").json()
    assert detail["performer"] == "Ava" and detail["items"]
    assert detail["stats"]["scenes"] == 2
    assert "similar" in detail and "distribution" in detail

    r = client.get("/api/performer/roulette").json()
    assert r["id"] in {"p1"}   # only performer in this fixture


def test_hall_of_fame_endpoint(cfg, tmp_path, monkeypatch):
    import peaks.web.service as svc_mod

    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    monkeypatch.setenv("PEAKS_COLLECTIONS_DIR", str(tmp_path / "coll"))
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: _perf_client_cls()())
    monkeypatch.setattr(svc_mod.Service, "scene_meta", lambda self, ids: {})
    monkeypatch.setattr(svc_mod.Service, "stream_url", lambda self, sid, start=None: f"http://s/{sid}?t={start}")
    client = TestClient(create_app(cfg))
    client.post("/api/label", params={"key": "k1", "t": 0.0, "label": 1, "scene_id": "1"})

    hof = client.post("/api/performers/hall-of-fame", params={"top_n": 5}).json()
    assert hof["created"]
    names = {c["name"] for c in client.get("/api/collections").json()["collections"]}
    assert any("best of" in n for n in names)


def test_collection_export_starts_job(cfg, tmp_path, monkeypatch):
    import peaks.web.service as svc_mod

    monkeypatch.setenv("PEAKS_COLLECTIONS_DIR", str(tmp_path / "coll"))
    captured = {}
    monkeypatch.setattr(svc_mod.Service, "export_collection",
                        lambda self, job=None, name="", limit=200: captured.update(name=name, limit=limit) or {"clips": 0})
    client = TestClient(create_app(cfg))
    client.post("/api/collection", json={"name": "Reel", "apexes": [{"scene_id": "1", "start": 0, "url": "u"}]})

    r = client.post("/api/collection/export", params={"name": "Reel"})
    assert r.status_code == 200 and "id" in r.json()  # a job was started
    import time
    for _ in range(50):
        if captured:
            break
        time.sleep(0.02)
    assert captured.get("name") == "Reel"


def test_board_performer_endpoint(cfg, monkeypatch):
    import peaks.web.service as svc_mod

    monkeypatch.setattr(svc_mod.Service, "scene_meta", lambda self, ids: {})
    monkeypatch.setattr(svc_mod.Service, "stream_url", lambda self, sid, start=None: f"http://s/{sid}?t={start}")

    class _C:
        def scene_details(self, ids):
            return {str(ids[0]): {"performers_detail": [{"id": "p1", "name": "Ava"}]}}

        def scenes_for_performer(self, pid, limit=500):
            return ["1", "2"]  # both embedded in the fixture (k1->1, k2->2)

    monkeypatch.setattr(svc_mod.Service, "client", lambda self: _C())
    client = TestClient(create_app(cfg))
    d = client.get("/api/board/performer?scene_id=1").json()
    assert d["performer"] == "Ava"
    assert d["items"] and all(it["scene_id"] in {"1", "2"} for it in d["items"])


def test_autotrain_kicks_after_threshold(cfg, tmp_path):
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    cfg.modeling.autotrain_every = 2
    client = TestClient(create_app(cfg))
    # frames k1@0 and k2@0 are seeded in the dinov2 cache by the fixture
    r1 = client.post("/api/label", params={"key": "k1", "t": 0.0, "label": 1})
    assert r1.json()["autotrain"] is False  # 1 rating, below threshold
    r2 = client.post("/api/label", params={"key": "k2", "t": 0.0, "label": 0})
    assert r2.json()["autotrain"] is True  # 2 ratings + both classes → background train


def test_no_auth_by_default(client):
    # empty password → open app, capabilities reports auth off
    assert client.get("/api/stats").status_code == 200
    assert client.get("/api/capabilities").json()["auth"] is False


def test_auth_gate_blocks_then_allows(cfg):
    cfg.auth.password = "hunter2"
    # a fresh client with no cookie jar shared across the login boundary
    client = TestClient(create_app(cfg))
    # API is 401 without a session; a browser GET gets the login page (200 HTML)
    assert client.get("/api/stats").status_code == 401
    page = client.get("/")
    assert page.status_code == 200 and "sign in" in page.text.lower()
    assert client.get("/api/capabilities").status_code == 401

    # wrong password stays locked
    assert client.post("/api/login", json={"password": "nope"}).status_code == 401
    # correct password sets the session cookie; the client jar carries it
    ok = client.post("/api/login", json={"password": "hunter2"})
    assert ok.status_code == 200 and "peaks_session" in ok.cookies
    assert client.get("/api/stats").status_code == 200

    # logout clears it → gated again
    client.post("/api/logout")
    assert client.get("/api/stats").status_code == 401


def test_auth_rejects_garbage_and_sets_ttl(cfg):
    cfg.auth.password = "pw"
    cfg.auth.session_hours = 1.0
    client = TestClient(create_app(cfg))
    # a forged/garbage session cookie is not a known token → still gated
    client.cookies.set("peaks_session", "not-a-real-token")
    assert client.get("/api/stats").status_code == 401
    # a real login stamps a cookie whose max-age matches the 1-hour session
    ok = client.post("/api/login", json={"password": "pw"})
    set_cookie = ok.headers["set-cookie"].lower()
    assert "max-age=3600" in set_cookie and "httponly" in set_cookie


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
    hits = r.json()["items"]
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


def test_label_by_scene_id_resolves_key(client, monkeypatch):
    """The megaboard rates by scene_id (no cache key) — the endpoint resolves it."""
    from peaks.web import service as svc

    seen = {}
    monkeypatch.setattr(svc.Service, "add_label",
                        lambda self, key, t, label, profile=None, scene_id=None:
                        seen.update(key=key, label=label, scene_id=scene_id) or {"positive": 1, "negative": 0})

    # no key given — resolved from scene_id (fixture seeds scene "1" -> key "k1")
    r = client.post("/api/label", params={"t": 0.0, "label": 1, "scene_id": "1"})
    assert r.status_code == 200 and seen["key"] == "k1" and seen["scene_id"] == "1"
    # neither key nor a resolvable scene_id -> 400
    assert client.post("/api/label", params={"t": 0.0, "label": 1}).status_code == 400


def test_search_forwards_taste_flag(client, monkeypatch):
    from peaks.web import service as svc

    seen = {}
    monkeypatch.setattr(svc.Service, "has_clip_index", lambda self: True)
    monkeypatch.setattr(svc.Service, "scene_meta", lambda self, ids: {})
    monkeypatch.setattr(svc.Service, "stream_url", lambda self, sid, start=None: "s")

    def fake_text(self, q, top_k=60, taste=False, **kw):
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
    hits = client.get("/api/search/similar", params={"key": "k1", "t": 0.0}).json()["items"]
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
