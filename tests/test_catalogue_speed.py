"""Catalogue navigation stays quick: thumbnails are decoded once, a few at a
time, never for a page you've left; saved moments and picture scores answer
from cache while they refresh; the reject memory isn't re-read every click."""

import threading
import time

import pytest

pytest.importorskip("fastapi")

from fakestash import FakeStash  # noqa: E402
from peaks.config import Config  # noqa: E402


@pytest.fixture
def stash():
    s = FakeStash({str(i): {"title": f"Scene {i}", "rating100": None} for i in range(1, 4)})
    s.markers = [{"scene_id": "1", "seconds": 30.0, "marker_id": "m1", "title": ""}]
    return s


@pytest.fixture
def svc(tmp_path, stash, monkeypatch):
    import peaks.web.service as svc_mod

    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    return svc_mod.Service(cfg)


@pytest.fixture
def app_svc(svc, monkeypatch, tmp_path):
    import peaks.web.app as app_mod

    monkeypatch.setenv("PEAKS_WARM_ON_START", "0")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(type(svc), "path_for_key", lambda self, key: str(video))
    monkeypatch.setattr(app_mod, "Service", lambda cfg: svc)
    app = app_mod.create_app(svc.cfg)
    return app, svc


def _client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


def test_a_thumbnail_is_decoded_once_then_served_from_disk(app_svc, monkeypatch):
    app, svc = app_svc
    decodes = []
    monkeypatch.setattr(type(svc), "frame_jpeg",
                        lambda self, path, t, size=320: decodes.append(t) or b"\xff\xd8one\xff\xd9")
    with _client(app) as c:
        a = c.get("/api/frame?key=fpA&t=12.5")
        b = c.get("/api/frame?key=fpA&t=12.5")
        other = c.get("/api/frame?key=fpA&t=40")
    assert a.status_code == b.status_code == other.status_code == 200
    assert a.content == b.content == b"\xff\xd8one\xff\xd9"
    assert decodes == [12.5, 40.0]                          # the repeat came from disk
    assert "max-age=604800" in b.headers["cache-control"]   # and the browser keeps it too


def test_failed_decodes_are_not_cached(app_svc, monkeypatch):
    app, svc = app_svc

    def boom(self, path, t, size=320):
        raise RuntimeError("bad file")
    monkeypatch.setattr(type(svc), "frame_jpeg", boom)
    monkeypatch.setattr(type(svc), "refresh_path", lambda self, key: None)
    with _client(app) as c:
        assert c.get("/api/frame?key=fpB&t=1").status_code == 500
    assert svc.frame_cached("fpB", 1.0, 320) is None


def test_only_a_few_thumbnails_decode_at_once(app_svc, monkeypatch):
    import peaks.web.app as app_mod

    app, svc = app_svc
    live, peak, lock = [0], [0], threading.Lock()

    def slow(self, path, t, size=320):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.15)
        with lock:
            live[0] -= 1
        return b"\xff\xd8x\xff\xd9"
    monkeypatch.setattr(type(svc), "frame_jpeg", slow)
    codes = []
    with _client(app) as c:
        threads = [threading.Thread(target=lambda i=i: codes.append(c.get(f"/api/frame?key=fpC&t={i}").status_code))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert codes == [200] * 10
    assert peak[0] <= app_mod._FRAME_DECODES


def test_a_thumbnail_for_a_page_you_left_is_skipped(app_svc, monkeypatch):
    import anyio

    app, svc = app_svc
    decodes = []
    monkeypatch.setattr(type(svc), "frame_jpeg", lambda self, path, t, size=320: decodes.append(t) or b"x")
    route = next(r for r in app.routes if getattr(r, "path", "") == "/api/frame")

    class Gone:
        async def is_disconnected(self):
            return True
    resp = anyio.run(lambda: route.endpoint(Gone(), "fpD", 3.0, 320))
    assert resp.status_code == 499 and decodes == []


def test_the_thumbnail_cache_is_pruned_oldest_first(svc):
    for i in range(5):
        svc.frame_store("fpE", float(i), 320, b"x" * 1000)
        p = svc._frame_cache_path("fpE", float(i), 320)
        import os
        os.utime(p, (1000 + i, 1000 + i))
    assert svc.prune_frame_cache(cap_bytes=2500) == 3
    assert [svc.frame_cached("fpE", float(i), 320) is not None for i in range(5)] == [False, False, False, True, True]


def test_backups_leave_the_thumbnail_cache_out(svc):
    svc.frame_store("fpF", 1.0, 320, b"jpg")
    names = [rel for _, rel in svc._part_files("embeddings")]
    assert not any(n.startswith("_frames") for n in names)


def test_the_reject_memory_is_kept_between_clicks(svc):
    assert svc.reject_memory() is svc.reject_memory()


def _wait(cond, secs=3.0):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_stale_saved_moments_answer_at_once_and_refresh_behind(svc, stash):
    first = svc._taste_markers("apex")
    assert len(first) == 1
    stash.markers = stash.markers + [{"scene_id": "2", "seconds": 5.0, "marker_id": "m2", "title": ""}]
    svc._marker_cache["apex"] = (time.monotonic() - svc._MARKER_TTL - 1, first)
    assert len(svc._taste_markers("apex")) == 1           # no wait on Stash
    assert _wait(lambda: len(svc._marker_cache["apex"][1]) == 2)


def test_a_save_during_the_background_refresh_is_not_lost(svc, stash, monkeypatch):
    svc._taste_markers("apex")
    gate = threading.Event()

    def slow_iter(tag):
        gate.wait(3)
        return iter([{"scene_id": "1", "seconds": 30.0, "marker_id": "m1", "title": ""}])
    monkeypatch.setattr(stash, "iter_markers_by_tag", slow_iter)
    svc._marker_cache["apex"] = (time.monotonic() - svc._MARKER_TTL - 1, svc._marker_cache["apex"][1])
    svc._taste_markers("apex")                              # starts the re-read
    svc._note_marker("apex", mk={"scene_id": "3", "seconds": 9.0, "marker_id": "m3", "title": ""})
    gate.set()
    assert _wait(lambda: "apex" not in svc._marker_refreshing)
    assert {m["marker_id"] for m in svc._marker_cache["apex"][1]} == {"m1", "m3"}


def test_picture_scores_are_not_recomputed_on_every_click(svc, monkeypatch):
    calls = []
    real = type(svc)._pic_measures
    monkeypatch.setattr(type(svc), "_pic_measures", lambda self: calls.append(1) or real(self))
    svc.picture_scores()                    # first use: trains + scores
    calls.clear()
    svc.picture_scores()
    svc.picture_scores()
    assert calls == []
    svc._pic_dirty = True                   # a new rating counts at once
    svc.picture_scores()
    assert calls
