"""Snappier pages: page snapshots (a write always invalidates), text-only gzip,
the Speed panel's timings, and job decoders at low priority."""

import gzip
import os
import subprocess

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from peaks.config import Config  # noqa: E402


def _client(tmp_path, monkeypatch, summary):
    import peaks.web.app as app_mod
    import peaks.web.service as svc_mod

    monkeypatch.setattr(svc_mod.Service, "library_summary", lambda self: summary())
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    app = app_mod.create_app(cfg)
    return TestClient(app), app


def test_a_snapshot_is_served_until_a_write(tmp_path, monkeypatch):
    calls = []

    def summary():
        calls.append(1)
        return {"n": len(calls)}
    c, app = _client(tmp_path, monkeypatch, summary)
    assert c.get("/api/catalogue/summary").json() == {"n": 1}
    assert c.get("/api/catalogue/summary").json() == {"n": 1} and len(calls) == 1     # served, not recomputed
    c.post("/api/memory/shed", params={"drop_indexes": False})                         # any write
    assert c.get("/api/catalogue/summary").json() == {"n": 2}                         # fresh, at once


def test_json_is_gzipped_but_video_and_ranges_never_are(tmp_path, monkeypatch):
    big = {"items": ["x" * 50] * 200}
    c, app = _client(tmp_path, monkeypatch, lambda: big)
    r = c.get("/api/catalogue/summary", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip" and r.json() == big
    assert "app;dur=" in r.headers["server-timing"]

    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Route

    from peaks.web.compress import TextGzip

    def video(_):
        return Response(b"\0" * 5000, media_type="video/mp4")

    def js(_):
        return Response("let x = 1;\n" * 500, media_type="text/javascript")
    tc = TestClient(TextGzip(Starlette(routes=[Route("/v", video), Route("/j", js)])))
    assert "content-encoding" not in tc.get("/v", headers={"Accept-Encoding": "gzip"}).headers
    r = tc.get("/j", headers={"Accept-Encoding": "gzip"})
    assert r.headers["content-encoding"] == "gzip" and r.text.startswith("let x")
    assert "content-encoding" not in tc.get("/j", headers={"Accept-Encoding": "gzip", "Range": "bytes=0-9"}).headers
    assert gzip.decompress(gzip.compress(b"ok")) == b"ok"


def test_the_speed_panel_lists_routes_slowest_first_and_purges(tmp_path, monkeypatch):
    from peaks.web import forensics, memwatch

    forensics._timings.clear()
    c, app = _client(tmp_path, monkeypatch, lambda: {"n": 1})
    for _ in range(3):
        c.get("/api/catalogue/summary")
    forensics.note_duration("GET", "/api/slow", 2.5, route="/api/slow")
    memwatch.PURGES.append({"t": __import__("time").time(), "level": "soft", "freed_mb": 300,
                            "rss_before_mb": 1, "rss_after_mb": 1, "dropped": []})
    d = c.get("/api/perf").json()
    assert d["routes"][0]["route"] == "GET /api/slow" and d["routes"][0]["p95_ms"] == 2500
    assert any(r["route"] == "GET /api/catalogue/summary" and r["n"] == 3 for r in d["routes"])
    assert d["purges"][0]["level"] == "soft" and "memory" in d and d["running"] == []
    assert "stash" in d and d["load"]["cores"]


def test_a_request_knows_how_long_it_waited_on_stash(tmp_path, monkeypatch):
    import time as _t

    from peaks import stashtiming
    from peaks.web import forensics

    forensics._timings.clear()
    stashtiming.reset()

    def slow_summary():
        with stashtiming.timed():                 # what StashClient.execute does around each call
            _t.sleep(0.05)
        return {"n": 1}
    c, app = _client(tmp_path, monkeypatch, slow_summary)
    r = c.get("/api/catalogue/summary")
    assert "stash;dur=" in r.headers["server-timing"]
    row = next(x for x in c.get("/api/perf").json()["routes"] if x["route"] == "GET /api/catalogue/summary")
    assert row["stash_share"] > 0.3
    assert c.get("/api/perf").json()["stash"]["calls"] == 1


@pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="Linux only")
def test_job_decoders_run_at_low_priority():
    from peaks.sampling import background_priority

    out = subprocess.run(["sh", "-c", "nice"], capture_output=True, text=True, preexec_fn=background_priority)
    assert int(out.stdout.strip()) >= 10


def test_thumbnails_decode_on_the_cpu_and_step_back_from_the_very_end(tmp_path, monkeypatch):
    import peaks.sampling as sampling
    import peaks.web.service as svc_mod

    made = []

    class Sampler:
        def __init__(self, hwaccel=None, **k):
            made.append(hwaccel)

        def grab_frame(self, path, t, width=None):
            from PIL import Image
            if t > 99:
                raise sampling.SamplerError(f"no frame decoded for {path}@{t}s")
            return Image.new("RGB", (64, 36), (200, 0, 0))
    monkeypatch.setattr(sampling, "FrameSampler", Sampler)
    monkeypatch.setattr(svc_mod.Service, "path_for_key", lambda self, k: str(tmp_path / "v.mp4"))
    (tmp_path / "v.mp4").write_bytes(b"x")
    c, app = _client(tmp_path, monkeypatch, lambda: {})
    assert c.get("/api/frame", params={"key": "k1", "t": 12}).headers["content-type"] == "image/jpeg"
    assert made and all(h is None for h in made)                  # never -hwaccel for a thumbnail
    r = c.get("/api/frame", params={"key": "k1", "t": 99.5})       # past the last frame
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
