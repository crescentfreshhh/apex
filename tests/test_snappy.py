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


@pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="Linux only")
def test_job_decoders_run_at_low_priority():
    from peaks.sampling import background_priority

    out = subprocess.run(["sh", "-c", "nice"], capture_output=True, text=True, preexec_fn=background_priority)
    assert int(out.stdout.strip()) >= 10
