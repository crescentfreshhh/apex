"""One live stream per open tab: the server pushes what changed (running jobs,
the pause, the warm-up banner, "the library changed", new log lines) and
nothing when nothing did — so an idle tab asks the server nothing."""

import json
import os
import sys
import threading

import pytest

pytest.importorskip("fastapi")

sys.path.insert(0, os.path.dirname(__file__))


def _feed(tmp_path, monkeypatch):
    from test_snappy import _client

    c, app = _client(tmp_path, monkeypatch, lambda: {"n": 1})
    from peaks.web.app import LiveFeed

    return c, app, LiveFeed(app.state.service, app.state.jobs)


def test_the_feed_sends_only_what_changed(tmp_path, monkeypatch):
    c, app, feed = _feed(tmp_path, monkeypatch)
    jobs = app.state.jobs
    first = feed.tick(full=True, now=0.0)
    assert set(first) == {"jobs", "hold", "warm", "version"} and first["jobs"] == []
    assert feed.tick(now=1.0) == {}                                 # nothing changed: nothing sent

    gate = threading.Event()
    j = jobs.start("sync", lambda job: (gate.wait(10), None)[1])
    out = feed.tick(now=2.0)
    assert [x["id"] for x in out["jobs"]] == [j.id]
    assert feed.tick(now=2.2) == {}                                 # only the elapsed clock moved
    j.progress = {"done": 1, "total": 5}
    assert "jobs" not in feed.tick(now=2.4)                         # at most once a second…
    assert feed.tick(now=3.1)["jobs"][0]["progress"] == {"done": 1, "total": 5}   # …then it goes out

    j.log("half way")
    out = feed.tick(now=5.0)
    assert [x["msg"] for x in out["lines"]] == ["half way"] and out["lines"][0]["kind"] == "sync"
    assert "lines" not in feed.tick(now=6.0)                        # a line is sent once

    jobs.hold(600)
    assert feed.tick(now=7.0)["hold"]["held"] is True
    jobs.unhold()
    assert feed.tick(now=8.0)["hold"]["held"] is False
    gate.set()
    j._thread.join(5) if hasattr(j, "_thread") else None


def test_a_library_write_is_announced_and_telemetry_is_not(tmp_path, monkeypatch):
    c, app, feed = _feed(tmp_path, monkeypatch)
    feed.tick(full=True, now=0.0)
    c.post("/api/hold?minutes=60")                                  # telemetry: no version bump
    out = feed.tick(now=1.0)
    assert "version" not in out and out["hold"]["held"]
    c.post("/api/collections", json={"name": "x"})                  # a write: the library changed
    assert "version" in feed.tick(now=2.0)


def test_the_stream_over_http(tmp_path, monkeypatch):
    c, app, _ = _feed(tmp_path, monkeypatch)
    svc = app.state.service
    before = svc.__dict__.get("_data_version", 0)
    r = c.get("/api/events?once=1", headers={"Accept-Encoding": "gzip"})   # the browser asks for gzip
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert "content-encoding" not in r.headers                        # never buffered for compression
    body = r.text
    assert body.startswith("retry: 3000")
    events = {}
    for block in body.split("\n\n"):
        lines = dict(ln.split(": ", 1) for ln in block.splitlines() if ln and not ln.startswith(("retry", ":")))
        if "event" in lines:
            events[lines["event"]] = json.loads(lines["data"])
    assert set(events) == {"jobs", "hold", "warm", "version"}
    assert events["jobs"] == [] and events["hold"]["held"] is False and "loading" in events["warm"]
    assert svc.__dict__.get("_data_version", 0) == before            # a GET: nothing bumped, nothing written
