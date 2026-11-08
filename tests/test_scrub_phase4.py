"""The browser's chatter costs the server little: the job tray asks for running
jobs only (no logs, no results), the Activity log's history file is parsed once
per change, and the taste metrics are a snapshot like the summary."""

import os
import sys
import time

import pytest

pytest.importorskip("fastapi")

sys.path.insert(0, os.path.dirname(__file__))


def test_the_tray_gets_running_jobs_only_without_logs(tmp_path, monkeypatch):
    from test_snappy import _client

    c, app = _client(tmp_path, monkeypatch, lambda: {"n": 1})
    assert c.get("/api/jobs/running").json() == []
    d = c.post("/api/hold?minutes=60").json()                     # (a write that's cheap to make)
    assert d["held"]
    running = c.get("/api/jobs/running").json()
    assert all(set(j) <= {"id", "kind", "status", "started", "elapsed", "progress", "timing", "paused_for"} for j in running)


def test_the_job_history_file_is_parsed_once_per_change(tmp_path):
    from peaks.web.jobs import JobManager

    p = tmp_path / "hist.jsonl"
    p.write_text('{"id": "a-1", "started": 1, "finished": 2, "kind": "embed", "status": "done"}\n')
    jm = JobManager(history_path=p)
    reads = []
    real = type(p).read_text
    import pathlib
    pathlib.Path.read_text = lambda self, *a, **k: (reads.append(str(self)) if str(self) == str(p) else None) or real(self, *a, **k)
    try:
        assert jm.history()[0]["id"] == "a-1"
        jm.history(); jm.history()
        assert reads.count(str(p)) == 1
        time.sleep(0.01)
        with p.open("a") as fh:
            fh.write('{"id": "b-2", "started": 3, "finished": 4, "kind": "sync", "status": "done"}\n')
        assert [h["id"] for h in jm.history()] == ["b-2", "a-1"] and reads.count(str(p)) == 2
    finally:
        pathlib.Path.read_text = real


def test_taste_metrics_is_a_snapshot(tmp_path, monkeypatch):
    from test_snappy import _client

    c, app = _client(tmp_path, monkeypatch, lambda: {"n": 1})
    import peaks.web.service as svc_mod

    calls = []
    monkeypatch.setattr(svc_mod.Service, "taste_metrics", lambda self, threshold=None: calls.append(threshold) or {"has_taste": False})
    c.get("/api/taste/metrics"); c.get("/api/taste/metrics"); c.get("/api/taste/metrics")
    assert calls == [None]                                        # one computation behind the snapshot
    c.get("/api/taste/metrics?threshold=0.5")
    assert calls == [None, 0.5]                                   # a specific bar: computed
