"""A week of how Peaks spent its time: 5-minute samples (routes, Stash, GPU,
disk), each job's hours and disk, each retrain's gains — one report a night,
with a plain verdict per background job."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from peaks import health  # noqa: E402

DAY = 86400.0
NOW = 1_800_000_000.0


def _week():
    samples, jobs = [], []
    for i in range(7 * 24 * 12):                     # every 5 minutes for a week
        at = NOW - 7 * DAY + i * 300 + 1
        busy = i % 4 == 0
        samples.append(health.sample(at - 300, [
            {"route": "/api/catalogue", "n": 10, "p50_ms": 80, "p95_ms": 400 if busy else 150},
            {"route": "/api/search/text", "n": 2, "p50_ms": 1500, "p95_ms": 2200},
        ], {"calls": 4, "median_ms": 30, "p90_ms": 120}, {"util": 85 if busy else 2, "mem_mb": 5000},
            {"read": 50_000_000, "write": 5_000_000}, ["motion"] if busy else [], now=at))
    for d in range(7):
        jobs.append({"kind": "motion", "at": NOW - d * DAY, "elapsed_s": 3 * 3600, "read": 40e9, "write": 1e9})
        jobs.append({"kind": "hear", "at": NOW - d * DAY, "elapsed_s": 0.5 * 3600, "read": 5e9, "write": 0.1e9})
        jobs.append({"kind": "embed", "at": NOW - d * DAY, "elapsed_s": 2 * 3600, "read": 30e9, "write": 2e9,
                     "overlapped": d == 3})
    jobs.append({"kind": "motion", "at": NOW - 20 * DAY, "elapsed_s": 99 * 3600})     # older than a week: left out
    gains = [{"at": NOW - 5 * DAY, "motion_gain": 0.02, "sound_gain": 0.01},
             {"at": NOW - 1 * DAY, "motion_gain": 0.003, "sound_gain": 0.034}]           # the latest counts
    return samples, jobs, gains


def test_the_week_report_numbers_and_verdicts():
    samples, jobs, gains = _week()
    r = health.build_report(samples, jobs, gains, now=NOW)
    assert r["samples"] == 7 * 24 * 12
    top = r["routes"][0]
    assert top["route"] == "/api/search/text" and top["p95_ms"] == 2200 and top["n"] == 2 * 7 * 24 * 12
    cat = next(x for x in r["routes"] if x["route"] == "/api/catalogue")
    assert cat["p50_ms"] == 80 and cat["p95_ms"] == 400            # a quarter of the week was slow: p95 sees it
    assert r["stash"]["median_ms"] == 30 and r["stash"]["p90_ms"] == 120
    assert r["gpu"]["busy_pct"] == 25.0 and r["gpu"]["peak_mem_mb"] == 5000
    assert r["disk"]["read_gb"] == pytest.approx(7 * 24 * 12 * 0.05, rel=1e-3)
    by = {j["kind"]: j for j in r["jobs"]}
    assert by["motion"]["runs"] == 7 and by["motion"]["hours"] == 21.0 and by["motion"]["read_gb"] == 280.0
    assert by["embed"]["approximate"] and not by["motion"]["approximate"]
    assert by["motion"]["gain_pts"] == 0.3 and by["hear"]["gain_pts"] == 3.4
    v = " | ".join(r["verdicts"])
    assert "Motion: 21.0 h and 280 GB read this week for +0.3 points of grade accuracy — consider turning it off" in v
    assert "Hearing: 3.5 h and 35 GB read this week, +3.4 points of grade accuracy — earns its keep" in v
    assert "Embedding" not in v                                       # core work: numbers, no verdict


def test_an_empty_week_and_missing_parts():
    r = health.build_report([], [], [], now=NOW)
    assert r["routes"] == [] and r["stash"] is None and r["gpu"] is None and r["disk"] is None and r["verdicts"] == []
    r = health.build_report([], [{"kind": "hear", "at": NOW, "elapsed_s": 3600}], [], now=NOW)
    assert r["verdicts"] == ["Hearing: 1.0 h this week — no accuracy measured yet"]


def test_proc_io_and_deltas():
    io = health.proc_io()
    if io is None:
        pytest.skip("no /proc/self/io here")
    assert io["read"] >= 0 and io["write"] >= 0
    assert health.io_delta({"read": 10, "write": 5}, {"read": 25, "write": 4}) == {"read": 15, "write": 0}
    assert health.io_delta(None, io) is None


def test_the_store_keeps_a_rolling_window(tmp_path):
    from peaks.store import Store

    st = Store(tmp_path / "peaks.db")
    for d in range(10):
        st.log_append("health", {"d": d}, at=NOW - d * DAY)
    assert [r["d"] for r in st.log_since("health", NOW - 2.5 * DAY)] == [0, 1, 2]
    assert st.log_prune("health", NOW - 7.5 * DAY) == 2 and st.log_count("health") == 8   # days 8 and 9 go


def test_samples_jobs_and_the_report_endpoint(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from test_snappy import _client

    import peaks.web.app as app_mod

    c, app = _client(tmp_path, monkeypatch, lambda: {"n": 1})
    svc, jobs = app.state.service, app.state.jobs
    state = {}
    app_mod._health_sample(svc, jobs, state)                       # nothing happened: a quiet sample
    rows = svc.store().log_read(health.SAMPLES)
    assert rows[-1]["quiet"] and state["at"]
    c.get("/api/capabilities")                                     # a request: the next sample measures
    app_mod._health_sample(svc, jobs, state)
    row = svc.store().log_read(health.SAMPLES)[-1]
    assert not row.get("quiet") and "/api/capabilities" in " ".join(row["routes"])
    import threading
    gate = threading.Event()
    j = jobs.start("hear", lambda job: (gate.wait(5), None)[1])
    gate.set()
    for _ in range(100):
        if svc.store().log_count(health.JOBS):
            break
        import time
        time.sleep(0.05)
    io = svc.store().log_read(health.JOBS)[-1]
    assert io["kind"] == "hear" and io["elapsed_s"] >= 0 and "overlapped" in io
    svc._log_gains({"sound_gain": 0.04, "cv": {"exact": 0.9}})
    rep = c.get("/api/perf/report").json()                         # none saved yet: built now
    assert rep["samples"] >= 2 and any(x["kind"] == "hear" for x in rep["jobs"])
    assert rep["jobs"][0].get("gain_pts") == 4.0 or any(x.get("gain_pts") == 4.0 for x in rep["jobs"])
    assert svc.store().map_get("health_report", "week")["samples"] == rep["samples"]
    names = [s.name for s in app_mod._scheduler_steps(svc, jobs, {"last": 0.0}, lambda j: None)]
    assert "health-sample" in names and "health-report" in names
