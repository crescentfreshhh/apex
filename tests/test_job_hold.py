"""The top bar's "Pause jobs for an hour": background senses wait on the waiting
list, Ingest / Embed / Fix stop and start again afterwards, things you started that
change Stash finish, and nothing starts on its own until the pause ends (or you
carry on)."""

import threading
import time

import pytest

pytest.importorskip("fastapi")

import peaks.web.app as app_mod  # noqa: E402
from peaks.web.jobs import JobManager  # noqa: E402


def _long(job):
    while not job.cancelled:
        time.sleep(0.01)
    raise InterruptedError("stopped")


def _wait(pred, s=5.0):
    end = time.time() + s
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


class Svc:
    def __init__(self):
        self.started = []

    def run_ingest(self, job, embed_busy=None):
        self.started.append("ingest")

    def run_embed(self, job):
        self.started.append("embed")

    def run_fix(self, job):
        self.started.append("fix")

    def __getattr__(self, name):            # the scheduler's other checks: nothing due
        if name.endswith("_due") or name in ("embed_followup", "seed_ingest_waiting"):
            return lambda *a, **k: False
        if name == "schedule_settings":
            return lambda: {"embed_seconds": 0}
        raise AttributeError(name)


def test_pausing_sends_senses_to_the_waiting_list_and_stops_library_work(tmp_path):
    jobs, svc = JobManager(), Svc()
    m = jobs.start("motion", _long)
    e = jobs.start("embed", _long)
    g = jobs.start("library", _long)                     # a bulk grade you started: it finishes
    paused = app_mod._hold_jobs(svc, jobs, 60)
    assert set(paused) == {"motion", "embed"} and jobs.held()
    assert jobs.paused_snapshot()["motion"]["for"] == app_mod.HOLD_FOR
    assert jobs.held_restart == ["embed"]
    assert _wait(lambda: m.status == "paused" and e.status == "paused")
    assert not g.cancelled
    g.request_cancel()


def test_nothing_starts_while_paused_then_what_stopped_comes_back(tmp_path):
    jobs, svc = JobManager(), Svc()
    i = jobs.start("ingest", _long)
    app_mod._hold_jobs(svc, jobs, 60)
    assert _wait(lambda: i.status == "paused")
    assert app_mod._hold_tick(svc, jobs) is True                   # held: the tick starts nothing
    app_mod._scheduler_once(svc, jobs, {"last": 0.0}, lambda j: None)
    assert svc.started == [] and not app_mod._resume_paused(svc, jobs)
    jobs.hold_until = time.time() - 1                               # the hour is up
    assert app_mod._hold_tick(svc, jobs) is False
    assert _wait(lambda: svc.started == ["ingest"]) and jobs.held_restart == []


def test_the_api_counts_down_and_carrying_on_restarts_at_once(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from test_snappy import _client

    c, app = _client(tmp_path, monkeypatch, lambda: {"n": 1})
    assert c.get("/api/hold").json() == {"held": False, "until": None, "left_s": 0}
    r = c.post("/api/hold?minutes=60").json()
    assert r["held"] and 3590 <= r["left_s"] <= 3600 and r["paused"] == []
    assert c.get("/api/hold").json()["held"]
    off = c.delete("/api/hold").json()
    assert off["held"] is False and off["started"] == []
