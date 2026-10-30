"""Heavy jobs take turns: library work (an ingest, an embed pass) goes first —
background work that can carry on later is paused, and started again after."""

import threading
import time

import pytest

pytest.importorskip("fastapi")

import peaks.web.app as app_mod  # noqa: E402
from peaks.web.jobs import JobManager  # noqa: E402


def _bg(gate: threading.Event, seen: list):
    def target(job):
        while not job.cancelled and not gate.is_set():
            time.sleep(0.01)
        seen.append("stopped" if job.cancelled else "finished")
        return {"stopped": True} if job.cancelled else {}
    return target


def _wait(pred, secs=3.0):
    t0 = time.time()
    while time.time() - t0 < secs:
        if pred():
            return True
        time.sleep(0.01)
    return False


class FakeService:
    def __init__(self):
        self.labs = []

    def run_lab(self, cand, job=None):
        self.labs.append(cand)
        return {}


def test_an_ingest_pauses_a_trial_which_resumes_after(tmp_path):
    jm = JobManager(history_path=tmp_path / "h.jsonl")
    svc = FakeService()
    gate, seen = threading.Event(), []
    lab = jm.start("lab", _bg(gate, seen))
    lab.args = {"cand": "dinov3_vitl16"}
    assert app_mod._make_way(jm, "Ingest") == ["lab"]
    assert _wait(lambda: lab.status != "running") and lab.status == "paused" and seen == ["stopped"]
    assert jm.paused["lab"]["args"] == {"cand": "dinov3_vitl16"} and jm.paused["lab"]["for"] == "Ingest"
    assert any("paused for Ingest" in m for _, m in lab.lines())
    hist = jm.history()
    assert hist[0]["status"] == "paused" and hist[0]["paused_for"] == "Ingest"

    ingest_gate = threading.Event()
    jm.start("ingest", _bg(ingest_gate, []))
    assert app_mod._resume_paused(svc, jm) == []                 # not while the ingest runs
    ingest_gate.set()
    assert _wait(lambda: jm.running("ingest") is None)
    assert app_mod._resume_paused(svc, jm) == ["lab"]            # its turn again
    assert _wait(lambda: svc.labs == ["dinov3_vitl16"]) and not jm.paused


def test_background_senses_make_way_and_their_timers_clear(tmp_path):
    jm = JobManager()
    gate, seen = threading.Event(), []
    jm.start("hear", _bg(gate, seen))
    app_mod._hear_next["at"] = time.time() + 3600
    app_mod._make_way(jm, "Embedding")
    assert _wait(lambda: jm.running("hear") is None) and "hear" in jm.paused
    assert app_mod._resume_paused(FakeService(), jm) == ["hear"]
    assert app_mod._hear_next["at"] == 0.0 and not jm.paused


def test_nothing_pauses_without_a_running_background_job():
    jm = JobManager()
    assert app_mod._make_way(jm, "Ingest") == [] and jm.paused == {}


def test_a_trial_asked_for_during_an_ingest_is_queued_and_shown(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from peaks.config import Config

    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("PEAKS_SCHEDULER", "0")
    monkeypatch.setenv("PEAKS_WARM_ON_START", "0")
    made = {}
    real = app_mod.JobManager
    monkeypatch.setattr(app_mod, "JobManager", lambda *a, **k: made.setdefault("jm", real(*a, **k)))
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    c = TestClient(app_mod.create_app(cfg))
    jm = made["jm"]
    gate = threading.Event()
    jm.start("ingest", _bg(gate, []))
    r = c.post("/api/lab/dinov3_vitl16/run").json()
    assert r == {"queued": True, "after": "Ingest"} and jm.running("lab") is None
    feed = c.get("/api/activity/feed").json()
    assert feed["waiting"][0]["kind"] == "lab" and feed["waiting"][0]["after"] == "Ingest"
    assert feed["waiting"][0]["detail"] == "DINOv3 ViT-L/16"
    gate.set()
