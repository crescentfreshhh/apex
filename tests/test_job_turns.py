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
    assert app_mod._hear_next["at"] == 0.0 and jm.paused["hear"]["resuming"]       # its turn; still holds the queue
    app_mod._hear_next["at"] = 99.0
    app_mod._resume_paused(FakeService(), jm)
    assert app_mod._hear_next["at"] == 99.0                                        # zeroed once, not every tick


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


def test_the_background_lane_runs_one_job_at_a_time_in_order(tmp_path):
    jm = JobManager()
    svc = FakeService()
    gates = {k: threading.Event() for k in ("lab", "hear")}
    lab = jm.start("lab", _bg(gates["lab"], []))
    lab.args = {"cand": "dinov3_vitl16"}
    jm.start("hear", _bg(gates["hear"], []))                     # (how it used to look)
    app_mod._make_way(jm, "Ingest")
    assert _wait(lambda: jm.running("lab") is None and jm.running("hear") is None)
    assert set(jm.paused) == {"lab", "hear"}
    # after the ingest: Hearing's own starter must not jump ahead of the trial
    app_mod._hear_next["at"] = 0.0
    assert app_mod._bg_turn_taken(jm, "hear")                     # the trial waits ahead of it
    assert app_mod._resume_paused(svc, jm) == ["lab"] and "hear" in jm.paused   # one at a time, trial first
    gate = threading.Event()
    jm.start("lab", _bg(gate, []))                                # (the resumed trial, running)
    assert app_mod._resume_paused(svc, jm) == []                  # nothing beside it
    assert app_mod._bg_turn_taken(jm, "hear") and app_mod._bg_turn_taken(jm, "picture")
    gate.set()
    assert _wait(lambda: jm.running("lab") is None)
    assert app_mod._resume_paused(svc, jm) == ["hear"] and jm.paused["hear"]["resuming"]
    assert not app_mod._bg_turn_taken(jm, "hear") and app_mod._bg_turn_taken(jm, "picture")   # picture can't jump it


def test_a_trial_asked_for_while_hearing_runs_goes_next(tmp_path, monkeypatch):
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
    jm.start("hear", _bg(gate, []))
    assert c.post("/api/lab/dinov3_vitl16/run").json() == {"queued": True, "after": "Hearing"}
    w = c.get("/api/activity/feed").json()["waiting"][0]
    assert w["kind"] == "lab" and w["queued"] and w["after"] == "Hearing"
    gate.set()


# --- one GPU lane for user-started work too -------------------------------------------------

def _app(tmp_path, monkeypatch):
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
    return c, made["jm"]


def test_heavy_work_waits_until_the_paused_job_has_really_stopped(tmp_path):
    jm = JobManager()
    gate, order = threading.Event(), []

    def slow_to_stop(job):
        while not job.cancelled:
            time.sleep(0.01)
        time.sleep(0.3)                                       # finishing its scene
        order.append("hear stopped")
        return {}
    jm.start("hear", slow_to_stop)

    def embed(job):
        order.append("embed ran")
        return {}
    jm.start("embed", app_mod._way_then(jm, "Embedding", embed))
    assert _wait(lambda: len(order) == 2, 3) and order == ["hear stopped", "embed ran"]
    assert "hear" in jm.paused


def test_a_sense_asked_for_while_the_lane_is_busy_is_queued_not_doubled(tmp_path, monkeypatch):
    c, jm = _app(tmp_path, monkeypatch)
    gate = threading.Event()
    jm.start("embed", _bg(gate, []))
    for path, kind, after in (("/api/hear", "hear", "Embedding"), ("/api/motion", "motion", "Embedding"),
                              ("/api/moments/zoom", "zoom", "Embedding")):
        assert c.post(path).json() == {"queued": True, "after": after}
        assert jm.running(kind) is None and jm.paused[kind]["args"]["manual"]
    gate.set()
    assert _wait(lambda: jm.running("embed") is None)


def test_no_model_switch_mid_trial_and_blockers_are_atomic(tmp_path, monkeypatch):
    from peaks.web.jobs import Busy

    c, jm = _app(tmp_path, monkeypatch)
    gate = threading.Event()
    jm.start("zoom", _bg(gate, []))
    r = c.post("/api/models", json={"dino_model": "dinov2_vitg14"})
    assert r.status_code == 409 and "Moment zoom" in r.json()["detail"]
    with pytest.raises(Busy):
        jm.start("hear", lambda j: {}, blockers=("zoom",))
    gate.set()


def test_a_stopped_ingest_is_a_stop_and_old_jobs_are_let_go():
    from peaks.web.jobs import KEEP_JOBS

    jm = JobManager()

    def ingest(job):
        while not job.cancelled:
            time.sleep(0.01)
        raise InterruptedError("stopped")
    j = jm.start("ingest", ingest)
    j.request_cancel()
    assert _wait(lambda: j.status != "running") and j.status == "cancelled"
    for i in range(KEEP_JOBS + 20):
        jm.start(f"k{i}", lambda job: {})
    assert _wait(lambda: all(x.status != "running" for x in jm.list()))
    jm.start("last", lambda job: {})
    assert len(jm.list()) <= KEEP_JOBS + 2
    assert "fix" in app_mod.LIBRARY_WORK


# --- your Stop sticks ------------------------------------------------------------------------

class DueService(FakeService):
    def index_loaded(self):
        return True

    def hear_due(self):
        return True

    def run_hear(self, job):
        while not job.cancelled:
            time.sleep(0.01)
        return {}


def test_a_stopped_sense_is_not_restarted_by_the_scheduler_today(tmp_path, monkeypatch):
    c, jm = _app(tmp_path, monkeypatch)
    svc = DueService()
    app_mod._hear_next["at"] = 0.0
    j = app_mod._start_hear(svc, jm)
    assert j is not None
    assert c.post(f"/api/jobs/{j.id}/cancel").status_code == 200
    assert _wait(lambda: jm.running("hear") is None) and j.status == "cancelled"
    app_mod._hear_next["at"] = 0.0
    assert app_mod._start_hear(svc, jm) is None and jm.stopped_by_you("hear")     # not on its own
    jm.stopped_until.pop("hear")
    assert app_mod._start_hear(svc, jm) is not None                               # tomorrow (or you) again
    jm.running("hear").request_cancel()


def test_stopping_a_job_that_was_pausing_cancels_its_resume_and_waiting_work_can_be_cancelled(tmp_path, monkeypatch):
    c, jm = _app(tmp_path, monkeypatch)
    gate = threading.Event()
    lab = jm.start("lab", _bg(gate, []), args={"cand": "dinov3_vitl16"})
    app_mod._make_way(jm, "Ingest")
    c.post(f"/api/jobs/{lab.id}/cancel")
    assert _wait(lambda: lab.status != "running") and lab.status == "cancelled" and "lab" not in jm.paused
    jm.pause_put("zoom", {"for": "Embedding", "args": {}, "at": 0})
    assert c.delete("/api/jobs/waiting/zoom").json() == {"ok": True} and "zoom" not in jm.paused
    assert jm.stopped_by_you("zoom") and c.delete("/api/jobs/waiting/zoom").status_code == 404
    jm.stopped_until.clear()


def test_switching_a_sense_off_stops_its_run(tmp_path, monkeypatch):
    c, jm = _app(tmp_path, monkeypatch)
    gate = threading.Event()
    jm.start("motion", _bg(gate, []))
    c.post("/api/motion/settings", json={"on": False})
    assert _wait(lambda: jm.running("motion") is None)
