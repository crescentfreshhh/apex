"""The central Activity log: job timing (percent, time per item, time left),
timestamped log lines, the merged feed, and finished jobs remembered across
restarts."""

import json
import threading
import time

import pytest

from peaks.web.jobs import HISTORY_KEEP, Job, JobManager


def test_timing_from_done_total_and_its_own_eta():
    j = Job(id="hear-1", kind="hear")
    j.started -= 100
    j.progress = {"done": 20, "total": 100}
    t = j.timing()
    assert t["pct"] == 0.2 and t["done"] == 20 and t["elapsed_s"] >= 100
    assert t["per_item_s"] == pytest.approx(5.0, rel=0.05)                # 20 in 100 s: the average so far
    assert t["eta_s"] == pytest.approx(400, rel=0.05)
    j._samples.clear()
    now = time.time()
    j._samples.extend([(now - 60, 10.0), (now - 1, 40.0)])                 # lately: 30 in ~59 s
    j.progress = {"done": 40, "total": 100}
    t = j.timing()
    assert t["rate_per_min"] == pytest.approx(30.5, rel=0.1) and t["eta_s"] == pytest.approx(118, rel=0.1)
    j.progress = {"done": 40, "total": 100, "eta_s": 77}                  # the job knows better
    assert j.timing()["eta_s"] == 77


def test_timing_for_a_staged_job_and_at_the_very_start():
    j = Job(id="ingest-1", kind="ingest")
    assert j.timing()["pct"] is None and j.timing()["eta_s"] is None       # nothing to go on yet
    j.started -= 50
    j.progress = {"stage": "identify", "pct": 0.25}
    t = j.timing()
    assert t["stage"] == "identify" and t["pct"] == 0.25 and t["eta_s"] == pytest.approx(150, rel=0.05)
    assert t["per_item_s"] is None                                         # no items, no per-item time
    j.status, j.finished = "done", time.time()
    assert j.timing()["eta_s"] is None


def test_log_lines_carry_their_time_and_the_old_shape_stays():
    j = Job(id="embed-1", kind="embed")
    j.log("first")
    t0 = time.time()
    time.sleep(0.01)
    j.log("second")
    assert [m for _, m in j.lines()] == ["first", "second"] and [m for _, m in j.lines(t0)] == ["second"]
    assert j.as_dict()["log"] == ["first", "second"] and "timing" in j.as_dict()


def _run(jm, kind, steps, fail=False, gap=0.02):
    def target(job):
        for i in range(steps):
            job.progress = {"done": i + 1, "total": steps}
            job.log(f"{kind} step {i + 1}")
            time.sleep(gap)
        if fail:
            raise RuntimeError("disk full")
        return {"heard": steps, "failed": 0}
    return jm.start(kind, target)


def _wait(jm):
    for _ in range(300):
        if all(j.status != "running" for j in jm.list()):
            return
        time.sleep(0.02)


def test_finished_jobs_are_remembered_across_restarts_and_kept_short(tmp_path):
    path = tmp_path / "job_history.jsonl"
    jm = JobManager(history_path=path)
    _run(jm, "hear", 3)
    _run(jm, "motion", 2, fail=True)
    _wait(jm)
    again = JobManager(history_path=path).history()                        # a restart
    by = {h["kind"]: h for h in again}
    assert by["hear"]["status"] == "done" and by["hear"]["done"] == 3 and "heard 3" in by["hear"]["result"]
    assert by["motion"]["status"] == "error" and "disk full" in by["motion"]["error"]
    path.write_text("\n".join(json.dumps({"id": f"x-{i}", "kind": "x", "started": i, "finished": i})
                              for i in range(HISTORY_KEEP + 30)) + "\n")
    jm2 = JobManager(history_path=path)
    _run(jm2, "sync", 1)
    _wait(jm2)
    assert len(path.read_text().splitlines()) == HISTORY_KEEP


def test_the_feed_merges_every_jobs_lines_in_time_order(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod
    from peaks.config import Config

    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("PEAKS_SCHEDULER", "0")
    monkeypatch.setenv("PEAKS_WARM_ON_START", "0")
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    made = {}
    real = app_mod.JobManager

    def capture(*a, **k):
        made["jm"] = real(*a, **k)
        return made["jm"]
    monkeypatch.setattr(app_mod, "JobManager", capture)
    c = TestClient(app_mod.create_app(cfg))
    jm = made["jm"]
    gate = threading.Event()

    def slow(job):
        job.progress = {"done": 1, "total": 4}
        job.log("hear: scene 1")
        gate.wait(5)
        return {}
    jm.start("hear", slow)
    _run(jm, "zoom", 3)
    time.sleep(0.2)
    d = c.get("/api/activity/feed").json()
    assert [r["kind"] for r in d["running"]] == ["hear"] and d["running"][0]["label"] == "Hearing"
    assert d["running"][0]["pct"] == 0.25 and d["running"][0]["family"] == "hear"
    ts = [x["t"] for x in d["lines"]]
    assert ts == sorted(ts) and {x["kind"] for x in d["lines"]} == {"hear", "zoom"}
    assert any(f["kind"] == "zoom" and f["label"] == "Moment zoom" for f in d["finished"])
    later = c.get("/api/activity/feed", params={"since": d["lines"][-1]["t"]}).json()
    assert later["lines"] == []
    gate.set()
    _wait(jm)
    assert (tmp_path / "job_history.jsonl").exists()


def test_a_job_asked_to_pause_says_so_while_it_winds_down():
    j = Job(id="lab-1", kind="lab")
    j.progress = {"done": 112, "total": 400}
    j.paused_for = "Ingest"
    j.request_cancel()
    t = j.timing()
    assert t["stage"] == "pausing for Ingest" and t["stopping"] and t["eta_s"] is None
