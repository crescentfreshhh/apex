"""The scheduler sleeps until something happens: an event (a job finishing, a
library write, the pause, the index or the scene list changing) runs every step
once; otherwise only a step whose own clock deadline arrives runs (the weekly
backup, the overnight measure, the seedbox times, the pause ending…). An idle
server touches nothing: no file stats, no state reads, no Stash requests."""

import builtins
import os
import pathlib
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from peaks.web.scheduler import Scheduler, Step  # noqa: E402


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def test_events_run_every_step_and_deadlines_only_their_own():
    clock, ran = Clock(1000.0), []
    daily = lambda now: 1500.0 if now < 1500 else 1500.0 + 86400      # noqa: E731 — like "every day at 05:00"
    steps = [Step("a", lambda: ran.append("a")),
             Step("b", lambda: ran.append("b"), daily),
             Step("lib", lambda: ran.append("lib"), every=1800, on_event=False)]
    s = Scheduler(steps, clock=clock)
    assert s.run_due(event=True) == ["a", "b"]                 # an event: every event step, once
    for t in range(1001, 1500, 7):
        clock.t = t
        assert s.run_due() == []                               # nothing due: nothing runs
    clock.t = 1500
    assert s.run_due() == ["b"]                                # its deadline: only it
    clock.t = 1700
    s.run_due(event=True)                                      # an event in between doesn't push the repeat back
    clock.t = 1000 + 1800
    assert s.run_due() == ["lib"]                              # the clock-only repeat, on time
    assert s.next["lib"] == 1000 + 3600 and s.next["b"] == 1500 + 86400


def test_the_pause_stops_the_rest_and_its_end_is_a_deadline():
    clock, ran, held = Clock(0.0), [], {"until": 600.0}
    steps = [Step("hold", lambda: clock.t < held["until"], lambda now: held["until"] if now < held["until"] else None,
                  stops_rest=True),
             Step("work", lambda: ran.append(clock.t))]
    s = Scheduler(steps, clock=clock)
    s.run_due(event=True)
    assert ran == [] and s.next["hold"] == 600.0               # paused: nothing else; wakes when it ends
    clock.t = 600.0
    s.run_due(event=True)
    assert ran == [600.0]


def test_wake_sets_the_event_and_counts_why():
    s = Scheduler([], clock=Clock(0.0))
    s.wake("job finished")
    assert s._ev.is_set() and s.reasons["job finished"] == 1
    assert s.sleep_for() == Scheduler.MAX_SLEEP                # no deadlines: a long heartbeat only


def _idle_world(tmp_path, monkeypatch):
    """A real Service and JobManager over the fake library, Wednesday 14:00."""
    from test_catalogue import _triage_svc

    import peaks.web.app as app_mod
    from peaks.web.jobs import JobManager

    monkeypatch.setenv("PEAKS_CLEANUP_ROOT", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    jobs = JobManager(history_path=tmp_path / "jh.jsonl")
    svc.catalogue(limit=5)
    svc.index()
    for k in ("zoom", "hear", "motion", "faces"):              # the senses are caught up
        monkeypatch.setattr(svc, f"{k}_due", lambda: False)
    base = time.mktime(time.strptime("2026-11-04 14:00:00", "%Y-%m-%d %H:%M:%S"))
    clock = Clock(base)
    monkeypatch.setattr(time, "time", clock)
    state = {"last": 0.0}
    emb = lambda j: app_mod._scheduled_embed_pass(svc, j)      # noqa: E731
    sch = Scheduler(app_mod._scheduler_steps(svc, jobs, state, emb), clock=clock)
    jobs.on_finish = lambda j: sch.wake("job finished")
    svc._on_change = sch.wake
    return svc, stash, jobs, sch, clock


def test_an_idle_server_touches_nothing_for_ten_minutes(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    svc, stash, jobs, sch, clock = _idle_world(tmp_path, monkeypatch)
    sch.run_due(event=True)                                    # start-up: every step looks once
    assert not [j for j in jobs.list() if j.status == "running"]
    touched = []

    def spy(name, fn):
        def w(*a, **k):
            touched.append(name)
            return fn(*a, **k)
        return w
    from peaks.store import Store

    monkeypatch.setattr(pathlib.Path, "stat", spy("Path.stat", pathlib.Path.stat))
    monkeypatch.setattr(os, "stat", spy("os.stat", os.stat))
    monkeypatch.setattr(os, "scandir", spy("os.scandir", os.scandir))
    monkeypatch.setattr(builtins, "open", spy("open", builtins.open))
    monkeypatch.setattr(Store, "map_all", spy("Store.map_all", Store.map_all))
    monkeypatch.setattr(Store, "log_read", spy("Store.log_read", Store.log_read))
    for n in [n for n in dir(stash) if not n.startswith("_") and callable(getattr(stash, n))]:
        monkeypatch.setattr(stash, n, spy("stash." + n, getattr(stash, n)))
    for _ in range(600):                                       # ten minutes, second by second
        clock.t += 1
        assert sch.run_due() == []
    assert touched == []


def test_a_job_finishing_wakes_it_and_what_waited_starts(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    import threading

    svc, stash, jobs, sch, clock = _idle_world(tmp_path, monkeypatch)
    sch.run_due(event=True)
    gate = threading.Event()
    j = jobs.start("embed", lambda job: (gate.wait(5), None)[1])
    monkeypatch.setattr(type(svc), "backup_due", lambda self, now=None: True)
    sch._ev.clear()
    sch.run_due(event=True)
    assert jobs.running("backup") is None                      # waits for the embed
    gate.set()
    j._thread.join(5) if hasattr(j, "_thread") else time.sleep(0.5)
    for _ in range(50):
        if sch._ev.is_set():
            break
        time.sleep(0.05)
    assert sch._ev.is_set() and sch.reasons["job finished"] >= 1
    sch.run_due(event=True)
    assert jobs.list()[0].kind in ("backup", "embed") and any(x.kind == "backup" for x in jobs.list())


def test_index_and_scene_list_changes_wake_it(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    svc, stash, jobs, sch, clock = _idle_world(tmp_path, monkeypatch)
    sch._ev.clear()
    svc.mark_index_stale()
    assert sch.reasons["index stale"] == 1
    svc.invalidate_index()
    assert sch.reasons["index dropped"] == 1
    v = svc.__dict__.get("_cat_ver", 0)
    svc._cat_changed()
    assert sch.reasons["catalogue"] == 1 and svc._cat_ver == v + 1


def test_clock_deadlines_are_read_from_memory(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    svc, stash, jobs, sch, clock = _idle_world(tmp_path, monkeypatch)
    now = clock.t                                              # Wednesday 14:00
    nb = svc.backup_next(now)                                  # default: Sunday 04:10
    lt = time.localtime(nb)
    assert lt.tm_wday == 6 and (lt.tm_hour, lt.tm_min) == (4, 10) and 0 < nb - now < 7 * 86400
    sch.run_due(event=True)
    assert sch.next["measure"] is not None and time.localtime(sch.next["measure"]).tm_hour == svc.cfg.modeling.taste_measure_hour
    assert sch.next["library"] == pytest.approx(now + 1800, abs=1)     # anchored on start-up, not pushed back
    assert sch.next["watch"] is None and sch.next["upgrades"] is None and sch.next["seed-pull"] is None


def test_the_folder_watch_skips_unchanged_folders_and_writes_only_on_change(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import peaks.web.service as svc_mod
    from fakestash import FakeStash
    from peaks.config import Config

    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    stash = FakeStash({"1": {"rating100": 100, "o_counter": 18}})
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    svc = svc_mod.Service(cfg)
    d = tmp_path / "data" / "in"
    (d / "deep" / "er").mkdir(parents=True)
    for i in range(5):
        (d / "deep" / "er" / f"{i}.mp4").write_bytes(b"x")
    svc.save_watch_settings(on=True, paths=[str(d)], settle=120, quiet=180)
    svc.watch_tick(now=0)
    saves, scans = [], []
    real_save, real_scan = type(svc)._watch_save, os.scandir
    monkeypatch.setattr(type(svc), "_watch_save", lambda self: saves.append(1) or real_save(self))
    monkeypatch.setattr(os, "scandir", lambda p: scans.append(str(p)) or real_scan(p))
    assert svc.watch_tick(now=60) is None
    assert saves == [] and scans == []                         # nothing changed: no folder listed, no write
    (d / "deep" / "new.mkv").write_bytes(b"n")
    assert svc.watch_tick(now=120) is None
    assert scans == [str(d / "deep")] and saves == [1]         # only the folder that changed; one write
    assert str(d / "deep" / "new.mkv") in svc._watch_state()["seen"]


def test_zoom_remembers_files_it_cannot_read(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from test_zoom import _svc

    svc, stash, saved, sampler = _svc(tmp_path, monkeypatch)
    todo = svc.zoom_todo()
    bad = todo[0]["scene_id"]
    real = svc.zoom_scene

    def flaky(row, *a, **k):
        if row["scene_id"] == bad:
            raise RuntimeError("moov atom not found")
        return real(row, *a, **k)
    svc.zoom_scene = flaky
    svc.run_zoom(limit=3)
    assert bad in svc.sense_failures("zoom")
    assert bad not in [r["scene_id"] for r in svc.zoom_todo()]   # not tried again on the next runs
    assert not getattr(svc, "_zoom_error", None)


def test_zoom_backs_off_when_every_file_fails(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from test_zoom import _svc

    svc, stash, saved, sampler = _svc(tmp_path, monkeypatch)

    def broken(row, *a, **k):
        raise RuntimeError("CUDA error")
    svc.zoom_scene = broken
    out = svc.run_zoom(limit=6)
    assert out["failed"] == 6 and svc.sense_failures("zoom") == {}   # the setup, not the files: none parked
    assert "every scene failed" in svc._zoom_error
