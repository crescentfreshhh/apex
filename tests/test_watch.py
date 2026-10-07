"""The folder watch: new downloads are ingested on their own — once settled,
in one batch after the folder goes quiet, scanning only the watched folder."""

import pytest

pytest.importorskip("fastapi")

from fakestash import FakeStash  # noqa: E402
from peaks.config import Config  # noqa: E402

SETTLE, QUIET = 120, 180


@pytest.fixture
def stash():
    return FakeStash({"1": {"rating100": 100, "o_counter": 18}})


@pytest.fixture
def svc(tmp_path, stash, monkeypatch):
    import peaks.web.service as svc_mod

    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "INGEST_POLL", 0.0)
    monkeypatch.setattr(svc_mod.Service, "run_embed", lambda self, job=None, **kw: {"embedded": 0})
    s = svc_mod.Service(cfg)
    return s


@pytest.fixture
def folder(tmp_path, stash, svc):
    d = tmp_path / "data" / "sysbackup" / "norating"
    d.mkdir(parents=True)
    (d / "old.mp4").write_bytes(b"o" * 10)
    stash.libraries = [str(tmp_path / "data")]
    svc.save_watch_settings(on=True, paths=[str(d)], settle=SETTLE, quiet=QUIET)
    assert svc.watch_tick(now=0) is None          # first look: baseline, nothing ingested
    return d


def test_settles_then_waits_for_quiet_then_one_batch(svc, folder):
    (folder / "a.mp4").write_bytes(b"a" * 5)
    assert svc.watch_tick(now=60) is None          # just seen
    (folder / "a.mp4").write_bytes(b"a" * 50)      # still growing
    assert svc.watch_tick(now=120) is None
    assert svc.watch_tick(now=240) is None         # settled (since 120) but not quiet yet
    (folder / "sub").mkdir()
    (folder / "sub" / "b.mkv").write_bytes(b"b")
    assert svc.watch_tick(now=300) is None         # b arrives: the batch keeps waiting
    assert svc.watch_tick(now=420) is None         # b settled, quiet since 300 only 120 s
    go = svc.watch_tick(now=480)
    assert go and go["paths"] == [str(folder)]
    assert sorted(go["files"]) == sorted([str(folder / "a.mp4"), str(folder / "sub" / "b.mkv")])
    svc.watch_started(go["files"], now=480)
    assert svc.watch_tick(now=1000) is None        # handled: never again
    assert svc.watch_status()["last_run"]["files"] == 2


def test_temp_hidden_and_non_video_files_are_ignored(svc, folder):
    for n in ("c.mp4.part", ".c.mp4.Xy12zz", "c.!qB", "c.nfo", "c.jpg", "c.mkv.crdownload"):
        (folder / n).write_bytes(b"t")
    (folder / ".hidden").mkdir()
    (folder / ".hidden" / "d.mp4").write_bytes(b"d")
    assert svc.watch_tick(now=60) is None
    assert svc.watch_tick(now=10_000) is None
    assert svc.watch_status()["settled"] == 0


def test_restart_does_not_retrigger_and_new_folder_rebaselines(svc, folder, tmp_path):
    (folder / "a.mp4").write_bytes(b"a")
    svc.watch_tick(now=60)
    go = svc.watch_tick(now=60 + SETTLE + QUIET)
    svc.watch_started(go["files"])
    svc.__dict__.pop("_watch_mem", None)           # a restart re-reads the state file
    assert svc.watch_tick(now=5_000) is None
    other = tmp_path / "data" / "other"
    other.mkdir()
    (other / "x.mp4").write_bytes(b"x")
    svc.save_watch_settings(paths=[str(other)])
    assert svc.watch_tick(now=6_000) is None       # new folder: baseline
    assert svc.watch_tick(now=9_000) is None


def test_check_now_skips_the_quiet_wait_not_the_settle(svc, folder):
    (folder / "a.mp4").write_bytes(b"a")
    assert svc.watch_tick(now=60, force=True) is None             # not settled yet
    assert svc.watch_tick(now=60 + SETTLE, force=True)["files"] == [str(folder / "a.mp4")]


def test_off_or_no_folder_does_nothing(svc, folder):
    svc.save_watch_settings(on=False)
    (folder / "a.mp4").write_bytes(b"a")
    assert svc.watch_tick(now=60) is None and svc.watch_tick(now=10_000) is None
    svc.save_watch_settings(on=True)               # back on: what's there now is the baseline
    assert svc.watch_tick(now=10_060) is None
    assert svc.watch_tick(now=20_000) is None


def test_warnings_for_missing_or_uncovered_folders(svc, stash, tmp_path):
    stash.libraries = [str(tmp_path / "data")]
    (tmp_path / "data").mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    w = svc.watch_warnings([str(tmp_path / "nope"), str(outside), str(tmp_path / "data")])
    assert len(w) == 2 and "can't see" in w[0] and "isn't inside any Stash library" in w[1]


def test_watched_ingest_scans_only_the_folder(svc, stash, folder):
    stash.arriving = {"9": {"rating100": None}}
    out = svc.run_ingest(paths=[str(folder)], trigger="watch")
    scans = [c[1] for c in stash.calls if c[0] == "scan"]
    assert scans[-1]["paths"] == [str(folder)]
    assert out["trigger"] == "watch" and out["stages"]["scan"] == "1 new scene(s)"
    assert svc.last_ingest()["trigger"] == "watch"
    stash.no_fields = {("ScanMetadataInput", "paths")}                  # older Stash: full scan
    svc.run_ingest(paths=[str(folder)], trigger="watch")
    assert "paths" not in [c[1] for c in stash.calls if c[0] == "scan"][-1]

def test_scheduler_waits_for_a_running_job_then_starts_one_ingest(svc, folder, monkeypatch):
    import peaks.web.app as app_mod

    running = {"sync"}
    started = []

    class Jobs:
        def running(self, k):
            return object() if k in running else None

        def start(self, kind, fn):
            started.append(kind)
            return type("J", (), {"as_dict": lambda self: {"kind": kind}})()

    (folder / "a.mp4").write_bytes(b"a")
    import time as _t
    t0 = _t.time()
    monkeypatch.setattr(_t, "time", lambda: t0)
    app_mod._watch_go(svc, Jobs())
    monkeypatch.setattr(_t, "time", lambda: t0 + SETTLE + QUIET + 1)
    assert app_mod._watch_go(svc, Jobs()) is None and started == []      # a Sync is running
    running.clear()
    assert app_mod._watch_go(svc, Jobs()) is not None and started == ["ingest"]
    assert app_mod._watch_go(svc, Jobs()) is None and started == ["ingest"]


def test_api_save_status_dirs_and_check(svc, folder, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    monkeypatch.setenv("PEAKS_CLEANUP_ROOT", str(tmp_path / "data"))
    api = TestClient(app_mod.create_app(svc.cfg))
    w = api.get("/api/watch").json()
    assert w["watch_on"] and w["watch_paths"] == [str(folder)] and w["warnings"] == []
    assert api.post("/api/watch", json={"paths": "nope"}).status_code == 400
    w = api.post("/api/watch", json={"on": True, "paths": [str(folder), "/nowhere"]}).json()
    assert any("can't see /nowhere" in x for x in w["warnings"])
    d = api.get("/api/watch/dirs").json()
    assert d["dirs"] == [str(tmp_path / "data" / "sysbackup")]
    r = api.post("/api/watch", params={"action": "check"}).json()
    assert r["started"] is None


# --- the scheduler: one heavy job at a time, owed embeds, retries ------------------------

class _Jobs:
    def __init__(self, running=()):
        self.live, self.started = set(running), []

    def running(self, k):
        return object() if k in self.live else None

    def start(self, kind, fn):
        if kind in self.live:
            raise RuntimeError("busy")
        self.started.append((kind, fn))
        return type("J", (), {"as_dict": lambda self: {"kind": kind}})()


def test_scheduled_embed_waits_for_an_ingest_then_starts(svc):
    import peaks.web.app as app_mod

    state = {"last": 0.0}
    jobs = _Jobs(running={"ingest"})
    assert app_mod._scheduled_embeds(svc, jobs, state, 3600, "run") is None
    assert state["last"] == 0.0 and jobs.started == []      # still due, not skipped
    jobs.live.clear()
    assert app_mod._scheduled_embeds(svc, jobs, state, 3600, "run") == "scheduled"
    assert jobs.started[0][0] == "embed" and state["last"] > 0


def test_owed_embeds_go_before_the_scheduled_pass(svc):
    import peaks.web.app as app_mod

    svc.add_embed_followup(["7"])
    jobs = _Jobs()
    assert app_mod._scheduled_embeds(svc, jobs, {"last": 0.0}, 0, "run") == "followup"
    assert jobs.started[0][1] == svc.run_embed_followup


def test_watch_waits_for_a_running_embed(svc, folder, monkeypatch):
    import time as _t

    import peaks.web.app as app_mod

    (folder / "a.mp4").write_bytes(b"a")
    t0 = _t.time()
    monkeypatch.setattr(_t, "time", lambda: t0)
    app_mod._watch_go(svc, _Jobs())
    monkeypatch.setattr(_t, "time", lambda: t0 + SETTLE + QUIET + 1)
    jobs = _Jobs(running={"embed"})
    assert app_mod._watch_go(svc, jobs) is None and jobs.started == []
    jobs.live.clear()
    assert app_mod._watch_go(svc, jobs) is not None and jobs.started[0][0] == "ingest"


def test_a_failed_auto_ingest_is_retried_after_ten_minutes(svc, folder, monkeypatch):
    import time as _t

    import peaks.web.app as app_mod
    import peaks.web.service as svc_mod

    (folder / "a.mp4").write_bytes(b"a")
    t = [_t.time()]
    monkeypatch.setattr(_t, "time", lambda: t[0])
    app_mod._watch_go(svc, _Jobs())
    t[0] += SETTLE + QUIET + 1
    jobs = _Jobs()
    app_mod._watch_go(svc, jobs)
    _, fn = jobs.started[0]

    def boom(self, job=None, **kw):
        raise RuntimeError("scan failed: Stash is down")

    monkeypatch.setattr(svc_mod.Service, "run_ingest", boom)
    with pytest.raises(RuntimeError):
        fn(None)
    st = svc.watch_status()
    assert "Stash is down" in st["last_error"]["error"] and st["retry_after"] == pytest.approx(t[0] + 600)
    t[0] += 120
    assert app_mod._watch_go(svc, _Jobs()) is None             # not before the retry time
    t[0] += 600
    jobs = _Jobs()
    assert app_mod._watch_go(svc, jobs) is not None             # the same files, again
    monkeypatch.setattr(svc_mod.Service, "run_ingest", lambda self, job=None, **kw: {"new": 1})
    jobs.started[0][1](None)
    assert svc.watch_status()["last_error"] is None


def test_stopping_an_auto_ingest_does_not_retry(svc, folder, monkeypatch):
    import time as _t

    import peaks.web.app as app_mod
    import peaks.web.service as svc_mod

    (folder / "a.mp4").write_bytes(b"a")
    t = [_t.time()]
    monkeypatch.setattr(_t, "time", lambda: t[0])
    app_mod._watch_go(svc, _Jobs())
    t[0] += SETTLE + QUIET + 1
    jobs = _Jobs()
    app_mod._watch_go(svc, jobs)

    def stopped(self, job=None, **kw):
        raise InterruptedError("scan stopped")

    monkeypatch.setattr(svc_mod.Service, "run_ingest", stopped)
    with pytest.raises(InterruptedError):
        jobs.started[0][1](None)
    assert svc.watch_status()["last_error"] is None
    t[0] += 10_000
    assert app_mod._watch_go(svc, _Jobs()) is None


# --- regression: grading (the renamer follower) must not break the folder watch -----------

def test_watch_still_works_after_a_grade(svc, folder, stash):
    import time as _t

    import peaks.web.app as app_mod

    svc.grade_scene("1", "exceptionnelle")                  # starts the renamer follower
    (folder / "after-grade.mp4").write_bytes(b"x")
    assert svc.watch_tick(now=_t.time()) is None            # no TypeError
    go = svc.watch_tick(now=_t.time() + SETTLE + QUIET + 5)
    assert go and go["files"] == [str(folder / "after-grade.mp4")]
    app_mod._watch_go(svc, _Jobs(), force=True)              # the scheduler path, no TypeError either


def test_no_instance_attribute_shadows_a_method(svc, folder, stash, tmp_path, monkeypatch):
    """Mixins share one object: an attribute stored under a method's name breaks
    that method for good (what silently stopped the folder watch)."""
    monkeypatch.setenv("PEAKS_SEEDBOX_DIR", str(tmp_path / "nope"))
    svc.grade_scene("1", "exceptionnelle")
    svc.watch_tick()
    svc.watch_status()
    svc.keep_as_is("1")
    svc.dupe_queue_status()
    svc.seedbox_status()
    svc.copies_status()
    svc.cleanup_status()
    svc.backup_status()
    cls = type(svc)
    clash = [n for n in vars(svc) if callable(getattr(cls, n, None)) or isinstance(getattr(cls, n, None), property)]
    assert clash == []


def test_a_crashing_check_is_reported_not_swallowed(svc, folder, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    broken = {"on": True}
    real = type(svc).watch_tick

    def boom(self, *a, **k):
        if broken["on"]:
            raise TypeError("'_thread.lock' object is not callable")
        return real(self, *a, **k)

    monkeypatch.setattr(type(svc), "watch_tick", boom)
    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(app_mod.create_app(svc.cfg))
    r = api.post("/api/watch", params={"action": "check"})
    assert r.status_code == 500 and "folder check failed" in r.json()["detail"]
    st = api.get("/api/watch").json()
    assert "not callable" in st["check_error"]["error"]
    broken["on"] = False
    svc.watch_tick()                                        # a good check clears it
    assert svc.watch_status()["check_error"] is None
