"""The seedbox jobs Peaks took over from the Unraid scripts: pull (only new
files, a few at a time, perms + .done per file, runs.log / pull.failing as
before), purge (the script's rules, a dry run first, the one qBittorrent
write), the report (counts only — never a file name), the cutover guard and
the schedule."""

import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fakestash import FakeStash  # noqa: E402
from peaks import seedjobs as sj  # noqa: E402
from peaks.config import Config  # noqa: E402

HERE = Path(__file__).parent

# --- pure rules ---------------------------------------------------------------------


def test_wanted_files_match_the_scripts_filter():
    assert sj.wanted("Show/clip.mp4") and sj.wanted("a/b.funscript")
    assert not sj.wanted("Show/Sample/clip.mp4") and not sj.wanted("clip-sample.mkv")
    assert not sj.wanted("x.nfo") and not sj.wanted("x.mp4.!qB") and not sj.wanted("cover.jpg")


def test_new_files_skip_what_is_done():
    assert sj.new_files(["b.mp4", "a.mp4", " ", "a.mp4"], {"a.mp4"}) == ["b.mp4"]


def test_rclone_stats_lines():
    assert sj.rclone_stats('{"stats": {"bytes": 50, "totalBytes": 100, "speed": 2.5}}') == \
        {"bytes": 50, "total": 100, "speed": 2.5, "eta": None}
    assert sj.rclone_stats('{"msg": "Copied"}') is None and sj.rclone_stats("not json") is None


BASE = "/home/me/downloads/qbittorrent"


def _t(h, folder, progress=1.0, size=1e9, name="T"):
    return {"hash": h, "save_path": f"{BASE}/{folder}", "progress": progress, "size": size, "name": name}


def test_purge_plan_follows_the_scripts_rules():
    torrents = [
        _t("a", "1"),                                    # all pulled → delete
        _t("b", "1"),                                    # one file missing → keep
        _t("c", "vr", progress=0.4),                     # still downloading → keep
        _t("d", "1"),                                    # file list unreadable → keep
        _t("e", "other"),                                # other folder → not even considered
        _t("f", "1"),                                    # only a sample + nfo → nothing to verify → keep
        _t("w1", "arr/whisparr"), _t("w2", "arr/whisparr"), _t("w3", "arr/whisparr"),
    ]
    files = {
        "a": [{"name": "A/a.mp4"}, {"name": "A/a.funscript"}, {"name": "A/a.nfo"},
              {"name": "A/extra.mp4", "priority": 0}],    # deselected: never downloaded
        "b": [{"name": "B/b1.mp4"}, {"name": "B/b2.mp4"}],
        "d": None,
        "f": [{"name": "F/Sample/s.mp4"}, {"name": "F/f.nfo"}],
    }
    done = {"1": {"A/a.mp4", "A/a.funscript", "B/b1.mp4"}, "vr": set()}
    w = {"ok": True, "imported": {"W1", "W2"}, "queued": {"W2"}}
    plan = {d["hash"]: d for d in sj.purge_plan(torrents, files, done, BASE, ("1", "vr"), "arr/whisparr", w)}
    assert "e" not in plan
    assert plan["a"]["delete"] and plan["a"]["why"] == "all 2 files confirmed on Jarvis"
    assert plan["b"]["why"] == "1 of 2 files not pulled yet" and not plan["b"]["delete"]
    assert plan["c"]["why"] == "still downloading" and plan["d"]["why"] == "couldn't list its files"
    assert plan["f"]["why"] == "no video files found to verify"
    assert plan["w1"]["delete"] and not plan["w2"]["delete"] and not plan["w3"]["delete"]
    down = {d["hash"]: d for d in sj.purge_plan(torrents, files, done, BASE, ("1", "vr"), "arr/whisparr", {"ok": False})}
    assert not any(down[h]["delete"] for h in ("w1", "w2", "w3"))     # Whisparr down: leave them


def test_the_report_never_names_a_file():
    now = 1_800_000_000.0
    runs = [{"at": now - 600, "kind": "pull", "ok": True, "count": 2, "bytes": None},
            {"at": now - 7200, "kind": "pull", "ok": False, "count": 0, "bytes": None},
            {"at": now - 3600, "kind": "purge", "ok": True, "count": 3, "bytes": 5_000_000_000}]
    pulled = [{"t": now - 500, "cat": "1", "rel": "Secret Title/secret.mp4", "path": "/x", "size": 2e9, "ext": "mp4",
               "scene_id": "9"},
              {"t": now - 400, "cat": "1", "rel": "Secret Title/secret.funscript", "size": 1e3, "ext": "funscript"},
              {"t": now - 90000, "cat": "vr", "rel": "old.mp4", "size": 1e9, "ext": "mp4"}]
    rep = sj.report(runs, pulled, {"9": {"tier": "legendaire", "identified": True}}, now, "live")
    assert rep["came_in"]["files"] == 2 and rep["came_in"]["graded"] == 1 and rep["came_in"]["identified"] == 1
    assert rep["purge"] == {"runs": 1, "deleted": 3, "freed_bytes": 5_000_000_000, "mode": "live"}
    assert rep["issues"] == ["1 pull run(s) failed"]
    assert "secret" not in repr(rep).lower() and "Secret" not in repr(rep)


# --- the service, with the fake rclone ------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    import peaks.web.service as svc_mod

    box = tmp_path / "box"
    for rel in ("1/Show A/a.mp4", "1/Show A/a.funscript", "1/b.mp4", "vr/v.mp4"):
        p = box / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 1000)
    state = tmp_path / "seedbox"
    state.mkdir()
    lib = tmp_path / "data"
    (lib / "norating").mkdir(parents=True)
    (lib / "VR").mkdir()
    fake = tmp_path / "rclone"
    fake.write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'fake_rclone.py'} \"$@\"\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PEAKS_RCLONE", str(fake))
    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.delenv("PEAKS_SEEDBOX_REMOTE", raising=False)
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.seedbox.dir = str(state)
    stash = FakeStash({})
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    svc = svc_mod.Service(cfg)
    svc.save_seedjob_settings({"remote": f"fake:{box}", "filter": str(tmp_path / "filters.txt"),
                               "owner": f"{os.getuid()}:{os.getgid()}",
                               "categories": [{"cat": "1", "dest": str(lib / "norating")},
                                              {"cat": "vr", "dest": str(lib / "VR")}]})
    return {"svc": svc, "box": box, "state": state, "lib": lib}


def test_nothing_runs_before_the_cutover(env):
    svc = env["svc"]
    with pytest.raises(RuntimeError, match="Unraid seedbox scripts"):
        svc.run_seed_pull()
    assert not svc.seedjob_due("pull") and not svc.seedjob_due("purge")


def test_pull_brings_only_new_files_and_records_each(env):
    svc, state, lib = env["svc"], env["state"], env["lib"]
    svc.save_seedjob_settings({"cutover": True})
    (state / "1.done").write_text("1/already.mp4\nb.mp4\n")          # b.mp4 came down before
    out = svc.run_seed_pull()
    assert out == {"pulled": 3, "bytes": 3000, "errors": []}
    assert (lib / "norating" / "Show A" / "a.mp4").exists() and not (lib / "norating" / "b.mp4").exists()
    assert (lib / "VR" / "v.mp4").exists()
    assert set((state / "1.done").read_text().splitlines()) >= {"Show A/a.mp4", "Show A/a.funscript", "b.mp4"}
    assert oct((lib / "norating" / "Show A" / "a.mp4").stat().st_mode & 0o777) == "0o666"
    assert oct((lib / "norating" / "Show A").stat().st_mode & 0o777) == "0o777"
    line = (state / "runs.log").read_text().strip().splitlines()[-1].split("|")
    assert line[1:] == ["pull", "ok", "3"]
    assert svc.run_seed_pull()["pulled"] == 0                       # second run: nothing new
    assert svc._sj_foreign_pull(time.time()) is None                # Peaks' own runs aren't "foreign"


def test_a_failed_file_fails_the_run_and_is_retried(env):
    svc, box, state = env["svc"], env["box"], env["state"]
    svc.save_seedjob_settings({"cutover": True})
    (box / "1" / "FAILME.mp4").write_bytes(b"x")
    out = svc.run_seed_pull()
    assert out["pulled"] == 4 and out["errors"] == ["[1] 1 of 4 file(s) failed; will retry next run"]
    assert (state / "pull.failing").exists() and "FAILME.mp4" not in (state / "1.done").read_text()
    assert (state / "runs.log").read_text().strip().splitlines()[-1].split("|")[1:3] == ["pull", "fail"]
    (box / "1" / "FAILME.mp4").unlink()
    svc.run_seed_pull()
    assert not (state / "pull.failing").exists()                     # recovered
    assert any("recovered" in a["text"] for a in svc.seedjob_status()["alerts"])


def test_a_listing_failure_skips_the_category(env, monkeypatch):
    svc = env["svc"]
    svc.save_seedjob_settings({"cutover": True})
    monkeypatch.setenv("FAKE_RCLONE_LSF_FAIL", "1")
    out = svc.run_seed_pull()
    assert out["pulled"] == 0 and len(out["errors"]) == 2 and "could not list" in out["errors"][0]


def test_pulls_run_a_few_at_a_time_and_never_overlap(env, monkeypatch):
    svc, box = env["svc"], env["box"]
    svc.save_seedjob_settings({"cutover": True, "parallel": 2})
    for i in range(6):
        (box / "1" / f"n{i}.mp4").write_bytes(b"x" * 100)
    monkeypatch.setenv("FAKE_RCLONE_SLEEP", "0.15")
    peak, live = [0], svc._sj_live()

    def watch():
        while not done.is_set():
            peak[0] = max(peak[0], len((live.get("pull") or {}).get("lanes") or {}))
            time.sleep(0.01)
    done = threading.Event()
    th = threading.Thread(target=watch)
    th.start()
    result = {}
    runner = threading.Thread(target=lambda: result.update(svc.run_seed_pull()))
    runner.start()
    time.sleep(0.3)
    assert svc.run_seed_pull() == {"skipped": "a pull is already running"}
    runner.join()
    done.set()
    th.join()
    assert result["pulled"] == 10 and 1 <= peak[0] <= 2          # 4 files + the 6 new ones


def test_a_pull_peaks_didnt_make_blocks_the_takeover(env):
    svc, state = env["svc"], env["state"]
    svc.save_seedjob_settings({"cutover": True})
    (state / "runs.log").write_text(f"{int(time.time()) - 300}|pull|ok|1\n")      # the Unraid script, still on
    assert "Unraid script still on" in svc.seedjob_status()["ready"]["pull"]
    with pytest.raises(RuntimeError, match="Peaks didn't make"):
        svc.run_seed_pull()


class FakeQbt:
    def __init__(self, torrents, files):
        self._t, self._f, self.deleted = torrents, files, []

    def torrents(self):
        return self._t

    def files(self, h):
        return self._f[h]

    def delete_torrents(self, hashes, delete_files=True):
        self.deleted.append((list(hashes), delete_files))


def _purge_env(env, mode="auto"):
    svc, state = env["svc"], env["state"]
    svc.cfg.seedbox.qbt_url, svc.cfg.seedbox.qbt_user, svc.cfg.seedbox.qbt_password = "http://q", "u", "p"
    svc.save_seedjob_settings({"cutover": True, "purge_mode": mode, "whisparr_folder": ""})
    (state / "1.done").write_text("Show A/a.mp4\n")
    base = str(env["box"])
    return FakeQbt([{"hash": "h1", "save_path": f"{base}/1", "progress": 1, "size": 4e9, "name": "Show A"},
                    {"hash": "h2", "save_path": f"{base}/1", "progress": 1, "size": 1e9, "name": "Other"}],
                   {"h1": [{"name": "Show A/a.mp4"}], "h2": [{"name": "Other/o.mp4"}]})


def test_purge_is_a_dry_run_for_the_first_three_days(env):
    q = _purge_env(env)
    out = env["svc"].run_seed_purge(client=q)
    assert out == {"dry": True, "delete": 1, "freed_bytes": 4_000_000_000, "kept": 1} and q.deleted == []
    st = env["svc"].seedjob_status()["purge"]
    assert st["dry"] and 2.9 < st["dry_days_left"] <= 3.0
    assert (env["state"] / "runs.log").read_text().strip().split("|")[1:] == ["purge", "ok", "0", "0"]


def test_purge_goes_live_after_the_dry_run(env):
    svc = env["svc"]
    q = _purge_env(env)
    s = svc._settings()["seedjobs"]
    s["purge_dry_until"] = time.time() - 1                       # three days later
    out = svc.run_seed_purge(client=q)
    assert not out["dry"] and q.deleted == [(["h1"], True)]
    assert (env["state"] / "runs.log").read_text().strip().split("|")[1:] == ["purge", "ok", "1", "4000000000"]
    assert any("dry run is over" in a["text"] for a in svc.seedjob_status()["alerts"])


def test_staying_in_dry_run_never_deletes(env):
    svc = env["svc"]
    q = _purge_env(env, mode="dry")
    svc._settings()["seedjobs"]["purge_dry_until"] = time.time() - 1
    assert svc.run_seed_purge(client=q)["dry"] and q.deleted == []


def test_the_report_counts_and_trims_runs_log(env):
    svc, state = env["svc"], env["state"]
    svc.save_seedjob_settings({"cutover": True})
    svc.run_seed_pull()
    old = int(time.time()) - 9 * 86400
    with open(state / "runs.log", "a") as fh:
        fh.write(f"{old}|pull|ok|5\n")
    rep = svc.run_seed_report()
    assert rep["came_in"]["files"] == 4 and rep["pull"]["ok"] == 1
    assert str(old) not in (state / "runs.log").read_text()          # older than 7 days: gone
    text = repr(rep) + repr(svc.seedjob_status()["alerts"])
    assert "a.mp4" not in text and "Show A" not in text and "v.mp4" not in text


def test_schedule(env):
    svc = env["svc"]
    svc.save_seedjob_settings({"cutover": True, "purge_at": "05:00"})
    now = svc._at_today("05:30", time.time())
    assert svc.seedjob_due("pull", now)
    svc._sj_update(lambda d: d["last"].__setitem__("pull", now - 600))
    assert not svc.seedjob_due("pull", now)                          # ran 10 min ago (every 30)
    assert svc.seedjob_due("report", svc._at_today("06:01", now))
    assert not svc.seedjob_due("report", svc._at_today("05:59", now))


def test_the_floor_never_shows_a_file_name_and_stays_ascii(env):
    import json

    from peaks.web.floor import floor_replay, floor_state
    from peaks.web.jobs import JobManager

    svc = env["svc"]
    svc.save_seedjob_settings({"cutover": True})
    jobs = JobManager()
    job = jobs.start("seed-pull", svc.run_seed_pull)
    for _ in range(200):
        if job.status != "running":
            break
        time.sleep(0.02)
    snap = json.dumps(floor_state(svc, jobs)) + json.dumps(floor_replay(svc))
    assert job.status == "done"
    assert "a.mp4" not in snap and "Show A" not in snap and "v.mp4" not in snap
    assert all(ord(c) < 127 for t in floor_state(svc, jobs)["ticker"] for c in t["text"])
