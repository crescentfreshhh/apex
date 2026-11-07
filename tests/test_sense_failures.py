"""A file Motion or Hearing can't read is remembered: the next run doesn't start
on it again (it waits a week; after 3 failures, until the file changes), one
damaged stretch only costs that window, and a broken model blames no files."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_motion import FakeSampler, FakeVJEPA, _wire  # noqa: E402
from test_zoom import _svc as _zoom_svc  # noqa: E402

pytest.importorskip("sklearn")


class BadFiles(FakeSampler):
    """Every window of `bad` scenes fails; in `patchy` scenes, only the first window."""

    def __init__(self, bad=(), patchy=()):
        super().__init__()
        self.bad, self.patchy, self.first_seen = set(bad), set(patchy), set()

    def window_frames(self, path, start, dur, step, size=None):
        from peaks.sampling import SamplerError

        sid = path.rsplit("/", 1)[-1].split(".")[0]
        if sid in self.bad or (sid in self.patchy and sid not in self.first_seen):
            self.first_seen.add(sid)
            self.calls.append((sid, start, size))
            raise SamplerError(f"ffmpeg window failed for {path}@{start}s: Invalid NAL unit size")
        return super().window_frames(path, start, dur, step, size)


def _world(tmp_path, monkeypatch, **kw):
    svc, stash, saved, _ = _zoom_svc(tmp_path, monkeypatch, pattern="random", seed=5)
    svc.run_zoom(limit=200)
    todo = svc.motion_todo()
    return svc, todo


def test_a_failed_file_is_not_first_next_run_and_waits_a_week(tmp_path, monkeypatch):
    svc, todo = _world(tmp_path, monkeypatch)
    bad = todo[0]["scene_id"]                                  # first in line (a scene with saves)
    sampler = BadFiles(bad=[bad])
    _wire(svc, monkeypatch, FakeVJEPA(), sampler)
    r = svc.run_motion(limit=3)
    assert r["failed"] == 1 and bad in svc.sense_failures("motion")
    assert bad not in [x["scene_id"] for x in svc.motion_todo()]          # skipped next run
    assert svc.motion_status()["unreadable"] == 1
    sampler.calls.clear()
    svc.run_motion(limit=3)
    assert bad not in {sid for sid, *_ in sampler.calls}                  # never touched again
    # a week later it's tried again — at the back of the line
    f = svc.sense_failures("motion")
    f[bad]["at"] = time.time() - 8 * 86400
    svc._sense_fail_edit("motion", lambda cur: cur.update(f))
    left = [x["scene_id"] for x in svc.motion_todo()]
    assert left and left[-1] == bad


def test_three_failures_park_it_until_the_file_changes(tmp_path, monkeypatch):
    svc, todo = _world(tmp_path, monkeypatch)
    row = todo[0]
    for _ in range(3):
        svc._sense_failed("motion", row, RuntimeError("Invalid NAL unit size"))
    fails = svc.sense_failures("motion")
    assert svc._sense_skip(fails, row, now=time.time() + 30 * 86400)        # parked, however long it's been
    changed = {**row, "fingerprint": "a-new-download"}
    assert not svc._sense_skip(fails, changed)                              # re-downloaded: tried again


def test_one_damaged_stretch_only_costs_that_window(tmp_path, monkeypatch):
    svc, todo = _world(tmp_path, monkeypatch)
    sid = todo[0]["scene_id"]
    _wire(svc, monkeypatch, FakeVJEPA(), BadFiles(patchy=[sid]))
    r = svc.run_motion(limit=1)
    assert r["seen"] == 1 and not r["failed"]
    assert sid not in svc.sense_failures("motion")
    assert sid not in [x["scene_id"] for x in svc.motion_todo()]            # done: not redone each run
    assert svc.motion_scene_vector(sid) is not None                         # its good windows count


def test_a_broken_model_blames_no_files(tmp_path, monkeypatch):
    svc, todo = _world(tmp_path, monkeypatch)

    class Broken(FakeVJEPA):
        def embed_clips(self, clips):
            raise RuntimeError("expected input to have 4 dimensions")
    _wire(svc, monkeypatch, Broken(), FakeSampler())
    r = svc.run_motion(limit=500)
    assert r["failed"] == 5 and svc.sense_failures("motion") == {}


def test_is_there_work_stops_at_the_first(tmp_path, monkeypatch):
    svc, todo = _world(tmp_path, monkeypatch)
    calls = []
    real = type(svc)._motion_centers
    monkeypatch.setattr(type(svc), "_motion_centers", lambda self, *a: calls.append(1) or real(self, *a))
    assert svc.motion_due() and len(calls) == 1


def test_retry_clears_and_the_api_says_how_many(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, todo = _world(tmp_path, monkeypatch)
    svc._sense_failed("motion", todo[0], RuntimeError("x"))
    c = TestClient(app_mod.create_app(svc.cfg))
    assert c.get("/api/motion").json()["unreadable"] == 1
    assert c.post("/api/motion/retry-failed").json() == {"cleared": 1}
    assert svc.sense_failures("motion") == {} and todo[0]["scene_id"] in [x["scene_id"] for x in svc.motion_todo()]


def test_hearing_skips_a_file_it_could_not_read(tmp_path, monkeypatch):
    svc, todo = _world(tmp_path, monkeypatch)
    first = svc.hear_todo()[0]
    svc._sense_failed("hear", first, RuntimeError("no audio decodable"))
    assert first["scene_id"] not in [x["scene_id"] for x in svc.hear_todo()]
    assert svc.hear_status()["unreadable"] == 1
