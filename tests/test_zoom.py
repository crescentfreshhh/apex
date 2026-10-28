"""Moment zoom: dense windows around candidates, shapes, the honest moment model, spans.

No ffmpeg and no real model: a fake sampler hands out (scene, time) "frames" and a
fake embedder turns them into vectors whose first component is the planted taste.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402

pytest.importorskip("sklearn")

CENTERS = [30.0, 90.0, 150.0, 210.0, 270.0]


def _plant(sid, t, slow_at, spike_at, pattern="slow"):
    """Taste over time in a scene: a slow build at `slow_at`, a one-frame spike at `spike_at`."""
    if pattern == "slow" and slow_at - 10 <= t <= slow_at + 18:
        return min(0.7, 0.3 + 0.4 * (t - (slow_at - 10)) / 16)
    if abs(t - spike_at) < 1:
        return 0.9
    return 0.3


class FakeSampler:
    def __init__(self, svc, plan):
        self.svc, self.plan, self.calls = svc, plan, []

    def window_frames(self, path, start, dur, step):
        sid = path.rsplit("/", 1)[-1].split(".")[0]
        self.calls.append((sid, start))
        return [(round(start + i * step, 2), (sid, round(start + i * step, 2)))
                for i in range(int(dur / step)) if start + i * step >= 0]


class FakeEmbedder:
    def __init__(self, plan, dim=24):
        self.plan, self.dim = plan, dim

    def embed_images(self, images):
        out = np.zeros((len(images), self.dim), np.float32)
        for i, (sid, t) in enumerate(images):
            out[i, 0] = self.plan(sid, t)
            out[i, 1 + int(t // 2) % (self.dim - 1)] = 1.0      # frame to frame, something changes
        return out


def _svc(tmp_path, monkeypatch, pattern="slow", seed=0):
    import peaks.web.service as svc_mod

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    rng = np.random.default_rng(seed)
    graded = [sid for sid, v in stash.s.items() if v["rating100"] == 100 and v["o_counter"] in (16, 17, 18)]
    slow = {sid: CENTERS[int(rng.integers(3, 5))] for sid in graded}           # slow build in a low-taste window
    spike = {sid: CENTERS[0] for sid in graded}
    saved = {sid: [slow[sid] + 4.0 if pattern == "slow" else CENTERS[int(rng.integers(0, 5))] + 4.0] for sid in graded}

    def plan(sid, t):
        return _plant(sid, t, slow.get(sid, -999), spike.get(sid, -999), pattern)

    def strips(self, ids, n=4, min_gap=20.0):
        out = {}
        for sid in ids:
            # frame taste: the spike looks best, the slow build ranks low (4th–5th)
            score = {CENTERS[0]: 0.9, CENTERS[1]: 0.5, CENTERS[2]: 0.45, CENTERS[3]: 0.4, CENTERS[4]: 0.35}
            if sid in slow:
                score[slow[sid]] = 0.35 if slow[sid] == CENTERS[4] else 0.4
            out[sid] = [{"key": "k", "t": c, "score": score[c]} for c in CENTERS[:n]]
        return out

    monkeypatch.setattr(svc_mod.Service, "_scene_moment_strips", strips)
    monkeypatch.setattr(svc_mod.Service, "_zoom_saved", lambda self: saved)
    monkeypatch.setattr(svc_mod.Service, "_taste_block", lambda self, v: np.asarray(v)[:, 0])
    sampler, emb = FakeSampler(svc, plan), FakeEmbedder(plan)
    monkeypatch.setattr(svc_mod.Service, "_zoom_sampler", lambda self: sampler)
    monkeypatch.setattr(svc_mod.Service, "_embedder", lambda self, model=None: emb)
    for sid in stash.s:
        stash.s[sid]["path"] = f"/data/{sid}.mp4"
    return svc, stash, saved, sampler


def test_zoom_windows_skip_rejects_resume_and_redo_on_a_new_save(tmp_path, monkeypatch):
    svc, stash, saved, sampler = _svc(tmp_path, monkeypatch)
    stash.s["u3"]["rating100"] = 20                                        # a reject: never zoomed
    todo = svc.zoom_todo()
    assert "u3" not in [r["scene_id"] for r in todo]
    assert todo[0]["scene_id"] in saved                                    # scenes with saves first
    sid = todo[0]["scene_id"]
    c = svc.zoom_candidates(sid, saved[sid])
    assert [x["c"] for x in c if x["kind"] == "pick"] == CENTERS and saved[sid][0] in [x["c"] for x in c if x["kind"] == "saved"]

    class Job:
        cancelled, progress = False, None
        n = 0

        def log(self, *_):
            pass
    j = Job()
    orig = svc.zoom_scene

    def stop_after_three(row, *a, **k):
        j.n += 1
        if j.n == 3:
            j.cancelled = True
        return orig(row, *a, **k)
    svc.zoom_scene = stop_after_three
    assert svc.run_zoom(job=j)["zoomed"] == 3                              # stopped …
    left = len(svc.zoom_todo())
    svc.zoom_scene = orig
    svc.run_zoom()
    assert len(svc.zoom_todo()) == 0 and left > 0                          # … and resumed
    starts = {round(s) for x, s in sampler.calls if x == sid}
    assert starts == {round(c - 20) for c in CENTERS} | {round(saved[sid][0] - 20)}   # windows only, ±20 s
    saved[sid].append(100.0)                                               # a new save not covered yet
    assert [r["scene_id"] for r in svc.zoom_todo()] == [sid]


def test_the_shape_model_wins_on_a_slow_build_and_picks_become_spans(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, stash, saved, _ = _svc(tmp_path, monkeypatch)
    svc.run_zoom(limit=200)                                                # trains at the end
    rep = svc.zoom_status()["model"]
    assert rep["trained"] and rep["compared"] >= 30
    assert rep["frame"]["top3"] == 0.0 and rep["shape"]["top1"] >= 0.9    # held out, same hit rule
    assert rep["use_shape"] and "shape picks win" in rep["verdict"]
    assert any(w["feature"] in ("hold", "rise") and w["weight"] > 0 for w in rep["weights"])
    sid = next(iter(saved))
    picks = svc.scene_picks(sid)
    assert picks[0]["shape"] and picks[0]["start"] < picks[0]["t"] <= picks[0]["end"]
    assert abs(picks[0]["t"] - saved[sid][0]) <= 15                        # the slow build comes first
    r = TestClient(app_mod.create_app(svc.cfg)).get(f"/api/scene/{sid}/picks").json()
    assert {"start", "end", "shape", "answered"} <= set(r[0])


def test_no_signal_keeps_frame_picks(tmp_path, monkeypatch):
    svc, _, saved, _ = _svc(tmp_path, monkeypatch, pattern="random", seed=3)
    svc.run_zoom(limit=200)
    rep = svc.zoom_status()["model"]
    assert not rep["use_shape"] and "not better yet" in (rep.get("verdict") or rep.get("reason", "not better yet"))
    sid = next(iter(saved))
    assert svc.scene_picks(sid)[0]["shape"] is False                       # spans, ranked by taste
