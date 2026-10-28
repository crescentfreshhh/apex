"""Motion: clips cut from candidate windows, the motion job, and motion as a
sense for the moment model — no torch, no ffmpeg: a fake video encoder."""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_zoom import _svc as _zoom_svc  # noqa: E402

pytest.importorskip("sklearn")


def test_clips_are_cut_every_four_seconds_from_a_four_fps_window():
    from peaks.web.motion import cut_clips

    frames = list(range(160))                       # 40 s at 4 fps, from t = 10
    clips = cut_clips(frames, 10.0)
    assert [c for c, _ in clips] == [12.0 + 4 * i for i in range(10)]
    assert all(len(cl) == 16 for _, cl in clips) and clips[1][1][0] == 16      # 4 s in → frame 16
    assert cut_clips(frames[:10], 0.0) == []                                   # not even one whole clip


class FakeSampler:
    def __init__(self):
        self.calls = []

    def window_frames(self, path, start, dur, step, size=None):
        sid = path.rsplit("/", 1)[-1].split(".")[0]
        self.calls.append((sid, start, size))
        return [(round(start + i * step, 2), (sid, round(start + i * step, 2))) for i in range(int(dur / step))]


class FakeVJEPA:
    """A clip's "motion": a planted direction near your saves (or nothing), plus noise."""

    size, dim = 384, 12

    def __init__(self, near_save=None, seed=0):
        self.near, self.rng = near_save, np.random.default_rng(seed)

    def embed_clips(self, clips):
        out = self.rng.standard_normal((len(clips), self.dim)).astype(np.float32) * 0.3
        for i, cl in enumerate(clips):
            sid, t0 = cl[0]
            c = t0 + 2.0
            if self.near and any(abs(c - s) <= 6 for s in self.near.get(sid, [])):
                out[i, 0] += 3.0
        return out


def _wire(svc, monkeypatch, emb, sampler):
    import peaks.web.service as svc_mod

    monkeypatch.setattr(svc_mod.Service, "_motion_embedder", lambda self: emb)
    monkeypatch.setattr(svc_mod.Service, "_motion_sampler", lambda self: sampler)


def test_motion_wins_when_your_saves_move_a_certain_way(tmp_path, monkeypatch):
    svc, stash, saved, _ = _zoom_svc(tmp_path, monkeypatch, pattern="random", seed=5)
    svc.run_zoom(limit=200)
    sampler = FakeSampler()
    _wire(svc, monkeypatch, FakeVJEPA(near_save=saved), sampler)
    assert len(svc.motion_todo()) == svc.motion_status()["embedded"]
    r = svc.run_motion(limit=500)
    assert r["seen"] >= 30 and not r["failed"] and not svc.motion_todo()
    assert all(size == 384 for *_, size in sampler.calls)                     # decoded at the model's size
    rep = svc.zoom_status()["model"]
    assert "shape + motion" in rep["sets"] and rep["uses"] in ("shape + motion", "all") and rep["use_shape"]
    assert any(w["feature"] == "motion_taste" and w["weight"] > 0 for w in rep["weights"])
    card = next(x for x in svc.lab()["senses"] if x["id"] == "motion")
    assert card["moments"]["in_use"] and card["status"]["seen"] >= 30
    sid = next(iter(saved))
    assert svc.motion_scene_vector(sid) is not None


def test_motion_that_says_nothing_is_not_used(tmp_path, monkeypatch):
    svc, stash, saved, _ = _zoom_svc(tmp_path, monkeypatch, pattern="slow")
    svc.run_zoom(limit=200)
    _wire(svc, monkeypatch, FakeVJEPA(near_save=None, seed=2), FakeSampler())
    svc.run_motion(limit=500)
    rep = svc.zoom_status()["model"]
    assert rep["uses"] == "shape"                                          # the shape already wins; noise adds nothing


def test_a_hub_load_failure_is_a_clear_error(monkeypatch):
    from peaks import embedding as E

    def boom(*a, **k):
        raise RuntimeError("HTTP Error 404")
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                       hub=types.SimpleNamespace(load=boom), float16="f16")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(E.MotionUnavailable, match="V-JEPA 2.1"):
        E.MotionEmbedder()


def test_the_motion_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, *_ = _zoom_svc(tmp_path, monkeypatch)
    svc.run_zoom(limit=200)
    c = TestClient(app_mod.create_app(svc.cfg))
    d = c.get("/api/motion").json()
    assert d["seen"] == 0 and d["embedded"] >= 30 and d["running"] is None
    assert c.post("/api/motion/settings", json={"on": False}).json()["on"] is False
