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


def _fake_torch(tmp_path, events, load=None, download=None):
    class Enc:
        embed_dim = 1024

        def eval(self):
            return self

        def to(self, _):
            return self

    def fetch(url, dst, progress=True):
        events.append(("download", url, dst))
        if download:
            download()
        open(dst, "wb").write(b"weights")

    def hub_load(repo, entry):
        events.append(("load", repo, entry))
        if load:
            load()
        return Enc(), object()                                   # (encoder, predictor)
    return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                 hub=types.SimpleNamespace(load=hub_load, get_dir=lambda: str(tmp_path / "hub"),
                                                           download_url_to_file=fetch),
                                 tensor=lambda *a, **k: types.SimpleNamespace(view=lambda *s: None), float16="f16")


def test_the_weights_come_from_meta_not_localhost_and_only_once(tmp_path, monkeypatch):
    """Meta's hub code points its weights at localhost:8300 (vjepa2 issue #174):
    Peaks fetches them from dl.fbaipublicfiles.com into the hub cache first."""
    from peaks import embedding as E

    ev, said = [], []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(tmp_path, ev))
    E.MotionEmbedder(log=said.append)
    assert ev[0][0] == "download" and ev[1][0] == "load"                      # cached before the hub looks
    assert ev[0][1] == "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
    assert ev[0][2] == str(tmp_path / "hub" / "checkpoints" / "vjepa2_1_vitl_dist_vitG_384.pt")
    assert said and "one time" in said[0]
    ev.clear()
    E.MotionEmbedder()
    assert [e[0] for e in ev] == ["load"]                                     # already cached: no download


def test_a_failed_download_or_hub_load_is_a_clear_error(tmp_path, monkeypatch):
    from peaks import embedding as E

    def refuse():
        raise OSError("[Errno 111] Connection refused")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(tmp_path, [], download=refuse))
    with pytest.raises(E.MotionUnavailable, match="dl.fbaipublicfiles.com"):
        E.MotionEmbedder()
    assert not (tmp_path / "hub" / "checkpoints" / "vjepa2_1_vitl_dist_vitG_384.pt").exists()   # retried next run

    def boom():
        raise RuntimeError("HTTP Error 404")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(tmp_path / "b", [], load=boom))
    with pytest.raises(E.MotionUnavailable, match="V-JEPA 2.1.*already in torch.hub's cache"):
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


def test_a_broken_motion_model_stops_early_and_says_why(tmp_path, monkeypatch):
    svc, stash, saved, _ = _zoom_svc(tmp_path, monkeypatch)
    svc.run_zoom(limit=200)

    class Broken(FakeVJEPA):
        def embed_clips(self, clips):
            raise RuntimeError("expected input[1, 3, 16, 384, 384] to have 4 dimensions")
    _wire(svc, monkeypatch, Broken(), FakeSampler())
    r = svc.run_motion(limit=500)
    assert r["seen"] == 0 and r["failed"] == 5
    st = svc.motion_status()
    assert "failed the same way" in st["last_run"]["stopped"] and "4 dimensions" in st["error"]


# --- motion helps set a moment's length ------------------------------------------------------

def test_a_motion_curve_moves_the_span_onto_the_movement():
    from peaks.moment_shape import blend_scores, overlap, span

    t = np.arange(28.0, 70.0, 2.0)
    vis = np.full(len(t), 0.3)
    vis[np.searchsorted(t, 44.0)] = 0.9                            # one frame looks best, at 44 s
    mot = np.where((t >= 40) & (t <= 56), 0.8, 0.1)                # the movement is 40–56 s
    a, _, b = span(t, vis)
    assert b - a == 6 and overlap((a, b), (40, 56)) < 0.4          # looks alone: a 6 s sliver
    a, _, b = span(t, blend_scores(vis, mot))
    assert overlap((a, b), (40, 56)) > 0.7
    assert overlap((0, 10), (5, 15)) == 5 / 15 and overlap((0, 1), (2, 3)) == 0.0


def _span_world(tmp_path, monkeypatch, n=25, motion="follows", seed=0):
    import peaks.web.service as svc_mod
    from test_catalogue import _triage_svc

    svc, _ = _triage_svc(tmp_path, monkeypatch)
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 120.0, 2.0)
    vis = {}
    for i in range(n):
        v = 0.3 + 0.02 * rng.standard_normal(len(t))
        v[np.searchsorted(t, 44.0)] = 0.9                           # one pretty frame inside the action
        vis[f"m{i}"] = v
    ranges = {sid: [(40.0, 56.0)] for sid in vis}
    scenes = {sid: {"key": sid} for sid in vis}

    class Cache:
        def load(self, key, ns):
            return t, np.stack([vis[key], np.zeros(len(t))], axis=1), {"candidates": [{"c": 48.0, "kind": "saved"}]}

    def curve(self, sid, times):
        if motion == "follows":
            return np.where((times >= 40) & (times <= 56), 0.8, 0.1)
        return np.random.default_rng(hash(sid) % 1000).random(len(times))
    monkeypatch.setattr(svc_mod.Service, "_zoom_view", lambda self: {"scenes": scenes})
    monkeypatch.setattr(svc_mod.Service, "_zoom_cache", lambda self: Cache())
    monkeypatch.setattr(svc_mod.Service, "_taste_block", lambda self, v: np.asarray(v)[:, 0])
    monkeypatch.setattr(svc_mod.Service, "motion_taste_curve", curve)
    monkeypatch.setattr(svc_mod.Service, "_zoom_saved_ranges", lambda self: ranges)
    monkeypatch.setattr(svc_mod.Service, "_window_senses", lambda self, *a: {})
    return svc


def test_motion_sets_lengths_only_when_it_lands_closer_to_yours(tmp_path, monkeypatch):
    svc = _span_world(tmp_path, monkeypatch)
    before = svc._zoom_windows("m0")[0]["span"]
    st = svc.motion_span_check()
    assert st["on"] and st["n"] == 25 and st["iou_with"] > st["iou_without"] + 0.1 and st["p"] <= 0.05
    after = svc._zoom_windows("m0")[0]["span"]                     # the players' spans now follow the movement
    assert after != before and after[0] >= 38 and after[2] <= 58
    card = next(x for x in svc.lab()["senses"] if x["id"] == "motion")
    assert card["length"]["on"]


def test_motion_noise_or_too_few_saves_leave_lengths_alone(tmp_path, monkeypatch):
    svc = _span_world(tmp_path, monkeypatch, motion="noise")
    assert not svc.motion_span_check()["on"] and not svc.motion_span_on()
    svc2 = _span_world(tmp_path / "b", monkeypatch, n=5)
    st = svc2.motion_span_check()
    assert not st["on"] and st["n"] == 5 and "not enough" in st["verdict"]
