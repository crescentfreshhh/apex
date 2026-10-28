"""Hearing: the listening job, sound features for moments and grading, and
search by sound — no torch, no ffmpeg: a fake CLAP and synthetic audio."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402
from test_zoom import _svc as _zoom_svc  # noqa: E402

pytest.importorskip("sklearn")

SR = 8000


class FakeClap:
    """Sound "meaning" = [loud?, tier hint, noise…]; the text "moaning" points at loud."""

    sr, dim = SR, 8

    def __init__(self, hint=None):
        self.hint = hint or {}

    def embed_audio(self, blocks):
        out = np.zeros((len(blocks), self.dim), np.float32)
        for i, b in enumerate(blocks):
            out[i, 0] = float(np.sqrt((b ** 2).mean()) > 0.2)
            out[i, 1] = float(b[1]) if len(b) > 1 else 0.0       # the planted tier hint rides in sample 1
            out[i, 2:] = 0.05
        return out

    def embed_text(self, prompts):
        return np.asarray([[1.0] + [0.0] * (self.dim - 1) for _ in prompts], np.float32)


def _blocks(duration, loud_at=(), hint=0.0, seed=0):
    """10 s blocks every 5 s of a quiet tone, loud (and rising) around `loud_at`."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / SR)
    amp = np.full(len(t), 0.05)
    for c in loud_at:
        m = (t >= c - 12) & (t <= c + 4)
        amp[m] = 0.1 + 0.8 * np.clip((t[m] - (c - 12)) / 12, 0, 1)
    x = (amp * np.sin(2 * np.pi * 300 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    x[1::int(5 * SR)] = hint
    out = []
    for s in np.arange(0, max(0.0, duration - 10) + 1e-6, 5.0):
        a = int(s * SR)
        out.append((float(s), x[a: a + 10 * SR].copy()))
    return out


def _wire(svc, monkeypatch, plan, clap=None):
    import peaks.web.service as svc_mod

    clap = clap or FakeClap()
    monkeypatch.setattr(svc_mod.Service, "_hear_embedder", lambda self: clap)
    monkeypatch.setattr(svc_mod.Service, "_hear_decode", lambda self, path, sr: plan(path))
    return clap


def test_the_job_listens_in_order_skips_rejects_and_handles_silence(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    for sid in stash.s:
        stash.s[sid]["path"] = f"/data/{sid}.mp4"
    stash.s["u3"]["rating100"] = 20
    saved = {"5": [100.0]}
    monkeypatch.setattr(type(svc), "_zoom_saved", lambda self: saved)
    _wire(svc, monkeypatch, lambda p: None if p.endswith("/7.mp4") else _blocks(60))
    todo = [r["scene_id"] for r in svc.hear_todo()]
    assert todo[0] == "5" and "u3" not in todo                       # saves first, rejects never

    class Job:
        cancelled, progress = False, None

        def log(self, *_):
            pass
    j = Job()
    orig = svc.hear_scene

    calls = []

    def stop_after_two(row, emb):
        calls.append(row["scene_id"])
        if len(calls) == 2:
            j.cancelled = True
        return orig(row, emb)
    svc.hear_scene = stop_after_two
    r = svc.run_hear(job=j)
    assert r["heard"] == 2
    svc.hear_scene = orig
    svc.run_hear()
    assert not svc.hear_todo() and svc.heard("7") and svc._sound_load("7") is None   # silent file: heard, no sound
    st = svc.hear_status()
    assert st["heard"] == st["embedded"] - 1 and st["settings"]["on"]


def test_sound_wins_the_moment_test_when_saves_follow_the_sound(tmp_path, monkeypatch):
    """Saves sit where it gets loud; the picture alone can't tell (random taste
    pattern), so shape + sound must beat shape alone, honestly, and Peaks says so."""
    svc, stash, saved, _ = _zoom_svc(tmp_path, monkeypatch, pattern="random", seed=5)
    _wire(svc, monkeypatch, lambda p: _blocks(300, loud_at=saved.get(p.rsplit("/", 1)[-1].split(".")[0], [])))
    svc.run_zoom(limit=200)
    svc.run_hear(limit=500)
    rep = svc.zoom_status()["model"]
    assert rep["trained"] and "shape + sound" in rep["sets"]
    assert rep["sets"]["shape + sound"]["top3"] >= rep["sets"]["shape"]["top3"] + 0.02
    assert rep["uses"] == "shape + sound" and rep["use_shape"]
    feats = {w["feature"] for w in rep["weights"] if w["weight"] > 0}
    assert feats & {"loud_peak", "loud_rise", "sound_taste"}
    lab = next(x for x in svc.lab()["senses"] if x["id"] == "sound")
    assert lab["moments"]["in_use"] and lab["status"]["heard"]


def test_the_tier_model_keeps_sound_only_when_it_helps(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    for sid in stash.s:
        stash.s[sid]["path"] = f"/data/{sid}.mp4"
    tiers = {sid: v["o_counter"] for sid, v in stash.s.items()}
    hint = {16: 0.0, 17: 0.4, 18: 0.9}

    def plan(p):
        sid = p.rsplit("/", 1)[-1].split(".")[0]
        return _blocks(40, hint=hint.get(tiers[sid], 0.2), seed=int.from_bytes(sid.encode(), "little") % 997)
    _wire(svc, monkeypatch, plan)
    svc.run_hear(limit=500)
    rep = svc.train_tier_model()
    assert "sound_gain" in rep                                         # tested, not assumed
    # the visual look already separates these tiers perfectly, so sound can't add: not kept
    assert rep["use_sound"] is False
    m = svc._tier_model_state()["model"]
    assert "sound" not in [g for g, *_ in m.groups()]


def test_search_by_sound(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    for sid in stash.s:
        stash.s[sid]["path"] = f"/data/{sid}.mp4"
    _wire(svc, monkeypatch, lambda p: _blocks(60, loud_at=[40.0] if p.endswith("/3.mp4") else []))
    svc.run_hear(limit=500)
    hits = svc.sound_search("moaning")
    assert hits[0].scene_id == "3" and 20 <= hits[0].time <= 40
    c = TestClient(app_mod.create_app(svc.cfg))
    r = c.get("/api/search/text", params={"q": "sound: moaning"}).json()
    assert r["sound"] == "moaning" and r["total"] >= 1
    assert c.get("/api/hear").json()["heard"] >= 30
    assert c.post("/api/hear/settings", json={"on": False}).json()["on"] is False
