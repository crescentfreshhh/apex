"""The Lab: a candidate model is judged against the current one on the same
scenes, the same honest way, and only a clear winner can be switched to.

No torch, no downloads: the "candidate embed" writes planted vectors straight
into the candidate's cache namespace.
"""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402

pytest.importorskip("sklearn")

CAND = "dinov3_vitl16"
NS = "dinov3-vitl16"
D = 24


def _lab_svc(tmp_path, monkeypatch, cand_quality="better", n_saved=16):
    import peaks.web.service as svc_mod
    from peaks.cache import EmbeddingCache
    from peaks.tier_model import train_class

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    cache = EmbeddingCache(svc.cfg.embedding.cache_dir)
    rng = np.random.default_rng(11)
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)  # noqa: E731
    look = {c: unit(rng.standard_normal(D)) for c in ("exceptionnelle", "merveilleuse", "legendaire", None)}
    moment = unit(rng.standard_normal(D))
    keys = {cache.peek_meta(k, "dinov2")["scene_id"]: k for k in cache.keys("dinov2")}
    saved = {sid: [120.0] for sid in sorted(keys)[:n_saved]}
    t = np.arange(10, dtype=np.float32) * 30
    # the current model barely sees the tiers; the candidate sees them (or sees nothing)
    tiers = {r["scene_id"]: train_class(r["tier"]) for r in svc._catalogue_all()}
    for sid, k in keys.items():
        base = look[tiers.get(sid)]
        cur = unit(0.15 * base + rng.standard_normal((10, D)))
        cache.save(k, "dinov2", t, cur.astype(np.float32), meta={"scene_id": sid})
    plan = {}
    for sid, k in keys.items():
        base = look[tiers.get(sid)]
        if cand_quality == "better":
            new = unit(base + 0.25 * rng.standard_normal((10, D)))
            if sid in saved:
                new[4] = unit(moment + 0.1 * rng.standard_normal(D))
        else:
            new = unit(rng.standard_normal((10, D)))
        plan[k] = (sid, new.astype(np.float32))
    calls = []

    def fake_embed(self, cand, rows, job=None):
        calls.append(len(rows))
        n = 0
        for k, (sid, v) in plan.items():
            if sid in {r["scene_id"] for r in rows} and not cache.has(k, NS):
                cache.save(k, NS, t, v, meta={"scene_id": sid, "interval": 30.0})
                n += 1
                if job is not None and job.cancelled:
                    break
        return {"embedded": n, "frames": 10 * n, "seconds": 2.0}

    monkeypatch.setattr(svc_mod.Service, "_lab_embed", fake_embed)
    monkeypatch.setattr(svc_mod.Service, "_zoom_saved", lambda self: saved)
    return svc, stash, saved, calls, cache


def test_the_sample_is_stratified_and_keeps_your_saved_scenes(tmp_path, monkeypatch):
    import peaks.web.lab as lab
    from peaks.tier_model import train_class

    monkeypatch.setattr(lab, "PER_TIER", 5)
    svc, _, saved, _, _ = _lab_svc(tmp_path, monkeypatch, n_saved=3)
    rows = svc.lab_sample()
    ids = {r["scene_id"] for r in rows}
    assert set(saved) <= ids
    per = {}
    for r in rows:
        per[train_class(r["tier"])] = per.get(train_class(r["tier"]), 0) + 1
    graded_extra = {c: n - sum(1 for s in saved if s in ids and train_class(next(
        x["tier"] for x in rows if x["scene_id"] == s)) == c) for c, n in per.items() if c}
    assert all(n <= 5 for n in graded_extra.values()) and len([c for c in per if c]) >= 3
    assert [r["scene_id"] for r in svc.lab_sample()] == [r["scene_id"] for r in rows]      # stable


def test_a_better_backbone_wins_and_can_be_switched_to(tmp_path, monkeypatch):
    svc, _, _, calls, cache = _lab_svc(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        svc.lab_switch(CAND)                                               # no trial, no switch
    tr = svc.run_lab(CAND)
    assert tr["win"] and tr["against"] == "dinov2_vits14" and tr["scenes"] >= 36
    assert tr["new"]["grading"]["accuracy"] >= tr["current"]["grading"]["accuracy"] + 0.02
    assert tr["new"]["moments"]["top3"] > tr["current"]["moments"]["top3"]
    assert tr["paired"]["only_candidate"] > tr["paired"]["only_current"]
    assert tr["switch_hours"] is not None and "wins" in tr["verdict"]
    lab = svc.lab()
    c = next(x for x in lab["candidates"] if x["id"] == CAND)
    assert c["trial"]["win"] and not c["stale"] and c["label"] == "DINOv3 ViT-L/16"
    svc.lab_switch(CAND)
    assert svc._settings()["dino_model"] == CAND and svc._model_name() == NS
    assert len(cache.keys(NS)) >= 36                                        # the sample is already embedded
    assert next(x for x in svc.lab()["candidates"] if x["id"] == CAND)["active"]


def test_a_backbone_that_sees_nothing_says_stay(tmp_path, monkeypatch):
    svc, *_ = _lab_svc(tmp_path, monkeypatch, cand_quality="noise")
    tr = svc.run_lab(CAND)
    assert not tr["win"] and "no better" in tr["verdict"]
    with pytest.raises(ValueError):
        svc.lab_switch(CAND)


def test_the_trial_resumes_after_a_stop(tmp_path, monkeypatch):
    svc, _, _, calls, cache = _lab_svc(tmp_path, monkeypatch)

    class Job:
        cancelled, progress = True, None

        def log(self, *_):
            pass
    assert svc.run_lab(CAND, job=Job())["stopped"]
    first = len(cache.keys(NS))
    assert 0 < first < 36
    tr = svc.run_lab(CAND)
    assert tr["win"] and len(cache.keys(NS)) > first


def test_the_lab_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, *_ = _lab_svc(tmp_path, monkeypatch)
    svc.run_lab(CAND)
    c = TestClient(app_mod.create_app(svc.cfg))
    d = c.get("/api/lab").json()
    assert d["active"] == "dinov2_vits14" and d["running"] is None
    assert {x["id"] for x in d["candidates"]} == {"dinov3_vitl16", "dinov3_vith16plus"}
    assert c.post("/api/lab/dinov3_vith16plus/switch").status_code == 409     # never trialled
    assert c.post(f"/api/lab/{CAND}/switch").json()["dino_model"] == CAND


def test_dinov3_names_and_a_clear_licence_error(monkeypatch):
    from peaks import embedding as E

    assert E.dino_cache_name("dinov3_vith16plus") == "dinov3-vith16plus"
    assert E.dino_label("dinov3_vith16plus") == "DINOv3 ViT-H+/16"
    assert {"dinov3_vitl16", "dinov3_vith16plus"} <= set(E.DINO_BACKBONES)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                       float16="f16", float32="f32")

    class AutoModel:
        @staticmethod
        def from_pretrained(repo, **_):
            raise OSError("401 gated repo")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=AutoModel))
    with pytest.raises(E.DinoV3Unavailable, match="HF_TOKEN"):
        E.get_embedder("dino", model_name="dinov3_vitl16")
