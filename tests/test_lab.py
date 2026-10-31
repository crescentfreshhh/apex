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
    assert {x["id"] for x in d["candidates"]} == {"dinov3_vitl16", "dinov3_vith16plus", "siglip2_so400m", "pe_core_l"}
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


def test_the_sample_is_held_at_half_precision_without_changing_the_verdict(tmp_path, monkeypatch):
    """The trial keeps the sample in float16 (half the memory) and widens one scene
    at a time to score — the same numbers as scoring float32 copies."""
    svc, _, saved, _, _ = _lab_svc(tmp_path, monkeypatch)
    rows = svc.lab_sample()
    svc._lab_embed(CAND, rows)
    half = svc._lab_vectors(NS, rows)
    assert half and all(v.dtype == np.float16 for _, v in half.values())
    full = {k: (t, v.astype(np.float32)) for k, (t, v) in half.items()}
    graded = [r for r in rows if r["scene_id"] in half]
    g16, g32 = svc._lab_grading(half, graded), svc._lab_grading(full, graded)
    assert g16["accuracy"] == g32["accuracy"] and abs(g16["log_loss"] - g32["log_loss"]) < 1e-3
    m16, m32 = svc._lab_moments(half, saved, set(half)), svc._lab_moments(full, saved, set(half))
    assert {k: m16[k] for k in ("top1", "top3", "tight")} == {k: m32[k] for k in ("top1", "top3", "tight")}


def test_many_saved_scenes_in_one_tier_never_crowd_out_the_others(tmp_path, monkeypatch):
    """Your saves are mostly top-tier scenes; with more of them than the cap, the
    sample used to be nearly all one tier — and grading couldn't be judged."""
    import peaks.web.lab as lab
    from peaks.tier_model import train_class

    svc, _, _, _, _ = _lab_svc(tmp_path, monkeypatch)
    rows = svc._catalogue_all()
    top = [r["scene_id"] for r in rows if train_class(r["tier"]) == "legendaire"]
    monkeypatch.setattr(type(svc), "_zoom_saved", lambda self: {sid: [60.0] for sid in top})
    monkeypatch.setattr(lab, "PER_TIER", 10)
    monkeypatch.setattr(lab, "SAMPLE_CAP", len(top))             # as many saves as the whole cap
    sample = svc.lab_sample()
    per = {}
    for r in sample:
        c = train_class(r["tier"])
        per[c] = per.get(c, 0) + 1
    share = min(10, len(top) // 3)
    assert len(sample) <= len(top)
    assert len([c for c, n in per.items() if c]) == 3 and all(n >= share for c, n in per.items() if c), per
    # the old order: saves first, then the cut — nothing but the saves' tier was left


def test_an_interrupted_trial_resumes_counting_what_is_already_embedded(tmp_path, monkeypatch):
    """A re-pull or a stop mid-trial keeps every scene already embedded; the next
    run says so and its progress starts there, not at zero."""
    import peaks.pipeline as pipeline
    import peaks.web.service as svc_mod

    svc, _, _, _, cache = _lab_svc(tmp_path, monkeypatch)

    class Job:
        cancelled, progress = False, None

        def __init__(self):
            self.lines = []

        def log(self, m):
            self.lines.append(m)
    j = Job()
    j.cancelled = True
    svc.run_lab(CAND, job=j)                                   # stopped part-way: some scenes embedded
    first = len(cache.keys(NS))
    assert 0 < first

    # the real _lab_embed (the fixture fakes it), with the embedding itself stubbed
    from peaks.web.lab import LabMixin
    monkeypatch.setattr(svc_mod.Service, "_lab_embed", LabMixin._lab_embed)
    monkeypatch.setattr(svc_mod.Service, "_lab_embedder", lambda self, cand: type("E", (), {"name": NS})())
    sig = svc._lab_sampler()
    sig = getattr(sig, "interval_signature", sig.interval)
    for k in cache.keys(NS):                                    # the fixture's entries, at the library's sampling
        t, v, meta = cache.load(k, NS)
        cache.save(k, NS, t, v, meta={**meta, "interval": sig})
    seen = {}

    def fake_library(scenes, sampler, embedder, cache_, **kw):
        seen["start"] = dict(j2.progress)
        return {"embedded": 0, "frames": 0}
    monkeypatch.setattr(pipeline, "embed_library", fake_library)
    j2 = Job()
    rows = svc.lab_sample()
    svc._lab_embed(CAND, rows, job=j2)
    idx = svc.index(svc._model_name())
    key_of = {str(m.get("scene_id")): k for k, m in idx.key_meta.items()}
    have = sum(1 for r in rows if key_of.get(r["scene_id"]) and cache.has(key_of[r["scene_id"]], NS, interval=sig))
    assert have > 0 and seen["start"]["done"] == have
    assert any(m.startswith(f"resuming — {have} of") for m in j2.lines)
    card = next(c for c in svc.lab()["candidates"] if c["id"] == CAND)
    assert card["sample"]["done"] == have and card["sample"]["total"] == len(rows)


def test_broken_nan_frames_are_redone_and_never_judged(tmp_path, monkeypatch):
    """DINOv3 in float16 overflowed and cached NaN frames; a trial drops those
    scenes so they're embedded again, and judging never sees a NaN."""
    svc, _, _, _, cache = _lab_svc(tmp_path, monkeypatch)
    svc.run_lab(CAND)
    keys = cache.keys(NS)
    t, v, meta = cache.load(keys[0], NS)
    v = np.array(v, np.float32)
    v[2] = np.nan
    cache.save(keys[0], NS, t, v, meta=meta)
    sid = meta["scene_id"]
    vecs = svc._lab_vectors(NS, [{"scene_id": sid}])
    assert np.isfinite(vecs[sid][1].astype(np.float32)).all() and len(vecs[sid][0]) == len(t) - 1
    from peaks.web.lab import LabMixin
    assert LabMixin._lab_drop_broken(cache, NS, keys) == 1 and not cache.has(keys[0], NS)
    assert LabMixin._lab_drop_broken(cache, NS, keys) == 0
    tr = svc.run_lab(CAND)                                       # judged fine; the scene embedded again
    assert "verdict" in tr and cache.has(keys[0], NS)


def test_image_text_candidates_are_registered(monkeypatch):
    import types

    from peaks import embedding as E
    from peaks.web.lab import CANDIDATES

    assert {"siglip2_so400m", "pe_core_l"} <= set(CANDIDATES) <= set(E.DINO_BACKBONES)
    assert E.dino_label("siglip2_so400m") == "SigLIP 2 So400m" and E.dino_label("pe_core_l") == "Perception Encoder L"
    assert E.dino_cache_name("siglip2_so400m") == "siglip2-so400m" and E.dino_cache_name("pe_core_l") == "pe-core-l"
    made = []
    monkeypatch.setattr(E, "OpenClipVisualEmbedder", lambda name, device=None: made.append(name) or "emb")
    assert E.make_dino("siglip2_so400m") == "emb" and made == ["siglip2_so400m"]
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "open_clip", None)              # an image without open_clip
    with pytest.raises(RuntimeError, match="open_clip"):
        E.OpenClipVisualEmbedder("pe_core_l")


def test_trial_data_is_listed_and_deleted_on_request(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, _, _, _, cache = _lab_svc(tmp_path, monkeypatch)
    svc.run_lab(CAND)
    c = TestClient(app_mod.create_app(svc.cfg))
    d = c.get("/api/lab/data").json()
    row = next(x for x in d["items"] if x["id"] == CAND)
    assert row["scenes"] == len(cache.keys(NS)) > 0 and row["bytes"] > 0 and row["deletable"]
    lab = c.get("/api/lab").json()
    card = next(x for x in lab["candidates"] if x["id"] == CAND)
    assert card["licence"] and not next(x for x in lab["candidates"] if x["id"] == "siglip2_so400m")["licence"]
    assert c.delete(f"/api/lab/{svc._active_dino_model()}/data").status_code in (404, 409)   # never the active model
    r = c.delete(f"/api/lab/{CAND}/data").json()
    assert r["freed_bytes"] == row["bytes"] and cache.keys(NS) == []
    card = next(x for x in c.get("/api/lab").json()["candidates"] if x["id"] == CAND)
    assert card["trial"] and card["data_deleted"]                                        # the result stays
    assert all(x["id"] != CAND for x in c.get("/api/lab/data").json()["items"])
    assert c.delete("/api/lab/nope/data").status_code == 404
