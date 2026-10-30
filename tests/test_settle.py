"""Settling the library — graded scenes only: the 9-in-10 threshold, what's left
to ask (disagreements and scenes in a retiring tier), the order, rulings, history."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402

pytest.importorskip("sklearn")


def _library(tmp_path, monkeypatch, n_cluster=8, n_wrong=1, mix=0.8, br=33e6, drain=True):
    """The triage library (Merveilleuse draining unless `drain=False`), plus unreviewed
    scenes Peaks can't call — `n_cluster` look-alikes between Exceptionnelle and
    Légendaire and one lone odd one — and `n_wrong` Exceptionnelle-looking scenes
    graded Légendaire."""
    from peaks.cache import EmbeddingCache

    svc, stash = _triage_svc(tmp_path, monkeypatch, settings={"draining_tiers": ["merveilleuse"]} if drain else None)
    cache = EmbeddingCache(svc.cfg.embedding.cache_dir)
    keys = {cache.peek_meta(k, "dinov2")["scene_id"]: k for k in cache.keys("dinov2")}
    look = lambda sid: cache.load(keys[sid], "dinov2")[1].mean(axis=0)  # noqa: E731
    exc = [sid for sid, v in stash.s.items() if v["o_counter"] == 17]
    leg = [sid for sid, v in stash.s.items() if v["o_counter"] == 18]
    mid = mix * look(exc[0]) + (1 - mix) * look(leg[0])
    rng = np.random.default_rng(3)
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)  # noqa: E731
    odd = unit(mid + 1.2 * unit(rng.standard_normal(mid.shape)))
    new = {f"c{i}": unit(mid) for i in range(n_cluster)} | {"lone": odd}
    for i, (sid, base) in enumerate(new.items()):
        frames = unit(base + 0.05 * rng.standard_normal((10, len(base)))).astype(np.float32)
        cache.save(f"n{i}", "dinov2", np.arange(10, dtype=np.float32) * 30, frames, meta={"scene_id": sid})
        stash.s[sid] = {"rating100": None, "o_counter": 0, "width": 3840, "height": 2160, "bit_rate": br,
                        "frame_rate": 30, "video_codec": "hevc", "date": "2025-01-01",
                        "tag_ids": [], "organized": False}
    for sid in exc[:n_wrong]:
        stash.s[sid]["o_counter"] = 18
    assert svc.train_tier_model()["trained"]
    return svc, stash, exc[:n_wrong], list(new)


def test_calibration_finds_the_nine_in_ten_threshold():
    from peaks.web.settle import FALLBACK, calibrate

    oof = {}
    for i in range(40):                                   # sure and right
        oof[f"a{i}"] = {"tier": "legendaire", "graded": "legendaire", "conf": 0.95 - i * 0.005}
    for i in range(20):                                   # unsure and often wrong
        oof[f"b{i}"] = {"tier": "legendaire", "graded": "reject" if i % 2 else "legendaire", "conf": 0.5 - i * 0.01}
    c = calibrate(oof)
    assert c["measured"] and 0.35 < c["t"] < 0.5 and c["accuracy"] >= 0.9 and 40 < c["above"] < 60   # the lowest still 9-in-10
    assert calibrate({"x": {"tier": "a", "graded": "a", "conf": 0.9}}) == {"t": FALLBACK, "measured": False, "n": 1}


def test_spread_order_spreads_the_picks():
    from peaks.web.settle import spread_order

    X = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], np.float32)
    assert spread_order(np.array([1.0, 0.99, 0.98, 0.6]), X, 2) == [0, 3]       # not three look-alikes


def test_only_graded_scenes_are_questions(tmp_path, monkeypatch):
    """Every honest disagreement and every scene left in a retiring tier — and no
    unreviewed scene, however unsure Peaks is about it."""
    from peaks.tier_model import train_class

    svc, stash, (wrong,), new = _library(tmp_path, monkeypatch)
    rows = svc._catalogue_all()
    preds = svc._tier_predictions(rows)
    merv = [r["scene_id"] for r in rows if r["tier"] == "merveilleuse"]
    expect = {r["scene_id"] for r in rows if train_class(r["tier"]) and r["scene_id"] in preds
              and preds[r["scene_id"]]["tier"] != train_class(r["tier"])}
    st = svc.settle_state()
    assert st["ok"] and wrong in expect and st["disagree"] == len(expect)
    assert st["drain"] == len(merv) > 0 and st["unsettled"] == st["disagree"] + st["drain"]
    assert st["agree"] + st["disagree"] == sum(1 for r in rows if train_class(r["tier"]) and r["scene_id"] in preds)
    q = svc.settle_queue(rows, preds)
    ids = [r["scene_id"] for r in q]
    assert set(ids) == expect | set(merv)
    assert set(new).isdisjoint(ids) and {"u1", "u2", "u3"}.isdisjoint(ids)     # unreviewed: never
    assert ids.index(wrong) <= 1                                               # interleaved, not last
    info = svc.settle_info(wrong)
    assert info["kind"] == "disagree" and info["peaks"] == "exceptionnelle" and "You said" in info["why"]
    d = svc.settle_info(merv[0])
    assert d["kind"] == "drain" and "retiring" in d["why"] and d["peaks"] in ("exceptionnelle", "legendaire")


def test_every_disagreement_counts_sure_or_leaning(tmp_path, monkeypatch):
    import peaks.web.settle as settle

    svc, stash, (wrong,), _ = _library(tmp_path, monkeypatch, drain=False)
    rows = svc._catalogue_all()
    monkeypatch.setattr(settle, "calibrate", lambda oof: {"t": 0.999, "measured": True, "accuracy": 0.9, "above": 20, "n": 36})
    st = svc.settle_state()
    assert st["disagree"] >= 1 and st["disagree_sure"] == 0                    # leaning ones count too
    svc.settle_queue(rows, svc._tier_predictions(rows))
    assert svc.settle_info(wrong)["sure"] is False and "leaning" in svc.settle_info(wrong)["why"]
    monkeypatch.setattr(settle, "calibrate", lambda oof: {"t": 0.0, "measured": True, "accuracy": 0.9, "above": 20, "n": 36})
    assert svc.settle_state()["disagree_sure"] == svc.settle_state()["disagree"]
    svc.settle_queue(rows, svc._tier_predictions(rows))
    assert "sure" in svc.settle_info(wrong)["why"]


def test_a_ruling_settles_a_disagreement_until_you_regrade(tmp_path, monkeypatch):
    svc, stash, (wrong,), _ = _library(tmp_path, monkeypatch)
    before = svc.settle_state()
    svc.settle_rule(wrong)
    after = svc.settle_state()
    assert after["disagree"] == before["disagree"] - 1 and after["unsettled"] == before["unsettled"] - 1
    svc.settle_unrule(wrong)
    assert svc.settle_state()["disagree"] == before["disagree"]
    svc.settle_rule(wrong)
    stash.s[wrong]["o_counter"] = 17                        # graded again → the ruling no longer applies…
    svc._cat_cache = None
    svc.__dict__.pop("_cat_cache", None)
    st = svc.settle_state()
    assert st["disagree"] == before["disagree"] - 1          # … and the new grade agrees with Peaks


def test_history_projection_and_the_api(tmp_path, monkeypatch):
    import json
    import time

    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, stash, (wrong,), _ = _library(tmp_path, monkeypatch)
    p = svc._settle_history_path()
    now = time.time()
    p.write_text("".join(json.dumps({"at": now - 86400 * (3 - i), "unsettled": 40 - 10 * i, "answers": 20 * i}) + "\n"
                         for i in range(4)))
    st = svc.settle_state()
    assert st["projection"] and st["projection"]["per_answer"] == 0.5
    assert st["projection"]["answers"] == int(np.ceil(st["unsettled"] / 0.5))
    c = TestClient(app_mod.create_app(svc.cfg))
    d = c.get("/api/settle").json()
    assert d["ok"] and d["unsettled"] >= 1 and d["threshold"]["t"] > 0
    cat = c.get("/api/catalogue", params={"view": "settle", "limit": 50}).json()
    assert cat["views"]["settle"] == d["unsettled"] and {x["settle"]["kind"] for x in cat["items"]} == {"disagree", "drain"}
    assert c.post(f"/api/settle/rule/{wrong}").json()["ruled"]
    assert c.get("/api/settle").json()["disagree"] == d["disagree"] - 1
    assert c.delete(f"/api/settle/rule/{wrong}").json()["ruled"] is False
    assert c.post("/api/settle/rule/nope").status_code == 404


def test_settling_retrains_sooner(tmp_path, monkeypatch):
    import peaks.web.settle as settle

    svc, *_ = _library(tmp_path, monkeypatch)
    started = []
    monkeypatch.setattr(type(svc), "_train_tier_quietly", lambda self: started.append(1))
    rows = svc._catalogue_all()
    svc.settle_queue(rows, svc._tier_predictions(rows))    # opening the list = settling
    assert svc.settling()
    for _ in range(settle.FAST_RETRAIN):
        svc._note_grade()
    import time
    for _ in range(50):
        if started:
            break
        time.sleep(0.02)
    assert started


def test_answering_by_grading_is_a_ruling_and_undo_takes_it_back(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, stash, (wrong,), _ = _library(tmp_path, monkeypatch, drain=False)
    c = TestClient(app_mod.create_app(svc.cfg))
    before = c.get("/api/settle").json()["disagree"]
    grade = lambda g, settle: c.post("/api/catalogue/grade", json={"scene_id": wrong, "grade": g, "settle": settle}).json()  # noqa: E731
    r = grade("legendaire", False)                               # graded from the Catalogue: not an answer
    assert not r.get("settled") and c.get("/api/settle").json()["disagree"] == before
    r = grade("legendaire", True)                                # re-confirmed in Settle: your grade stands
    assert r["settled"] and c.get("/api/settle").json()["disagree"] == before - 1
    svc.train_tier_model()                                       # still settled after a retrain
    assert c.get("/api/settle").json()["disagree"] == before - 1
    c.delete(f"/api/settle/rule/{wrong}")                        # what undo does
    assert c.get("/api/settle").json()["disagree"] == before
    r = grade("merveilleuse", True)                              # a third grade, neither yours nor Peaks'
    assert r["settled"] and c.get("/api/settle").json()["disagree"] == before - 1
    assert svc.rulings()[wrong]["grade"] == "merveilleuse"
    r = c.post("/api/catalogue/grade", json={"scene_id": wrong, "grade": "reject", "no_learn": True,
                                              "settle": True}).json()       # "reject, don't learn": never a question
    assert r["settled"] is False
