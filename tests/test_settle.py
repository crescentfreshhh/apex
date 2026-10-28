"""Settling the library: the 9-in-10 threshold, what's left to ask, the order
questions come in, rulings, and the progress history."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402

pytest.importorskip("sklearn")


def _library(tmp_path, monkeypatch, n_cluster=8, n_wrong=1, mix=0.8, br=33e6):
    """The triage library plus unsure scenes: `n_cluster` near-identical ones that
    look half Exceptionnelle, half Légendaire, and one lone unsure scene that looks
    like nothing else; and `n_wrong` Exceptionnelle-looking scenes graded Légendaire."""
    from peaks.cache import EmbeddingCache

    svc, stash = _triage_svc(tmp_path, monkeypatch)
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


def test_what_is_left_and_the_order_questions_come_in(tmp_path, monkeypatch):
    svc, stash, (wrong,), new = _library(tmp_path, monkeypatch)
    st = svc.settle_state()
    assert st["ok"] and st["disagree"] == 1 and st["unsure"] >= 5
    assert st["unsettled"] == st["unsure"] + st["disagree"]
    rows = svc._catalogue_all()
    q = svc.settle_queue(rows, svc._tier_predictions(rows))
    ids = [r["scene_id"] for r in q]
    assert ids[0].startswith("c")                          # representative + unsure first
    assert ids.index("lone") < ids.index(sorted((i for i in ids if i.startswith("c")), key=ids.index)[-1])  # spread out
    assert wrong in ids and ids.index(wrong) == 4          # every 5th question is a disagreement
    info = svc.settle_info(wrong)
    assert info["kind"] == "disagree" and info["peaks"] == "exceptionnelle" and "You said" in info["why"]
    assert svc.settle_info(ids[0])["kind"] == "unsure" and "Peaks unsure" in svc.settle_info(ids[0])["why"]
    assert {"u1", "u2"}.isdisjoint(ids)                      # Peaks is sure about these: settled


def test_a_ruling_settles_a_disagreement_until_you_regrade(tmp_path, monkeypatch):
    svc, stash, (wrong,), _ = _library(tmp_path, monkeypatch)
    before = svc.settle_state()["unsettled"]
    svc.settle_rule(wrong)
    assert svc.settle_state()["disagree"] == 0 and svc.settle_state()["unsettled"] == before - 1
    svc.settle_unrule(wrong)
    assert svc.settle_state()["disagree"] == 1
    svc.settle_rule(wrong)
    stash.s[wrong]["o_counter"] = 17                        # graded again → the ruling no longer applies…
    svc._cat_cache = None
    svc.__dict__.pop("_cat_cache", None)
    st = svc.settle_state()
    assert st["disagree"] == 0                               # … and the new grade agrees with Peaks


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
    assert cat["views"]["settle"] == d["unsettled"] and cat["items"][0]["settle"]["kind"] == "unsure"
    assert c.post(f"/api/settle/rule/{wrong}").json()["ruled"]
    assert c.get("/api/settle").json()["disagree"] == 0
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
