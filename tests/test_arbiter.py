"""The Arbiter (honest verdicts + the argument), moment checks, and the shrink trendline."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402

pytest.importorskip("sklearn")


def _planted(tmp_path, monkeypatch, n_wrong=1, studio=None, studio_n=0):
    """The triage library with `n_wrong` Exceptionnelle-looking scenes graded
    Légendaire; `studio` goes on the first `studio_n` Exceptionnelle-looking ones."""
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    exc = [sid for sid, v in stash.s.items() if v["o_counter"] == 17]
    for sid in exc[:n_wrong]:
        stash.s[sid]["o_counter"] = 18
    for sid in exc[:studio_n]:
        stash.s[sid]["studio"] = studio
    rep = svc.train_tier_model()
    assert rep["trained"]
    return svc, stash, exc[:n_wrong]


def test_honest_verdicts_catch_a_mislabel(tmp_path, monkeypatch):
    svc, stash, (wrong,) = _planted(tmp_path, monkeypatch)
    m = svc._tier_model_state()["model"]
    assert m.oof[wrong]["tier"] == "exceptionnelle"           # judged without its grade
    assert len(m.folds) >= 2
    rows = svc._catalogue_all()
    preds = svc._tier_predictions(rows)
    assert preds[wrong]["honest"] and preds[wrong]["tier"] == "exceptionnelle"
    assert not preds["u1"].get("honest")                        # not graded: the full model

    e = svc.tier_explain(wrong)
    assert e["ok"] and e["honest"] and e["yours"] == "legendaire"
    assert e["verdict"] == "exceptionnelle" and e["agree"] == "lower"
    total = sum(g["push"] for g in e["groups"]) + e["baseline"]
    assert abs(total - e["gap"]) < 0.01                         # the parts add up to the verdict
    assert e["gap"] > 0
    assert {g["group"] for g in e["groups"]} >= {"look", "file"}
    looks = e["lookalikes"]
    exc_ids = {sid for sid, v in stash.s.items() if v["o_counter"] == 17}
    assert {x["scene_id"] for x in looks["exceptionnelle"]} <= exc_ids
    assert e["moments"]["exceptionnelle"] and e["moments"]["legendaire"]
    assert e["words"] == {}                                     # no CLIP here: no words, not noise
    assert "look" in e["summary"]


def test_arbiter_summary_grid_and_cell(tmp_path, monkeypatch):
    svc, stash, (wrong,) = _planted(tmp_path, monkeypatch)
    s = svc.arbiter_summary()
    assert s["ok"] and s["grid"]["legendaire"]["exceptionnelle"] >= 1
    assert 0 < s["agree"] < 1 and s["calibration"]
    assert s["drivers"]["groups"][0]["group"] == "look"
    d = svc.catalogue(view="arbiter", cell="legendaire>exceptionnelle")
    assert wrong in [r["scene_id"] for r in d["items"]]


def test_disagreements_cluster_by_studio(tmp_path, monkeypatch):
    # two of Slick's five scenes are graded a tier above how they look
    svc, stash, wrong = _planted(tmp_path, monkeypatch, n_wrong=2, studio="Slick", studio_n=5)
    clusters = svc.arbiter_summary()["clusters"]
    top = next((c for c in clusters if c["value"] == "Slick"), None)
    assert top and top["way"] == "lower" and top["n"] == 5 and top["same_way"] >= 2


# --- moments ----------------------------------------------------------------------------

def test_live_moment_checks_and_pick_approval(tmp_path, monkeypatch):
    svc, _ = _triage_svc(tmp_path, monkeypatch)
    monkeypatch.setattr(type(svc), "scene_picks",
                        lambda self, sid, n=4: [{"t": 120.0, "score": 0.9, "key": "k0"},
                                                {"t": 30.0, "score": 0.7, "key": "k0"}])
    svc.moment_check_on_save("0", 125.0)            # by the first pick
    svc.moment_check_on_save("0", 36.0)             # by the second
    svc.moment_check_on_save("0", 250.0)            # nowhere near
    svc.record_pick_feedback("0", 120.0, 1)
    svc.record_pick_feedback("0", 30.0, 0)
    m = svc.moment_insights()
    assert m["live"]["n"] == 3 and m["live"]["top1"] == round(1 / 3, 3) and m["live"]["top3"] == round(2 / 3, 3)
    assert m["approval"] == {"n": 2, "rate": 0.5, "by_month": m["approval"]["by_month"]}


def test_no_taste_means_no_check(tmp_path, monkeypatch):
    svc, _ = _triage_svc(tmp_path, monkeypatch)
    monkeypatch.setattr(type(svc), "scene_picks", lambda self, sid, n=4: [{"t": 120.0, "score": None}])
    svc.moment_check_on_save("0", 120.0)            # evenly spaced frames aren't picks
    assert svc.moment_insights()["live"]["n"] == 0


def test_estimate_leaves_each_scenes_own_saves_out(tmp_path, monkeypatch):
    import numpy as np

    svc, _ = _triage_svc(tmp_path, monkeypatch)
    idx = svc.index(svc._model_name())
    k0 = next(k for k, m in idx.key_meta.items() if m["scene_id"] == "0")
    k1 = next(k for k, m in idx.key_meta.items() if m["scene_id"] == "1")
    vecs = np.stack([idx.vector_at(k0, 30.0), idx.vector_at(k1, 60.0)])
    src = [{"key": k0, "time": 30.0, "scene_id": "0", "kind": "apex"},
           {"key": k1, "time": 60.0, "scene_id": "1", "kind": "apex"}]
    monkeypatch.setattr(type(svc), "_taste_sources", lambda self, model, rebuild=False, profile=None: (vecs, src))
    est = svc._mc_estimate()
    assert est["scenes"] == 2 and est["saves"] == 2
    # alone, a scene's own save would be its own nearest moment; with it left out there's nothing to score
    monkeypatch.setattr(type(svc), "_taste_sources", lambda self, model, rebuild=False, profile=None: (vecs[:1], src[:1]))
    svc._mc_est_cache = None
    assert svc._mc_estimate()["scenes"] == 0


def test_pick_answer_is_a_label_and_a_record(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, _ = _triage_svc(tmp_path, monkeypatch)
    app = app_mod.create_app(svc.cfg)
    c = TestClient(app)
    r = c.post("/api/label", params={"t": 30.0, "label": 1, "scene_id": "0", "pick": "true"})
    assert r.status_code == 200 and r.json()["positive"] >= 1
    s = app.state.service
    assert s.moment_insights()["approval"]["n"] == 1
    assert c.get("/api/scene/0/picks").status_code == 200


# --- the shrink trendline ------------------------------------------------------------------

def test_trend_backfill_and_daily_snapshot(tmp_path, monkeypatch):
    from datetime import date, timedelta

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    today = date.today()
    old = (today - timedelta(days=20)).isoformat()
    for sid, v in stash.s.items():
        v["created_at"] = f"{old}T10:00:00Z"
        v["size"] = 1_000_000_000
    svc.catalogue(refresh=True)
    gone_day = (today - timedelta(days=10)).isoformat()
    svc.action_log().path.parent.mkdir(parents=True, exist_ok=True)
    with svc.action_log().path.open("a") as fh:
        fh.write(json.dumps({"ts": f"{gone_day}T09:00:00", "action": "delete", "size": 5_000_000_000,
                             "file_deleted": True}) + "\n")
    t = svc.library_trend(days=30)
    by_d = {p["d"]: p for p in t["points"]}
    n = len(stash.s)
    before = by_d[(today - timedelta(days=15)).isoformat()]
    assert before["rec"] and before["scenes"] == n + 1 and before["gb"] == n + 5.0   # the deleted one was here
    after = by_d[(today - timedelta(days=5)).isoformat()]
    assert after["scenes"] == n
    assert not by_d[today.isoformat()]["rec"]                   # today: recorded, not reconstructed
    assert t["freed"][-1]["gb"] == 5.0
    assert svc._trend_snapshot(svc._catalogue_all()) is False   # once a day
