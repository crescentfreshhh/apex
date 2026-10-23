"""Picture quality in grading: the tier model tests the measured picture like
performer history (kept only when it helps); the Reject nudge and the trim
list weigh it next to taste (content wins); duplicate choices say why, call
close scores a tie, measure their copies first — and the page can play them."""

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fakestash import FakeStash  # noqa: E402
from peaks.config import Config  # noqa: E402
from peaks.tier_model import (  # noqa: E402
    CLASSES, PIC_FEATURES, TierModel, fit_best, picture_features, quality_features, visual_features,
)

D = 24


def _unit(v):
    return v / np.linalg.norm(v)


# --- the tier model -----------------------------------------------------------------

def test_picture_features_mark_missing_and_vr():
    none, vr = picture_features(None), picture_features({"vr": True, "score": 90})
    assert len(none) == len(PIC_FEATURES) and none[-1] == 1 and vr[-1] == 1
    p = picture_features({"score": 80, "subs": {"detail": 9, "noise": None}, "real": 2160, "upscaled": False})
    assert p[-1] == 0 and p[0] == pytest.approx(0.8) and p[1] == pytest.approx(0.9)
    assert picture_features({"score": None, "rank": 30})[0] == pytest.approx(0.3)   # before calibration


def _library(rng, picture_matters, n_per=30):
    """The look and the file carry nothing; with `picture_matters` each tier has
    its own picture score."""
    level = {"reject": 15, "merveilleuse": 45, "exceptionnelle": 70, "legendaire": 92}
    Xv, Xq, P, y = [], [], [], []
    for c in CLASSES:
        for _ in range(n_per):
            Xv.append(visual_features(np.stack([_unit(rng.standard_normal(D)) for _ in range(6)])))
            Xq.append(quality_features({"mbps": 20, "res": "4K", "fps": 30, "codec": "hevc"}))
            score = level[c] + rng.normal(0, 4) if picture_matters else rng.uniform(10, 95)
            P.append(picture_features({"score": float(score), "subs": {}, "real": 2160}))
            y.append(c)
    return np.stack(Xv), np.stack(Xq), np.stack(P), y


def test_picture_is_kept_when_it_helps():
    Xv, Xq, P, y = _library(np.random.default_rng(0), picture_matters=True)
    model, rep = fit_best(Xv, Xq, y, pic=P)
    assert rep["use_picture"] and rep["picture_gain"] > 0.3 and model.use_picture
    hi = model.summarize(model.predict_proba(Xv[:1], Xq[:1], pic=P[-1:]))[0]      # a Légendaire's picture
    lo = model.summarize(model.predict_proba(Xv[:1], Xq[:1], pic=P[:1]))[0]       # a reject's
    assert hi["expected"] > lo["expected"]
    # unmeasured files still get a prediction
    assert model.predict_proba(Xv[:2], Xq[:2]).shape == (2, len(model.classes_))


def test_picture_is_dropped_when_it_doesnt_help():
    Xv, Xq, P, y = _library(np.random.default_rng(1), picture_matters=False)
    model, rep = fit_best(Xv, Xq, y, pic=P)
    assert rep["use_picture"] is False and not model.use_picture


def test_too_few_measured_files_skip_the_test():
    Xv, Xq, P, y = _library(np.random.default_rng(2), picture_matters=True)
    P[:] = picture_features(None)
    _, rep = fit_best(Xv, Xq, y, pic=P)
    assert "picture_gain" not in rep


def test_models_saved_before_pictures_still_predict():
    Xv, Xq, P, y = _library(np.random.default_rng(3), picture_matters=True)
    m = TierModel(pca_dim=8).fit(Xv, Xq, y)
    del m.__dict__["use_picture"]                                  # a pre-picture pickle
    assert m.predict_proba(Xv[:3], Xq[:3], pic=P[:3]).shape == (3, len(m.classes_))


# --- the Reject nudge and the trim list -----------------------------------------------

@pytest.fixture
def stash():
    return FakeStash({str(i): {"rating100": 100, "o_counter": 16, "title": f"Scene {i}",
                               "created_at": "2024-01-01T00:00:00Z"} for i in range(1, 5)})


@pytest.fixture
def svc(tmp_path, stash, monkeypatch):
    import peaks.web.service as svc_mod

    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    s = svc_mod.Service(cfg)
    s._best_q25, s._best_q50 = 0.30, 0.50
    return s


def _row(svc, sid="1"):
    return svc._cat_update_row(sid)


def test_content_wins_a_good_picture_is_noted_not_a_reprieve(svc):
    sig = {"saves": 0, "best": 0.20, "new": False, "pic": {"score": 78, "band": "Good"}}
    r = svc._recommend(_row(svc), {"keeper": 0.3}, sig)
    assert r["grade"] == "reject" and "(picture is Good · 78)" in r["why"]
    assert "picture Good 78" in r["detail"]


def test_so_so_content_with_a_poor_picture_gets_a_nudge(svc):
    sig = {"saves": 0, "best": 0.45, "new": False, "pic": {"score": 18, "band": "Poor"}}
    r = svc._recommend(_row(svc), {"keeper": 0.4}, sig)
    assert r["grade"] == "reject" and r["why"].startswith("so-so content and a poor picture")
    # the same content with a Watchable picture, or no measure: no nudge (as before)
    assert svc._recommend(_row(svc), {"keeper": 0.4}, {**sig, "pic": {"score": 40, "band": "Watchable"}}) is None
    assert svc._recommend(_row(svc), {"keeper": 0.4}, {k: v for k, v in sig.items() if k != "pic"}) is None
    # never for a scene the model thinks is a keeper, a new one, or one with saves
    assert svc._recommend(_row(svc), {"keeper": 0.8}, sig) is None
    assert svc._recommend(_row(svc), {"keeper": 0.4}, {**sig, "new": True}) is None
    assert svc._recommend(_row(svc), {"keeper": 0.4}, {**sig, "saves": 1})["grade"] == "legendaire"


def test_trim_puts_the_poor_picture_first_among_similar_taste(svc):
    rows = [_row(svc, s) for s in ("1", "2", "3")]
    sig = {"1": {"saves": 0, "best": 0.40, "new": False, "pic": {"score": 80, "band": "Good"}},
           "2": {"saves": 0, "best": 0.42, "new": False, "pic": {"score": 15, "band": "Poor"}},
           "3": {"saves": 0, "best": 0.41, "new": False}}
    order = [r["scene_id"] for r in svc._triage_raw("trim", rows, {}, {}, sig)]
    assert order == ["2", "1", "3"]


# --- duplicates -----------------------------------------------------------------------

def _trusted(monkeypatch, svc, scores):
    monkeypatch.setattr(type(svc), "picture_trusted", lambda self: True)
    monkeypatch.setattr(type(svc), "picture_scores", lambda self: scores)


def test_close_picture_scores_are_a_tie_decided_by_the_file(svc, stash, monkeypatch):
    stash.s["1"].update(width=3840, height=2160)
    stash.s["2"].update(width=1920, height=1080)
    svc.invalidate_meta()
    rows = [_row(svc, "1"), _row(svc, "2")]
    _trusted(monkeypatch, svc, {"1": {"score": 70}, "2": {"score": 73}})
    keep, by, why = svc._dupe_keeper_why(rows)
    assert (keep, by) == ("1", "tie") and why.startswith("pictures are a tie (73 vs 70)")


def test_a_clear_picture_gap_decides_and_says_why(svc, monkeypatch):
    rows = [_row(svc, "1"), _row(svc, "2")]
    _trusted(monkeypatch, svc, {"1": {"score": 40, "reasons": ["soft"]},
                                "2": {"score": 81, "reasons": ["real 4K detail", "clean"]}})
    keep, by, why = svc._dupe_keeper_why(rows)
    assert (keep, by) == ("2", "picture") and why == "best picture: 81 vs 40 — real 4K detail"
    g = svc._dupe_group(rows, {})
    assert g["keep_why"] == why and all(r["stream"] for r in g["scenes"])


def test_duplicate_copies_are_measured_first(svc, monkeypatch):
    monkeypatch.setattr(type(svc), "_pic_needs", lambda self, r: True)
    monkeypatch.setattr(type(svc), "last_ingest", lambda self: {"new": ["1"]})
    svc._dupe_cache = {"groups": [{"scenes": [{"scene_id": "3"}, {"scene_id": "4"}]}]}
    owed, _ = svc._pic_todo()
    assert [r["scene_id"] for r in owed][:2] == ["3", "4"] and {r["scene_id"] for r in owed} == {"1", "3", "4"}


def test_measure_a_groups_copies_now(svc, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    measured = []
    monkeypatch.setattr(type(svc), "_pic_needs", lambda self, r: r["scene_id"] != "2")
    monkeypatch.setattr(type(svc), "picture_measure_row",
                        lambda self, r: measured.append(r["scene_id"]) or {"key": r["scene_id"]})
    monkeypatch.setenv("PEAKS_WARM_ON_START", "0")
    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(app_mod.create_app(svc.cfg))
    assert api.post("/api/duplicates/measure", json={"scene_ids": []}).status_code == 400
    job = api.post("/api/duplicates/measure", json={"scene_ids": ["1", "2", "3"]}).json()
    import time
    for _ in range(200):
        j = api.get(f"/api/jobs/{job['id']}").json()
        if j["status"] != "running":
            break
        time.sleep(0.02)
    assert j["status"] == "done" and j["result"]["measured"] == 2
    assert sorted(measured) == ["1", "3"]                     # 2 was already measured
