"""Quick guesses between retrains: a grade nudges the tier model's last layer and
the ungraded scenes are re-scored at once, marked `quick`; honest numbers
(Settle, the Arbiter, the report's accuracy) never read them; the next honest
retrain replaces them."""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

pytest.importorskip("sklearn")

from test_catalogue import _triage_svc  # noqa: E402


def _regrade(svc, stash, sid, rating=100, o=18):
    stash.s[sid].update(rating100=rating, o_counter=o)
    return svc._cat_put_row(sid, stash.s[sid])


def test_a_grade_moves_the_lookalikes_guess_at_once(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    svc.train_tier_model()
    rows = svc._catalogue_all()
    before = svc._tier_predictions(rows)["u2"]                      # looks exceptionnelle
    assert before["tier"] == "exceptionnelle" and not before.get("quick")
    held = svc._tier_preds[1]
    took = []
    for sid in [r["scene_id"] for r in rows if r["tier"] == "exceptionnelle"][:6]:
        row = _regrade(svc, stash, sid)                             # you call that look Légendaire now
        t = time.perf_counter()
        svc._quick_learn(sid, row)
        took.append(time.perf_counter() - t)
    after = svc._tier_preds[1]
    assert after is not held                                        # a new dict: the views recompute
    assert after["u2"]["quick"] and after["u2"]["probs"]["legendaire"] > before["probs"]["legendaire"] + 0.3
    assert max(took) < 0.05                                         # well under 50 ms
    graded = [r["scene_id"] for r in rows if r["tier"] in ("legendaire", "merveilleuse")]
    assert not any(after[s].get("quick") for s in graded if s in after)   # graded scenes keep their honest verdict


def test_honest_numbers_never_read_quick_guesses(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    svc.train_tier_model()
    rows = svc._catalogue_all()
    svc._tier_predictions(rows)
    rep = dict(svc.tier_model_status()["report"]["cv"])
    oof = dict(svc._tier_model_state()["model"].oof)
    for sid in [r["scene_id"] for r in rows if r["tier"] == "exceptionnelle"][:6]:
        svc._quick_learn(sid, _regrade(svc, stash, sid))
    assert svc.tier_model_status()["report"]["cv"] == rep and svc._tier_model_state()["model"].oof == oof
    preds = svc._tier_preds[1]
    quick = {s for s, p in preds.items() if p.get("quick")}
    assert quick
    out = svc._settle_classify(svc._catalogue_all(), preds, {"t": 0.0})
    asked = {x["scene_id"] if isinstance(x, dict) else x for k in ("disagree", "drain") for x in out.get(k, [])}
    assert not (asked & quick)                                      # Settle never asks about a quick guess


def test_the_honest_retrain_replaces_the_quick_guesses(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    svc.train_tier_model()
    rows = svc._catalogue_all()
    svc._tier_predictions(rows)
    sid = [r["scene_id"] for r in rows if r["tier"] == "exceptionnelle"][0]
    svc._quick_learn(sid, _regrade(svc, stash, sid))
    assert any(p.get("quick") for p in svc._tier_preds[1].values())
    svc.train_tier_model()
    svc._tier_preds = None
    fresh = svc._tier_predictions(svc._catalogue_all())
    assert not any(p.get("quick") for p in fresh.values())


def test_a_grade_without_a_model_or_row_is_a_no_op(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    assert svc._quick_learn("u1", svc._catalogue_all()[0]) is None    # no model yet
    svc.train_tier_model()
    svc._tier_predictions(svc._catalogue_all())
    assert svc._quick_learn("nope", {"tier": "legendaire"}) is None   # not a scored scene
    assert svc._quick_learn("u1", {"tier": "unreviewed"}) is None      # not a training grade


def test_the_quick_head_handles_two_classes():
    from types import SimpleNamespace

    from peaks.quick_head import QuickHead

    clf = SimpleNamespace(coef_=np.array([[1.0, -1.0]]), intercept_=np.array([0.0]))
    head = QuickHead(SimpleNamespace(_clf=clf, classes_=["reject", "legendaire"]))
    x = np.array([[0.2, 0.1]])
    p0 = head.proba(x)[0, 1]
    assert abs(p0 - 1 / (1 + np.exp(-0.1))) < 1e-9                   # same as the binary logistic
    head.learn(x[0], "reject")
    assert head.proba(x)[0, 1] < p0
