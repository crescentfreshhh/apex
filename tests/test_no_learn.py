"""'Reject, don't learn': 1★ like any reject, but left out of everything Peaks
learns from — the tier model, performer/studio records, taste, the reject
memory, and Settle's questions. Any other grade (or an undo) clears it."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402

pytest.importorskip("sklearn")


def _svc(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    for sid in ("0", "1", "2"):
        stash.s[sid]["fingerprint"] = f"fp{sid}"
        stash.s[sid]["performers_detail"] = [{"id": "p1", "name": "Ava"}]
        stash.s[sid]["performer_ids"] = ["p1"]
    return svc, stash


def _row(svc, sid):
    return next(r for r in svc._catalogue_all() if r["scene_id"] == sid)


def test_a_no_learn_reject_is_a_reject_that_teaches_nothing(tmp_path, monkeypatch):
    svc, stash = _svc(tmp_path, monkeypatch)
    out = svc.grade_scene("0", "reject", no_learn=True)
    assert out["scene"]["tier"] == "rejected" and stash.s["0"]["rating100"] == 20     # a real 1★
    assert _row(svc, "0")["no_learn"] is True
    rep = svc.train_tier_model()
    assert rep["counts"]["reject"] == 0                                               # not a training example
    svc.grade_scene("1", "reject")                                                     # a normal reject still counts
    assert svc.train_tier_model()["counts"]["reject"] == 1
    graded = svc._who_records(svc._catalogue_all())
    assert graded is not None
    # Settle never questions it
    rows = svc._catalogue_all()
    c = svc._settle_classify(rows, svc._tier_predictions(rows), {"t": 0.9})
    assert "0" not in {r["scene_id"] for r in c["disagree"]}
    # the reject memory skips it (deleted later, Peaks still won't learn from it)
    svc._remember_rejects(rows)
    fps = svc.reject_memory().fingerprints()
    assert "fp1" in fps and "fp0" not in fps


def test_regrading_or_undoing_clears_it(tmp_path, monkeypatch):
    svc, stash = _svc(tmp_path, monkeypatch)
    prev = svc.grade_scene("0", "reject", no_learn=True)["previous"]
    assert svc._is_no_learn("0")
    svc.grade_scene("0", "reject")                                                     # a plain reject now: it teaches
    assert not svc._is_no_learn("0")
    svc.grade_scene("0", "reject", no_learn=True)
    svc.restore_scene_grade("0", prev["rating100"], prev["o_counter"], tag_ids=prev["tag_ids"],
                            organized=prev["organized"])                               # undo
    assert not svc._is_no_learn("0")
    svc.grade_scene("0", "reject", no_learn=True)
    svc.grade_scene("0", "exceptionnelle")
    assert not svc._is_no_learn("0") and _row(svc, "0")["no_learn"] is False


def test_the_grade_api_takes_no_learn(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, stash = _svc(tmp_path, monkeypatch)
    c = TestClient(app_mod.create_app(svc.cfg))
    r = c.post("/api/catalogue/grade", json={"scene_id": "0", "grade": "reject", "no_learn": True}).json()
    assert r["scene"]["tier"] == "rejected" and r["scene"]["no_learn"] is True
    hist = [e for e in svc.history() if e["action"] == "grade" and e["scene_id"] == "0"]
    assert hist and "don't learn" in str(hist[-1].get("grade"))
