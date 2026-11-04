"""Your criteria: ▲/▼ reasons and performer ratings are kept and shown; each
criterion's model learns from them (a scene's own answer never feeds its own
score), generalises to look-alikes, and stays out of grading until it helps."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402

pytest.importorskip("sklearn")

GOOD = {str(p) for p in range(6)}            # performers 0–5 have the face you like


def _faces_world(tmp_path, monkeypatch):
    import peaks.web.service as svc_mod

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    rng = np.random.default_rng(3)
    like = rng.standard_normal(32)
    like /= np.linalg.norm(like)
    face = {}
    for p in range(12):
        v = rng.standard_normal(32) * 0.6 + (1.0 if str(p) in GOOD else -1.0) * like * 1.5
        face[str(p)] = v / np.linalg.norm(v)
    who = {}
    for i, sid in enumerate(sorted(s for s in stash.s if s.isdigit())):
        pid = str(i % 12)
        stash.s[sid]["performers_detail"] = [{"id": pid, "name": f"P{pid}"}]
        stash.s[sid]["performers"] = [f"P{pid}"]
        who[sid] = pid

    def scene_faces(self, sid):
        pid = who.get(str(sid))
        if pid is None:
            return None
        v = face[pid] + rng.standard_normal(32) * 0.05
        v = (v / np.linalg.norm(v))[None].astype(np.float32)
        return np.zeros(1, np.float32), v, {}
    monkeypatch.setattr(svc_mod.Service, "_scene_faces", scene_faces)
    svc.catalogue(limit=5, refresh=True)
    return svc, stash, who


def test_reasons_and_ratings_are_kept_undone_and_shown(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, stash, who = _faces_world(tmp_path, monkeypatch)
    c = TestClient(app_mod.create_app(svc.cfg))
    r = c.post("/api/scene/3/criteria", json={"aspect": "face", "value": 1}).json()
    assert r["reasons"] == {"face": 1}
    c.post("/api/scene/3/criteria", json={"aspect": "quality", "value": -1})
    assert c.get("/api/scene/3/criteria").json()["reasons"] == {"face": 1, "quality": -1}
    c.post("/api/scene/3/criteria", json={"aspect": "face", "value": 0})          # tap again: cleared
    assert c.get("/api/scene/3/criteria").json()["reasons"] == {"quality": -1}
    assert c.post("/api/scene/3/criteria", json={"aspect": "smell", "value": 1}).status_code == 400
    assert c.post("/api/performer/7/criteria", json={"aspect": "face", "value": 2}).json()["ratings"] == {"face": 2}
    story = c.get("/api/scene/3/story").json()
    assert any(e["kind"] == "criteria" and "▼ Quality" in e["why"] for e in story["events"])
    assert svc.criteria_scene_vector("3") is None                                  # nothing learned yet


def test_the_face_criterion_learns_from_her_ratings_and_carries_to_look_alikes(tmp_path, monkeypatch):
    svc, stash, who = _faces_world(tmp_path, monkeypatch)
    for p in ("0", "1", "2", "3"):
        svc.set_performer_criterion(p, "face", 2)
    for p in ("6", "7", "8", "9"):
        svc.set_performer_criterion(p, "face", -1)
    m = svc.criteria_models()["face"]
    assert m["used"] and m["acc"] >= 0.65 and m["labels"] == 24
    unrated_good = [s for s, p in who.items() if p in ("4", "5")]
    unrated_bad = [s for s, p in who.items() if p in ("10", "11")]
    assert np.mean([m["scores"][s] for s in unrated_good]) > 0.6 > 0.4 > np.mean([m["scores"][s] for s in unrated_bad])
    v = svc.criteria_scene_vector(unrated_good[0])
    assert v.shape == (5,) and v[1] > 0.6 and v[0] == 0.5                         # face scored, body unknown
    read = svc.criteria_for_scene(unrated_good[0])["read"]
    assert read["face"]["band"] in ("strong", "ok") and "body" not in read
    rep = svc.train_tier_model()
    assert rep["trained"] and "use_criteria" in rep                               # tested for grading, honestly
    s = svc.criteria_summary()
    assert next(x for x in s["criteria"] if x["aspect"] == "face")["used"] and s["performers_rated"] == 8


def test_noise_reasons_teach_nothing(tmp_path, monkeypatch):
    svc, stash, who = _faces_world(tmp_path, monkeypatch)
    rng = np.random.default_rng(1)
    for sid in [s for s in stash.s if s.isdigit()][:30]:
        svc.set_scene_criterion(sid, "outfit", int(rng.choice([-1, 1])))
    m = svc.criteria_models()["outfit"]
    assert m["labels"] == 30 and not m["used"]
    assert svc.criteria_teach("5") in ("body", "face", "outfit", "action", "quality")


def test_every_settle_question_carries_the_chips_even_if_one_fails(tmp_path, monkeypatch):
    import peaks.web.service as svc_mod
    from test_settle import _library

    svc, stash, (wrong,), _ = _library(tmp_path / "s", monkeypatch)
    rows = svc._catalogue_all()
    q = svc.settle_queue(rows, svc._tier_predictions(rows))
    assert q and all("criteria" in svc.settle_info(r["scene_id"]) for r in q)       # not just the first ones
    bad = q[0]["scene_id"]
    real = svc_mod.Service.criteria_read

    def flaky(self, sid):
        if str(sid) == bad:
            raise RuntimeError("boom")
        return real(self, sid)
    monkeypatch.setattr(svc_mod.Service, "criteria_read", flaky)
    q = svc.settle_queue(rows, svc._tier_predictions(rows))
    assert "criteria" not in svc.settle_info(bad)
    assert all("criteria" in svc.settle_info(r["scene_id"]) for r in q if r["scene_id"] != bad)
