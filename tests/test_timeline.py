"""One scene's story: grades (and what Peaks thought), rulings, pick answers,
upgrades and 'not them', newest first, with Peaks' view today."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_settle import _library  # noqa: E402

pytest.importorskip("sklearn")


def test_a_scene_timeline_tells_the_story(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, stash, (wrong,), _ = _library(tmp_path, monkeypatch, drain=False)
    svc.catalogue(limit=5, refresh=True)
    svc._tier_predictions(svc._catalogue_all())              # Peaks has an opinion before you grade
    svc.grade_scene(wrong, "exceptionnelle")
    svc.settle_rule(wrong)
    svc.record_pick_feedback(wrong, 42.0, 1)
    svc.orphan_not_them(wrong, "77")
    c = TestClient(app_mod.create_app(svc.cfg))
    d = c.get(f"/api/scene/{wrong}/story").json()
    assert "events" in d, d
    kinds = [e["kind"] for e in d["events"]]
    assert {"grade", "ruling", "pick", "performers"} <= set(kinds)
    g = next(e for e in d["events"] if e["kind"] == "grade")
    assert g["who"] == "you" and "→" in g["text"] and "Peaks thought" in g["why"]
    ats = [e["at"] for e in d["events"] if e["at"]]
    assert ats == sorted(ats, reverse=True)                  # newest first
    assert d["now"]["peaks"]["tier"] and d["now"]["tier"]
    assert c.get("/api/scene/nope/story").status_code == 404
