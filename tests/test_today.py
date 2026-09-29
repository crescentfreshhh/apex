# ruff: noqa: F811 — `svc` is the shared fixture, imported from test_taste_quality
"""For You as a daily curation page: a mixed set of decisions fixed for the
day, one-job streams, skip/done, the task board and the week's tally."""

import pytest

pytest.importorskip("sklearn")

from test_taste_quality import _curated, _settle, svc  # noqa: E402,F401


def _ready(svc):
    _curated(svc)
    _settle(svc)
    return svc


def test_todays_set_is_mixed_capped_and_fixed_for_the_day(svc):
    _ready(svc)
    t = svc.curation_today(n=10)
    p = t["progress"]
    assert 0 < p["total"] <= p["goal"] == 20 and p["done"] == 0
    jobs = [d["job"] for d in t["next"]]
    assert jobs[0] == "saved"                                   # a quick confirmation first
    assert "new" in jobs and len(set(jobs)) >= 2                # …interleaved with real decisions
    ids = [d["scene_id"] for d in t["next"]]
    assert len(ids) == len(set(ids))
    first = t["next"][0]
    assert first["pick"] == "legendaire" and "saved" in first["why"] and first["stream"]
    again = svc.curation_today(n=10)
    assert [d["scene_id"] for d in again["next"]] == ids        # the same all day


def test_grading_anywhere_counts_and_skip_moves_to_the_back(svc):
    _ready(svc)
    t = svc.curation_today(n=5)
    a, b = t["next"][0]["scene_id"], t["next"][1]["scene_id"]
    svc.grade_scene(a, "legendaire")                            # e.g. from the Catalogue
    t2 = svc.curation_today(n=5)
    assert a not in [d["scene_id"] for d in t2["next"]] and t2["progress"]["done"] == 1
    svc.today_mark(b, skip=True)
    order = [d["scene_id"] for d in svc.curation_today(n=10)["next"]]
    assert order[0] != b and (b not in order or order[-1] == b)
    before = svc.curation_today()["progress"]["total"]
    assert svc.curation_today(more=True)["progress"]["total"] > before    # "keep going"


def test_one_job_streams_and_the_grace_period(svc):
    _ready(svc)
    saved = svc.curation_today(job="saved", n=10)
    assert {d["job"] for d in saved["next"]} == {"saved"}
    assert {d["scene_id"] for d in saved["next"]} == {"0", "15"}
    new = svc.curation_today(job="new", n=10)["next"]
    assert all(d["tier"] == "unreviewed" for d in new)
    fresh = [d for d in svc.curation_today(job="new", n=10)["next"] if d["scene_id"] == "30"]
    assert all(d["pick"] != "reject" for d in fresh)            # new scenes: never pushed to Reject
    with pytest.raises(ValueError):
        svc.curation_today(job="nonsense")


def test_board_counts_match_the_catalogue_and_week_tally(svc):
    _ready(svc)
    t = svc.curation_today()
    views = svc.library_summary()["views"]
    assert t["board"]["jobs"]["saved"] == views["saved"] and t["board"]["likely"] == views["likely"]
    assert t["board"]["rejected"]["count"] == svc.library_summary()["counts"]["rejected"]
    w0 = t["week"]["decisions"]
    svc.grade_scene("0", "legendaire")
    svc.grade_scene("2", "reject")
    w = svc.curation_today()["week"]
    assert w["decisions"] == w0 + 2 and w["promoted"] >= 1 and w["let_go"] >= 1


def test_today_api(svc):
    _ready(svc)
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    c = TestClient(app_mod.create_app(svc.cfg))
    r = c.get("/api/today?n=2")
    assert r.status_code == 200 and len(r.json()["next"]) == 2
    sid = r.json()["next"][0]["scene_id"]
    assert c.post(f"/api/today/done?scene_id={sid}").json()["done"] == 1
    assert c.post(f"/api/today/done?scene_id={sid}&undo=true").json()["done"] == 0
    assert c.get("/api/today?job=bogus").status_code == 400
    assert c.post("/api/library/curation?today_goal=5").json()["today_goal"] == 5
