# ruff: noqa: F811 — `svc` is the shared fixture, imported from test_taste_quality
"""The Performers directory: every Stash performer, pictures cached on disk in
a sensible order (your pick, Stash's photo, a solo cover, a frame), records in
plain words, and the page's sections."""

import time

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("PIL")

from test_taste_quality import svc  # noqa: E402,F401 — the shared fixture library


def _cast(svc):
    """Scenes 0–59: evens star Ann (1) alone, graded Légendaire; odds star Bo (2)
    with Cy (3), rejected. Ann and Bo have Stash photos, Cy doesn't; Dee (4) has
    no scenes at all."""
    st = svc.client()
    for sid, v in st.s.items():
        i = int(sid)
        if i % 2 == 0:
            v.update(rating100=100, o_counter=18, performers=["Ann"],
                     performers_detail=[{"id": "1", "name": "Ann"}], studio="Good")
        else:
            v.update(rating100=20, performers=["Bo", "Cy"], studio="Bad",
                     performers_detail=[{"id": "2", "name": "Bo"}, {"id": "3", "name": "Cy"}])
        v["created_at"] = "2024-01-01T00:00:00Z"
    st.performers = [
        {"id": "1", "name": "Ann", "aliases": ["Annie Lou"], "has_image": True, "scene_count": 30, "favorite": True},
        {"id": "2", "name": "Bo", "has_image": True, "scene_count": 30},
        {"id": "3", "name": "Cy", "has_image": False, "scene_count": 30},
        {"id": "4", "name": "Dée", "has_image": False, "scene_count": 0},
    ]
    svc.invalidate_meta()
    svc._cat_cache = None
    return svc


def test_directory_merges_stash_and_records(svc):
    _cast(svc)
    d = svc.performer_directory()
    by = {p["id"]: p for p in d["performers"]}
    assert set(by) == {"1", "2", "3", "4"}                      # everyone, even with no scenes here
    assert by["1"]["aliases"] == ["Annie Lou"] and by["1"]["fav"] and by["1"]["lib"] == 30
    assert by["1"]["record"]["verdict"] == "a favourite" and by["2"]["record"]["verdict"] == "usually a miss"
    assert by["4"]["lib"] == 0 and by["4"]["record"] is None
    assert by["1"]["stale"]                                     # a favourite the board hasn't shown lately
    svc.exposure().record({"0": 1.0})
    assert not {p["id"]: p for p in svc.performer_directory()["performers"]}["1"]["stale"]
    # served from disk after a restart, before Stash is asked again
    svc.__dict__.pop("_perf_dir_cache")
    svc.client().performers = []
    assert len(svc._stash_performers()) == 4


def test_photos_follow_the_source_order_and_are_cached(svc):
    _cast(svc)
    from PIL import Image

    # Ann: her Stash photo (portrait 200×300)
    data = svc.performer_photo("1")
    assert Image.open(__import__("io").BytesIO(data)).size == (200, 300)
    v = svc.performer_photo_version("1")
    assert v and svc.performer_photo("1") == data               # cached: same bytes, no rebuild
    # Cy: no Stash photo, never alone in a scene → a cover of a scene she's in
    assert Image.open(__import__("io").BytesIO(svc.performer_photo("3"))).size == (320, 180)
    # Dee: nothing at all
    assert svc.performer_photo("4") is None and svc.performer_photo_version("4") == 0
    # choosing a picture overrides the automatic one and bumps the version
    opts = svc.performer_photo_options("1")["options"]
    kinds = [o["kind"] for o in opts]
    assert kinds[0] == "stash" and "cover" in kinds and all(o["thumb"] for o in opts)
    cover = next(o for o in opts if o["kind"] == "cover")
    assert cover["solo"]
    time.sleep(1.1)
    r = svc.choose_performer_photo("1", {"kind": "cover", "scene_id": cover["scene_id"]})
    assert r["photo"] > v
    assert Image.open(__import__("io").BytesIO(svc.performer_photo("1"))).size == (320, 180)
    assert next(o for o in svc.performer_photo_options("1")["options"] if o.get("scene_id") == cover["scene_id"])["chosen"]
    svc.choose_performer_photo("1", None)                        # back to automatic
    assert Image.open(__import__("io").BytesIO(svc.performer_photo("1"))).size == (200, 300)
    with pytest.raises(ValueError):
        svc.choose_performer_photo("1", {"kind": "nonsense"})


def test_warm_up_caches_everyone_in_the_library(svc):
    _cast(svc)
    r = svc.warm_performer_photos()
    assert r == {"cached": 3, "of": 3}                          # Ann, Bo, Cy (Dee has no scenes)
    assert svc.performer_directory()["photos_missing"] == 0


def test_profile_and_api(svc):
    _cast(svc)
    p = svc.performer_profile("1")
    assert p["record"]["verdict"] == "a favourite" and p["record"]["top_words"] == "all"
    assert p["studios"] == [{"name": "Good", "n": 30}] and len(p["scenes"]) == 30
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc.warm_performer_photos()                                 # settle what the page would change
    c = TestClient(app_mod.create_app(svc.cfg))
    c.get("/api/performers/directory")
    for _ in range(100):                                        # the leaderboard builds in the background
        if c.get("/api/performers/directory").json()["stats_ready"]:
            break
        time.sleep(0.1)
    r = c.get("/api/performers/directory")
    assert r.status_code == 200 and len(r.json()["performers"]) == 4
    assert c.get("/api/performers/directory", headers={"If-None-Match": r.headers["etag"]}).status_code == 304
    ph = c.get("/api/performer/2/photo?v=1")
    assert ph.status_code == 200 and "max-age=604800" in ph.headers["cache-control"]
    assert c.get("/api/performer/4/photo").status_code == 404
    assert c.post("/api/performer/1/photo", json={"kind": "stash"}).status_code == 200
