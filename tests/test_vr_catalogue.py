"""VR in the catalogue, graded the way you grade VR: stars only.

Légendaire = 5★ + organized, Exceptionnelle = 4★ + organized, unrated =
unreviewed, 1★ = reject (as flat). No O-count, no tier tags, no Merveilleuse.
A scene is VR when its file is in a VR pull folder (/data/#VR by default) or
it carries the "Virtual Reality" tag."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

pytest.importorskip("fastapi")

from test_ingest import stash, svc  # noqa: E402,F401 — the fake Stash + service fixtures

from peaks.tiers import tier_of  # noqa: E402


def test_vr_tiers_read_the_stars_alone():
    assert tier_of(100, 0, vr=True) == "legendaire"
    assert tier_of(80, 0, vr=True) == "exceptionnelle"
    assert tier_of(100, 17, vr=True) == "legendaire"          # an O-count means nothing for VR
    assert tier_of(None, 0, vr=True) == "unreviewed"
    assert tier_of(20, 0, vr=True) == "rejected"
    assert tier_of(60, 0, vr=True) == "unreviewed"            # 3★: not a VR grade
    assert tier_of(90, 0, vr=True) == "anomaly"               # 4½★: to revalidate
    assert tier_of(80, 0) == "unreviewed" and tier_of(100, 0) == "anomaly"   # flat is unchanged


def _vr(stash, sid="2", **kw):
    stash.s[sid].update(path=f"/data/#VR/{sid}.mp4", **kw)
    return sid


def test_a_vr_row_says_so_and_reads_its_grade(svc, stash):
    _vr(stash, "2", rating100=80, organized=True)
    stash.tags = {"9": "Virtual Reality"}
    stash.s["1"].update(rating100=100, o_counter=0, tag_ids=["9"])     # tagged VR, outside the folder
    rows = {r["scene_id"]: r for r in svc._catalogue_all(refresh=True)}
    assert rows["2"]["vr"] and rows["2"]["tier"] == "exceptionnelle"
    assert rows["1"]["vr"] and rows["1"]["tier"] == "legendaire"
    assert not rows["2"]["tag_state"]["needs_sync"] and rows["2"]["tag_state"]["conflict"] is None


def test_grading_vr_writes_only_the_stars_and_organized(svc, stash):
    stash.tags = {"7": "Blonde"}
    sid = _vr(stash, "2", tag_ids=["7"])
    svc._catalogue_all(refresh=True)
    out = svc.grade_scene(sid, "exceptionnelle")
    writes = [c for c in stash.calls if c[0] in ("update", "add_o", "o_dates", "delete_o")]
    assert writes == [("update", sid, {"rating100": 80, "organized": True})]   # no O-count, no tags
    assert stash.s[sid]["tag_ids"] == ["7"] and out["scene"]["tier"] == "exceptionnelle"
    assert out["previous"] == {"rating100": None, "o_counter": 0, "tag_ids": ["7"], "organized": False}
    stash.calls.clear()
    out = svc.grade_scene(sid, "legendaire")
    assert [c for c in stash.calls if c[0] == "update"] == [("update", sid, {"rating100": 100, "organized": True})]
    assert out["scene"]["tier"] == "legendaire"


def test_vr_has_no_merveilleuse_and_rejects_like_flat(svc, stash):
    sid = _vr(stash, "2")
    svc._catalogue_all(refresh=True)
    with pytest.raises(ValueError, match="VR scenes are graded"):
        svc.grade_scene(sid, "merveilleuse")
    out = svc.grade_scene(sid, "reject")
    assert stash.s[sid]["rating100"] == 20 and out["scene"]["tier"] == "rejected"


def test_vr_folders_are_in_scope_beside_the_library_path(svc, stash):
    svc.cfg.library.path = "/data/flat"
    stash.s["1"]["path"] = "/data/flat/a.mp4"
    _vr(stash, "2")
    stash.s.setdefault("3", {"rating100": None, "o_counter": 0, "tag_ids": [], "organized": False,
                             "fingerprint": "fp3"})
    stash.s["3"]["path"] = "/data/elsewhere/b.mp4"
    assert svc._scope_prefixes() == ["/data/flat", "/data/#VR"]
    ids = {r["scene_id"] for r in svc._catalogue_fetch()}            # the light listing, scoped here
    assert ids == {"1", "2"}


def test_the_flat_model_never_learns_from_or_scores_vr(svc, stash):
    _vr(stash, "2", rating100=100, organized=True)
    rows = svc._catalogue_all(refresh=True)
    seen = []
    real = svc._scene_features
    svc._scene_features = lambda rs: (seen.append({r["scene_id"] for r in rs}), real(rs))[1]
    try:
        svc.train_tier_model()
    except Exception:  # noqa: BLE001 — too few grades to fit is fine: what it read is the point
        pass
    assert seen and all("2" not in s for s in seen)
    st = {"model": object()}
    seen.clear()
    svc._tier_predictions_compute(rows, {**st, "model": type("M", (), {"fallback": None, "oof": {}})()}, {}, "k")
    assert seen and all("2" not in s for s in seen)


def test_the_sampler_crops_vr_to_the_left_eye():
    from peaks.sampling import FrameSampler, vr_crop_filter

    s = FrameSampler(interval_seconds=2.0, mode="interval")
    s.vr_test = lambda p: "/#VR/" in p
    assert vr_crop_filter() == "crop=iw*0.3:ih*0.6:iw*0.1:ih*0.2"
    assert vr_crop_filter() in s._raw_vf(256, 224, vr=True)
    assert "crop=iw" not in s._raw_vf(256, 224)
    assert s._is_vr("/data/#VR/a.mp4") and not s._is_vr("/data/flat/a.mp4")


def test_vr_keepers_never_set_the_flat_quality_floor():
    from peaks.tier_model import quality_flag, quality_floor

    vr = [{"tier": "legendaire", "vr": True, "quality": {"res": "6K", "mbps": 60}} for _ in range(6)]
    flat = [{"tier": "legendaire", "quality": {"res": "1080p", "mbps": 8}} for _ in range(6)]
    floor = quality_floor(vr + flat)
    assert floor["lowest_res"] == "1080p" and "6K" not in floor["by_res"]
    assert quality_flag({"res": "4K", "mbps": 1, "vr": True}, floor) is None


def _vr_library(stash, n_keep=6):
    """VR keepers (real ~6K per eye) plus an upscale and a good unreviewed file."""
    import peaks.picture as pq

    meas = {}
    for i in range(n_keep):
        sid = str(100 + i)
        stash.s[sid] = {"rating100": 100 if i % 2 else 80, "o_counter": 0, "tag_ids": [], "organized": True,
                        "fingerprint": f"fp{sid}", "path": f"/data/#VR/k{sid}.mp4", "width": 8192, "height": 4096,
                        "bit_rate": 60e6, "frame_rate": 60, "studio": "Alpha", "date": "2023-01-01"}
        meas[f"fp{sid}"] = {"v": pq.VERSION, "eff_h": 2900 + 50 * i, "eff_factor": 1.1}
    for sid, eye in (("200", 1800), ("201", 3200)):
        stash.s[sid] = {"rating100": None, "o_counter": 0, "tag_ids": [], "organized": False,
                        "fingerprint": f"fp{sid}", "path": f"/data/#VR/u{sid}.mp4", "width": 8192, "height": 4096,
                        "bit_rate": 60e6, "frame_rate": 60, "studio": "Alpha", "date": "2023-06-01"}
        meas[f"fp{sid}"] = {"v": pq.VERSION, "eff_h": eye, "eff_factor": 4096 / eye}
    return meas


def test_vr_quality_reaches_the_listing_and_sorts_it(svc, stash):
    meas = _vr_library(stash)
    svc._pic_measures = lambda: meas
    svc._catalogue_all(refresh=True)
    d = svc.catalogue(tier="unreviewed", vr="vr", sort="vrquality")
    order = [i["scene_id"] for i in d["items"]]
    assert order == ["201", "200"]                                  # meets your bar first
    by = {i["scene_id"]: i["vrq"] for i in d["items"]}
    assert by["201"]["verdict"] == "meets" and by["200"]["verdict"] == "below"
    assert "under a 8K label" in by["200"]["reasons"][0]


def test_vr_files_are_measured_without_the_backfill_switch(svc, stash):
    _vr_library(stash)
    svc._pic_measures = lambda: {}
    svc._catalogue_all(refresh=True)
    owed, rest = svc._pic_todo_raw(svc._catalogue_all())
    ids = [r["scene_id"] for r in owed]
    assert rest == [] and set(ids) >= {"100", "101", "200", "201"} and "1" not in ids
    assert ids.index("101") < ids.index("200")                      # keepers first: they set the bar


def test_vr_files_are_sampled_every_10s_but_count_as_done_at_the_library_rate():
    from types import SimpleNamespace

    from peaks.pipeline import _vr_meta
    from peaks.sampling import FrameSampler

    s = FrameSampler(interval_seconds=2.0, mode="sparse")
    s.vr_test = lambda p: "/#VR/" in p
    sig = s.interval_signature
    s.vr_interval = 10.0
    assert s._interval_for("/data/#VR/a.mp4") == 10.0 and s._interval_for("/data/flat/a.mp4") == 2.0
    assert s.interval_signature == sig                                 # recorded as the library's sampling
    i = FrameSampler(interval_seconds=2.0, mode="interval")
    i.vr_test, i.vr_interval = s.vr_test, 10.0
    assert i._raw_vf(256, 224, True, i._interval_for("/data/#VR/a.mp4")).startswith("fps=1/10,crop=")
    assert i._raw_vf(256, 224, False, i._interval_for("/data/flat/a.mp4")).startswith("fps=1/2,scale=")
    assert i._linear_fallback_sampler().vr_interval == 10.0
    assert _vr_meta(s, SimpleNamespace(path="/data/#VR/a.mp4")) == {"vr_interval": 10.0}
    assert _vr_meta(s, SimpleNamespace(path="/data/flat/a.mp4")) == {}
    s.vr_interval = 1.0                                                # never denser than the library
    assert s._interval_for("/data/#VR/a.mp4") == 2.0


def _graded_vr(stash, n=10):
    """n VR scenes per grade: rejects soft (upscaled), keepers sharp; plus one unreviewed sharp and one soft."""
    import peaks.picture as pq

    meas, sid = {}, 300
    for tier, rating, eyes in (("reject", 20, (1500, 2100)), ("exceptionnelle", 80, (2900, 3300)),
                               ("legendaire", 100, (3400, 3900))):
        for i in range(n):
            sid += 1
            s = str(sid)
            eye = eyes[0] + (eyes[1] - eyes[0]) * i / max(1, n - 1)
            stash.s[s] = {"rating100": rating, "o_counter": 0, "tag_ids": [], "organized": rating > 20,
                          "fingerprint": f"fp{s}", "path": f"/data/#VR/{s}.mp4", "width": 8192, "height": 4096,
                          "bit_rate": 60e6, "frame_rate": 60, "studio": "Alpha", "date": "2023-01-01"}
            meas[f"fp{s}"] = {"v": pq.VERSION, "eff_h": eye, "eff_factor": 4096 / eye, "sharp": eye / 1000}
    for s, eye in (("398", 3700), ("399", 1600)):
        stash.s[s] = {"rating100": None, "o_counter": 0, "tag_ids": [], "organized": False, "fingerprint": f"fp{s}",
                      "path": f"/data/#VR/{s}.mp4", "width": 8192, "height": 4096, "bit_rate": 60e6,
                      "frame_rate": 60, "studio": "Alpha", "date": "2023-01-01"}
        meas[f"fp{s}"] = {"v": pq.VERSION, "eff_h": eye, "eff_factor": 4096 / eye, "sharp": eye / 1000}
    return meas


def test_the_vr_model_trains_on_vr_grades_and_suggests_for_unreviewed_vr(svc, stash):
    pytest.importorskip("sklearn")
    meas = _graded_vr(stash)
    svc._pic_measures = lambda: meas
    svc._catalogue_all(refresh=True)
    rep = svc.train_vr_model()
    assert rep["trained"] and rep["counts"] == {"reject": 10, "exceptionnelle": 10, "legendaire": 10}
    d = svc.catalogue(tier="unreviewed", vr="vr")
    by = {i["scene_id"]: i["pred"] for i in d["items"]}
    assert by["399"]["tier"] == "reject" and by["398"]["tier"] in ("exceptionnelle", "legendaire")
    assert by["399"]["from"] == "vr" and d["vr_model"]["trained"]


def test_vr_rejects_are_remembered_apart_and_count_after_deletion(svc, stash):
    pytest.importorskip("sklearn")
    meas = _graded_vr(stash, n=8)
    svc._pic_measures = lambda: meas
    rows = svc._catalogue_all(refresh=True)
    assert svc._remember_rejects(rows) == 0                       # never into the flat model's memory
    assert svc._remember_vr_rejects(rows) == 8
    mem = svc.store().map_all("vr_rejects")
    assert mem["fp301"]["y"] == "reject" and mem["fp301"]["pic"]["eff_h"] == 1500
    for s in [k for k, v in stash.s.items() if v.get("rating100") == 20]:   # the files are deleted
        del stash.s[s]
    svc._catalogue_all(refresh=True)
    ex = svc._vr_examples(svc._catalogue_all())
    assert sum(e["y"] == "reject" for e in ex) == 8                # still examples, from memory


def test_without_enough_vr_grades_the_model_says_what_it_needs(svc, stash):
    meas = _graded_vr(stash, n=3)
    svc._pic_measures = lambda: meas
    svc._catalogue_all(refresh=True)
    rep = svc.train_vr_model()
    assert not rep["trained"] and rep["needs"] == {"reject": 5, "exceptionnelle": 5, "legendaire": 5}
    assert svc.vr_model_status()["trained"] is False


def test_a_rating_set_in_heresphere_arrives_with_the_next_routine_check(svc, stash):
    import time

    sid = _vr(stash, "2")
    svc._catalogue_fetch()                                          # the first listing
    svc._catalogue_fetch()                                          # a check: now it has a "last time"
    stash.s[sid].update(rating100=100)                              # you rate it 5★ in HereSphere
    stash.updated = {sid: time.time()}
    before = getattr(svc, "_grades_since_train", 0)
    rows = {r["scene_id"]: r for r in svc._catalogue_fetch()}
    assert rows[sid]["tier"] == "legendaire"
    assert svc._grades_since_train == before + 1                    # counts toward the next retrain


def test_vr_keepers_rated_outside_peaks_are_marked_organized_and_nothing_else(svc, stash):
    _vr(stash, "2", rating100=100, organized=False)                 # VR 5★ from HereSphere
    stash.s["1"].update(rating100=100, o_counter=18, organized=False)   # a flat Légendaire: not ours to touch
    stash.s.setdefault("3", {"rating100": None, "o_counter": 0, "tag_ids": [], "organized": False,
                             "fingerprint": "fp3"})
    stash.s["3"]["path"] = "/data/#VR/u.mp4"                        # VR unreviewed: left alone
    svc._catalogue_all(refresh=True)
    stash.calls.clear()
    assert svc.organize_vr_keepers() == 1
    assert [c for c in stash.calls if c[0] == "update"] == [("update", "2", {"organized": True})]
    assert stash.s["2"]["organized"] and not stash.s["1"]["organized"] and not stash.s["3"]["organized"]
    stash.calls.clear()
    assert svc.organize_vr_keepers() == 0 and stash.calls == []    # done once


def test_a_stash_without_the_update_filter_still_lists_new_scenes(svc, stash):
    stash.scene_ids_updated_since = None                            # an older Stash: calling it fails
    svc._catalogue_fetch()
    svc._catalogue_fetch()
    stash.s["9"] = {"rating100": None, "o_counter": 0, "tag_ids": [], "organized": False, "fingerprint": "fp9"}
    assert "9" in {r["scene_id"] for r in svc._catalogue_fetch()}


def test_the_client_asks_for_scenes_updated_after_a_time():
    from peaks.stash_client import StashClient

    c = StashClient(url="http://stash.invalid/graphql")
    seen = {}
    c.execute = lambda q, v=None: (seen.update(q=q, v=v), {"findScenes": {"scenes": [{"id": 4}]}})[1]
    assert c.scene_ids_updated_since(1_800_000_000) == {"4"}
    assert "scene_filter: $f" in seen["q"]
    assert seen["v"] == {"f": {"updated_at": {"value": "2027-01-15T08:00:00Z", "modifier": "GREATER_THAN"}}}


def test_a_failed_check_leaves_its_window_for_the_next_one(svc, stash):
    import time

    sid = _vr(stash, "2")
    svc._catalogue_fetch()
    svc._catalogue_fetch()
    t0 = svc._cat_checked_at
    stash.s[sid].update(rating100=80)                               # rated while Stash is struggling
    stash.updated = {sid: time.time()}
    real = stash.scene_details
    stash.scene_details = lambda ids: (_ for _ in ()).throw(TimeoutError("Stash busy"))
    with pytest.raises(TimeoutError):
        svc._catalogue_fetch()
    assert svc._cat_checked_at == t0                                # not moved on
    stash.scene_details = real
    real_upd = stash.scene_ids_updated_since
    stash.scene_ids_updated_since = lambda when: (_ for _ in ()).throw(TimeoutError("blip"))
    svc._catalogue_fetch()
    assert svc._cat_checked_at == t0                                # couldn't ask: still not moved on
    stash.scene_ids_updated_since = real_upd
    rows = {r["scene_id"]: r for r in svc._catalogue_fetch()}
    assert rows[sid]["tier"] == "exceptionnelle"                    # caught up
