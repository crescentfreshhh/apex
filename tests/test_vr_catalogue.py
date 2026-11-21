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
