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
