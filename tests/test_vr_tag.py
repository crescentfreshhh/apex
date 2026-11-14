"""Videos in a VR pull folder (Seedbox → categories, e.g. /data/#VR) carry the
"Virtual Reality" tag in Stash once ingested — added on top of their other
tags, the ones already there included, and never twice."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

pytest.importorskip("fastapi")

from test_ingest import stash, svc  # noqa: E402,F401 — the ingest fixtures


def _vr_calls(stash):
    return [c for c in stash.calls if c[0] in ("add_tags", "tag_create")]


def test_an_ingest_tags_new_and_existing_scenes_in_the_vr_folder(svc, stash):
    stash.tags = {"7": "Blonde"}
    stash.s["2"].update(path="/data/#VR/old.mp4", tag_ids=["7"])          # already in the library
    stash.s["1"]["path"] = "/data/Rando/norating/flat.mp4"
    stash.arriving["10"]["path"] = "/data/#VR/new.mp4"                   # pulled into #VR
    out = svc.run_ingest()
    vr = next(t for t, n in stash.tags.items() if n == "Virtual Reality")
    assert _vr_calls(stash) == [("tag_create", "Virtual Reality"), ("add_tags", ["10", "2"], [vr])]
    assert stash.s["2"]["tag_ids"] == ["7", vr]                            # its other tags stay
    assert vr not in stash.s["1"]["tag_ids"] and vr not in stash.s["11"]["tag_ids"]
    assert out["stages"]["vr tag"] == "2 tagged 'Virtual Reality'"
    row = next(r for r in svc._catalogue_all() if r["scene_id"] == "10")
    assert vr in row["tag_ids"]                                            # the listing shows it at once
    stash.calls.clear()
    stash.arriving = {}                                                    # (the fake re-adds them per scan)
    out = svc.run_ingest()                                                 # nothing missing: no write
    assert _vr_calls(stash) == [] and "vr tag" not in out["stages"]


def test_an_existing_tag_is_reused_and_a_custom_vr_category_counts(svc, stash):
    stash.tags = {"5": "virtual reality"}
    svc.save_seedjob_settings({"categories": [{"cat": "1", "dest": "/data/Rando/norating"},
                                              {"cat": "VR-6dof", "dest": "/mnt/Headset"}]})
    assert svc.vr_folders() == ["/mnt/Headset"]
    stash.s["2"]["path"] = "/mnt/Headset/a.mp4"
    stash.s["1"]["path"] = "/data/#VR/not-a-pull-folder-any-more.mp4"
    svc.run_ingest()
    assert _vr_calls(stash) == [("add_tags", ["2"], ["5"])]               # no second tag created


def test_a_folder_whose_name_starts_the_same_is_not_inside(svc, stash):
    stash.s["2"]["path"] = "/data/#VR-old/a.mp4"
    svc.run_ingest()
    assert _vr_calls(stash) == []
