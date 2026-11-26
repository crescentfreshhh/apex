"""Peaks's picks as Stash markers (opt-in): each scene's top 3 moments under the
"Peaks pick" tag — kept in step with the picks, never mixed with your saved
moments, and removed completely when switched off."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

pytest.importorskip("fastapi")

from test_ingest import stash, svc  # noqa: E402,F401 — the fake Stash + service fixtures


def _strips(picks):
    """Stand-in for the index's best moments: scene_id -> [(t, score)]."""
    def f(sids, n=4, min_gap=20.0):
        return {s: [{"t": t, "score": sc} for t, sc in picks.get(s, [])][:n] for s in sids}
    return f


def _on(svc, stash, picks):
    stash.markers = [{"scene_id": "1", "seconds": 30.0, "marker_id": "m1", "title": "saved"}]   # your saved moment
    svc._scene_moment_strips = _strips(picks)
    svc._catalogue_all(refresh=True)
    svc.save_pick_markers(True)


def test_off_by_default_and_nothing_written(svc, stash):
    assert svc.pick_markers_settings()["on"] is False
    svc._scene_moment_strips = _strips({"1": [(100.0, 0.9)]})
    svc._catalogue_all(refresh=True)
    out = svc.run_pick_markers()
    assert not any(c[0] == "marker_create" for c in stash.calls) and out == {"removed": 0}


def test_top_three_picks_become_markers_under_their_own_tag(svc, stash):
    _on(svc, stash, {"1": [(100.0, 0.5), (300.0, 0.9), (500.0, 0.7), (700.0, 0.2)], "2": [(40.0, 0.6)]})
    out = svc.run_pick_markers()
    made = [c for c in stash.calls if c[0] == "marker_create"]
    assert [(c[1], c[2], c[3]) for c in made if c[1] == "1"] == [
        ("1", 300.0, "Peaks pick 1"), ("1", 500.0, "Peaks pick 2"), ("1", 100.0, "Peaks pick 3")]
    assert out["written"] == 2
    picks = [m for m in stash.markers if m.get("tag") == "Peaks pick"]
    assert len(picks) == 4 and all(m["end_seconds"] == m["seconds"] + 20 for m in picks)
    assert [m["marker_id"] for m in stash.iter_markers_by_tag("peaks")] == ["m1"]   # saved moments unmixed


def test_only_moved_picks_are_rewritten_and_rejects_lose_theirs(svc, stash):
    picks = {"1": [(100.0, 0.9)], "2": [(40.0, 0.6)]}
    _on(svc, stash, picks)
    svc.run_pick_markers()
    stash.calls.clear()
    picks["1"] = [(105.0, 0.9)]                                       # moved 5 s: left alone
    picks["2"] = [(90.0, 0.6)]                                        # moved 50 s: rewritten
    svc.run_pick_markers()
    assert [c[1] for c in stash.calls if c[0] == "marker_create"] == ["2"]
    assert sum(c[0] == "marker_destroy" for c in stash.calls) == 1
    stash.calls.clear()
    stash.s["2"]["rating100"] = 20                                    # rejected
    svc._catalogue_all(refresh=True)
    out = svc.run_pick_markers()
    assert out["removed"] == 1 and not [m for m in stash.markers if m["scene_id"] == "2" and m.get("tag")]


def test_no_taste_model_means_no_markers(svc, stash):
    _on(svc, stash, {"1": [(100.0, None), (300.0, None)]})            # evenly spread frames, no scores
    svc.run_pick_markers()
    assert not any(c[0] == "marker_create" for c in stash.calls)


def test_switching_off_removes_every_pick_marker_and_nothing_else(svc, stash):
    _on(svc, stash, {"1": [(100.0, 0.9), (300.0, 0.5)], "2": [(40.0, 0.6)]})
    svc.run_pick_markers()
    stash.markers.append({"scene_id": "2", "seconds": 1.0, "marker_id": "lost", "title": "Peaks pick 1",
                          "tag": "Peaks pick"})                       # one Peaks forgot about
    svc.save_pick_markers(False)
    assert svc.pick_markers_due()
    out = svc.run_pick_markers()
    assert out["removed"] == 4
    assert [m["marker_id"] for m in stash.markers] == ["m1"] and svc._pm_state() == {}


def test_the_api_switch_and_remove(svc, stash, monkeypatch):
    from test_ingest import _api

    client = _api(svc, monkeypatch)
    assert client.get("/api/pick-markers").json()["on"] is False
    assert client.post("/api/pick-markers", json={"on": True}).json()["on"] is True
    r = client.post("/api/pick-markers/remove")
    assert r.status_code == 200 and client.get("/api/pick-markers").json()["on"] is False
