"""Catalogue: the user's grading scheme (5★ + O-count tiers), exact O-count
writes, grading/undo through the API, and quality metadata. Stash is a fake
that records every mutation, so we assert the exact calls Peaks makes."""

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from peaks.config import Config  # noqa: E402
from peaks.tiers import GRADES, quality_of, res_class, tier_names, tier_of  # noqa: E402


# --- pure tier logic ---------------------------------------------------------

@pytest.mark.parametrize("rating,o,want", [
    (None, 0, "unreviewed"), (0, 18, "unreviewed"),
    (20, 18, "rejected"), (20, 0, "rejected"),
    (60, 17, "unreviewed"), (80, 18, "unreviewed"),      # 2-4★ = not reviewed yet
    (100, 0, "upscale"),
    (100, 1, "anomaly"), (100, 15, "anomaly"), (100, 19, "anomaly"),
    (100, 16, "merveilleuse"), (100, 17, "exceptionnelle"), (100, 18, "legendaire"),
])
def test_tier_of_boundaries(rating, o, want):
    assert tier_of(rating, o) == want


def test_grades_round_trip_to_their_tier():
    for grade, (rating, o) in GRADES.items():
        want = "rejected" if grade == "reject" else grade
        assert tier_of(rating, o if o is not None else 5) == want


def test_quality_and_resolution():
    assert res_class(3840, 2160) == "4K" and res_class(1080, 1920) == "1080p"
    assert res_class(None, 720) is None
    q = quality_of({"width": 3840, "height": 2160, "bit_rate": 42_000_000,
                    "frame_rate": 60, "video_codec": "HEVC"})
    assert q == {"res": "4K", "mbps": 42.0, "fps": 60.0, "codec": "hevc",
                 "bpp": round(42e6 / (3840 * 2160 * 60), 4)}
    assert quality_of({})["mbps"] is None and quality_of({})["bpp"] is None


def test_tier_names_overrides():
    n = tier_names({"legendaire": " Legendaire ", "bogus": "x", "merveilleuse": ""})
    assert n["legendaire"] == "Legendaire"            # trimmed override
    assert n["merveilleuse"] == "Merveilleuse"        # blank → default
    assert "bogus" not in n


# --- fake Stash --------------------------------------------------------------

class FakeStash:
    def __init__(self, scenes: dict):
        self.s = scenes            # sid -> {rating100, o_counter, ...}
        self.calls: list = []

    def iter_scenes(self, path_prefix=None):
        for sid in self.s:
            yield SimpleNamespace(id=sid)

    def scene_details(self, ids):
        return {str(i): {"path": f"/data/{i}.mp4", "title": f"Scene {i}",
                         "performers": ["Jane"], **self.s[str(i)]}
                for i in ids if str(i) in self.s}

    def update_scene(self, scene_id, clear=(), **fields):
        self.calls.append(("update", scene_id, fields.get("rating100"), tuple(clear)))
        if "rating100" in clear:
            self.s[scene_id]["rating100"] = None
        elif fields.get("rating100") is not None:
            self.s[scene_id]["rating100"] = fields["rating100"]
        return {}

    def scene_add_o(self, sid):
        self.calls.append(("add_o", sid))
        self.s[sid]["o_counter"] += 1
        return self.s[sid]["o_counter"]

    def scene_delete_o(self, sid):
        self.calls.append(("del_o", sid))
        self.s[sid]["o_counter"] -= 1
        return self.s[sid]["o_counter"]

    def scene_reset_o(self, sid):
        self.calls.append(("reset_o", sid))
        self.s[sid]["o_counter"] = 0
        return 0

    def stream_url(self, sid, start=None):
        return f"http://stash/{sid}?t={start}"


@pytest.fixture
def stash():
    return FakeStash({
        "1": {"rating100": None, "o_counter": 0, "width": 3840, "height": 2160,
              "bit_rate": 40_000_000, "frame_rate": 60, "video_codec": "hevc", "date": "2024-01-01"},
        "2": {"rating100": 100, "o_counter": 16, "width": 1920, "height": 1080,
              "bit_rate": 8_000_000, "frame_rate": 30, "video_codec": "h264", "date": "2023-01-01"},
        "3": {"rating100": 100, "o_counter": 7, "width": 1920, "height": 1080,
              "bit_rate": 5_000_000, "frame_rate": 30, "video_codec": "h264", "date": "2022-01-01"},
        "4": {"rating100": 20, "o_counter": 0, "date": "2021-01-01"},
    })


@pytest.fixture
def svc(tmp_path, stash, monkeypatch):
    import peaks.web.service as svc_mod

    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    return svc_mod.Service(cfg)


# --- exact O-count writes ----------------------------------------------------

def test_set_o_count_moves_by_the_difference(svc, stash):
    assert svc.set_o_count("2", 17) == 17
    assert stash.calls == [("add_o", "2")]                      # 16 → 17: one add
    stash.calls.clear()
    assert svc.set_o_count("2", 16) == 16
    assert stash.calls == [("del_o", "2")]                      # 17 → 16: one delete
    stash.calls.clear()
    assert svc.set_o_count("3", 0) == 0
    assert stash.calls == [("reset_o", "3")]                    # → 0: reset
    stash.calls.clear()
    assert svc.set_o_count("3", 0) == 0 and stash.calls == []   # already there: no-op
    with pytest.raises(ValueError):
        svc.set_o_count("3", 101)


def test_grade_then_undo(svc, stash):
    out = svc.grade_scene("1", "legendaire")
    assert out["previous"] == {"rating100": None, "o_counter": 0}
    assert out["scene"]["tier"] == "legendaire"
    assert stash.s["1"]["rating100"] == 100 and stash.s["1"]["o_counter"] == 18
    # undo restores an UNRATED scene by clearing the rating, not setting a value
    back = svc.restore_scene_grade("1", out["previous"]["rating100"], out["previous"]["o_counter"])
    assert back["scene"]["tier"] == "unreviewed"
    assert stash.s["1"]["rating100"] is None and stash.s["1"]["o_counter"] == 0
    assert ("update", "1", None, ("rating100",)) in stash.calls


def test_reject_hides_and_leaves_o_alone(svc, stash):
    svc.grade_scene("3", "reject")
    assert stash.s["3"] == {**stash.s["3"], "rating100": 20, "o_counter": 7}
    assert "3" in svc.hidden_scene_ids()
    svc.grade_scene("3", "merveilleuse")                        # re-grade unhides
    assert "3" not in svc.hidden_scene_ids() and stash.s["3"]["o_counter"] == 16


def test_unknown_grade_rejected(svc):
    with pytest.raises(ValueError):
        svc.grade_scene("1", "legendary")


# --- listing ------------------------------------------------------------------

def test_catalogue_counts_filters_and_hidden_sync(svc):
    d = svc.catalogue()
    assert d["counts"]["unreviewed"] == 1 and d["counts"]["merveilleuse"] == 1
    assert d["counts"]["anomaly"] == 1 and d["counts"]["rejected"] == 1
    assert [r["scene_id"] for r in d["items"]] == ["1", "2", "3", "4"]   # newest first
    # a scene rejected directly in Stash is hidden from feeds after a listing
    assert "4" in svc.hidden_scene_ids()
    only = svc.catalogue(tier="anomaly")
    assert [r["scene_id"] for r in only["items"]] == ["3"]
    assert only["counts"]["unreviewed"] == 1          # chips still count every tier
    assert [r["scene_id"] for r in svc.catalogue(res="4K")["items"]] == ["1"]
    assert [r["scene_id"] for r in svc.catalogue(min_mbps=6)["items"]] == ["1", "2"]
    assert [r["scene_id"] for r in svc.catalogue(sort="bitrate")["items"]][:2] == ["1", "2"]
    assert svc.catalogue(q="scene 3")["total"] == 1


def test_catalogue_picks_up_grades_made_in_stash(svc, stash):
    assert svc.catalogue()["counts"]["legendaire"] == 0
    stash.s["2"]["o_counter"] = 18                    # graded in Stash, not Peaks
    assert svc.catalogue()["counts"]["legendaire"] == 0          # cached listing
    assert svc.catalogue(refresh=True)["counts"]["legendaire"] == 1


# --- API ------------------------------------------------------------------------

def test_catalogue_api_grade_and_restore(svc, monkeypatch):
    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    client = TestClient(app_mod.create_app(svc.cfg))
    r = client.post("/api/catalogue/grade", json={"scene_id": "2", "grade": "exceptionnelle"})
    assert r.status_code == 200 and r.json()["scene"]["tier"] == "exceptionnelle"
    prev = r.json()["previous"]
    assert prev == {"rating100": 100, "o_counter": 16}
    r = client.post("/api/catalogue/restore", json={"scene_id": "2", **prev})
    assert r.json()["scene"]["tier"] == "merveilleuse"
    assert client.post("/api/catalogue/grade", json={"scene_id": "2", "grade": "nope"}).status_code == 400
    listing = client.get("/api/catalogue", params={"tier": "merveilleuse"}).json()
    assert listing["total"] == 1 and listing["names"]["legendaire"] == "Légendaire"
    names = client.post("/api/catalogue/names", json={"names": {"legendaire": "Legendaire"}}).json()
    assert names["legendaire"] == "Legendaire"


def test_js_tier_rules_match_python():
    """The browser copies of tier_of (app.js tierOf, megaboard.js mbTierBadge)
    must agree with peaks.tiers.tier_of on every boundary."""
    import json
    import pathlib
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    root = pathlib.Path(__file__).resolve().parents[1]
    app = (root / "src/peaks/web/static/app.js").read_text()
    mb = (root / "webapp/megaboard.js").read_text()
    tier_fn = re.search(r"function tierOf\(rating100, o\) \{.*?\n\}", app, re.S).group(0)
    mb_fn = re.search(r"function mbTierBadge\(rating100, o\) \{.*?\n\}", mb, re.S).group(0)
    cases = [(r, o) for r in (None, 0, 20, 40, 60, 80, 100) for o in (0, 1, 7, 15, 16, 17, 18, 19)]
    script = (
        "const esc = (s) => s; let MB_TIER_NAMES = {};\n" + tier_fn + "\n" + mb_fn + "\n"
        f"const cases = {json.dumps(cases)};\n"
        "console.log(JSON.stringify(cases.map(([r, o]) => [tierOf(r, o),"
        " (mbTierBadge(r, o).match(/tier-(\\w+)/) || [null, 'unreviewed'])[1]])));"
    )
    out = json.loads(subprocess.run([node, "-e", script], capture_output=True,
                                    text=True, check=True).stdout)
    for (r, o), (js_app, js_mb) in zip(cases, out):
        want = tier_of(r, o)
        assert js_app == want, (r, o, js_app, want)
        assert js_mb == want, (r, o, js_mb, want)
