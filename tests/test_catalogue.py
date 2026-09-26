"""Catalogue: the user's grading scheme (5★ + O-count tiers), exact O-count
writes, grading/undo through the API, and quality metadata. Stash is a fake
that records every mutation, so we assert the exact calls Peaks makes."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fakestash import FakeStash  # noqa: E402
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
    assert q == {"res": "4K", "w": 3840, "h": 2160, "mbps": 42.0, "fps": 60.0, "codec": "hevc",
                 "bpp": round(42e6 / (3840 * 2160 * 60), 4)}
    assert quality_of({})["mbps"] is None and quality_of({})["bpp"] is None


@pytest.mark.parametrize("w,h,want", [
    (3840, 2160, "4K"), (4096, 2160, "4K"), (2160, 3840, "4K"),     # UHD, DCI, vertical
    (3840, 1920, "4K"),                                              # VR 2:1 — was "1080p"
    (5120, 2560, "5K"), (5760, 2880, "6K"), (7200, 3600, "7K"),      # VR side-by-side
    (7680, 4320, "8K"), (8192, 4096, "8K"),
    (3440, 1440, "1440p"), (2560, 1440, "1440p"), (1920, 1080, "1080p"),
])
def test_res_class_above_4k_and_vr(w, h, want):
    assert res_class(w, h) == want


def test_tier_features_keep_their_size_for_8k():
    from peaks.tier_model import quality_features
    base = quality_features(quality_of({"width": 3840, "height": 2160, "bit_rate": 40e6, "frame_rate": 60}))
    vr8k = quality_features(quality_of({"width": 8192, "height": 4096, "bit_rate": 80e6, "frame_rate": 60}))
    assert base.shape == vr8k.shape                      # saved models / reject memory stay valid


def test_tier_names_overrides():
    n = tier_names({"legendaire": " Legendaire ", "bogus": "x", "merveilleuse": ""})
    assert n["legendaire"] == "Legendaire"            # trimmed override
    assert n["merveilleuse"] == "Merveilleuse"        # blank → default
    assert "bogus" not in n


# --- fake Stash --------------------------------------------------------------

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
    prev = out["previous"]
    assert prev == {"rating100": None, "o_counter": 0, "tag_ids": [], "organized": False}
    assert out["scene"]["tier"] == "legendaire"
    assert stash.s["1"]["rating100"] == 100 and stash.s["1"]["o_counter"] == 18
    # undo restores an UNRATED scene by clearing the rating, not setting a value
    back = svc.restore_scene_grade("1", prev["rating100"], prev["o_counter"],
                                   tag_ids=prev["tag_ids"], organized=prev["organized"])
    assert back["scene"]["tier"] == "unreviewed"
    assert stash.s["1"]["rating100"] is None and stash.s["1"]["o_counter"] == 0
    assert stash.calls[-1] == ("update", "1", {"clear": ("rating100",), "tag_ids": [],
                                               "organized": False})


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
    assert prev == {"rating100": 100, "o_counter": 16, "tag_ids": [], "organized": False}
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


# --- keeper triage through the service ---------------------------------------------

def _triage_svc(tmp_path, monkeypatch):
    """36 graded scenes in three visually distinct tiers + unreviewed/anomaly ones."""
    import numpy as np

    import peaks.web.service as svc_mod
    from peaks.cache import EmbeddingCache

    pytest.importorskip("sklearn")
    rng = np.random.default_rng(7)
    D = 24
    unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
    look = {t: unit(rng.standard_normal(D)) for t in ("upscale", "merveilleuse", "legendaire")}
    scenes, embed = {}, {}
    grade = {"upscale": (100, 0, 30e6), "merveilleuse": (100, 16, 12e6), "legendaire": (100, 18, 45e6)}
    n = 0
    for t, (r, o, br) in grade.items():
        for _ in range(12):
            sid = str(n)
            n += 1
            scenes[sid] = {"rating100": r, "o_counter": o, "width": 3840, "height": 2160,
                           "bit_rate": br, "frame_rate": 30, "video_codec": "hevc", "date": "2020-01-01"}
            embed[sid] = look[t]
    # unreviewed: one that looks legendary, one that looks like an upscale, one low-bitrate 4K
    for sid, t, br in (("u1", "legendaire", 40e6), ("u2", "upscale", 30e6), ("u3", "merveilleuse", 4e6)):
        scenes[sid] = {"rating100": None, "o_counter": 0, "width": 3840, "height": 2160,
                       "bit_rate": br, "frame_rate": 30, "video_codec": "hevc", "date": "2025-01-01"}
        embed[sid] = look[t]
    scenes["a1"] = {"rating100": 100, "o_counter": 5, "width": 3840, "height": 2160,
                    "bit_rate": 44e6, "frame_rate": 30, "video_codec": "hevc", "date": "2024-01-01"}
    embed["a1"] = look["legendaire"]
    stash = FakeStash(scenes)
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    cfg.embedding.model = "dino"
    cfg.embedding.dino_model = "dinov2_vits14"         # legacy "dinov2" cache namespace
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    for i, (sid, base) in enumerate(embed.items()):
        frames = np.stack([unit(base + 0.3 * rng.standard_normal(D)) for _ in range(10)]).astype(np.float32)
        cache.save(f"k{i}", "dinov2", np.arange(10, dtype=np.float32) * 30, frames,
                   meta={"scene_id": sid})
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_taste_scores", lambda self, m, profile=None: (None, None))
    return svc_mod.Service(cfg), stash


def test_triage_train_views_and_suggestions(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    rep = svc.train_tier_model()
    assert rep["trained"] and rep["n"] == 36
    assert set(rep["classes"]) == {"upscale", "merveilleuse", "legendaire"}
    assert rep["cv"]["exact"] >= 0.9
    assert rep["short"] == {} and rep["counts"]["reject"] == 0

    d = svc.catalogue(view="likely")
    order = [r["scene_id"] for r in d["items"]]
    assert order[0] == "u1"                                    # looks Légendaire → first
    assert order[-1] == "u3"                                   # below the quality floor → last
    assert d["items"][0]["pred"]["tier"] == "legendaire"
    assert d["views"]["likely"] == 3

    q = svc.catalogue(view="quality")
    assert [r["scene_id"] for r in q["items"]] == ["u3"]        # 4 Mbps 4K, below every tiered 4K
    assert "below every 4K scene" in q["items"][0]["flag"]

    a = svc.catalogue(view="anomaly")
    assert a["items"][0]["suggest"]["grade"] == "legendaire"    # O=5 anomaly that looks legendary
    assert "your Légendaire scenes" in a["items"][0]["suggest"]["why"]   # display name, not key
    stash.s["a1"]["tags"] = ["Upscale"]                         # an explicit tag wins
    a2 = svc.catalogue(view="anomaly", refresh=True)
    assert a2["items"][0]["suggest"] == {"grade": "upscale", "why": "tagged 'upscale' in Stash"}

    # a persisted model survives a restart
    from peaks.web.service import Service
    again = Service(svc.cfg)
    again.client = svc.client
    assert again.tier_model_status()["trained"] is True


def test_triage_refuses_without_enough_grades(tmp_path, stash, svc):
    pytest.importorskip("sklearn")
    rep = svc.train_tier_model()
    assert rep["trained"] is False and "two or more tiers" in rep["reason"]


# --- tier board & reels ---------------------------------------------------------------

def test_tier_board_and_reel_use_only_the_chosen_tiers(tmp_path, monkeypatch):
    import peaks.web.service as svc_mod

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    legend = {sid for sid, v in stash.s.items() if v["rating100"] == 100 and v["o_counter"] == 18}
    b = svc.tier_board("legendaire", per_scene=2)
    assert b["scenes"] == 12 and b["label"] == "Légendaire"
    assert b["hits"] and {str(h.scene_id) for h in b["hits"]} <= legend
    both = svc.tier_board("merveilleuse,legendaire")
    assert both["scenes"] == 24 and both["label"] == "Merveilleuse + Légendaire"
    with pytest.raises(ValueError):
        svc.tier_board("rejected")                   # boards/reels are for keeper tiers

    captured = {}
    monkeypatch.setattr(svc_mod.Service, "_build_reel",
                        lambda self, specs, out, job=None, log=print:
                        captured.update(specs=specs, out=str(out)) or {"clips": len(specs)})
    monkeypatch.setattr(svc_mod.Service, "clip_span",
                        lambda self, key, t, model=None: (float(t), float(t) + 10))
    for sid in stash.s:                              # the reel needs file paths
        stash.s[sid]["path"] = f"/data/{sid}.mp4"
    res = svc.export_tiers(tiers="legendaire", count=20)
    assert res["tiers"] == "Légendaire" and res["clips"] == len(captured["specs"]) > 0
    assert {str(s["scene_id"]) for s in captured["specs"]} <= legend
    assert "reel-Légendaire-" in captured["out"]


def test_tier_api_validation(svc, monkeypatch):
    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    client = TestClient(app_mod.create_app(svc.cfg))
    assert client.get("/api/board/tier", params={"tiers": "bogus"}).status_code == 400
    assert client.post("/api/catalogue/reel", params={"tiers": "rejected"}).status_code == 400
    src = client.get("/api/board/sources").json()
    assert [t["key"] for t in src["tiers"]][0] == "legendaire"
