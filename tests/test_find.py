"""Finding things: search words (all must match; aliases, accents, phrases,
-exclude), the filter chips (several performers / studios / tags, VR, saves,
date added, Os, size, organised), suggestions while typing, studio pages and
co-stars — and the old single-value filters / saved views still working."""

import pytest

pytest.importorskip("fastapi")

from fakestash import FakeStash  # noqa: E402
from peaks.config import Config  # noqa: E402


def _sc(title, who, studio="", tags=(), date="2024-01-01", added="2024-02-01", size=2e9,
        w=3840, h=2160, rating=None, o=0, path=None, details="", organized=False):
    return {"title": title, "performers": [n for _, n in who],
            "performers_detail": [{"id": i, "name": n} for i, n in who],
            "studio": studio, "tags": list(tags), "date": date, "created_at": added + "T10:00:00Z",
            "size": int(size), "width": w, "height": h, "rating100": rating, "o_counter": o,
            "details": details, "organized": organized, **({"path": path} if path else {})}


ANNA, BELLA, CARA = ("p1", "Ánna Rose"), ("p2", "Bella"), ("p3", "Cara Mia")


@pytest.fixture
def stash():
    s = FakeStash({
        "1": _sc("Red Dress Evening", [ANNA], "Velvet", ["Lingerie", "Solo"], rating=100, o=18,
                 details="A slow evening by the fire"),
        "2": _sc("Poolside", [ANNA, BELLA], "Velvet", ["Outdoor"], rating=100, o=17, added="2025-06-01"),
        "3": _sc("VR Night", [BELLA], "Nebula", ["VR"], w=5760, h=2880, size=9e9, organized=True),
        "4": _sc("Kitchen", [CARA], "Nebula", ["Solo"], size=1e9, path="/data/vr/kitchen.mp4"),
        "5": _sc("Garden", [ANNA, CARA], "Sunhouse", ["Outdoor", "Lingerie"], rating=20),
        "6": _sc("Studio Session", [BELLA, CARA], "Sunhouse", [], w=1920, h=1080, o=2, added="2025-09-10"),
    })
    s.performers = [{"id": "p1", "name": "Ánna Rose", "aliases": ["Anya Star"]},
                    {"id": "p2", "name": "Bella", "aliases": []},
                    {"id": "p3", "name": "Cara Mia", "aliases": ["Kara"]}]
    s.markers = [{"scene_id": "2", "seconds": 30.0, "marker_id": "m1", "title": ""}]
    return s


@pytest.fixture
def svc(tmp_path, stash, monkeypatch):
    import peaks.web.service as svc_mod

    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_perf_stats_nowait", lambda self: None)
    return svc_mod.Service(cfg)


def ids(svc, **kw):
    return sorted(x["scene_id"] for x in svc.catalogue(limit=100, **kw)["items"])


# --- search words ------------------------------------------------------------------------

def test_every_word_must_match_somewhere(svc):
    assert ids(svc, q="anna velvet") == ["1", "2"]          # performer + studio, accent-free
    assert ids(svc, q="anna outdoor") == ["2", "5"]         # performer + tag
    assert ids(svc, q="fire") == ["1"]                      # the details text
    assert ids(svc, q="ANYA") == ["1", "2", "5"]            # an alias finds her scenes
    assert ids(svc, q="kara sunhouse") == ["5", "6"]


def test_phrases_and_exclusions(svc):
    assert ids(svc, q='"red dress"') == ["1"]
    assert ids(svc, q='"dress red"') == []
    assert ids(svc, q="anna -velvet") == ["5"]
    assert ids(svc, q='outdoor -"cara mia"') == ["2"]


# --- filter chips ----------------------------------------------------------------------------

def test_performers_all_or_any(svc):
    assert ids(svc, performer_ids=["p1", "p2"]) == ["2"]                     # both in the scene
    assert ids(svc, performer_ids=["p1", "p2"], perf_mode="any") == ["1", "2", "3", "5", "6"]
    assert ids(svc, performer_id="p3") == ["4", "5", "6"]                    # the old single id


def test_studios_any_tags_all_and_not(svc):
    assert ids(svc, studios=["Velvet", "nebula"]) == ["1", "2", "3", "4"]
    assert ids(svc, tags=["lingerie", "outdoor"]) == ["5"]
    assert ids(svc, tags=["Solo"], tags_not=["lingerie"]) == ["4"]
    assert ids(svc, tags_not=["outdoor", "solo"]) == ["3", "6"]
    assert ids(svc, studio="Sunhouse", tag="Outdoor") == ["5"]               # old single values


def test_vr_saves_added_os_size_organised(svc):
    assert ids(svc, vr="vr") == ["3", "4"]                    # 2:1 frame + VR tag; a /vr/ folder
    assert ids(svc, vr="flat") == ["1", "2", "5", "6"]
    assert ids(svc, saves="yes") == ["2"] and "2" not in ids(svc, saves="no")
    assert ids(svc, added_from="2025-01-01") == ["2", "6"]
    assert ids(svc, added_to="2024-12-31") == ["1", "3", "4", "5"]
    assert ids(svc, min_o=17) == ["1", "2"]
    assert ids(svc, size_min=5) == ["3"] and ids(svc, size_max=1.5) == ["4"]
    assert ids(svc, organized="yes") == ["3"]


def test_filters_combine_with_tiers_and_counts(svc):
    r = svc.catalogue(q="anna", limit=100)
    assert r["total"] == 3 and r["counts"]["rejected"] == 1
    assert ids(svc, q="anna", tier="rejected") == ["5"]


def test_saved_views_keep_the_new_filters(svc):
    views = svc.save_view("Anna & Bella outside", {"performer_ids": ["p1", "p2"], "pnames": ["Ánna Rose", "Bella"],
                                                   "tags": ["Outdoor"], "vr": "flat", "studios": [], "junk": 1})
    p = views[-1]["params"]
    assert p == {"performer_ids": ["p1", "p2"], "pnames": ["Ánna Rose", "Bella"], "tags": ["Outdoor"], "vr": "flat"}


# --- suggestions -------------------------------------------------------------------------------

def test_suggest_groups_and_aliases(svc):
    s = svc.catalogue_suggest("an")
    assert [p["id"] for p in s["performers"]][:1] == ["p1"]                # Ánna, accent-free
    s = svc.catalogue_suggest("anya")
    assert s["performers"][0]["id"] == "p1" and s["performers"][0]["alias"] == "Anya Star"
    assert svc.catalogue_suggest("neb")["studios"] == [{"name": "Nebula", "n": 2}]
    assert svc.catalogue_suggest("out")["tags"] == [{"name": "Outdoor", "n": 2}]
    assert [x["scene_id"] for x in svc.catalogue_suggest("pool")["scenes"]] == ["2"]
    assert svc.catalogue_suggest("  ") == {"performers": [], "studios": [], "tags": [], "scenes": []}


# --- studios and co-stars ----------------------------------------------------------------------

def test_studios_list_and_page(svc):
    names = [s["name"] for s in svc.studios_list()["studios"]]
    assert set(names) == {"Velvet", "Nebula", "Sunhouse"}
    sun = next(s for s in svc.studios_list()["studios"] if s["name"] == "Sunhouse")
    assert sun["scenes"] == 1 and sun["rejected"] == 1
    assert [s["name"] for s in svc.studios_list(q="vel")["studios"]] == ["Velvet"]
    assert svc.studios_list(sort="best")["studios"][0]["name"] == "Velvet"
    p = svc.studio_profile("Velvet")
    assert p["scenes"] == 2 and p["tiers"] == {"legendaire": 1, "exceptionnelle": 1}
    assert p["performers"][0]["id"] == "p1" and p["performers"][0]["n"] == 2
    assert p["first"] == "2024-01-01" and [b["scene_id"] for b in p["best"]] == ["1", "2"]
    with pytest.raises(LookupError):
        svc.studio_profile("Nobody")


def test_costars(svc):
    c = svc.performer_costars("p3")
    assert [(x["id"], x["n"]) for x in c] == [("p1", 1), ("p2", 1)]


def test_routes(svc, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(app_mod.create_app(svc.cfg))
    r = api.get("/api/catalogue", params=[("performer_ids", "p1"), ("performer_ids", "p2"), ("limit", 50)]).json()
    assert [x["scene_id"] for x in r["items"]] == ["2"]
    r = api.get("/api/catalogue", params=[("tags_not", "Solo"), ("vr", "vr")]).json()
    assert [x["scene_id"] for x in r["items"]] == ["3"]
    assert api.get("/api/catalogue", params={"vr": "maybe"}).status_code == 422
    assert api.get("/api/catalogue/suggest", params={"q": "bel"}).json()["performers"][0]["id"] == "p2"
    assert api.get("/api/studios").json()["total"] == 3
    assert api.get("/api/studio", params={"name": "Nebula"}).json()["scenes"] == 2
    assert api.get("/api/studio", params={"name": "zzz"}).status_code == 404
    assert api.get("/api/performer/p1/costars").json()["costars"][0]["id"] in {"p2", "p3"}


def test_studio_megaboard_channel(svc, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    r = svc.studio_board("Sunhouse")
    assert r["scenes"] == 1 and r["label"] == "Sunhouse"                       # the rejected scene is left out
    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(app_mod.create_app(svc.cfg))
    d = api.get("/api/board/studio", params={"name": "Velvet"}).json()
    assert d["scenes"] == 2 and d["label"] == "Velvet" and isinstance(d["items"], list)
    assert api.get("/api/board/studio", params={"name": "Nobody"}).status_code == 404
