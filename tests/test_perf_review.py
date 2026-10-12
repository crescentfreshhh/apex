"""Performer review: find a performer (names + aliases), a random pick that
favours ones due for a look, the lead-in brief, her scenes best first by id,
and the memory of finished reviews."""

import random

import pytest

pytest.importorskip("fastapi")

from fakestash import FakeStash  # noqa: E402
from peaks.config import Config  # noqa: E402

DAY = 86400.0
NOW = 1_760_000_000.0


def _p(*pairs):
    return {"performers": [n for _, n in pairs], "performers_detail": [{"id": i, "name": n} for i, n in pairs]}


@pytest.fixture
def stash():
    leg, exc, mer, ups, rej = (100, 18), (100, 17), (100, 16), (100, 0), (20, 0)

    def sc(t, who, date, size=2e9, w=3840):
        r, o = t if t else (None, 0)
        return {"rating100": r, "o_counter": o, "date": date, "created_at": date + "T10:00:00Z",
                "size": int(size), "width": w, "height": int(w * 9 / 16), **_p(*who)}

    s = FakeStash({
        "1": sc(leg, [("p1", "Ánna")], "2024-05-01"),
        "2": sc(exc, [("p1", "Ánna"), ("p2", "Bella")], "2023-02-01", w=1920),
        "3": sc(mer, [("p1", "Ánna")], "2024-01-01"),
        "4": sc(None, [("p1", "Ánna")], "2025-01-01", size=1e9),
        "5": sc(rej, [("p1", "Ánna")], "2022-01-01"),
        "6": sc(ups, [("p2", "Bella")], "2024-03-01"),
        "7": sc(mer, [("p3", "Anna")], "2021-01-01"),          # a different Anna (same name, other id)
        "8": sc(leg, [("p2", "Bella")], "2020-01-01"),
        "9": sc(None, [("p2", "Bella")], "2025-02-01"),
    })
    s.performers = [{"id": "p1", "name": "Ánna", "aliases": ["Anya Star"]},
                    {"id": "p2", "name": "Bella", "aliases": []},
                    {"id": "p3", "name": "Anna", "aliases": []}]
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


def test_search_names_and_aliases_counts_exclude_rejects(svc):
    allp = {p["id"]: p for p in svc.perf_review_list(now=NOW)["performers"]}
    assert allp["p1"]["scenes"] == 4 and allp["p1"]["rejected"] == 1
    assert [p["id"] for p in svc.perf_review_list(now=NOW)["performers"]][0] == "p1"   # most scenes first
    assert {p["id"] for p in svc.perf_review_list(q="anna", now=NOW)["performers"]} == {"p1", "p3"}   # accent-insensitive
    assert [p["id"] for p in svc.perf_review_list(q="anya", now=NOW)["performers"]] == ["p1"]         # alias
    assert allp["p1"]["reviewed_days"] is None


def test_random_needs_three_scenes_and_skips_this_weeks(svc):
    rng = random.Random(1)
    picks = {svc.perf_review_random(now=NOW, rng=rng)["id"] for _ in range(40)}
    assert picks == {"p1", "p2"}                                  # p3 has only 1 scene
    svc.perf_review_done("p1", {"seen": 4}, now=NOW - 2 * DAY)
    picks = {svc.perf_review_random(now=NOW, rng=rng)["id"] for _ in range(40)}
    assert picks == {"p2"}                                        # reviewed 2 days ago: not due
    svc.perf_review_done("p2", {"seen": 4}, now=NOW - DAY)
    assert svc.perf_review_random(now=NOW, rng=rng)["id"] in {"p1", "p2"}   # everyone recent: still a pick
    assert svc.perf_review_random(exclude="p2", now=NOW, rng=rng)["id"] == "p1"


def test_random_favours_never_reviewed(svc):
    rng = random.Random(7)
    svc.perf_review_done("p2", {"seen": 4}, now=NOW - 10 * DAY)     # due again, but recently-ish
    picks = [svc.perf_review_random(now=NOW, rng=rng)["id"] for _ in range(300)]
    assert picks.count("p1") > picks.count("p2") * 5               # never (365 d) vs 10 days


def test_brief_has_the_lead_in(svc):
    b = svc.perf_review_brief("p1", now=NOW)
    assert b["name"] == "Ánna" and b["repertoire"]["scenes"] == 4 and b["repertoire"]["rejected"] == 1
    assert b["repertoire"]["first"] == "2023-02-01" and b["repertoire"]["latest"] == "2025-01-01"
    shape = {s["tier"]: s for s in b["shape"]}
    assert shape["legendaire"]["n"] == 1 and shape["legendaire"]["pct"] == 25.0
    assert shape["legendaire"]["lib_pct"] == pytest.approx(100 * 2 / 8, abs=0.1)   # 2 of 8 non-rejected
    assert b["open"]["unreviewed"] == 1
    assert b["footprint"]["gb"] == 7.0 and b["footprint"]["low_gb"] == 1.0 and b["footprint"]["hi_res"] == 3
    assert b["taste"]["best"]["scene_id"] == "1" and b["last_review"] is None
    svc.perf_review_done("p1", {"seen": 4, "regraded": 1, "rejected": 1, "freed_bytes": 2e9}, now=NOW - 3 * DAY)
    lr = svc.perf_review_brief("p1", now=NOW)["last_review"]
    assert lr["days"] == 3 and lr["regraded"] == 1 and lr["times"] == 1
    with pytest.raises(LookupError):
        svc.perf_review_brief("nobody")


def test_catalogue_by_performer_id_best_first(svc):
    r = svc.catalogue(performer_id="p1", tier="legendaire,exceptionnelle,merveilleuse,upscale,anomaly,unreviewed",
                      sort="tier", limit=50)
    assert [x["scene_id"] for x in r["items"]] == ["1", "2", "3", "4"]          # best first, no reject
    assert [x["scene_id"] for x in svc.catalogue(performer_id="p3", limit=50)["items"]] == ["7"]   # the other Anna
    r = svc.catalogue(performer_id="p2", sort="tier", limit=50)
    assert [x["scene_id"] for x in r["items"]][:1] == ["8"]
    up = [x["scene_id"] for x in svc.catalogue(performer_id="p2", sort="tier", dir="asc", limit=50)["items"]]
    assert up[0] == "9"                                                           # worst first when flipped


def test_routes(svc, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(app_mod.create_app(svc.cfg))
    assert api.get("/api/perf-review", params={"q": "bell"}).json()["performers"][0]["id"] == "p2"
    assert api.get("/api/perf-review/random").json()["id"] in {"p1", "p2"}
    assert api.get("/api/perf-review/p1").json()["repertoire"]["scenes"] == 4
    assert api.get("/api/perf-review/zzz").status_code == 404
    assert api.post("/api/perf-review/p1/done", json={"seen": 4, "kept": 3}).json()["times"] == 1
    ids = [x["scene_id"] for x in api.get("/api/catalogue", params={"performer_id": "p1", "sort": "tier"}).json()["items"]]
    assert ids[0] == "1"
