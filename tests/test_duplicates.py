"""Duplicates: Stash's phash groups laid out by file quality, a recommended
keeper, 'keep this' carries the best grade to the kept copy BEFORE deleting the
others (never recorded as rejects), and 'not duplicates' is remembered."""

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fakestash import FakeStash  # noqa: E402
from peaks.config import Config  # noqa: E402


@pytest.fixture
def stash():
    s = FakeStash({
        # group A: a 4K Légendaire-graded low-bitrate copy vs an ungraded 4K high-bitrate one
        "1": {"width": 3840, "height": 2160, "bit_rate": 20e6, "size": 4_000_000_000,
              "rating100": 100, "o_counter": 18, "tag_ids": ["11"], "organized": True},
        "2": {"width": 3840, "height": 2160, "bit_rate": 60e6, "size": 9_000_000_000},
        "3": {"width": 1920, "height": 1080, "bit_rate": 8e6, "size": 1_500_000_000,
              "rating100": 20},
        # group B: two ungraded copies
        "4": {"width": 1920, "height": 1080, "bit_rate": 9e6, "size": 2_000_000_000},
        "5": {"width": 1920, "height": 1080, "bit_rate": 9e6, "size": 2_100_000_000},
        "6": {"width": 1280, "height": 720, "bit_rate": 3e6, "size": 500_000_000},
    }, tags={"11": "legendaire"})
    s.dupes = [["1", "2", "3"], ["4", "5"]]
    return s


@pytest.fixture
def svc(tmp_path, stash, monkeypatch):
    import peaks.web.service as svc_mod

    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    return svc_mod.Service(cfg)


def _api(svc, monkeypatch):
    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    return TestClient(app_mod.create_app(svc.cfg))


def _wait(client, job):
    for _ in range(300):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_groups_with_keeper_and_reclaimable_space(svc, stash, monkeypatch):
    client = _api(svc, monkeypatch)
    assert client.post("/api/duplicates/scan", params={"accuracy": "nope"}).status_code == 400
    j = _wait(client, client.post("/api/duplicates/scan", params={"accuracy": "high"}).json())
    assert ("dupes", 4, -1.0) in stash.calls
    d = client.get("/api/duplicates").json()
    a, b = d["groups"]                                       # biggest reclaim first
    assert a["keep"] == "2" and a["best_grade"] == "legendaire"
    assert a["reclaim"] == 5_500_000_000
    assert b["keep"] == "5" and b["best_grade"] is None     # same quality → bigger file
    assert j["status"] == "done"


def test_keep_carries_the_grade_first_then_deletes_others(svc, stash, monkeypatch):
    client = _api(svc, monkeypatch)
    _wait(client, client.post("/api/duplicates/scan").json())
    assert client.post("/api/duplicates/resolve", json={"keep": "2", "delete": ["1", "3"]}).status_code == 409
    j = _wait(client, client.post("/api/duplicates/resolve",
                                  json={"keep": "2", "delete": ["1", "3", "2"], "confirm": True}).json())
    r = j["result"]
    assert r["deleted"] == 2 and r["carried_grade"] == "legendaire", j
    # order: the kept copy is graded (O moves, one tagged update) BEFORE the destroy
    kinds = [c[0] for c in stash.calls]
    assert kinds.index("update") < kinds.index("destroy")
    assert stash.calls[kinds.index("destroy")][1] == ("1", "3")
    assert stash.s["2"]["o_counter"] == 18 and stash.s["2"]["tag_ids"] == ["11"]
    assert "1" not in stash.s and "3" not in stash.s
    # duplicates are not rejects: nothing remembered, logged as duplicate deletions
    assert len(svc.reject_memory()) == 0
    dels = [e for e in svc.history() if e["action"] == "delete"]
    assert {e["reason"] for e in dels} == {"duplicate"}
    assert any(e["action"] == "duplicate" and e["scene_id"] == "2" for e in svc.history())
    # the resolved group is gone from the cached view
    assert [g["keep"] for g in client.get("/api/duplicates").json()["groups"]] == ["5"]


def test_keep_never_downgrades_the_kept_copy(svc, stash, monkeypatch):
    stash.s["2"].update(rating100=100, o_counter=18, tag_ids=["11"], organized=True)  # already Légendaire, tagged
    stash.s["1"].update(o_counter=16)                       # the other is only Merveilleuse
    client = _api(svc, monkeypatch)
    j = _wait(client, client.post("/api/duplicates/resolve",
                                  json={"keep": "2", "delete": ["1"], "confirm": True}).json())
    assert j["result"]["carried_grade"] is None
    assert stash.s["2"]["o_counter"] == 18


def test_not_duplicates_is_remembered(svc, stash, monkeypatch):
    client = _api(svc, monkeypatch)
    assert client.post("/api/duplicates/ignore", json={"scene_ids": ["4"]}).status_code == 400
    client.post("/api/duplicates/ignore", json={"scene_ids": ["5", "4"]})
    _wait(client, client.post("/api/duplicates/scan").json())
    d = client.get("/api/duplicates").json()
    assert [g["keep"] for g in d["groups"]] == ["2"] and d["ignored"] == 1
    stash.dupes[1].append("6")                              # a new copy → a new group, shown again
    _wait(client, client.post("/api/duplicates/scan").json())
    assert len(client.get("/api/duplicates").json()["groups"]) == 2


def test_only_ids_limits_to_groups_with_new_scenes(svc):
    got = svc.find_duplicates(only_ids={"5"})
    assert [g["keep"] for g in got["groups"]] == ["5"]
    assert svc.cached_duplicates() is None                  # a scoped check alone does not fill the view


def test_keeper_prefers_more_pixels_8k_vr_over_4k(svc):
    from peaks.tiers import quality_of
    rows = [{"scene_id": "a", "tier": "unreviewed", "size": 9e9,
             "quality": quality_of({"width": 3840, "height": 2160, "bit_rate": 60e6})},
            {"scene_id": "b", "tier": "unreviewed", "size": 7e9,
             "quality": quality_of({"width": 8192, "height": 4096, "bit_rate": 45e6})}]
    assert rows[1]["quality"]["res"] == "8K"
    assert svc.dupe_keeper(rows) == "b"


def test_resolve_can_keep_the_other_files(svc, stash, monkeypatch):
    client = _api(svc, monkeypatch)
    j = _wait(client, client.post("/api/duplicates/resolve",
                                  json={"keep": "5", "delete": ["4"], "confirm": True, "delete_file": False}).json())
    assert ("destroy", ("4",), False, True) in stash.calls and j["result"]["freed_bytes"] == 0


# --- the winner takes the group's highest tier, tier TAGS included ------------------

def _tags(stash, sid):
    return sorted(stash.tags[t] for t in stash.s[sid]["tag_ids"])


@pytest.fixture
def tagged(stash):
    stash.tags.update({"12": "exceptionnelle", "13": "merveilleuse", "14": "blonde"})
    return stash


def test_a_tier_tag_on_a_deleted_copy_carries_to_the_keeper(svc, tagged):
    tagged.s["5"]["tag_ids"] = ["12"]                       # ungraded, but tagged exceptionnelle
    tagged.s["4"]["tag_ids"] = ["14"]                       # keeper: unreviewed, a normal tag
    r = svc.resolve_duplicate(None, "4", ["5"], confirm=True)
    assert r["carried_grade"] == "exceptionnelle"
    kinds = [c[0] for c in tagged.calls]
    assert kinds.index("update") < kinds.index("destroy")
    assert tagged.s["4"]["rating100"] == 100 and tagged.s["4"]["o_counter"] == 17
    assert _tags(tagged, "4") == ["blonde", "exceptionnelle"] and tagged.s["4"]["organized"]
    assert any("highest tag" in (e.get("detail") or "") for e in svc.history() if e["action"] == "duplicate")


def test_disagreeing_tags_the_higher_one_wins_and_the_lower_goes(svc, tagged):
    tagged.s["4"].update(rating100=100, o_counter=16, tag_ids=["13"], organized=True)   # Merveilleuse
    tagged.s["5"]["tag_ids"] = ["12"]                                                    # tagged exceptionnelle
    r = svc.resolve_duplicate(None, "4", ["5"], confirm=True)
    assert r["carried_grade"] == "exceptionnelle"
    assert tagged.s["4"]["o_counter"] == 17 and _tags(tagged, "4") == ["exceptionnelle"]


def test_keeper_already_highest_and_tagged_is_left_alone(svc, tagged):
    tagged.s["1"]["tag_ids"] = ["11"]
    tagged.s["2"].update(tag_ids=["12"])                    # the other copy: tagged exceptionnelle
    r = svc.resolve_duplicate(None, "1", ["2"], confirm=True)
    assert r["carried_grade"] is None and tagged.s["1"]["o_counter"] == 18
    assert not [c for c in tagged.calls if c[0] == "update"]


def test_keeper_missing_its_own_tier_tag_gets_it(svc, tagged):
    tagged.s["4"].update(rating100=100, o_counter=17, organized=False)                  # Exceptionnelle, untagged
    r = svc.resolve_duplicate(None, "4", ["5"], confirm=True)
    assert r["carried_grade"] == "exceptionnelle" and _tags(tagged, "4") == ["exceptionnelle"]
    assert tagged.s["4"]["organized"]


def test_rejected_or_unreviewed_copies_set_nothing(svc, tagged):
    tagged.s["5"].update(rating100=20)                      # a reject
    r = svc.resolve_duplicate(None, "4", ["5"], confirm=True)
    assert r["carried_grade"] is None and tagged.s["4"]["rating100"] is None


def test_custom_tier_tag_names_are_honoured(svc, tagged):
    svc.save_tier_tags({"exceptionnelle": "Tier 2"})
    tagged.tags["15"] = "Tier 2"
    tagged.s["5"]["tag_ids"] = ["15"]
    r = svc.resolve_duplicate(None, "4", ["5"], confirm=True)
    assert r["carried_grade"] == "exceptionnelle" and _tags(tagged, "4") == ["Tier 2"]


def test_group_label_counts_tags(svc, tagged, monkeypatch):
    tagged.s["5"]["tag_ids"] = ["12"]
    client = _api(svc, monkeypatch)
    _wait(client, client.post("/api/duplicates/scan").json())
    b = next(g for g in client.get("/api/duplicates").json()["groups"] if g["keep"] in ("4", "5"))
    assert b["best_grade"] == "exceptionnelle"
