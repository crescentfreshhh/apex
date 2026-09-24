"""Tier tags for the renamer plugin: grading into a tagged tier moves the O-count
first, then makes exactly ONE sceneUpdate carrying the rating, the complete tag
list (one tier tag, other tags kept) and organized. Rejects touch the rating
only. Conflicts are detected; tag sync is explicit; bulk grading undoes as one."""

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fakestash import FakeStash  # noqa: E402
from peaks.config import Config  # noqa: E402
from peaks.tiers import tag_state  # noqa: E402

TAGS = {"7": "blonde", "11": "legendaire", "12": "exceptionnelle", "13": "merveilleuse"}


@pytest.fixture
def stash():
    return FakeStash({
        "1": {"rating100": None, "o_counter": 0, "tag_ids": ["7"]},
        "2": {"rating100": 100, "o_counter": 16, "tag_ids": ["7", "13"], "organized": True},
        "3": {"rating100": 100, "o_counter": 17, "tag_ids": ["11", "12"]},          # two tier tags
        "4": {"rating100": 100, "o_counter": 18, "tag_ids": []},                   # graded in Stash, untagged
        "5": {"rating100": 60, "o_counter": 0, "tag_ids": ["13"], "organized": True},
    }, tags=TAGS)


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
    for _ in range(200):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_tag_state_rules():
    assert tag_state("legendaire", ["legendaire", "pov"], True) == \
        {"present": ["legendaire"], "conflict": None, "needs_sync": False}
    assert tag_state("legendaire", ["legendaire"], False)["needs_sync"]            # not organized
    assert tag_state("legendaire", ["Merveilleuse"], True)["conflict"] == "tier tag disagrees with the grade"
    assert tag_state("exceptionnelle", ["legendaire", "exceptionnelle"], True)["conflict"] == "several tier tags"
    assert tag_state("rejected", ["merveilleuse"], True)["needs_sync"] is False     # rejects never synced
    assert tag_state("upscale", ["personal upscale"], True)["conflict"] is None


def test_grade_is_o_first_then_one_update_with_full_tag_list(svc, stash):
    svc.grade_scene("2", "legendaire")          # 16 → 18, had merveilleuse + blonde
    assert stash.calls == [
        ("add_o", "2"), ("add_o", "2"),
        ("update", "2", {"rating100": 100, "tag_ids": ["7", "11"], "organized": True}),
    ]


def test_grade_creates_a_missing_tier_tag_once(svc, stash):
    svc.grade_scene("1", "upscale")
    assert ("tag_create", "personal upscale") in stash.calls
    new = stash.find_tag_by_name("personal upscale").id
    upd = [c for c in stash.calls if c[0] == "update"]
    assert len(upd) == 1 and upd[0][2]["tag_ids"] == ["7", new] and upd[0][2]["organized"]
    stash.calls.clear()
    svc.grade_scene("5", "upscale")
    assert not [c for c in stash.calls if c[0] == "tag_create"]                  # reused
    assert stash.s["5"]["tag_ids"] == [new]                                    # merveilleuse removed


def test_reject_touches_rating_only(svc, stash):
    svc.grade_scene("2", "reject")
    assert stash.calls == [("update", "2", {"rating100": 20})]
    assert stash.s["2"]["tag_ids"] == ["7", "13"] and stash.s["2"]["organized"] is True


def test_undo_restores_tags_and_organized_in_one_update(svc, stash):
    out = svc.grade_scene("2", "exceptionnelle")
    stash.calls.clear()
    p = out["previous"]
    svc.restore_scene_grade("2", p["rating100"], p["o_counter"], tag_ids=p["tag_ids"],
                            organized=p["organized"])
    assert stash.calls == [("del_o", "2"),
                           ("update", "2", {"rating100": 100, "tag_ids": ["7", "13"],
                                            "organized": True})]


def test_conflict_view_and_rows(svc):
    d = svc.catalogue(view="conflict")
    assert [r["scene_id"] for r in d["items"]] == ["3", "5"]
    assert d["views"]["conflict"] == 2
    row3 = next(r for r in svc._catalogue_all() if r["scene_id"] == "3")
    assert row3["tag_state"]["present"] == ["legendaire", "exceptionnelle"]


def test_tag_sync_preview_then_confirmed_apply(svc, stash, monkeypatch):
    client = _api(svc, monkeypatch)
    prev = client.get("/api/catalogue/tag-sync").json()
    # 3: two tier tags → exceptionnelle only; 4: legendaire untagged. 2 is already right.
    assert sorted(i["scene_id"] for i in prev["items"]) == ["3", "4"]
    assert prev["count"] == 2 and prev["moves"] == 2
    assert stash.calls == []                                                     # preview writes nothing
    assert client.post("/api/catalogue/tag-sync", json={}).status_code == 409
    j = _wait(client, client.post("/api/catalogue/tag-sync", json={"confirm": True}).json())
    assert j["result"]["synced"] == 2, j
    assert stash.s["3"]["tag_ids"] == ["12"] and stash.s["3"]["organized"]
    assert stash.s["4"]["tag_ids"] == ["11"] and stash.s["4"]["organized"]
    updates = [c for c in stash.calls if c[0] == "update"]
    assert all("rating100" not in c[2] for c in updates)                          # grades untouched
    assert {e["action"] for e in svc.history()} == {"tag-sync"}


def test_bulk_grade_and_bulk_undo(svc, stash, monkeypatch):
    client = _api(svc, monkeypatch)
    assert client.post("/api/catalogue/grade-bulk", json={"scene_ids": ["1"], "grade": "x"}).status_code == 400
    j = _wait(client, client.post("/api/catalogue/grade-bulk",
                                  json={"scene_ids": ["1", "5", "1"], "grade": "merveilleuse"}).json())
    r = j["result"]
    assert r["graded"] == 2 and [p["scene_id"] for p in r["previous"]] == ["1", "5"]
    assert stash.s["1"]["tag_ids"] == ["7", "13"] and stash.s["5"]["o_counter"] == 16
    u = _wait(client, client.post("/api/catalogue/restore-bulk", json={"items": r["previous"]}).json())
    assert u["result"]["restored"] == 2
    assert stash.s["1"] == {**stash.s["1"], "rating100": None, "o_counter": 0, "tag_ids": ["7"], "organized": False}
    assert stash.s["5"]["rating100"] == 60 and stash.s["5"]["tag_ids"] == ["13"]


def test_custom_tier_tag_names(svc, stash, monkeypatch):
    client = _api(svc, monkeypatch)
    got = client.post("/api/catalogue/tier-tags", json={"tags": {"legendaire": "Tier 18", "bogus": "x"}}).json()
    assert got["legendaire"] == "Tier 18" and "bogus" not in got
    svc.grade_scene("1", "legendaire")
    assert "Tier 18" in [stash.tags[t] for t in stash.s["1"]["tag_ids"]]
