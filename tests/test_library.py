"""Library management: Stash scene facts, the capability check, the action log
and fingerprint-keyed grade backups (dry-run diff + confirmed restore)."""

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fakestash import FakeStash  # noqa: E402
from peaks.config import Config  # noqa: E402
from peaks.ledger import ActionLog, backup_diff, list_backups, write_backup  # noqa: E402
from peaks.stash_client import StashClient  # noqa: E402


class _Canned(StashClient):
    def __init__(self, responses):
        super().__init__(url="http://stash.test:6969")
        self._responses = list(responses)
        self.calls = []

    def execute(self, query, variables=None):
        self.calls.append((query, variables))
        return self._responses.pop(0)


@pytest.fixture
def stash():
    return FakeStash({
        "1": {"rating100": None, "o_counter": 0, "date": "2024-01-01"},
        "2": {"rating100": 100, "o_counter": 16, "date": "2023-01-01", "tag_ids": ["7"]},
        "3": {"rating100": 100, "o_counter": 18, "date": "2022-01-01"},
        "4": {"rating100": 20, "o_counter": 0, "date": "2021-01-01"},
    }, tags={"7": "blonde"})


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


# --- Stash client ---------------------------------------------------------------

def test_scene_details_carries_tags_fingerprint_and_added_date():
    c = _Canned([{"findScenes": {"scenes": [{
        "id": "5", "title": "", "rating100": 100, "o_counter": 17, "organized": True,
        "created_at": "2025-03-01T10:00:00Z",
        "tags": [{"id": "3", "name": "exceptionnelle"}, {"id": "9", "name": "pov"}],
        "files": [{"path": "/m/5.mp4", "size": 4_000_000_000, "fingerprints": [
            {"type": "phash", "value": "ph5"}, {"type": "oshash", "value": "os5"}]}],
    }]}}])
    d = c.scene_details(["5"])["5"]
    assert d["tag_ids"] == ["3", "9"] and d["tags"] == ["exceptionnelle", "pov"]
    assert d["fingerprint"] == "os5"            # same preference as the embedding cache key
    assert d["phash"] == "ph5" and d["size"] == 4_000_000_000
    assert d["created_at"].startswith("2025-03-01") and d["organized"] is True
    assert "fingerprints { type value }" in c.calls[0][0]


def test_capabilities_introspects_the_schema():
    c = _Canned([{"query": {"fields": [{"name": "findDuplicateScenes"}, {"name": "findJob"}]},
                  "mutation": {"fields": [{"name": "scenesDestroy"}, {"name": "metadataScan"}]}}])
    caps = c.capabilities()
    assert caps["scenesDestroy"] and caps["findDuplicateScenes"] and caps["metadataScan"]
    assert caps["metadataAutoTag"] is False and caps["metadataIdentify"] is False


def test_capability_api_and_unreachable_stash(svc, stash, monkeypatch):
    client = _api(svc, monkeypatch)
    stash.caps["scenesDestroy"] = False
    ops = client.get("/api/stash/capabilities").json()
    assert ops["ok"] and ops["ops"]["scenesDestroy"] is False
    with pytest.raises(RuntimeError, match="scenesDestroy"):
        svc.require_op("scenesDestroy")

    def boom():
        raise ConnectionError("down")
    monkeypatch.setattr(stash, "capabilities", boom)
    down = svc.capabilities(refresh=True)
    assert down["ok"] is False and "down" in down["reason"]
    assert not any(down["ops"].values())


# --- action log --------------------------------------------------------------------

def test_action_log_tail_is_newest_first(tmp_path):
    log = ActionLog(tmp_path / "a.jsonl")
    for i in range(5):
        log.append("grade", scene_id=str(i))
    (tmp_path / "a.jsonl").open("a").write("not json\n")
    assert [e["scene_id"] for e in log.tail(4)] == ["4", "3", "2"]
    assert ActionLog(tmp_path / "missing.jsonl").tail() == []


def test_grades_and_undo_are_logged(svc, monkeypatch):
    svc.grade_scene("1", "legendaire")
    svc.restore_scene_grade("1", None, 0)
    items = _api(svc, monkeypatch).get("/api/history").json()["items"]
    undo, grade = items[0], items[1]
    assert grade["action"] == "grade" and grade["fingerprint"] == "fp1"
    assert grade["before"]["tier"] == "unreviewed" and grade["after"]["tier"] == "legendaire"
    assert grade["path"] == "/data/1.mp4" and grade["title"] == "Scene 1"
    assert undo["action"] == "restore" and undo["after"]["tier"] == "unreviewed"


# --- grade backups -------------------------------------------------------------------

def test_backup_keys_by_fingerprint_and_prunes(tmp_path):
    rows = [{"scene_id": "1", "fingerprint": "a", "rating100": 100, "o_counter": 18},
            {"scene_id": "2", "fingerprint": "b", "rating100": None, "o_counter": 0},   # ungraded
            {"scene_id": "3", "fingerprint": None, "rating100": 100, "o_counter": 16}]  # no file id
    p = write_backup(tmp_path, rows, stamp="20250101")
    data = json.loads(p.read_text())
    assert list(data["scenes"]) == ["a"] and data["scenes"]["a"]["tier"] == "legendaire"
    for d in range(2, 20):
        write_backup(tmp_path, rows, stamp=f"202501{d:02d}", keep=14)
    names = [b["name"] for b in list_backups(tmp_path)]
    assert len(names) == 14 and "grades-backup-20250101.json" not in names


def test_backup_diff_matches_moved_files_by_fingerprint():
    backup = {"scenes": {"a": {"rating100": 100, "o_counter": 18},
                         "b": {"rating100": 100, "o_counter": 16},
                         "gone": {"rating100": 20, "o_counter": 0}}}
    rows = [{"scene_id": "91", "fingerprint": "a", "rating100": 100, "o_counter": 18,
             "tier": "legendaire", "path": "/new/place.mp4"},             # moved, same grade
            {"scene_id": "92", "fingerprint": "b", "rating100": None, "o_counter": 0,
             "tier": "unreviewed"}]                                       # grade lost
    d = backup_diff(backup, rows)
    assert d["same"] == 1 and d["missing"] == 1
    assert [c["scene_id"] for c in d["changes"]] == ["92"]
    assert d["changes"][0]["to"]["tier"] == "merveilleuse"


def test_daily_backup_on_first_listing_then_restore_via_api(svc, stash, monkeypatch):
    client = _api(svc, monkeypatch)
    client.get("/api/catalogue")
    backups = client.get("/api/backups").json()["items"]
    assert len(backups) == 1 and backups[0]["count"] == 3        # scenes 2, 3, 4 are graded
    name = backups[0]["name"]

    stash.s["2"]["o_counter"] = 0          # a grade lost in Stash (e.g. database reset)
    stash.s["3"]["rating100"] = None
    prev = client.get(f"/api/backups/{name}/preview").json()
    assert sorted(c["scene_id"] for c in prev["changes"]) == ["2", "3"]
    assert prev["same"] == 1

    assert client.post(f"/api/backups/{name}/restore", json={}).status_code == 409   # needs confirm
    assert client.get("/api/backups/../../etc/passwd/preview").status_code in (400, 404)
    job = client.post(f"/api/backups/{name}/restore", json={"confirm": True}).json()
    import time
    for _ in range(100):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] != "running":
            break
        time.sleep(0.05)
    assert j["status"] == "done" and j["result"]["restored"] == 2, j
    assert stash.s["2"]["o_counter"] == 16 and stash.s["3"]["rating100"] == 100
    sources = {e.get("source") for e in svc.history() if e["action"] == "grade"}
    assert sources == {f"backup {name}"}
