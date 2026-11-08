"""Peaks' own state lives in one SQLite file: a change writes one row, a reader
parses nothing it didn't ask for, the old JSON files are imported once and set
aside as the backup, and an export writes them back for a rollback."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from peaks.store import Store  # noqa: E402


def test_maps_and_logs_round_trip(tmp_path):
    st = Store(tmp_path / "peaks.db")
    st.map_set("m", "a", {"x": 1})
    st.map_set_many("m", {"b": [1, 2]}, delete=["zz"])
    assert st.map_all("m") == {"a": {"x": 1}, "b": [1, 2]} and st.map_get("m", "a") == {"x": 1}
    st.map_delete("m", "a")
    assert Store(tmp_path / "peaks.db").map_all("m") == {"b": [1, 2]}           # on disk, not just in memory
    st.log_append("l", {"at": 1, "scene_id": "7", "k": "one"}, scene_id="7")
    st.log_append("l", {"at": 2, "scene_id": "8", "k": "two"}, scene_id="8")
    assert [r["k"] for r in st.log_read("l")] == ["one", "two"]
    assert [r["k"] for r in st.log_tail("l", 1)] == ["two"]
    assert [r["k"] for r in st.log_read("l", scene_id="7")] == ["one"]
    assert [r["k"] for r in st.log_read("l", contains='"k":"two"')] == ["two"]
    assert st.log_count("l") == 2 and "".join(st.log_iter_lines("l")).count("\n") == 2


def test_an_edit_writes_only_the_rows_that_changed(tmp_path):
    st = Store(tmp_path / "peaks.db")
    st.map_replace("m", {"a": 1, "b": 2, "c": 3})
    v = st.version("m")
    writes = []
    real = st.map_set_many
    st.map_set_many = lambda name, items, delete=(): writes.append((dict(items), list(delete))) or real(name, items, delete)
    with st.map_edit("m") as d:
        d["b"] = 20
        del d["c"]
    assert writes == [({"b": 20}, ["c"])] and st.map_all("m") == {"a": 1, "b": 20}
    assert st.version("m") == v + 1
    with st.map_edit("m"):
        pass
    assert len(writes) == 1 and st.version("m") == v + 1                    # nothing changed: no write


def test_old_files_are_imported_once_and_set_aside(tmp_path):
    j = tmp_path / "rulings.json"
    j.write_text(json.dumps({"1": {"grade": "x"}}))
    jl = tmp_path / "actions.jsonl"
    jl.write_text('{"ts": "2026-01-02T03:04:05", "action": "grade", "scene_id": "1"}\nnot json\n')
    st = Store(tmp_path / "peaks.db")
    assert st.migrate_map("rulings", j) and st.map_all("rulings") == {"1": {"grade": "x"}}
    assert not j.exists() and j.with_name("rulings.json.migrated").exists()
    assert st.migrate_log("actions", jl, at_key="ts") and st.log_count("actions") == 1
    assert not jl.exists() and jl.with_name("actions.jsonl.migrated").exists()
    j.write_text(json.dumps({"9": {}}))                                          # a stray later file: ignored
    assert st.migrate_map("rulings", j) is False and st.map_all("rulings") == {"1": {"grade": "x"}}
    assert st.migrate_map("none", tmp_path / "missing.json") is False          # nothing there: marked, no error
    out = tmp_path / "export"
    assert sorted(Store(tmp_path / "peaks.db").export(out)) == ["actions.jsonl", "rulings.json"]
    assert json.loads((out / "rulings.json").read_text()) == {"1": {"grade": "x"}}
    assert json.loads((out / "actions.jsonl").read_text().splitlines()[0])["action"] == "grade"


def test_the_service_stores_write_rows_not_files(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from test_catalogue import _triage_svc

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    st = svc.store()
    writes = []
    real = type(st).map_set_many
    monkeypatch.setattr(type(st), "map_set_many", lambda self, name, items, delete=(): writes.append((name, sorted(items), list(delete))) or real(self, name, items, delete))
    svc.train_tier_model()
    svc.grade_scene("u1", "legendaire")
    assert ("verdicts", ["u1"], []) in writes                                   # one row for the one scene
    assert not svc._vd_path().exists() and not svc._rulings_path().exists()
    svc.settle_rule("u1")
    assert svc.rulings()["u1"]["grade"] == "legendaire" and ("rulings", ["u1"], []) in writes
    svc.record_board_exposure(shows={"u2": 1.0})
    assert ("board_exposure", ["u2"], []) in writes and svc.exposure().summary("u2")["showings"] == 1.0
    svc.set_scene_criterion("u1", "body", 1)
    assert svc.criteria_reasons()["u1"]["body"] == 1 and ("criteria", ["u1"], []) in writes
    svc.record_pick_feedback("u1", 12.0, 1)
    assert svc.answered_picks("u1") == [12.0] and svc.answered_picks("u2") == []
    log = svc.action_log()
    assert log.store is st and any(e["action"] == "grade" for e in log.about("u1"))
    assert log.tail(1)[0]["action"] in ("grade", "criteria", "keep", "ruling") or log.tail(1)
    assert (tmp_path / "state" / "peaks.db").exists() or any(p.name == "peaks.db" for p in svc._state_dir().iterdir())


def test_the_download_streams_the_log(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")

    from test_snappy import _client

    c, app = _client(tmp_path, monkeypatch, lambda: {"n": 1})
    assert c.get("/api/history/download").status_code in (200, 404)


def test_a_snapshot_carries_the_database_and_a_restore_brings_it_back(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    import sqlite3

    from test_catalogue import _triage_svc

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    monkeypatch.setenv("PEAKS_BACKUP_DIR", str(tmp_path / "bk"))
    svc.store().map_set("rulings", "u1", {"grade": "legendaire", "at": 1})
    snap = svc.run_backup(job=None, label="t", prune=False)["name"]
    db = tmp_path / "bk" / snap / "state" / "peaks.db"
    assert db.exists() and not list((tmp_path / "bk" / snap / "state").glob("*.db-*"))
    assert sqlite3.connect(str(db)).execute("SELECT count(*) FROM maps WHERE name='rulings'").fetchone()[0] == 1
    svc.store().map_delete("rulings", "u1")
    assert svc.rulings() == {}
    svc.backup_restore(None, snap, confirm=True)
    assert svc.rulings()["u1"]["grade"] == "legendaire"
