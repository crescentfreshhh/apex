"""Saves that overlap must never leave a damaged file (the "Extra data" bug)."""

from __future__ import annotations

import json
import threading

import pytest

from peaks import atomic
from peaks.labels import LabelStore


@pytest.fixture(autouse=True)
def _fresh_repairs():
    atomic.clear_repairs()
    yield
    atomic.clear_repairs()


def _leftovers(folder):
    return [p.name for p in folder.iterdir() if p.name.endswith(".tmp")]


def _labels(n, profile="apex"):
    return [{"key": f"k{i}", "time": float(i), "label": i % 2, "profile": profile,
             "scene_id": str(i), "ts": 1.0, "source": "explicit", "weight": 1.0} for i in range(n)]


def test_overlapping_saves_always_leave_valid_json(tmp_path):
    path = tmp_path / "labels.json"
    start = threading.Barrier(20)
    errors = []

    def save(n):
        try:
            start.wait()
            for _ in range(5):
                atomic.write_json(path, _labels(n), indent=2)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(50 + 37 * i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    data = json.loads(path.read_text())          # parses: no "Extra data"
    assert len(data) in {50 + 37 * i for i in range(20)}
    assert _leftovers(tmp_path) == []


def test_editing_loses_no_label(tmp_path):
    path = tmp_path / "labels.json"
    LabelStore(path).save()
    start = threading.Barrier(16)

    def rate(i):
        start.wait()
        for j in range(5):
            with LabelStore.editing(path) as store:
                store.add(f"k{i}-{j}", 1.0, 1, "apex")

    threads = [threading.Thread(target=rate, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(LabelStore(path).for_profile("apex")) == 80


def test_editing_does_not_save_on_error(tmp_path):
    path = tmp_path / "labels.json"
    with LabelStore.editing(path) as store:
        store.add("a", 1.0, 1, "apex")
    with pytest.raises(RuntimeError), LabelStore.editing(path) as store:
        store.add("b", 1.0, 1, "apex")
        raise RuntimeError("boom")
    assert [l.key for l in LabelStore(path).for_profile("apex")] == ["a"]


def test_extra_data_is_repaired_keeping_the_first_snapshot(tmp_path):
    """What the old shared temp file left: a whole save, then another's tail."""
    path = tmp_path / "labels.json"
    whole = json.dumps(_labels(40), indent=2)
    longer = json.dumps(_labels(55), indent=2)
    path.write_text(whole + longer[len(whole):])
    with pytest.raises(json.JSONDecodeError, match="Extra data"):
        json.loads(path.read_text())

    store = LabelStore(path)
    assert len(store.for_profile("apex")) == 40
    assert json.loads(path.read_text()) == _labels(40)      # rewritten clean
    assert path.read_text().startswith("[\n  {")              # same layout as before
    damaged = list(tmp_path.glob("labels.json.damaged-*"))
    assert len(damaged) == 1 and damaged[0].read_text().startswith(whole)
    (fix,) = atomic.repairs()
    assert fix["file"] == "labels.json" and fix["kept"] == 40
    assert fix["damaged_copy"] == damaged[0].name and fix["dropped_bytes"] > 0


@pytest.mark.parametrize("text", ['[{"key": "a", "time": 1', "not json at all", ""])
def test_other_damage_raises_and_is_left_untouched(tmp_path, text):
    path = tmp_path / "labels.json"
    path.write_text(text)
    with pytest.raises(atomic.DamagedFile, match="labels.json"):
        LabelStore(path)
    assert path.read_text() == text
    assert not list(tmp_path.glob("*.damaged-*"))


def test_settings_style_read_sets_a_damaged_file_aside(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"backup_on": tru')
    assert atomic.read_json(path, default={}, set_aside=True) == {}
    assert not path.exists()
    (aside,) = tmp_path.glob("settings.json.damaged-*")
    assert aside.read_text() == '{"backup_on": tru'
    (fix,) = atomic.repairs()
    assert fix["set_aside"] and fix["kept"] is None


def test_missing_file_gives_default(tmp_path):
    assert atomic.read_json(tmp_path / "nope.json", default=[]) == []


def test_no_temp_file_left_after_a_failed_write(tmp_path):
    path = tmp_path / "state.json"
    atomic.write_json(path, {"a": 1})
    with pytest.raises(TypeError):
        atomic.write_json(path, {"a": object()})
    with pytest.raises(RuntimeError), atomic.temp_for(path) as tmp:
        tmp.write_text("half")
        raise RuntimeError("crash mid-write")
    assert json.loads(path.read_text()) == {"a": 1}
    assert _leftovers(tmp_path) == []


def test_concurrent_settings_saves_stay_valid(tmp_path):
    path = tmp_path / "settings.json"
    start = threading.Barrier(12)

    def save(i):
        start.wait()
        for j in range(10):
            atomic.write_json(path, {"who": i, "pad": "x" * (i * 97 + j)}, indent=2, trailing_newline=True)

    threads = [threading.Thread(target=save, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert "who" in json.loads(path.read_text())
    assert path.read_text().endswith("}\n")
    assert _leftovers(tmp_path) == []


def test_temp_names_are_unique_and_next_to_the_file(tmp_path):
    path = tmp_path / "x.npz"
    with atomic.temp_for(path) as a:
        with atomic.temp_for(path) as b:          # same thread re-enters the lock
            assert a != b and a.parent == b.parent == tmp_path
            b.write_bytes(b"2")
        a.write_bytes(b"1")
    assert path.read_bytes() == b"1"
