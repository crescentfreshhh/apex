"""Backup & restore: complete snapshot folders where unchanged files are hard
links to the previous snapshot; restore puts everything back and reloads; the
library cleanup never touches the backup folder."""

import io
import json
import os
import tarfile
import time

import pytest

pytest.importorskip("fastapi")

from peaks.config import Config  # noqa: E402
from peaks.web.service import Service  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    conf, data = tmp_path / "config", tmp_path / "data"
    (conf / "backups").mkdir(parents=True)
    (conf / "models" / "perf_photos").mkdir(parents=True)
    (conf / "cache" / "embeddings" / "dino").mkdir(parents=True)
    (conf / "collections").mkdir()
    (data / "Scenes").mkdir(parents=True)
    (data / "Scenes" / "a.mp4").write_bytes(b"v")
    monkeypatch.setenv("PEAKS_SETTINGS", str(conf / "settings.json"))
    monkeypatch.setenv("PEAKS_COLLECTIONS_DIR", str(conf / "collections"))
    monkeypatch.setenv("PEAKS_CLEANUP_ROOT", str(data))
    monkeypatch.delenv("PEAKS_BACKUP_DIR", raising=False)
    (conf / "settings.json").write_text(json.dumps({"tier_names": {"legendaire": "Leg"}}))
    (conf / "actions.jsonl").write_text('{"action": "grade"}\n')
    (conf / "backups" / "grades-backup-20260101.json").write_text("{}")
    (conf / "labels.json").write_text('{"labels": [1, 2, 3]}')
    (conf / "models" / "taste.pkl").write_bytes(b"model-v1")
    (conf / "models" / "tier_model.pkl").write_bytes(b"tier")
    (conf / "models" / "taste.scores.npy").write_bytes(b"cache")          # rebuildable: skipped
    (conf / "models" / "perf_photos" / "1.jpg").write_bytes(b"jpg")
    (conf / "collections" / "Faves.json").write_text("[]")
    (conf / "cache" / "failures.json").write_text("{}")
    for i in range(5):
        (conf / "cache" / "embeddings" / "dino" / f"k{i}.npz").write_bytes(b"e" * 1000)
    cfg = Config()
    cfg.embedding.cache_dir = str(conf / "cache" / "embeddings")
    cfg.modeling.dir = str(conf / "models")
    cfg.modeling.labels_path = str(conf / "labels.json")
    return Service(cfg), conf, data


def _snap_dir(svc, name):
    return svc.backup_root() / name


def test_snapshot_holds_every_part_and_links_unchanged_files(env):
    svc, conf, data = env
    a = svc.run_backup()
    d1 = _snap_dir(svc, a["name"])
    assert d1.parent == data / ".peaks-backups" and (data / ".peaks-backups" / ".peaks-keep").exists()
    man = json.loads((d1 / "manifest.json").read_text())
    assert {p: man["parts"][p]["files"] for p in man["parts"]} == {
        "state": 3, "labels": 1, "models": 3, "collections": 1, "failures": 1, "embeddings": 5}
    assert not (d1 / "models" / "taste.scores.npy").exists()           # a cache, not your data
    assert (d1 / "state" / "backups" / "grades-backup-20260101.json").exists()
    # a week later: one embedding changed, one new
    time.sleep(1.1)
    (conf / "cache" / "embeddings" / "dino" / "k0.npz").write_bytes(b"E" * 2000)
    (conf / "cache" / "embeddings" / "dino" / "k9.npz").write_bytes(b"n" * 500)
    b = svc.run_backup()
    d2 = _snap_dir(svc, b["name"])
    same = os.stat(d1 / "embeddings" / "dino" / "k1.npz").st_ino == os.stat(d2 / "embeddings" / "dino" / "k1.npz").st_ino
    changed = os.stat(d1 / "embeddings" / "dino" / "k0.npz").st_ino != os.stat(d2 / "embeddings" / "dino" / "k0.npz").st_ino
    assert same and changed and b["links_ok"]
    assert b["new_bytes"] == 2500 and b["copied"] == 2                  # only what changed took space
    assert [s["name"] for s in svc.list_snapshots()] == [b["name"], a["name"]]


def test_partial_runs_are_cleaned_and_old_snapshots_pruned(env):
    svc, conf, data = env
    svc.save_backup_settings(backup_keep=2)
    root = svc.backup_root()
    root.mkdir(parents=True)
    (root / "peaks-2020-01-01-0000.partial").mkdir()
    names = [svc.run_backup()["name"] for _ in range(3)]
    assert not list(root.glob("*.partial"))
    assert [s["name"] for s in svc.list_snapshots()] == names[:0:-1]   # newest two kept


def test_no_hard_links_falls_back_to_copies(env, monkeypatch):
    svc, conf, data = env
    svc.run_backup()

    def nolink(*a):
        raise OSError("hard links not supported")

    monkeypatch.setattr(os, "link", nolink)
    r = svc.run_backup()
    assert not r["links_ok"] and r["copied"] == 14
    assert svc.backup_status()["links_ok"] is False


def test_unwritable_data_means_off(env, monkeypatch):
    svc, conf, data = env
    monkeypatch.setattr(os, "access", lambda p, m: False)
    st = svc.backup_status()
    assert not st["writable"] and "read-write" in st["reason"] and not st["due"]
    with pytest.raises(RuntimeError):
        svc.run_backup()


def test_cleanup_never_touches_the_backup_folder(env):
    svc, conf, data = env
    svc.run_backup()
    root = svc.backup_root()
    (root / "stray.zip").write_bytes(b"z")                              # even a zip in there
    (root / "empty-inside").mkdir()
    (data / "Mine").mkdir()
    (data / "Mine" / ".peaks-keep").write_text("")
    (data / "Mine" / "set.zip").write_bytes(b"z")                       # marked folder: left alone
    (data / "Junk" / "Deep").mkdir(parents=True)
    (data / "Junk" / "pics.zip").write_bytes(b"z")                      # everything else: cleaned
    r = svc.cleanup_library_root(apply=True)
    assert r["zips"] == 1 and r["folders"] == 2
    assert (root / "stray.zip").exists() and (root / "empty-inside").exists()
    assert (data / "Mine" / "set.zip").exists() and not (data / "Junk").exists()
    assert svc.list_snapshots()


def test_restore_puts_everything_back_and_reloads(env):
    svc, conf, data = env
    snap = svc.run_backup()["name"]
    assert svc._settings()["tier_names"]["legendaire"] == "Leg"
    # things change / get lost
    (conf / "labels.json").write_text('{"labels": []}')
    (conf / "models" / "taste.pkl").unlink()
    (conf / "models" / "junk-new.pkl").write_bytes(b"x")
    (conf / "settings.json").write_text(json.dumps({"tier_names": {"legendaire": "Changed"}}))
    svc._settings_cache = None
    assert svc._settings()["tier_names"]["legendaire"] == "Changed"
    prev = svc.backup_restore_snapshot_preview(snap)
    assert prev["parts"]["models"] == {"snapshot": 3, "now": 3} and prev["parts"]["labels"]["snapshot"] == 1
    with pytest.raises(PermissionError):
        svc.backup_restore(None, snap)
    r = svc.backup_restore(None, snap, confirm=True)
    assert (conf / "labels.json").read_text() == '{"labels": [1, 2, 3]}'
    assert (conf / "models" / "taste.pkl").read_bytes() == b"model-v1"
    assert not (conf / "models" / "junk-new.pkl").exists()
    assert svc._settings()["tier_names"]["legendaire"] == "Leg"          # reloaded, no restart
    labels = [s["label"] for s in svc.list_snapshots()]
    assert "before restore" in labels and r["safety"] in [s["name"] for s in svc.list_snapshots()]


def test_small_archive_round_trip_and_bad_archives_refused(env):
    svc, conf, data = env
    snap = svc.run_backup()["name"]
    path = svc.export_small_backup(snap)
    with tarfile.open(path) as t:
        assert not [m for m in t.getnames() if "/embeddings" in m]
    new = svc.import_small_backup(path.read_bytes())["name"]
    assert new.startswith("peaks-upload-") and new in [s["name"] for s in svc.list_snapshots()]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, payload in (("peaks-evil/manifest.json", b"{}"), ("peaks-evil/../../escape.txt", b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            t.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError):
        svc.import_small_backup(buf.getvalue())
    assert not (conf.parent / "escape.txt").exists()


def test_weekly_schedule(env):
    svc, conf, data = env
    now = time.time()
    t = time.localtime(now)
    svc.save_backup_settings(backup_day=t.tm_wday, backup_hour=0, backup_minute=0)
    assert svc.backup_due(now)
    svc.run_backup()
    assert not svc.backup_due(now)                                      # done this week
    svc.save_backup_settings(backup_day=(t.tm_wday + 1) % 7)
    assert not svc.backup_due(now + 3 * 86400) or time.localtime(now + 3 * 86400).tm_wday == (t.tm_wday + 1) % 7
    svc.save_backup_settings(backup_on=False)
    assert not svc.backup_due(now)
