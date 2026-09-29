"""Sync's cleanup of the library root: *.zip image sets and folders left with
nothing in them — never the root, never a folder with any other file, and
nothing at all when the root is missing, empty (unmounted) or read-only."""

import pytest

pytest.importorskip("fastapi")

from peaks.config import Config  # noqa: E402
from peaks.web.service import Service  # noqa: E402


@pytest.fixture
def lib(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "Scenes" / "Kept").mkdir(parents=True)
    (root / "Scenes" / "Kept" / "video.mp4").write_bytes(b"v" * 10)
    (root / "Scenes" / "Kept" / "set.zip").write_bytes(b"z" * 100)          # zip beside a video: zip goes
    (root / "Sets" / "A" / "B").mkdir(parents=True)
    (root / "Sets" / "A" / "B" / "pics.ZIP").write_bytes(b"z" * 50)          # only a zip → the whole chain empties
    (root / "Empty" / "Deeper").mkdir(parents=True)
    (root / "Junk").mkdir()
    (root / "Junk" / "Thumbs.db").write_bytes(b"t")                          # a file is a file: stays
    monkeypatch.setenv("PEAKS_CLEANUP_ROOT", str(root))
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache" / "embeddings")
    cfg.modeling.dir = str(tmp_path / "models")
    svc = Service(cfg)
    return svc, root


def test_preview_changes_nothing(lib):
    svc, root = lib
    r = svc.cleanup_library_root(apply=False)
    assert r["zips"] == 2 and r["folders"] == 5 and r["bytes"] == 150 and not r["applied"]
    assert (root / "Sets" / "A" / "B" / "pics.ZIP").exists() and (root / "Empty").exists()


def test_apply_removes_zips_and_empty_folders_bottom_up(lib):
    svc, root = lib
    r = svc.cleanup_library_root(apply=True)
    assert r["applied"] and r["zips"] == 2 and r["folders"] == 5 and r["errors"] == []
    assert not (root / "Sets").exists() and not (root / "Empty").exists()
    assert (root / "Scenes" / "Kept" / "video.mp4").exists() and not (root / "Scenes" / "Kept" / "set.zip").exists()
    assert (root / "Junk" / "Thumbs.db").exists() and root.exists()
    logged = [e for e in svc.history(50) if e.get("action") == "cleanup"]
    assert len(logged) == 7


def test_guards_missing_empty_and_read_only_roots(lib, tmp_path, monkeypatch):
    svc, root = lib
    monkeypatch.setenv("PEAKS_CLEANUP_ROOT", str(tmp_path / "nope"))
    assert "isn't there" in svc.cleanup_library_root(apply=True)["skipped"]
    (tmp_path / "unmounted").mkdir()
    monkeypatch.setenv("PEAKS_CLEANUP_ROOT", str(tmp_path / "unmounted"))
    assert "empty" in svc.cleanup_library_root(apply=True)["skipped"]
    monkeypatch.setenv("PEAKS_CLEANUP_ROOT", str(root))
    import os
    monkeypatch.setattr(os, "access", lambda p, m: False)
    r = svc.cleanup_library_root(apply=True)
    assert "read-write" in r["skipped"] and (root / "Sets").exists()


def test_first_run_previews_then_approval_runs_it(lib):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, root = lib
    c = TestClient(app_mod.create_app(svc.cfg))
    s = c.get("/api/library/cleanup").json()
    assert s["on"] and not s["approved"]
    r = c.post("/api/library/cleanup?action=run").json()       # not approved yet: preview only
    assert not r["last"]["applied"] and (root / "Sets").exists()
    r = c.post("/api/library/cleanup?action=approve").json()
    assert r["approved"] and r["last"]["applied"] and not (root / "Sets").exists()
    assert c.post("/api/library/cleanup?action=off").json()["on"] is False
