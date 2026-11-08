"""Idle Peaks touches disk and Stash only when something changed: the 5-minute
catalogue check is one light id query; a Sync reads metadata only and never
prunes the performer face references; "is there work?" is remembered; the zoom
run writes its state in batches; small state files are parsed once per change."""

import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402

pytest.importorskip("sklearn")


def test_sync_reads_metadata_only_and_keeps_performer_face_references(tmp_path):
    from types import SimpleNamespace

    from peaks.cache import EmbeddingCache
    from peaks.pipeline import scene_key, sync_cache

    cache = EmbeddingCache(tmp_path / "cache")
    v = np.zeros((2, 4), np.float32)
    k1 = scene_key(SimpleNamespace(id="1", path="/a.mp4", fingerprint="fp1"))
    cache.save(k1, "dinov2", np.zeros(2, np.float32), v, meta={"scene_id": "1", "path": "/a.mp4"})
    cache.save("gone", "dinov2", np.zeros(2, np.float32), v, meta={"scene_id": "9", "path": "/z.mp4"})
    cache.save("p7", "arcface@performers", np.zeros(2, np.float32), v, meta={"pid": "7"})
    loads = []
    real = EmbeddingCache.load
    cache.load = lambda key, model, _r=real: loads.append(key) or _r(cache, key, model)
    scenes = [SimpleNamespace(id="1", path="/a.mp4", fingerprint="fp1")]
    s = sync_cache(scenes, cache, "dinov2", prune=True, log=lambda *_: None)
    assert s["pruned"] == 1 and loads == []                      # nothing in step: no vectors read
    s2 = sync_cache(scenes, cache, "arcface@performers", prune=True, log=lambda *_: None)
    assert s2["pruned"] == 0 and cache.has("p7", "arcface@performers")   # keyed by performer: never an orphan
    scenes2 = [SimpleNamespace(id="1", path="/moved.mp4", fingerprint="fp1")]
    s3 = sync_cache(scenes2, cache, "dinov2", prune=False, log=lambda *_: None)
    assert s3["moved"] == 1 and loads == [k1]                    # a move: that one entry rewritten


def test_the_catalogue_check_asks_stash_for_ids_only(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    svc._catalogue_all(refresh=True)
    asked = []
    stash.all_scene_ids = lambda: asked.append("ids") or set(stash.s)
    monkeypatch.setattr(type(stash), "iter_scenes", lambda self, path_prefix=None: asked.append("listing") or iter(()))
    details = []
    real = stash.scene_details
    stash.scene_details = lambda ids: details.append(list(ids)) or real(ids)
    svc._catalogue_fetch(refresh=False)
    assert asked == ["ids"] and details == []                    # one light call, nothing re-fetched
    stash.s["new1"] = {"rating100": None, "o_counter": 0, "width": 1920, "height": 1080, "tag_ids": [], "organized": False,
                       "fingerprint": "fpnew1", "bit_rate": 5e6, "frame_rate": 30, "video_codec": "h264", "date": "2025-02-02"}
    rows = svc._catalogue_fetch(refresh=False)
    assert details == [["new1"]] and any(r["scene_id"] == "new1" for r in rows)   # only the newcomer's details


def test_the_catalogue_file_is_not_rewritten_when_nothing_changed(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    saves = []
    monkeypatch.setattr(type(svc), "_save_catalogue_disk", lambda self, ids: saves.append(len(ids)))
    svc._catalogue_fetch(refresh=False)
    svc._catalogue_fetch(refresh=False)
    svc._catalogue_fetch(refresh=False)
    assert len(saves) == 1


def test_is_there_work_is_remembered_until_something_changes(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(type(svc), "hear_todo", lambda self, first=False: calls.append(1) or [])
    assert svc.hear_due() is False and svc.hear_due() is False and svc.hear_due() is False
    assert calls == [1]                                           # a "no" stands
    svc._sense_ran()                                              # a run finished: asked again
    assert svc.hear_due() is False and calls == [1, 1]
    svc._cat_edit_seq = svc.__dict__.get("_cat_edit_seq", 0) + 1  # the library changed: asked again
    assert svc.hear_due() is False and calls == [1, 1, 1]


def test_small_state_files_are_parsed_once_per_change(tmp_path):
    from peaks import atomic

    p = tmp_path / "s.json"
    p.write_text(json.dumps({"a": 1}))
    a = atomic.read_json_cached(p)
    a["a"] = 99                                                   # a copy: the cache isn't poisoned
    assert atomic.read_json_cached(p) == {"a": 1}
    time.sleep(0.01)
    p.write_text(json.dumps({"a": 2}))
    assert atomic.read_json_cached(p) == {"a": 2}
    assert atomic.read_json_cached(tmp_path / "none.json", default={"d": 1}) == {"d": 1}


def test_zoom_notes_scenes_in_memory_and_writes_in_batches(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    writes = []
    real = type(svc)._zoom_save
    monkeypatch.setattr(type(svc), "_zoom_save", lambda self, d: writes.append(len(d["scenes"])) or real(self, d))
    for i in range(150):
        svc._zoom_note(str(i), {"key": f"k{i}", "centers": [1.0], "at": 0, "frames": 3})
    assert writes == [100]                                        # one write at 100, not 150
    assert len(svc._zoom_view()["scenes"]) == 150                 # readers see them all meanwhile
    svc._zoom_flush()
    assert writes == [100, 150] and len(svc._zoom_state()["scenes"]) == 150
    svc._zoom_flush()
    assert writes == [100, 150]                                   # nothing pending: no write
