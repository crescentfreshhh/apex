"""The request path does no repeated work: what a page derives from the whole
library is computed once per library state and shared; a GET never writes a
state file; telemetry writes don't throw the page snapshots away; the index is
never built as a side effect of a status call, and a build never blocks readers
of a loaded index."""

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402

pytest.importorskip("sklearn")


def test_pages_share_one_derived_state_until_the_library_changes(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    svc.train_tier_model()
    calls = []
    real = type(svc)._scene_signals_raw
    monkeypatch.setattr(type(svc), "_scene_signals_raw", lambda self, rows: calls.append(1) or real(self, rows))
    svc.catalogue(limit=20)
    svc.library_summary()
    svc.catalogue(view="likely", limit=20)
    svc.catalogue(tier="legendaire", limit=20)
    assert calls == [1]                                       # one computation for all of them
    svc.grade_scene("u1", "legendaire")                       # the library changed: once more
    svc.catalogue(limit=20)
    svc.library_summary()
    assert calls == [1, 1]
    d1 = svc.catalogue(limit=5)["views"]
    assert d1 == svc.library_summary()["views"]               # the same numbers everywhere


def test_a_single_scene_lookup_never_evicts_the_library_answer(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    svc.catalogue(limit=5)
    key = svc.__dict__["_signals_cache"][0]
    row = svc._catalogue_all()[0]
    svc._scene_signals([row])                                 # the timeline asks about one scene
    assert svc.__dict__["_signals_cache"][0] == key


def test_a_get_never_writes_verdicts(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    svc.train_tier_model()
    svc.grade_scene("u1", "legendaire")                       # an answer, recorded
    before = svc.store().version("verdicts")
    writes = []
    real = type(svc)._vd_save
    monkeypatch.setattr(type(svc), "_vd_save", lambda self: writes.append(1) or real(self))
    for _ in range(3):
        svc.catalogue(view="saved", limit=20)
        svc.catalogue(view="trim", limit=20)
        svc.library_summary()
    assert writes == [] and svc.store().version("verdicts") == before    # nothing written inside the GETs


def test_telemetry_writes_keep_the_snapshots(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    c = TestClient(app_mod.create_app(svc.cfg))
    calls = []
    real = type(svc).library_summary
    monkeypatch.setattr(type(svc), "library_summary", lambda self: calls.append(1) or real(self))
    c.get("/api/catalogue/summary")
    c.post("/api/hold?minutes=60")                             # telemetry: the summary snapshot stands
    c.delete("/api/hold")
    c.get("/api/catalogue/summary")
    assert calls == [1]
    c.post("/api/catalogue/grade", json={"scene_id": "u1", "grade": "legendaire"})   # a library change
    c.get("/api/catalogue/summary")
    assert calls == [1, 1]


def test_readers_of_a_loaded_index_never_wait_on_a_rebuild(tmp_path, monkeypatch):
    from peaks.search import SearchIndex

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    idx = svc.index()
    gate = threading.Event()
    real = SearchIndex.build

    def slow(self, *a, **k):
        gate.wait(10)
        return real(self, *a, **k)
    monkeypatch.setattr(SearchIndex, "build", slow)
    t = threading.Thread(target=lambda: svc.index(_rebuild=True), daemon=True)
    t.start()
    time.sleep(0.2)                                            # the rebuild is under way (index dropped)
    t0 = time.time()
    with svc._index_lock:                                      # the lock is free while it builds
        pass
    assert time.time() - t0 < 0.5
    assert svc.index_if_loaded() is None and svc.index_state() == "building"
    gate.set()
    t.join(10)
    assert svc.index().size == idx.size
