"""Grading while an ingest runs: a scene you reject (and leave rejected past
the 5 s undo window) is dropped from what's still ahead of it — never
embedded, left out of auto tag and the duplicate check. An undo in time
keeps it in."""

import time

import numpy as np
import pytest

from peaks.cache import EmbeddingCache
from peaks.embedding import FakeEmbedder
from peaks.models import Scene
from peaks.pipeline import _skipping, embed_library


def _scene(id_):
    return Scene.from_dict({"id": id_, "title": "", "scene_markers": [],
                            "files": [{"path": f"/m/{id_}.mp4", "fingerprints": [{"type": "oshash", "value": f"k{id_}"}]}]})


class _Raw:
    interval, mode, pipeline = 2.0, "interval", "raw"

    def __init__(self, on_decode=None):
        self.decoded, self.on_decode = [], on_decode

    def iter_frames_raw(self, path, *, resize_short, crop):
        self.decoded.append(path)
        if self.on_decode:
            self.on_decode(path)
        for i in range(3):
            yield i * 2.0, np.full((crop, crop, 3), i, dtype=np.uint8)


def _emb():
    e = FakeEmbedder(dim=8)
    e.raw_resize, e.raw_crop = 16, 16
    return e


# --- the order and the skipping -------------------------------------------------------

def test_skipping_drops_settled_rejects_and_asks_fresh_ones_again():
    answers = {"2": ["skip"], "3": ["wait", "wait", None], "4": ["wait", "skip"]}
    asked = []

    def should_skip(sc):
        asked.append(sc.id)
        a = answers.get(sc.id)
        return a.pop(0) if a else None
    counts, slept = {}, []
    order = [s.id for s in _skipping([_scene(i) for i in "12345"], should_skip, lambda *_: None, counts, slept.append)]
    assert order == ["1", "5", "3"]                 # 3 was undone in time: embedded last
    assert counts == {"skipped_rejected": 2} and slept


def test_a_scene_still_waiting_after_the_window_is_embedded():
    out = list(_skipping([_scene("1")], lambda sc: "wait", lambda *_: None, {}, lambda _s: None))
    assert [s.id for s in out] == ["1"]             # never stuck: undo window long over


@pytest.mark.parametrize("workers", [1, 3])
def test_embed_never_decodes_a_settled_reject(tmp_path, workers):
    sampler, cache = _Raw(), EmbeddingCache(tmp_path)
    stats = embed_library([_scene(i) for i in "123"], sampler, _emb(), cache, workers=workers,
                          log=lambda *_: None, should_skip=lambda sc: "skip" if sc.id == "2" else None)
    assert stats["embedded"] == 2 and stats["skipped_rejected"] == 1
    assert "/m/2.mp4" not in sampler.decoded and not cache.has("k2", "fake")


def test_a_reject_made_while_its_scene_decodes_skips_the_gpu(tmp_path):
    rejected = set()
    sampler = _Raw(on_decode=lambda path: rejected.add("2") if path == "/m/2.mp4" else None)
    cache = EmbeddingCache(tmp_path)
    stats = embed_library([_scene(i) for i in "123"], sampler, _emb(), cache, workers=2, log=lambda *_: None,
                          should_skip=lambda sc: "skip" if sc.id in rejected else None)
    assert "/m/2.mp4" in sampler.decoded and not cache.has("k2", "fake")
    assert stats["embedded"] == 2 and stats["skipped_rejected"] == 1


def test_without_should_skip_nothing_changes(tmp_path):
    stats = embed_library([_scene(i) for i in "12"], _Raw(), _emb(), EmbeddingCache(tmp_path), log=lambda *_: None)
    assert stats["embedded"] == 2 and stats["skipped_rejected"] == 0


# --- the service: what counts as a settled reject ---------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fakestash import FakeStash

    import peaks.web.service as svc_mod
    from peaks.config import Config

    stash = FakeStash({str(i): {"rating100": None, "o_counter": 0} for i in range(1, 5)})
    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    s = svc_mod.Service(cfg)
    s._stash = stash
    return s


def test_a_reject_settles_after_the_undo_window(svc):
    out = svc.grade_scene("1", "reject")
    now = time.time()
    assert svc.reject_settled("1", now) == "wait" and svc.reject_settled("1", now + 5.1) == "skip"
    assert out["in_ingest"] is False
    svc.restore_scene_grade("1", out["previous"]["rating100"], out["previous"]["o_counter"])
    assert svc.reject_settled("1", now + 10) is None             # undone: back in
    svc.grade_scene("2", "reject")
    svc.grade_scene("2", "merveilleuse")
    assert svc.reject_settled("2", now + 10) is None             # re-graded: back in


def test_the_grade_says_when_a_running_ingest_will_drop_it(svc):
    svc._write_ingest({"running": True, "new": ["3"], "stage": "embed"})
    assert svc.grade_scene("3", "reject")["in_ingest"] is True
    assert svc.grade_scene("4", "reject")["in_ingest"] is False      # not in this ingest
