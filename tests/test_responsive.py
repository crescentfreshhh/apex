"""Pages keep answering while jobs run: new embeds refresh the search index beside
the loaded one instead of dropping it, tier predictions answer from the last
result while they re-score behind, and criteria never retrain inside a request."""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")

from test_index_memory import MODEL, _limit, _vecs, svc  # noqa: E402,F401

from peaks.cache import EmbeddingCache, path_key  # noqa: E402
from peaks.search import SearchIndex  # noqa: E402

GB = 1 << 30


def _add_scene(svc, i):
    c = EmbeddingCache(svc.cfg.embedding.cache_dir)
    c.save(path_key(f"/data/{i}.mp4"), MODEL, np.arange(6, dtype=np.float32), _vecs(np.random.default_rng(i), 6),
           meta={"scene_id": str(i), "path": f"/data/{i}.mp4"})


def _wait(pred, s=10.0):
    end = time.time() + s
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_new_embeds_keep_the_loaded_index_until_the_new_one_swaps_in(svc, monkeypatch):
    _limit(monkeypatch, 20 * GB, 1 * GB)
    idx = svc.index()
    gate = threading.Event()
    real = SearchIndex.build

    def slow(self, *a, **k):
        gate.wait(10)                                 # the rebuild takes a while
        return real(self, *a, **k)
    monkeypatch.setattr(SearchIndex, "build", slow)
    _add_scene(svc, 8)
    svc.refresh_index_soon(MODEL)                     # what an embed pass does when it's done
    t0 = time.time()
    assert svc.index() is idx and time.time() - t0 < 1.0   # pages answer at once, from the old one
    assert svc.index_state() in ("stale", "building")
    gate.set()
    assert _wait(lambda: svc.index() is not idx)
    assert svc.index().size == idx.size + 6 and svc.index_state() == "warm"


def test_nothing_loaded_drops_as_before(svc):
    seen = []
    svc.invalidate_index = lambda model=None: seen.append(model)
    svc.refresh_index_soon(MODEL)
    assert seen == [MODEL]


def test_predictions_answer_from_the_last_result_while_they_rescore(tmp_path, monkeypatch):
    import peaks.web.service as svc_mod

    s = svc_mod.Service.__new__(svc_mod.Service)
    model = object()
    gen = {"g": 1}
    computed = []
    gate = threading.Event()

    class Idx:
        @property
        def gen(self):
            return gen["g"]
    s._tier_model_state = lambda: {"model": model}
    s._picture_entries = lambda: PICS
    s.index = lambda m=None: Idx()
    s._model_name = lambda: MODEL

    def compute(self, rows, st, pics, key):
        if computed:
            gate.wait(10)
        computed.append(key[1])
        out = {"1": {"tier": "legendaire", "gen": key[1]}}
        self._tier_preds = (key, out)
        return out
    PICS = {}
    monkeypatch.setattr(svc_mod.Service, "_tier_predictions_compute", compute)
    rows = [{"scene_id": "1"}]
    assert s._tier_predictions(rows)["1"]["gen"] == 1                 # cold: computed in the request
    gen["g"] = 2                                                      # new embeds: a newer index
    t0 = time.time()
    assert s._tier_predictions(rows)["1"]["gen"] == 1 and time.time() - t0 < 1.0   # the last result, at once
    assert s._tier_predictions(rows)["1"]["gen"] == 1                 # still one re-score, not two
    gate.set()
    assert _wait(lambda: s._tier_predictions(rows)["1"]["gen"] == 2)
    assert computed == [1, 2]


def test_two_cold_requests_compute_once(monkeypatch):
    import peaks.web.service as svc_mod

    s = svc_mod.Service.__new__(svc_mod.Service)
    model, n, pics = object(), [], {}

    class Idx:
        gen = 1
    s._tier_model_state = lambda: {"model": model}
    s._picture_entries = lambda: pics
    s.index = lambda m=None: Idx()
    s._model_name = lambda: MODEL

    def compute(self, rows, st, pics, key):
        n.append(1)
        time.sleep(0.3)
        self._tier_preds = (key, {"x": 1})
        return {"x": 1}
    monkeypatch.setattr(svc_mod.Service, "_tier_predictions_compute", compute)
    ts = [threading.Thread(target=s._tier_predictions, args=([{"scene_id": "1"}],)) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert n == [1]


def test_criteria_vectors_never_retrain_in_a_request(monkeypatch):
    import peaks.web.service as svc_mod

    s = svc_mod.Service.__new__(svc_mod.Service)
    asked = []
    monkeypatch.setattr(svc_mod.Service, "criteria_models",
                        lambda self, wait=True: asked.append(wait) or {a: {"used": False, "scores": {}} for a in
                                                                       ("body", "face", "outfit", "action", "quality")})
    assert s.criteria_scene_vector("1") is None and asked == [False]
