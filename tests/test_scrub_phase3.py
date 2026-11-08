"""GPU, decode and index work costs what it has to and no more: the model helpers
pick precision and batch from the device, a library pass over the bf16 index is
spread over threads with the same answer, PCA is fitted once per fold while the
tier model tries its configurations, the senses retrain once the queue is
empty, each scene's sense files are read once per moment-model fit, and the
text-search models leave the GPU when idle."""

import contextlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

pytest.importorskip("sklearn")


def test_precision_and_batch_helpers_are_safe_on_the_cpu():
    from peaks import embedding as emb

    class T:                                            # a torch without CUDA
        class cuda:
            @staticmethod
            def is_available():
                return False
    assert isinstance(emb.autocast_for(T, "cpu"), contextlib.AbstractContextManager)
    assert emb.pick_batch(16) == 16                     # no GPU: the configured size


def test_a_threaded_library_pass_gives_the_same_answer(tmp_path):
    from peaks.cache import EmbeddingCache, path_key
    from peaks.search import SearchIndex

    cache = EmbeddingCache(tmp_path / "c")
    rng = np.random.default_rng(0)
    for i in range(40):
        v = rng.standard_normal((300, 16)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        cache.save(path_key(f"/v/{i}.mp4"), "m", np.arange(300, dtype=np.float32), v, meta={"scene_id": str(i)})
    os.environ["PEAKS_INDEX_F32_MAX_MB"] = "0"          # force bf16 storage
    try:
        idx = SearchIndex(cache, "m").build(cache.keys("m"))
    finally:
        os.environ.pop("PEAKS_INDEX_F32_MAX_MB", None)
    assert idx._bf16 and idx.size == 12000
    q = rng.standard_normal(16).astype(np.float32)
    fast = idx.dot(q)
    slow = np.concatenate([idx.rows(s, min(idx.size, s + 1000)) @ q for s in range(0, idx.size, 1000)])
    assert np.allclose(fast, slow, atol=1e-5)
    hits = idx.search(q, top_k=None, min_score=-1.0, per_scene=3)
    assert 0 < len(hits) <= 50000


def test_pca_is_fitted_once_per_fold_across_configurations(monkeypatch):
    import sklearn.decomposition as dec

    from peaks import tier_model as tm

    fits = []
    real = dec.PCA.fit

    def counting(self, X, y=None):
        fits.append(X.shape[0])
        return real(self, X, y)
    monkeypatch.setattr(dec.PCA, "fit", counting)
    rng = np.random.default_rng(1)
    n = 120
    y = ["merveilleuse"] * 40 + ["exceptionnelle"] * 40 + ["legendaire"] * 40
    Xv = np.concatenate([rng.standard_normal((40, 32)) + off for off in (0, 3, 6)]).astype(np.float32)
    Xq = rng.standard_normal((n, 4)).astype(np.float32)
    tm.fit_best(Xv, Xq, y, who=None)
    folds = 5
    # the grid's PCA sizes + the final fit: one PCA per (fold, k) and one per configuration's final model,
    # not one per fold per configuration (which was ~55)
    assert len(fits) <= len(set(k for k in tm.PCA_GRID if k <= max(tm.PCA_GRID[0], n // 4)) or {1}) * folds + 2


def test_the_senses_retrain_once_the_queue_is_empty(tmp_path, monkeypatch):
    from test_catalogue import _triage_svc

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    trained = []
    monkeypatch.setattr(type(svc), "train_moment_model", lambda self: trained.append("m") or {"verdict": "x"})
    monkeypatch.setattr(type(svc), "train_tier_model", lambda self: trained.append("t") or {"trained": True})
    monkeypatch.setattr(type(svc), "hear_todo", lambda self, first=False: [{"scene_id": "1"}])
    monkeypatch.setattr(type(svc), "motion_todo", lambda self, first=False: [])
    svc._senses_retrain(lambda *_: None, "hear")
    assert trained == [] and svc._retrain_owed is True       # more to hear: owed, not run
    monkeypatch.setattr(type(svc), "hear_todo", lambda self, first=False: [])
    svc._senses_retrain(lambda *_: None, "hear")
    assert trained == ["m", "t"]                              # the queue is empty: now


def test_each_scene_is_read_once_while_the_moment_model_fits(tmp_path, monkeypatch):
    from test_catalogue import _triage_svc

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    reads = []
    monkeypatch.setattr(type(svc), "_sound_load_raw", lambda self, sid: reads.append(sid) or None)
    svc._sense_memo = {}
    for _ in range(6):
        svc._sound_load("7")
    assert reads == ["7"]
    del svc._sense_memo
    svc._sound_load("7")
    assert reads == ["7", "7"]                                # outside a fit: no memo


def test_idle_text_models_leave_the_gpu(tmp_path, monkeypatch):
    from test_catalogue import _triage_svc

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    svc._clip = object()
    svc._clap_text = object()
    svc._text_models_used()
    assert svc.unload_idle_text_models(idle_s=600) == []     # just used: stays
    svc._text_models_at -= 3600
    assert sorted(svc.unload_idle_text_models(idle_s=600)) == ["clap-text", "clip-text"]
    assert svc._clip is None and "_clap_text" not in svc.__dict__
