"""Hybrid pivot: visual 'more like this moment' narrowed by CLIP keywords.

Service-level, offline: a small DINO cache doubles as the CLIP index (via a
monkeypatched `_clip_name`), and `_clip_query_vector` is stubbed so a keyword maps
to a known direction — so we can assert the blend re-ranks deterministically.
"""

import numpy as np
import pytest

pytest.importorskip("fastapi")

from peaks.cache import EmbeddingCache  # noqa: E402
from peaks.config import Config  # noqa: E402
from peaks.search import Hit  # noqa: E402
import peaks.web.service as svc_mod  # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _seed(cache, key, sid, vec, t=0.0):
    cache.save(key, "dinov2", np.array([t], dtype=np.float32),
               np.stack([_unit(vec)]), meta={"scene_id": sid})


def _service(tmp_path, monkeypatch):
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache" / "embeddings")
    cfg.embedding.model = "dino"
    cfg.embedding.dino_model = "dinov2_vits14"   # → cache namespace "dinov2"
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    # source moment ~ [1,0,0]; A is visually close but keyword-irrelevant; B is
    # less visually similar but matches the keyword direction [0,0,1].
    _seed(cache, "k0", "10", [1, 0, 0])
    _seed(cache, "kA", "11", [0.96, 0.28, 0])
    _seed(cache, "kB", "12", [0.6, 0, 0.8])
    svc = svc_mod.Service(cfg)
    monkeypatch.setattr(svc, "_clip_name", lambda: "dinov2")      # reuse the index as CLIP
    monkeypatch.setattr(svc, "has_clip_index", lambda: True)
    monkeypatch.setattr(svc, "_clip_text_vector", lambda text: np.array([0, 0, 1], np.float32))
    return svc


def test_clip_weight_dials_between_visual_and_keyword(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch)

    # pure keyword → the keyword-matching moment (scene 12) wins despite lower visual sim
    kw = svc.search_by_frame("k0", 0.0, top_k=5, clip="anything", clip_weight=1.0)
    assert kw and kw[0].scene_id == "12"
    # pure visual → the visually-closest moment (scene 11) wins, keywords ignored
    vis = svc.search_by_frame("k0", 0.0, top_k=5, clip="anything", clip_weight=0.0)
    assert vis and vis[0].scene_id == "11"


def test_clip_ignored_without_clip_index(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(svc, "has_clip_index", lambda: False)   # no CLIP → fall back to visual
    hits = svc.search_by_frame("k0", 0.0, top_k=5, clip="anything", clip_weight=1.0)
    assert hits and hits[0].scene_id == "11"   # visual order, never empty


def test_clip_rerank_drops_candidates_without_a_clip_vector(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch)

    class _FakeClipIdx:
        def vector_at(self, key, t):
            return np.array([0, 0, 1], np.float32) if key == "ka" else None

    monkeypatch.setattr(svc, "index", lambda model=None: _FakeClipIdx())
    monkeypatch.setattr(svc, "_clip_query_vector", lambda text, neg_weight=0.5: np.array([0, 0, 1], np.float32))
    out = svc._clip_rerank([Hit("A", "ka", 0.0, 0.9), Hit("B", "kb", 0.0, 0.8)], "x", weight=0.5)
    assert [h.scene_id for h in out] == ["A"]   # B dropped — no CLIP vector to judge
