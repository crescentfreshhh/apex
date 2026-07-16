"""Service-level scoring/playlist glue: playlist writes to the webapp dir, a
write-scoring run rebuilds it, and the calibration read-out surfaces scores."""

import json

import numpy as np
import pytest

pytest.importorskip("fastapi")

from peaks.config import Config  # noqa: E402
import peaks.web.service as svc_mod  # noqa: E402


class _FakeEmbedder:
    name = "dinov2"


def _service(tmp_path):
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache" / "embeddings")
    cfg.embedding.model = "dino"
    return svc_mod.Service(cfg), cfg


class _MarkerClient:
    def iter_markers_by_tag(self, tag, page_size=200):
        yield {
            "marker_id": "1", "scene_id": "7", "seconds": 10.0,
            "end_seconds": 25.0, "title": "apex 0.90", "primary_tag": "apex",
        }

    def stream_url(self, sid, start=None):
        return f"http://stash/scene/{sid}/stream?start={start}"


def test_run_playlist_writes_to_webapp_dir(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)
    monkeypatch.setattr(svc, "client", lambda: _MarkerClient())
    board = tmp_path / "webapp"
    monkeypatch.setenv("PEAKS_WEBAPP_DIR", str(board))

    res = svc.run_playlist(tags=["apex"])
    assert res["count"] == 1
    pl = json.loads((board / "playlist.json").read_text())
    assert pl["count"] == 1 and pl["apexes"][0]["scene_id"] == "7"
    assert pl["apexes"][0]["score"] == 0.90  # parsed from the marker title


def test_run_score_write_rebuilds_playlist(tmp_path, monkeypatch):
    import peaks.pipeline as pl
    import peaks.scoring as sc

    svc, _ = _service(tmp_path)
    monkeypatch.setattr(svc_mod, "get_embedder_for_references", lambda cfg: _FakeEmbedder())
    monkeypatch.setattr(pl, "resolve_references_dir", lambda base, tag: tmp_path)
    monkeypatch.setattr(pl, "load_references", lambda emb, d: np.zeros((2, 4), dtype="float32"))
    monkeypatch.setattr(sc, "make_similarity_scorer", lambda refs, reduce: (lambda v: np.zeros(len(v))))
    monkeypatch.setattr(pl, "score_library", lambda *a, **k: {"scenes": 1, "segments": 3, "skipped": 0, "existing": 0})
    monkeypatch.setattr(svc_mod.Service, "scenes", lambda self, limit=0: [])
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: _MarkerClient())

    seen = {}

    def fake_playlist(self, tags=None, log=None):
        seen["tags"] = tags
        return {"tag": "apex", "count": 3, "out": "x"}

    monkeypatch.setattr(svc_mod.Service, "run_playlist", fake_playlist)

    stats = svc.run_score(write=True, tag="apex")
    assert stats["segments"] == 3 and stats["playlist"] == 3
    assert seen["tags"] == ["apex"]  # board rebuilt for the scored tag


def test_score_calibration_reports_distribution(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    # two scenes of unit vectors; scorer returns fixed scores so percentiles are known
    for k in ("a", "b"):
        cache.save(k, "dinov2", np.array([0.0, 1.0], dtype="float32"),
                   np.ones((2, 4), dtype="float32"), meta={})
    lines = []
    svc._log_score_calibration(cache, "dinov2", lambda v: np.full(len(v), 0.3), cfg.scoring, lines.append)
    text = "\n".join(lines)
    assert "calibration" in text and "max=0.300" in text
    assert "nothing reaches" in text  # 0.3 < default high 0.45