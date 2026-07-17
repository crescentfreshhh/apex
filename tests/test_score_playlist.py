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


def test_scene_timeline_text_mode(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    vecs = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype="float32")
    cache.save("kk", "clip", np.array([0.0, 8.0, 16.0], dtype="float32"), vecs, meta={"scene_id": "5"})
    monkeypatch.setattr(svc, "_clip_text_vector", lambda text: np.array([1, 0, 0], dtype="float32"))

    out = svc.scene_timeline("kk", text="red")
    assert out["model"] == "clip" and out["scene_id"] == "5"
    pts = out["points"]
    assert len(pts) == 3
    assert pts[0][1] > 0.9 and pts[2][1] > 0.9 and pts[1][1] < 0.1  # matches vs not


def test_scene_timeline_missing_scene_is_empty(tmp_path):
    svc, _ = _service(tmp_path)
    assert svc.scene_timeline("nope", text="x")["points"] == []


def test_clip_query_vector_handles_negatives(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)
    vecs = {"beach sunset": np.array([1, 0, 0], dtype="float32"),
            "crowd": np.array([0, 1, 0], dtype="float32")}
    monkeypatch.setattr(svc, "_clip_text_vector", lambda phrase: vecs[phrase])

    q = svc._clip_query_vector("beach sunset -crowd")
    # positive along beach, pushed away from crowd
    assert q[0] > 0.8 and q[1] < -0.1

    plain = svc._clip_query_vector("beach sunset")
    assert plain[0] > 0.99 and abs(plain[1]) < 1e-6  # no negative → pure positive


def test_create_apex_writes_marker(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)

    class C:
        def find_or_create_tag(self, name):
            return type("T", (), {"id": "9", "name": name})()

        def create_scene_marker(self, *, scene_id, seconds, primary_tag_id, title, end_seconds):
            return {"id": "m1", "scene_id": scene_id, "seconds": seconds, "end_seconds": end_seconds}

    monkeypatch.setattr(svc, "client", lambda: C())
    m = svc.create_apex("5", 42.0)
    assert m["seconds"] == 42.0 and m["end_seconds"] == 57.0  # default +15s clip


def test_classify_frame_top_labels(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    cache.save("k", "clip", np.array([0.0], dtype="float32"),
               np.array([[1, 0, 0]], dtype="float32"), meta={"scene_id": "1"})
    monkeypatch.setattr(svc, "_vocab", lambda: ["beach", "office"])
    vmap = {"beach": np.array([1, 0, 0], dtype="float32"), "office": np.array([0, 1, 0], dtype="float32")}
    monkeypatch.setattr(svc, "_clip_text_vector", lambda t: vmap[t])

    out = svc.classify_frame("k", 0.0, top_k=2)
    assert out["labels"][0][0] == "beach"  # frame vector matches "beach"
    labs = dict(out["labels"])
    assert labs["beach"] > labs["office"]


def test_classify_frame_missing_is_empty(tmp_path):
    svc, _ = _service(tmp_path)
    assert svc.classify_frame("nope", 0.0)["labels"] == []


def test_taste_label_train_and_rerank(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache
    from peaks.search import Hit

    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    # two scenes in dinov2 space: scene A frames ~[1,0], scene B ~[0,1]
    cache.save("A", "dinov2", np.array([0.0, 8.0], dtype="float32"),
               np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype="float32"), meta={"scene_id": "1"})
    cache.save("B", "dinov2", np.array([0.0, 8.0], dtype="float32"),
               np.array([[0, 1, 0, 0], [0, 1, 0, 0]], dtype="float32"), meta={"scene_id": "2"})

    # thumbs: love A frames, skip B frames
    svc.add_label("A", 0.0, 1)
    svc.add_label("A", 8.0, 1)
    svc.add_label("B", 0.0, 0)
    svc.add_label("B", 8.0, 0)
    counts = svc.label_counts()
    assert counts["positive"] == 2 and counts["negative"] == 2

    stats = svc.train_taste(model="dinov2")
    assert stats["samples"] == 4 and svc.has_taste()

    # re-rank: a B-ish hit ranked above an A-ish hit should flip toward A
    hits = [Hit(scene_id="2", key="B", time=0.0, score=0.9),
            Hit(scene_id="1", key="A", time=0.0, score=0.85)]
    ranked = svc._rerank_by_taste(hits, "dinov2")
    assert ranked[0].key == "A"  # taste pulls the loved scene to the top


class _ReelClient:
    def iter_markers_by_tag(self, tag, page_size=200):
        yield {"marker_id": "1", "scene_id": "7", "seconds": 10.0, "end_seconds": 25.0,
               "title": "apex", "primary_tag": tag}
        yield {"marker_id": "2", "scene_id": "8", "seconds": 5.0, "end_seconds": 20.0,
               "title": "apex", "primary_tag": tag}

    def scene_details(self, ids):
        return {i: {"path": f"/data/Rando/{i}.mp4"} for i in ids}


def test_export_reel_extracts_and_concats(tmp_path, monkeypatch):
    import subprocess

    svc, _ = _service(tmp_path)
    monkeypatch.setattr(svc, "client", lambda: _ReelClient())
    monkeypatch.setenv("PEAKS_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setattr("os.path.exists", lambda p: True)  # pretend sources + segs exist
    monkeypatch.setattr("os.path.getsize", lambda p: 1000)

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        # the concat step must write the output file
        if "concat" in cmd:
            (tmp_path / "exports").mkdir(parents=True, exist_ok=True)
            out = cmd[-1]
            with open(out, "wb") as f:
                f.write(b"x" * 2_000_000)
        return type("R", (), {"returncode": 0, "stderr": b""})()

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = svc.export_reel(tag="apex")
    assert res["clips"] == 2  # both segments extracted
    assert res["name"].endswith(".mp4") and res["bytes"] > 0
    # two extract calls + one concat call
    assert sum(1 for c in calls if "-f" in c and "mpegts" in c) == 2
    assert any("concat" in c for c in calls)


def test_export_reel_no_markers(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)

    class Empty:
        def iter_markers_by_tag(self, tag, page_size=200):
            return iter(())

    monkeypatch.setattr(svc, "client", lambda: Empty())
    assert svc.export_reel(tag="apex")["clips"] == 0


def test_reel_path_rejects_traversal(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)
    monkeypatch.setenv("PEAKS_EXPORT_DIR", str(tmp_path / "exports"))
    (tmp_path / "exports").mkdir()
    (tmp_path / "exports" / "good.mp4").write_bytes(b"x")
    assert svc.reel_path("good") is not None
    assert svc.reel_path("../../etc/passwd") is None  # sanitized away


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