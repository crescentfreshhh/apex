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
    cfg.embedding.dino_model = "dinov2_vits14"  # legacy "dinov2" namespace (tests seed it)
    cfg.embedding.clip_model = "ViT-B-32"  # so _clip_name() == "clip" (tests seed "clip")
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


def test_create_apex_adds_positive_taste_label(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    cache.save("A", "dinov2", np.array([0.0, 45.0], dtype="float32"),
               np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype="float32"), meta={"scene_id": "5"})

    class C:
        def find_or_create_tag(self, name):
            return type("T", (), {"id": "9", "name": name})()

        def create_scene_marker(self, **kw):
            return {"id": "m1", **kw}

    monkeypatch.setattr(svc, "client", lambda: C())
    assert svc.label_counts()["positive"] == 0
    svc.create_apex("5", 42.0)  # scene 5 is embedded (key "A")
    c = svc.label_counts()
    assert c["positive"] == 1 and c["negative"] == 0  # the save became a taste 👍

    # saving a moment whose scene isn't embedded doesn't error or mislabel
    svc.create_apex("999", 3.0)
    assert svc.label_counts()["positive"] == 1  # unchanged — no cache key for scene 999


def test_auto_tag_scores_and_writes(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    # scene A looks like "beach", scene B like "office"
    cache.save("A", "clip", np.array([0.0], dtype="float32"),
               np.array([[1, 0, 0]], dtype="float32"), meta={"scene_id": "1"})
    cache.save("B", "clip", np.array([0.0], dtype="float32"),
               np.array([[0, 1, 0]], dtype="float32"), meta={"scene_id": "2"})
    monkeypatch.setattr(svc, "_vocab", lambda: ["beach", "office"])
    vmap = {"beach": np.array([1, 0, 0], dtype="float32"), "office": np.array([0, 1, 0], dtype="float32")}
    monkeypatch.setattr(svc, "_clip_text_vector", lambda t: vmap[t])
    monkeypatch.setattr(svc, "_clip_text_batch", lambda labels: np.stack([vmap[t] for t in labels]))

    writes = []

    class C:
        def find_or_create_tag(self, name):
            return type("T", (), {"id": {"beach": "10", "office": "20"}[name], "name": name})()

        def add_scene_tags(self, scene_ids, tag_ids):
            writes.append((sorted(scene_ids), tag_ids)); return len(scene_ids)

    monkeypatch.setattr(svc, "client", lambda: C())

    res = svc.auto_tag(top=1)
    assert res["scenes"] == 2 and res["tags"] == 2
    w = {tid[0]: sids for sids, tid in writes}
    assert w["10"] == ["1"] and w["20"] == ["2"]  # beach→sceneA, office→sceneB


def test_add_scene_tags_uses_add_mode(monkeypatch):
    from peaks.stash_client import StashClient

    c = StashClient(url="http://x")
    seen = {}
    monkeypatch.setattr(c, "execute", lambda q, v=None: seen.update(v or {}) or {"bulkSceneUpdate": []})
    n = c.add_scene_tags(["1", "2"], ["10"])
    assert n == 2
    assert seen["input"]["tag_ids"] == {"ids": ["10"], "mode": "ADD"}
    assert seen["input"]["ids"] == ["1", "2"]


def test_vocab_get_save_roundtrip(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)
    monkeypatch.setenv("PEAKS_VOCAB", str(tmp_path / "vocab.txt"))
    # defaults before any file
    d = svc.get_vocab()
    assert d["from_file"] is False and d["count"] > 20

    r = svc.save_vocab("beach\noffice\n# a comment\n")
    assert r["count"] == 2  # comment + blank not counted
    d2 = svc.get_vocab()
    assert d2["from_file"] is True and "beach" in d2["vocab"]
    # saving drops the cached matrix so classification rebuilds on the new terms
    assert svc._vocab_cache is None


def test_models_save_overrides_active_backbone(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    # before any save: the configured defaults are active and unsaved
    m = svc.get_models()
    assert m["dino_model"] == cfg.embedding.dino_model
    assert m["dino_saved"] is False
    assert svc._model_name() == "dinov2"  # small backbone → legacy namespace

    # saving a bigger backbone flips the whole pipeline's cache namespace
    r = svc.save_models(dino_model="dinov2_vitb14", clip_model="ViT-L-14")
    assert r["dino_model"] == "dinov2_vitb14" and r["dino_saved"] is True
    assert svc._model_name() == "dinov2-vitb14"
    assert svc._clip_name() == "clip-vit-l-14"
    assert svc._active_clip_pretrained() == "laion2b_s32b_b82k"
    assert svc.stats()["dino_model"] == "dinov2_vitb14"

    # a fresh Service reads the persisted choice back
    svc2 = svc_mod.Service(cfg)
    assert svc2._active_dino_model() == "dinov2_vitb14"

    # a blank value clears the override back to the container default
    svc.save_models(dino_model="")
    assert svc._active_dino_model() == cfg.embedding.dino_model
    assert svc.get_models()["dino_saved"] is False


def test_models_save_rejects_unknown(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)
    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    with pytest.raises(ValueError):
        svc.save_models(dino_model="dinov2_enormous")
    with pytest.raises(ValueError):
        svc.save_models(clip_model="ViT-Z-99")


def test_classify_frame_top_labels(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    cache.save("k", "clip", np.array([0.0], dtype="float32"),
               np.array([[1, 0, 0]], dtype="float32"), meta={"scene_id": "1"})
    monkeypatch.setattr(svc, "_vocab", lambda: ["beach", "office"])
    vmap = {"beach": np.array([1, 0, 0], dtype="float32"), "office": np.array([0, 1, 0], dtype="float32")}
    monkeypatch.setattr(svc, "_clip_text_vector", lambda t: vmap[t])
    monkeypatch.setattr(svc, "_clip_text_batch", lambda labels: np.stack([vmap[t] for t in labels]))

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


class _MarkerTagClient:
    """Yields one apex marker on scene 1, so the taste centroid has a source."""

    def iter_markers_by_tag(self, tag, page_size=200):
        yield {"marker_id": "1", "scene_id": "1", "seconds": 0.0,
               "end_seconds": 15.0, "title": "apex", "primary_tag": tag}


def _seed_two_scenes(cfg):
    from peaks.cache import EmbeddingCache

    cache = EmbeddingCache(cfg.embedding.cache_dir)
    cache.save("A", "dinov2", np.array([0.0, 8.0], dtype="float32"),
               np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype="float32"), meta={"scene_id": "1"})
    cache.save("B", "dinov2", np.array([0.0, 8.0], dtype="float32"),
               np.array([[0, 1, 0, 0], [0, 1, 0, 0]], dtype="float32"), meta={"scene_id": "2"})


def test_recommend_ranks_by_taste_centroid(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    _seed_two_scenes(cfg)
    monkeypatch.setattr(svc, "client", lambda: _MarkerTagClient())
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")

    r = svc.recommend(top_k=10, per_scene=2)
    assert r["sources"] == 1  # one apex marker on scene A
    assert r["reranked"] is False  # no trained model yet → pure centroid
    # centroid ≈ A's direction, so A's frames should top the list
    assert r["hits"][0].scene_id == "1"
    assert r["hits"][0].score > r["hits"][-1].score


def test_board_pool_is_threshold_driven_and_spans_the_library(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    _seed_two_scenes(cfg)  # apex on scene "1" -> centroid ≈ [1,0,0,0]

    # A spread of scenes at known centroid-cosines from 0.55 up to 0.99, well
    # past the old top-N cap, so we can prove the board reaches the long tail.
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    cosines = np.linspace(0.55, 0.99, 40)
    for i, cs in enumerate(cosines):
        d = np.array([cs, float(np.sqrt(1 - cs * cs)), 0, 0], dtype="float32")
        cache.save(f"S{i}", "dinov2", np.array([0.0, 8.0], dtype="float32"),
                   np.stack([d, d]), meta={"scene_id": str(100 + i)})

    monkeypatch.setattr(svc, "client", lambda: _MarkerTagClient())
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")

    # taste floor honored: every returned moment is at/above it, per_scene capped.
    hi = svc.board_pool(count=500, per_scene=2, min_score=0.8, seed=1)
    assert hi["hits"] and all(h.score >= 0.8 for h in hi["hits"])
    from collections import Counter
    assert max(Counter(h.scene_id for h in hi["hits"]).values()) <= 2

    # a lower floor admits strictly more distinct scenes than a higher one.
    lo = svc.board_pool(count=500, per_scene=2, min_score=0.6, seed=1)
    assert len({h.scene_id for h in lo["hits"]}) > len({h.scene_id for h in hi["hits"]})

    # min_score=0 => no floor: the whole seeded library is eligible.
    allp = svc.board_pool(count=500, per_scene=2, min_score=0.0, seed=1)
    assert len({h.scene_id for h in allp["hits"]}) >= len({h.scene_id for h in lo["hits"]})

    # marching the exclude set forward drains the ≥0.8 set instead of looping.
    seen, rounds = set(), 0
    while rounds < 50:
        r = svc.board_pool(count=500, per_scene=2, min_score=0.8, exclude=seen, seed=1)
        if not r["hits"]:
            break
        seen |= {h.scene_id for h in r["hits"]}
        rounds += 1
    assert not svc.board_pool(count=500, per_scene=2, min_score=0.8, exclude=seen)["hits"]
    assert len(seen) > 2  # spanned many scenes, not just the top couple


def test_board_pool_uniform_random_within_floor(tmp_path, monkeypatch):
    """With a floor set, the board is a UNIFORM-random draw within it — different
    every load (seeded for the test), not a replay of the same top-scored moments."""
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    _seed_two_scenes(cfg)  # apex on scene "1" → centroid ≈ [1,0,0,0]
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    for i, cs in enumerate(np.linspace(0.55, 0.99, 40)):
        d = np.array([cs, float(np.sqrt(1 - cs * cs)), 0, 0], dtype="float32")
        cache.save(f"S{i}", "dinov2", np.array([0.0, 8.0], dtype="float32"),
                   np.stack([d, d]), meta={"scene_id": str(100 + i)})
    monkeypatch.setattr(svc, "client", lambda: _MarkerTagClient())
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")

    ids = lambda r: [(h.scene_id, round(h.time, 1)) for h in r["hits"]]
    a = svc.board_pool(count=12, per_scene=2, min_score=0.6, seed=1)
    a2 = svc.board_pool(count=12, per_scene=2, min_score=0.6, seed=1)
    b = svc.board_pool(count=12, per_scene=2, min_score=0.6, seed=2)

    assert a["hits"] and all(h.score >= 0.6 for h in a["hits"])   # floor honoured
    assert ids(a) == ids(a2)                                      # reproducible per seed
    # a different seed draws a different set → it's random, not a fixed top-N
    assert {s for s, _ in ids(a)} != {s for s, _ in ids(b)}


def test_board_pool_covers_every_scene_scored_by_classifier(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    _seed_two_scenes(cfg)

    # a spread of extra scenes, each pointing a different direction (diverse taste)
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    for i, cs in enumerate(np.linspace(0.4, 0.95, 12)):
        d = np.array([cs, float(np.sqrt(1 - cs * cs)), 0, 0], dtype="float32")
        cache.save(f"S{i}", "dinov2", np.array([0.0, 8.0], dtype="float32"),
                   np.stack([d, d]), meta={"scene_id": str(200 + i)})

    monkeypatch.setattr(svc, "client", lambda: _MarkerTagClient())
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")

    # stub the trained model: probability = first component (higher = more "taste")
    class _Clf:
        def predict_proba(self, m):
            return np.clip(np.asarray(m)[:, 0], 0.0, 1.0)

    monkeypatch.setattr(svc, "_taste_model", lambda profile, model: _Clf())

    idx = svc.index(svc._model_name())
    n_scenes = len({str(s) for s in idx.scene_ids})

    # one peak per scene: every indexed scene appears, scored by the classifier.
    r = svc.board_pool(count=1000, per_scene=1, min_score=0.0)
    assert r["scored_by"] == "classifier"
    assert r["scenes"] == n_scenes and r["moments"] == n_scenes
    assert len({str(h.scene_id) for h in r["hits"]}) == n_scenes  # ≥1 moment for ALL scenes

    # depth: per_scene=2 gives up to 2 moments/scene but still covers every scene.
    from collections import Counter
    r2 = svc.board_pool(count=1000, per_scene=2, min_score=0.0)
    assert len({str(h.scene_id) for h in r2["hits"]}) == n_scenes
    assert max(Counter(str(h.scene_id) for h in r2["hits"]).values()) <= 2

    # the floor is an optional tightener: raising it drops low-scoring scenes.
    tight = svc.board_pool(count=1000, per_scene=1, min_score=0.9)
    assert tight["scenes"] < n_scenes
    assert all(h.score >= 0.9 for h in tight["hits"])


class _NoMarkerClient:
    def iter_markers_by_tag(self, tag, page_size=200):
        return iter(())


def test_board_scores_by_nearest_mode_not_the_average(tmp_path, monkeypatch):
    """Two distinct loved interests → BOTH light up, not just their midpoint.
    A single averaged centroid would score the A/B midpoint highest; nearest-mode
    scoring scores pure-A and pure-B moments above the midpoint — which no single
    centroid can do. This is the 'spans my whole taste' guarantee."""
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")

    cache = EmbeddingCache(cfg.embedding.cache_dir)
    A = np.array([1, 0, 0, 0], dtype="float32")
    B = np.array([0, 1, 0, 0], dtype="float32")
    mid = np.array([1, 1, 0, 0], dtype="float32") / np.sqrt(2.0)  # 45° between A and B
    t0 = np.array([0.0], dtype="float32")
    # two loved moments, one in each cluster; three held-out test scenes
    cache.save("loveA", "dinov2", t0, np.stack([A]), meta={"scene_id": "10"})
    cache.save("loveB", "dinov2", t0, np.stack([B]), meta={"scene_id": "11"})
    cache.save("testA", "dinov2", t0, np.stack([A]), meta={"scene_id": "20"})
    cache.save("testB", "dinov2", t0, np.stack([B]), meta={"scene_id": "21"})
    cache.save("mid", "dinov2", t0, np.stack([mid]), meta={"scene_id": "22"})

    monkeypatch.setattr(svc, "client", lambda: _NoMarkerClient())  # thumbs are the only taste
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")

    # love one moment in each cluster — no trained model, so scoring uses modes
    svc.add_label("loveA", 0.0, 1)
    svc.add_label("loveB", 0.0, 1)

    r = svc.board_pool(count=1000, per_scene=1, min_score=0.0)
    assert r["scored_by"] == "modes"
    best = {str(h.scene_id): h.score for h in r["hits"]}
    # both distinct interests beat their midpoint — impossible for one centroid
    assert best["20"] > best["22"] and best["21"] > best["22"]
    assert best["20"] > 0.99 and best["21"] > 0.99  # each sits on its own mode

    # the diversify path (pool > count) runs on real vectors and returns `count`
    d = svc.board_pool(count=2, per_scene=1, min_score=0.0, seed=1)
    assert len(d["hits"]) == 2 and d["scored_by"] == "modes"


def test_taste_metrics_distribution_and_bands(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    _seed_two_scenes(cfg)  # scene A dir [1,0,0,0], scene B dir [0,1,0,0]
    monkeypatch.setattr(svc, "client", lambda: _MarkerTagClient())  # apex on scene "1" (A)
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")

    m = svc.taste_metrics()
    assert m["has_taste"] is True
    assert m["indexed"]["frames"] == 4 and m["indexed"]["scenes"] == 2
    assert m["sources"]["apex"] >= 1 and m["sources"]["used_in_centroid"] == m["sources"]["total"]

    # bands: cutoffs non-increasing as the band loosens; moments non-decreasing;
    # distinct scenes never exceed the moment count.
    cuts = [b["cutoff"] for b in m["bands"]]           # Top1% .. Top25%
    moms = [b["moments"] for b in m["bands"]]
    assert cuts == sorted(cuts, reverse=True)
    assert moms == sorted(moms)
    assert all(b["scenes"] <= b["moments"] for b in m["bands"])
    d = m["distribution"]
    assert d["p50"] <= d["p90"] <= d["p99"] <= d["max"]
    assert len(m["histogram"]["counts"]) == 24 and len(m["histogram"]["edges"]) == 25
    # CDF for the client-side threshold slider: monotonically non-increasing.
    cdf = m["cdf"]
    assert len(cdf["thresholds"]) == 101
    assert cdf["moments_ge"] == sorted(cdf["moments_ge"], reverse=True)
    assert cdf["scenes_ge"] == sorted(cdf["scenes_ge"], reverse=True)

    # absolute threshold: centroid ~ scene A, so A's 2 frames score ~1, B's ~0.
    mt = svc.taste_metrics(threshold=0.5)["threshold"]
    assert mt["value"] == 0.5 and mt["moments"] == 2 and mt["scenes"] == 1
    assert mt["percentile"] == 50.0


def test_taste_metrics_empty_without_taste(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    _seed_two_scenes(cfg)

    class _NoMarkers:
        def iter_markers_by_tag(self, tag, page_size=200):
            return iter(())

    monkeypatch.setattr(svc, "client", lambda: _NoMarkers())
    m = svc.taste_metrics()  # no apexes, no thumbs -> no centroid
    assert m["has_taste"] is False
    assert m["labels"] == {"positive": 0, "negative": 0, "has_model": False}
    assert "bands" not in m


def test_recommend_shuffle_varies_but_is_seeded(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    # 20 scenes all near the taste direction, so scores are close and the
    # rank-weighted shuffle has room to reorder the top of the feed.
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    rng = np.random.default_rng(0)
    for i in range(1, 21):
        v = np.array([1.0, 0, 0, 0], dtype="float32") + rng.normal(0, 0.15, 4).astype("float32")
        v = v / np.linalg.norm(v)
        cache.save(f"s{i}", "dinov2", np.array([0.0], dtype="float32"),
                   v.reshape(1, 4), meta={"scene_id": str(i)})
    monkeypatch.setattr(svc, "client", lambda: _MarkerTagClient())
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")

    order = lambda r: [h.scene_id for h in r["hits"]]
    # no shuffle → deterministic, same feed every call
    assert order(svc.recommend(top_k=6, shuffle=False)) == order(svc.recommend(top_k=6, shuffle=False))
    # different seeds → a different mix (the whole point of "Rebuild")
    a = order(svc.recommend(top_k=6, shuffle=True, seed=1))
    b = order(svc.recommend(top_k=6, shuffle=True, seed=2))
    assert a != b
    assert a == order(svc.recommend(top_k=6, shuffle=True, seed=1))  # same seed reproduces
    # shuffle defaults to rebuild, so the Rebuild button varies without a flag
    assert order(svc.recommend(top_k=6, rebuild=True, seed=1)) == a
    # still a real, correctly-sized feed drawn from the library
    assert len(a) == 6 and set(a) <= {str(i) for i in range(1, 21)}


def test_recommend_min_score_floors_the_pool(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    _seed_two_scenes(cfg)  # centroid ~ scene A ([1,0,0,0]); A frames ~1.0, B frames ~0.0
    monkeypatch.setattr(svc, "client", lambda: _MarkerTagClient())
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")

    allh = svc.recommend(top_k=10)["hits"]
    assert any(h.scene_id == "2" for h in allh)  # scene B present without a floor
    floored = svc.recommend(top_k=10, min_score=0.5)["hits"]
    assert floored and all(h.score >= 0.5 for h in floored)
    assert all(h.scene_id == "1" for h in floored)  # only on-taste scene A survives


def test_performer_board_plays_her_embedded_scenes(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    for sid in ("1", "2", "3"):  # three embedded scenes
        cache.save(f"k{sid}", "dinov2", np.array([0.0, 5.0], dtype="float32"),
                   np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype="float32"),
                   meta={"scene_id": sid, "path": f"/x/{sid}.mp4"})

    class _C:
        def scene_details(self, ids):
            return {str(ids[0]): {"performers_detail": [{"id": "p1", "name": "Jane Doe"}]}}

        def scenes_for_performer(self, pid, limit=500):
            return ["1", "2", "3", "99"]  # 99 is not embedded

    monkeypatch.setattr(svc, "client", lambda: _C())
    r = svc.performer_board("1", count=100)
    assert r["performer"] == "Jane Doe"
    sids = {h.scene_id for h in r["hits"]}
    assert sids and sids <= {"1", "2", "3"} and "99" not in sids


def test_default_vocab_is_rich_and_unique():
    from peaks.vocab import DEFAULT_VOCAB

    assert len(DEFAULT_VOCAB) > 250
    assert all(isinstance(t, str) and t.strip() for t in DEFAULT_VOCAB)
    assert len(set(DEFAULT_VOCAB)) == len(DEFAULT_VOCAB)  # no duplicates
    joined = " ".join(DEFAULT_VOCAB).lower()
    for term in ("breasts", "butt", "doggy", "lingerie", "blonde", "thick", "bbw", "stockings"):
        assert term in joined


def test_recommend_reranks_with_trained_model(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    _seed_two_scenes(cfg)
    monkeypatch.setattr(svc, "client", lambda: _MarkerTagClient())
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")

    # centroid points at scene A (the apex marker), but teach the classifier the
    # opposite: dislike A, like B — then train.
    svc.add_label("A", 0.0, 0); svc.add_label("A", 8.0, 0)
    svc.add_label("B", 0.0, 1); svc.add_label("B", 8.0, 1)
    svc.train_taste(model="dinov2")

    r = svc.recommend(top_k=10, per_scene=2)
    assert r["reranked"] is True  # a trained model exists → retrieve-then-rerank
    # reranking overrides the centroid: the liked scene B rises to the top
    assert r["hits"][0].scene_id == "2"


def test_labelstore_remove_all_and_recent(tmp_path):
    import time

    from peaks.labels import LabelStore

    store = LabelStore(tmp_path / "labels.json")
    store.add("A", 0.0, 1, "apex")
    store.add("B", 0.0, 0, "apex")
    # backdate A so only B counts as "recent"
    for lab in store.for_profile("apex"):
        if lab.key == "A":
            lab.ts = time.time() - 3600
    assert store.remove("apex", newer_than=time.time() - 600) == 1  # drops recent B
    assert {lab.key for lab in store.for_profile("apex")} == {"A"}
    assert store.remove("apex") == 1 and store.for_profile("apex") == []  # wipe the rest


def test_delete_taste_full_wipe_removes_model(tmp_path):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    _seed_two_scenes(cfg)
    svc.add_label("A", 0.0, 1); svc.add_label("A", 8.0, 1)
    svc.add_label("B", 0.0, 0); svc.add_label("B", 8.0, 0)
    svc.train_taste(model="dinov2")
    assert svc.has_taste()

    r = svc.delete_taste()  # full wipe
    assert r["removed"] == 4 and r["positive"] == 0 and r["model_deleted"] is True
    assert not svc.has_taste()
    assert svc.label_counts()["positive"] == 0


def test_delete_taste_purge_apexes_destroys_markers(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    _seed_two_scenes(cfg)
    svc.add_label("A", 0.0, 1); svc.add_label("B", 0.0, 0)

    destroyed = []

    class _PurgeClient:
        def iter_markers_by_tag(self, tag, page_size=200):
            yield {"marker_id": "11", "scene_id": "1", "seconds": 0.0}
            yield {"marker_id": "22", "scene_id": "2", "seconds": 8.0}

        def destroy_scene_markers(self, ids, chunk=100):
            destroyed.extend(ids)
            return len(ids)

    monkeypatch.setattr(svc, "client", lambda: _PurgeClient())

    r = svc.delete_taste(purge_apexes=True)
    assert destroyed == ["11", "22"]        # both apex markers deleted from Stash
    assert r["apexes_removed"] == 2
    assert r["removed"] == 2 and r["positive"] == 0

    # regression: a plain full wipe must NOT touch apex markers
    destroyed.clear()
    svc.add_label("A", 0.0, 1); svc.add_label("B", 0.0, 0)
    r2 = svc.delete_taste()  # no purge_apexes
    assert destroyed == []
    assert "apexes_removed" not in r2


def test_delete_taste_recent_window_keeps_and_retrains(tmp_path):
    import time

    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    _seed_two_scenes(cfg)
    svc.add_label("A", 0.0, 1); svc.add_label("B", 0.0, 0)  # will backdate → "old"
    svc.add_label("A", 8.0, 1); svc.add_label("B", 8.0, 0)  # recent
    st = svc._label_store()
    for lab in st.for_profile("apex"):
        if lab.time == 0.0:
            lab.ts = time.time() - 7200  # 2h old
    st.save()

    r = svc.delete_taste(within_minutes=60)  # drop the 2 recent, keep the 2 old
    assert r["removed"] == 2
    assert r["positive"] == 1 and r["negative"] == 1
    assert r["retrained"] is True and svc.has_taste()


def test_build_and_get_galaxy(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    monkeypatch.setenv("PEAKS_GALAXY_DIR", str(tmp_path / "galaxy"))
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    rng = np.random.default_rng(1)
    for s in range(8):
        base = np.zeros(6, dtype="float32"); base[s % 2] = 1.0
        v = base + rng.normal(0, 0.05, size=(3, 6)).astype("float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        cache.save(f"k{s}", "dinov2", np.array([0.0, 5.0, 10.0], dtype="float32"),
                   v, meta={"scene_id": str(s + 1)})

    assert svc.get_galaxy("dino")["built"] is False  # nothing cached yet
    out = svc.build_galaxy(space="dino", method="pca")  # pca → no numba in CI
    assert out["scenes"] == 8

    g = svc.get_galaxy("dino")
    assert g["built"] is True and len(g["scenes"]) == 8
    r = g["scenes"][0]
    assert {"scene_id", "key", "t", "x", "y", "c", "taste", "url"} <= set(r)
    assert 0.0 <= r["x"] <= 1.0 and 0.0 <= r["y"] <= 1.0


def test_diversify_breaks_up_near_duplicates(tmp_path):
    from peaks.cache import EmbeddingCache
    from peaks.search import Hit

    svc, cfg = _service(tmp_path)
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    # three scenes: A and B look near-identical; C is different
    cache.save("A", "dinov2", np.array([0.0], dtype="float32"),
               np.array([[1, 0, 0, 0]], dtype="float32"), meta={"scene_id": "1"})
    cache.save("B", "dinov2", np.array([0.0], dtype="float32"),
               np.array([[0.99, 0.14, 0, 0]], dtype="float32"), meta={"scene_id": "2"})
    cache.save("C", "dinov2", np.array([0.0], dtype="float32"),
               np.array([[0, 0, 1, 0]], dtype="float32"), meta={"scene_id": "3"})
    # ranking has the two look-alikes on top, the different one last
    ranked = [Hit("1", "A", 0.0, 0.99), Hit("2", "B", 0.0, 0.98), Hit("3", "C", 0.0, 0.90)]

    pure = svc._diversify(ranked, "dinov2", k=2, diversity=0.0)
    assert [h.key for h in pure] == ["A", "B"]  # pure ranking keeps the duplicates
    diverse = svc._diversify(ranked, "dinov2", k=2, diversity=0.6)
    assert [h.key for h in diverse] == ["A", "C"]  # variety pulls C above the near-dupe B


def test_recommend_exclude_drops_scenes(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    _seed_two_scenes(cfg)
    monkeypatch.setattr(svc, "client", lambda: _MarkerTagClient())
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")

    sids = {str(h.scene_id) for h in svc.recommend(top_k=10)["hits"]}
    assert sids  # baseline has results
    r = svc.recommend(top_k=10, exclude={"1"})
    assert all(str(h.scene_id) != "1" for h in r["hits"])  # excluded scene gone
    assert svc.recommend(top_k=10, exclude=sids)["hits"] == []  # exclude all → empty


def test_recommend_empty_without_taste(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    _seed_two_scenes(cfg)

    class _NoMarkers:
        def iter_markers_by_tag(self, tag, page_size=200):
            return iter(())

    monkeypatch.setattr(svc, "client", lambda: _NoMarkers())
    r = svc.recommend()
    assert r["sources"] == 0 and r["hits"] == []


def test_index_rebuilds_as_cache_grows(tmp_path):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    assert svc.index("dinov2").size == 0  # nothing embedded yet
    # a scene embeds mid-session → an empty cached index is rebuilt on access
    cache.save("A", "dinov2", np.array([0.0], dtype="float32"),
               np.array([[1, 0, 0, 0]], dtype="float32"), meta={"scene_id": "1"})
    assert svc.index("dinov2").size == 1
    # another embeds; a plain access reuses the cached (now-stale) index,
    # but refresh=True picks up the growth (what the swipe trainer uses)
    cache.save("B", "dinov2", np.array([0.0], dtype="float32"),
               np.array([[0, 1, 0, 0]], dtype="float32"), meta={"scene_id": "2"})
    assert svc.index("dinov2").size == 1
    assert svc.index("dinov2", refresh=True).size == 2


def test_autotrain_due_needs_threshold_and_both_classes(tmp_path):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.autotrain_every = 3
    svc.add_label("A", 0.0, 1)
    assert svc.autotrain_due() is False  # below the threshold
    svc.add_label("A", 8.0, 1)
    svc.add_label("B", 0.0, 1)
    assert svc.autotrain_due() is False  # threshold hit, but only one class (👍)
    svc.add_label("B", 8.0, 0)
    assert svc.autotrain_due() is True  # now both classes present
    svc.reset_labels_since_train()
    assert svc.autotrain_due() is False  # counter cleared after a (re)train

    cfg.modeling.autotrain_every = 0
    svc._labels_since_train = 99
    assert svc.autotrain_due() is False  # 0 disables auto-training


def test_next_uncertain_skips_labeled(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    _seed_two_scenes(cfg)
    # no trained model, no centroid → random pick, but always an unlabeled frame
    item = svc.next_uncertain(pool=10)
    assert item is not None and (item["key"], round(item["time"], 2)) not in (
        svc._label_store().labeled_ids(cfg.markers.tag_name)
    )


def test_sample_frames_distinct_and_bounded(tmp_path):
    svc, cfg = _service(tmp_path)
    _seed_two_scenes(cfg)  # 4 frames total (A x2, B x2)

    got = svc.sample_frames(count=3, seed=0)
    assert len(got) == 3
    pairs = {(h["key"], h["time"]) for h in got}
    assert len(pairs) == 3  # distinct rows, no repeats
    assert all(h["scene_id"] in {"1", "2"} for h in got)

    # asking for more than the library holds is clamped to the library size
    assert len(svc.sample_frames(count=99)) == 4


def test_sample_frames_empty_without_cache(tmp_path):
    svc, _ = _service(tmp_path)  # nothing seeded
    assert svc.sample_frames(count=10) == []


def test_taste_words_from_centroid(tmp_path, monkeypatch):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    # CLIP-space frames for the same scenes (so the centroid lives in clip space)
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    cache.save("A", "clip", np.array([0.0], dtype="float32"),
               np.array([[1, 0, 0]], dtype="float32"), meta={"scene_id": "1"})
    monkeypatch.setattr(svc, "client", lambda: _MarkerTagClient())
    monkeypatch.setattr(svc, "_vocab", lambda: ["beach", "office"])
    vmap = {"beach": np.array([1, 0, 0], dtype="float32"), "office": np.array([0, 1, 0], dtype="float32")}
    monkeypatch.setattr(svc, "_clip_text_vector", lambda t: vmap[t])
    monkeypatch.setattr(svc, "_clip_text_batch", lambda labels: np.stack([vmap[t] for t in labels]))

    out = svc.taste_words(top_k=2)
    assert out["sources"] == 1 and out["labels"][0][0] == "beach"


def test_collections_save_list_load(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)
    monkeypatch.setenv("PEAKS_COLLECTIONS_DIR", str(tmp_path / "coll"))
    apexes = [{"scene_id": "7", "start": 3, "url": "u"}]
    saved = svc.save_collection("Beach Days!", apexes)
    assert saved["count"] == 1 and saved["safe"] == "Beach-Days"

    listed = svc.list_collections()
    assert listed and listed[0]["name"] == "Beach Days!" and listed[0]["count"] == 1

    loaded = svc.load_collection("Beach-Days")
    assert loaded["apexes"][0]["scene_id"] == "7"
    assert svc.load_collection("missing") is None


def test_collection_rename_and_delete(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)
    monkeypatch.setenv("PEAKS_COLLECTIONS_DIR", str(tmp_path / "coll"))
    svc.save_collection("My Faves", [{"scene_id": "1", "start": 0, "url": "u"}])
    safe = svc.list_collections()[0]["safe"]

    # rename changes the display name but keeps the file/safe stem (URLs don't break)
    r = svc.rename_collection("My Faves", "Best Of")
    assert r["name"] == "Best Of" and r["safe"] == safe
    listed = svc.list_collections()
    assert listed[0]["name"] == "Best Of" and listed[0]["safe"] == safe
    assert svc.load_collection(safe)["apexes"][0]["scene_id"] == "1"  # contents intact

    # delete removes it
    assert svc.delete_collection(safe)["removed"] is True
    assert svc.list_collections() == []
    assert svc.delete_collection(safe)["removed"] is False  # already gone


def _seed_performer_scenes(cfg, scene_ids=("1", "2", "3")):
    from peaks.cache import EmbeddingCache

    cache = EmbeddingCache(cfg.embedding.cache_dir)
    # scene 1 has an on-axis (high-taste) frame + an off one; others middling
    frames = {
        "1": [[1, 0, 0, 0], [0, 1, 0, 0]],
        "2": [[0.9, 0.1, 0, 0], [0.5, 0.5, 0, 0]],
        "3": [[0.8, 0.2, 0, 0], [0.3, 0.7, 0, 0]],
    }
    for sid in scene_ids:
        v = np.array(frames[sid], dtype="float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        cache.save(f"k{sid}", "dinov2", np.array([0.0, 5.0], dtype="float32"), v,
                   meta={"scene_id": sid, "path": f"/x/{sid}.mp4"})


class _PerfClient:
    def scene_details(self, ids):
        return {str(i): {"performers_detail": [{"id": "p1", "name": "Jane Doe"}]} for i in ids}

    def scenes_for_performer(self, pid, limit=500):
        return ["1", "2", "3", "99"]  # 99 not embedded

    def find_performers(self, name=None, limit=500):
        return [{"id": "p1", "name": "Jane Doe", "image": "/img", "scene_count": 4}]


def test_ranked_moments_orders_by_taste(tmp_path):
    svc, cfg = _service(tmp_path)
    _seed_performer_scenes(cfg)
    c = svc._unit(np.array([1, 0, 0, 0], dtype="float32"))
    hits = svc._ranked_moments_for_scenes(["1", "2", "3"], c, per_scene=1)
    assert [h.scene_id for h in hits] == ["1", "2", "3"]  # scene 1's best is closest
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True) and scores[0] > 0.99


def test_performer_best_ranks_embedded_only(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    _seed_performer_scenes(cfg)
    monkeypatch.setattr(svc, "client", lambda: _PerfClient())
    # taste centroid ~ [1,0,0,0] via a thumbs-up on scene 1's on-axis frame
    svc.add_label("k1", 0.0, 1, scene_id="1")

    r = svc.performer_best(name="Jane", per_scene=2, count=50)
    assert r["performer"] == "Jane Doe"
    sids = {h.scene_id for h in r["hits"]}
    assert sids and sids <= {"1", "2", "3"} and "99" not in sids   # embedded only
    assert [h.score for h in r["hits"]] == sorted((h.score for h in r["hits"]), reverse=True)


def test_performer_best_spread_and_floor(tmp_path, monkeypatch):
    """The megaboard performer board: floor 0 → a diverse span of her scenes;
    floor > 0 → only her moments matching your taste (tightening toward 100%)."""
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    _seed_performer_scenes(cfg)
    monkeypatch.setattr(svc, "client", lambda: _PerfClient())
    svc.add_label("k1", 0.0, 1, scene_id="1")   # taste centroid ~ [1,0,0,0]

    # spread → diverse coverage: every embedded scene present, no taste ranking
    sp = svc.performer_best(name="Jane", spread=True, per_scene=2, count=50)
    assert {h.scene_id for h in sp["hits"]} == {"1", "2", "3"}
    assert all(h.score == 0.0 for h in sp["hits"])   # coverage, not taste-scored

    # floor > 0 → only her taste-matching moments survive, still score-sorted
    hi = svc.performer_best(name="Jane", min_score=0.99, per_scene=2, count=50)
    assert hi["hits"] and all(h.score >= 0.99 for h in hi["hits"])
    assert [h.score for h in hi["hits"]] == sorted((h.score for h in hi["hits"]), reverse=True)
    # tightening drops the middling scenes the 0% spread happily included
    assert "1" in {h.scene_id for h in hi["hits"]}
    assert len({h.scene_id for h in hi["hits"]}) < 3


def test_performer_moment_matches_ranks_across_her_scenes(tmp_path, monkeypatch):
    """'More of this moment (same actress)': her embedded scenes ranked by
    similarity to a given moment — embedded-only, best match first."""
    svc, cfg = _service(tmp_path)
    _seed_performer_scenes(cfg)
    monkeypatch.setattr(svc, "client", lambda: _PerfClient())

    r = svc.performer_moment_matches("1", 0.0, per_scene=2)   # scene 1 @ 0 ~ [1,0,0,0]
    assert r["performer"] == "Jane Doe"
    sids = [h.scene_id for h in r["hits"]]
    assert sids and set(sids) <= {"1", "2", "3"} and "99" not in sids  # embedded only
    assert sids[0] == "1"                                              # its own frame is closest
    assert [h.score for h in r["hits"]] == sorted((h.score for h in r["hits"]), reverse=True)


def test_scene_moments_stays_in_one_scene(tmp_path):
    """'More moments in this scene': a spread drawn only from the given scene."""
    svc, cfg = _service(tmp_path)
    _seed_performer_scenes(cfg)
    r = svc.scene_moments("2", per_scene=10)
    assert r["hits"] and {h.scene_id for h in r["hits"]} == {"2"}   # only that scene


def test_performer_stats_leaderboard(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    _seed_performer_scenes(cfg)
    monkeypatch.setattr(svc, "client", lambda: _PerfClient())
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")
    rows = svc.performer_stats(rebuild=True)
    assert len(rows) == 1
    jane = rows[0]
    assert jane["name"] == "Jane Doe" and jane["scenes"] == 3 and jane["moments"] == 6
    # extended fields + centroid + top moments
    assert "affinity" in jane and "o_counter" in jane and jane["top"]
    assert jane["top"][0]["thumb"].startswith("/api/frame?key=")
    assert svc._perf_centroids and "p1" in svc._perf_centroids


class _PerfClient2:
    """Two performers: p1 (scenes 1,2) and p2 (scene 3), with engagement data."""
    def scene_details(self, ids):
        who = {"1": [{"id": "p1", "name": "Ava"}], "2": [{"id": "p1", "name": "Ava"}],
               "3": [{"id": "p2", "name": "Mia"}]}
        return {str(i): {"performers_detail": who.get(str(i), []),
                         "o_counter": 5 if str(i) == "1" else 0, "rating100": 80} for i in ids}

    def scenes_for_performer(self, pid, limit=500):
        return {"p1": ["1", "2"], "p2": ["3"]}.get(pid, [])

    def find_performers(self, name=None, limit=500):
        return [{"id": "p1", "name": "Ava"}, {"id": "p2", "name": "Mia"}]


def test_performer_detail_and_similar(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    _seed_performer_scenes(cfg)
    monkeypatch.setattr(svc, "client", lambda: _PerfClient2())
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")
    svc.add_label("k1", 0.0, 1, scene_id="1")   # taste ~ scene 1's on-axis frame

    d = svc.performer_detail(performer_id="p1")
    assert d["performer"] == "Ava" and d["hits"]
    assert d["stats"]["o_counter"] == 5 and d["stats"]["scenes"] == 2
    assert d["distribution"] and len(d["distribution"]["counts"]) == 20
    sim = svc.similar_performers("p1")
    assert sim and sim[0]["id"] == "p2" and -1.0 <= sim[0]["score"] <= 1.0


def test_performer_stats_excludes_upscale(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    _seed_performer_scenes(cfg)

    class _C:
        def scene_details(self, ids):
            return {str(i): {"performers_detail": [
                {"id": "p1", "name": "Jane Doe"}, {"id": "upx", "name": "Upscale"}]} for i in ids}
        def scenes_for_performer(self, pid, limit=500):
            return ["1", "2", "3"]
        def find_performers(self, name=None, limit=500):
            return [{"id": "upx", "name": "Upscale"}]

    monkeypatch.setattr(svc, "client", lambda: _C())
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")
    names = {r["name"] for r in svc.performer_stats(rebuild=True)}
    assert "Jane Doe" in names and "Upscale" not in names  # excluded from the leaderboard


def test_export_collection_writes_mp4(tmp_path, monkeypatch):
    import subprocess
    svc, cfg = _service(tmp_path)
    monkeypatch.setenv("PEAKS_COLLECTIONS_DIR", str(tmp_path / "coll"))
    monkeypatch.setenv("PEAKS_EXPORT_DIR", str(tmp_path / "exp"))
    svc.save_collection("My Reel", [
        {"scene_id": "1", "start": 0, "duration": 20, "url": "u", "score": 0.9},
        {"scene_id": "2", "start": 5, "end": 25, "url": "u", "score": 0.5},
    ])

    class _C:
        def scene_details(self, ids):
            return {str(i): {"path": str(tmp_path / f"{i}.mp4")} for i in ids}
    for i in ("1", "2"):
        (tmp_path / f"{i}.mp4").write_bytes(b"x")   # files must "exist"
    monkeypatch.setattr(svc, "client", lambda: _C())

    calls = []
    def fake_run(cmd, **kw):
        calls.append(cmd)
        # emulate ffmpeg writing each segment / final output
        out = cmd[-1]
        if out.endswith(".ts") or out.endswith(".mp4"):
            open(out, "wb").write(b"video")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")
    monkeypatch.setattr(subprocess, "run", fake_run)

    r = svc.export_collection(name="My Reel", limit=200)
    assert r["clips"] == 2 and r["name"] == "My-Reel.mp4"
    assert (tmp_path / "exp" / "My-Reel.mp4").exists()
    assert svc.reel_path("My-Reel") and any("concat" in " ".join(c) for c in calls)


def test_hall_of_fame_and_roulette(tmp_path, monkeypatch):
    svc, cfg = _service(tmp_path)
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    cfg.modeling.dir = str(tmp_path / "models")
    import os
    os.environ["PEAKS_COLLECTIONS_DIR"] = str(tmp_path / "coll")
    _seed_performer_scenes(cfg)
    monkeypatch.setattr(svc, "client", lambda: _PerfClient2())
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"u/{sid}")
    svc.add_label("k1", 0.0, 1, scene_id="1")

    hof = svc.hall_of_fame(top_n=2)
    assert len(hof["created"]) >= 1
    names = {c["name"] for c in svc.list_collections()}
    assert any("best of" in n for n in names)
    r = svc.performer_roulette(min_moments=1)
    assert r["id"] in {"p1", "p2"}


def test_find_duplicates_threshold(tmp_path):
    from peaks.cache import EmbeddingCache

    svc, cfg = _service(tmp_path)
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    # query scene A; B is near-identical (same vec), C is different
    cache.save("A", "dinov2", np.array([0.0], dtype="float32"),
               np.array([[1, 0, 0]], dtype="float32"), meta={"scene_id": "1"})
    cache.save("B", "dinov2", np.array([0.0], dtype="float32"),
               np.array([[1, 0, 0]], dtype="float32"), meta={"scene_id": "2"})
    cache.save("C", "dinov2", np.array([0.0], dtype="float32"),
               np.array([[0, 1, 0]], dtype="float32"), meta={"scene_id": "3"})

    dupes = svc.find_duplicates("A", 0.0, threshold=0.9)
    keys = {h.key for h in dupes}
    assert "B" in keys and "C" not in keys and "A" not in keys  # only the near-identical other scene


def test_scene_pool_lists_scenes_with_urls(tmp_path, monkeypatch):
    from peaks.models import Scene

    svc, _ = _service(tmp_path)

    class C:
        def iter_scenes(self, path_prefix=""):
            for i in ("1", "2"):
                yield Scene.from_dict({"id": i, "title": "", "files": [{"path": f"/data/{i}.mp4", "duration": 100.0}], "scene_markers": []})

        def stream_url(self, sid, start=None):
            return f"http://stash/scene/{sid}/stream?start={start}"

    monkeypatch.setattr(svc, "client", lambda: C())
    pool = svc.scene_pool()
    assert len(pool) == 2
    assert pool[0]["scene_id"] == "1" and pool[0]["duration"] == 100.0
    assert "start=0" in pool[0]["url"]
    # cached: a second call doesn't rebuild (client swapped to a raiser)
    monkeypatch.setattr(svc, "client", lambda: (_ for _ in ()).throw(AssertionError("rebuilt")))
    assert svc.scene_pool() is pool


def test_board_sources(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)
    monkeypatch.setenv("PEAKS_COLLECTIONS_DIR", str(tmp_path / "coll"))
    svc.save_collection("Faves", [{"scene_id": "7"}])
    s = svc.board_sources()
    assert s["tag"] == "apex"
    assert any(c["name"] == "Faves" for c in s["collections"])


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