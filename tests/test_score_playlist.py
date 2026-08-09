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