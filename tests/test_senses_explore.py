"""The Lab's Sound and Motion explorers: the clip bank, decks, "more of this",
named sound concepts, and what the moment model gets — planted clusters, no torch."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402

pytest.importorskip("sklearn")

D = 16


def _heard(tmp_path, monkeypatch, n_blocks=50):
    """Every embedded scene "heard": scene i's sound sits in cluster i % 3 (A, B, C)."""
    from peaks.cache import EmbeddingCache
    from peaks.web.hear import CLAP_NS, ENV_NS

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    cache = EmbeddingCache(svc.cfg.embedding.cache_dir)
    rng = np.random.default_rng(1)
    centres = np.eye(D, dtype=np.float32)[:3] * 3
    cluster = {}
    for i, key in enumerate(sorted(cache.keys("dinov2"))):
        sid = cache.peek_meta(key, "dinov2")["scene_id"]
        c = i % 3
        cluster[sid] = c
        t = np.arange(n_blocks, dtype=np.float32) * 5 + 5
        v = centres[c] + 0.6 * rng.standard_normal((n_blocks, D)).astype(np.float32)
        cache.save(key, CLAP_NS, t, v, meta={"scene_id": sid})
        cache.save(key, ENV_NS, t, np.zeros((n_blocks, 4), np.float32), meta={"scene_id": sid})
    return svc, stash, cluster


def test_the_bank_is_capped_per_scene_and_leaves_rejects_out(tmp_path, monkeypatch):
    svc, stash, cluster = _heard(tmp_path, monkeypatch)
    stash.s["0"]["rating100"] = 20
    svc._cat_cache = None
    V, S, T = svc._se_bank("sound")
    assert V.dtype == np.float16 and "0" not in set(S)
    per = {s: int((S == s).sum()) for s in set(S)}
    assert max(per.values()) == 30                                     # 50 blocks → 30 kept, spread out
    t1 = T[S == "1"]
    assert t1.min() == 5 and t1.max() == 250


def test_discover_spreads_out_and_more_follows_your_marks(tmp_path, monkeypatch):
    svc, stash, cluster = _heard(tmp_path, monkeypatch)
    d = svc.sense_deck("sound", "discover", n=12)
    sids = [x["scene_id"] for x in d["items"]]
    assert len(set(sids)) == len(sids) == 12                           # one clip per scene
    assert {cluster[s] for s in sids} == {0, 1, 2}                     # every corner of the sound space
    it = d["items"][0]
    assert it["end"] - it["start"] == 10 and it["stream"] and it["why"]
    assert svc.sense_deck("sound", "more", n=6)["note"]                # no taste yet: says so, discovers
    a = [s for s, c in cluster.items() if c == 0]
    b = [s for s, c in cluster.items() if c == 1]
    for s in a[:4]:
        svc.sense_mark("sound", s, 50.0, 1)
    for s in b[:2]:
        svc.sense_mark("sound", s, 50.0, -1)
    assert svc.sense_taste("sound")["trained"]
    more = svc.sense_deck("sound", "more", n=12)["items"]
    liked = [x for x in more if cluster[x["scene_id"]] == 0]
    assert len(liked) >= 8                                             # mostly what you liked…
    assert len(more) - len(liked) >= 2                                 # …with discovery mixed in
    per = {}
    for x in more:
        per[x["scene_id"]] = per.get(x["scene_id"], 0) + 1
    assert max(per.values()) <= 2
    assert all(not (x["scene_id"] == a[0] and abs(x["t"] - 50) < 2.5) for x in more)   # never a marked clip
    svc.sense_mark("sound", a[0], 50.0, 0)                             # undo
    assert svc.sense_taste("sound")["likes"] == 3


def test_a_named_sound_concept_searches_filters_and_teaches(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, stash, cluster = _heard(tmp_path, monkeypatch)
    a = [s for s, c in cluster.items() if c == 0]
    for s in a[:4]:
        svc.sense_mark("sound", s, 50.0, 1)
    for s in [s for s, c in cluster.items() if c == 2][:3]:
        svc.sense_mark("sound", s, 50.0, -1)
    c = svc.sense_concept_save("sound", "Breathless")
    assert c["slug"] == "sound:breathless" and c["trained"] and c["kind"] == "sound"
    assert any(x["slug"] == "sound:breathless" for x in svc.concepts())
    assert all(x["slug"] != "sound:breathless" for x in svc._visual_concepts())
    scores = svc.concept_scene_scores("sound:breathless")
    top = sorted(scores, key=lambda s: -scores[s])[:5]
    assert all(cluster[s] == 0 for s in top)
    teach = svc.sense_deck("sound", "teach", n=9, concept="sound:breathless")["items"]
    assert len(teach) == 9
    app = TestClient(app_mod.create_app(svc.cfg))
    r = app.get("/api/search/text", params={"q": "sound: breathless"}).json()
    assert r["concept"] == "breathless" and cluster[r["items"][0]["scene_id"]] == 0
    cat = app.get("/api/catalogue", params={"concept": "sound:breathless", "sort": "concept", "limit": 5}).json()
    assert cat["items"] and all(cluster.get(x["scene_id"]) == 0 for x in cat["items"][:3])
    assert app.get("/api/lab/sound/taste").json()["concepts"][0]["name"] == "Breathless"
    assert app.post("/api/lab/sound/mark", json={"scene_id": a[5], "t": 20, "label": 1}).json()["likes"] == 5
    assert app.get("/api/lab/sound/deck", params={"mode": "discover", "n": 4}).json()["items"]
    assert app.get("/api/search/text", params={"q": "motion: nothing"}).status_code == 404
    assert app.delete("/api/concepts/sound:breathless").json()["deleted"] == "sound:breathless"


def test_your_lab_taste_is_a_moment_feature(tmp_path, monkeypatch):
    svc, stash, cluster = _heard(tmp_path, monkeypatch)
    a = [s for s, c in cluster.items() if c == 0]
    for s in a[:4]:
        svc.sense_mark("sound", s, 50.0, 1)
    for s in [s for s, c in cluster.items() if c == 1][:3]:
        svc.sense_mark("sound", s, 50.0, -1)
    svc.sense_concept_save("sound", "breathless")
    f_a = svc.sense_window_likes("sound", a[6], 100.0, 20.0)
    f_b = svc.sense_window_likes("sound", [s for s, c in cluster.items() if c == 1][6], 100.0, 20.0)
    assert f_a["sound_like"] > 0.5 and f_a["sound_like"] > f_b["sound_like"] + 0.3
    assert "sound:breathless" in f_a
    groups = svc._zoom_feature_groups()
    assert "sound_like" in groups["sound"] and "sound:breathless" in groups["sound"]
    assert "motion_like" in groups["motion"]


def test_motion_explore_windows_are_fixed_and_spread(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    e = {"centers": [30.0, 300.0]}
    one = svc._motion_centers("42", e, {"duration": 1800})
    assert one[:2] == [30.0, 300.0] and len(one) == 4
    assert one == svc._motion_centers("42", e, {"duration": 1800})          # the same every time
    assert all(abs(x - y) >= 40 for i, x in enumerate(one) for y in one[i + 1:])
    assert svc._motion_centers("42", e, {"duration": 60}) == [30.0, 300.0]  # too short: picks only
    svc.save_motion_settings(on=True)
    s = dict(svc._settings())
    s["motion"]["explore_windows"] = 0
    svc._save_settings_dict(s)
    assert svc._motion_centers("42", e, {"duration": 1800}) == [30.0, 300.0]
