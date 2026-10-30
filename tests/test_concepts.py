"""Concepts: Peaks learns your words — teach, probe, scores, search, the Arbiter."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402

pytest.importorskip("sklearn")


def _frames(svc, tier_o):
    """(key, t, scene_id) of every frame in scenes graded with O-count `tier_o`."""
    idx = svc.index(svc._model_name())
    rows = {r["scene_id"]: r for r in svc._catalogue_all()}
    out = []
    for i in range(idx.size):
        sid = idx.scene_ids[i]
        r = rows.get(str(sid))
        if r and r.get("o_counter") == tier_o and r.get("rating100") == 100:
            out.append((idx.keys[i], float(idx.times[i]), str(sid)))
    return out


def _teach(svc, name, yes, no):
    for k, t, sid in yes:
        svc.concept_label(name, k, t, 1, scene_id=sid)
    for k, t, sid in no:
        svc.concept_label(name, k, t, 0, scene_id=sid)


def test_a_concept_is_learned_from_a_few_marks(tmp_path, monkeypatch):
    svc, _ = _triage_svc(tmp_path, monkeypatch)
    c = svc.concept_create("Golden look")
    assert c["slug"] == "golden-look" and not c["trained"]
    assert len(svc.concept_candidates("golden look")) == 12          # seeds, one per scene
    leg, exc = _frames(svc, 18), _frames(svc, 17)
    _teach(svc, "Golden look", leg[::10][:6], exc[::10][:6])
    st = next(x for x in svc.concepts() if x["slug"] == "golden-look")
    assert st["trained"] and st["yes"] == 6 and st["no"] == 6 and st["accuracy"] >= 0.8
    scores = svc.concept_scene_scores("Golden look")
    rows = {r["scene_id"]: r for r in svc._catalogue_all()}
    leg_s = [v for s, v in scores.items() if rows[s]["o_counter"] == 18]
    exc_s = [v for s, v in scores.items() if rows[s]["o_counter"] == 17]
    assert min(leg_s) > max(exc_s)                                   # it separates what you taught
    hits = svc.concept_search("golden-look", top_k=5, per_scene=1)
    assert all(rows[h.scene_id]["o_counter"] == 18 for h in hits) and 0.5 < hits[0].score <= 1
    # after training, teaching asks about what it's least sure of — never a frame already marked
    marked = {(k, round(t, 2)) for k, t, _ in leg[::10][:6] + exc[::10][:6]}
    assert not marked & {(c["key"], c["t"]) for c in svc.concept_candidates("golden-look")}


def test_too_few_marks_wait_and_delete_cleans_up(tmp_path, monkeypatch):
    svc, _ = _triage_svc(tmp_path, monkeypatch)
    svc.concept_create("slow build")
    k, t, sid = _frames(svc, 18)[0]
    out = svc.concept_label("slow build", k, t, 1, scene_id=sid)
    assert out["yes"] == 1 and "report" not in out
    with pytest.raises(LookupError):
        svc.concept_search("slow build")
    svc.concept_delete("slow build")
    assert svc.concepts() == []
    with pytest.raises(ValueError):
        svc.concept_create("  !!  ")


def test_concepts_argue_filter_sort_and_search(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, _ = _triage_svc(tmp_path, monkeypatch)
    svc.concept_create("Golden look")
    _teach(svc, "Golden look", _frames(svc, 18)[::10][:6], _frames(svc, 17)[::10][:4] + _frames(svc, 16)[::10][:4])
    assert svc.train_tier_model()["trained"]

    # the Arbiter speaks your concept first — it really separates Légendaire from Exceptionnelle
    words = svc.arbiter_summary()["words"]
    assert words[0] == {"term": "Golden look", "d": words[0]["d"], "kind": "concept",
                        "favours": "Légendaire"} and words[0]["d"] >= 0.3
    leg_sid = next(r["scene_id"] for r in svc._catalogue_all() if r["o_counter"] == 18)
    e = svc.tier_explain(leg_sid)
    side = e["words"].get("legendaire") or []
    assert side and side[0]["kind"] == "concept" and side[0]["term"] == "Golden look"

    # the Catalogue filters and sorts by it
    scores = svc.concept_scene_scores("golden-look")
    d = svc.catalogue(concept="golden-look", concept_min=0.5, limit=100)
    assert d["items"] and all(r["concept"] >= 0.5 for r in d["items"])
    assert d["total"] == sum(1 for sid in scores if scores[sid] >= 0.5 and sid in svc.client().s)
    srt = svc.catalogue(concept="Golden look", sort="concept", limit=100)["items"]
    vals = [r["concept"] for r in srt]
    assert vals == sorted(vals, reverse=True)
    golden = {sid for sid, v in svc.client().s.items() if v["o_counter"] == 18} | {"u1", "a1"}   # u1/a1 look it
    assert {r["scene_id"] for r in srt[:12]} <= golden                # the golden-looking ones come first

    # search by its name uses the concept (no CLIP index here at all)
    c = TestClient(app_mod.create_app(svc.cfg))
    r = c.get("/api/search/text", params={"q": "golden look", "per_scene": 1, "top_k": 5})
    assert r.status_code == 200 and r.json()["concept"] == "golden look" and r.json()["items"]
    assert c.get("/api/search/text", params={"q": "something else"}).status_code == 400   # still needs CLIP
    assert [x["name"] for x in c.get("/api/concepts").json()] == ["Golden look"]


def test_a_concept_is_a_megaboard_channel(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    svc, _ = _triage_svc(tmp_path, monkeypatch)
    svc.concept_create("Golden look")
    svc.concept_create("Untaught")
    leg, exc = _frames(svc, 18), _frames(svc, 17)
    _teach(svc, "Golden look", leg[::10][:6], exc[::10][:6])
    c = TestClient(app_mod.create_app(svc.cfg))
    src = c.get("/api/board/sources").json()
    assert {"key": "golden-look", "label": "Golden look", "kind": "visual"} in src["concepts"]
    assert all(x["key"] != "untaught" for x in src["concepts"])            # only trained ones are channels
    d = c.get("/api/board/concept", params={"key": "golden-look", "per_scene": 2}).json()
    rows = {r["scene_id"]: r for r in svc._catalogue_all()}
    assert d["concept"] == "Golden look" and d["total"] > 12 and all(x["stream"] for x in d["items"])
    per = {}
    for x in d["items"]:
        per[x["scene_id"]] = per.get(x["scene_id"], 0) + 1
    assert max(per.values()) <= 2                                          # spread across scenes
    top = [x for x in d["items"][:6]]
    assert all(rows[x["scene_id"]]["o_counter"] == 18 for x in top)        # strongest first
    strong = c.get("/api/board/concept", params={"key": "golden-look", "min_score": 0.9}).json()
    assert all(x["score"] >= 0.9 for x in strong["items"]) and strong["total"] < d["total"] * 2
    assert c.get("/api/board/concept", params={"key": "untaught"}).status_code == 404
    assert c.get("/api/board/concept", params={"key": "nope"}).status_code == 404
