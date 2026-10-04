"""Answered suggestions stay answered: a grade (or 'Keep as is') takes a scene
out of the promote / second / saved / trim / passed lists and suggestions,
until there's new evidence — retraining alone never brings it back."""

import pytest

pytest.importorskip("fastapi")

from fakestash import FakeStash  # noqa: E402
from peaks.config import Config  # noqa: E402


@pytest.fixture
def stash():
    return FakeStash({
        "1": {"rating100": 100, "o_counter": 16},     # Merveilleuse — a promotion candidate
        "2": {"rating100": 100, "o_counter": 17},     # Exceptionnelle — a second-look candidate
        "3": {"rating100": 100, "o_counter": 0},      # Upscale with saves — saved → Légendaire
        "4": {"rating100": 100, "o_counter": 16},     # passed over on the megaboard — trim / Reject
    })


@pytest.fixture
def svc(tmp_path, stash, monkeypatch):
    import peaks.web.service as svc_mod

    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    s = svc_mod.Service(cfg)
    s._best_q25, s._best_q50 = 0.5, 0.6
    return s


def _pred(tier):
    probs = {t: 0.0 for t in ("reject", "upscale", "merveilleuse", "exceptionnelle", "legendaire")}
    probs[tier] = 0.9
    return {"tier": tier, "probs": probs, "expected": 2.0, "keeper": 0.3, "conf": 0.9}


def _state(svc, guess1="exceptionnelle", guess2="merveilleuse", saves3=2, shows4=10, passed4=True):
    rows = [svc._cat_update_row(s) for s in ("1", "2", "3", "4")]
    preds = {"1": _pred(guess1), "2": _pred(guess2), "3": _pred("upscale"), "4": _pred("merveilleuse")}
    sig = {"1": {"saves": 0}, "2": {"saves": 0}, "3": {"saves": saves3},
           "4": {"saves": 0, "best": 0.2, "passed": passed4, "showings": shows4, "shown_days": 4}}
    return rows, preds, sig


def _ids(svc, view, st):
    rows, preds, sig = st
    return [r["scene_id"] for r in svc._triage(view, rows, preds, {}, sig)]


def test_unanswered_scenes_are_in_their_lists(svc):
    st = _state(svc)
    assert _ids(svc, "promote", st) == ["1"] and _ids(svc, "second", st) == ["2"]
    assert _ids(svc, "saved", st) == ["3"] and _ids(svc, "passed", st) == ["4"]


def test_regrading_at_the_same_tier_answers_promotion_until_the_guess_moves_up(svc, stash):
    svc.grade_scene("1", "merveilleuse")                      # "no, it stays Merveilleuse"
    assert _ids(svc, "promote", _state(svc)) == []
    assert _ids(svc, "promote", _state(svc)) == []            # retrained, same guess: still answered
    assert _ids(svc, "promote", _state(svc, guess1="legendaire")) == ["1"]   # a further tier up


def test_second_look_comes_back_only_if_the_guess_drops_further(svc):
    svc.keep_as_is("2")
    assert _ids(svc, "second", _state(svc)) == []
    assert _ids(svc, "second", _state(svc, guess2="upscale")) == ["2"]


def test_saved_comes_back_only_after_a_new_save(svc):
    svc.keep_as_is("3")
    assert _ids(svc, "saved", _state(svc)) == []
    rows, preds, sig = _state(svc)
    assert svc._recommend(rows[2], preds["3"], sig["3"]) is None              # no Légendaire nudge either
    assert _ids(svc, "saved", _state(svc, saves3=3)) == ["3"]


def test_keep_as_is_writes_nothing_to_stash_and_answers_trim(svc, stash):
    rows, preds, sig = _state(svc)
    assert svc._recommend(rows[3], preds["4"], sig["4"])["grade"] == "reject"
    before = len([c for c in stash.calls if c[0] in ("update", "o")])
    svc.keep_as_is("4")
    assert len([c for c in stash.calls if c[0] in ("update", "o")]) == before
    st = _state(svc, shows4=0, passed4=False)                 # the megaboard record starts over
    assert _ids(svc, "passed", st) == [] and _ids(svc, "trim", st) == []
    assert svc._recommend(st[0][3], st[1]["4"], st[2]["4"]) is None
    st = _state(svc, shows4=9, passed4=True)                  # 8+ more showings, still no reaction
    assert _ids(svc, "passed", st) == ["4"]
    assert svc._recommend(st[0][3], st[1]["4"], st[2]["4"])["grade"] == "reject"


def test_undo_brings_it_back(svc, stash):
    r = svc.grade_scene("1", "merveilleuse")
    assert _ids(svc, "promote", _state(svc)) == []
    p = r["previous"]
    svc.restore_scene_grade("1", p["rating100"], p["o_counter"], tag_ids=p["tag_ids"], organized=p["organized"])
    assert _ids(svc, "promote", _state(svc)) == ["1"]
    svc.keep_as_is("2")
    svc.undo_keep("2")
    assert _ids(svc, "second", _state(svc)) == ["2"]


def test_a_tier_changed_outside_peaks_voids_the_verdict(svc, stash):
    svc.keep_as_is("1")
    _ids(svc, "promote", _state(svc))                         # evidence recorded
    stash.s["2"]["o_counter"] = 18                            # (unrelated)
    stash.s["1"]["o_counter"] = 17                            # changed in Stash: now Exceptionnelle
    svc.invalidate_meta("1")
    st = _state(svc, guess1="merveilleuse")
    assert svc._answered("second", st[0][0], st[1]["1"], st[2]["1"]) is False


def test_automatic_changes_are_not_verdicts(svc):
    svc.grade_scene("1", "legendaire", source="auto: saved a moment")
    svc.grade_scene("2", "legendaire", source="duplicate")
    assert svc.verdict_status()["answered"] == 0


def test_clear_all_and_restart(svc):
    import peaks.web.service as svc_mod

    svc.keep_as_is("1")
    svc.keep_as_is("2")
    fresh = svc_mod.Service(svc.cfg)
    assert fresh.verdict_status()["answered"] == 2
    assert fresh.clear_verdicts()["answered"] == 0


def test_api_keep_and_clear(svc, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(app_mod.create_app(svc.cfg))
    assert api.post("/api/catalogue/keep", params={"scene_id": "1"}).json()["scene"]["tier"] == "merveilleuse"
    assert api.get("/api/verdicts").json()["answered"] == 1
    assert api.post("/api/catalogue/keep", params={"scene_id": "1", "undo": True}).status_code == 200
    assert api.get("/api/verdicts").json()["answered"] == 0
    assert api.post("/api/catalogue/keep", params={"scene_id": "999"}).status_code == 404
    api.post("/api/catalogue/keep", params={"scene_id": "2"})
    assert api.post("/api/verdicts/clear").json()["answered"] == 0
