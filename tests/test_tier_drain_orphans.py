"""Retiring Merveilleuse (drain, then retire) and orphans (no performer)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_catalogue import _triage_svc  # noqa: E402

from peaks import tiers  # noqa: E402
from peaks.tier_model import live_ordinal, train_class  # noqa: E402


# --- the scheme ------------------------------------------------------------------------

def test_retired_tier_reads_as_anomaly_and_cannot_be_graded():
    assert tiers.tier_of(100, 16) == "merveilleuse"
    tiers.set_retired(["merveilleuse"])
    assert tiers.tier_of(100, 16) == "anomaly"
    assert tiers.tier_of(100, 17) == "exceptionnelle"
    assert "merveilleuse" not in tiers.grades() and "reject" in tiers.grades()
    assert tiers.keeper_tiers() == ("exceptionnelle", "legendaire")
    assert "merveilleuse" not in tiers.tier_tags()
    st = tiers.tag_state("anomaly", ["merveilleuse"], True)
    assert st["conflict"] and "retired" in st["conflict"]          # a straggler surfaces


def test_draining_tier_is_not_training_data():
    tiers.set_retired((), draining=["merveilleuse"])
    assert train_class("merveilleuse") is None
    assert train_class("exceptionnelle") == "exceptionnelle" and train_class("rejected") == "reject"
    assert tiers.learned_tiers() == ("exceptionnelle", "legendaire")
    # no gap where the middle tier was: reject and Exceptionnelle are one step apart
    assert live_ordinal(["reject", "exceptionnelle", "legendaire"]) == {
        "reject": 0, "exceptionnelle": 1, "legendaire": 2}


# --- drain, then retire ------------------------------------------------------------------

def test_drain_view_suggests_live_tiers_then_retires(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch, settings={"draining_tiers": ["merveilleuse"]})
    rep = svc.train_tier_model()
    assert rep["trained"] and set(rep["classes"]) == {"exceptionnelle", "legendaire"}   # Merveilleuse left out
    assert rep["counts"]["merveilleuse"] == 0

    d = svc.catalogue(view="promote", limit=100)
    merv = {sid for sid, v in stash.s.items() if v["rating100"] == 100 and v["o_counter"] == 16}
    assert {r["scene_id"] for r in d["items"]} == merv                  # every scene left in it
    confs = [r["pred"]["conf"] for r in d["items"]]
    assert confs == sorted(confs, reverse=True)                         # surest first
    sug = d["items"][0]["suggest"]
    assert sug["drain"] and sug["grade"] in ("exceptionnelle", "legendaire")
    assert "retiring Merveilleuse" in sug["why"]
    assert d["tier_state"]["draining"] == ["merveilleuse"]
    assert d["drain"] is not None
    assert all(set(ids) <= merv for ids in d["drain"].values())

    # predictions only ever name live classes
    assert all(r["pred"]["tier"] != "merveilleuse" for r in d["items"])

    # re-grade the last of them (here: straight in Stash) → it retires on its own
    for sid in merv:
        stash.s[sid]["o_counter"] = 17
    svc.catalogue(refresh=True)
    st = svc.tier_state()
    assert st["retired"] == ["merveilleuse"] and st["draining"] == []
    assert st["retired_note"]["tiers"] == ["merveilleuse"]
    saved = json.loads(open(os.environ["PEAKS_SETTINGS"]).read())
    assert saved["retired_tiers"] == ["merveilleuse"]
    with pytest.raises(ValueError, match="retired"):
        svc.grade_scene("0", "merveilleuse")
    stash.s["0"]["o_counter"] = 16                       # a straggler
    rows = {r["scene_id"]: r for r in svc.catalogue(refresh=True, limit=200)["items"]}
    assert rows["0"]["tier"] == "anomaly"


def test_no_retirement_without_a_real_listing(tmp_path, monkeypatch):
    svc, _ = _triage_svc(tmp_path, monkeypatch, settings={"draining_tiers": ["merveilleuse"]})
    svc._check_drained([])                              # Stash down / empty: never retire
    assert svc.tier_state()["retired"] == []


# --- orphans -------------------------------------------------------------------------------

def _orphan_svc(tmp_path, monkeypatch):
    svc, stash = _triage_svc(tmp_path, monkeypatch)
    stash.performers = [{"id": "1", "name": "Ava"}, {"id": "2", "name": "Bea"}, {"id": "3", "name": "Cleo"},
                        {"id": "9", "name": "upscale"}]
    who = {17: ("1", "Ava"), 18: ("2", "Bea"), 16: ("3", "Cleo")}
    for sid, v in stash.s.items():
        if v["rating100"] == 100 and v["o_counter"] in who:
            pid, name = who[v["o_counter"]]
            v["performers_detail"] = [{"id": pid, "name": name}]
            v["performers"] = [name]
        else:
            v["performers_detail"], v["performers"] = [], []
    stash.s["u2"]["performers_detail"] = [{"id": "9", "name": "upscale"}]   # junk only: still an orphan
    stash.s["u2"]["performers"] = ["upscale"]
    stash.s["u3"]["rating100"] = 20                     # a reject: not an orphan
    return svc, stash


def test_orphans_view_and_suggestions(tmp_path, monkeypatch):
    svc, stash = _orphan_svc(tmp_path, monkeypatch)
    d = svc.catalogue(view="orphans", limit=50, refresh=True)
    ids = [r["scene_id"] for r in d["items"]]
    assert set(ids) == {"u1", "u2", "a1"}               # no performer, or only junk; rejects left out
    assert ids[0] == "a1"                               # keepers / graded first
    assert d["views"]["orphans"] == 3

    s1 = svc.orphan_suggestions("u1")["suggestions"]    # looks like the Légendaire scenes → Bea
    assert s1[0]["name"] == "Bea" and s1[0]["via"]["scene_id"] in stash.s
    s2 = svc.orphan_suggestions("u2")["suggestions"]
    assert s2[0]["name"] == "Ava"
    assert all(x["name"] != "upscale" for x in s1 + s2)

    svc.orphan_not_them("u1", "2")                      # remembered
    assert all(x["id"] != "2" for x in svc.orphan_suggestions("u1")["suggestions"])


def test_add_performer_merges_and_undoes(tmp_path, monkeypatch):
    svc, stash = _orphan_svc(tmp_path, monkeypatch)
    stash.s["u2"]["performers_detail"] = [{"id": "9", "name": "upscale"}]
    r = svc.add_performer("u2", "1")
    assert r["previous"] == ["9"]
    assert [p["id"] for p in stash.s["u2"]["performers_detail"]] == ["9", "1"]      # merged, not replaced
    assert ("update", "u2", {"performer_ids": ["9", "1"]}) in stash.calls
    assert r["scene"]["performers"] == ["upscale", "Ava"]
    assert "u2" not in {x["scene_id"] for x in svc.catalogue(view="orphans", limit=50)["items"]}
    svc.add_performer("u2", "1")                         # adding twice changes nothing
    assert [p["id"] for p in stash.s["u2"]["performers_detail"]] == ["9", "1"]
    svc.set_scene_performers("u2", r["previous"])        # the undo
    assert [p["id"] for p in stash.s["u2"]["performers_detail"]] == ["9"]
    log = svc.history(10)
    assert any(e["action"] == "performers" and e["after"] == ["9", "1"] for e in log)
