"""Teach Peaks: the ungraded scenes that would teach the grade models most —
torn guesses, thin records, spread out — and, for VR before its model, the
gaps it still has. Plus the HereSphere tag."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

pytest.importorskip("fastapi")

from peaks import teach  # noqa: E402
from test_ingest import stash, svc  # noqa: E402,F401

NAMES = {"exceptionnelle": "Exceptionnelle", "legendaire": "Légendaire", "reject": "Reject"}


def _r(sid, studio="A", perfs=("p1",)):
    return {"scene_id": sid, "studio": studio, "performer_ids": list(perfs)}


def test_torn_guesses_and_unknown_studios_come_first_with_a_reason():
    graded = [_r(f"g{i}") for i in range(10)]
    cands = [_r("sure"), _r("torn"), _r("new", studio="Brand New", perfs=("p1",))]
    preds = {"sure": {"probs": {"legendaire": 0.95, "exceptionnelle": 0.04, "reject": 0.01}},
             "torn": {"probs": {"legendaire": 0.45, "exceptionnelle": 0.42, "reject": 0.13}},
             "new": {"probs": {"legendaire": 0.8, "exceptionnelle": 0.15, "reject": 0.05}}}
    out = teach.rank(cands, preds, graded, NAMES)
    assert [r["scene_id"] for r in out] == ["torn", "new", "sure"]
    assert out[0]["teach"]["why"] == "torn between Légendaire and Exceptionnelle"
    assert out[1]["teach"]["why"] == "first Brand New scene you'd grade"


def test_spread_caps_a_studio_and_skips_near_duplicates():
    cands = [_r(str(i), studio="Same") for i in range(5)] + [_r("x", studio="Other")]
    preds = {str(i): {"probs": {"legendaire": 0.5, "exceptionnelle": 0.5}} for i in range(5)}
    preds["x"] = {"probs": {"legendaire": 0.9, "exceptionnelle": 0.1}}
    out = [r["scene_id"] for r in teach.rank(cands, preds, [], NAMES)]
    assert out[:3] == ["0", "1", "x"]                       # two from "Same", then someone else
    v = np.ones(4) / 2
    out = [r["scene_id"] for r in teach.rank([_r("a", "S1"), _r("b", "S2")], {}, [], NAMES,
                                              vecs={"a": v, "b": v})]
    assert out == ["a", "b"]                                # the near-twin is listed, just after


def test_vr_gaps_ask_for_likely_rejects_while_rejects_are_short():
    cands = [_r("meets"), _r("below"), _r("fresh", studio="Nobody")]
    vrq = {"meets": {"verdict": "meets"}, "below": {"verdict": "below"}, "fresh": {"verdict": "meets"}}
    graded = [_r("g1"), _r("g2")]
    out = teach.rank_vr_gaps(cands, vrq, graded, {"reject": 8}, NAMES)
    assert [r["scene_id"] for r in out] == ["below", "fresh", "meets"]   # rejects needed most, then a new studio
    assert "likely reject" in out[0]["teach"]["why"] and "no VR from Nobody" in out[1]["teach"]["why"]
    out = teach.rank_vr_gaps(cands, vrq, graded, {"legendaire": 5}, NAMES)
    assert out[-1]["scene_id"] == "below"                   # keepers short: likely keepers first


def test_teaching_order_beats_random_on_a_synthetic_library():
    """The measurement behind the feature, small: accuracy on held-out scenes
    after 100 grades, Teach Peaks order vs random order."""
    from sklearn.linear_model import LogisticRegression

    cls = ["reject", "exceptionnelle", "legendaire"]
    gains = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        centers = rng.normal(0, 1.0, (3, 12))
        lean = rng.integers(0, 3, 40)
        rows, X, y = [], [], []
        for i in range(900):
            st = int(rng.integers(0, 40))
            c = lean[st] if rng.random() < 0.6 else int(rng.integers(0, 3))
            X.append(np.append(centers[c] + rng.normal(0, 1.3, 12), np.eye(40)[st] * 1.5))
            y.append(c)
            rows.append(_r(str(i), studio=f"S{st}", perfs=()))
        X, y = np.array(X), np.array(y)
        test = np.arange(600, 900)
        acc = {}
        for order in ("teach", "random"):
            r2 = np.random.default_rng(100 + seed)
            lab = list(r2.choice(600, 10, replace=False))
            while len(lab) < 100:
                cand = [i for i in range(600) if i not in set(lab)]
                if order == "random" or len(set(y[lab])) < 3:
                    lab += list(r2.choice(cand, 5, replace=False))
                    continue
                m = LogisticRegression(max_iter=2000).fit(X[lab], y[lab])
                P = m.predict_proba(X[cand])
                preds = {str(i): {"probs": dict(zip(cls, map(float, p)))} for i, p in zip(cand, P)}
                lab += [int(r["scene_id"]) for r in teach.rank([rows[i] for i in cand], preds,
                                                                [rows[i] for i in lab], {}, limit=5)]
            m = LogisticRegression(max_iter=2000).fit(X[lab[:100]], y[lab[:100]])
            acc[order] = float((m.predict(X[test]) == y[test]).mean())
        gains.append(acc["teach"] - acc["random"])
    assert np.mean(gains) > 0


def test_the_view_and_the_heresphere_tag(svc, stash):
    for sid in ("50", "51", "52"):
        stash.s[sid] = {"rating100": None, "o_counter": 0, "tag_ids": [], "organized": False, "fingerprint": f"fp{sid}",
                        "path": f"/data/#VR/{sid}.mp4", "studio": f"Studio{sid}"}
    svc._catalogue_all(refresh=True)
    d = svc.catalogue(view="teach")
    assert {i["scene_id"] for i in d["items"]} >= {"50", "51", "52"} and all(i.get("teach") for i in d["items"])
    import time

    for _ in range(100):                                    # (the listing may have started a background refresh)
        if not getattr(svc, "_cat_refreshing", False):
            break
        time.sleep(0.05)
    svc.sync_teach_tag()
    tid = next(t for t, n in stash.tags.items() if n == "Peaks · teach me")
    assert all(tid in stash.s[s]["tag_ids"] for s in ("50", "51", "52"))
    assert svc.sync_teach_tag() == {"added": 0, "removed": 0}       # nothing changed: nothing written
    stash.s["50"]["rating100"] = 100                        # graded in HereSphere
    svc._catalogue_all(refresh=True)
    out = svc.sync_teach_tag()
    assert out["removed"] == 1 and tid not in stash.s["50"]["tag_ids"]
    svc.save_teach_tag(False)
    svc.sync_teach_tag()
    assert not any(tid in v["tag_ids"] for v in stash.s.values())
