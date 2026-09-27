"""Keeper-triage model on synthetic libraries built to test specific claims:
it learns tiers from the picture, learns a bitrate preference only when quality
features are on, and tells a high-bitrate upscale from a real 4K master by the
picture — plus the transparent quality floor."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from peaks.tier_model import (  # noqa: E402
    CLASSES, TierModel, cross_validate, fit_best, quality_features, quality_flag,
    quality_floor, usable_classes, visual_features,
)

D = 32


def _unit(v):
    return v / np.linalg.norm(v)


def _library(rng, n_per=24, visual_sep=True, quality_sep=False, upscale_mbps=45):
    """Scenes for every class; `visual_sep` gives each class its own look,
    `quality_sep` makes bitrate rise with the tier (upscale excepted)."""
    centers = {c: _unit(rng.standard_normal(D)) for c in CLASSES}
    Xv, Xq, y = [], [], []
    for c in CLASSES:
        for _ in range(n_per):
            base = centers[c] if visual_sep else centers["merveilleuse"]
            frames = np.stack([_unit(base + 0.35 * rng.standard_normal(D)) for _ in range(12)])
            Xv.append(visual_features(frames, rng.random(12)))
            mbps = 8.0
            if quality_sep:
                mbps = {"reject": 3, "upscale": upscale_mbps, "merveilleuse": 12,
                        "exceptionnelle": 25, "legendaire": 50}[c] * float(rng.uniform(0.85, 1.15))
            Xq.append(quality_features({"mbps": mbps, "res": "4K", "fps": 30, "codec": "hevc",
                                        "bpp": mbps * 1e6 / (3840 * 2160 * 30)}))
            y.append(c)
    return np.stack(Xv), np.stack(Xq), y


def test_learns_tiers_from_the_picture():
    Xv, Xq, y = _library(np.random.default_rng(0), visual_sep=True)
    cv = cross_validate(Xv, Xq, y, use_quality=True)
    assert cv["exact"] >= 0.9 and cv["within_one"] >= 0.95


def test_bitrate_preference_needs_quality_features():
    # the picture carries NO tier signal; each tier has its own bitrate band
    Xv, Xq, y = _library(np.random.default_rng(1), visual_sep=False, quality_sep=True,
                         upscale_mbps=35)
    model, rep = fit_best(Xv, Xq, y)
    assert rep["cv"]["exact"] >= 0.8
    assert rep["cv_without_quality"]["exact"] <= 0.4        # ~chance over 5 classes
    assert rep["quality_gain"] >= 0.4
    assert rep["pca_dim"] <= 16                              # small: the picture is noise here


def test_fixed_large_pca_would_drown_quality():
    """Why the PCA size is chosen by CV: fixed at 48 on 120 scenes, noise
    dimensions swamp the real bitrate signal."""
    Xv, Xq, y = _library(np.random.default_rng(1), visual_sep=False, quality_sep=True,
                         upscale_mbps=35)
    big = cross_validate(Xv, Xq, y, use_quality=True, pca_dim=48)
    small = cross_validate(Xv, Xq, y, use_quality=True, pca_dim=8)
    assert small["exact"] - big["exact"] >= 0.3


def test_high_bitrate_upscale_separated_by_picture():
    # upscale and legendaire are BOTH 4K at ~45-50 Mbps — only the picture differs
    rng = np.random.default_rng(2)
    Xv, Xq, y = _library(rng, visual_sep=True, quality_sep=True)
    m = TierModel().fit(Xv, Xq, y)
    pred = [m.classes_[i] for i in np.argmax(m.predict_proba(Xv, Xq), axis=1)]
    for c in ("upscale", "legendaire"):
        rows = [p for p, t in zip(pred, y) if t == c]
        assert sum(p == c for p in rows) / len(rows) >= 0.9


def test_summary_and_keeper_probability():
    Xv, Xq, y = _library(np.random.default_rng(3))
    m = TierModel().fit(Xv, Xq, y)
    s = m.summarize(m.predict_proba(Xv[:1], Xq[:1]))[0]
    assert set(s["probs"]) == set(CLASSES)
    assert abs(sum(s["probs"].values()) - 1) < 0.01
    assert 0 <= s["keeper"] <= 1 and 0 <= s["expected"] <= len(CLASSES) - 1
    assert s["keeper"] == pytest.approx(1 - s["probs"]["reject"], abs=0.002)


def test_short_classes_reported():
    y = ["legendaire"] * 12 + ["merveilleuse"] * 30 + ["reject"] * 3
    ok, short = usable_classes(y)
    assert ok == ["merveilleuse", "legendaire"] and short == {"reject": 3}


def test_save_load_round_trip(tmp_path):
    Xv, Xq, y = _library(np.random.default_rng(4))
    m = TierModel().fit(Xv, Xq, y)
    back = TierModel.load(m.save(tmp_path / "tier.pkl"))
    assert np.allclose(back.predict_proba(Xv[:5], Xq[:5]), m.predict_proba(Xv[:5], Xq[:5]))


def test_quality_features_mark_missing():
    known = quality_features({"mbps": 40, "res": "4K", "fps": 60, "codec": "hevc", "bpp": 0.08})
    unknown = quality_features({})
    assert known.shape == unknown.shape
    assert unknown[3] == 1.0 and unknown[4] == 1.0 and known[3] == 0.0   # explicit missing flags


def _row(tier, res, mbps):
    return {"tier": tier, "quality": {"res": res, "mbps": mbps}}


def test_quality_floor_and_flag():
    rows = ([_row("legendaire", "4K", m) for m in (40, 55, 30, 22, 60)]
            + [_row("merveilleuse", "1080p", m) for m in (9, 12, 15, 10, 8)]
            + [_row("upscale", "4K", 5), _row("unreviewed", "4K", 3)])   # not tiered: ignored
    floor = quality_floor(rows)
    assert floor["by_res"]["4K"] == {"mbps": 22, "n": 5}
    assert floor["by_res"]["1080p"]["mbps"] == 8 and floor["lowest_res"] == "1080p"
    assert "below every 4K scene" in quality_flag({"res": "4K", "mbps": 12}, floor)
    assert quality_flag({"res": "4K", "mbps": 30}, floor) is None
    assert "never tiered anything below 1080p" in quality_flag({"res": "720p", "mbps": 50}, floor)
    assert quality_flag({"res": "1440p", "mbps": 1}, floor) is None      # no floor for 1440p yet


# --- performer / studio track records ----------------------------------------

from peaks.tier_model import (  # noqa: E402
    WHO_FEATURES, WhoEncoder, _oof_who, fit_fallback, who_of,
)


def _feat(enc, who, own=None):
    return dict(zip(WHO_FEATURES, enc.encode([who], None if own is None else [own])[0]))


def test_records_are_smoothed_toward_the_library_mean():
    # library of mid-grade scenes, then one performer with 1 vs 10 Légendaires
    who = [{"performers": ["x"], "studio": "s"} for _ in range(40)]
    y = ["merveilleuse"] * 40
    one = WhoEncoder().fit(who + [{"performers": ["star"], "studio": ""}], y + ["legendaire"])
    ten = WhoEncoder().fit(who + [{"performers": ["star"], "studio": ""}] * 10, y + ["legendaire"] * 10)
    base1 = one.base[0] / 4
    g1 = _feat(one, {"performers": ["star"], "studio": ""})["perf_grade_max"]
    g10 = _feat(ten, {"performers": ["star"], "studio": ""})["perf_grade_max"]
    assert g1 - base1 < 0.2                     # one grade barely moves the record
    assert g10 > 0.85                           # ten grades make it (nearly) Légendaire
    new = _feat(ten, {"performers": ["nobody"], "studio": ""})
    assert new["perf_known_frac"] == 0 and new["perf_graded_log"] == 0
    assert _feat(ten, {"performers": [], "studio": ""})["no_performers"] == 1
    assert _feat(ten, None)["who_missing"] == 1


def test_a_scenes_own_grade_never_feeds_its_record():
    who = [{"performers": ["p"], "studio": "s"}]
    enc = WhoEncoder().fit(who, ["legendaire"])
    loo = _feat(enc, who[0], own="legendaire")
    assert loo["perf_graded_log"] == 0 and loo["studio_graded_log"] == 0
    assert loo["perf_grade_max"] == pytest.approx(enc.base[0] / 4)
    # out-of-fold training encodings: a lone performer's only scene knows nothing of itself
    oof = _oof_who(who * 1 + [{"performers": ["q"], "studio": "t"}] * 9, ["legendaire"] + ["reject"] * 9)
    assert oof[0, WHO_FEATURES.index("perf_known_frac")] == 0
    ex = enc.explain(who[0], own="legendaire")
    assert ex["performers"][0]["n"] == 0 and ex["studio"]["n"] == 0
    assert enc.explain(who[0])["performers"][0] == {
        "id": "p", "n": 1, "grade": 4.0, "keep": 1.0, "top": 1.0,
        "shrunk": pytest.approx((4 + 3 * enc.base[0]) / 4, abs=1e-3)}


def _who_library(rng, who_matters=True, n=200):
    """Picture and file quality carry nothing; with `who_matters`, performers
    A/B only make Légendaire scenes and studio 'Bad' only rejects."""
    Xv = np.stack([visual_features(np.stack([_unit(rng.standard_normal(D)) for _ in range(6)]))
                   for _ in range(n)])
    Xq = np.stack([quality_features({"mbps": 8, "res": "1080p", "fps": 30, "codec": "h264"})] * n)
    y, who = [], []
    others = [f"p{i}" for i in range(12)]
    for i in range(n):
        c = CLASSES[i % 5]
        if who_matters and c == "legendaire":
            w = {"performers": [["A", "B"][i % 2], str(rng.choice(others))], "studio": "Good"}
        elif who_matters and c == "reject":
            w = {"performers": [str(rng.choice(others))], "studio": "Bad"}
        else:
            w = {"performers": [str(rng.choice(others))], "studio": str(rng.choice(["Good", "Mid", "Bad"]))}
        y.append(c)
        who.append(w)
    return Xv, Xq, y, who


def test_who_is_kept_when_it_helps_and_predictions_follow():
    Xv, Xq, y, who = _who_library(np.random.default_rng(3))
    model, rep = fit_best(Xv, Xq, y, who=who)
    assert rep["use_who"] and rep["who_gain"] > 0.1 and model.use_who
    assert rep["cv"] == rep["cv_with_who"]
    fresh = np.stack([visual_features(np.stack([_unit(np.ones(D))] * 3))] * 2)
    s = model.summarize(model.predict_proba(fresh, Xq[:2], who=[
        {"performers": ["A"], "studio": "Good"}, {"performers": ["p1"], "studio": "Bad"}]))
    assert s[0]["expected"] > s[1]["expected"]
    assert s[1]["probs"]["reject"] > s[0]["probs"]["reject"]


def test_who_is_dropped_when_it_doesnt_help():
    Xv, Xq, y, who = _who_library(np.random.default_rng(4), who_matters=False)
    model, rep = fit_best(Xv, Xq, y, who=who)
    assert rep["use_who"] is False and not model.use_who
    assert "cv_without_who" not in rep
    model.predict_proba(Xv[:2], Xq[:2])                    # no who needed


def test_fallback_model_judges_unembedded_scenes_by_who():
    _, Xq, y, who = _who_library(np.random.default_rng(5))
    m, rep = fit_fallback(Xq, y, who)
    assert rep["trained"] and rep["who_gain"] > 0.1
    s = m.summarize(m.predict_proba(None, Xq[:2], who=[
        {"performers": ["B"], "studio": "Good"}, {"performers": ["p2"], "studio": "Bad"}]))
    assert s[0]["tier"] == "legendaire" and s[1]["tier"] == "reject"
    _, Xq2, y2, who2 = _who_library(np.random.default_rng(6), who_matters=False)
    m2, rep2 = fit_fallback(Xq2, y2, who2)
    assert m2 is None and not rep2["trained"]


def test_old_pickled_models_still_predict():
    Xv, Xq, y = _library(np.random.default_rng(0))
    m = TierModel(pca_dim=8).fit(Xv, Xq, y)
    del m.__dict__["use_who"], m.__dict__["use_visual"], m.__dict__["who_enc"]   # a pre-who pickle
    assert m.predict_proba(Xv[:3], Xq[:3]).shape == (3, len(m.classes_))


def test_who_of_reads_catalogue_rows():
    assert who_of({"performer_ids": ["7", None, 9], "studio": "S"}) == {"performers": ["7", "9"], "studio": "S"}
    assert who_of({}) == {"performers": [], "studio": ""}
