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
