"""The shape of a moment: rise, hold, change, and where it starts and ends."""

import numpy as np

from peaks.moment_shape import FEATURES, shape_features, span

T = np.arange(0, 42, 2.0)            # a ±20 s window at 2 s


def _vecs(n, cut_at=None):
    v = np.tile(np.eye(4)[0], (n, 1)).astype(np.float32)
    if cut_at is not None:
        v[cut_at:] = np.eye(4)[1]
    return v


def test_a_slow_build_rises_and_a_drop_falls():
    s = np.clip((T - 10) / 20, 0, 1)                 # builds from 10 s to a peak at 30 s
    s[T > 30] = 0.2                                  # then drops
    f = shape_features(T, s, _vecs(len(T)), hold_level=0.7)
    assert set(f) == set(FEATURES)
    assert f["rise"] > 0.03 and f["fall"] > 0.01
    assert f["peak"] == 1.0 and f["contrast"] > 0.4
    assert f["hold"] == 2.0 * int((s >= 0.7).sum())


def test_a_plateau_holds_and_a_cut_shows_as_change():
    s = np.where((T >= 10) & (T <= 30), 0.9, 0.1)
    f = shape_features(T, s, _vecs(len(T), cut_at=12), hold_level=0.8)
    assert f["hold"] == 22.0                         # 10 s … 30 s inclusive, at 2 s
    assert f["change_max"] > 0.9 and f["change_mean"] < 0.1


def test_span_bounds_and_clamps():
    s = np.where((T >= 12) & (T <= 24), 1.0, 0.0)
    s[9] = 1.2                                       # the peak at 18 s, inside the plateau
    assert span(T, s) == (12.0, 18.0, 24.0)
    narrow = np.zeros(len(T)); narrow[10] = 1.0      # a single spike at 20 s → at least 6 s
    a, p, b = span(T, narrow)
    assert p == 20.0 and b - a == 6.0
    wide = np.ones(len(T)); wide[0] = 2.0            # peak at the very start, all high → ≤ 40 s, not before 0
    a, p, b = span(T, wide)
    assert a == 0.0 and b - a <= 40.0
