"""The shape of a moment: how a stretch of a scene unfolds, not one frame.

Given a window's frames — their times, taste scores and embeddings — these pure
functions describe how it builds, peaks and holds, how much changes, and where
the moment starts and ends. The moment model (peaks.web.zoom) ranks candidate
moments on these features, trained on your saves and 👍/👎.
"""

from __future__ import annotations

import numpy as np

FEATURES: tuple[str, ...] = ("peak", "contrast", "rise", "fall", "hold", "change_mean", "change_max")
# how each feature reads in plain words when it pushes a moment up
WORDS = {"peak": "a strong best frame", "contrast": "stands out from around it", "rise": "a build-up into it",
         "fall": "a quick drop after", "hold": "it holds for a while", "change_mean": "lots of movement",
         "change_max": "a cut or sudden change"}
SPAN_MIN, SPAN_MAX = 6.0, 40.0
SLOPE_SPAN = 10.0                 # seconds either side of the peak for rise / fall


def _slope(t: np.ndarray, s: np.ndarray) -> float:
    """Least-squares slope of s over t (per second); 0 with fewer than 2 points."""
    if len(t) < 2 or float(np.ptp(t)) == 0:
        return 0.0
    tc = t - t.mean()
    return float((tc * (s - s.mean())).sum() / (tc * tc).sum())


def shape_features(times, scores, vecs, hold_level: float) -> dict[str, float]:
    """The shape features of one window (times ascending, seconds)."""
    t = np.asarray(times, np.float64)
    s = np.asarray(scores, np.float64)
    if len(t) == 0:
        return {k: 0.0 for k in FEATURES}
    p = int(np.argmax(s))
    peak = float(s[p])
    before = (t >= t[p] - SLOPE_SPAN) & (t <= t[p])
    after = (t >= t[p]) & (t <= t[p] + SLOPE_SPAN)
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    hold = float((s >= hold_level).sum() * step)
    v = np.asarray(vecs, np.float32)
    if len(v) > 1:
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        change = 1.0 - (v[1:] * v[:-1]).sum(axis=1)
        cm, cx = float(change.mean()), float(change.max())
    else:
        cm = cx = 0.0
    return {"peak": peak, "contrast": peak - float(np.median(s)),
            "rise": _slope(t[before], s[before]), "fall": -_slope(t[after], s[after]),
            "hold": hold, "change_mean": cm, "change_max": cx}


def span(times, scores) -> tuple[float, float, float]:
    """(start, peak, end) of the moment in a window: out from the peak while the
    score stays above halfway between the window's median and its peak, then
    clamped to SPAN_MIN..SPAN_MAX seconds around the peak."""
    t = np.asarray(times, np.float64)
    s = np.asarray(scores, np.float64)
    if len(t) == 0:
        return 0.0, 0.0, 0.0
    p = int(np.argmax(s))
    level = s[p] - 0.5 * (s[p] - float(np.median(s)))
    a = p
    while a > 0 and s[a - 1] >= level:
        a -= 1
    b = p
    while b < len(s) - 1 and s[b + 1] >= level:
        b += 1
    start, end, pk = float(t[a]), float(t[b]), float(t[p])
    if end - start < SPAN_MIN:                         # too short: widen evenly around the peak
        half = SPAN_MIN / 2
        start, end = pk - half, pk + half
    if end - start > SPAN_MAX:                         # too long: keep the part around the peak
        start = max(start, pk - SPAN_MAX / 2)
        end = min(end, start + SPAN_MAX)
    lo = float(t[0])
    if start < lo:                                     # never before the window (or the scene)
        end += lo - start
        start = lo
    return round(max(0.0, start), 2), round(pk, 2), round(end, 2)
