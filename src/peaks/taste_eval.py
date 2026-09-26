"""Honest measurement of the taste model: held-out, scene-grouped evaluation.

Training-set CV with random background rows mixed into the test folds flatters
the model (random frames are easy negatives). This scores only what matters:

- `auc`: ROC-AUC over your *explicit* ratings (👍/👎/⭐), predicted by a model that
  never saw that scene — frames of one scene never straddle train and test.
- `p_at_50`: of the 50 held-out moments the model ranks highest (or the top R
  when you've loved fewer than 50), the share you actually loved — compare
  against `base_rate`, the share a coin would get.
- `peak_hit`: for held-out scenes holding a moment you loved, how often the
  model's single best moment in the scene lands within 10 s of one of them —
  i.e. how often "jump to peak" would land on your moment.
"""

from __future__ import annotations

import numpy as np

PEAK_TOLERANCE = 10.0  # seconds


def _auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def grouped_oof(
    fit, X: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray,
    eval_mask: np.ndarray, bg_mask: np.ndarray, *, scenes: dict | None = None,
    folds: int = 5, seed: int = 0,
) -> dict | None:
    """Out-of-fold evaluation. `fit(X, y, w)` returns a `predict(X)` callable.

    Rows are split by `groups` (scene key), stratified by label. Background rows
    (`bg_mask`) only ever train, and never from a test fold's own scenes.
    `scenes`, if given, is {key: (times, feats, positive_times)} for the
    peak-hit metric (feats already in the same feature space as X).
    Returns None when there aren't enough scenes of each class to evaluate."""
    from sklearn.model_selection import StratifiedGroupKFold

    ev = np.where(eval_mask & ~bg_mask)[0]
    if ev.size == 0:
        return None
    pos_groups = {groups[i] for i in ev if y[i] == 1}
    neg_groups = {groups[i] for i in ev if y[i] == 0}
    k = min(folds, len(pos_groups), len(neg_groups))
    if k < 2:
        return None
    lab = np.where(~bg_mask)[0]           # every labeled row takes part in splitting
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(y.shape[0], np.nan, dtype=np.float64)
    hits, tried = 0, 0
    for tr_i, te_i in splitter.split(lab, y[lab], groups[lab]):
        test_groups = set(groups[lab[te_i]].tolist())
        train = np.concatenate([lab[tr_i], np.where(bg_mask & ~np.isin(groups, list(test_groups)))[0]])
        if len(set(y[train].tolist())) < 2:
            continue
        predict = fit(X[train], y[train], w[train])
        te = lab[te_i]
        te = te[eval_mask[te]]
        if te.size:
            oof[te] = predict(X[te])
        for g in test_groups:
            sc = (scenes or {}).get(g)
            if sc is None:
                continue
            times, feats, pos_t = sc
            if not len(pos_t) or not len(times):
                continue
            best = float(times[int(np.argmax(predict(feats)))])
            tried += 1
            hits += int(np.min(np.abs(np.asarray(pos_t) - best)) <= PEAK_TOLERANCE)
    done = ev[~np.isnan(oof[ev])]
    if done.size == 0:
        return None
    ys, ss = y[done], oof[done]
    # precision in the top 50 — or the top R (R = held-out positives) when
    # there are fewer than 50, so a small benchmark isn't just the base rate
    top_n = min(50, max(1, int(ys.sum())))
    top = np.argsort(-ss)[:top_n]
    return {
        "auc": None if (a := _auc(ys, ss)) is None else round(a, 3),
        "p_at_50": round(float(ys[top].mean()), 3),
        "base_rate": round(float(ys.mean()), 3),
        "peak_hit": round(hits / tried, 3) if tried else None,
        "peak_scenes": tried,
        "folds": k, "evaluated": int(done.size),
    }


def better(a: dict | None, b: dict | None, margin: float = 0.01) -> bool:
    """Does result `a` beat `b` (by more than `margin` AUC — the extra
    complexity has to earn its place)?"""
    if not a or a.get("auc") is None:
        return False
    if not b or b.get("auc") is None:
        return True
    return a["auc"] > b["auc"] + margin
