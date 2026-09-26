"""Keeper triage: learn the user's tiers from the scenes they've already graded.

Each graded scene becomes one example with two kinds of features:

* **visual** — the mean of the scene's frame embeddings, plus the mean of its
  most on-taste frames (what the scene looks like overall, and at its best),
  compressed with PCA so ~300 examples aren't drowned in 2k dimensions;
* **quality** — log bitrate, bits-per-pixel-per-frame, resolution class, frame
  rate and codec. Learned jointly with the visual side rather than hand-set,
  because high bitrate isn't simply "better": AI upscales (the Upscale tier) are
  often 4K and high-bitrate too, and only the picture tells them apart.

A multinomial logistic regression over the grades
(reject < upscale < merveilleuse < exceptionnelle < legendaire) produces, for
any scene, a probability per tier, an expected tier (ordinal mean) and a keeper
probability (1 − P(reject)). Everything here is suggestion only — nothing is
graded without a click.

sklearn is imported lazily (the `[ml]` extra), like the taste classifier.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np

from .tiers import RES_CLASSES

CLASSES: tuple[str, ...] = ("reject", "upscale", "merveilleuse", "exceptionnelle", "legendaire")
ORDINAL: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
# catalogue tier → training class (anomaly / unreviewed are not training data)
TIER_CLASS: dict[str, str] = {
    "rejected": "reject", "upscale": "upscale", "merveilleuse": "merveilleuse",
    "exceptionnelle": "exceptionnelle", "legendaire": "legendaire",
}
MIN_PER_CLASS = 10
_CODECS = ("h264", "hevc", "av1", "vp9")


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def visual_features(frames: np.ndarray, scores: np.ndarray | None = None,
                    top_frac: float = 0.1) -> np.ndarray:
    """(n, d) float32 frame embeddings (+ optional per-frame taste scores) →
    (2d,) = [unit mean of all frames, unit mean of the top `top_frac` frames]."""
    frames = np.asarray(frames, dtype=np.float32)
    mean = _unit(frames.mean(axis=0))
    if scores is None or len(scores) != len(frames):
        top = mean
    else:
        k = max(1, int(math.ceil(top_frac * len(frames))))
        idx = np.argsort(-np.asarray(scores))[:k]
        top = _unit(frames[idx].mean(axis=0))
    return np.concatenate([mean, top]).astype(np.float32)


_FEAT_RES: tuple[str, ...] = ("SD", "720p", "1080p", "1440p", "4K")


def quality_features(q: dict) -> np.ndarray:
    """A scene's quality facts (peaks.tiers.quality_of) → a fixed-length vector.
    Unknown values become 0 with an explicit 'missing' flag, so a file Stash
    hasn't probed isn't mistaken for a terrible one."""
    mbps = q.get("mbps")
    bpp = q.get("bpp")
    fps = q.get("fps")
    res = q.get("res")
    codec = (q.get("codec") or "").lower()
    feats = [
        math.log1p(mbps) if mbps else 0.0,
        min(float(bpp), 1.0) if bpp else 0.0,
        min(float(fps), 120.0) / 60.0 if fps else 0.0,
        0.0 if mbps else 1.0,                       # bitrate missing
        0.0 if res else 1.0,                        # resolution missing
    ]
    # one-hot over the classes the model was built with; 5K–8K share the top
    # slot so saved models and reject memories keep their feature size (bitrate
    # and bpp still tell them apart)
    top = _FEAT_RES[-1]
    r = top if res and RES_CLASSES.index(res) >= RES_CLASSES.index(top) else res
    feats += [1.0 if r == c else 0.0 for c in _FEAT_RES]
    feats += [1.0 if codec == c else 0.0 for c in _CODECS]
    feats.append(1.0 if codec and codec not in _CODECS else 0.0)
    return np.asarray(feats, dtype=np.float32)


class TierModel:
    """PCA(visual) ⊕ quality → standardize → multinomial logistic regression."""

    def __init__(self, use_quality: bool = True, pca_dim: int = 48, C: float = 0.5):
        self.use_quality = use_quality
        self.pca_dim = pca_dim
        self.C = C
        self.classes_: list[str] = []
        self._pca = self._scaler = self._clf = None

    def _design(self, Xv: np.ndarray, Xq: np.ndarray, fit: bool) -> np.ndarray:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        if fit:
            k = max(1, min(self.pca_dim, Xv.shape[0] - 1, Xv.shape[1]))
            self._pca = PCA(n_components=k, random_state=0).fit(Xv)
        parts = [self._pca.transform(Xv)]
        if self.use_quality:
            parts.append(Xq)
        X = np.hstack(parts)
        if fit:
            self._scaler = StandardScaler().fit(X)
        return self._scaler.transform(X)

    def fit(self, Xv: np.ndarray, Xq: np.ndarray, y: list[str]) -> "TierModel":
        from sklearn.linear_model import LogisticRegression

        X = self._design(np.asarray(Xv, np.float32), np.asarray(Xq, np.float32), fit=True)
        self._clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000)
        self._clf.fit(X, np.asarray(y))
        self.classes_ = [str(c) for c in self._clf.classes_]
        return self

    def predict_proba(self, Xv: np.ndarray, Xq: np.ndarray) -> np.ndarray:
        X = self._design(np.asarray(Xv, np.float32), np.asarray(Xq, np.float32), fit=False)
        return self._clf.predict_proba(X)

    def summarize(self, proba: np.ndarray) -> list[dict]:
        """Per row: predicted class, its confidence, expected tier on the
        CLASSES ordinal scale, keeper probability, and the full distribution."""
        out = []
        ords = np.array([ORDINAL[c] for c in self.classes_], dtype=np.float32)
        rej = self.classes_.index("reject") if "reject" in self.classes_ else None
        for p in proba:
            j = int(np.argmax(p))
            out.append({
                "tier": self.classes_[j],
                "conf": round(float(p[j]), 3),
                "expected": round(float(p @ ords), 3),
                "keeper": round(1.0 - float(p[rej]), 3) if rej is not None else None,
                "probs": {c: round(float(v), 3) for c, v in zip(self.classes_, p)},
            })
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(self, fh)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TierModel":
        """Pickle executes code on load — only load models you created (they
        live in your local models/ dir)."""
        with open(path, "rb") as fh:
            return pickle.load(fh)


def usable_classes(y: list[str], min_per_class: int = MIN_PER_CLASS) -> tuple[list[str], dict]:
    """Classes with enough examples to learn, and the counts of those without."""
    counts = {c: y.count(c) for c in CLASSES}
    ok = [c for c in CLASSES if counts[c] >= min_per_class]
    short = {c: counts[c] for c in CLASSES if 0 < counts[c] < min_per_class}
    return ok, short


def cross_validate(Xv: np.ndarray, Xq: np.ndarray, y: list[str], use_quality: bool,
                   folds: int = 5, pca_dim: int = 48) -> dict:
    """Stratified k-fold: exact accuracy and 'within one tier' accuracy (the
    predicted tier is at most one step from the real one on the ordinal)."""
    from sklearn.model_selection import StratifiedKFold

    y_arr = np.asarray(y)
    k = max(2, min(folds, min(np.unique(y_arr, return_counts=True)[1])))
    exact = near = 0
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=0).split(Xv, y_arr):
        m = TierModel(use_quality=use_quality, pca_dim=pca_dim).fit(Xv[tr], Xq[tr], list(y_arr[tr]))
        pred = [m.classes_[int(i)] for i in np.argmax(m.predict_proba(Xv[te], Xq[te]), axis=1)]
        for p, t in zip(pred, y_arr[te]):
            exact += p == t
            near += abs(ORDINAL[p] - ORDINAL[str(t)]) <= 1
    n = len(y)
    return {"exact": round(exact / n, 3), "within_one": round(near / n, 3), "folds": k}


def quality_floor(rows: list[dict], min_count: int = 5) -> dict:
    """The transparent quality flag: per resolution class, the lowest bitrate
    among scenes the user has graded Merveilleuse or better (needs `min_count`
    such scenes to set a floor). Scenes under it are 'below every scene you've
    tiered at this resolution'."""
    tiered = [r for r in rows if r.get("tier") in ("merveilleuse", "exceptionnelle", "legendaire")]
    floors: dict[str, dict] = {}
    for res in RES_CLASSES:
        mb = [r["quality"]["mbps"] for r in tiered
              if r["quality"].get("res") == res and r["quality"].get("mbps")]
        if len(mb) >= min_count:
            floors[res] = {"mbps": min(mb), "n": len(mb)}
    tiered_res = {r["quality"].get("res") for r in tiered if r["quality"].get("res")}
    lowest = min((RES_CLASSES.index(x) for x in tiered_res), default=None)
    return {"by_res": floors,
            "lowest_res": RES_CLASSES[lowest] if lowest is not None else None,
            "tiered": len(tiered)}


def quality_flag(q: dict, floor: dict) -> str | None:
    """A one-line reason when a scene falls under the learned floor, else None."""
    res, mbps = q.get("res"), q.get("mbps")
    low = floor.get("lowest_res")
    if res and low and RES_CLASSES.index(res) < RES_CLASSES.index(low):
        return f"{res} — you've never tiered anything below {low}"
    f = floor.get("by_res", {}).get(res or "")
    if f and mbps and mbps < f["mbps"]:
        return (f"{mbps:g} Mbps · {res} — below every {res} scene you've tiered "
                f"(lowest {f['mbps']:g} Mbps of {f['n']})")
    return None


PCA_GRID: tuple[int, ...] = (8, 16, 32, 48)


def fit_best(Xv: np.ndarray, Xq: np.ndarray, y: list[str]) -> tuple[TierModel, dict]:
    """Pick the visual PCA size by cross-validation, then fit on everything.

    The right size depends on how many scenes are graded and how much the
    picture carries: on ~120 examples, 48 components of mostly-noise visual
    dims swamped a real bitrate signal (37% vs 89% at 8 components in the
    synthetic test), while a library whose tiers really live in the picture
    wants more. Candidates are capped at n/4 so tiny sets stay small. The CV
    score of the chosen size is reported (mildly optimistic: it was also
    used to choose among a handful of sizes). Also reports the same size
    WITHOUT quality features, so the value of bitrate is measured, not assumed.
    """
    n = len(y)
    grid = [k for k in PCA_GRID if k <= max(PCA_GRID[0], n // 4)] or [PCA_GRID[0]]
    best_k, best = grid[0], None
    for k in grid:
        cv = cross_validate(Xv, Xq, y, use_quality=True, pca_dim=k)
        if best is None or (cv["exact"], cv["within_one"]) > (best["exact"], best["within_one"]):
            best_k, best = k, cv
    without = cross_validate(Xv, Xq, y, use_quality=False, pca_dim=best_k)
    model = TierModel(use_quality=True, pca_dim=best_k).fit(Xv, Xq, y)
    return model, {"pca_dim": best_k, "cv": best, "cv_without_quality": without,
                   "quality_gain": round(best["exact"] - without["exact"], 3)}

