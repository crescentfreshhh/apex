"""The Tier-2 taste classifier.

A small supervised model trained on cached frame embeddings: positives = frames
you labeled "want it", negatives = "don't". It emits a per-frame probability in
[0, 1] with the *same shape* the Tier-1 similarity scorer produces, so the
downstream segment extraction is identical — only the score source changes.

sklearn is imported lazily (it lives in the `[ml]` extra), so importing this
module stays cheap. Models are pickled with their embedder name + dim so we can
refuse to score a cache built with a different embedder.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _weighted_resample(X, y, w, seed: int = 0):
    """Per-class weighted bootstrap: resample each class to its own size, drawing
    with probability ∝ weight. Emulates sample_weight for estimators that lack
    it (MLP) while guaranteeing both classes survive at their original balance."""
    rng = np.random.default_rng(seed)
    w = np.clip(np.asarray(w, dtype=np.float64), 1e-9, None)
    picks = []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        if idx.size:
            p = w[idx] / w[idx].sum()
            picks.append(rng.choice(idx, size=idx.size, replace=True, p=p))
    sel = np.concatenate(picks)
    rng.shuffle(sel)
    return X[sel], y[sel]


def with_context(vecs: np.ndarray, w: int) -> np.ndarray:
    """Temporal context for one scene's contiguous frame sequence: each frame's
    vector joined with the mean of its ±`w` neighbours (clipped at the scene's
    edges) → (n, 2·dim). A peak is a multi-second event, so the window mean lets
    the model see "what's happening around this frame", not one still."""
    v = np.asarray(vecs, dtype=np.float32)
    if w <= 0 or v.shape[0] == 0:
        return v
    n = v.shape[0]
    csum = np.vstack([np.zeros((1, v.shape[1]), dtype=np.float64), np.cumsum(v, axis=0, dtype=np.float64)])
    i = np.arange(n)
    lo, hi = np.clip(i - w, 0, n), np.clip(i + w + 1, 0, n)
    mean = ((csum[hi] - csum[lo]) / (hi - lo)[:, None]).astype(np.float32)
    return np.hstack([v, mean])


class TasteClassifier:
    def __init__(self, kind: str = "logreg", model_name: str = "", profile: str = "",
                 context: int = 0):
        self.kind = kind
        self.model_name = model_name  # which embedder produced the features
        self.profile = profile
        self.context = int(context)   # ±frames of temporal context (0 = single frame)
        self.dim: int | None = None   # feature dim the estimator was fit on
        self._clf = None
        self._pos_index = 1  # column of predict_proba that is the positive class

    # --- build ---------------------------------------------------------------

    def _new_estimator(self):
        if self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=1000, class_weight="balanced")
        if self.kind == "mlp":
            from sklearn.neural_network import MLPClassifier

            # a touch of L2 (alpha) to keep the non-linear model honest
            return MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, alpha=1e-3,
                                 random_state=0)
        raise ValueError(f"unknown classifier kind: {self.kind!r}")

    def train(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "TasteClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("training matrix X must be (n_samples, dim) and non-empty")
        classes = set(np.unique(y).tolist())
        if classes != {0, 1}:
            raise ValueError(
                "need both positive (1) and negative (0) labels to train; "
                f"got classes {sorted(classes)}"
            )
        self._clf = self._new_estimator()
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if sample_weight is not None and self.kind == "mlp":
            # MLPClassifier has no sample_weight → emulate it with a per-class
            # weighted bootstrap (keeps both classes and their balance).
            Xr, yr = _weighted_resample(X, y, sample_weight)
            self._clf.fit(Xr, yr)
        elif sample_weight is not None:
            self._clf.fit(X, y, sample_weight=np.asarray(sample_weight, dtype=np.float64))
        else:
            self._clf.fit(X, y)
        self.dim = X.shape[1]
        self._pos_index = int(np.where(self._clf.classes_ == 1)[0][0])
        return self

    @property
    def fitted(self) -> bool:
        return self._clf is not None

    # --- score ---------------------------------------------------------------

    @property
    def raw_dim(self) -> int | None:
        """The embedder's frame dim (before context features)."""
        if self.dim is None:
            return None
        return self.dim // 2 if self.context else self.dim

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class for each row → (n,) in [0, 1].

        A context model accepts either ready-made context features (2·dim, from
        a caller that knows each row's neighbours) or raw frames, which are then
        read as ONE scene's contiguous sequence (the per-scene scoring path)."""
        if not self.fitted:
            raise RuntimeError("classifier is not trained")
        X = np.asarray(X, dtype=np.float32)
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        if self.context and X.ndim == 2 and X.shape[1] == self.raw_dim:
            X = with_context(X, self.context)
        if self.dim is not None and X.shape[1] != self.dim:
            raise ValueError(
                f"feature dim {X.shape[1]} != classifier dim {self.dim} "
                "(was the cache built with a different embedder?)"
            )
        proba = self._clf.predict_proba(X)[:, self._pos_index]
        return proba.astype(np.float32)

    # --- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # atomic write: a background retrain must never leave a reader (the
        # swipe trainer / rerank) seeing a half-written pickle.
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(
                {
                    "kind": self.kind,
                    "model_name": self.model_name,
                    "profile": self.profile,
                    "dim": self.dim,
                    "pos_index": self._pos_index,
                    "context": self.context,
                    "clf": self._clf,
                },
                fh,
            )
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TasteClassifier":
        """Load a saved model. Note: pickle executes code on load — only load
        model files you created yourself (they live in your local models/)."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls(
            kind=data["kind"],
            model_name=data.get("model_name", ""),
            profile=data.get("profile", ""),
            context=data.get("context", 0),
        )
        obj.dim = data["dim"]
        obj._pos_index = data.get("pos_index", 1)
        obj._clf = data["clf"]
        return obj
