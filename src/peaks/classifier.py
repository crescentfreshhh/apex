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


class TasteClassifier:
    def __init__(self, kind: str = "logreg", model_name: str = "", profile: str = ""):
        self.kind = kind
        self.model_name = model_name  # which embedder produced the features
        self.profile = profile
        self.dim: int | None = None
        self._clf = None
        self._pos_index = 1  # column of predict_proba that is the positive class

    # --- build ---------------------------------------------------------------

    def _new_estimator(self):
        if self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=1000, class_weight="balanced")
        if self.kind == "mlp":
            from sklearn.neural_network import MLPClassifier

            return MLPClassifier(hidden_layer_sizes=(128,), max_iter=500)
        raise ValueError(f"unknown classifier kind: {self.kind!r}")

    def train(self, X: np.ndarray, y: np.ndarray) -> "TasteClassifier":
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
        self._clf.fit(X, y)
        self.dim = X.shape[1]
        self._pos_index = int(np.where(self._clf.classes_ == 1)[0][0])
        return self

    @property
    def fitted(self) -> bool:
        return self._clf is not None

    # --- score ---------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class for each row → (n,) in [0, 1]."""
        if not self.fitted:
            raise RuntimeError("classifier is not trained")
        X = np.asarray(X, dtype=np.float32)
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
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
        )
        obj.dim = data["dim"]
        obj._pos_index = data.get("pos_index", 1)
        obj._clf = data["clf"]
        return obj
