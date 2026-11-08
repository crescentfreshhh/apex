"""Quick guesses between retrains: the tier model's last layer, nudged by each grade.

The honest tier model is a full cross-validated retrain — every 25 grades (10
while you settle), tens of fits. Between two retrains a grade changed nothing
about the other scenes' suggestions. Here, after each grade, a copy of the
model's final (multinomial logistic) layer takes a few small gradient steps on
that one scene's feature row — the features themselves stay as the honest
training left them — and the ungraded scenes are re-scored in one matrix
product. An L2 pull back toward the honest weights keeps it from drifting; the
next honest retrain replaces it outright. Guesses it changed carry
`quick: True`, so the pages can say so and nothing honest-only (settle
calibration, the Arbiter, accuracy figures) ever reads them.
"""

from __future__ import annotations

import numpy as np

STEPS = 4
RATE = 0.3
ANCHOR = 0.05        # pull toward the honest weights, per step


class QuickHead:
    def __init__(self, model):
        clf = model._clf
        self.model = model
        self.classes = list(model.classes_)
        coef = np.asarray(clf.coef_, dtype=np.float64)
        icpt = np.asarray(clf.intercept_, dtype=np.float64)
        if len(self.classes) == 2:            # binary: log-odds of classes[1] → a 2-row softmax
            coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
            icpt = np.array([0.0, icpt[0]])
        self.W0, self.b0 = coef.copy(), icpt.copy()
        self.W, self.b = coef, icpt
        self.updates = 0

    def proba(self, X: np.ndarray) -> np.ndarray:
        z = np.asarray(X, dtype=np.float64) @ self.W.T + self.b
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def learn(self, x: np.ndarray, cls: str) -> bool:
        """A few small steps toward `cls` for feature row `x`."""
        if cls not in self.classes:
            return False
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.zeros(len(self.classes))
        y[self.classes.index(cls)] = 1.0
        for _ in range(STEPS):
            p = self.proba(x[None, :])[0]
            g = p - y
            self.W -= RATE * (np.outer(g, x) / max(1.0, float(x @ x) ** 0.5) + ANCHOR * (self.W - self.W0))
            self.b -= RATE * (g + ANCHOR * (self.b - self.b0))
        self.updates += 1
        return True
