"""Keeper triage: learn the user's tiers from the scenes they've already graded.

Each graded scene becomes one example with two kinds of features:

* **visual** — the mean of the scene's frame embeddings, plus the mean of its
  most on-taste frames (what the scene looks like overall, and at its best),
  compressed with PCA so ~300 examples aren't drowned in 2k dimensions;
* **quality** — log bitrate, bits-per-pixel-per-frame, resolution class, frame
  rate and codec. Learned jointly with the visual side rather than hand-set,
  because high bitrate isn't simply "better": AI upscales (the Upscale tier) are
  often 4K and high-bitrate too, and only the picture tells them apart;
* **who** (optional, kept only when cross-validation says it helps) — the track
  record of the scene's performers and studio in your grades: average grade,
  keep rate and share graded Exceptionnelle or better, each shrunk toward the
  library average so one or two grades count for little (`WhoEncoder`).

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


# --- who: performer & studio track records ---------------------------------

WHO_PRIOR = 3.0        # smoothing weight: a record of n grades is n/(n+3) its own
# performer/studio features must raise CV exact accuracy by more than this to be
# kept: on a few hundred scenes, records that carry nothing still "gain" ~1–2
# points by chance
WHO_MARGIN = 0.02
_TOP = ORDINAL["exceptionnelle"]
WHO_FEATURES: tuple[str, ...] = (
    "perf_grade_max", "perf_grade_mean", "perf_keep_mean", "perf_top_max",
    "perf_graded_log", "perf_known_frac", "no_performers",
    "studio_grade", "studio_keep", "studio_top", "studio_graded_log", "no_studio",
    "who_missing",
)


def who_of(row: dict) -> dict:
    """A catalogue row → the 'who' of a scene: {performers: [stash ids], studio}."""
    return {"performers": [str(p) for p in (row.get("performer_ids") or []) if p],
            "studio": row.get("studio") or ""}


class WhoEncoder:
    """Per-performer (Stash id) and per-studio (name) tallies of your grades,
    turned into features for a scene. `who` entries are {performers, studio}
    dicts, or None when unknown (old remembered rejects) — those neither count
    toward a record nor get one (they get the `who_missing` flag instead)."""

    def __init__(self, prior: float = WHO_PRIOR):
        self.prior = prior
        self.perf: dict[str, list[float]] = {}     # id -> [n, sum_ord, keeps, tops]
        self.studio: dict[str, list[float]] = {}
        self.base = (2.0, 0.5, 0.2)                # library mean grade, keep, top share

    @staticmethod
    def _tally(y: str) -> tuple[float, float, float, float]:
        o = ORDINAL[y]
        return 1.0, float(o), float(o > 0), float(o >= _TOP)

    def fit(self, who: list, y: list[str]) -> "WhoEncoder":
        self.perf, self.studio = {}, {}
        tot = np.zeros(4)
        for w, c in zip(who, y):
            if w is None:
                continue
            t = self._tally(c)
            tot += t
            for pid in set(w.get("performers") or []):
                a = self.perf.setdefault(pid, [0.0, 0.0, 0.0, 0.0])
                for i in range(4):
                    a[i] += t[i]
            st = w.get("studio")
            if st:
                a = self.studio.setdefault(st, [0.0, 0.0, 0.0, 0.0])
                for i in range(4):
                    a[i] += t[i]
        if tot[0]:
            self.base = (tot[1] / tot[0], tot[2] / tot[0], tot[3] / tot[0])
        return self

    def record(self, tally: list[float] | None, own: str | None = None) -> tuple[float, float, float, float]:
        """(n, grade, keep, top) for a tally, shrunk toward the library mean —
        with `own` (a grade) taken back out first, so a scene's own grade never
        feeds its own record."""
        n, so, sk, st = tally or (0.0, 0.0, 0.0, 0.0)
        if own is not None and n:
            d = self._tally(own)
            n, so, sk, st = n - d[0], so - d[1], sk - d[2], st - d[3]
        k = self.prior
        g0, k0, t0 = self.base
        return (n, (so + k * g0) / (n + k), (sk + k * k0) / (n + k), (st + k * t0) / (n + k))

    def encode(self, who: list, own: list | None = None) -> np.ndarray:
        """(n, len(WHO_FEATURES)) features. `own[i]`, if given, is row i's own
        grade (leave-one-out: used when encoding scenes the records were built from)."""
        out = np.zeros((len(who), len(WHO_FEATURES)), dtype=np.float32)
        for i, w in enumerate(who):
            if w is None:
                out[i, -1] = 1.0
                continue
            me = own[i] if own is not None else None
            recs = [self.record(self.perf.get(p), me) for p in dict.fromkeys(w.get("performers") or [])]
            if recs:
                gr = [r[1] for r in recs]
                out[i, 0:7] = [max(gr) / 4, float(np.mean(gr)) / 4, float(np.mean([r[2] for r in recs])),
                               max(r[3] for r in recs), math.log1p(sum(r[0] for r in recs)),
                               sum(1 for r in recs if r[0] > 0) / len(recs), 0.0]
            else:
                g0, k0, t0 = self.base
                out[i, 0:7] = [g0 / 4, g0 / 4, k0, t0, 0.0, 0.0, 1.0]
            st = w.get("studio")
            if st:
                n, g, kp, tp = self.record(self.studio.get(st), me)
                out[i, 7:12] = [g / 4, kp, tp, math.log1p(n), 0.0]
            else:
                g0, k0, t0 = self.base
                out[i, 7:12] = [g0 / 4, k0, t0, 0.0, 1.0]
        return out

    def explain(self, w: dict | None, own: str | None = None) -> dict:
        """Raw (unshrunk) records for display: {performers: [{id, n, keep, top,
        grade}], studio: {...} | None}; n = 0 means no history."""
        def rec(t):
            n, g, k, tp = self.record(t, own)   # n after leave-one-out
            if not n:
                return {"n": 0}
            raw_n, so, sk, st = t
            if own is not None:
                d = self._tally(own)
                so, sk, st = so - d[1], sk - d[2], st - d[3]
            return {"n": int(n), "grade": round(so / n, 2), "keep": round(sk / n, 3),
                    "top": round(st / n, 3), "shrunk": round(g, 3)}
        if not w:
            return {"performers": [], "studio": None}
        perfs = [{"id": p, **rec(self.perf.get(p))} for p in dict.fromkeys(w.get("performers") or [])]
        st = w.get("studio")
        return {"performers": perfs, "studio": {"name": st, **rec(self.studio.get(st))} if st else None}


def _oof_who(who: list, y: list[str], folds: int = 5, seed: int = 0) -> np.ndarray:
    """Training-row who features, each built only from the other folds' grades
    (out-of-fold target encoding: a scene's own grade never shapes its record,
    and — unlike plain leave-one-out — the value doesn't tilt against its label)."""
    n = len(y)
    out = np.zeros((n, len(WHO_FEATURES)), dtype=np.float32)
    k = max(2, min(folds, n))
    order = np.random.default_rng(seed).permutation(n)
    for f in range(k):
        te = order[f::k]
        te_set = set(te.tolist())
        tr = [i for i in range(n) if i not in te_set]
        enc = WhoEncoder().fit([who[i] for i in tr], [y[i] for i in tr])
        out[te] = enc.encode([who[i] for i in te])
    return out


class TierModel:
    """PCA(visual) ⊕ quality ⊕ who → standardize → multinomial logistic regression.
    `use_visual=False` gives the fallback for scenes not embedded yet."""

    def __init__(self, use_quality: bool = True, pca_dim: int = 48, C: float = 0.5,
                 use_who: bool = False, use_visual: bool = True):
        self.use_quality = use_quality
        self.use_who = use_who
        self.use_visual = use_visual
        self.pca_dim = pca_dim
        self.C = C
        self.classes_: list[str] = []
        self.who_enc: WhoEncoder | None = None
        self._pca = self._scaler = self._clf = None

    def _design(self, Xv, Xq, Xw, fit: bool) -> np.ndarray:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        parts = []
        if getattr(self, "use_visual", True):
            if fit:
                k = max(1, min(self.pca_dim, Xv.shape[0] - 1, Xv.shape[1]))
                self._pca = PCA(n_components=k, random_state=0).fit(Xv)
            parts.append(self._pca.transform(Xv))
        if self.use_quality:
            parts.append(Xq)
        if getattr(self, "use_who", False):
            parts.append(Xw)
        X = np.hstack(parts)
        if fit:
            self._scaler = StandardScaler().fit(X)
        return self._scaler.transform(X)

    @staticmethod
    def _arr(X):
        return None if X is None else np.asarray(X, np.float32)

    def fit(self, Xv, Xq, y: list[str], who: list | None = None) -> "TierModel":
        from sklearn.linear_model import LogisticRegression

        Xw = None
        if self.use_who:
            if who is None:
                raise ValueError("use_who needs each scene's performers/studio")
            self.who_enc = WhoEncoder().fit(who, list(y))
            Xw = _oof_who(who, list(y))
        X = self._design(self._arr(Xv), self._arr(Xq), Xw, fit=True)
        self._clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000)
        self._clf.fit(X, np.asarray(y))
        self.classes_ = [str(c) for c in self._clf.classes_]
        return self

    def predict_proba(self, Xv, Xq, who: list | None = None, own: list | None = None) -> np.ndarray:
        """`own[i]`: scene i's current grade when it's one the records were
        built from (so its own grade is left out of its encoding), else None."""
        Xw = None
        if getattr(self, "use_who", False):
            n = len(Xq)
            Xw = self.who_enc.encode(who if who is not None else [None] * n, own)
        X = self._design(self._arr(Xv), self._arr(Xq), Xw, fit=False)
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
                   folds: int = 5, pca_dim: int = 48, use_who: bool = False,
                   who: list | None = None, use_visual: bool = True) -> dict:
    """Stratified k-fold: exact accuracy and 'within one tier' accuracy (the
    predicted tier is at most one step from the real one on the ordinal).
    Performer/studio records are rebuilt inside each fold from its training
    part only — the held-out scenes' grades are never in them."""
    from sklearn.model_selection import StratifiedKFold

    y_arr = np.asarray(y)
    k = max(2, min(folds, min(np.unique(y_arr, return_counts=True)[1])))
    exact = near = 0
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=0).split(Xq, y_arr):
        m = TierModel(use_quality=use_quality, pca_dim=pca_dim, use_who=use_who,
                      use_visual=use_visual)
        m.fit(None if Xv is None else Xv[tr], Xq[tr], list(y_arr[tr]),
              who=[who[i] for i in tr] if use_who else None)
        proba = m.predict_proba(None if Xv is None else Xv[te], Xq[te],
                                who=[who[i] for i in te] if use_who else None)
        pred = [m.classes_[int(i)] for i in np.argmax(proba, axis=1)]
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
    """A one-line, plain-words reason when a scene falls under the learned floor,
    else None. `quality_flag_detail` gives the figures behind it."""
    res, mbps = q.get("res"), q.get("mbps")
    low = floor.get("lowest_res")
    if res and low and RES_CLASSES.index(res) < RES_CLASSES.index(low):
        return f"{res} — you've never tiered anything below {low}"
    f = floor.get("by_res", {}).get(res or "")
    if f and mbps and mbps < f["mbps"]:
        return f"lower quality than every {res} scene you've tiered"
    return None


def quality_flag_detail(q: dict, floor: dict) -> str | None:
    """The numbers behind `quality_flag` (shown small beside the words)."""
    res, mbps = q.get("res"), q.get("mbps")
    f = floor.get("by_res", {}).get(res or "")
    if f and mbps and mbps < f["mbps"] and quality_flag(q, floor):
        return f"{mbps:g} Mbps · lowest you've tiered {f['mbps']:g} Mbps, of {f['n']}"
    return None


PCA_GRID: tuple[int, ...] = (8, 16, 32, 48)


def fit_best(Xv: np.ndarray, Xq: np.ndarray, y: list[str],
             who: list | None = None) -> tuple[TierModel, dict]:
    """Pick the visual PCA size by cross-validation, then fit on everything.

    The right size depends on how many scenes are graded and how much the
    picture carries: on ~120 examples, 48 components of mostly-noise visual
    dims swamped a real bitrate signal (37% vs 89% at 8 components in the
    synthetic test), while a library whose tiers really live in the picture
    wants more. Candidates are capped at n/4 so tiny sets stay small. The CV
    score of the chosen size is reported (mildly optimistic: it was also
    used to choose among a handful of sizes). Also reports the same size
    WITHOUT quality features, so the value of bitrate is measured, not assumed.

    With `who` (each scene's performers/studio, None where unknown), the
    performer/studio records are cross-validated too and kept only when they
    raise exact accuracy (`who_gain`) by more than WHO_MARGIN — never assumed
    to help.
    """
    n = len(y)
    grid = [k for k in PCA_GRID if k <= max(PCA_GRID[0], n // 4)] or [PCA_GRID[0]]
    best_k, best = grid[0], None
    for k in grid:
        cv = cross_validate(Xv, Xq, y, use_quality=True, pca_dim=k)
        if best is None or (cv["exact"], cv["within_one"]) > (best["exact"], best["within_one"]):
            best_k, best = k, cv
    without = cross_validate(Xv, Xq, y, use_quality=False, pca_dim=best_k)
    rep = {"pca_dim": best_k, "cv": best, "cv_without_quality": without,
           "quality_gain": round(best["exact"] - without["exact"], 3)}
    use_who = False
    if who is not None and any(w for w in who):
        with_who = cross_validate(Xv, Xq, y, use_quality=True, pca_dim=best_k, use_who=True, who=who)
        gain = round(with_who["exact"] - best["exact"], 3)
        use_who = gain > WHO_MARGIN
        rep.update({"cv_with_who": with_who, "who_gain": gain, "use_who": use_who})
        if use_who:
            rep["cv_without_who"] = best
            rep["cv"] = with_who
    model = TierModel(use_quality=True, pca_dim=best_k, use_who=use_who)
    model.fit(Xv, Xq, y, who=who if use_who else None)
    return model, rep


def fit_fallback(Xq: np.ndarray, y: list[str], who: list) -> tuple[TierModel | None, dict]:
    """The performer/studio + file-quality model for scenes that aren't embedded
    yet (so there's no picture to judge). Trained on every graded scene with
    known performers/studio, embedded or not. Returns (None, report) when the
    records don't beat a quality-only guess — then new scenes simply wait."""
    ok = [i for i, w in enumerate(who) if w is not None]
    if len(ok) < 2 * MIN_PER_CLASS:
        return None, {"trained": False, "n": len(ok)}
    Xq2, y2, w2 = Xq[ok], [y[i] for i in ok], [who[i] for i in ok]
    classes, _ = usable_classes(y2)
    keep = [i for i, c in enumerate(y2) if c in classes]
    if len(classes) < 2:
        return None, {"trained": False, "n": len(ok)}
    Xq2, y2, w2 = Xq2[keep], [y2[i] for i in keep], [w2[i] for i in keep]
    with_who = cross_validate(None, Xq2, y2, use_quality=True, use_who=True, who=w2, use_visual=False)
    base = cross_validate(None, Xq2, y2, use_quality=True, use_visual=False)
    rep = {"n": len(y2), "cv": with_who, "cv_quality_only": base,
           "who_gain": round(with_who["exact"] - base["exact"], 3)}
    if rep["who_gain"] <= WHO_MARGIN:
        return None, {"trained": False, **rep}
    m = TierModel(use_quality=True, use_who=True, use_visual=False).fit(None, Xq2, y2, who=w2)
    return m, {"trained": True, **rep}
