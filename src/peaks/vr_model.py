"""The VR grade model: Reject / Exceptionnelle (4★) / Légendaire (5★), from what
can be known without the headset.

VR is graded on its picture first, so the model starts from the file:
- **file** — the picture measure of one eye at full resolution (real detail,
  upscale factor, sharpness, noise, blockiness, banding) and the file facts
  (bitrate for its size, frame rate, codec, bit depth);
- **studio** — how that studio's VR has turned out for you, and how far this
  file's year is from the years you've kept of theirs (studios change cameras);
- **who** — how the performers' VR has turned out.

Studio and performer records are worked out from the training scenes only
(and leave each scene's own grade out), and each group is kept only when the
honest cross-validated accuracy says it adds — so if the file alone predicts
your VR grades best, the file alone is what it uses. Examples are your graded
VR scenes plus remembered VR rejects (deleted files: their measures, studio,
year and performers were kept). It switches on once every grade has
MIN_PER_CLASS examples. Only ever suggestions.
"""

from __future__ import annotations

import math

import numpy as np

CLASSES = ("reject", "exceptionnelle", "legendaire")
ORD = {"reject": 0, "exceptionnelle": 1, "legendaire": 2}
MIN_PER_CLASS = 8
MARGIN = 0.02                   # a richer set must beat a simpler one by this much (exact accuracy)
PRIOR = 3.0                     # pseudo-scenes pulling a small record toward the library's rates
SETS = (("file",), ("file", "studio"), ("file", "studio", "who"))
MODERN = ("hevc", "h265", "av1", "vp9")


def label_of(tier: str | None) -> str | None:
    return {"rejected": "reject", "exceptionnelle": "exceptionnelle", "legendaire": "legendaire"}.get(tier or "")


def example(row: dict, pic: dict | None, label: str | None = None) -> dict:
    """What the model knows of one VR scene (a listing row + its picture record),
    small enough to keep after the file is deleted."""
    d = str(row.get("date") or "")[:4]
    keep = ("eff_h", "eff_factor", "sharp", "noise", "block", "band", "bit_depth")
    q = row.get("quality") or {}
    return {"y": label if label is not None else label_of(row.get("tier")),
            "studio": (row.get("studio") or "").strip().lower(),
            "year": int(d) if d.isdigit() else None,
            "performers": [str(p) for p in row.get("performer_ids") or []],
            "q": {k: q.get(k) for k in ("mbps", "bpp", "fps", "codec", "w", "h")},
            "pic": {k: (pic or {}).get(k) for k in keep} if pic and not pic.get("error") else None}


# --- features -------------------------------------------------------------------------

def _f(x, scale=1.0) -> float:
    try:
        v = float(x)
        return 0.0 if math.isnan(v) else v / scale
    except (TypeError, ValueError):
        return 0.0


def file_features(ex: dict) -> list[float]:
    p, q = ex.get("pic") or {}, ex.get("q") or {}
    has_pic = 1.0 if p.get("eff_h") else 0.0
    bpp = q.get("bpp")
    if not bpp and q.get("mbps") and q.get("fps") and q.get("w") and q.get("h"):
        bpp = q["mbps"] * 1e6 / (q["fps"] * q["w"] * q["h"])
    return [has_pic, _f(p.get("eff_h"), 4096), math.log(max(1.0, _f(p.get("eff_factor")) or 1.0)),
            _f(p.get("sharp")), _f(p.get("noise")), _f(p.get("block")), _f(p.get("band")),
            1.0 if bpp else 0.0, _f(bpp, 0.1), _f(q.get("mbps"), 100), _f(q.get("fps"), 60),
            1.0 if str(q.get("codec") or "").lower() in MODERN else 0.0,
            1.0 if _f(p.get("bit_depth")) > 8 else 0.0]


class Records:
    """Studio and performer tallies of VR grades, from the training scenes only."""

    def fit(self, exs: list[dict]) -> "Records":
        self.studio: dict[str, list] = {}
        self.perf: dict[str, list] = {}
        n = len(exs) or 1
        self.base = (sum(e["y"] != "reject" for e in exs) / n, sum(e["y"] == "legendaire" for e in exs) / n)
        for e in exs:
            k, t = float(e["y"] != "reject"), float(e["y"] == "legendaire")
            if e["studio"]:
                s = self.studio.setdefault(e["studio"], [0.0, 0.0, 0.0, []])
                s[0] += 1
                s[1] += k
                s[2] += t
                if k and e["year"]:
                    s[3].append(e["year"])
            for pid in set(e["performers"]):
                p = self.perf.setdefault(pid, [0.0, 0.0, 0.0])
                p[0] += 1
                p[1] += k
                p[2] += t
        return self

    def _rate(self, tally, own: dict | None) -> tuple[float, float, float]:
        n, k, t = tally[0], tally[1], tally[2]
        if own is not None:                       # a training scene: its own grade left out
            n, k, t = n - 1, k - float(own["y"] != "reject"), t - float(own["y"] == "legendaire")
        kr = (k + PRIOR * self.base[0]) / (n + PRIOR)
        tr = (t + PRIOR * self.base[1]) / (n + PRIOR)
        return kr, tr, math.log1p(max(0.0, n))

    def studio_features(self, ex: dict, own: bool) -> list[float]:
        s = self.studio.get(ex["studio"]) if ex["studio"] else None
        if s is None or (own and s[0] <= 1):
            return [0.0, self.base[0], self.base[1], 0.0, 0.0, 0.0]
        kr, tr, n = self._rate(s, ex if own else None)
        years = list(s[3])
        if own and ex["y"] != "reject" and ex["year"] in years:
            years.remove(ex["year"])
        gap = ((ex["year"] - float(np.median(years))) / 5.0) if (years and ex["year"]) else 0.0
        return [1.0, kr, tr, n, gap, 1.0 if (years and ex["year"]) else 0.0]

    def who_features(self, ex: dict, own: bool) -> list[float]:
        rates = [self._rate(self.perf[p], ex if own else None) for p in set(ex["performers"])
                 if p in self.perf and not (own and self.perf[p][0] <= 1)]
        if not rates:
            return [0.0, self.base[0], self.base[1], 0.0]
        return [1.0, float(np.mean([r[0] for r in rates])), float(np.mean([r[1] for r in rates])),
                float(np.mean([r[2] for r in rates]))]


def design(exs: list[dict], groups: tuple[str, ...], rec: Records, own: bool) -> np.ndarray:
    rows = []
    for e in exs:
        x = []
        if "file" in groups:
            x += file_features(e)
        if "studio" in groups:
            x += rec.studio_features(e, own)
        if "who" in groups:
            x += rec.who_features(e, own)
        rows.append(x)
    return np.asarray(rows, dtype=np.float64)


# --- fitting ------------------------------------------------------------------------------

def _fit(exs, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rec = Records().fit(exs)
    X = design(exs, groups, rec, own=True)
    y = [e["y"] for e in exs]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
    return rec, sc, clf


def cross_validate(exs: list[dict], groups: tuple[str, ...], seed: int = 0) -> float:
    """Exact accuracy, each scene predicted by a model (and records) that never saw it."""
    from sklearn.model_selection import StratifiedKFold

    y = np.array([e["y"] for e in exs])
    k = max(2, min(5, min(int((y == c).sum()) for c in CLASSES)))
    hits = 0
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        train = [exs[i] for i in tr]
        rec, sc, clf = _fit(train, groups)
        pred = clf.predict(sc.transform(design([exs[i] for i in te], groups, rec, own=False)))
        hits += int((pred == y[te]).sum())
    return round(hits / len(y), 3)


class VRModel:
    def __init__(self, groups, rec, sc, clf, report):
        self.groups, self.rec, self.sc, self.clf, self.report = groups, rec, sc, clf, report

    def predict(self, exs: list[dict]) -> list[dict]:
        if not exs:
            return []
        P = self.clf.predict_proba(self.sc.transform(design(exs, self.groups, self.rec, own=False)))
        classes = [str(c) for c in self.clf.classes_]
        ords = np.array([ORD[c] for c in classes], dtype=np.float64)
        rej = classes.index("reject")
        out = []
        for p in P:
            j = int(np.argmax(p))
            out.append({"tier": classes[j], "conf": round(float(p[j]), 3), "expected": round(float(p @ ords), 3),
                        "keeper": round(1.0 - float(p[rej]), 3),
                        "probs": {c: round(float(v), 3) for c, v in zip(classes, p)}, "from": "vr"})
        return out


def train(exs: list[dict]) -> tuple[VRModel | None, dict]:
    """Fit the VR model, choosing its feature groups by honest cross-validation.
    Returns (model or None, report)."""
    exs = [e for e in exs if e.get("y") in CLASSES]
    counts = {c: sum(e["y"] == c for e in exs) for c in CLASSES}
    rep = {"counts": counts, "n": len(exs), "min_per_class": MIN_PER_CLASS}
    short = {c: MIN_PER_CLASS - n for c, n in counts.items() if n < MIN_PER_CLASS}
    if short:
        rep.update(trained=False, needs=short)
        return None, rep
    cv = {"+".join(g): cross_validate(exs, g) for g in SETS}
    chosen = SETS[0]
    for g in SETS[1:]:
        if cv["+".join(g)] >= cv["+".join(chosen)] + MARGIN:
            chosen = g
    rec, sc, clf = _fit(exs, chosen)
    baseline = round(max(counts.values()) / len(exs), 3)
    rep.update(trained=True, cv=cv, uses="+".join(chosen), accuracy=cv["+".join(chosen)], baseline=baseline)
    return VRModel(chosen, rec, sc, clf, rep), rep
