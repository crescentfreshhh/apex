"""Picture quality: what a file actually looks like, not what its bitrate says.

The embeddings are computed from ~224 px frames, where detail and compression
damage are gone, so this is a separate light pass: a few stills per file,
decoded by ffmpeg at **full resolution** (one eye of a side-by-side VR file),
measured with numpy/scipy:

- effective resolution — where the picture's detail runs out. Real footage
  keeps (roughly) the same share of fine detail at every scale up to its pixel
  size; an upscale has none past the resolution it came from, so a "4K" file
  made from 1080p reads as "real detail ≈ 1080p";
- sharpness, noise, blockiness (jumps on the 8-px coding grid), banding
  (smooth gradients that step instead of blend), exposure, letterboxing,
  interlacing (comb pattern) and duplicated frames;
- plus file facts from ffprobe: bit depth, HDR, audio bitrate.

`measure(path)` returns one record of those numbers. What they mean for *you*
is learned elsewhere (`PictureModel`, from your "which looks better?" picks).
"""

from __future__ import annotations

import json
import math
import subprocess

import numpy as np

VERSION = 1                     # bump when a metric changes: old records are re-measured
N_STILLS = 8
EFF_FIT = (0.02, 0.09)          # cycles/px: the band surely carrying real detail (fits the expected spectrum)
EFF_DROP = 0.3                  # detail ends where the spectrum falls below this share of the expected
EFF_SLOPE = (-3.2, -1.2)        # natural footage falls off within this range (log power / log frequency)
EFF_MIN_SIDE = 480              # smaller frames hold too little fine detail above codec noise to judge
CROP = 1024                     # metrics that are local run on a central crop this big


# --- reading the file -----------------------------------------------------------------

def probe(path: str, ffprobe: str = "ffprobe", timeout: float = 30.0) -> dict:
    """Size, running time and file facts of the first video / audio stream."""
    r = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                       capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or b"").decode(errors="replace").strip()[-300:] or "ffprobe failed")
    d = json.loads(r.stdout or b"{}")
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"
              and not (s.get("disposition") or {}).get("attached_pic")), None)
    if v is None:
        raise RuntimeError("no video stream")
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), None)
    pix = v.get("pix_fmt") or ""
    depth = 12 if "12" in pix else 10 if ("10" in pix or "p010" in pix) else 8
    dur = _num(v.get("duration")) or _num((d.get("format") or {}).get("duration")) or 0.0
    return {"w": int(v.get("width") or 0), "h": int(v.get("height") or 0), "duration": dur,
            "codec": v.get("codec_name"), "bit_depth": depth,
            "hdr": (v.get("color_transfer") or "") in ("smpte2084", "arib-std-b67"),
            "audio_kbps": round(_num((a or {}).get("bit_rate")) / 1000) if a and _num(a.get("bit_rate")) else None}


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def is_side_by_side(w: int, h: int) -> bool:
    return bool(w and h and 1.95 <= w / h <= 2.05)


def read_stills(path: str, info: dict, n: int = N_STILLS, ffmpeg: str = "ffmpeg", timeout: float = 90.0):
    """Yield (still, the frame after it) as grey uint8 arrays at full resolution,
    spread over 10–90 % of the running time; one eye of a side-by-side file."""
    w, h, dur = info["w"], info["h"], info["duration"]
    if not (w and h):
        return
    vf = "format=gray"
    if is_side_by_side(w, h):
        w = w // 2
        vf = f"crop={w}:{h}:0:0," + vf
    size = w * h
    for i in range(n):
        t = dur * (0.1 + 0.8 * i / max(1, n - 1)) if dur else 0.0
        r = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", path, "-an", "-sn",
                            "-frames:v", "2", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                           capture_output=True, timeout=timeout)
        buf = r.stdout or b""
        if len(buf) < size:
            continue
        a = np.frombuffer(buf[:size], dtype=np.uint8).reshape(h, w)
        b = np.frombuffer(buf[size:2 * size], dtype=np.uint8).reshape(h, w) if len(buf) >= 2 * size else None
        yield a, b


# --- the metrics (one grey frame at a time) -----------------------------------------------

def _radial_power(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum averaged over rings: (cycles per pixel, power)."""
    n = min(y.shape)
    y = y[:n, :n] - y[:n, :n].mean()
    win = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    f = np.abs(np.fft.fftshift(np.fft.fft2(y * win))) ** 2
    c = n // 2
    yy, xx = np.indices(f.shape)
    r = np.hypot(yy - c, xx - c) / n
    bins = np.linspace(0, 0.5, 101)
    idx = np.digitize(r.ravel(), bins)
    total = np.bincount(idx, f.ravel(), minlength=102)[1:101]
    count = np.maximum(1, np.bincount(idx, minlength=102)[1:101])
    return (bins[:-1] + bins[1:]) / 2, total / count


def effective_factor(y: np.ndarray) -> float:
    """How many times larger the frame is than its real detail: 1 = every pixel
    earns its keep, 2 = it holds about half its resolution's worth (a 2× upscale).
    Natural footage has power falling off smoothly with frequency; fit that
    trend where detail is surely real, then find where the actual spectrum drops
    well below it."""
    if float(y.std()) < 3.0 or min(y.shape) < EFF_MIN_SIDE:   # flat, or too small to tell
        return float("nan")
    r, p = _radial_power(y)
    m = (r >= EFF_FIT[0]) & (r <= EFF_FIT[1]) & (p > 0)
    if m.sum() < 5:
        return float("nan")
    slope, icpt = np.polyfit(np.log(r[m]), np.log(p[m]), 1)
    held = float(np.clip(slope, *EFF_SLOPE))
    if held != slope:           # big smooth areas (walls, skin) bend the fit: hold a natural slope
        icpt = float(np.mean(np.log(p[m]) - held * np.log(r[m])))
    rel = p / np.exp(icpt + held * np.log(r))
    cut = EFF_FIT[1]
    for ri, q in zip(r, rel):
        if ri > EFF_FIT[1] and q > EFF_DROP:
            cut = float(ri)
    return round(max(1.0, 0.5 / cut), 3)


def blockiness(y: np.ndarray) -> float:
    """Jumps across the 8-px coding grid versus mid-block (1 = none). Compared
    with the column half a block away — same parity, so an upscale's
    every-other-pixel pattern doesn't read as blocks."""
    out = []
    for d in (np.abs(np.diff(y, axis=1)), np.abs(np.diff(y, axis=0)).T):
        x = np.arange(d.shape[1]) % 8
        off = float(d[:, x == 3].mean())
        if off > 0.5:
            out.append(float(d[:, x == 7].mean()) / off)
    return float(np.mean(out)) if out else float("nan")


def sharpness(y: np.ndarray) -> float:
    """Edge energy relative to contrast (scale-free)."""
    from scipy import ndimage

    return float(np.mean(np.abs(ndimage.laplace(y)))) / (float(y.std()) + 1.0)


def noise(y: np.ndarray) -> float:
    """Grain (σ in grey levels), measured in the frame's least-textured areas
    (Immerkær's estimator: a kernel that cancels smooth shading)."""
    from scipy import ndimage

    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = np.abs(ndimage.convolve(y, k))
    g = np.hypot(*np.gradient(ndimage.gaussian_filter(y, 2.0)))
    flat = g <= np.percentile(g, 15)
    return float(np.sqrt(np.pi / 2) * resp[flat].mean() / 6)


def banding(y: np.ndarray) -> float:
    """Share of the steps in smooth gradients that jump 2+ grey levels at once
    (0 = smooth or dithered, 1 = staircase)."""
    from scipy import ndimage

    yi = np.rint(y).astype(np.int16)
    smooth = ndimage.uniform_filter(np.abs(np.diff(y, axis=1, append=y[:, -1:])), 9) < 1.5
    d = np.diff(yi, axis=1)
    m = smooth[:, :-1]
    steps = (d != 0) & m
    n = int(steps.sum())
    if n < 200:
        return 0.0
    density = n / max(1, int(m.sum()))
    big = float(((np.abs(d) >= 2) & steps).sum()) / n
    return big * max(0.0, 1.0 - density / 0.3)


def combing(y: np.ndarray) -> float:
    """Interlace comb: rows that disagree with both neighbours, beyond what
    rows two apart (the same field) do."""
    a, b, c = y[:-2], y[1:-1], y[2:]
    near = float((((b - a) * (b - c)) > 400).mean())
    a2, c2 = y[:-4], y[4:]
    far = float((((y[2:-2] - a2) * (y[2:-2] - c2)) > 400).mean())
    return max(0.0, near - far)


def active_box(frames: list[np.ndarray]) -> tuple[int, int, int, int]:
    """The picture inside black bars: (x0, y0, x1, y1), from every still (a dark
    scene alone isn't a bar)."""
    h, w = frames[0].shape
    rows = np.zeros(h, bool)
    cols = np.zeros(w, bool)
    for f in frames:
        rows |= (f.mean(axis=1) > 20) | (f.std(axis=1) > 6)
        cols |= (f.mean(axis=0) > 20) | (f.std(axis=0) > 6)
    ry, cx = np.flatnonzero(rows), np.flatnonzero(cols)
    if not len(ry) or not len(cx):
        return 0, 0, w, h
    return int(cx[0]), int(ry[0]), int(cx[-1]) + 1, int(ry[-1]) + 1


def _crop(y: np.ndarray, size: int = CROP) -> np.ndarray:
    h, w = y.shape
    ch, cw = min(h, size), min(w, size)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return y[y0:y0 + ch, x0:x0 + cw]


def measure_frames(pairs: list[tuple[np.ndarray, np.ndarray | None]]) -> dict:
    """The metrics over a file's stills (medians, so one odd still doesn't decide)."""
    if not pairs:
        raise ValueError("no stills")
    stills = [a for a, _ in pairs]
    h, w = stills[0].shape
    x0, y0, x1, y1 = active_box(stills)
    per: dict[str, list[float]] = {k: [] for k in ("eff", "block", "sharp", "noise", "band", "comb",
                                                     "p1", "p99", "mean")}
    frozen = 0
    for a, b in pairs:
        full = a.astype(np.float32)
        per["block"].append(blockiness(full))                    # on the coding grid: uncropped
        pic = full[y0:y1, x0:x1]
        c = _crop(pic)
        per["eff"].append(effective_factor(c))
        per["sharp"].append(sharpness(c))
        per["noise"].append(noise(c))
        per["band"].append(banding(c))
        per["comb"].append(combing(c))
        lo, hi = np.percentile(pic, [1, 99])
        per["p1"].append(float(lo))
        per["p99"].append(float(hi))
        per["mean"].append(float(pic.mean()))
        if b is not None and float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))) < 0.15:
            frozen += 1

    def med(k):
        v = [x for x in per[k] if not math.isnan(x)]
        return round(float(np.median(v)), 4) if v else None

    eff = med("eff")
    ah = y1 - y0
    return {"v": VERSION, "n": len(pairs), "w": w, "h": h,
            "active": round((x1 - x0) * (y1 - y0) / float(w * h), 3), "box": [x0, y0, x1, y1],
            "eff_factor": eff, "eff_h": round(ah / eff) if eff else None,
            "block": med("block"), "sharp": med("sharp"), "noise": med("noise"), "band": med("band"),
            "comb": med("comb"), "p1": med("p1"), "p99": med("p99"), "mean": med("mean"),
            "frozen": round(frozen / len(pairs), 3)}


def stills_for_people(path: str, rec: dict, out_full, out_crop, ffmpeg: str = "ffmpeg",
                      at: float = 0.5, duration: float = 0.0, timeout: float = 90.0) -> None:
    """The two JPEGs the comparison screen shows: the whole frame (960 px wide)
    and a 1:1 crop of real picture from its middle (quality differences vanish
    when a frame is shrunk, so the crop is never scaled)."""
    x0, y0, x1, y1 = rec.get("box") or [0, 0, rec["w"], rec["h"]]
    eye = f"crop={rec['w']}:{rec['h']}:0:0," if rec.get("vr") else ""
    cw, ch = min(640, x1 - x0), min(640, y1 - y0)
    cx, cy = x0 + (x1 - x0 - cw) // 2, y0 + (y1 - y0 - ch) // 2
    t = f"{max(0.0, duration * at):.3f}"
    for out, vf in ((out_full, eye + "scale=960:-2"), (out_crop, eye + f"crop={cw}:{ch}:{cx}:{cy}")):
        subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-ss", t, "-i", path, "-an", "-sn",
                        "-frames:v", "1", "-vf", vf, "-q:v", "3", str(out)], capture_output=True, timeout=timeout)


def measure(path: str, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", n: int = N_STILLS) -> dict:
    info = probe(path, ffprobe)
    pairs = list(read_stills(path, info, n=n, ffmpeg=ffmpeg))
    if not pairs:
        raise RuntimeError("couldn't decode any stills")
    rec = measure_frames(pairs)
    rec.update(bit_depth=info["bit_depth"], hdr=info["hdr"], audio_kbps=info["audio_kbps"],
               codec=info["codec"], vr=is_side_by_side(info["w"], info["h"]), duration=info["duration"])
    return rec


# --- what it means to you: features, the pairwise model, reasons in words ---------------

FEATURES = ("real_detail", "eff_h", "sharp", "noise", "block", "band", "comb", "frozen", "active",
            "contrast", "dark", "mbps", "bpp", "fps", "modern_codec", "deep_color", "hdr", "audio")


def features(rec: dict, q: dict | None = None) -> np.ndarray:
    """A file's measures (+ Stash's file facts `q`, from tiers.quality_of) as a
    fixed-length vector. Missing values sit at a neutral 0 — the model learns
    from differences, so a gap neither helps nor hurts a file."""
    q = q or {}
    eff = rec.get("eff_factor") or 1.0

    def g(k, d=0.0):
        v = rec.get(k)
        return float(v) if v is not None else d

    p1, p99 = g("p1", 16.0), g("p99", 235.0)
    codec = (rec.get("codec") or q.get("codec") or "").lower()
    return np.asarray([
        1.0 / max(1.0, eff),                                  # 1 = all the pixels are real detail
        math.log(max(120.0, g("eff_h", 720.0))),
        math.log1p(g("sharp")),
        math.log1p(g("noise")),
        min(3.0, g("block", 1.0)) - 1.0,
        g("band"), g("comb"), g("frozen"), g("active", 1.0),
        (p99 - p1) / 255.0,
        1.0 if g("mean", 100.0) < 35 else 0.0,
        math.log1p(q.get("mbps") or 0.0),
        min(float(q.get("bpp") or 0.0), 1.0),
        min(float(q.get("fps") or 0.0), 120.0) / 60.0,
        1.0 if codec in ("hevc", "h265", "av1", "vp9") else 0.0,
        1.0 if g("bit_depth", 8) > 8 else 0.0,
        1.0 if rec.get("hdr") else 0.0,
        math.log1p(g("audio_kbps")),
    ], dtype=np.float32)


class PictureModel:
    """Your bar for picture quality, learned from "which looks better?" picks.

    Pairwise logistic regression (Bradley–Terry) on standardised feature
    differences: P(a looks better than b) = σ(w·(za − zb)). A file's score is
    w·z, shown as a 0–100 percentile across the measured library. "Both fine" /
    "Both bad" answers set the acceptable line. It's only trusted once its
    cross-validated agreement with your picks beats bitrate alone."""

    MIN_PAIRS = 25
    MARGIN = 0.03                      # must beat the bitrate-only baseline by this much

    def __init__(self):
        self.w = None
        self.mu = None
        self.sd = None
        self.lib = np.zeros(0, np.float32)
        self.line = None
        self.report: dict = {"trained": False}

    @property
    def trained(self) -> bool:
        return self.w is not None

    @property
    def trusted(self) -> bool:
        return bool(self.report.get("trusted"))

    def _z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd

    @staticmethod
    def _fit(dx: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression

        # symmetric: every pick also teaches the reverse, so there's no left/right bias
        X = np.vstack([dx, -dx])
        Y = np.concatenate([y, 1 - y])
        m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=500)
        m.fit(X, Y)
        return m.coef_[0]

    @staticmethod
    def _agree(dx: np.ndarray, y: np.ndarray, cols: list[int] | None, folds: int = 5) -> float:
        """Cross-validated share of picks the model gets right."""
        rng = np.random.default_rng(0)
        order = rng.permutation(len(y))
        hits = 0
        for k in range(folds):
            test = order[k::folds]
            train = np.setdiff1d(order, test)
            if not len(test) or len(set(y[train])) < 2:
                continue
            d = dx if cols is None else dx[:, cols]
            w = PictureModel._fit(d[train], y[train])
            hits += int(((d[test] @ w > 0).astype(int) == y[test]).sum())
        return hits / max(1, len(y))

    def train(self, feats: dict[str, np.ndarray], picks: list[dict]) -> dict:
        """`feats`: file key → feature vector (every measured file);
        `picks`: [{a, b, ans: "a"|"b"|"fine"|"bad"}]."""
        keys = list(feats)
        allx = np.stack([feats[k] for k in keys]) if keys else np.zeros((0, len(FEATURES)), np.float32)
        ranked = [p for p in picks if p.get("ans") in ("a", "b") and p["a"] in feats and p["b"] in feats]
        rep = {"trained": False, "pairs": len(ranked),
               "fine_bad": sum(1 for p in picks if p.get("ans") in ("fine", "bad")), "need": self.MIN_PAIRS}
        if len(ranked) < self.MIN_PAIRS or len(keys) < 4:
            self.w, self.report = None, rep
            return rep
        self.mu = allx.mean(axis=0)
        self.sd = allx.std(axis=0) + 1e-6
        z = {k: self._z(feats[k]) for k in keys}
        dx = np.stack([z[p["a"]] - z[p["b"]] for p in ranked])
        y = np.asarray([1 if p["ans"] == "a" else 0 for p in ranked])
        if len(set(y)) < 2:                     # all "left" (or all "right"): mirror one so it can fit
            dx[0], y[0] = -dx[0], 1 - y[0]
        self.w = self._fit(dx, y)
        agree = self._agree(dx, y, None)
        base = self._agree(dx, y, [FEATURES.index("mbps")])
        self.lib = np.sort(np.asarray([float(z[k] @ self.w) for k in keys], np.float32))
        # the acceptable line: the raw score that best separates "both fine" from "both bad" files
        good = [float(z[k] @ self.w) for p in picks if p.get("ans") == "fine" for k in (p["a"], p["b"]) if k in z]
        bad = [float(z[k] @ self.w) for p in picks if p.get("ans") == "bad" for k in (p["a"], p["b"]) if k in z]
        self.line = None
        if len(good) >= 4 and len(bad) >= 4:
            cands = sorted(good + bad)
            acc = [(np.mean([g >= t for g in good]) + np.mean([b < t for b in bad])) / 2 for t in cands]
            self.line = float(cands[int(np.argmax(acc))])
        rep.update(trained=True, agree=round(agree, 3), baseline=round(base, 3),
                   trusted=agree >= base + self.MARGIN, line=self.percentile(self.line) if self.line is not None else None)
        self.report = rep
        return rep

    def raw(self, x: np.ndarray) -> float:
        return float(self._z(x) @ self.w)

    def percentile(self, raw: float) -> int:
        if not len(self.lib):
            return 50
        return int(round(100 * np.searchsorted(self.lib, raw) / len(self.lib)))

    def score(self, x: np.ndarray) -> int:
        return self.percentile(self.raw(x))

    def acceptable(self, x: np.ndarray) -> bool | None:
        return None if self.line is None else self.raw(x) >= self.line

    def pulls(self, x: np.ndarray) -> list[tuple[str, float]]:
        """Which features pull this file down (most first): (name, contribution)."""
        c = self._z(x) * self.w
        return sorted(((FEATURES[i], float(c[i])) for i in range(len(FEATURES)) if c[i] < 0), key=lambda t: t[1])

    def state(self) -> dict:
        return {"w": None if self.w is None else self.w.tolist(),
                "mu": None if self.mu is None else self.mu.tolist(),
                "sd": None if self.sd is None else self.sd.tolist(),
                "lib": self.lib.tolist(), "line": self.line, "report": self.report}

    @classmethod
    def from_state(cls, d: dict) -> "PictureModel":
        m = cls()
        if d.get("w") is not None:
            m.w, m.mu, m.sd = (np.asarray(d[k], np.float32) for k in ("w", "mu", "sd"))
        m.lib = np.asarray(d.get("lib") or [], np.float32)
        m.line, m.report = d.get("line"), d.get("report") or {"trained": False}
        return m


def res_words(h: float | None) -> str:
    if not h:
        return "?"
    for name, top in (("SD", 600), ("720p", 900), ("1080p", 1300), ("1440p", 1800), ("4K", 2600), ("5K", 3100)):
        if h < top:
            return name
    return "6K+"


def reasons(rec: dict, model: PictureModel | None = None, x: np.ndarray | None = None, limit: int = 3) -> list[str]:
    """What's wrong with the picture, in words — worst first. Fixed thresholds
    decide what's worth saying; a trained model decides the order (what pulls
    *your* score down most)."""
    out: dict[str, str] = {}
    h, eff = rec.get("h") or 0, rec.get("eff_factor") or 1.0
    if eff >= 1.3:
        word = "upscaled" if h >= 1000 else "soft"
        out["real_detail"] = f"{word} — real detail ≈ {res_words(rec.get('eff_h'))}"
    if (rec.get("block") or 1.0) >= 1.2:
        out["block"] = "blocky (heavy compression)"
    if (rec.get("noise") or 0.0) >= 5.0:
        out["noise"] = "grainy / noisy"
    if (rec.get("band") or 0.0) >= 0.3:
        out["band"] = "banding in smooth areas"
    if (rec.get("comb") or 0.0) >= 0.03:
        out["comb"] = "interlaced (combing on motion)"
    if (rec.get("frozen") or 0.0) >= 0.25:
        out["frozen"] = "repeated frames (judder)"
    if (rec.get("active") or 1.0) <= 0.85:
        out["active"] = f"letterboxed — {round(100 * rec['active'])}% of the frame is picture"
    p1, p99 = rec.get("p1"), rec.get("p99")
    if p1 is not None and p99 is not None and p99 - p1 < 120:
        out["contrast"] = "washed out / low contrast"
    if (rec.get("mean") or 100) < 35:
        out["dark"] = "very dark"
    if model is not None and model.trained and x is not None:
        rank = {name: i for i, (name, _) in enumerate(model.pulls(x))}
        keys = sorted(out, key=lambda k: rank.get(k, 99))
    else:
        keys = list(out)
    return [out[k] for k in keys[:limit]]
