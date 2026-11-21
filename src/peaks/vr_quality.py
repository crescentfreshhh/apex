"""VR quality check: does a VR file meet the bar your VR keepers set?

You can't judge a VR picture without the headset, and for VR the picture comes
first: a perfect scene is worthless if the file isn't sharp, whatever its label
says. So Peaks judges it two ways, both from what you've already kept:

- **the file itself** (peaks.picture measures one eye at full resolution):
  where its real detail stops (an "8K" upscaled from 5K reads as 5K), and the
  bitrate it spends on each pixel of its frame (a starved encode);
- **its studio and year**: studios change cameras and encodes over the years,
  so a file from a year none of that studio's keepers come from is a question.

Each VR scene gets a verdict — ``meets`` / ``borderline`` / ``below`` /
``unmeasured`` — and a few plain reasons. The bar is your graded VR keepers
(Exceptionnelle, Légendaire), with the scene itself always left out, so a
keeper is judged by the others. Only ever a suggestion.
"""

from __future__ import annotations

import math

KEEPERS = ("exceptionnelle", "legendaire")
MIN_BASE = 5            # measured keepers needed before there's a bar at all
MIN_STUDIO = 3          # a studio's own keepers needed before its bar/years count
UPSCALE = 1.3           # effective factor at which a file reads as upscaled (peaks.picture)
VERDICTS = ("below", "borderline", "meets", "unmeasured")
RANK = {"meets": 3, "borderline": 2, "unmeasured": 1, "below": 0}


def real_label(eye_h: float | None) -> str | None:
    """The usual VR name for a side-by-side file whose eyes hold `eye_h` lines
    of real detail: VR is named by its whole width in thousands (8192×4096 is
    "8K", 5760×2880 "6K", 3840×1920 "4K")."""
    if not eye_h:
        return None
    return f"{max(1, round(2 * eye_h / 1024))}K"


def bits_per_real_pixel(mbps: float | None, fps: float | None, eye_h: float | None) -> float | None:
    """Bitrate spent on each pixel of real detail, both eyes (square eyes)."""
    if not mbps or not fps or not eye_h:
        return None
    return mbps * 1e6 / (fps * 2 * eye_h * eye_h)


def _bpp(q: dict) -> float | None:
    """Bits per pixel of the frame, per frame (Stash's file facts). Judged on the
    frame, not on the real detail: the same bitrate on a sharper file isn't thinner."""
    if q.get("bpp"):
        return float(q["bpp"])
    w, h, mbps, fps = q.get("w"), q.get("h"), q.get("mbps"), q.get("fps")
    if not (w and h and mbps and fps):
        return None
    return mbps * 1e6 / (fps * w * h)


def _year(row: dict) -> int | None:
    d = str(row.get("date") or "")[:4]
    return int(d) if d.isdigit() and 1990 < int(d) < 2100 else None


def _quantile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    i = (len(v) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def facts(row: dict, rec: dict | None) -> dict:
    """What a VR scene's file and listing say, in the numbers the bar compares."""
    q = row.get("quality") or {}
    ok = bool(rec) and not rec.get("error") and rec.get("eff_h")
    eye_h = float(rec["eff_h"]) if ok else None
    return {"eye_h": eye_h, "label": q.get("res"), "real": real_label(eye_h),
            "upscaled": bool(ok and (rec.get("eff_factor") or 1) >= UPSCALE),
            "bpp": _bpp(q),
            "studio": (row.get("studio") or "").strip(), "year": _year(row)}


def assess(rows: list[dict], measures: dict[str, dict]) -> dict[str, dict]:
    """scene_id → {verdict, reasons, real, label, upscaled, bpp, year} for every
    VR row. `measures`: scene_id → the picture record (peaks.picture.measure)."""
    vr = [r for r in rows if r.get("vr")]
    f = {r["scene_id"]: facts(r, measures.get(r["scene_id"])) for r in vr}
    keepers = [r["scene_id"] for r in vr if r.get("tier") in KEEPERS]
    out = {}
    for r in vr:
        sid, me = r["scene_id"], f[r["scene_id"]]
        base = [k for k in keepers if k != sid]               # a keeper is judged by the others
        out[sid] = _verdict(me, [f[k] for k in base])
    return out


def _verdict(me: dict, base: list[dict]) -> dict:
    reasons: list[str] = []
    level = "meets"

    def worse(to: str) -> None:
        nonlocal level
        if RANK[to] < RANK[level]:
            level = to

    # studio and year — from the keepers, whether or not anything is measured
    studio = me["studio"]
    mine = [b for b in base if studio and b["studio"].lower() == studio.lower()]
    years = sorted(b["year"] for b in mine if b["year"])
    if studio and not mine:
        reasons.append(f"no VR keepers from {studio} yet")
    elif me["year"] and len(years) >= MIN_STUDIO and me["year"] < years[0] - 1:
        reasons.append(f"your kept {studio} VR is all {years[0]}+, this is {me['year']}")
        worse("borderline")

    measured = [b for b in base if b["eye_h"]]
    if not me["eye_h"]:
        return _out("unmeasured", ["picture not measured yet", *reasons], me)
    if len(measured) < MIN_BASE:
        return _out("unmeasured", [f"no bar yet — {len(measured)} of {MIN_BASE} VR keepers measured", *reasons], me)

    # real detail per eye, against every keeper and against this studio's
    eye = [b["eye_h"] for b in measured]
    st_eye = [b["eye_h"] for b in measured if b in mine]
    real = me["real"] or "?"
    claim = f" under a {me['label']} label" if me["label"] and me["label"] != me["real"] else ""
    if me["eye_h"] < min(eye) * 0.97:
        reasons.insert(0, f"real detail ≈ {real}{claim} — below every VR keeper you have")
        worse("below")
    elif len(st_eye) >= MIN_STUDIO and me["eye_h"] < min(st_eye) * 0.97:
        reasons.insert(0, f"real detail ≈ {real}{claim} — below every {studio} VR keeper you have")
        worse("below")
    elif me["eye_h"] < _quantile(eye, 0.25):
        reasons.insert(0, f"real detail ≈ {real}{claim} — in the bottom quarter of your VR keepers")
        worse("borderline")
    elif me["upscaled"]:
        reasons.insert(0, f"upscaled (real detail ≈ {real}{claim}), though as sharp as your keepers")
    # bitrate on that real detail
    bpp = [b["bpp"] for b in measured if b["bpp"]]
    if me["bpp"] and len(bpp) >= MIN_BASE:
        if me["bpp"] < min(bpp) * 0.9:
            reasons.append("starved encode — less bitrate for its size than any VR keeper")
            worse("below")
        elif me["bpp"] < _quantile(bpp, 0.25):
            reasons.append("thin encode — bitrate for its size in the bottom quarter of your keepers")
            worse("borderline")
    if level == "meets":
        reasons.insert(0, f"real detail ≈ {real} — as good as your VR keepers")
    return _out(level, reasons, me)


def _out(verdict: str, reasons: list[str], me: dict) -> dict:
    return {"verdict": verdict, "reasons": reasons[:3], "real": me["real"], "eye_h": me["eye_h"], "label": me["label"],
            "upscaled": me["upscaled"], "bpp": round(me["bpp"], 4) if me["bpp"] else None, "year": me["year"]}
