"""Plain words for Peaks' numbers — one vocabulary for the whole app.

Every judgment Peaks shows (taste, how sure the tier model is, track records,
model accuracy…) reads as a phrase first; the exact figure travels alongside
for anyone who wants it. `static/words.js` mirrors this file line for line so
the browser and the server always say the same thing — tests hold them equal.

Taste words are always relative to *your* library (percentile cutoffs of your
own scores): a raw taste score means nothing across models, "top 5% of your
library" always does.
"""

from __future__ import annotations

import math

def _jsround(x: float) -> int:
    """Halves round up, as the browser's Math.round does (Python's round() goes
    to even) — so both halves of the app print the same number."""
    return math.floor(x + 0.5)


# --- taste, relative to your library --------------------------------------------

TASTE_BANDS: tuple[tuple[str, str, str], ...] = (
    # key, word, which cutoff it needs (a percentile of your own scores)
    ("standout", "Standout", "p99"),
    ("strong", "Strong", "p95"),
    ("good", "Good", "p90"),
    ("decent", "Decent", "p75"),
    ("soso", "So-so", "p50"),
)
TASTE_WEAK = ("weak", "Weak")
TASTE_HINT = {"standout": "top 1% of your library", "strong": "top 5%", "good": "top 10%",
              "decent": "top 25%", "soso": "top half", "weak": "bottom half"}


def taste_band(score: float | None, cuts: dict | None) -> tuple[str, str] | None:
    """(key, word) for a taste score against your library's cutoffs
    ({p50, p75, p90, p95, p99}); None when either is unknown."""
    if score is None or not cuts:
        return None
    for key, word, q in TASTE_BANDS:
        c = cuts.get(q)
        if c is not None and score >= c:
            return key, word
    return TASTE_WEAK


# --- how sure the tier model is ----------------------------------------------------

def confidence(conf: float) -> str:
    """'Almost certainly' / 'Probably' / 'Leaning' / 'Hard to call' — for the
    top tier's probability among five."""
    if conf >= 0.75:
        return "Almost certainly"
    if conf >= 0.5:
        return "Probably"
    if conf >= 0.35:
        return "Leaning"
    return "Hard to call"


def close_runner_up(conf: float, runner_up: float | None) -> bool:
    """Is the second-likeliest tier worth naming ('…, maybe Exceptionnelle')?"""
    return runner_up is not None and conf < 0.75 and runner_up >= 0.6 * conf


def keeper_word(keeper: float | None) -> str | None:
    """Chance it's not a reject, in words — None when it adds nothing (≥ 90%)."""
    if keeper is None or keeper >= 0.9:
        return None
    if keeper >= 0.65:
        return "Likely a keeper"
    if keeper >= 0.35:
        return "Could go either way"
    return "Likely a reject"


# --- shares, records, accuracy -------------------------------------------------------

def share(p: float) -> str:
    """A proportion in words: none · hardly any · a few · about a third ·
    about half · about two-thirds · most · nearly all · all."""
    if p <= 0:
        return "none"
    if p >= 1:
        return "all"
    for cut, word in ((0.1, "hardly any"), (0.25, "a few"), (0.4, "about a third"),
                      (0.6, "about half"), (0.75, "about two-thirds"), (0.9, "most")):
        if p < cut:
            return word
    return "nearly all"


def verdict(n: float, shrunk: float | None, keep: float | None, base: float) -> tuple[str, str]:
    """(word, tone) for a performer's / studio's record against your library
    average grade (`shrunk` and `base` on the 0–4 tier scale)."""
    if not n:
        return "new to you", "new"
    if n < 3:
        return "early days", "neutral"
    d = (shrunk if shrunk is not None else base) - base
    if keep is not None and keep < 0.5:
        return "usually a miss", "bad"
    if d >= 0.6:
        return "a favourite", "good"
    if d >= 0.2:
        return "a good bet", "good"
    if d > -0.2:
        return "mixed", "neutral"
    if d > -0.6:
        return "hit and miss", "neutral"
    return "usually a miss", "bad"


def accuracy(exact: float) -> str:
    if exact >= 0.8:
        return "very reliable"
    if exact >= 0.6:
        return "usually right"
    if exact >= 0.4:
        return "often close"
    return "rough guesses"


def within_one(p: float) -> str:
    if p >= 0.9:
        return "almost always within a tier"
    if p >= 0.75:
        return "usually within a tier"
    return "often more than a tier off"


def gain(g: float) -> str:
    """A cross-validated accuracy gain (0–1) in words."""
    pts = g * 100
    if pts >= 8:
        return "helps a lot"
    if pts >= 3:
        return "helps"
    if pts > 0:
        return "helps a little"
    return "no clear help yet"


# --- the taste benchmark ------------------------------------------------------------

def auc(a: float) -> str:
    if a >= 0.9:
        return "Excellent"
    if a >= 0.8:
        return "Good"
    if a >= 0.7:
        return "Fair"
    if a >= 0.6:
        return "Weak"
    return "No better than a coin"


def lift(p: float, base: float) -> str:
    if not base:
        return ""
    x = p / base
    if x < 1.15:
        return "no better than chance"
    return f"{x:.1f}× better than chance" if x < 3 else f"{_jsround(x)}× better than chance"


def how_often(p: float) -> str:
    if p >= 0.75:
        return "most of the time"
    if p >= 0.5:
        return "more often than not"
    if p >= 0.3:
        return "some of the time"
    return "rarely"


def trend(delta: float, eps: float = 0.02) -> str:
    if delta > eps:
        return "better than last time"
    if delta < -eps:
        return "slipped since last time"
    return "about the same as last time"


# --- time, rank, similarity ---------------------------------------------------------

def age(days: int | None) -> str | None:
    if days is None:
        return None
    if days <= 0:
        return "added today"
    if days == 1:
        return "added yesterday"
    if days < 7:
        return "added this week"
    if days < 45:
        w = _jsround(days / 7)
        return f"added {w} week{'s' if w != 1 else ''} ago"
    if days < 540:
        return f"added {_jsround(days / 30)} months ago"
    return f"added {days / 365:.1f} years ago"


def rank(pct: float) -> tuple[str, str]:
    """(key, words) for a rank among your performers — `pct` 0 = best."""
    if pct < 0.1:
        return "top", "a favourite"
    if pct < 0.3:
        return "strong", "strong"
    if pct < 0.6:
        return "mixed", "mixed"
    return "low", "rarely your taste"


def clip_length(sim: float) -> str:
    """The moment-length slider (0.3 long … 0.95 tight) in words."""
    if sim >= 0.8:
        return "very tight"
    if sim >= 0.6:
        return "tight"
    if sim >= 0.45:
        return "balanced"
    return "long"
