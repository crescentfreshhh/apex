"""The browser (static/words.js) and the server (peaks/words.py) must say the
same thing for the same number — run both over a grid of values and compare."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from peaks import words

JS = Path(__file__).resolve().parents[1] / "src/peaks/web/static/words.js"
GRID = [i / 100 for i in range(0, 101)]
CUTS = {"p50": 0.2, "p75": 0.4, "p90": 0.6, "p95": 0.7, "p99": 0.9}


def _py():
    return {
        "taste": [list(words.taste_band(x, CUTS)) for x in GRID],
        "confidence": [words.confidence(x) for x in GRID],
        "runner": [words.close_runner_up(x, x * 0.7) for x in GRID],
        "keeper": [words.keeper_word(x) for x in GRID],
        "share": [words.share(x) for x in GRID],
        "verdict": [list(words.verdict(n, 2 + d / 10, k, 2.0)) for n in (0, 2, 5)
                    for d in range(-10, 11, 2) for k in (0.3, 0.9)],
        "accuracy": [words.accuracy(x) for x in GRID],
        "within": [words.within_one(x) for x in GRID],
        "gain": [words.gain(x / 5 - 0.05) for x in GRID],
        "auc": [words.auc(x) for x in GRID],
        "lift": [words.lift(x, 0.1) for x in GRID],
        "often": [words.how_often(x) for x in GRID],
        "trend": [words.trend(x / 10 - 0.05) for x in GRID],
        "age": [words.age(d) for d in (None, 0, 1, 3, 6, 7, 20, 44, 45, 200, 539, 540, 900)],
        "rank": [list(words.rank(x)) for x in GRID],
        "strict": [words.clip_length(x) for x in GRID],
    }


_JS_CALLS = """
const W = window.Words, G = [...Array(101).keys()].map((i) => i / 100);
const C = {p50: 0.2, p75: 0.4, p90: 0.6, p95: 0.7, p99: 0.9};
const v = [];
for (const n of [0, 2, 5]) for (let d = -10; d <= 10; d += 2) for (const k of [0.3, 0.9]) v.push(W.verdict(n, 2 + d / 10, k, 2.0));
console.log(JSON.stringify({
  taste: G.map((x) => W.tasteBand(x, C)), confidence: G.map(W.confidence),
  runner: G.map((x) => W.closeRunnerUp(x, x * 0.7)), keeper: G.map(W.keeperWord), share: G.map(W.share),
  verdict: v, accuracy: G.map(W.accuracy), within: G.map(W.withinOne), gain: G.map((x) => W.gain(x / 5 - 0.05)),
  auc: G.map(W.auc), lift: G.map((x) => W.lift(x, 0.1)), often: G.map(W.howOften),
  trend: G.map((x) => W.trend(x / 10 - 0.05)),
  age: [null, 0, 1, 3, 6, 7, 20, 44, 45, 200, 539, 540, 900].map(W.age), rank: G.map(W.rank),
  strict: G.map(W.clipLength),
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_browser_and_server_words_agree():
    shim = ("globalThis.window = globalThis; globalThis.localStorage = {getItem(){return null}, setItem(){}};"
            "globalThis.document = {documentElement: {classList: {toggle(){}}}};")
    out = subprocess.run(["node", "-e", shim + JS.read_text() + _JS_CALLS],
                         capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    py = _py()
    for k in py:
        assert js[k] == py[k], k


def test_the_card_in_the_screenshot_reads_well():
    assert words.confidence(0.56) == "Probably"
    assert words.keeper_word(1.0) is None                      # says nothing new
    assert words.share(14 / 21) == "about two-thirds" and words.share(1.0) == "all"
    assert words.share(0.66) == "about two-thirds"
    assert words.taste_band(1.0, CUTS) == ("standout", "Standout")
    assert words.age(0) == "added today"


def test_edges():
    assert words.taste_band(None, CUTS) is None and words.taste_band(0.5, None) is None
    assert words.taste_band(0.1, CUTS) == ("weak", "Weak")
    assert words.verdict(0, None, None, 2.0) == ("new to you", "new")
    assert words.verdict(10, 3.0, 0.4, 2.0)[0] == "usually a miss"   # rejected more than kept
    assert words.lift(0.5, 0.1) == "5× better than chance" and words.lift(0.21, 0.1) == "2.1× better than chance"
    assert words.gain(-0.02) == "no clear help yet"
