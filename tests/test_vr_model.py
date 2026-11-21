"""The VR grade model: file quality first, studio/era and performers only when
the honest cross-validation says they add."""

import random

import pytest

pytest.importorskip("sklearn")

from peaks import vr_model as vm  # noqa: E402


def _ex(y, eye, studio="a", year=2023, perf=("p1",), mbps=60.0):
    return {"y": y, "studio": studio, "year": year, "performers": list(perf),
            "q": {"mbps": mbps, "fps": 60, "w": 8192, "h": 4096, "codec": "hevc"},
            "pic": {"eff_h": eye, "eff_factor": 4096 / eye, "sharp": eye / 1000, "noise": 1.0, "block": 0.1,
                    "band": 0.1, "bit_depth": 8}}


def _library(rng, n=30):
    """Rejects are soft (upscales); between keepers, studio 'leg' makes Légendaire."""
    exs = []
    for _ in range(n):
        exs.append(_ex("reject", rng.uniform(1500, 2300), studio=rng.choice(["leg", "exc"])))
        exs.append(_ex("exceptionnelle", rng.uniform(2900, 3800), studio="exc", perf=(f"p{rng.randint(1, 9)}",)))
        exs.append(_ex("legendaire", rng.uniform(2900, 3800), studio="leg", perf=(f"p{rng.randint(1, 9)}",)))
    return exs


def test_not_ready_until_every_grade_has_enough_examples():
    m, rep = vm.train([_ex("legendaire", 3500)] * 20 + [_ex("exceptionnelle", 3000)] * 3)
    assert m is None and not rep["trained"]
    assert rep["needs"] == {"reject": vm.MIN_PER_CLASS, "exceptionnelle": vm.MIN_PER_CLASS - 3}


def test_file_quality_finds_the_rejects_and_studio_is_kept_because_it_helps():
    m, rep = vm.train(_library(random.Random(1)))
    assert rep["trained"] and rep["uses"].startswith("file+studio")
    assert rep["accuracy"] > rep["baseline"] + 0.3
    assert rep["cv"]["file+studio"] > rep["cv"]["file"] + vm.MARGIN
    up, good_leg = m.predict([_ex(None, 1700, studio="leg"), _ex(None, 3600, studio="leg")])
    assert up["tier"] == "reject" and good_leg["tier"] == "legendaire" and up["from"] == "vr"
    assert abs(sum(up["probs"].values()) - 1) < 5e-3 and 0 <= up["keeper"] < 0.5


def test_a_group_that_doesnt_help_is_left_out():
    rng = random.Random(2)
    exs = [_ex("reject", rng.uniform(1500, 2000), studio=rng.choice("xyz")) for _ in range(20)]
    exs += [_ex(c, rng.uniform(3000, 3800) if c == "legendaire" else rng.uniform(2300, 2900),
                studio=rng.choice("xyz")) for c in ("exceptionnelle", "legendaire") for _ in range(20)]
    m, rep = vm.train(exs)
    assert rep["uses"] == "file"                      # studios here are noise: not kept


def test_records_leave_each_scenes_own_grade_out():
    exs = [_ex("legendaire", 3500, studio="solo")] + [_ex("reject", 2000, studio="other") for _ in range(5)]
    rec = vm.Records().fit(exs)
    own = rec.studio_features(exs[0], own=True)
    assert own[0] == 0.0                              # its only scene is itself: no record at all
    seen = rec.studio_features(exs[0], own=False)
    assert seen[0] == 1.0 and seen[2] > rec.base[1]


def test_examples_keep_what_survives_deletion():
    row = {"tier": "rejected", "studio": "Alpha ", "date": "2021-02-03", "performer_ids": [5],
           "quality": {"mbps": 9.0, "fps": 60, "w": 8192, "h": 4096, "codec": "h264", "res": "8K"}}
    e = vm.example(row, {"eff_h": 1800, "eff_factor": 2.2, "sharp": 1.0})
    assert e["y"] == "reject" and e["studio"] == "alpha" and e["year"] == 2021 and e["performers"] == ["5"]
    assert e["pic"]["eff_h"] == 1800 and e["q"]["mbps"] == 9.0
    assert vm.example(row, {"error": "x"})["pic"] is None
