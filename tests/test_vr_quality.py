"""The VR quality check: real detail and bitrate against your VR keepers, plus
the studio's years — the smell test you'd otherwise need the headset for."""

from peaks.vr_quality import assess, bits_per_real_pixel, real_label


def _row(sid, tier="unreviewed", studio="Alpha", date="2023-05-01", res="8K", mbps=60.0, fps=60):
    return {"scene_id": sid, "vr": True, "tier": tier, "studio": studio, "date": date,
            "quality": {"res": res, "mbps": mbps, "fps": fps, "w": 8192, "h": 4096}}


def _rec(eye_h, factor=1.0):
    return {"eff_h": eye_h, "eff_factor": factor}


def _library():
    rows, meas = [], {}
    for i in range(8):                      # keepers: real 6K-8K per eye, healthy bitrate, 2021-2024
        sid = f"k{i}"
        rows.append(_row(sid, tier="legendaire" if i % 2 else "exceptionnelle", date=f"{2021 + i % 4}-01-01"))
        meas[sid] = _rec(2880 + 150 * i)
    return rows, meas


def test_labels_and_bitrate_per_real_pixel():
    assert real_label(4096) == "8K" and real_label(2880) == "6K" and real_label(None) is None
    assert round(bits_per_real_pixel(60, 60, 4096), 4) == round(60e6 / (60 * 2 * 4096 * 4096), 4)


def test_an_upscale_under_an_8k_label_is_below_the_bar():
    rows, meas = _library()
    rows.append(_row("up", res="8K"))
    meas["up"] = _rec(1900, factor=2.1)     # "8K" made from ~4K
    v = assess(rows, meas)["up"]
    assert v["verdict"] == "below" and "under a 8K label" in v["reasons"][0] and v["upscaled"]


def test_a_sharp_well_encoded_file_meets_it():
    rows, meas = _library()
    rows.append(_row("good"))
    meas["good"] = _rec(3600)
    v = assess(rows, meas)["good"]
    assert v["verdict"] == "meets" and v["reasons"][0].startswith("real detail ≈ 7K")


def test_a_starved_encode_is_below_the_bar_even_when_sharp():
    rows, meas = _library()
    rows.append(_row("thin", mbps=8.0))
    meas["thin"] = _rec(3600)
    v = assess(rows, meas)["thin"]
    assert v["verdict"] == "below" and any("starved" in x for x in v["reasons"])


def test_studio_years_and_new_studios():
    rows, meas = _library()
    rows.append(_row("old", date="2018-06-01"))
    meas["old"] = _rec(3600)
    rows.append(_row("new", studio="Beta"))
    meas["new"] = _rec(3600)
    v = assess(rows, meas)
    assert v["old"]["verdict"] == "borderline" and any("all 2021+, this is 2018" in x for x in v["old"]["reasons"])
    assert v["new"]["verdict"] == "meets" and any("no VR keepers from Beta yet" in x for x in v["new"]["reasons"])


def test_no_bar_until_enough_keepers_are_measured_and_unmeasured_files_say_so():
    rows, meas = _library()
    rows.append(_row("x"))
    v = assess(rows, meas)["x"]
    assert v["verdict"] == "unmeasured" and v["reasons"][0] == "picture not measured yet"
    few = {k: m for k, m in list(meas.items())[:3]}
    few["x"] = _rec(3600)
    v = assess(rows, few)["x"]
    assert v["verdict"] == "unmeasured" and "3 of 5 VR keepers measured" in v["reasons"][0]


def test_a_keeper_is_judged_by_the_others_and_flat_rows_are_ignored():
    rows, meas = _library()
    meas["k0"] = _rec(1500)                 # your weakest keeper: below the rest, not its own bar
    rows.append({"scene_id": "flat", "tier": "legendaire", "studio": "Alpha", "quality": {}})
    meas["flat"] = _rec(500)
    v = assess(rows, meas)
    assert v["k0"]["verdict"] == "below" and "flat" not in v
    assert all(x["verdict"] != "below" for k, x in v.items() if k != "k0")   # 500-line flat set no bar
