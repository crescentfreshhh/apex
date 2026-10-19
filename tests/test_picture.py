"""Picture quality: the metrics tell real detail from upscales, blocks, grain,
banding, interlacing and letterboxing; real files go through ffmpeg (one eye of
VR); the pairwise model learns a preference and is only trusted when it beats
bitrate alone; the app measures new files, keeps measures current, serves
comparison pairs, and uses the score in the Catalogue and for duplicates."""

import io
import os
import random
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

import peaks.picture as pq

RNG = np.random.default_rng(3)


def natural(n=512, beta=1.0, seed=0):
    """A frame with a natural-image spectrum (amplitude ∝ 1/f), plus a smooth 'wall'."""
    rng = np.random.default_rng(seed)
    fx, fy = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    r = np.hypot(fx, fy)
    r[0, 0] = 1
    img = np.real(np.fft.ifft2((rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / r ** beta))
    img = 128 + 40 * (img - img.mean()) / img.std()
    from scipy import ndimage

    img[:, : n // 3] = ndimage.gaussian_filter(img[:, : n // 3], 12)
    return np.clip(img, 0, 255).astype(np.float32)


def upscale(img, k, how=Image.LANCZOS):
    h, w = img.shape
    s = Image.fromarray(img, "F").resize((int(w / k), int(h / k)), Image.BOX)
    return np.rint(np.asarray(s.resize((w, h), how), dtype=np.float32))


def jpeg(img, q):
    b = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(b, "JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())), dtype=np.float32)


# --- the metrics ---------------------------------------------------------------------------

def test_effective_resolution_sees_upscales():
    nat = np.rint(natural(1024))
    assert pq.effective_factor(nat) < 1.1
    assert 1.2 < pq.effective_factor(upscale(nat, 1.5)) < 1.6
    assert 1.7 < pq.effective_factor(upscale(nat, 2)) < 2.3
    assert pq.effective_factor(upscale(nat, 2, Image.BICUBIC)) > 1.7
    assert pq.effective_factor(upscale(nat, 3)) > 2.5
    assert np.isnan(pq.effective_factor(np.full((256, 256), 90, np.float32)))      # flat: no verdict


def test_blockiness_is_compression_not_upscaling():
    nat = natural(512)
    assert abs(pq.blockiness(nat) - 1) < 0.05
    assert abs(pq.blockiness(upscale(nat, 2, Image.BICUBIC)) - 1) < 0.05      # every-other-pixel pattern ≠ blocks
    assert pq.blockiness(jpeg(nat, 50)) > 1.2 and pq.blockiness(jpeg(nat, 10)) > 1.8


def test_noise_banding_combing_and_bars():
    nat = natural(512)
    clean = pq.noise(nat)
    assert clean < 1.0 and 3.5 < pq.noise(np.clip(nat + RNG.standard_normal(nat.shape) * 4, 0, 255)) < 5
    grad = np.tile(np.linspace(40, 104, 1024, dtype=np.float32), (512, 1))
    assert pq.banding(np.rint(grad)) < 0.05                                   # an honest 8-bit gradient
    assert pq.banding(np.floor(grad / 4) * 4) > 0.8                           # a staircase
    assert pq.banding(np.rint(grad + RNG.standard_normal(grad.shape) * 1.2)) < 0.05   # dithered
    moved = np.roll(nat, 6, axis=1)
    il = nat.copy()
    il[1::2] = moved[1::2]
    assert pq.combing(nat) < 0.01 and pq.combing(il) > 0.05
    boxed = np.zeros((360, 640), np.float32)
    boxed[45:315] = natural(640)[:270]
    assert pq.active_box([boxed]) == (0, 45, 640, 315)


def test_measure_frames_medians_and_frozen():
    nat = np.rint(natural(512)).astype(np.uint8)
    up = upscale(nat.astype(np.float32), 2).astype(np.uint8)
    rec = pq.measure_frames([(nat, nat), (nat, None), (up, up)])
    assert rec["frozen"] == round(2 / 3, 3) and rec["active"] == 1.0
    assert rec["eff_factor"] < 1.1 and rec["eff_h"] >= 460
    with pytest.raises(ValueError):
        pq.measure_frames([])


# --- real files through ffmpeg ---------------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="no ffmpeg")


def _clip(tmp, name, vf=None, crf=16, size=(1280, 720), frames=12):
    src = tmp / "src"
    if not src.exists():
        src.mkdir()
        big = np.rint(natural(1600)).astype(np.uint8)
        for i in range(frames):
            Image.fromarray(big[i: i + size[1], 2 * i: 2 * i + size[0]]).convert("RGB").save(src / f"{i:03d}.png")
    out = tmp / name
    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", str(src / "%03d.png")]
    if vf:
        cmd += ["-vf", vf]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(out)], check=True)
    return str(out)


@needs_ffmpeg
def test_real_files(tmp_path):
    native = pq.measure(_clip(tmp_path, "native.mp4"), n=3)
    up = pq.measure(_clip(tmp_path, "up.mp4", vf="scale=640:360:flags=area,scale=1280:720:flags=lanczos"), n=3)
    vr = pq.measure(_clip(tmp_path, "vr.mp4", vf="split[a][b];[a][b]hstack,scale=1920:960"), n=3)
    small = pq.measure(_clip(tmp_path, "small.mp4", vf="scale=640:360"), n=2)
    assert native["w"] == 1280 and native["h"] == 720 and native["eff_factor"] < 1.2
    assert up["eff_factor"] > 1.6 and pq.res_words(up["eff_h"]) == "SD"
    assert vr["vr"] and vr["w"] == 960 and vr["h"] == 960              # one eye
    assert small["eff_factor"] is None                                 # too small to judge: no verdict
    assert native["bit_depth"] == 8 and native["hdr"] is False and native["duration"] >= 0.4
    with pytest.raises(RuntimeError):
        pq.measure(str(tmp_path / "nope.mp4"))


# --- learning your bar -------------------------------------------------------------------------

def _library(n=120, seed=1):
    rng = random.Random(seed)
    feats, facts = {}, {}
    for i in range(n):
        rec = {"eff_factor": rng.choice([1.0, 1.0, 1.05, 1.5, 2.0, 2.4]), "h": 2160, "sharp": rng.random(),
               "noise": rng.random() * 6, "block": 1 + rng.random() * 0.3, "band": rng.random() * 0.2,
               "comb": 0.0, "frozen": 0.0, "active": 1.0, "p1": 16, "p99": 235, "mean": 110}
        rec["eff_h"] = 2160 / rec["eff_factor"]
        q = {"mbps": rng.uniform(5, 60), "bpp": rng.random() * 0.3, "fps": 30}
        feats[f"k{i}"], facts[f"k{i}"] = pq.features(rec, q), (rec, q)
    return feats, facts


def _picks(feats, judge, n=150, seed=2, ties=0):
    rng = random.Random(seed)
    keys = list(feats)
    out = []
    for _ in range(n):
        a, b = rng.sample(keys, 2)
        out.append({"a": a, "b": b, "ans": "a" if judge(a) > judge(b) else "b"})
    return out


def test_model_learns_real_detail_and_beats_bitrate():
    feats, facts = _library()
    real = lambda k: 1 / facts[k][0]["eff_factor"] + 0.05 * facts[k][0]["sharp"]          # noqa: E731
    m = pq.PictureModel()
    rep = m.train(feats, _picks(feats, real))
    assert rep["trained"] and rep["agree"] > 0.85 and rep["trusted"]
    best = max(feats, key=real)
    worst = min(feats, key=real)
    assert m.score(feats[best]) > m.score(feats[worst]) and 0 <= m.score(feats[worst]) <= 100
    assert pq.PictureModel.from_state(m.state()).score(feats[best]) == m.score(feats[best])


def test_a_bitrate_only_taste_isnt_trusted_and_few_picks_wait():
    feats, facts = _library()
    rep = pq.PictureModel().train(feats, _picks(feats, lambda k: facts[k][1]["mbps"]))
    assert rep["trained"] and not rep["trusted"]               # bitrate alone already explains it
    rep = pq.PictureModel().train(feats, _picks(feats, lambda k: 1, n=10))
    assert not rep["trained"] and rep["need"] == pq.PictureModel.MIN_PAIRS


def test_acceptable_line_from_fine_and_bad():
    feats, facts = _library()
    real = lambda k: 1 / facts[k][0]["eff_factor"]                                       # noqa: E731
    picks = _picks(feats, real)
    good = [k for k in feats if facts[k][0]["eff_factor"] <= 1.05]
    bad = [k for k in feats if facts[k][0]["eff_factor"] >= 2.0]
    picks += [{"a": good[i], "b": good[i + 1], "ans": "fine"} for i in range(0, 10, 2)]
    picks += [{"a": bad[i], "b": bad[i + 1], "ans": "bad"} for i in range(0, 10, 2)]
    m = pq.PictureModel()
    m.train(feats, picks)
    assert m.line is not None
    assert all(m.acceptable(feats[k]) for k in good[:10]) and not any(m.acceptable(feats[k]) for k in bad[:10])


def test_reasons_in_words():
    rec = {"h": 2160, "eff_factor": 2.0, "eff_h": 1080, "block": 1.4, "noise": 1, "band": 0, "comb": 0,
           "frozen": 0, "active": 0.74, "p1": 16, "p99": 235, "mean": 100}
    words = pq.reasons(rec, limit=5)
    assert words[0] == "upscaled — real detail ≈ 1080p"
    assert "blocky (heavy compression)" in words and "letterboxed — 74% of the frame is picture" in words
    assert pq.reasons({"h": 1080, "eff_factor": 1.0}) == []


# --- in the app ----------------------------------------------------------------------------------

@pytest.fixture
def lib(tmp_path):
    pytest.importorskip("fastapi")
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    native = _clip(tmp_path, "native.mp4")
    up = _clip(tmp_path, "up.mp4", vf="scale=640:360:flags=area,scale=1280:720:flags=lanczos")
    noisy = _clip(tmp_path, "noisy.mp4", vf="noise=alls=30:allf=t")
    return {"native": native, "up": up, "noisy": noisy}


@pytest.fixture
def svc(tmp_path, lib, monkeypatch):
    from fakestash import FakeStash

    import peaks.web.service as svc_mod
    from peaks.config import Config

    stash = FakeStash({
        "1": {"path": lib["native"], "rating100": 100, "o_counter": 18, "width": 1280, "height": 720},
        "2": {"path": lib["up"], "rating100": None, "width": 1280, "height": 720},
        "3": {"path": lib["noisy"], "rating100": 100, "o_counter": 16, "width": 1280, "height": 720},
        "4": {"path": str(tmp_path / "missing.mp4"), "rating100": None},
        "5": {"path": lib["native"] + ".rej", "rating100": 20},
    })
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    monkeypatch.setattr(pq, "N_STILLS", 3)
    monkeypatch.setattr(pq.measure, "__defaults__", ("ffmpeg", "ffprobe", 3))
    s = svc_mod.Service(cfg)
    s.last_ingest = lambda: {"new": ["2"]}
    return s


def test_new_files_first_backfill_only_when_asked(svc):
    assert svc.picture_due()
    assert svc.run_picture()["measured"] == 1                       # just the new file
    assert not svc.picture_due()                                    # no backfill until you start it
    svc.picture_backfill(True)
    r = svc.run_picture()
    assert r["measured"] == 2 and r["failed"] == 1                  # the missing file is recorded, not retried
    st = svc.picture_status()
    assert st["measured"] == 3 and st["failed"] == 1 and st["backfill"] is False   # done: switched itself off
    assert not svc.picture_due()


def test_a_replaced_file_is_measured_again(svc, lib):
    svc.picture_backfill(True)
    svc.run_picture()
    row = next(r for r in svc._catalogue_all() if r["scene_id"] == "1")
    assert not svc._pic_needs(row)
    st = os.stat(lib["native"])
    os.utime(lib["native"], (st.st_atime, st.st_mtime + 100))
    assert svc._pic_needs(row)


def test_pairs_answers_scores_and_catalogue(svc):
    svc.picture_backfill(True)
    svc.run_picture()
    p = svc.picture_pair(random.Random(0))
    assert p and p["a"]["key"] != p["b"]["key"] and p["a"]["crop"].startswith("/api/picture/still/")
    safe, kind = p["a"]["crop"].split("/")[-2:]
    assert svc.picture_still(safe, kind) is not None and svc.picture_still("../x", "crop") is None
    with pytest.raises(ValueError):
        svc.picture_answer("a", "b", "maybe")
    pics = svc.picture_scores()
    assert pics["2"]["upscaled"] and pics["2"]["reasons"][0].startswith("soft — real detail")
    assert pics["1"]["score"] is None                               # no model yet: no score, still reasons
    ids = [x["scene_id"] for x in svc.catalogue(pic="upscaled", tier="", limit=50)["items"]]
    assert ids == ["2"]
    order = [x["scene_id"] for x in svc.catalogue(sort="picture", tier="", limit=50)["items"]]
    assert order.index("2") > order.index("1")                     # less real detail sorts lower
    item = next(x for x in svc.catalogue(tier="", limit=50)["items"] if x["scene_id"] == "2")
    assert item["picture"]["upscaled"]


def test_duplicates_follow_a_trusted_score(svc, monkeypatch):
    rows = [r for r in svc._catalogue_all() if r["scene_id"] in ("1", "2")]
    assert svc._dupe_keeper(rows) == svc.dupe_keeper(rows)          # not trusted: the file-facts rule
    monkeypatch.setattr(type(svc), "picture_trusted", lambda self: True)
    monkeypatch.setattr(type(svc), "picture_scores", lambda self: {"1": {"score": 20}, "2": {"score": 80}})
    assert svc._dupe_keeper(rows) == "2"


def test_keeping_a_duplicate_is_a_pick(svc):
    svc.picture_backfill(True)
    svc.run_picture()
    svc.picture_note_keep("1", ["2"])
    picks = svc._pic_picks()
    assert picks[-1]["ans"] == "a" and picks[-1]["src"] == "duplicate"
    assert svc.picture_status()["from_duplicates"] == 1 and svc.picture_status()["picks"] == 0


def test_routes(svc, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(app_mod.create_app(svc.cfg))
    st = api.get("/api/picture/status").json()
    assert st["total"] == 4 and st["model"]["trained"] is False
    assert api.post("/api/picture/answer", json={"a": "x", "b": "y", "ans": "nope"}).status_code == 400
    assert api.get("/api/picture/still/abc/crop").status_code == 404
    assert api.get("/api/catalogue", params={"pic": "bogus"}).status_code == 422
    assert api.post("/api/picture/backfill", params={"on": "true"}).json()["backfill"] is True
