"""Reel export QC: the concat must be valid across mixed sources.

No real ffmpeg here — `subprocess.run` is faked to dispatch on argv (ffprobe vs
ffmpeg) so we can assert the control flow: uniform sources → lossless stream-copy,
mixed sources → re-encode to a common canvas, and a malformed result → raises.
"""

import json
import subprocess

import pytest

pytest.importorskip("fastapi")

from peaks.config import Config  # noqa: E402
import peaks.web.service as svc_mod  # noqa: E402


def _svc(tmp_path):
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    return svc_mod.Service(cfg)


def _touch(path, data=b"x"):
    with open(path, "wb") as f:
        f.write(data)


def _make_fake_run(calls, *, out_video=True, out_dur=30.0):
    """A subprocess.run stand-in: ffprobe on a SOURCE returns that path's canned
    profile (registered on `calls.profiles`), ffprobe on the OUTPUT returns a
    validation payload, and ffmpeg 'creates' its output file and records argv."""
    def fake_run(cmd, capture_output=False, **kw):
        calls.list.append(cmd)
        exe = cmd[0]
        if exe == "ffprobe":
            target = cmd[-1]
            if target in calls.profiles:                     # source probe
                vc, w, h, fps, pix, ac, ar, ch = calls.profiles[target]
                streams = [{"codec_type": "video", "codec_name": vc, "width": w,
                            "height": h, "avg_frame_rate": f"{int(fps)}/1", "pix_fmt": pix}]
                if ac:
                    streams.append({"codec_type": "audio", "codec_name": ac,
                                    "sample_rate": str(ar), "channels": ch})
                body = {"streams": streams}
            else:                                            # output validation
                body = {"format": {"duration": str(out_dur)},
                        "streams": ([{"codec_type": "video"}] if out_video else [])}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(body).encode(), b"")
        # ffmpeg: fabricate the output file (last argv) and succeed
        _touch(cmd[-1], b"\x00" * 2048)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")
    return fake_run


class _Calls:
    def __init__(self):
        self.list = []
        self.profiles = {}


def _seg_cmds(calls):
    return [c for c in calls.list if c[0] == "ffmpeg" and c[-1].endswith(".ts")]


def test_uniform_sources_stream_copy(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    a, b = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
    _touch(a)
    _touch(b)
    prof = ("h264", 1920, 1080, 30.0, "yuv420p", "aac", 48000, 2)
    calls = _Calls()
    calls.profiles = {a: prof, b: prof}
    monkeypatch.setattr(subprocess, "run", _make_fake_run(calls))

    res = svc._build_reel(
        [{"path": a, "start": 0, "end": 10, "scene_id": "1"},
         {"path": b, "start": 5, "end": 20, "scene_id": "2"}],
        tmp_path / "out.mp4",
    )
    assert res["mode"] == "copy" and res["clips"] == 2
    segs = _seg_cmds(calls)
    assert segs and all("-c" in c and "copy" in c for c in segs)
    assert not any("libx264" in c for c in segs)


def test_mixed_sources_reencode(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    a, b = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
    _touch(a)
    _touch(b)
    calls = _Calls()
    calls.profiles = {
        a: ("h264", 1920, 1080, 30.0, "yuv420p", "aac", 48000, 2),
        b: ("hevc", 1280, 720, 24.0, "yuv420p", "aac", 44100, 2),   # different res+fps+codec
    }
    monkeypatch.setattr(subprocess, "run", _make_fake_run(calls))

    res = svc._build_reel(
        [{"path": a, "start": 0, "end": 10, "scene_id": "1"},
         {"path": b, "start": 5, "end": 20, "scene_id": "2"}],
        tmp_path / "out.mp4",
    )
    assert res["mode"] == "reencode" and res["clips"] == 2
    segs = _seg_cmds(calls)
    assert segs and all("libx264" in c and "20" in c for c in segs)   # -crf 20
    assert all("1080" in "".join(c) for c in segs)                    # normalized canvas


def test_missing_audio_gets_silent_track(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    a, b = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
    _touch(a)
    _touch(b)
    calls = _Calls()
    calls.profiles = {
        a: ("h264", 1920, 1080, 30.0, "yuv420p", "aac", 48000, 2),
        b: ("h264", 1280, 720, 30.0, "yuv420p", None, 0, 0),          # no audio → mixed
    }
    monkeypatch.setattr(subprocess, "run", _make_fake_run(calls))
    # mixed set (a has audio, b doesn't) → re-encode; the no-audio clip gets a
    # synthesized silent track so the joined audio stays continuous.
    svc._build_reel(
        [{"path": a, "start": 0, "end": 10, "scene_id": "1"},
         {"path": b, "start": 0, "end": 20, "scene_id": "2"}],
        tmp_path / "o.mp4",
    )
    segs = _seg_cmds(calls)
    assert any("anullsrc=r=48000:cl=stereo" in "".join(c) for c in segs)  # silent track synthesized
    assert not all("anullsrc=r=48000:cl=stereo" in "".join(c) for c in segs)  # only the no-audio clip


def test_malformed_output_raises(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    a = str(tmp_path / "a.mp4")
    _touch(a)
    prof = ("h264", 1920, 1080, 30.0, "yuv420p", "aac", 48000, 2)
    calls = _Calls()
    calls.profiles = {a: prof}
    # validation ffprobe reports NO video stream → must raise, not return success
    monkeypatch.setattr(subprocess, "run", _make_fake_run(calls, out_video=False))
    with pytest.raises(RuntimeError, match="no video stream"):
        svc._build_reel([{"path": a, "start": 0, "end": 10, "scene_id": "1"}], tmp_path / "o.mp4")


def test_export_reel_caps_by_default(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    monkeypatch.setattr(svc_mod.Service, "REEL_DEFAULT_CAP", 2)
    captured = {}

    class _C:
        def iter_markers_by_tag(self, tag, page_size=200):
            return iter([{"scene_id": str(i), "seconds": 0.0, "end_seconds": 5.0} for i in range(10)])

        def scene_details(self, ids):
            return {str(i): {"path": f"/data/{i}.mp4"} for i in ids}

    monkeypatch.setattr(svc_mod.Service, "client", lambda self: _C())
    monkeypatch.setattr(svc_mod.Service, "_build_reel",
                        lambda self, specs, out, job=None, log=print: captured.update(n=len(specs)) or {"clips": len(specs)})
    svc.export_reel(tag="apex")                 # no limit → default cap (2)
    assert captured["n"] == 2
    svc.export_reel(tag="apex", limit=0)        # 0 → all 10
    assert captured["n"] == 10


# --- configurable export quality --------------------------------------------

def test_export_settings_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    svc = _svc(tmp_path)
    # defaults when unset
    e = svc.get_export_settings()
    assert (e["res"], e["fps"], e["codec"]) == ("1080", "30", "h264")
    # persist + read back (fresh instance re-reads the file)
    svc.save_export_settings(res="2160", fps="60", codec="hevc")
    e2 = _svc(tmp_path).get_export_settings()
    assert (e2["res"], e2["fps"], e2["codec"]) == ("2160", "60", "hevc")
    # validation rejects junk
    with pytest.raises(ValueError):
        svc.save_export_settings(res="720")
    with pytest.raises(ValueError):
        svc.save_export_settings(fps="144")
    with pytest.raises(ValueError):
        svc.save_export_settings(codec="av1")


def test_reel_honors_4k_hevc_settings(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    svc._settings_cache = {"export_res": "2160", "export_fps": "60", "export_codec": "hevc"}
    a, b = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
    _touch(a)
    _touch(b)
    calls = _Calls()
    calls.profiles = {
        a: ("h264", 1920, 1080, 30.0, "yuv420p", "aac", 48000, 2),
        b: ("hevc", 1280, 720, 24.0, "yuv420p", "aac", 44100, 2),
    }
    monkeypatch.setattr(subprocess, "run", _make_fake_run(calls))
    svc._build_reel(
        [{"path": a, "start": 0, "end": 10, "scene_id": "1"},
         {"path": b, "start": 5, "end": 20, "scene_id": "2"}],
        tmp_path / "o.mp4",
    )
    segs = _seg_cmds(calls)
    assert segs and all("scale=3840:2160" in "".join(c) for c in segs)
    assert all("fps=60" in "".join(c) for c in segs)
    assert all("libx265" in c for c in segs)
    assert not any("libx264" in c for c in segs)


def test_reel_defaults_stay_1080_h264(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    svc._settings_cache = {}   # unset → 1080/30/h264 defaults
    a, b = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
    _touch(a)
    _touch(b)
    calls = _Calls()
    calls.profiles = {
        a: ("h264", 1920, 1080, 30.0, "yuv420p", "aac", 48000, 2),
        b: ("hevc", 1280, 720, 24.0, "yuv420p", "aac", 44100, 2),
    }
    monkeypatch.setattr(subprocess, "run", _make_fake_run(calls))
    svc._build_reel(
        [{"path": a, "start": 0, "end": 10, "scene_id": "1"},
         {"path": b, "start": 5, "end": 20, "scene_id": "2"}],
        tmp_path / "o.mp4",
    )
    segs = _seg_cmds(calls)
    assert segs and all("scale=1920:1080" in "".join(c) for c in segs)
    assert all("fps=30" in "".join(c) for c in segs)
    assert all("libx264" in c for c in segs)


def test_reel_source_res_takes_max(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    svc._settings_cache = {"export_res": "source", "export_fps": "source"}
    a, b = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
    _touch(a)
    _touch(b)
    calls = _Calls()
    calls.profiles = {
        a: ("h264", 3840, 2160, 60.0, "yuv420p", "aac", 48000, 2),
        b: ("hevc", 1280, 720, 24.0, "yuv420p", "aac", 44100, 2),
    }
    monkeypatch.setattr(subprocess, "run", _make_fake_run(calls))
    svc._build_reel(
        [{"path": a, "start": 0, "end": 10, "scene_id": "1"},
         {"path": b, "start": 5, "end": 20, "scene_id": "2"}],
        tmp_path / "o.mp4",
    )
    segs = _seg_cmds(calls)
    assert segs and all("scale=3840:2160" in "".join(c) for c in segs)   # max of the two
    assert all("fps=60" in "".join(c) for c in segs)


# --- one-click performer reel -----------------------------------------------

def test_export_performer_builds_centered_specs(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    monkeypatch.setenv("PEAKS_EXPORT_DIR", str(tmp_path / "exports"))

    class _Hit:
        def __init__(self, sid, t):
            self.scene_id, self.time, self.score = sid, t, 0.9

    hits = [_Hit("10", 100.0), _Hit("11", 5.0), _Hit("12", 500.0)]
    monkeypatch.setattr(svc_mod.Service, "performer_best",
                        lambda self, **kw: {"performer": "Jane Doe", "hits": hits})

    class _C:
        def scene_details(self, ids):
            return {"10": {"path": "/data/10.mp4", "duration": 600.0},
                    "11": {"path": "/data/11.mp4", "duration": 30.0},
                    "12": {"path": "/data/12.mp4", "duration": 505.0}}

    monkeypatch.setattr(svc_mod.Service, "client", lambda self: _C())
    captured = {}
    monkeypatch.setattr(svc_mod.Service, "_build_reel",
                        lambda self, specs, out, job=None, log=print:
                        captured.update(specs=specs, out=str(out)) or
                        {"clips": len(specs), "mode": "reencode", "path": str(out),
                         "bytes": 1})

    res = svc.export_performer(name="Jane Doe", count=300, window=20.0)
    specs = captured["specs"]
    assert len(specs) == 3
    # centered 20s window, clamped into the scene
    assert specs[0]["start"] == 90.0 and specs[0]["end"] == 110.0          # t=100 mid-scene
    assert specs[1]["start"] == 0.0                                        # t=5 clamps to 0
    assert specs[1]["end"] == 20.0
    assert specs[2]["end"] == 505.0                                        # t=500 clamps to dur
    # performer name in the output filename
    assert "jane-doe" in captured["out"].lower()
    assert res["performer"] == "Jane Doe" and res["name"].endswith(".mp4")


def test_export_performer_groups_by_scene_best_first(tmp_path, monkeypatch):
    """Reel plays through one scene at a time (timestamp order within), scenes
    ordered best-first. Hits arrive globally taste-score-descending."""
    svc = _svc(tmp_path)
    monkeypatch.setenv("PEAKS_EXPORT_DIR", str(tmp_path / "exports"))

    class _Hit:
        def __init__(self, sid, t, score):
            self.scene_id, self.time, self.score = sid, t, score

    # interleaved across 3 scenes, out of time order, score-desc (as performer_best returns):
    hits = [
        _Hit("20", 200.0, 0.95),   # scene A top → best scene
        _Hit("22", 50.0, 0.90),    # scene C top
        _Hit("20", 10.0, 0.80),    # scene A, earlier timestamp
        _Hit("21", 300.0, 0.70),   # scene B top
        _Hit("21", 100.0, 0.60),   # scene B, earlier timestamp
    ]
    monkeypatch.setattr(svc_mod.Service, "performer_best",
                        lambda self, **kw: {"performer": "Jane Doe", "hits": hits})

    class _C:
        def scene_details(self, ids):
            return {sid: {"path": f"/data/{sid}.mp4", "duration": 1000.0}
                    for sid in ("20", "21", "22")}

    monkeypatch.setattr(svc_mod.Service, "client", lambda self: _C())
    captured = {}
    monkeypatch.setattr(svc_mod.Service, "_build_reel",
                        lambda self, specs, out, job=None, log=print:
                        captured.update(specs=specs) or {"clips": len(specs)})

    svc.export_performer(name="Jane Doe", window=20.0)
    specs = captured["specs"]
    # scenes grouped, best-first: A(0.95), C(0.90), B(0.70)
    assert [s["scene_id"] for s in specs] == ["20", "20", "22", "21", "21"]
    # within each scene, clips play in timestamp (start) order
    a_starts = [s["start"] for s in specs if s["scene_id"] == "20"]
    b_starts = [s["start"] for s in specs if s["scene_id"] == "21"]
    assert a_starts == sorted(a_starts) and a_starts == [0.0, 190.0]   # t=10 then t=200
    assert b_starts == sorted(b_starts) and b_starts == [90.0, 290.0]  # t=100 then t=300


def test_export_performer_no_hits(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    monkeypatch.setattr(svc_mod.Service, "performer_best",
                        lambda self, **kw: {"performer": "Nobody", "hits": []})
    res = svc.export_performer(name="Nobody")
    assert res["clips"] == 0 and res["performer"] == "Nobody"
