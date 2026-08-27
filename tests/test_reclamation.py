"""Reclamation report + per-scene peak reel (dry-run POC).

Service-level tests with the scorer/index/Stash stubbed (no torch, no ffmpeg,
no Stash) — mirrors tests/test_fix_service.py.
"""

import pytest

pytest.importorskip("fastapi")

from peaks.config import Config  # noqa: E402
from peaks.scoring import Segment  # noqa: E402
from peaks.web.service import Service  # noqa: E402


class _Idx:
    def __init__(self, sids):
        self.size = max(len(sids), 1)
        # key_meta: one key per scene, carrying its scene_id
        self.key_meta = {f"k{s}": {"scene_id": s} for s in sids}


class _Client:
    def __init__(self, details):
        self._d = details

    def scene_details(self, ids):
        return {i: self._d[i] for i in ids if i in self._d}

    def stream_url(self, sid, start=None):
        return f"http://s/{sid}?start={start}"


def _service(tmp_path, segments, details, hidden=frozenset()):
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache" / "embeddings")
    cfg.modeling.dir = str(tmp_path / "models")
    svc = Service(cfg)
    sids = list(details.keys())
    svc.index = lambda model=None, refresh=False: _Idx(sids)
    svc._scene_segments = lambda model=None, rebuild=False: (segments, "taste model")
    svc.hidden_scene_ids = lambda: set(hidden)
    svc.client = lambda: _Client(details)
    svc._model_name = lambda alias=None: "dinov2"
    return svc


def _mkfile(tmp_path, name, size):
    p = tmp_path / name
    p.write_bytes(b"\0" * size)
    return str(p)


def test_reclamation_buckets(tmp_path):
    details = {
        "1": {"path": _mkfile(tmp_path, "s1.mp4", 2000), "duration": 100.0, "title": "dead"},
        "2": {"path": _mkfile(tmp_path, "s2.mp4", 1000), "duration": 100.0, "title": "sparse"},
        "3": {"path": _mkfile(tmp_path, "s3.mp4", 1000), "duration": 100.0, "title": "peaky"},
    }
    segments = {
        # "1" has no segments → no_peak
        "2": [Segment(start=0, end=10, peak_score=0.5, mean_score=0.5)],   # 10% kept → sparse
        "3": [Segment(start=0, end=60, peak_score=0.9, mean_score=0.9)],   # 60% kept → keep whole
    }
    svc = _service(tmp_path, segments, details)
    rep = svc.reclamation_report(waste_ratio=0.4)

    assert [r["scene_id"] for r in rep["no_peak"]] == ["1"]
    assert rep["no_peak"][0]["reclaim_bytes"] == 2000
    assert [r["scene_id"] for r in rep["sparse"]] == ["2"]
    assert rep["sparse"][0]["est_bytes"] == 100  # 1000 * 0.10
    assert rep["sparse"][0]["reclaim_bytes"] == 900
    # "3" is peaky enough → in neither bucket
    assert "3" not in {r["scene_id"] for r in rep["no_peak"] + rep["sparse"]}
    t = rep["totals"]
    assert t["no_peak_bytes"] == 2000 and t["sparse_bytes"] == 900
    assert t["total_bytes"] == 2900 and t["considered"] == 3


def test_reclamation_floor_moves_scene_to_no_peak(tmp_path):
    details = {"2": {"path": _mkfile(tmp_path, "s2.mp4", 1000), "duration": 100.0, "title": "s"}}
    segments = {"2": [Segment(start=0, end=10, peak_score=0.5, mean_score=0.5)]}
    svc = _service(tmp_path, segments, details)
    # a high floor filters the 0.5 peak out entirely → whole file dead weight
    rep = svc.reclamation_report(floor=0.9, waste_ratio=0.4)
    assert [r["scene_id"] for r in rep["no_peak"]] == ["2"]
    assert not rep["sparse"]


def test_reclamation_excludes_hidden_and_missing(tmp_path):
    details = {
        "1": {"path": _mkfile(tmp_path, "s1.mp4", 2000), "duration": 100.0, "title": "dead"},
        "2": {"path": "/nope/missing.mp4", "duration": 100.0, "title": "gone"},
        "3": {"path": _mkfile(tmp_path, "s3.mp4", 500), "duration": 100.0, "title": "hidden"},
    }
    segments = {}  # all no-peak
    svc = _service(tmp_path, segments, details, hidden={"3"})
    rep = svc.reclamation_report()
    ids = {r["scene_id"] for r in rep["no_peak"]}
    assert ids == {"1"}                      # 3 hidden, 2 unresolved
    assert rep["totals"]["unresolved"] == 1
    assert rep["totals"]["considered"] == 1


def test_reel_scene_dry_run_writes_nothing(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("PEAKS_EXPORT_DIR", str(tmp_path / "exports"))
    details = {"2": {"path": _mkfile(tmp_path, "s2.mp4", 1000), "duration": 100.0, "title": "s"}}
    segments = {"2": [
        Segment(start=30, end=40, peak_score=0.8, mean_score=0.8),
        Segment(start=5, end=10, peak_score=0.7, mean_score=0.7),
    ]}
    svc = _service(tmp_path, segments, details)

    def _boom(*a, **k):
        raise AssertionError("subprocess.run must NOT run in dry-run")

    monkeypatch.setattr(subprocess, "run", _boom)

    plan = svc.reel_scene("2", dry_run=True)
    assert plan["dry_run"] is True
    # chronological order (5–10 before 30–40)
    assert [s["start"] for s in plan["segments"]] == [5.0, 30.0]
    assert plan["kept_secs"] == 15.0
    assert plan["est_bytes"] == 150  # 1000 * (15/100)
    # nothing written to the reels dir
    assert not (tmp_path / "exports" / "reels").exists()


def test_reel_scene_real_path_uses_stream_copy(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("PEAKS_EXPORT_DIR", str(tmp_path / "exports"))
    src = _mkfile(tmp_path, "s2.mp4", 1000)
    details = {"2": {"path": src, "duration": 100.0, "title": "s"}}
    segments = {"2": [Segment(start=5, end=10, peak_score=0.7, mean_score=0.7)]}
    svc = _service(tmp_path, segments, details)

    cmds = []

    def _fake_run(cmd, *a, **k):
        cmds.append(cmd)
        # write the ffmpeg output file (last arg) so the caller sees a real file
        with open(cmd[-1], "wb") as f:
            f.write(b"\0" * 50)

        class _R:
            returncode = 0
            stderr = b""
        return _R()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    out = svc.reel_scene("2", dry_run=False)
    assert out["dry_run"] is False
    assert out["out_path"].endswith(".mp4")
    assert "/reels/" in out["out_path"] and out["out_path"] != src  # never the source
    # every ffmpeg invocation is a lossless stream copy — no re-encode flags
    joined = " ".join(" ".join(c) for c in cmds)
    assert "-c copy" in joined
    assert "libx264" not in joined and "-crf" not in joined


def test_keep_set_store_roundtrip(tmp_path):
    details = {"7": {"path": _mkfile(tmp_path, "s7.mp4", 1000), "duration": 100.0, "title": "s"}}
    svc = _service(tmp_path, {}, details)
    assert svc.get_keep_segments("7") is None
    entry = svc.save_keep_segments("7", [[10, 20], [5, 8]], approved=True)
    # normalised: sorted, and stored
    assert entry["segments"] == [[5.0, 8.0], [10.0, 20.0]]
    assert entry["approved"] is True
    # a fresh Service instance reads the persisted file
    again = _service(tmp_path, {}, details)
    got = again.get_keep_segments("7")
    assert got["segments"] == [[5.0, 8.0], [10.0, 20.0]] and got["approved"] is True
    assert svc.clear_keep_segments("7") is True
    assert svc.get_keep_segments("7") is None


def test_reel_scene_uses_explicit_keepset(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("PEAKS_EXPORT_DIR", str(tmp_path / "exports"))
    details = {"2": {"path": _mkfile(tmp_path, "s2.mp4", 1000), "duration": 100.0, "title": "s"}}
    # peaks would propose one segment, but the human keep-set overrides it
    segments = {"2": [Segment(start=0, end=5, peak_score=0.9, mean_score=0.9)]}
    svc = _service(tmp_path, segments, details)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no ffmpeg in dry-run")))

    plan = svc.reel_scene("2", dry_run=True, segments=[[10, 30], [40, 45]])
    assert [[s["start"], s["end"]] for s in plan["segments"]] == [[10.0, 30.0], [40.0, 45.0]]
    assert plan["kept_secs"] == 25.0
    assert plan["est_bytes"] == 250  # 1000 * 25/100


def test_reel_scene_falls_back_to_saved_keepset(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("PEAKS_EXPORT_DIR", str(tmp_path / "exports"))
    details = {"2": {"path": _mkfile(tmp_path, "s2.mp4", 1000), "duration": 100.0, "title": "s"}}
    segments = {"2": [Segment(start=0, end=5, peak_score=0.9, mean_score=0.9)]}
    svc = _service(tmp_path, segments, details)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no ffmpeg in dry-run")))
    svc.save_keep_segments("2", [[50, 70]])
    plan = svc.reel_scene("2", dry_run=True)   # no explicit segments → saved set
    assert [[s["start"], s["end"]] for s in plan["segments"]] == [[50.0, 70.0]]


def test_reclamation_scene_payload(tmp_path):
    details = {"2": {"path": _mkfile(tmp_path, "s2.mp4", 1000), "duration": 100.0, "title": "sc"}}
    segments = {"2": [Segment(start=10, end=20, peak_score=0.9, mean_score=0.9)]}
    svc = _service(tmp_path, segments, details)
    d = svc.reclamation_scene("2", strip=5)
    assert d["seeded_segments"] == [[10.0, 20.0]]
    assert d["keep_segments"] == [[10.0, 20.0]]  # no saved edit → the seed
    assert d["has_keepset"] is False and d["approved"] is False
    assert len(d["filmstrip_times"]) == 5 and d["filmstrip_times"][0] == 0.0
    assert d["stream"] and d["duration"] == 100.0
    # once saved, the payload reflects the human keep-set
    svc.save_keep_segments("2", [[0, 40]], approved=True)
    d2 = svc.reclamation_scene("2", strip=5)
    assert d2["keep_segments"] == [[0.0, 40.0]] and d2["approved"] is True
    assert d2["has_keepset"] is True


def test_scene_frame_jpeg_any_timestamp(tmp_path, monkeypatch):
    details = {"2": {"path": _mkfile(tmp_path, "s2.mp4", 1000), "duration": 100.0, "title": "s"}}
    svc = _service(tmp_path, {}, details)
    seen = {}

    def _fake_frame(path, time, size=320):
        seen["call"] = (path, time, size)
        return b"JPG"

    monkeypatch.setattr(svc, "frame_jpeg", _fake_frame)
    out = svc.scene_frame_jpeg("2", 87.5, size=160)
    assert out == b"JPG"
    # resolved the scene's path and decoded at the requested (never-embedded) time
    assert seen["call"][1] == 87.5 and seen["call"][0].endswith("s2.mp4")


# --- Phase 5c: subtractive cut-marks + worst-first queue + dry-run ledger ----

class _QIdx:
    """A fake index exposing the positional arrays trash_queue reads."""
    def __init__(self, scene_ids, times):
        self.size = len(scene_ids)
        self.scene_ids = scene_ids
        self.keys = [f"k{s}" for s in scene_ids]
        self.times = times


def test_cut_snaps_inward_to_keyframes(tmp_path, monkeypatch):
    details = {"2": {"path": _mkfile(tmp_path, "s2.mp4", 1000), "duration": 100.0, "title": "s"}}
    svc = _service(tmp_path, {}, details)
    monkeypatch.setattr(svc, "_scene_keyframes", lambda sid: [0.0, 10.0, 20.0, 30.0])
    # marking [5,27] may only drop the fully-enclosed GOP [10,20]
    entry = svc.add_cut_segment("2", 5, 27)
    assert entry["segments"] == [[10.0, 20.0]]
    # never widens beyond the marked span
    assert 10.0 >= 5 and 20.0 <= 27
    # a span too small to enclose a whole GOP records nothing
    entry2 = svc.add_cut_segment("2", 21, 29)
    assert entry2["segments"] == [[10.0, 20.0]]  # unchanged


def test_cut_without_keyframes_keeps_range(tmp_path, monkeypatch):
    details = {"2": {"path": _mkfile(tmp_path, "s2.mp4", 1000), "duration": 100.0, "title": "s"}}
    svc = _service(tmp_path, {}, details)
    monkeypatch.setattr(svc, "_scene_keyframes", lambda sid: [])  # no ffprobe data
    entry = svc.add_cut_segment("2", 5, 27)
    assert entry["segments"] == [[5.0, 27.0]]


def test_cut_store_merge_and_remove(tmp_path, monkeypatch):
    details = {"2": {"path": _mkfile(tmp_path, "s2.mp4", 1000), "duration": 100.0, "title": "s"}}
    svc = _service(tmp_path, {}, details)
    monkeypatch.setattr(svc, "_scene_keyframes", lambda sid: [])
    svc.add_cut_segment("2", 10, 20)
    svc.add_cut_segment("2", 18, 30)   # overlaps → merges
    assert svc.get_cut_segments("2")["segments"] == [[10.0, 30.0]]
    assert svc.remove_cut("2") is True
    assert svc.get_cut_segments("2") is None


def test_trash_scene_whole(tmp_path):
    details = {"2": {"path": _mkfile(tmp_path, "s2.mp4", 1000), "duration": 100.0, "title": "s"}}
    svc = _service(tmp_path, {}, details)
    entry = svc.trash_scene("2")
    assert entry["whole"] is True
    assert svc.get_cut_segments("2")["whole"] is True


def test_trash_queue_worst_first_one_per_scene(tmp_path, monkeypatch):
    details = {s: {"path": _mkfile(tmp_path, f"s{s}.mp4", 1000), "duration": 100.0, "title": s}
               for s in ["1", "2", "3"]}
    svc = _service(tmp_path, {}, details)
    # two moments for scene 1, one each for 2 and 3
    idx = _QIdx(["1", "2", "1", "3"], [0.0, 0.0, 5.0, 0.0])
    monkeypatch.setattr(svc, "index", lambda model=None, refresh=False: idx)
    import numpy as np
    # scene 3 lowest, then 2, then 1 — ascending = worst first
    monkeypatch.setattr(svc, "_taste_scores",
                        lambda model, profile=None: (np.array([0.8, 0.5, 0.9, 0.1], np.float32), "classifier"))
    monkeypatch.setattr(svc, "stream_url", lambda sid, start=None: f"http://s/{sid}")
    q = svc.trash_queue(count=10)
    ids = [it["scene_id"] for it in q["items"]]
    assert ids == ["3", "2", "1"]      # worst first, one per scene
    # a whole-trashed scene drops out of the queue
    svc.trash_scene("3")
    q2 = svc.trash_queue(count=10)
    assert "3" not in {it["scene_id"] for it in q2["items"]}


def test_reclaim_ledger_dry_run(tmp_path, monkeypatch):
    details = {
        "1": {"path": _mkfile(tmp_path, "s1.mp4", 1000), "duration": 100.0, "title": "part"},
        "2": {"path": _mkfile(tmp_path, "s2.mp4", 4000), "duration": 100.0, "title": "whole"},
    }
    svc = _service(tmp_path, {}, details)
    monkeypatch.setattr(svc, "_scene_keyframes", lambda sid: [])
    svc.add_cut_segment("1", 0, 25)   # 25% of a 1000-byte file → ~250
    svc.trash_scene("2")               # whole 4000-byte file
    led = svc.reclaim_ledger()
    assert led["scenes"] == 2
    assert led["reclaim_bytes"] == 250 + 4000
    assert led["cut_secs"] == 125.0    # 25 + full 100
    # biggest contributor first
    assert led["top"][0]["scene_id"] == "2"
    # nothing was written to disk beyond the cut-store json
    assert not (tmp_path / "exports").exists()
