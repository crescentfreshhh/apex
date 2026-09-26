"""Sparse (seek-based) sampling, tested against REAL video generated with
PyAV — this exercises actual seek + keyframe-decode behaviour, not stubs."""

import numpy as np
import pytest

av = pytest.importorskip("av")

from peaks.sampling import FrameSampler, SamplerError  # noqa: E402


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A 60s, 5fps, 64x48 mp4 with a keyframe every 2s (gop=10)."""
    path = str(tmp_path_factory.mktemp("vid") / "test.mp4")
    container = av.open(path, "w")
    stream = container.add_stream("mpeg4", rate=5)
    stream.width, stream.height = 64, 48
    stream.pix_fmt = "yuv420p"
    stream.codec_context.gop_size = 10  # keyframe every 2s

    for i in range(300):  # 60 seconds
        # frame content varies with i so different frames differ
        arr = np.full((48, 64, 3), (i * 7) % 256, dtype=np.uint8)
        arr[:, : (i % 64), 0] = 255
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def _sample(video, interval, crop=32, resize_short=32):
    s = FrameSampler(interval_seconds=interval, mode="sparse")
    return list(s.iter_frames_raw(video, resize_short=resize_short, crop=crop))


def test_sparse_samples_scale_with_interval(video):
    out8 = _sample(video, 8.0)
    out4 = _sample(video, 4.0)
    # 60s / 8s ≈ 7-8 samples; 60s / 4s ≈ 14-16 — roughly double
    assert 6 <= len(out8) <= 9
    assert 12 <= len(out4) <= 17
    assert len(out4) > len(out8)


def test_sparse_timestamps_strictly_increasing_and_near_grid(video):
    out = _sample(video, 8.0)
    times = [t for t, _ in out]
    assert times == sorted(times)
    assert len(set(times)) == len(times)  # no duplicate keyframes
    # keyframes are every 2s, so each sample lands within a GOP of its target
    for i, t in enumerate(times):
        assert abs(t - i * 8.0) <= 2.01


def test_sparse_frames_have_model_geometry(video):
    out = _sample(video, 10.0, crop=32, resize_short=40)
    for _t, arr in out:
        assert arr.shape == (32, 32, 3)
        assert arr.dtype == np.uint8
        assert arr.flags["C_CONTIGUOUS"]


def test_sparse_dedupes_when_grid_finer_than_gop(video):
    # 0.5s grid vs ~2s keyframes: must not yield the same keyframe repeatedly
    out = _sample(video, 0.5)
    times = [t for t, _ in out]
    assert len(set(times)) == len(times)  # every sample is a distinct keyframe
    # density collapses toward keyframe spacing: far fewer than the 120 grid
    # points (the encoder's scene-change detection adds a few extra I-frames)
    assert 25 <= len(out) <= 60


def test_sparse_frames_content_varies(video):
    out = _sample(video, 8.0)
    a = out[0][1].astype(int)
    b = out[-1][1].astype(int)
    assert np.abs(a - b).mean() > 1  # different moments look different


def test_sparse_signature_distinct_from_other_modes():
    sparse8 = FrameSampler(interval_seconds=8.0, mode="sparse")
    sparse4 = FrameSampler(interval_seconds=4.0, mode="sparse")
    interval8 = FrameSampler(interval_seconds=8.0, mode="interval")
    kf = FrameSampler(mode="keyframes")
    sigs = {
        sparse8.interval_signature,
        sparse4.interval_signature,
        interval8.interval_signature,
        kf.interval_signature,
    }
    assert len(sigs) == 4  # all four invalidate each other


def test_wants_raw_matrix():
    assert FrameSampler(mode="sparse").wants_raw
    assert FrameSampler(mode="sparse", pipeline="jpeg").wants_raw  # sparse is always raw
    assert FrameSampler(mode="interval", pipeline="raw").wants_raw
    assert not FrameSampler(mode="interval", pipeline="jpeg").wants_raw
    assert not FrameSampler(mode="keyframes").wants_raw


def test_sparse_missing_file_raises():
    s = FrameSampler(mode="sparse")
    with pytest.raises(Exception):  # av raises on open; surfaced to caller
        list(s.iter_frames_raw("/nope/missing.mp4", resize_short=32, crop=32))


def test_sparse_corrupt_file_fails_gracefully(video, tmp_path):
    """A truncated/garbage file must surface an error (so the scene is marked
    failed) rather than hanging or returning silently-bad data."""
    good = open(video, "rb").read()
    corrupt = tmp_path / "corrupt.mp4"
    # keep enough header that av may open it, then append pure garbage
    corrupt.write_bytes(good[: len(good) // 3] + b"\x00\xff" * 5000)

    s = FrameSampler(interval_seconds=4.0, mode="sparse")
    with pytest.raises(Exception):  # av error on open/decode, or the bailout
        list(s.iter_frames_raw(str(corrupt), resize_short=32, crop=32))


def test_sparse_runs_in_killable_subprocess(video):
    """The scene decode is isolated in a child process the parent can kill,
    with in-worker guards a corrupt file can't defeat."""
    import inspect

    from peaks import sampling

    orch = inspect.getsource(sampling.FrameSampler._iter_frames_sparse)
    assert "get_context(\"spawn\")" in orch
    assert ".terminate()" in orch and ".kill()" in orch
    assert "scene_timeout" in orch

    worker = inspect.getsource(sampling._sparse_extract_worker) + inspect.getsource(
        sampling._sparse_extract)
    assert "total_errors" in worker and "max_total_errors" in worker
    assert "implausible duration" in worker


def test_sparse_failure_reason_reaches_the_log(tmp_path):
    """A worker crash used to surface only as 'worker exit 1'; the child's actual
    exception must now be in the error the parent raises (→ the job log)."""
    junk = tmp_path / "not-a-video.mp4"
    junk.write_bytes(b"this is not a video file" * 100)
    s = FrameSampler(interval_seconds=4.0, mode="sparse", scene_timeout=60)
    with pytest.raises(SamplerError) as ei:
        list(s.iter_frames_raw(str(junk), resize_short=32, crop=32))
    msg = str(ei.value)
    assert "worker exit 1: " in msg                      # reason appended…
    assert len(msg.split("worker exit 1: ", 1)[1]) > 5   # …and non-empty
    assert not list(tmp_path.glob("*.err"))              # no temp debris left behind


def test_sparse_error_budget_scales_with_density():
    from peaks.sampling import _sparse_error_budget

    # 40-min scene: at 8s (~300 seeks) the budget is the floor of 60; at 2s
    # (~1200 seeks) a fixed 60 would fail a file with a ~5% bad stretch — the
    # budget grows to 20% of the samples instead
    assert _sparse_error_budget(2400, 8.0) == 60
    assert _sparse_error_budget(2400, 2.0) == 240
    assert _sparse_error_budget(60, 2.0) == 60          # short scenes keep the floor


def test_sampler_signature_override():
    """The Fix job's fallback decoders stamp rescued scenes with the LIBRARY's
    signature so they count as done instead of being retried every run."""
    s = FrameSampler(interval_seconds=2.0, mode="interval", signature=-102.0)
    assert s.interval_signature == -102.0
    assert FrameSampler(interval_seconds=2.0, mode="interval").interval_signature == 2.0


def test_sparse_timeout_kills_the_worker(video):
    """An absurdly small timeout must trip the kill path and raise, proving the
    parent can stop a child that isn't done (the corrupt-hang safety net)."""
    s = FrameSampler(interval_seconds=4.0, mode="sparse", scene_timeout=0.001)
    with pytest.raises(SamplerError, match="exceeded|killed"):
        list(s.iter_frames_raw(video, resize_short=32, crop=32))


def test_scene_timeout_default_and_env(monkeypatch, tmp_path):
    from peaks.config import Config

    assert Config.load(tmp_path / "none.toml").sampling.scene_timeout == 180.0
    monkeypatch.setenv("PEAKS_SCENE_TIMEOUT", "45")
    assert Config.load(tmp_path / "none.toml").sampling.scene_timeout == 45.0


def test_scene_timeout_stored_on_sampler():
    assert FrameSampler(mode="sparse", scene_timeout=30).scene_timeout == 30


def test_sparse_feeds_embed_library(video, tmp_path):
    """Full loop: real video -> sparse sampler -> raw embed -> cache."""
    from peaks.cache import EmbeddingCache
    from peaks.embedding import FakeEmbedder
    from peaks.models import Scene
    from peaks.pipeline import embed_library

    scene = Scene.from_dict(
        {
            "id": "1",
            "title": "",
            "files": [{"path": video, "fingerprints": [
                {"type": "oshash", "value": "k1"}]}],
            "scene_markers": [],
        }
    )
    emb = FakeEmbedder(dim=16)
    emb.raw_resize, emb.raw_crop = 40, 32
    sampler = FrameSampler(interval_seconds=8.0, mode="sparse")
    cache = EmbeddingCache(tmp_path)

    stats = embed_library([scene], sampler, emb, cache, log=lambda *_: None)
    assert stats["embedded"] == 1
    assert stats["frames"] >= 6

    times, vecs, meta = cache.load("k1", "fake")
    assert vecs.shape[1] == 16 and len(times) == stats["frames"]
    assert meta["mode"] == "sparse" and meta["pipeline"] == "raw"
    assert meta["interval"] == -(100.0 + 8.0)  # sparse signature


# --- files whose seeks mostly fail are decoded linearly on the spot ---------------------

def _budget_error(path="x.mp4"):
    from peaks.sampling import SamplerError
    return SamplerError(f"sparse extraction failed for {path} (worker exit 1: RuntimeError: "
                        f"292 seek/decode errors (budget 292) in {path})")


def test_seek_error_budget_falls_back_to_linear_decode(monkeypatch):
    import numpy as np

    from peaks.sampling import FrameSampler, sampling_signature

    s = FrameSampler(interval_seconds=2, mode="sparse", hwaccel="cuda")

    def sparse(self, path, **kw):
        raise _budget_error(path)
        yield  # pragma: no cover

    used = {}

    def linear(self, path, **kw):
        used.update(mode=self.mode, hwaccel=self.hwaccel, sig=self.interval_signature)
        yield 0.0, np.zeros((4, 4, 3), np.uint8)
        yield 2.0, np.zeros((4, 4, 3), np.uint8)
    monkeypatch.setattr(FrameSampler, "_iter_frames_sparse", sparse)
    monkeypatch.setattr(FrameSampler, "_iter_frames_raw_interval", linear)
    notes = []
    s.on_fallback = lambda path, why: notes.append((path, why))
    got = [t for t, _ in s.iter_frames_raw("bad.mp4", resize_short=4, crop=4)]
    assert got == [0.0, 2.0]
    # CPU linear decode, stored at the library's sparse signature (counts as done)
    assert used == {"mode": "interval", "hwaccel": "", "sig": sampling_signature("sparse", 2)}
    assert notes and notes[0][0] == "bad.mp4" and "292 seek/decode errors" in notes[0][1]


def test_other_sparse_failures_still_fail(monkeypatch):
    from peaks.sampling import FrameSampler, SamplerError

    def sparse(self, path, **kw):
        raise SamplerError("scene sampling exceeded 180s on x — killed")
        yield  # pragma: no cover
    monkeypatch.setattr(FrameSampler, "_iter_frames_sparse", sparse)
    with pytest.raises(SamplerError, match="exceeded"):
        list(FrameSampler(mode="sparse").iter_frames_raw("x", resize_short=4, crop=4))
    s = FrameSampler(mode="sparse")
    s.linear_fallback = False
    monkeypatch.setattr(FrameSampler, "_iter_frames_sparse",
                        lambda self, path, **kw: (_ for _ in ()).throw(_budget_error()))
    with pytest.raises(SamplerError, match="seek/decode"):
        list(s.iter_frames_raw("x", resize_short=4, crop=4))


def test_worker_exits_quietly_with_its_reason(tmp_path):
    pytest.importorskip("av")
    from peaks.sampling import _sparse_extract_worker

    out = str(tmp_path / "o.npz")
    with pytest.raises(SystemExit) as e:
        _sparse_extract_worker(str(tmp_path / "missing.mp4"), 2.0, 16, 16, out)
    assert e.value.code == 1
    assert (tmp_path / "o.npz.err").read_text()
