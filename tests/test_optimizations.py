"""Tests for the throughput/robustness optimizations: streaming JPEG split,
showinfo parsing, sampling signatures, client retries, f16 cache, playlist
score weights, and train-time cross-validation."""

from io import BytesIO

import numpy as np
import pytest
import requests

from peaks.cache import EmbeddingCache
from peaks.labels import LabelStore
from peaks.pipeline import train_profile
from peaks.playlist import build_playlist
from peaks.sampling import (
    FrameSampler,
    iter_jpegs,
    parse_showinfo_times,
)
from peaks.stash_client import StashClient, StashError


# --- streaming JPEG splitter ---------------------------------------------------


def _fake_jpeg(payload: bytes) -> bytes:
    assert b"\xff\xd9" not in payload and b"\xff\xd8" not in payload
    return b"\xff\xd8" + payload + b"\xff\xd9"


def test_iter_jpegs_splits_stream():
    blobs = [_fake_jpeg(b"aaa"), _fake_jpeg(b"bbbbb"), _fake_jpeg(b"c")]
    out = list(iter_jpegs(BytesIO(b"".join(blobs))))
    assert out == blobs


def test_iter_jpegs_survives_chunk_boundaries():
    blobs = [_fake_jpeg(bytes([i]) * 40) for i in range(1, 6)]
    # tiny chunks force every marker to straddle a read boundary at some point
    out = list(iter_jpegs(BytesIO(b"".join(blobs)), chunk_size=7))
    assert out == blobs


def test_iter_jpegs_ignores_leading_garbage_and_partial_tail():
    stream = b"junk" + _fake_jpeg(b"ok") + b"\xff\xd8partial-without-end"
    out = list(iter_jpegs(BytesIO(stream)))
    assert out == [_fake_jpeg(b"ok")]


def test_iter_jpegs_empty():
    assert list(iter_jpegs(BytesIO(b""))) == []


# --- showinfo pts parsing --------------------------------------------------------


def test_parse_showinfo_times():
    stderr = (
        "[Parsed_showinfo_0 @ 0x1] n:   0 pts:  512 pts_time:0.512 duration...\n"
        "some unrelated line\n"
        "[Parsed_showinfo_0 @ 0x1] n:   1 pts: 8512 pts_time:8.512 duration...\n"
        "[Parsed_showinfo_0 @ 0x1] n:   2 pts:18000 pts_time:18 duration...\n"
    )
    assert parse_showinfo_times(stderr) == [0.512, 8.512, 18.0]


def test_parse_showinfo_times_empty():
    assert parse_showinfo_times("nothing here") == []


# --- sampler configuration --------------------------------------------------------


def test_interval_signature_distinguishes_modes():
    a = FrameSampler(interval_seconds=2.0, mode="interval")
    b = FrameSampler(interval_seconds=2.0, mode="keyframes")
    assert a.interval_signature == 2.0
    assert b.interval_signature == -1.0


def test_keyframes_vf_uses_showinfo():
    s = FrameSampler(mode="keyframes", frame_size=288)
    vf = s._vf()
    assert vf.startswith("showinfo")
    assert "scale=" in vf and "fps=" not in vf


def test_hwaccel_flag_placement():
    s = FrameSampler(hwaccel="cuda")
    assert s._input_args() == ["-hwaccel", "cuda"]
    assert FrameSampler()._input_args() == []


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        FrameSampler(mode="every-other-tuesday")


# --- client retries -----------------------------------------------------------------


class _FlakySession:
    """Fails with a connection error N times, then succeeds."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0
        self.headers: dict = {}

    def post(self, *a, **k):
        self.calls += 1
        if self.calls <= self.failures:
            raise requests.ConnectionError("blip")

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"data": {"ok": True}}

        return _Resp()


def _flaky_client(failures: int) -> StashClient:
    client = StashClient("http://stash.test:6969")
    client.RETRY_SLEEPS = (0, 0, 0)  # don't actually sleep in tests
    client.session = _FlakySession(failures)
    return client


def test_execute_retries_transient_errors():
    client = _flaky_client(failures=2)
    assert client.execute("query { ok }") == {"ok": True}
    assert client.session.calls == 3


def test_execute_gives_up_after_retries():
    client = _flaky_client(failures=99)
    with pytest.raises(StashError, match="after 4 attempts"):
        client.execute("query { ok }")


# --- float16 cache -------------------------------------------------------------------


def test_cache_f16_roundtrip_close_and_smaller(tmp_path):
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((50, 64)).astype(np.float32)
    times = np.arange(50, dtype=np.float32)

    f16 = EmbeddingCache(tmp_path / "a")  # default float16
    f32 = EmbeddingCache(tmp_path / "b", dtype="float32")
    p16 = f16.save("k", "fake", times, vecs, meta={"interval": 2.0})
    p32 = f32.save("k", "fake", times, vecs, meta={"interval": 2.0})

    _, loaded, _ = f16.load("k", "fake")
    assert loaded.dtype == np.float32  # always returned as f32
    np.testing.assert_allclose(loaded, vecs, atol=2e-3)
    assert p16.stat().st_size < p32.stat().st_size * 0.75  # meaningfully smaller


# --- playlist score parsing ------------------------------------------------------------


class _MarkerClient:
    def __init__(self, markers):
        self._markers = markers

    def iter_markers_by_tag(self, tag_name):
        yield from self._markers

    def stream_url(self, scene_id, start=None):
        return f"http://t/scene/{scene_id}/stream?start={start:g}"


def test_playlist_extracts_score_from_title():
    m = {
        "marker_id": "m1", "scene_id": "1", "seconds": 5.0,
        "end_seconds": 15.0, "title": "apex 0.873", "primary_tag": "apex",
    }
    pl = build_playlist(_MarkerClient([m]), "apex")
    assert pl["apexes"][0]["score"] == 0.873


def test_playlist_no_score_when_title_plain():
    m = {
        "marker_id": "m1", "scene_id": "1", "seconds": 5.0,
        "end_seconds": 15.0, "title": "hand-made marker", "primary_tag": "apex",
    }
    pl = build_playlist(_MarkerClient([m]), "apex")
    assert "score" not in pl["apexes"][0]


# --- cross-validated training stats ------------------------------------------------------


def test_train_profile_reports_cv_auc(tmp_path):
    cache = EmbeddingCache(tmp_path)
    rng = np.random.default_rng(1)
    n, dim = 40, 16
    pos = rng.normal(1.0, 0.3, size=(n // 2, dim)).astype(np.float32)
    neg = rng.normal(-1.0, 0.3, size=(n // 2, dim)).astype(np.float32)
    vecs = np.vstack([pos, neg])
    times = np.arange(n, dtype=np.float32) * 2.0
    cache.save("k1", "fake", times, vecs, meta={"scene_id": "1"})

    store = LabelStore(tmp_path / "labels.json")
    for i in range(n // 2):
        store.add("k1", float(times[i]), 1, "apex")
    for i in range(n // 2, n):
        store.add("k1", float(times[i]), 0, "apex")

    _, stats = train_profile(store, cache, "fake", "apex")
    assert stats["cv_folds"] >= 2
    assert stats["cv_auc"] > 0.9  # separable clusters -> near-perfect AUC


def test_train_profile_skips_cv_when_too_few(tmp_path):
    cache = EmbeddingCache(tmp_path)
    vecs = np.array([[1.0] * 4, [-1.0] * 4], dtype=np.float32)
    times = np.array([0.0, 2.0], dtype=np.float32)
    cache.save("k1", "fake", times, vecs, meta={"scene_id": "1"})
    store = LabelStore(tmp_path / "labels.json")
    store.add("k1", 0.0, 1, "apex")
    store.add("k1", 2.0, 0, "apex")

    _, stats = train_profile(store, cache, "fake", "apex")
    assert "cv_auc" not in stats  # 1 sample per class: can't fold
