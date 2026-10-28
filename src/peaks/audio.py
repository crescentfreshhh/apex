"""Hearing: a scene's sound, block by block, and how it moves over time.

`decode_audio` streams a file's audio through ffmpeg as mono float32 PCM, in
overlapping blocks, so a long scene never sits in memory whole. `envelope`
describes the sound every couple of seconds — loudness, how much is happening
(onsets), brightness, and how rhythmic it is — the same step as the picture
index, so sound and picture line up. Pure numpy; the CLAP embedder (the sound's
"meaning", searchable with words) lives in peaks.embedding.
"""

from __future__ import annotations

import subprocess
from typing import Iterator

import numpy as np

ENV_FEATURES: tuple[str, ...] = ("loudness", "onset", "brightness", "rhythm")
FRAME = 0.05              # seconds per analysis frame inside the envelope
RHYTHM_SPAN = 8.0         # seconds of onsets looked at for a steady beat


class AudioError(RuntimeError):
    pass


def has_audio(path: str, ffprobe: str = "ffprobe") -> bool:
    try:
        out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                              "-of", "csv=p=0", path], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(out.stdout.strip())


def decode_audio(path: str, sr: int, ffmpeg: str = "ffmpeg", block: float = 10.0,
                 hop: float = 5.0) -> Iterator[tuple[float, np.ndarray]] | None:
    """(start seconds, samples) blocks of `block` s every `hop` s, mono float32
    at `sr`. None when the file has no audio stream."""
    if not has_audio(path, ffmpeg.replace("ffmpeg", "ffprobe") if "ffmpeg" in ffmpeg else "ffprobe"):
        return None
    return _blocks(path, sr, ffmpeg, block, hop)


def _blocks(path, sr, ffmpeg, block, hop):
    cmd = [ffmpeg, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise AudioError(f"{ffmpeg} not found") from exc
    nb, nh = int(block * sr), int(hop * sr)
    buf = np.zeros(0, np.float32)
    start = 0
    try:
        while True:
            chunk = proc.stdout.read(nh * 4)
            if chunk:
                buf = np.concatenate([buf, np.frombuffer(chunk[: len(chunk) // 4 * 4], np.float32)])
            while len(buf) >= nb:
                yield start / sr, buf[:nb].copy()
                buf = buf[nh:]
                start += nh
            if not chunk:
                break
        if len(buf) > sr:                                       # the tail, if there's a second of it
            yield start / sr, buf.copy()
    finally:
        proc.stdout.close()
        proc.wait()


def _frames(x: np.ndarray, n: int) -> np.ndarray:
    k = len(x) // n
    return x[: k * n].reshape(k, n) if k else np.zeros((0, n), np.float32)


def envelope(samples: np.ndarray, sr: int, step: float = 2.0, t0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(times, features) every `step` s from `t0`: loudness (dB, RMS), onset
    strength (spectral flux), brightness (spectral centroid, kHz) and rhythm
    (how strongly the onsets repeat at one beat over RHYTHM_SPAN s, 0..1)."""
    x = np.asarray(samples, np.float32)
    n = max(16, int(FRAME * sr))
    F = _frames(x, n)
    if not len(F):
        return np.zeros(0, np.float32), np.zeros((0, len(ENV_FEATURES)), np.float32)
    rms = np.sqrt((F.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    spec = np.abs(np.fft.rfft(F * np.hanning(n), axis=1))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    centroid = (spec * freqs).sum(axis=1) / np.maximum(spec.sum(axis=1), 1e-9) / 1000.0
    flux = np.r_[0.0, np.maximum(np.diff(spec, axis=0), 0).sum(axis=1)]
    flux = flux / max(float(np.percentile(flux, 95)), 1e-9)
    per = max(1, int(round(step / FRAME)))
    k = len(F) // per
    times, out = [], []
    span = int(RHYTHM_SPAN / FRAME)
    for i in range(k):
        a, b = i * per, (i + 1) * per
        c = (a + b) // 2
        o = flux[max(0, c - span // 2): c + span // 2]
        times.append(t0 + (a + b) / 2 * FRAME)
        out.append([float(db[a:b].mean()), float(flux[a:b].mean()), float(centroid[a:b].mean()), _rhythm(o)])
    return np.asarray(times, np.float32), np.asarray(out, np.float32)


def _rhythm(onsets: np.ndarray) -> float:
    """Autocorrelation peak of the onset curve at a beat between 0.25 and 2 s."""
    o = np.asarray(onsets, np.float64)
    if len(o) < 40 or o.std() < 1e-6:
        return 0.0
    o = o - o.mean()
    ac = np.correlate(o, o, mode="full")[len(o) - 1:]
    ac = ac / max(ac[0], 1e-12)
    lo, hi = int(0.25 / FRAME), min(len(ac) - 1, int(2.0 / FRAME))
    return float(np.clip(ac[lo: hi + 1].max(), 0, 1)) if hi > lo else 0.0
