"""Hearing: the sound envelope on synthetic signals, and ffmpeg decoding."""

import shutil
import subprocess

import numpy as np
import pytest

from peaks.audio import ENV_FEATURES, decode_audio, envelope

SR = 16000


def test_a_rising_sound_gets_louder_and_silence_is_quiet():
    t = np.arange(0, 20, 1 / SR)
    x = (np.linspace(0.01, 0.8, len(t)) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    times, f = envelope(x, SR, step=2.0)
    assert f.shape == (10, len(ENV_FEATURES)) and abs(times[0] - 1.0) < 0.05
    loud = f[:, 0]
    assert (np.diff(loud) > 0).all() and loud[-1] - loud[0] > 20       # dB climbs
    _, q = envelope(np.zeros(SR * 4, np.float32), SR)
    assert q[:, 0].max() < -100


def test_clicks_on_a_beat_are_rhythmic_and_noise_is_not():
    x = np.zeros(SR * 20, np.float32)
    for s in np.arange(0, 20, 0.5):                       # 2 clicks a second
        i = int(s * SR)
        x[i: i + 200] = np.hanning(200) * 0.9
    _, f = envelope(x, SR)
    rng = np.random.default_rng(0)
    _, g = envelope((0.3 * rng.standard_normal(SR * 20)).astype(np.float32), SR)
    r = ENV_FEATURES.index("rhythm")
    assert np.median(f[:, r]) > 0.6 and np.median(g[:, r]) < 0.3
    hi = (0.5 * np.sin(2 * np.pi * 3000 * np.arange(0, 4, 1 / SR))).astype(np.float32)
    lo = (0.5 * np.sin(2 * np.pi * 200 * np.arange(0, 4, 1 / SR))).astype(np.float32)
    b = ENV_FEATURES.index("brightness")
    assert envelope(hi, SR)[1][:, b].mean() > envelope(lo, SR)[1][:, b].mean() + 2


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_decode_audio_blocks_and_a_silent_file(tmp_path):
    a = tmp_path / "tone.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=23", "-f", "lavfi",
                    "-i", "color=c=black:s=64x64:d=23", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(a)], check=True)
    blocks = list(decode_audio(str(a), SR))
    starts = [s for s, _ in blocks]
    assert starts[:4] == [0.0, 5.0, 10.0, 15.0] and len(blocks[0][1]) == 10 * SR
    assert 0.05 < float(np.abs(blocks[1][1]).max()) <= 1.0           # lavfi sine is quiet (1/8)
    v = tmp_path / "mute.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3", "-c:v", "libx264", str(v)],
                   check=True)
    assert decode_audio(str(v), SR) is None
