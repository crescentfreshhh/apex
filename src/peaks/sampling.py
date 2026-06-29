"""Frame sampling via ffmpeg.

The actual decode shells out to ffmpeg/ffprobe at runtime (not bundled in this
repo). The *planning* logic — which timestamps to sample — is a pure function
(`plan_timestamps`) and is unit-tested offline.

Typical use (on a box with ffmpeg + the `[ml]` extra installed):

    sampler = FrameSampler(interval_seconds=2.0)
    for ts, img in sampler.iter_frames("/path/to/video.mp4"):
        ...  # img is a PIL.Image
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image


def plan_timestamps(
    duration: float, interval: float, *, offset: float | None = None
) -> list[float]:
    """Evenly spaced sample timestamps across a clip.

    Starts half an interval in (so the first frame isn't the very first frame,
    which is often black/a title card) and steps by `interval`. Pure + tested.
    """
    if duration <= 0 or interval <= 0:
        return []
    if offset is None:
        offset = interval / 2.0
    times: list[float] = []
    t = offset
    while t < duration:
        times.append(round(t, 3))
        t += interval
    return times


class SamplerError(RuntimeError):
    pass


class FrameSampler:
    def __init__(
        self,
        interval_seconds: float = 2.0,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
    ):
        self.interval = interval_seconds
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    def probe_duration(self, path: str) -> float:
        """Return duration in seconds via ffprobe."""
        cmd = [
            self.ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            path,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise SamplerError(f"{self.ffprobe} not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise SamplerError(f"ffprobe failed for {path}: {exc.stderr[:300]}") from exc
        try:
            return float(json.loads(out.stdout)["format"]["duration"])
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise SamplerError(f"could not parse duration for {path}") from exc

    def iter_frames(self, path: str) -> Iterator[tuple[float, "Image"]]:
        """Yield (timestamp_seconds, PIL.Image) sampled every `interval` seconds.

        Extracts with a single ffmpeg pass (`fps=1/interval`) into a temp dir,
        loads frames, and cleans up. Frames map to planned timestamps in order.
        """
        from PIL import Image as PILImage  # lazy

        duration = self.probe_duration(path)
        times = plan_timestamps(duration, self.interval)
        if not times:
            return

        tmpdir = Path(tempfile.mkdtemp(prefix="peaks-frames-"))
        try:
            cmd = [
                self.ffmpeg,
                "-v", "error",
                "-i", path,
                "-vf", f"fps=1/{self.interval}",
                "-q:v", "3",
                str(tmpdir / "f-%06d.jpg"),
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except FileNotFoundError as exc:
                raise SamplerError(f"{self.ffmpeg} not found on PATH") from exc
            except subprocess.CalledProcessError as exc:
                raise SamplerError(
                    f"ffmpeg failed for {path}: {exc.stderr[:300]}"
                ) from exc

            frame_files = sorted(tmpdir.glob("f-*.jpg"))
            for ts, fp in zip(times, frame_files):
                with PILImage.open(fp) as im:
                    yield ts, im.convert("RGB")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
