"""Frame sampling via ffmpeg.

Two modes:

  "interval"  — one frame every `interval_seconds`, streamed straight out of
                ffmpeg over a pipe (no temp files) through a bounded queue, so
                decode runs ahead while the consumer (the GPU embedder) works.
                Frame i comes from source time ~`i * interval`.

  "keyframes" — decode only keyframes (`-skip_frame nokey`): ~1-5% of the
                decode cost, at the price of irregular, encode-dependent
                spacing (typically every 2-10s). Exact timestamps are parsed
                from ffmpeg's showinfo output. The throughput escape hatch for
                huge libraries.

Optional `hwaccel` ("cuda"/"auto") offloads decode to the GPU (NVDEC) — decode,
not embedding, is the pipeline bottleneck.

The pure helpers (`plan_timestamps`, `iter_jpegs`, `parse_showinfo_times`) are
unit-tested offline; the ffmpeg calls themselves need a real box.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
from io import BytesIO
from pathlib import Path
from queue import Queue
from typing import IO, TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"


def plan_timestamps(
    duration: float, interval: float, *, offset: float = 0.0
) -> list[float]:
    """Expected sample timestamps for a clip: `offset, offset+interval, ...`.

    Matches how ffmpeg's `fps=1/interval` filter emits frames: the first output
    frame comes from the very start of the clip, then one per interval. Frame i
    therefore represents source time ~`i * interval` — timestamps must NOT be
    shifted, or every downstream marker/label lands offset from its frame.
    """
    if duration <= 0 or interval <= 0:
        return []
    times: list[float] = []
    t = offset
    while t < duration:
        times.append(round(t, 3))
        t += interval
    return times


def iter_jpegs(stream: IO[bytes], chunk_size: int = 65536) -> Iterator[bytes]:
    """Split a concatenated-MJPEG byte stream into individual JPEG blobs.

    Scans for SOI (FFD8) / EOI (FFD9) markers across chunk boundaries. Safe for
    ffmpeg mjpeg output: 0xFF bytes inside entropy-coded data are always
    followed by 0x00 stuffing, and there are no embedded EXIF thumbnails, so a
    raw FFD9 only ever terminates a frame.
    """
    buf = b""
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        buf += chunk
        while True:
            soi = buf.find(_JPEG_SOI)
            if soi < 0:
                buf = buf[-1:]  # keep a possible split FF
                break
            eoi = buf.find(_JPEG_EOI, soi + 2)
            if eoi < 0:
                buf = buf[soi:]  # incomplete frame; wait for more bytes
                break
            yield buf[soi : eoi + 2]
            buf = buf[eoi + 2 :]


_SHOWINFO_PTS = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_showinfo_times(stderr_text: str) -> list[float]:
    """Extract per-frame pts_time values from ffmpeg showinfo stderr output."""
    return [round(float(m), 3) for m in _SHOWINFO_PTS.findall(stderr_text)]


class SamplerError(RuntimeError):
    pass


class FrameSampler:
    def __init__(
        self,
        interval_seconds: float = 2.0,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        frame_size: int = 288,
        mode: str = "interval",
        hwaccel: str = "",
        queue_frames: int = 256,
    ):
        """`frame_size`: short-side pixels frames are downscaled to during
        extraction (0 = original size). Embedders resize to ~224px anyway, so
        decoding small keeps RAM/temp usage low while staying above model input
        size. `queue_frames` bounds how far decode may run ahead of the
        consumer in interval mode (~100MB at 256 frames of 288p)."""
        if mode not in ("interval", "keyframes"):
            raise ValueError(f"unknown sampling mode: {mode!r}")
        self.interval = interval_seconds
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.frame_size = frame_size
        self.mode = mode
        self.hwaccel = hwaccel
        self.queue_frames = queue_frames

    @property
    def interval_signature(self) -> float:
        """Value stored/checked in the embedding cache so a sampling-config
        change invalidates old entries. Keyframe mode uses -1.0 (its spacing
        is encode-dependent, not an interval)."""
        return -1.0 if self.mode == "keyframes" else self.interval

    # --- probing ---------------------------------------------------------------

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

    # --- command assembly -------------------------------------------------------

    def _scale_filter(self) -> str:
        if not self.frame_size:
            return ""
        # short side -> frame_size, other side scales up to keep aspect
        return (
            f"scale=w={self.frame_size}:h={self.frame_size}"
            ":force_original_aspect_ratio=increase:force_divisible_by=2"
        )

    def _input_args(self) -> list[str]:
        args = []
        if self.hwaccel:
            args += ["-hwaccel", self.hwaccel]
        return args

    def _vf(self) -> str:
        """The ffmpeg filtergraph for the current mode."""
        if self.mode == "keyframes":
            parts = ["showinfo"]  # exact pts_time per emitted frame
        else:
            parts = [f"fps=1/{self.interval:g}"]
        scale = self._scale_filter()
        if scale:
            parts.append(scale)
        return ",".join(parts)

    # --- single frame (labeler) ---------------------------------------------------

    def grab_frame(self, path: str, time: float) -> "Image":
        """Decode a single frame at `time` seconds (used by the labeler).

        `-ss` before `-i` with re-encoding is frame-accurate in modern ffmpeg
        (it seeks to the prior keyframe, then decodes forward to the target).
        """
        from PIL import Image as PILImage  # lazy

        cmd = [
            self.ffmpeg,
            "-v", "error",
            *self._input_args(),
            "-ss", f"{time:g}",
            "-i", path,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, check=True)
        except FileNotFoundError as exc:
            raise SamplerError(f"{self.ffmpeg} not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise SamplerError(
                f"ffmpeg frame grab failed for {path}@{time}s: {exc.stderr[:200]}"
            ) from exc
        if not out.stdout:
            raise SamplerError(f"no frame decoded for {path}@{time}s")
        return PILImage.open(BytesIO(out.stdout)).convert("RGB")

    # --- bulk sampling --------------------------------------------------------------

    def iter_frames(self, path: str) -> Iterator[tuple[float, "Image"]]:
        """Yield (timestamp_seconds, PIL.Image) samples for a video."""
        if self.mode == "keyframes":
            yield from self._iter_frames_keyframes(path)
        else:
            yield from self._iter_frames_interval(path)

    def _iter_frames_interval(self, path: str) -> Iterator[tuple[float, "Image"]]:
        """Stream frames over a pipe: no temp files, and ffmpeg decodes ahead
        (up to `queue_frames`) while the consumer embeds. Frame i is from
        source time ~i*interval — timestamps come from the frame index."""
        from PIL import Image as PILImage  # lazy

        if self.interval <= 0:
            return
        cmd = [
            self.ffmpeg,
            "-v", "error",
            *self._input_args(),
            "-i", path,
            "-vf", self._vf(),
            "-q:v", "3",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError as exc:
            raise SamplerError(f"{self.ffmpeg} not found on PATH") from exc

        frames: Queue = Queue(maxsize=self.queue_frames)
        stderr_buf = bytearray()

        def _read_frames():
            try:
                for jpg in iter_jpegs(proc.stdout):
                    frames.put(jpg)
            finally:
                frames.put(None)  # sentinel: stream finished (or died)

        def _drain_stderr():
            for line in proc.stderr:
                stderr_buf.extend(line)

        threading.Thread(target=_read_frames, daemon=True).start()
        threading.Thread(target=_drain_stderr, daemon=True).start()

        i = 0
        try:
            while True:
                jpg = frames.get()
                if jpg is None:
                    break
                with PILImage.open(BytesIO(jpg)) as im:
                    yield round(i * self.interval, 3), im.convert("RGB")
                i += 1
            rc = proc.wait()
            if rc != 0:
                raise SamplerError(
                    f"ffmpeg failed for {path}: "
                    f"{stderr_buf.decode(errors='replace')[:300]}"
                )
            if i == 0:
                raise SamplerError(f"no frames decoded for {path}")
        finally:
            if proc.poll() is None:  # consumer bailed early: stop decoding
                proc.kill()
                proc.wait()

    def _iter_frames_keyframes(self, path: str) -> Iterator[tuple[float, "Image"]]:
        """Keyframe-only pass: `-skip_frame nokey` decodes ~1-5% of frames.
        Exact timestamps are parsed from showinfo output (which logs at info
        level, hence `-v info` + post-run parse rather than streaming)."""
        from PIL import Image as PILImage  # lazy

        tmpdir = Path(tempfile.mkdtemp(prefix="peaks-frames-"))
        try:
            cmd = [
                self.ffmpeg,
                "-v", "info",  # showinfo logs at info level
                "-nostats",
                *self._input_args(),
                "-skip_frame", "nokey",
                "-i", path,
                "-vf", self._vf(),
                "-fps_mode", "passthrough",
                "-q:v", "3",
                str(tmpdir / "f-%06d.jpg"),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except FileNotFoundError as exc:
                raise SamplerError(f"{self.ffmpeg} not found on PATH") from exc
            except subprocess.CalledProcessError as exc:
                raise SamplerError(
                    f"ffmpeg failed for {path}: {exc.stderr[-300:]}"
                ) from exc

            times = parse_showinfo_times(proc.stderr)
            files = sorted(tmpdir.glob("f-*.jpg"))
            if abs(len(times) - len(files)) > 2:
                raise SamplerError(
                    f"keyframe pts/frame mismatch for {path}: "
                    f"{len(times)} pts vs {len(files)} frames"
                )
            for ts, fp in zip(times, files):
                with PILImage.open(fp) as im:
                    yield ts, im.convert("RGB")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
