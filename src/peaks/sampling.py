"""Frame sampling.

Three modes:

  "sparse"    — THE FAST PATH for long videos: seek to each sample time and
                decode exactly one keyframe (PyAV, in-process). Cost scales
                with the number of SAMPLES, not the video's duration — unlike
                every other mode, which must decode/discard the whole file.
                Timestamps are the decoded keyframe's exact pts, so samples
                snap to keyframes (typically within a couple of seconds of the
                grid). Requires the `av` package (bundled ffmpeg libs).

  "interval"  — one frame every `interval_seconds`, streamed out of ffmpeg
                over a pipe through a bounded queue. Decodes the ENTIRE video
                to keep a fraction of it — fine for short clips, brutal for
                hour-long files. Frame i comes from source time ~i*interval.

  "keyframes" — decode only keyframes (`-skip_frame nokey`): far less decode
                than interval, but still reads/parses the whole file and
                yields every keyframe (dense-GOP files gain little).

Optional `hwaccel` ("cuda"/"auto") offloads interval-mode decode to the GPU
(NVDEC). Sparse mode doesn't need it: it decodes ~one I-frame per sample.

The pure helpers (`plan_timestamps`, `iter_jpegs`, `iter_fixed_chunks`,
`parse_showinfo_times`) are unit-tested offline; sparse mode is tested against
real generated video (PyAV bundles codecs); the ffmpeg-binary paths need a
real box.
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


def iter_fixed_chunks(stream: IO[bytes], chunk_size: int) -> Iterator[bytes]:
    """Yield exact `chunk_size` slices from a stream (rawvideo framing).

    Handles short reads; a trailing partial chunk (truncated stream) is
    dropped rather than yielded corrupt.
    """
    buf = bytearray()
    while True:
        data = stream.read(chunk_size - len(buf) if len(buf) < chunk_size else chunk_size)
        if not data:
            break
        buf.extend(data)
        while len(buf) >= chunk_size:
            yield bytes(buf[:chunk_size])
            del buf[:chunk_size]


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
        pipeline: str = "raw",
    ):
        """`frame_size`: short-side pixels for the JPEG pipeline (0 = original
        size). `queue_frames` bounds how far decode runs ahead of the consumer.
        `pipeline`: "raw" pipes raw RGB frames pre-sized to the model's input
        straight to the embedder — no JPEG encode, no PIL decode, no CPU
        resize (the fast path). "jpeg" is the legacy/escape-hatch path."""
        if mode not in ("sparse", "interval", "keyframes"):
            raise ValueError(f"unknown sampling mode: {mode!r}")
        if pipeline not in ("raw", "jpeg"):
            raise ValueError(f"unknown pipeline: {pipeline!r}")
        self.interval = interval_seconds
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.frame_size = frame_size
        self.mode = mode
        self.hwaccel = hwaccel
        self.queue_frames = queue_frames
        self.pipeline = pipeline

    @property
    def interval_signature(self) -> float:
        """Value stored/checked in the embedding cache so a sampling-config
        change invalidates old entries. Keyframe mode uses -1.0 (its spacing
        is encode-dependent, not an interval); sparse encodes as -(100 +
        interval) so each sparse grid is distinct from every interval grid."""
        if self.mode == "keyframes":
            return -1.0
        if self.mode == "sparse":
            return -(100.0 + self.interval)
        return self.interval

    @property
    def wants_raw(self) -> bool:
        """True when this sampler produces numpy frames for embed_array.
        Sparse mode is always raw (it decodes straight to arrays); interval
        mode honours the pipeline setting; keyframes stays on the PIL path."""
        return self.mode == "sparse" or (
            self.mode == "interval" and self.pipeline == "raw"
        )

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

    def _raw_vf(self, resize_short: int, crop: int) -> str:
        """Filtergraph for the raw pipeline: sample, then reproduce the
        model's preprocessing in ffmpeg (bicubic short-side resize + center
        crop), so frames arrive at exactly the network's input size."""
        return (
            f"fps=1/{self.interval:g},"
            f"scale=w={resize_short}:h={resize_short}"
            ":force_original_aspect_ratio=increase:flags=bicubic,"
            f"crop={crop}:{crop}"
        )

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

    def iter_frames_raw(self, path: str, *, resize_short: int, crop: int):
        """Yield (timestamp, HxWx3 uint8 numpy frame) at the model's input
        geometry — the raw path. Dispatches on mode: sparse seeks per sample;
        interval streams a full decode over a pipe."""
        if self.mode == "sparse":
            yield from self._iter_frames_sparse(
                path, resize_short=resize_short, crop=crop
            )
        else:
            yield from self._iter_frames_raw_interval(
                path, resize_short=resize_short, crop=crop
            )

    def _iter_frames_sparse(self, path: str, *, resize_short: int, crop: int):
        """Seek-based sampling: for each grid time, seek to the prior keyframe
        and decode just that one frame. Decodes ~one I-frame per sample, so a
        56-minute file at interval=8 costs ~420 tiny decodes instead of a
        ~100k-frame full decode. Timestamps are the keyframes' exact pts.

        When the grid is finer than the keyframe spacing, consecutive seeks
        land on the same keyframe — duplicates are skipped, so the effective
        density is min(grid, GOP)."""
        import numpy as np  # lazy

        try:
            import av  # lazy: the `av` extra dependency
        except ImportError as exc:  # pragma: no cover - guarded in CLI too
            raise SamplerError(
                "sparse mode needs the 'av' package (pip install av)"
            ) from exc

        if self.interval <= 0:
            return

        with av.open(path) as container:
            if not container.streams.video:
                raise SamplerError(f"no video stream in {path}")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            try:  # decode keyframes only — big speedup, and seeks land on them
                stream.codec_context.skip_frame = "NONKEY"
            except (AttributeError, ValueError):  # pragma: no cover
                pass

            tb = stream.time_base
            if stream.duration and tb:
                duration = float(stream.duration * tb)
            elif container.duration:
                duration = container.duration / av.time_base
            else:
                raise SamplerError(f"could not determine duration of {path}")

            interp_kw: dict = {"interpolation": "BICUBIC"}
            last_pts = None
            yielded = 0
            t = 0.0
            while t < duration:
                target = t
                t += self.interval
                try:
                    container.seek(int(target / tb), stream=stream)
                    frame = next(container.decode(stream), None)
                except StopIteration:  # pragma: no cover
                    break
                except Exception:
                    continue  # unseekable spot / decode hiccup: skip sample
                if frame is None:
                    break
                if frame.pts is not None and last_pts is not None and frame.pts <= last_pts:
                    continue  # same keyframe as the previous sample
                last_pts = frame.pts
                ts = frame.time if frame.time is not None else target

                # reproduce the model preprocessing: bicubic short-side resize
                # then center crop, all inside libswscale + numpy
                scale = resize_short / min(frame.width, frame.height)
                nw = max(crop, int(round(frame.width * scale)))
                nh = max(crop, int(round(frame.height * scale)))
                try:
                    out = frame.reformat(
                        width=nw, height=nh, format="rgb24", **interp_kw
                    )
                except TypeError:  # older PyAV without interpolation kwarg
                    interp_kw = {}
                    out = frame.reformat(width=nw, height=nh, format="rgb24")
                arr = out.to_ndarray()
                y0 = (nh - crop) // 2
                x0 = (nw - crop) // 2
                arr = np.ascontiguousarray(
                    arr[y0 : y0 + crop, x0 : x0 + crop]
                )
                yield round(float(ts), 3), arr
                yielded += 1
            if yielded == 0:
                raise SamplerError(f"no frames decoded for {path}")

    def _iter_frames_raw_interval(self, path: str, *, resize_short: int, crop: int):
        """Fast path: yield (timestamp, HxWx3 uint8 numpy frame) with frames
        already at the model's input geometry.

        ffmpeg decodes (optionally NVDEC), resizes and crops, and writes raw
        RGB24 to a pipe. Framing is trivial — every frame is exactly
        crop*crop*3 bytes — so the Python side is a memcpy, not a JPEG decode.
        A reader thread keeps decode running ahead of the consumer.
        """
        import numpy as np  # lazy: keeps module import light

        if self.interval <= 0:
            return
        frame_bytes = crop * crop * 3
        cmd = [
            self.ffmpeg,
            "-v", "error",
            *self._input_args(),
            "-i", path,
            "-vf", self._raw_vf(resize_short, crop),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
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
                for chunk in iter_fixed_chunks(proc.stdout, frame_bytes):
                    frames.put(chunk)
            finally:
                frames.put(None)

        def _drain_stderr():
            for line in proc.stderr:
                stderr_buf.extend(line)

        threading.Thread(target=_read_frames, daemon=True).start()
        threading.Thread(target=_drain_stderr, daemon=True).start()

        i = 0
        try:
            while True:
                chunk = frames.get()
                if chunk is None:
                    break
                arr = np.frombuffer(chunk, dtype=np.uint8).reshape(crop, crop, 3)
                yield round(i * self.interval, 3), arr
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
        level, hence `-v info` + post-run parse rather than streaming).

        Note: `-skip_frame nokey` is a *software* decoder feature; NVDEC/cuvid
        ignores it and would decode every frame, defeating the whole point. So
        this path deliberately does NOT use hwaccel — sparse I-frame decode on
        the CPU is already cheap (it's skipping ~95% of frames), and the GPU is
        still used for the embedding itself."""
        from PIL import Image as PILImage  # lazy

        tmpdir = Path(tempfile.mkdtemp(prefix="peaks-frames-"))
        try:
            cmd = [
                self.ffmpeg,
                "-v", "info",  # showinfo logs at info level
                "-nostats",
                "-skip_frame", "nokey",  # CPU decoder skips non-key frames
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
