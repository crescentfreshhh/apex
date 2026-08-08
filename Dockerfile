# peaks (Opus) — container image for Unraid / Docker
#
# CUDA-enabled torch wheels are installed first (heavy, rarely change) so code
# updates rebuild fast. ffmpeg is a static BtbN build that includes NVDEC/cuvid
# so `hwaccel = "cuda"` decode works (the stock Debian build's nvidia support
# is not guaranteed). The NVDEC driver libs are injected at runtime by the
# nvidia container runtime (NVIDIA_DRIVER_CAPABILITIES must include "video").
# Model weights (DINOv2/CLIP) download on first use into /config so they persist.
#
# torch is the CUDA 12.8 build (cu128): it carries kernels for Blackwell
# (RTX 50-series, sm_120) AND older cards (Ampere/Ada), so the same image runs
# on a 5070 or a 3080 Ti. Blackwell REQUIRES cu128 — the older cu124 wheels have
# no sm_120 kernels and fail with "no kernel image is available". Needs a recent
# driver (>= 570); on Blackwell that means the open-source kernel module.

FROM python:3.11-slim

# static ffmpeg/ffprobe with full nvidia hwaccel (nvdec, cuvid, nvenc)
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget xz-utils ca-certificates \
    && wget -qO /tmp/ffmpeg.tar.xz \
        https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
    && mkdir -p /tmp/ff && tar -xf /tmp/ffmpeg.tar.xz -C /tmp/ff --strip-components=1 \
    && cp /tmp/ff/bin/ffmpeg /tmp/ff/bin/ffprobe /usr/local/bin/ \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ff /tmp/ffmpeg.tar.xz \
    && apt-get purge -y wget xz-utils && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# heavy ML deps pinned to CUDA 12.8 wheels — cover Blackwell (sm_120) through
# Ampere, and still run fine on CPU when no GPU is present
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

WORKDIR /opt/peaks
COPY pyproject.toml README.md config.example.toml ./
COPY src ./src
COPY webapp ./webapp
RUN pip install --no-cache-dir ".[ml,label,web]"

COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/refresh_config.py /opt/peaks/refresh_config.py
RUN chmod +x /entrypoint.sh

# persist model downloads + all working data under the /config volume
ENV TORCH_HOME=/config/torch \
    HF_HOME=/config/hf

ENV PEAKS_WEBAPP_DIR=/config/webapp
WORKDIR /config
EXPOSE 8800 7860
ENTRYPOINT ["/entrypoint.sh"]
# default process: the control-panel + explorer web app (megaboard mounted at
# /megaboard). Other commands still run via the container console.
CMD ["peaks", "web", "--host", "0.0.0.0", "--port", "8800"]
