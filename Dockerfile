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

# NOTE: the search-index RSS blow-up during an embed is fixed in code —
# SearchIndex.build preallocates its matrix and Service.index frees the old
# index before rebuilding, so a rebuild no longer holds 2-3x the matrix. glibc
# malloc tuning (MALLOC_ARENA_MAX / MALLOC_TRIM_THRESHOLD_) was tried here as
# extra insurance but roughly doubled per-scene embed time (aggressive trimming
# munmaps/re-faults on every large free in the decode loop, and fewer arenas
# serialize allocs across the decode workers), so it's deliberately NOT set. If
# a very long embed ever shows gradual RSS creep, MALLOC_ARENA_MAX=2 can be set
# as a container Variable — but expect a decode slowdown for it.

# All model caches under the /config volume so they download once and persist
# across updates: TORCH_HOME (DINOv2/torch.hub), HF_HOME (open_clip via HF hub),
# and XDG_CACHE_HOME so open_clip's ~/.cache/clip fallback for URL-hosted CLIP
# checkpoints doesn't escape to the ephemeral /root/.cache and re-download.
ENV TORCH_HOME=/config/torch \
    HF_HOME=/config/hf \
    HF_HUB_CACHE=/config/hf/hub \
    XDG_CACHE_HOME=/config/.cache

ENV PEAKS_WEBAPP_DIR=/config/webapp
WORKDIR /config
EXPOSE 8800 7860
ENTRYPOINT ["/entrypoint.sh"]
# default process: the control-panel + explorer web app (megaboard mounted at
# /megaboard). Other commands still run via the container console.
CMD ["peaks", "web", "--host", "0.0.0.0", "--port", "8800"]
