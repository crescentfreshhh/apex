# peaks (Opus) — container image for Unraid / Docker
#
# CUDA-enabled torch wheels are installed first (heavy, rarely change) so code
# updates rebuild fast. ffmpeg comes from Debian (includes cuda/nvdec hwaccel).
# Model weights (DINOv2/CLIP) download on first use into /config so they
# persist across container recreates.

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# heavy ML deps pinned to CUDA 12.4 wheels (work on any recent driver via the
# nvidia container runtime; also run fine on CPU)
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

WORKDIR /opt/peaks
COPY pyproject.toml README.md config.example.toml ./
COPY src ./src
COPY webapp ./webapp
RUN pip install --no-cache-dir ".[ml,label]"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# persist model downloads + all working data under the /config volume
ENV TORCH_HOME=/config/torch \
    HF_HOME=/config/hf

WORKDIR /config
EXPOSE 8800 7860
ENTRYPOINT ["/entrypoint.sh"]
# default process: keep the megaboard served; everything else runs via console
CMD ["peaks", "serve", "--host", "0.0.0.0", "--port", "8800", "--directory", "/config/webapp"]
