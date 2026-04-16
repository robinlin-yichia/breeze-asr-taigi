# syntax=docker/dockerfile:1.7
# Taigi ASR — CUDA 12.1 runtime + Python 3.11 + ffmpeg
# Target: WSL2 Ubuntu with NVIDIA Container Toolkit
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TAIGI_ASR_HOST=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy packaging metadata first so dependency layers cache across source edits.
COPY pyproject.toml README.md requirements.txt requirements-dev.txt ./
COPY src/ ./src/

# CUDA 12.1 PyTorch wheels (bitsandbytes for HF int8 only works on Linux).
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -e ".[hf]"

# Ship non-essential assets last so editing them doesn't invalidate the deps.
COPY data/ ./data/
COPY examples/ ./examples/

EXPOSE 7860

# Healthcheck hits the Gradio root so `docker compose ps` shows real status.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1

CMD ["python", "-m", "taigi_asr.ui.launcher", "--host", "0.0.0.0", "--no-browser"]
