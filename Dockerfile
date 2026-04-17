# syntax=docker/dockerfile:1.7
# Taigi ASR - CUDA 12.1 runtime + Python 3.11 + ffmpeg
# Target: WSL2 Ubuntu with NVIDIA Container Toolkit
#
# Layer strategy (source->install ordering is deliberate so editing src/
# only invalidates the small final install layer, not the 2 GB torch layer):
#   1. apt system packages    (rarely changes)
#   2. PyTorch CUDA wheel     (pinned to cu121, ~2 GB)
#   3. project metadata + src (invalidated on any repo change)
#   4. runtime assets         (data/ + examples/)
# Every pip layer uses a BuildKit cache mount so wheel downloads persist
# across rebuilds even when the layer itself is invalidated.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/root/.cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TAIGI_ASR_HOST=0.0.0.0

# ---------- Layer 1: system packages ----------
# Resilient to transient apt mirror sync issues (observed in CI where
# archive.ubuntu.com returned partially-updated Packages.gz files, failing
# the SHA check mid-build). We retry `apt-get update` with exponential
# backoff and tell apt itself to retry individual HTTP fetches.
RUN set -eux; \
    for i in 1 2 3 4 5; do \
        apt-get update -o Acquire::Retries=5 -o Acquire::http::Timeout=30 && break; \
        echo "apt-get update attempt $i failed, sleeping $((i*5))s"; \
        sleep $((i*5)); \
    done; \
    apt-get install -y --no-install-recommends \
        -o Acquire::Retries=5 \
        -o Acquire::http::Timeout=30 \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# ---------- Layer 2: PyTorch CUDA wheel (~2 GB, rarely invalidated) ----------
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cu121

# ---------- Layer 3: project metadata + source + install ----------
# Copy metadata first (for completeness / inspection) then full src/ before
# install so `pip install -e .` finds the package layout.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -e ".[hf]"

# ---------- Layer 4: runtime-only assets ----------
# Separated so editing examples/ or data/ doesn't invalidate the install layer.
COPY data/ ./data/
COPY examples/ ./examples/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1

CMD ["python", "-m", "taigi_asr.ui.launcher", "--host", "0.0.0.0", "--no-browser"]
