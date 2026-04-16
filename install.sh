#!/usr/bin/env bash
# ============================================================================
# Taigi ASR — one-click installer (Linux / WSL2 / macOS)
#
# Usage:
#   ./install.sh                 native venv install
#   ./install.sh --docker        build and start docker-compose stack
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")"

echo "================================================================"
echo "  Taigi ASR Installer"
echo "================================================================"

if [[ "${1:-}" == "--docker" ]]; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "[ERROR] docker not found. Install Docker + NVIDIA Container Toolkit first."
        exit 1
    fi
    echo "[docker] Building image..."
    docker compose build
    echo "[docker] Starting container..."
    docker compose up -d
    echo ""
    echo "Taigi ASR is running at http://localhost:7860"
    exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[WARN] ffmpeg not found. Install with:"
    echo "       sudo apt install -y ffmpeg   (Ubuntu/Debian/WSL2)"
    echo "       brew install ffmpeg          (macOS)"
    echo "       (continuing with pydub fallback — slower)"
fi

echo "[1/4] Creating virtualenv .venv ..."
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/4] Upgrading pip ..."
python -m pip install --upgrade pip >/dev/null

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[3a/4] Installing CUDA 12.1 PyTorch ..."
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    echo "[3a/4] CUDA not detected, installing CPU PyTorch ..."
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "[3b/4] Installing taigi-asr (editable + HF extras + dev tools) ..."
python -m pip install -e ".[hf,dev]"

echo "[4/4] Pre-downloading Breeze-ASR-26 model (~2.9 GB) ..."
python -c "from taigi_asr.engines.faster_whisper import FasterWhisperEngine; FasterWhisperEngine.preload()" \
    || echo "   (preload skipped; will download on first run)"

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [[ -f .pre-commit-config.yaml ]] && command -v pre-commit >/dev/null 2>&1; then
    echo "[bonus] Installing pre-commit hooks ..."
    pre-commit install || true
fi

echo ""
echo "================================================================"
echo "  Install finished. Launch with:"
echo "      ./start.sh        (opens browser automatically)"
echo "  or:"
echo "      taigi-asr-ui"
echo "================================================================"
