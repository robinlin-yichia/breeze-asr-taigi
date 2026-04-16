#!/usr/bin/env bash
# Launch Taigi ASR Gradio UI and auto-open browser.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .venv/bin/activate ]]; then
    echo "[ERROR] .venv not found. Run ./install.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
exec python -m taigi_asr.ui.launcher "$@"
