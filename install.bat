@echo off
REM ============================================================================
REM  Taigi ASR - One-click installer (Windows)
REM ----------------------------------------------------------------------------
REM  Steps:
REM    1. Create .venv
REM    2. Install CUDA-enabled PyTorch (cu121)
REM    3. Install taigi-asr + HF extras + dev tools
REM    4. Pre-download Breeze-ASR-26 CT2 model (~2.9 GB)
REM ============================================================================
setlocal enableextensions

chcp 65001 >nul
cd /d "%~dp0"

echo.
echo ================================================================
echo   Taigi ASR Installer
echo ================================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.10+ from
    echo         https://www.python.org/downloads/  and rerun.
    exit /b 1
)

REM ffmpeg check (not fatal; pydub fallback exists but much slower)
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [WARN] ffmpeg not found on PATH. Strongly recommended:
    echo        winget install Gyan.FFmpeg
    echo        (continuing anyway, using pydub fallback)
    echo.
)

echo [1/4] Creating virtualenv .venv ...
if not exist .venv (
    python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat

echo [2/4] Installing CUDA 12.1 PyTorch ...
python -m pip install --upgrade pip >nul
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 || goto :error

echo [3/4] Installing taigi-asr (editable + HF extras + dev tools) ...
python -m pip install -e ".[hf,dev]" || goto :error

echo [4/4] Pre-downloading Breeze-ASR-26 model (~2.9 GB) ...
python -c "from taigi_asr.engines.faster_whisper import FasterWhisperEngine; FasterWhisperEngine.preload()" || echo    (preload skipped; will download on first run)

echo.
echo ================================================================
echo   Install finished. Launch with:
echo       start.bat        (opens browser automatically)
echo   or:
echo       taigi-asr-ui
echo ================================================================
echo.
exit /b 0

:error
echo.
echo [ERROR] Install failed. See messages above.
exit /b 1
