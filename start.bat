@echo off
REM Launch Taigi ASR Gradio UI and auto-open browser.
setlocal enableextensions
chcp 65001 >nul
cd /d "%~dp0"

if not exist .venv\Scripts\activate.bat (
    echo [ERROR] .venv not found. Run install.bat first.
    exit /b 1
)
call .venv\Scripts\activate.bat

python -m taigi_asr.ui.launcher %*
