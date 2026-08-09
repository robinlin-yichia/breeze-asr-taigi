@echo off
chcp 65001 >nul
REM ==========================================================
REM  Standalone Speaker Diarization UI  ->  http://127.0.0.1:7861
REM
REM  Only needed to label an EXISTING transcript (audio + SRT/JSON
REM  you already have). For a fresh recording just use start.bat
REM  and tick "標記發言者" — the main UI now does both in one pass.
REM
REM  Runs from the same .venv as the ASR UI.
REM ==========================================================

cd /d "%~dp0"

set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=%~dp0venv\Scripts\python.exe"

if not exist "%PYEXE%" (
    echo [ERROR] Python venv not found.
    echo         Expected: %~dp0.venv\Scripts\python.exe
    echo         Run install.bat first.
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0diarize_app.py" (
    echo [ERROR] diarize_app.py not found in %~dp0
    echo.
    pause
    exit /b 1
)

if "%HF_TOKEN%"=="" (
    echo [WARN] HF_TOKEN environment variable is not set.
    echo        You can still paste the token into the UI field.
    echo.
)

echo [INFO] Launching Speaker Diarization UI ...
echo [INFO] URL: http://127.0.0.1:7861
echo [INFO] Press Ctrl+C in this window to stop.
echo.

"%PYEXE%" "%~dp0diarize_app.py"

set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [ERROR] Exited with code %RC%. See the message above.
    pause
)
