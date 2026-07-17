@echo off
REM VisionInspect — Runner (Windows)
REM Otomatis buat .vision venv + install deps jika belum ada.
setlocal
set "PROJECT_DIR=%~dp0"

if not exist "%PROJECT_DIR%.vision\Scripts\python.exe" (
    echo ⏳ Virtual env belum ada, setup otomatis...
    python --version >nul 2>&1 || (
        echo ❌ Python tidak ditemukan di PATH
        pause
        exit /b 1
    )
    python -m venv "%PROJECT_DIR%.vision"
    "%PROJECT_DIR%.vision\Scripts\python.exe" -m pip install -q -r "%PROJECT_DIR%requirements.txt"
    echo ✅ Siap
)

set HF_HUB_OFFLINE=1
"%PROJECT_DIR%.vision\Scripts\python.exe" "%PROJECT_DIR%run.py" %*
