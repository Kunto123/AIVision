@echo off
REM VisionInspect — Runner (Windows)
REM Pertama kali dijalankan: otomatis buat venv .vision\ + install requirements.
REM Argumen apa pun diteruskan ke run.py (mis. run.bat --log-level DEBUG).

setlocal
set "PROJECT_DIR=%~dp0"

REM 1. Bootstrap venv kalau belum ada
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

REM 2. Jalankan (HF_HUB_OFFLINE=1 → HuggingFace tidak diakses saat runtime)
set HF_HUB_OFFLINE=1
"%PROJECT_DIR%.vision\Scripts\python.exe" "%PROJECT_DIR%run.py" %*
