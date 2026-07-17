@echo off
REM VisionInspect — Offline Bundle Preparation (jalankan di PC development dengan internet)
REM Hasil: folder offline_bundle/ siap di-copy ke USB untuk instalasi edge PC.
REM
REM Usage:
REM   tools\prepare_offline_bundle.bat

setlocal
set PROJECT_DIR=%~dp0..
set BUNDLE_DIR=%PROJECT_DIR%\offline_bundle
set VENV_DIR=%PROJECT_DIR%\.vision

echo === VisionInspect — Prepare Offline Bundle ===
echo Project    : %PROJECT_DIR%
echo Bundle dir : %BUNDLE_DIR%
echo.

REM 1. Setup virtual environment jika belum ada
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [0/3] Virtual env belum ada, membuat .vision...
    python --version >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo ❌ Python tidak ditemukan di PATH. Install Python 3.10+ dulu.
        pause
        exit /b 1
    )
    python -m venv "%VENV_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo ❌ Gagal membuat virtual environment.
        pause
        exit /b %ERRORLEVEL%
    )
    echo ✅ Virtual environment dibuat
) else (
    echo [0/3] Virtual environment sudah ada, skip
)

REM 2. Buat folder bundle
if not exist "%BUNDLE_DIR%" mkdir "%BUNDLE_DIR%"
if not exist "%BUNDLE_DIR%\wheels" mkdir "%BUNDLE_DIR%\wheels"

REM 3. Download semua wheel (termasuk transitive dependencies)
echo [1/3] Downloading wheels from PyPI...
"%VENV_DIR%\Scripts\python.exe" -m pip download ^
    -r "%PROJECT_DIR%\requirements.txt" ^
    -d "%BUNDLE_DIR%\wheels"
if %ERRORLEVEL% neq 0 (
    echo ❌ pip download gagal, errorlevel=%ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo ✅ Wheels selesai

REM 3. Download backbone weights via timm (HuggingFace cache)
echo [2/3] Downloading backbone weights (timm/HuggingFace)...
"%VENV_DIR%\Scripts\python.exe" "%PROJECT_DIR%\tools\bundling_weights.py" ^
    --bundle "%BUNDLE_DIR%"
if %ERRORLEVEL% neq 0 (
    echo ❌ Weight download gagal
    exit /b %ERRORLEVEL%
)
echo ✅ Backbone weights selesai

REM 4. Copy install script ke bundle
echo [3/3] Copying install script...
copy "%PROJECT_DIR%\tools\install_offline.bat" "%BUNDLE_DIR%\install.bat" >nul
copy "%PROJECT_DIR%\requirements.txt" "%BUNDLE_DIR%\requirements.txt" >nul
echo ✅ Install script copied

REM 5. Ringkasan
echo.
echo === Bundle siap! ===
echo.
echo Folder: %BUNDLE_DIR%
dir /s "%BUNDLE_DIR%\wheels" 2>nul | find "File(s)"
dir /s "%BUNDLE_DIR%\hf_cache" 2>nul | find "File(s)"
echo.
echo Cara pakai di edge PC:
echo   1. Copy folder offline_bundle ke edge PC (USB)
echo   2. Jalankan: offline_bundle\install.bat
echo   3. Jalankan: run.bat
