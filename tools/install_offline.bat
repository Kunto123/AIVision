@echo off
REM VisionInspect — Offline Install (jalankan di edge PC, TANPA internet)
REM
REM Prasyarat:
REM   - Folder ini berisi: wheels/, hf_cache/, requirements.txt
REM   - Python 3.10+ sudah terinstall dan di PATH
REM
REM Cara:
REM   install.bat
REM   lalu run.bat

setlocal
set BUNDLE_DIR=%~dp0
set PROJECT_DIR=%BUNDLE_DIR%..\..\

echo === VisionInspect — Offline Install ===
echo Bundle dir    : %BUNDLE_DIR%
echo Target proyek : %PROJECT_DIR%
echo.

REM 1. Cek Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Python tidak ditemukan di PATH.
    echo   Install Python 3.10+ dulu dari https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 2. Buat virtual environment di project
echo [1/4] Membuat virtual environment...
if not exist "%PROJECT_DIR%.vision" (
    python -m venv "%PROJECT_DIR%.vision"
) else (
    echo   .vision sudah ada, skip
)

REM 3. Install wheels dari folder lokal (offline)
echo [2/4] Menginstall wheels dari offline_bundle...
"%PROJECT_DIR%.vision\Scripts\python.exe" -m pip install ^
    --no-index ^
    --find-links="%BUNDLE_DIR%wheels" ^
    -r "%BUNDLE_DIR%requirements.txt"
if %ERRORLEVEL% neq 0 (
    echo ⚠️  Ada error saat install. Beberapa wheel mungkin tidak terdownload lengkap.
    echo    Cek folder wheels/ dan ulangi prepare_offline_bundle.bat di PC development.
    pause
    exit /b %ERRORLEVEL%
)
echo ✅ Dependencies terinstall

REM 4. Copy HuggingFace cache (backbone weights)
echo [3/4] Meng-cache backbone weights (HuggingFace)...
if exist "%BUNDLE_DIR%hf_cache" (
    set HF_CACHE_DIR=%USERPROFILE%\.cache\huggingface\hub
    if not exist "%HF_CACHE_DIR%" mkdir "%HF_CACHE_DIR%"
    xcopy /E /I /Y "%BUNDLE_DIR%hf_cache\*" "%HF_CACHE_DIR%" >nul
    echo ✅ Weights di-cache ke %HF_CACHE_DIR%
) else (
    echo ⚠️  Folder hf_cache tidak ditemukan. Training mungkin gagal nanti.
)

REM 5. Set environment variable untuk offline mode
echo [4/4] Konfigurasi environment offline...
echo.
echo Setelah ini, jalankan aplikasi via run.bat
echo Atau set permanen: setx HF_HUB_OFFLINE 1
echo.
echo === Install selesai! ===
echo.
echo Jalankan: run.bat
