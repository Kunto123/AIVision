@echo off
REM VisionInspect — Retrain via WSL (PyTorch jalan di WSL, bukan Windows)
REM
REM Flow:
REM   1. Capture OK/NG di Windows (kamera jalan)
REM   2. Tutup VisionInspect
REM   3. Jalankan: retrain_wsl.bat [Program] [TemplateID]
REM   4. Buka lagi VisionInspect — model baru siap
REM
REM Usage:
REM   retrain_wsl.bat                       -> menu interaktif
REM   retrain_wsl.bat Default template_1    -> langsung train
REM

setlocal

REM ── Cari project root (folder tempat batch ini berada) ──
set "PROJECT_DIR=%~dp0"
REM Hapus trailing backslash
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

REM ── Konversi Windows path → WSL path (contoh: C:\Proj -> /mnt/c/Proj) ──
set "DRIVE=%PROJECT_DIR:~0,1%"
set "WSL_PATH=/mnt/%DRIVE%/%PROJECT_DIR:~3%"
set "WSL_PATH=%WSL_PATH:\=/%"

set WSL_VENV=%WSL_PATH%/.venv

REM ── Cek argumen ──
if not "%1"=="" if not "%2"=="" goto :train_direct

REM ── Mode interaktif ──
echo === VisionInspect — Retrain via WSL ===
echo Project : %PROJECT_DIR%
echo WSL path: %WSL_PATH%
echo.

REM Cek WSL + venv
wsl -e bash -c "test -f '%WSL_VENV%/bin/activate' && echo 'WSL venv: OK' || echo 'WSL venv: NOT FOUND'"
echo.

echo  Masukkan parameter template yang akan di-retrain.
echo.
set /p "PROG=Program [Default]: "
if "%PROG%"=="" set "PROG=Default"
set /p "TMPL=Template ID [template_1]: "
if "%TMPL%"=="" set "TMPL=template_1"
goto :run

:train_direct
set "PROG=%1"
set "TMPL=%2"
:run

echo.
echo 🧠 Training via WSL...
echo   Program  : %PROG%
echo   Template : %TMPL%
echo   WSL path : %WSL_PATH%
echo.

wsl -e bash -c "
    source '%WSL_VENV%/bin/activate' && \
    cd '%WSL_PATH%' && \
    python tools/train_cli.py --program '%PROG%' --template '%TMPL%'
"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ✅ Training selesai! Restart VisionInspect untuk memuat model baru.
) else (
    echo.
    echo ❌ Training gagal. Periksa:
    echo   1. WSL sudah di-setup?
    echo      - wsl
    echo      - cd %WSL_PATH%
    echo      - python3 -m venv .venv ^&^& source .venv/bin/activate ^&^& pip install -r requirements.txt
    echo   2. Template ID benar? Cek folder data\programs\%PROG%\templates\
)

echo.
pause
