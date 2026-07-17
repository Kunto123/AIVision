@echo off
REM VisionInspect — One-Time Setup
REM Menyiapkan environment Windows (.vision) dan WSL (.venv) sekali saja.
REM
REM Usage:
REM   setup.bat              -> setup Windows + WSL
REM   setup.bat --windows    -> setup Windows saja
REM   setup.bat --wsl        -> setup WSL saja
REM

setlocal
set "PROJECT_DIR=%~dp0"

echo === VisionInspect — Setup ===
echo Project: %PROJECT_DIR%
echo.

REM ── Cek Python di PATH ──
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Python tidak ditemukan di PATH.
    echo    Install Python 3.10+ dari https://www.python.org/downloads/
    pause
    exit /b 1
)

set "SETUP_WINDOWS=1"
set "SETUP_WSL=1"
if /i "%1"=="--windows" set "SETUP_WSL=0"
if /i "%1"=="--wsl" set "SETUP_WINDOWS=0"

REM ═══════════════════════════════════════════════
REM  Windows: .vision\ venv
REM ═══════════════════════════════════════════════
if "%SETUP_WINDOWS%"=="1" (
    echo [1/2] Windows — .vision\ ...
    if not exist "%PROJECT_DIR%.vision\Scripts\python.exe" (
        echo   Membuat virtual environment...
        python -m venv "%PROJECT_DIR%.vision"
        if %ERRORLEVEL% neq 0 (
            echo ❌ Gagal membuat .vision venv
            pause
            exit /b %ERRORLEVEL%
        )
    ) else (
        echo   .vision sudah ada, skip
    )

    echo   Menginstall dependencies...
    "%PROJECT_DIR%.vision\Scripts\python.exe" -m pip install -q -r "%PROJECT_DIR%requirements.txt"
    if %ERRORLEVEL% neq 0 (
        echo ⚠️  Install warning, lanjut...
    ) else (
        echo ✅ Windows dependencies siap
    )
)

REM ═══════════════════════════════════════════════
REM  WSL: .venv/ venv
REM ═══════════════════════════════════════════════
if "%SETUP_WSL%"=="1" (
    echo [2/2] WSL — .venv/ ...
    where wsl >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo ⚠️  WSL tidak ditemukan, skip
    ) else (
        REM Konversi path Windows → WSL path
        set "DRIVE=%PROJECT_DIR:~0,1%"
        set "WSL_PATH=/mnt/%DRIVE%/%PROJECT_DIR:~3%"
        set "WSL_PATH=%WSL_PATH:\=/%"

        wsl -e bash -c "
            cd '%WSL_PATH%' && \
            if [ ! -d '.venv' ]; then \
                echo '  Membuat .venv/...'; \
                python3 -m venv .venv; \
            else \
                echo '  .venv sudah ada, skip'; \
            fi && \
            echo '  Menginstall dependencies...' && \
            .venv/bin/pip install -q -r requirements.txt && \
            echo '✅ WSL dependencies siap'
        "
    )
)

echo.
echo === Setup selesai! ===
echo.
echo Cara menjalankan:
echo   Windows: run.bat
echo   WSL:     wsl -e bash -c "cd %WSL_PATH% ^&^& ./run.sh"
echo.
pause
