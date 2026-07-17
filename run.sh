#!/bin/bash
# VisionInspect — Runner (WSL / Linux)
# Otomatis buat .venv + install deps jika belum ada.
# Usage: ./run.sh
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "⏳ Virtual env belum ada, setup otomatis..."
    python3 --version || { echo "❌ Python3 tidak ditemukan"; exit 1; }
    python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt
    echo "✅ Siap"
fi

export HF_HUB_OFFLINE=1
exec .venv/bin/python run.py "$@"
