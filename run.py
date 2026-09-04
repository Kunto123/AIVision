#!/usr/bin/env python3
"""
VisionInspect — Runner.
Jalankan aplikasi dari folder proyek: `python run.py` (atau `run.bat` di Windows).
"""

import os
import sys
import platform
from pathlib import Path

# 1. Deteksi WSL → paksa software rendering (hindari error EGL/MESA/ZINK)
_on_wsl = False
if platform.system() == "Linux":
    try:
        with open("/proc/version") as f:
            _on_wsl = "microsoft" in f.read().lower()
    except Exception:
        pass

if _on_wsl:
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
        print("INFO: Running in WSL tanpa display server.",
              "Install VcXsrv/X410 di Windows lalu export DISPLAY=:0")

# 2. Arahkan folder data ke <proyek>/data (konsisten Windows & WSL)
#    Override lewat env VISIONINSPECT_DATA kalau perlu.
_project_root = Path(__file__).resolve().parent
_data_dir = str(_project_root / "data")
if "VISIONINSPECT_DATA" not in os.environ:
    os.environ["VISIONINSPECT_DATA"] = _data_dir

# 3. Package root ke sys.path, lalu jalankan
sys.path.insert(0, str(_project_root))

from visioninspect.main import main

if __name__ == "__main__":
    sys.exit(main())
