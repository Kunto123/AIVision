#!/usr/bin/env python3
"""
VisionInspect — Runner script.
Gunakan ini untuk menjalankan aplikasi dari folder proyek.
"""

import os
import sys
import platform
from pathlib import Path

# === WSL-specific fixes ===
_on_wsl = False
if platform.system() == "Linux":
    try:
        with open("/proc/version") as f:
            _on_wsl = "microsoft" in f.read().lower()
    except Exception:
        pass

if _on_wsl:
    # Suppress EGL/MESA/ZINK errors — pakai software rendering
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    # Pastikan display environment ada
    if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
        print("INFO: Running in WSL tanpa display server.",
              "Install VcXsrv/X410 di Windows lalu export DISPLAY=:0")

# Auto-set VISIONINSPECT_DATA ke data/ proyek (agar konsisten
# di Windows & WSL tanpa perlu config override manual)
_project_root = Path(__file__).resolve().parent
_data_dir = str(_project_root / "data")
if "VISIONINSPECT_DATA" not in os.environ:
    os.environ["VISIONINSPECT_DATA"] = _data_dir

# Add project root to path
sys.path.insert(0, str(_project_root))

from visioninspect.main import main

if __name__ == "__main__":
    sys.exit(main())
