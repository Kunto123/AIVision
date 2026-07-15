#!/usr/bin/env python3
"""
VisionInspect — Runner script.
Gunakan ini untuk menjalankan aplikasi dari folder proyek.
"""

import os
import sys
from pathlib import Path

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
