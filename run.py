#!/usr/bin/env python3
"""
VisionInspect — Runner script.
Gunakan ini untuk menjalankan aplikasi dari folder proyek.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from visioninspect.main import main

if __name__ == "__main__":
    sys.exit(main())
