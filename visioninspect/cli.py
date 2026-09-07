#!/usr/bin/env python3
"""
VisionInspect - CLI konsol (dibundel jadi VisionInspect-cli.exe).
Sub-perintah db.txt tanpa memuat GUI / torch:
  --check-db                  validasi db.txt (koneksi, tabel, mapping)
  --encrypt-secret "<teks>"   cetak token enc:v2: untuk DB_PASSWORD
  --db-user-add U P [ROLE]    tambah akun ke tabel user DB
"""

import sys
from pathlib import Path

# 1. Package root ke sys.path (dev; frozen sudah include package)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from visioninspect.storage import db_cli

if __name__ == "__main__":
    sys.exit(db_cli.main())
