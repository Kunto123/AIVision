"""
VisionInspect - Utility functions
"""

import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any


def normalize_wsl_path(path_str: str) -> Path:
    """Konversi path Windows (C:\\foo) → WSL (/mnt/c/foo). Perlu karena di WSL
    Path('C:\\foo') dianggap relative (tidak diawali '/')."""
    p = Path(path_str)

    # Cek: apakah ini path Windows? (C:\, D:\, d:\)
    if len(path_str) > 1 and path_str[1] == ':':
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/').lstrip('/')
        return Path(f'/mnt/{drive}/{rest}')

    # Cek: path yang mengandung backslash (C:\foo\bar disimpan tanpa colon?)
    if '\\' in path_str:
        # Coba parse sebagai Windows path
        cleaned = path_str.replace('\\', '/')
        return Path(cleaned)

    return p


def is_wsl() -> bool:
    """Detect if running inside WSL."""
    if platform.system() != 'Linux':
        return False
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower() or 'wsl' in f.read().lower()
    except Exception:
        return False


def atomic_write(path: Path, data: Any) -> None:
    """
    Atomic JSON write: write to temp file, then rename.
    Mencegah korupsi data jika crash saat menulis.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    )
    try:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        try:
            os.fsync(tmp.fileno())
        except (OSError, AttributeError):
            pass
        tmp.close()
        os.replace(tmp.name, str(path))
    except Exception:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise


def load_json(path: Path, default: Any = None) -> Any:
    """Load JSON file safely."""
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def sanitize_filename(name: str) -> str:
    """Sanitize string for use as filename."""
    import re
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip())


def format_timestamp(ts: float = None) -> str:
    """Format timestamp to string."""
    import time
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))
