#!/usr/bin/env python3
"""Pastikan requirements.txt & requirements-gpu.txt identik kecuali baris torch.

Exit 0 = sinkron, exit 1 = ada drift (dipakai di pre-commit / CI).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pkg_lines(path: Path, drop: str) -> list[str]:
    """Baris paket (diawali huruf), buang baris yang mengandung `drop`."""
    return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()
            if ln[:1].isalpha() and drop not in ln]


def main() -> int:
    cpu = pkg_lines(ROOT / "requirements.txt", "+cpu")
    gpu = pkg_lines(ROOT / "requirements-gpu.txt", "+cu124")
    if cpu == gpu:
        print(f"OK — {len(cpu)} paket identik di kedua file")
        return 0
    import difflib
    print("DRIFT antara requirements.txt (cpu) dan requirements-gpu.txt (gpu):")
    print("\n".join(difflib.unified_diff(cpu, gpu, "cpu", "gpu", lineterm="")))
    return 1


if __name__ == "__main__":
    sys.exit(main())
