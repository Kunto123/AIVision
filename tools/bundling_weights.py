#!/usr/bin/env python3
"""
VisionInspect — Weight Bundling Script
Download pretrained backbone weights dan bundle untuk offline install.
"""

import sys
import os
from pathlib import Path

# Tambahkan project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BACKBONES = {
    "resnet18": "pytorch/vision:v0.10.0",
    "wide_resnet50_2": "pytorch/vision:v0.10.0",
}


def download_weights(target_dir: Path = None):
    """
    Download pretrained backbone weights.
    Target dir: biasanya untuk PyInstaller bundle.
    """
    import torch

    if target_dir is None:
        target_dir = Path.home() / ".cache" / "torch" / "hub"

    print(f"=== Download Backbone Weights ===")
    print(f"Target: {target_dir}")
    print()

    for backbone, repo in BACKBONES.items():
        print(f"Downloading {backbone} dari {repo}...")
        try:
            torch.hub.load(repo, backbone, pretrained=True)
            print(f"  ✅ {backbone} selesai")
        except Exception as e:
            print(f"  ❌ {backbone} gagal: {e}")

    # Tampilkan cache size
    cache_dir = Path.home() / ".cache" / "torch"
    if cache_dir.exists():
        total_size = sum(
            f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
        )
        print(f"\nTotal cache: {total_size / 1024 / 1024:.1f} MB")
        print(f"Lokasi cache: {cache_dir}")
        print("\nCopy folder ini ke PC target untuk offline use.")


def bundle_for_pyinstaller(target_dir: Path):
    """
    Copy cached weights ke folder PyInstaller bundle.
    """
    src = Path.home() / ".cache" / "torch"
    dst = target_dir / "torch_cache"

    if not src.exists():
        print("❌ Cache torch tidak ditemukan. Jalankan download_weights() dulu.")
        return

    import shutil
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"✅ Weights dibundel ke {dst}")
    print(f"   Size: {sum(f.stat().st_size for f in dst.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download backbone weights")
    parser.add_argument("--bundle", type=str, default="",
                        help="Target directory for PyInstaller bundle")
    args = parser.parse_args()

    if args.bundle:
        bundle_for_pyinstaller(Path(args.bundle))
    else:
        download_weights()
