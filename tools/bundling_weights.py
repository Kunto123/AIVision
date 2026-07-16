#!/usr/bin/env python3
"""
VisionInspect — Weight Bundling Script (Offline Deployment)
Download pretrained backbone weights via timm (jalur yang benar-benar dipakai
TimmFeatureExtractor saat training) dan bundle untuk offline install di edge PC.

Cache location: ~/.cache/huggingface/hub/ (default huggingface_hub)
BUKAN ~/.cache/torch/hub/ (torch.hub — jalur LAMA yang tidak dipakai runtime).

Usage:
    # Download + bundle ke folder
    python tools/bundling_weights.py --bundle offline_bundle

    # Download saja (cek cache)
    python tools/bundling_weights.py
"""

import sys
import os
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BACKBONES = [
    "resnet18",         # default project
    "wide_resnet50_2",  # opsi high-accuracy
]


def _get_hf_cache_dir() -> Path:
    """Return huggingface_hub cache directory."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except ImportError:
        return Path.home() / ".cache" / "huggingface" / "hub"


def download_weights():
    """Download backbone weights via timm (jalur yang dipakai Anomalib).
    Cache otomatis tersimpan di ~/.cache/huggingface/hub/ oleh huggingface_hub.
    """
    import timm

    print("=== Download Backbone Weights (timm → HuggingFace Hub) ===")
    cache_dir = _get_hf_cache_dir()
    print(f"Cache target: {cache_dir}")
    print()

    for backbone in BACKBONES:
        print(f"Downloading {backbone}...")
        try:
            model = timm.create_model(backbone, pretrained=True)
            print(f"  ✅ {backbone} — {sum(p.numel() for p in model.parameters()):,} params")
        except Exception as e:
            print(f"  ❌ {backbone} gagal: {e}")

    # Tampilkan cache size
    if cache_dir.exists():
        total_size = sum(
            f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
        )
        print(f"\nTotal HF cache: {total_size / 1024 / 1024:.1f} MB")
        print(f"Lokasi cache: {cache_dir}")
        print()
        print("Copy folder ini ke PC target:")
        print(f"  {cache_dir}  →  %USERPROFILE%\\.cache\\huggingface\\")
    else:
        print("\n⚠️  Cache folder tidak ditemukan — mungkin download gagal.")


def bundle_for_offline(target_dir: Path):
    """Download weights + copy HF cache ke target_dir untuk offline bundle."""
    download_weights()

    src = _get_hf_cache_dir()
    dst = target_dir / "hf_cache"

    if not src.exists():
        print("❌ HF cache tidak ditemukan. Download mungkin gagal.")
        return

    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        s_dst = dst / item.name
        if item.is_dir():
            shutil.copytree(item, s_dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, s_dst)

    total_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1048576
    print(f"\n✅ Weights dibundel ke {dst}")
    print(f"   Size: {total_mb:.1f} MB")
    print()
    print("Instalasi di edge PC:")
    print(f"  1. Copy {dst} ke %USERPROFILE%\\.cache\\huggingface\\")
    print(f"  2. Set environment: set HF_HUB_OFFLINE=1")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Download backbone weights untuk offline deployment")
    parser.add_argument("--bundle", type=str, default="",
                        help="Target directory untuk offline bundle")
    args = parser.parse_args()

    if args.bundle:
        bundle_for_offline(Path(args.bundle))
    else:
        download_weights()
