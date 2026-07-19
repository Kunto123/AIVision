#!/usr/bin/env python3
r"""
VisionInspect — Weight Bundling Script (Offline Deployment)
Download pretrained backbone weights untuk offline install di edge PC.

Dua jalur:
  1. Via timm (normal) — butuh torch, kena Windows AppControl policy?
  2. Via huggingface_hub langsung — fallback, tidak butuh torch.

Cache location: ~/.cache/huggingface/hub/ (default huggingface_hub)

Usage:
    python tools/bundling_weights.py
    python tools/bundling_weights.py --bundle offline_bundle
    python tools/bundling_weights.py --install-deps   # install deps dulu

CATATAN: Jalankan dengan PYTHON DARI VENV, BUKAN py launcher!
  ✅ BENAR: .vision\Scripts\python.exe tools\bundling_weights.py
  ❌ SALAH: py tools\bundling_weights.py
"""

import sys
import os
import subprocess
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── HF repo mapping: backbone name → timm HF hub repo id ──────────────
BACKBONE_REPOS = {
    "resnet18":        "timm/resnet18.a1_in1k",
    "wide_resnet50_2": "timm/wide_resnet50_2.racm_in1k",
}
BACKBONES = list(BACKBONE_REPOS.keys())


# ── Project root detection ────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_venv_python():
    """Cari Python dari virtual environment proyek (.vision atau .venv)."""
    for venv_name in (".vision", ".venv"):
        venv_dir = _PROJECT_ROOT / venv_name
        if not venv_dir.exists():
            continue
        # Coba path Windows
        candidate = venv_dir / "Scripts" / "python.exe"
        if candidate.exists():
            return candidate
        # Coba path Unix
        candidate = venv_dir / "bin" / "python"
        if candidate.exists():
            return candidate
    return None


def _is_running_in_project_venv():
    """Cek apakah kita sudah jalan dari dalam virtual environment proyek."""
    venv = os.environ.get("VIRTUAL_ENV")
    if venv and _PROJECT_ROOT in Path(venv).parents:
        return True
    # Cek dari path executable
    exe = Path(sys.executable).resolve()
    for part in exe.parts:
        if part in (".vision", ".venv"):
            return True
    return False


def _install_deps():
    """Install requirements.txt ke venv yang aktif atau buat yang baru."""
    req_file = _PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        print(f"❌ requirements.txt tidak ditemukan di {_PROJECT_ROOT}")
        return False

    # Cek apakah dalam venv
    if _is_running_in_project_venv():
        python_exe = Path(sys.executable).resolve()
    else:
        venv_python = _find_venv_python()
        if venv_python:
            python_exe = venv_python
        else:
            print("⏳ Membuat virtual environment .vision/ ...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", str(_PROJECT_ROOT / ".vision")],
                    check=True, capture_output=True, timeout=60,
                )
            except Exception as e:
                print(f"❌ Gagal membuat venv: {e}")
                return False
            python_exe = _PROJECT_ROOT / ".vision" / "Scripts" / "python.exe"
            if not python_exe.exists():
                python_exe = _PROJECT_ROOT / ".vision" / "bin" / "python"
            if not python_exe.exists():
                print("❌ Tidak bisa menemukan python executable di venv")
                return False

    print(f"📦 Menginstall dependencies ke {python_exe.parent} ...")
    try:
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "-r", str(req_file)],
            check=True, timeout=300,
        )
        print("✅ Dependencies terinstall.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Install gagal (exit code {e.returncode})")
        print("   Coba manual: pip install -r requirements.txt")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Install timeout (5 menit). Coba manual: pip install -r requirements.txt")
        return False


# ── Core download functions ───────────────────────────────────────────

def _get_hf_cache_dir():
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except ImportError:
        return Path.home() / ".cache" / "huggingface" / "hub"


def download_weights():
    """Download backbone weights. Fallback jika torch tidak bisa dimuat."""
    cache_dir = _get_hf_cache_dir()
    print("=== Download Backbone Weights (timm -> HuggingFace Hub) ===")
    print(f"Cache target: {cache_dir}")
    print()

    # ── Coba jalur 1: timm (normal, butuh torch) ──
    timm_ok = False
    try:
        import timm
        import torch
        # torch terload -> jalur timm
        timm_ok = True
        print("[Jalur timm + torch]")
        for backbone in BACKBONES:
            print(f"  Downloading {backbone}...")
            try:
                model = timm.create_model(backbone, pretrained=True)
                print(f"  ✅ {backbone} - {sum(p.numel() for p in model.parameters()):,} params")
            except Exception as e:
                print(f"  ❌ {backbone} via timm gagal: {e}")
                # fallback ke HF hub untuk backbone ini
                _download_via_hf(backbone)
    except Exception as e:
        print(f"[Jalur timm + torch GAGAL: {e}]")
        print("[Fallback: huggingface_hub langsung - tidak butuh torch]")
        timm_ok = False

    # ── Jalur 2: huggingface_hub langsung (fallback penuh) ──
    if not timm_ok:
        print("[Jalur huggingface_hub]")
        for backbone in BACKBONES:
            _download_via_hf(backbone)

    # Tampilkan cache size
    _show_cache_size(cache_dir)


def _download_via_hf(backbone: str):
    """Download model files via huggingface_hub snapshot_download."""
    repo_id = BACKBONE_REPOS.get(backbone)
    if not repo_id:
        print(f"  ❌ {backbone}: tidak ada HF repo mapping")
        return
    try:
        from huggingface_hub import snapshot_download
        print(f"  Downloading {backbone} ({repo_id})...")
        path = snapshot_download(repo_id)
        print(f"  ✅ {backbone} -> {path}")
    except ImportError:
        print(f"  ❌ {backbone} via HF gagal: huggingface_hub tidak terinstall")
        print(f"     Install dulu: pip install huggingface_hub")
    except Exception as e:
        print(f"  ❌ {backbone} via HF gagal: {e}")


def _show_cache_size(cache_dir):
    """Show HF cache stats."""
    if cache_dir.exists():
        total_size = sum(
            f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
        )
        print(f"\nTotal HF cache: {total_size / 1024 / 1024:.1f} MB")
        print(f"Lokasi cache: {cache_dir}")
        print()
        print("Copy folder ini ke PC target:")
        print(f"  {cache_dir}  ->  %USERPROFILE%\\.cache\\huggingface\\")
    else:
        print("\n⚠️  Cache folder tidak ditemukan - mungkin download gagal.")


# ── Offline bundle ────────────────────────────────────────────────────

def bundle_for_offline(target_dir):
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


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download backbone weights untuk offline deployment")
    parser.add_argument("--bundle", type=str, default="",
                        help="Target directory untuk offline bundle")
    parser.add_argument("--install-deps", action="store_true",
                        help="Install dependencies (pip install -r requirements.txt) dulu")
    args = parser.parse_args()

    # ── Install deps dulu jika diminta ──
    if args.install_deps:
        ok = _install_deps()
        if not ok:
            sys.exit(1)

    # ── Cek venv (hanya peringatan, tidak blocking) ──
    if not _is_running_in_project_venv():
        venv_python = _find_venv_python()
        if venv_python:
            print("⚠️  PERINGATAN: Sebaiknya jalankan dari virtual environment.")
            print(f"   Gunakan: {venv_python} tools/bundling_weights.py [options]")
            print()
        else:
            print("⚠️  PERINGATAN: Virtual environment (.vision/.venv) tidak ditemukan.")
            print("   Jalankan 'setup.bat' dulu untuk membuat environment.")
            print()

    # ── Eksekusi ──
    if args.bundle:
        bundle_for_offline(Path(args.bundle))
    else:
        download_weights()
