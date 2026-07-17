#!/usr/bin/env python3
"""
VisionInspect — CLI Training Wrapper
Jalankan training dari command line (termasuk dari WSL) tanpa GUI/Qt.
GUI tetap jalan di Windows, training jalan di WSL tempat PyTorch bisa load.

Usage (dari WSL):
  python tools/train_cli.py --program Default --template template_1

Usage via retrain_wsl.bat (double-click dari Windows):
  retrain_wsl.bat
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Pastikan project root di path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Set data directory
_DATA_DIR = _PROJECT_ROOT / "data"
if "VISIONINSPECT_DATA" not in os.environ:
    os.environ["VISIONINSPECT_DATA"] = str(_DATA_DIR)


def train(program: str, template_id: str):
    """Run training pipeline for a specific template, no Qt needed."""
    from visioninspect.core.program import ProgramManager
    from visioninspect.core.training import TrainingPipeline, TrainingConfig, TrainingError
    from visioninspect.core.simple_train import SimpleThresholdTrainer
    from visioninspect.gui.training_worker import _crop_images_to_rois

    pm = ProgramManager(_DATA_DIR / "programs")

    # Validate
    tmpl_cfg = pm.get_template_config(program, template_id)
    if not tmpl_cfg:
        print(f"❌ Template '{template_id}' tidak ditemukan di program '{program}'")
        return False

    print(f"📋 Program: {program}")
    print(f"📋 Template: {template_id} ({tmpl_cfg.get('name', template_id)})")

    # Image dirs
    tmpl_dir = pm._get_template_dir(program) / template_id
    ok_dir = tmpl_dir / "images" / "ok"
    ng_dir = tmpl_dir / "images" / "ng"

    if not ok_dir.exists():
        print("❌ Folder images/ok/ tidak ditemukan")
        return False

    ok_images = list(ok_dir.glob("*.png")) + list(ok_dir.glob("*.jpg"))
    if not ok_images:
        print("❌ Tidak ada gambar OK untuk training")
        return False
    print(f"📸 OK images: {len(ok_images)}")

    ng_images = []
    if ng_dir.exists():
        ng_images = list(ng_dir.glob("*.png")) + list(ng_dir.glob("*.jpg"))
    print(f"📸 NG images: {len(ng_images)}")

    # ROI crop (matching inference pipeline)
    from visioninspect.gui.training_worker import TrainingWorker
    rois = TrainingWorker._get_enabled_rois(tmpl_cfg)
    input_size = tmpl_cfg.get("input_size", 256)

    if rois:
        print(f"✂️  Crop ke {len(rois)} ROI(s)...")
        ok_crop = Path(tempfile.mkdtemp(prefix="vi_ok_"))
        ng_crop = Path(tempfile.mkdtemp(prefix="vi_ng_"))
        n_ok = _crop_images_to_rois(ok_dir, rois, ok_crop, input_size)
        n_ng = _crop_images_to_rois(ng_dir, rois, ng_crop, input_size)
        print(f"   {n_ok} OK crops, {n_ng} NG crops")
        ok_dir = ok_crop
        ng_path = ng_crop if ng_images else None
    else:
        print("ℹ️  Tanpa ROI — training full-frame")
        ng_path = ng_dir if ng_images else None

    # Pilih trainer
    torch_ok = True
    try:
        import torch  # noqa
    except Exception:
        torch_ok = False

    if torch_ok:
        print("🧠 PyTorch tersedia → Anomalib training")
        train_cfg = TrainingConfig(
            algorithm=tmpl_cfg.get("algorithm", "patchcore"),
            backbone=tmpl_cfg.get("backbone", "resnet18"),
            input_size=input_size,
            coreset_sampling_ratio=tmpl_cfg.get("coreset_sampling_ratio", 0.1),
            threshold_mode=tmpl_cfg.get("threshold_mode", "adaptive"),
            manual_threshold=tmpl_cfg.get("manual_threshold", 0.5),
            enable_int8=tmpl_cfg.get("enable_int8", True),
        )
        pipeline = TrainingPipeline(train_cfg)

        def _progress(pct, msg):
            print(f"  [{pct:3d}%] {msg}")

        pipeline.set_progress_callback(_progress)
        output_dir = Path(tempfile.mkdtemp(prefix="vi_train_"))
        result = pipeline.train(ok_dir=ok_dir, ng_dir=ng_path, output_dir=output_dir)
    else:
        print("⚠️  PyTorch tidak tersedia → SimpleThreshold fallback")
        output_dir = Path(tempfile.mkdtemp(prefix="vi_train_"))
        trainer = SimpleThresholdTrainer(input_size=input_size)
        result = trainer.train(ok_dir=ok_dir, ng_dir=ng_path, output_dir=output_dir)

    # Simpan model ke template
    print("💾 Menyimpan model...")
    version = pm.save_template_model(program, template_id, result)
    print(f"✅ Model v{version} tersimpan (threshold={result['threshold']:.4f})")
    return True


def main():
    parser = argparse.ArgumentParser(description="VisionInspect CLI Training")
    parser.add_argument("--program", default="Default", help="Nama program")
    parser.add_argument("--template", required=True, help="Template ID (contoh: template_1)")
    args = parser.parse_args()

    success = train(args.program, args.template)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
