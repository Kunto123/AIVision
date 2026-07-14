"""
VisionInspect - Simple Threshold Training (Fallback for Windows)
Training tanpa PyTorch/Anomalib. Menggunakan statistik piksel sederhana.
Cocok untuk environment Windows di mana torch DLL bermasalah.
"""

import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("training")


class SimpleThresholdTrainer:
    """
    Fallback trainer: compute mean + std of OK images.
    At inference, pixel-wise z-score → anomaly score.
    Tidak perlu PyTorch, hanya numpy + OpenCV.
    """

    def __init__(self, input_size: int = 256):
        self._input_size = input_size
        self._progress_callback: Optional[Callable[[int, str], None]] = None

    def set_progress_callback(self, cb: Optional[Callable[[int, str], None]]) -> None:
        self._progress_callback = cb

    def train(self, ok_dir: Path, ng_dir: Optional[Path], output_dir: Path) -> dict:
        """
        Train simple statistical model from OK images.
        Returns model artifacts similar to TrainingPipeline.
        """
        self._report(10, "Memuat gambar OK...")

        ok_images = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            ok_images.extend(list(ok_dir.glob(ext)))

        if len(ok_images) < 1:
            raise ValueError("Minimal 1 gambar OK diperlukan")

        # Load and resize all OK images
        loaded = []
        for img_path in ok_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self._input_size, self._input_size))
            loaded.append(img.astype(np.float32) / 255.0)

        if not loaded:
            raise ValueError("Tidak ada gambar OK yang bisa dimuat")

        imgs = np.stack(loaded, axis=0)  # (N, H, W, 3)

        self._report(30, "Menghitung mean & std...")
        mean = np.mean(imgs, axis=0)
        std = np.std(imgs, axis=0)
        std = np.clip(std, 0.01, None)  # avoid division by zero

        self._report(50, "Mengevaluasi threshold...")

        # Compute scores on training data
        ok_scores = []
        for i in range(len(loaded)):
            z = np.abs(imgs[i] - mean) / std
            score = float(np.mean(z))
            ok_scores.append(score)

        # Also compute NG scores if available
        ng_scores = []
        if ng_dir and ng_dir.exists():
            ng_images = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                ng_images.extend(list(ng_dir.glob(ext)))
            for img_path in ng_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self._input_size, self._input_size))
                img_f = img.astype(np.float32) / 255.0
                z = np.abs(img_f - mean) / std
                score = float(np.mean(z))
                ng_scores.append(score)

        # Threshold: mean + 3*std of OK scores
        ok_mean = float(np.mean(ok_scores))
        ok_std = float(np.std(ok_scores))
        threshold = min(1.0, ok_mean + 3.0 * ok_std)

        self._report(70, f"Threshold: {threshold:.4f}")

        # Save model (mean + std arrays)
        model_dir = output_dir / "simple_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "mean.npy", mean)
        np.save(model_dir / "std.npy", std)

        # Also save as PNG preview for visual reference
        mean_img = (mean * 255).astype(np.uint8)
        cv2.imwrite(str(model_dir / "mean_preview.png"),
                     cv2.cvtColor(mean_img, cv2.COLOR_RGB2BGR))

        self._report(90, "Model tersimpan")

        metadata = {
            "algorithm": "simple_threshold",
            "input_size": self._input_size,
            "threshold": threshold,
            "ok_threshold_used": ok_mean + 3.0 * ok_std,
            "num_ok": len(ok_images),
            "num_ng": len(ng_images) if ng_dir else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "export_path": str(output_dir),
        }

        self._report(100, "Training selesai!")

        return {
            "threshold": threshold,
            "model_path": str(model_dir),
            "export_path": str(output_dir),
            "ok_scores": ok_scores,
            "ng_scores": ng_scores,
            "metadata": metadata,
        }

    def _report(self, percent: int, message: str):
        if self._progress_callback:
            self._progress_callback(percent, message)
