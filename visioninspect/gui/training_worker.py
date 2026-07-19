"""
VisionInspect - Training Worker (QThread)
Menjalankan TrainingPipeline di QThread terpisah agar tidak memblokir GUI.
"""

import time
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from visioninspect.core.training import TrainingPipeline, TrainingConfig, TrainingError
from visioninspect.core.program import ProgramManager
from visioninspect.utils.logging_setup import get_logger

logger = get_logger("training")


def _crop_images_to_rois(
    src_dir: Path,
    rois: List[dict],
    dst_dir: Path,
    input_size: int = 256,
) -> int:
    """Crop all images in *src_dir* to each enabled ROI, resize, and save to *dst_dir*.

    Returns number of cropped images saved.
    Handles multiple ROIs: 1 image × N ROIs = N training images.
    Used so training data matches inference pipeline (ROI-crop → resize).
    """
    import cv2
    import uuid

    if not rois:
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for fpath in sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg")):
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        for roi in rois:
            x = max(0, min(int(roi["x"]), w_img - 1))
            y = max(0, min(int(roi["y"]), h_img - 1))
            w = max(1, min(int(roi.get("width", 256)), w_img - x))
            h = max(1, min(int(roi.get("height", 256)), h_img - y))
            crop = img[y:y + h, x:x + w].copy()
            # Resize to input_size x input_size (matching inference)
            crop_resized = cv2.resize(crop, (input_size, input_size))
            uid = uuid.uuid4().hex[:8]
            dest = dst_dir / f"{fpath.stem}_roi{roi.get('uid', 'x')[:4]}_{uid}.png"
            cv2.imwrite(str(dest), crop_resized)
            count += 1
    return count


class TrainingWorker(QObject):
    """Worker untuk training yang berjalan di QThread terpisah."""

    # Signals
    progress = Signal(int, str)   # percent, message
    finished = Signal(dict)       # hasil training {threshold, model_path, ...}
    error = Signal(str)           # pesan error
    done = Signal()               # selesai (apapun hasilnya)

    def __init__(self, program_manager: ProgramManager, parent=None):
        super().__init__(parent)
        self._pm = program_manager
        self._pipeline: Optional[TrainingPipeline] = None

    @Slot(str, str)
    def start_training(self, program: str, template_id: str):
        """
        Start training for a template.
        Dipanggil dari QThread via signal.
        """
        try:
            self._do_training(program, template_id)
        except TrainingError as e:
            self.error.emit(str(e))
            logger.error("Training failed: %s", e)
        except Exception as e:
            self.error.emit(f"Unexpected error: {e}")
            logger.exception("Training unexpected error")
        finally:
            self.done.emit()

    def _do_training(self, program: str, template_id: str):
        """Internal: jalankan training dengan ROI cropping otomatis.

        Full-frame images dari galeri di-crop ke setiap enabled ROI
        sebelum training, sehingga data training identik dengan yang
        dilihat inference pipeline (ROI-crop → resize). Support multi-ROI.
        """
        # Get template config
        tmpl_cfg = self._pm.get_template_config(program, template_id)
        if not tmpl_cfg:
            raise TrainingError(f"Template '{template_id}' tidak ditemukan")

        # Get image directories (full-frame asli dari galeri)
        tmpl_dir = self._pm._get_template_dir(program) / template_id
        ok_dir = tmpl_dir / "images" / "ok"
        ng_dir = tmpl_dir / "images" / "ng"

        if not ok_dir.exists() or len(list(ok_dir.glob("*.png")) + list(ok_dir.glob("*.jpg"))) == 0:
            raise TrainingError("Tidak ada gambar OK di template ini")

        # Extract enabled ROIs and crop images to match inference pipeline
        rois = self._get_enabled_rois(tmpl_cfg)
        input_size = tmpl_cfg.get("input_size", 256)

        if rois:
            self.progress.emit(3, f"Menyiapkan data: crop ke {len(rois)} ROI...")
            import tempfile
            ok_crop_dir = Path(tempfile.mkdtemp(prefix="visioninspect_ok_crop_"))
            ng_crop_dir = Path(tempfile.mkdtemp(prefix="visioninspect_ng_crop_"))
            n_ok = _crop_images_to_rois(ok_dir, rois, ok_crop_dir, input_size)
            n_ng = _crop_images_to_rois(ng_dir, rois, ng_crop_dir, input_size)
            logger.info(
                "ROI crop: %d OK originals → %d crops across %d ROI(s); "
                "%d NG originals → %d crops",
                len(list(ok_dir.glob("*"))), n_ok, len(rois),
                len(list(ng_dir.glob("*"))), n_ng,
            )
            ok_dir = ok_crop_dir
            ng_dir = ng_crop_dir
        else:
            logger.info("Tidak ada ROI — training dengan full-frame images")

        ng_path = ng_dir if (ng_dir.exists() and list(ng_dir.glob("*"))) else None

        # Check if torch is available (it fails on Windows with DLL error)
        torch_ok = True
        try:
            import torch  # noqa: F401
        except Exception:
            torch_ok = False
            logger.warning("Torch tidak tersedia, gunakan SimpleThresholdTrainer")

        if torch_ok:
            self._do_anomalib_training(program, template_id, tmpl_cfg, ok_dir, ng_path)
        else:
            self._do_simple_training(program, template_id, tmpl_cfg, ok_dir, ng_path)

    @staticmethod
    def _get_enabled_rois(tmpl_cfg: dict) -> List[dict]:
        """Extract enabled ROIs from template config, handling legacy format."""
        roi_dicts = tmpl_cfg.get("rois", [])
        if not roi_dicts and "roi" in tmpl_cfg:
            old = tmpl_cfg["roi"]
            roi_dicts = [{
                "uid": "default",
                "x": old.get("x", 0), "y": old.get("y", 0),
                "width": old.get("width", 256), "height": old.get("height", 256),
                "enabled": True, "label": "ROI 1",
            }]
        return [r for r in roi_dicts if r.get("enabled", True)]

    def _do_simple_training(self, program, template_id, tmpl_cfg, ok_dir, ng_dir):
        """Fallback: training tanpa PyTorch."""
        from visioninspect.core.simple_train import SimpleThresholdTrainer

        self.progress.emit(5, "Menyiapkan data (mode sederhana)...")

        import tempfile
        output_dir = Path(tempfile.mkdtemp(prefix="visioninspect_train_"))

        trainer = SimpleThresholdTrainer(
            input_size=tmpl_cfg.get("input_size", 256))
        trainer.set_progress_callback(self._on_progress)

        result = trainer.train(
            ok_dir=ok_dir,
            ng_dir=ng_dir,
            output_dir=output_dir,
        )

        logger.info("Simple training selesai, threshold=%.4f", result["threshold"])

        self.progress.emit(95, "Menyimpan model ke template...")
        version = self._pm.save_template_model(program, template_id, result)
        result["version"] = version
        result["template_id"] = template_id
        self.finished.emit(result)

    def _do_anomalib_training(self, program, template_id, tmpl_cfg, ok_dir, ng_dir):
        """Anomalib training (torch-based)."""
        from visioninspect.core.training import TrainingPipeline, TrainingConfig

        train_cfg = TrainingConfig(
            algorithm=tmpl_cfg.get("algorithm", "patchcore"),
            backbone=tmpl_cfg.get("backbone", "resnet18"),
            input_size=tmpl_cfg.get("input_size", 256),
            coreset_sampling_ratio=tmpl_cfg.get("coreset_sampling_ratio", 0.1),
            threshold_mode=tmpl_cfg.get("threshold_mode", "adaptive"),
            manual_threshold=tmpl_cfg.get("manual_threshold", 0.5),
            enable_int8=tmpl_cfg.get("enable_int8", True),
        )

        # Create pipeline
        self._pipeline = TrainingPipeline(train_cfg)
        self._pipeline.set_progress_callback(self._on_progress)

        # Output directory (temp)
        import tempfile
        output_dir = Path(tempfile.mkdtemp(prefix="visioninspect_train_"))

        # Run training
        self.progress.emit(5, "Menyiapkan data...")
        result = self._pipeline.train(
            ok_dir=ok_dir,
            ng_dir=ng_dir,
            output_dir=output_dir,
        )

        # Kalibrasi normalisasi skor PER-ROI (skor PatchCore mentah beda skala
        # tiap ROI; 1 ref global tak cukup untuk multi-ROI). Tulis norm.json ke
        # output_dir SEBELUM disalin ke template oleh save_template_model.
        try:
            norm_ok, norm_ng = self._calibrate_per_roi(
                program, template_id, tmpl_cfg, output_dir)
            if norm_ok or norm_ng:
                result["ok_scores"] = norm_ok
                result["ng_scores"] = norm_ng
        except Exception as e:
            logger.warning("Kalibrasi per-ROI gagal: %s", e)

        # Save model to template
        self.progress.emit(95, "Menyimpan model ke template...")
        version = self._pm.save_template_model(program, template_id, result)

        result["version"] = version
        result["template_id"] = template_id

        logger.info("Training selesai: template=%s, version=%d", template_id, version)
        self.finished.emit(result)

    def _calibrate_per_roi(self, program, template_id, tmpl_cfg, output_dir):
        """Hitung score_ref per ROI & tulis norm.json.

        Skor PatchCore mentah punya skala berbeda tiap ROI, jadi 1 ref global
        salah untuk multi-ROI (bisa lewatkan semua NG). Untuk tiap ROI: crop
        gambar OK/NG asli ke ROI itu → skor via model OpenVINO → hitung ref.
        Returns (norm_ok_scores, norm_ng_scores) untuk histogram (dinormalisasi
        per ROI, 0.5 = ambang), atau ([], []) bila tak bisa dikalibrasi.
        """
        import tempfile
        import json
        import shutil

        rois = self._get_enabled_rois(tmpl_cfg)
        model_xml = output_dir / "openvino" / "model.xml"
        if not rois or not model_xml.exists() or self._pipeline is None:
            return [], []

        tmpl_dir = self._pm._get_template_dir(program) / template_id
        ok_src = tmpl_dir / "images" / "ok"
        ng_src = tmpl_dir / "images" / "ng"
        input_size = tmpl_cfg.get("input_size", 256)

        per_roi = {}
        norm_ok, norm_ng = [], []
        global_ok, global_ng = [], []
        for roi in rois:
            okd = Path(tempfile.mkdtemp(prefix="vi_cal_ok_"))
            ngd = Path(tempfile.mkdtemp(prefix="vi_cal_ng_"))
            try:
                _crop_images_to_rois(ok_src, [roi], okd, input_size)
                if ng_src.exists():
                    _crop_images_to_rois(ng_src, [roi], ngd, input_size)
                ok_imgs = sorted(okd.glob("*.png"))
                ng_imgs = sorted(ngd.glob("*.png"))
                ok_raw = self._pipeline._score_images_openvino(model_xml, ok_imgs)
                ng_raw = (self._pipeline._score_images_openvino(model_xml, ng_imgs)
                          if ng_imgs else [])
                if not ok_raw:
                    continue
                ref = self._pipeline._compute_score_ref(ok_raw, ng_raw)
                per_roi[roi.get("uid", "default")] = ref
                norm_ok += [self._pipeline._normalize_score(s, ref) for s in ok_raw]
                norm_ng += [self._pipeline._normalize_score(s, ref) for s in ng_raw]
                global_ok += ok_raw
                global_ng += ng_raw
            finally:
                shutil.rmtree(okd, ignore_errors=True)
                shutil.rmtree(ngd, ignore_errors=True)

        if not per_roi:
            return [], []

        global_ref = self._pipeline._compute_score_ref(global_ok, global_ng)
        payload = {"score_ref": global_ref, "input_size": input_size,
                   "per_roi": per_roi}
        for sub in ("openvino", "openvino_int8"):
            d = output_dir / sub
            if d.exists():
                try:
                    with open(d / "norm.json", "w") as f:
                        json.dump(payload, f, indent=2)
                except Exception as e:
                    logger.warning("Gagal tulis norm.json (%s): %s", sub, e)
        logger.info("Kalibrasi per-ROI: %s (global_ref=%.3f)",
                    {k: round(v, 3) for k, v in per_roi.items()}, global_ref)
        return norm_ok, norm_ng

    def _on_progress(self, percent: int, message: str):
        """Forward progress callback from pipeline."""
        self.progress.emit(percent, message)

    @Slot()
    def cancel(self):
        """Cancel running training."""
        if self._pipeline:
            self._pipeline.cancel()
            logger.info("Training cancelled by user")


class TrainingThread(QThread):
    """QThread khusus untuk TrainingWorker."""

    def __init__(self, program_manager: ProgramManager, parent=None):
        super().__init__(parent)
        self.worker = TrainingWorker(program_manager)
        self.worker.moveToThread(self)

    def run(self):
        self.exec()
