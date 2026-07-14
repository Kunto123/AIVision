"""
VisionInspect - Training Worker (QThread)
Menjalankan TrainingPipeline di QThread terpisah agar tidak memblokir GUI.
"""

import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from visioninspect.core.training import TrainingPipeline, TrainingConfig, TrainingError
from visioninspect.core.program import ProgramManager
from visioninspect.utils.logging_setup import get_logger

logger = get_logger("training")


class TrainingWorker(QObject):
    """
    Worker untuk training yang berjalan di QThread terpisah.
    """

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
        """Internal: jalankan training."""
        # Get template config
        tmpl_cfg = self._pm.get_template_config(program, template_id)
        if not tmpl_cfg:
            raise TrainingError(f"Template '{template_id}' tidak ditemukan")

        # Get image directories
        tmpl_dir = self._pm._get_template_dir(program) / template_id
        ok_dir = tmpl_dir / "images" / "ok"
        ng_dir = tmpl_dir / "images" / "ng"

        if not ok_dir.exists() or len(list(ok_dir.glob("*.png")) + list(ok_dir.glob("*.jpg"))) == 0:
            raise TrainingError("Tidak ada gambar OK di template ini")

        # Check if torch is available (it fails on Windows with DLL error)
        torch_ok = True
        try:
            import torch  # noqa: F401
        except Exception:
            torch_ok = False
            logger.warning("Torch tidak tersedia, gunakan SimpleThresholdTrainer")

        if torch_ok:
            self._do_anomalib_training(program, template_id, tmpl_cfg, ok_dir, ng_dir)
        else:
            self._do_simple_training(program, template_id, tmpl_cfg, ok_dir, ng_dir)

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
            ng_dir=ng_dir if (ng_dir.exists() and list(ng_dir.glob("*"))) else None,
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
            ng_dir=ng_dir if (ng_dir.exists() and list(ng_dir.glob("*"))) else None,
            output_dir=output_dir,
        )

        # Save model to template
        self.progress.emit(95, "Menyimpan model ke template...")
        version = self._pm.save_template_model(program, template_id, result)

        result["version"] = version
        result["template_id"] = template_id

        logger.info("Training selesai: template=%s, version=%d", template_id, version)
        self.finished.emit(result)

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
