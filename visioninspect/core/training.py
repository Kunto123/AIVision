"""
VisionInspect - Training Pipeline
Anomalib-based training: PatchCore / EfficientAd → export OpenVINO → INT8 PTQ.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("training")


class TrainingError(Exception):
    pass


class TrainingConfig:
    """Configuration for training pipeline."""

    def __init__(
        self,
        algorithm: str = "patchcore",       # patchcore | efficientad
        backbone: str = "resnet18",          # resnet18 | wide_resnet50_2
        input_size: int = 256,
        coreset_sampling_ratio: float = 0.1,
        threshold_mode: str = "adaptive",    # adaptive | manual
        manual_threshold: float = 0.5,
        threshold_margin_sigma: float = 3.0,
        enable_int8: bool = True,
        max_epochs: Optional[int] = None,
    ):
        self.algorithm = algorithm
        self.backbone = backbone
        self.input_size = input_size
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.threshold_mode = threshold_mode
        self.manual_threshold = manual_threshold
        self.threshold_margin_sigma = threshold_margin_sigma
        self.enable_int8 = enable_int8
        # PatchCore is one-shot (memory-bank, no backprop) — 1 epoch is correct.
        # EfficientAd trains an actual network via backprop and needs many more
        # epochs to converge; None picks a sensible per-algorithm default.
        if max_epochs is not None:
            self.max_epochs = max_epochs
        else:
            self.max_epochs = 1 if algorithm == "patchcore" else 100


class TrainingPipeline:
    """
    Pipeline training Anomalib.
    Langkah: load data → fit model → kalibrasi threshold → export OpenVINO → INT8.
    Berjalan di worker thread terpisah.
    """

    def __init__(self, config: TrainingConfig):
        self._config = config
        self._progress_callback: Optional[Callable[[int, str], None]] = None
        self._cancelled = False

    # ---- Callbacks ----

    def set_progress_callback(self, cb: Optional[Callable[[int, str], None]]) -> None:
        self._progress_callback = cb

    def cancel(self) -> None:
        self._cancelled = True

    # ---- Main Training ----

    def train(
        self,
        ok_dir: Path,
        ng_dir: Optional[Path],
        output_dir: Path,
    ) -> dict:
        """
        Run full training pipeline.
        
        Args:
            ok_dir: Directory with OK images
            ng_dir: Optional directory with NG images
            output_dir: Directory to save model artifacts
            
        Returns:
            dict with keys: threshold, model_path, export_path, int8_path, stats
        """
        self._cancelled = False
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect images
        ok_images = list(ok_dir.glob("*.png")) + list(ok_dir.glob("*.jpg")) + list(ok_dir.glob("*.jpeg"))
        ng_images = []
        if ng_dir and ng_dir.exists():
            ng_images = list(ng_dir.glob("*.png")) + list(ng_dir.glob("*.jpg")) + list(ng_dir.glob("*.jpeg"))

        if len(ok_images) < 1:
            raise TrainingError("Minimal 1 gambar OK diperlukan")

        self._report(5, f"Menyiapkan data: {len(ok_images)} OK, {len(ng_images)} NG")

        # Try Anomalib import
        try:
            import torch  # noqa: F401 — check torch is loadable first

            # ── Patch create_versioned_dir SEBELUM Engine di-import ──
            # Engine.fit() → _setup_workspace(versioned_dir=True) →
            # create_versioned_dir() → symlink_to() → WinError 1314 di Windows.
            # Karena engine.py lakukan `from anomalib.utils.path import create_versioned_dir`
            # di module level, kita harus patch di source module SEBELUM engine di-import.
            import anomalib.utils.path as _anom_path

            def _safe_versioned_dir(root_dir):
                """Buat folder 'latest' biasa, tanpa symlink (anti-WinError 1314)."""
                root_dir = Path(root_dir).resolve()
                root_dir.mkdir(parents=True, exist_ok=True)
                latest = root_dir / "latest"
                latest.mkdir(parents=True, exist_ok=True)
                return latest

            _anom_path.create_versioned_dir = _safe_versioned_dir

            # Sekarang import Engine — dia akan mengambil create_versioned_dir
            # yang sudah di-patch dari namespace anomalib.utils.path
            from anomalib.data import Folder
            from anomalib.models import Patchcore, EfficientAd
            from anomalib.engine import Engine
            from anomalib.deploy import ExportType
            from anomalib import TaskType
            from anomalib.data import Folder
            logger.info("Anomalib imported successfully")
        except ImportError as e:
            logger.error("Anomalib import failed: %s", e)
            raise TrainingError(f"Anomalib tidak terinstall: {e}")
        except OSError as e:
            logger.error("Torch/Anomalib DLL error: %s", e)
            raise TrainingError(
                "PyTorch tidak bisa dimuat di Windows. "
                "Gunakan WSL untuk training, atau install Visual C++ Redistributable. "
                f"Detail: {e}")

        if self._cancelled:
            raise TrainingError("Training dibatalkan")

        self._report(10, "Inisialisasi model...")

        # Create model
        try:
            if self._config.algorithm == "patchcore":
                model = Patchcore(
                    backbone=self._config.backbone,
                    coreset_sampling_ratio=self._config.coreset_sampling_ratio,
                )
            elif self._config.algorithm == "efficientad":
                # EfficientAd doesn't take backbone/input_size — it uses its
                # own fixed small teacher-student network (no swappable
                # torchvision backbone like PatchCore); image size is
                # controlled entirely by the datamodule below.
                model = EfficientAd()
            else:
                raise TrainingError(f"Unknown algorithm: {self._config.algorithm}")
        except Exception as e:
            raise TrainingError(f"Gagal membuat model: {e}")

        # Create datamodule
        try:
            import torch  # noqa: F401 - needed by Anomalib
            # EfficientAd's teacher-student normalization stats are computed
            # per-sample and require batch_size=1 (anomalib raises otherwise);
            # PatchCore has no such constraint.
            batch_size = 1 if self._config.algorithm == "efficientad" else 16
            datamodule = Folder(
                name="visioninspect",
                task=TaskType.CLASSIFICATION,
                root=ok_dir.parent,
                normal_dir=ok_dir.name,
                abnormal_dir=ng_dir.name if ng_dir else None,
                image_size=(self._config.input_size, self._config.input_size),
                train_batch_size=batch_size,
                eval_batch_size=batch_size,
                num_workers=0,
            )
            datamodule.setup()
        except Exception as e:
            raise TrainingError(f"Gagal setup data: {e}")

        if self._cancelled:
            raise TrainingError("Training dibatalkan")

        self._report(20, "Memulai training...")

        # Engine — default_root_dir di temp, matikan checkpointing versi
        # (Lightning bikin symlink v0→latest yg error di Windows tanpa privilege)
        import tempfile as _tf
        _train_work_dir = _tf.mkdtemp(prefix="visioninspect_")
        engine = Engine(
            task="classification",  # or "segmentation" depending on model
            image_metrics=["F1Score", "AUROC"],
            pixel_metrics=None,
            accelerator="cpu",
            devices=1,
            max_epochs=self._config.max_epochs,
            default_root_dir=_train_work_dir,
            #enable_checkpointing=False,  # cegah symlink v0→latest (WinError 1314)
            logger=False,              # TensorBoard logger jg bikin symlink yg sama
        )

        # Fit
        try:
            engine.fit(model=model, datamodule=datamodule)
        except Exception as e:
            raise TrainingError(f"Training gagal: {e}")

        if self._cancelled:
            raise TrainingError("Training dibatalkan")

        self._report(50, "Training selesai, mengevaluasi...")

        # Evaluate to get threshold
        try:
            test_results = engine.test(model=model, datamodule=datamodule)
            logger.info("Test results: %s", test_results)
        except Exception as e:
            logger.warning("Evaluasi gagal: %s", e)
            test_results = {}

        self._report(55, "Mengumpulkan skor untuk histogram...")
        ok_scores, ng_scores = self._collect_scores(model, ok_images, ng_images)
        self._report(60, f"OK: {len(ok_scores)} scores, NG: {len(ng_scores)} scores")

        if self._cancelled:
            raise TrainingError("Training dibatalkan")

        # Threshold calibration
        threshold = self._calibrate_threshold(model, datamodule)
        self._report(65, f"Threshold: {threshold:.4f}")

        if self._cancelled:
            raise TrainingError("Training dibatalkan")

        # Export to OpenVINO
        self._report(70, "Export ke OpenVINO...")
        export_dir = output_dir / "openvino"
        export_dir.mkdir(parents=True, exist_ok=True)

        ov_export_ok = False
        try:
            import torch
            import openvino as ov

            # Bypass Anomalib engine.export() — langsung torch → OpenVINO
            # (engine.export() gagal karena torch.export.export() tidak
            #  support model PatchCore di PyTorch 2.6+)
            model.eval()
            dummy = torch.randn(1, 3, self._config.input_size,
                                self._config.input_size)
            ov_model = ov.convert_model(model, example_input=dummy)
            # Pin input ke static [1,3,H,W] — convert_model menghasilkan
            # shape dinamis (?,3,?,?) yang bikin .shape throw di load time.
            ov_model.reshape([1, 3, self._config.input_size, self._config.input_size])
            ov_xml = export_dir / "model.xml"
            ov.save_model(ov_model, str(ov_xml))
            logger.info("OpenVINO export selesai (direct): %s", ov_xml)
            ov_export_ok = True
        except Exception as e:
            logger.warning("OpenVINO export gagal: %s", e)

        if not ov_export_ok:
            # Fallback: train SimpleThreshold model (mean/std) — pasti bisa dipake inference
            self._report(75, "OpenVINO gagal, fallback ke SimpleThreshold...")
            try:
                from visioninspect.core.simple_train import SimpleThresholdTrainer
                st_trainer = SimpleThresholdTrainer(
                    input_size=self._config.input_size)
                st_trainer.set_progress_callback(self._progress_callback)
                st_result = st_trainer.train(
                    ok_dir=ok_dir, ng_dir=ng_dir, output_dir=output_dir)
                logger.info("SimpleThreshold fallback selesai, threshold=%.4f",
                            st_result["threshold"])
                # Override export_path biar save_template_model nemu
                export_dir = Path(st_result["export_path"])
                threshold = st_result["threshold"]
            except Exception as e2:
                raise TrainingError(
                    f"OpenVINO export gagal: {e}. "
                    f"SimpleThreshold fallback juga gagal: {e2}")

        if self._cancelled:
            raise TrainingError("Training dibatalkan")

        self._report(85, "INT8 Quantization...")

        # INT8 PTQ via NNCF
        int8_path = None
        if self._config.enable_int8 and (export_dir / "model.xml").exists():
            try:
                int8_path = self._quantize_int8(export_dir / "model.xml", output_dir)
                self._report(90, "INT8 quantization selesai")
            except Exception as e:
                logger.warning("INT8 quantization failed: %s", e)

        # ── Kalibrasi normalisasi skor ──────────────────────────────────────
        # Skor PatchCore mentah tidak berada di [0,1] (mis. OK ~21). Hitung
        # score_ref (titik pisah OK vs NG) dari model OpenVINO, simpan ke
        # norm.json di samping model.xml, dan normalisasi skor histogram
        # (score_ref → 0.5) agar sebanding dgn threshold saat inferensi.
        score_ref = None
        if ov_export_ok and (export_dir / "model.xml").exists():
            self._report(92, "Kalibrasi skor...")
            try:
                ok_raw = self._score_images_openvino(export_dir / "model.xml", ok_images)
                ng_raw = (self._score_images_openvino(export_dir / "model.xml", ng_images)
                          if ng_images else [])
                score_ref = self._compute_score_ref(ok_raw, ng_raw)
                norm_payload = {"score_ref": score_ref,
                                "input_size": self._config.input_size}
                # norm.json ikut tercopy ke template bersama folder 'openvino'
                with open(export_dir / "norm.json", "w") as f:
                    json.dump(norm_payload, f, indent=2)
                if int8_path:
                    try:
                        with open(Path(int8_path).parent / "norm.json", "w") as f:
                            json.dump(norm_payload, f, indent=2)
                    except Exception:
                        pass
                # Normalisasi skor untuk histogram (0.5 = ambang)
                ok_scores = [self._normalize_score(s, score_ref) for s in ok_raw]
                ng_scores = [self._normalize_score(s, score_ref) for s in ng_raw]
                logger.info("Kalibrasi skor: score_ref=%.4f (OK n=%d, NG n=%d)",
                            score_ref, len(ok_raw), len(ng_raw))
            except Exception as e:
                logger.warning("Kalibrasi skor gagal: %s", e)

        # Save metadata
        metadata = {
            "algorithm": self._config.algorithm,
            "backbone": self._config.backbone,
            "input_size": self._config.input_size,
            "threshold": threshold,
            "threshold_mode": self._config.threshold_mode,
            "score_ref": score_ref,
            "num_ok": len(ok_images),
            "num_ng": len(ng_images),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "export_path": str(export_dir),
            "int8_path": str(int8_path) if int8_path else "",
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self._report(100, "Training selesai!")

        return {
            "threshold": threshold,
            "score_ref": score_ref,
            "model_path": str(export_dir),
            "export_path": str(output_dir),  # parent dir agar save_template_model temukan subfolder
            "int8_path": str(int8_path) if int8_path else "",
            "ok_scores": ok_scores,
            "ng_scores": ng_scores,
            "metadata": metadata,
        }

    # ---- Score Normalization / Calibration ----

    @staticmethod
    def _normalize_score(raw: float, score_ref: Optional[float]) -> float:
        """Map skor mentah → [0,1] dgn score_ref sbg titik tengah (0.5)."""
        if not score_ref or score_ref <= 0:
            return max(0.0, min(1.0, float(raw)))
        return max(0.0, min(1.0, 0.5 * float(raw) / score_ref))

    def _score_images_openvino(self, xml_path: Path, image_paths) -> list:
        """Jalankan model OpenVINO pada tiap gambar → ambil pred_score mentah."""
        import cv2
        import numpy as np
        import openvino as ov

        core = ov.Core()
        cm = core.compile_model(core.read_model(str(xml_path)), "CPU")
        ps_port = None
        for port in cm.outputs:
            try:
                if "pred_score" in port.get_names():
                    ps_port = port
                    break
            except Exception:
                pass
        size = self._config.input_size
        scores = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (size, size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            res = cm({0: t})
            if ps_port is not None:
                raw = float(np.asarray(res[ps_port]).reshape(-1)[0])
            else:
                # fallback: cari output 1D, atau max output pertama
                raw = None
                for val in res.values():
                    arr = np.asarray(val)
                    if arr.ndim <= 1:
                        raw = float(arr.reshape(-1)[0])
                        break
                if raw is None:
                    raw = float(np.max(np.asarray(list(res.values())[0])))
            scores.append(raw)
        return scores

    @staticmethod
    def _compute_score_ref(ok_raw, ng_raw) -> float:
        """Titik pisah OK vs NG (skor mentah); dipetakan ke normalized 0.5.

        Dgn NG: titik tengah antara ekor atas OK & ekor bawah NG.
        Tanpa NG: sedikit di atas sebaran OK (mean + 3σ).
        """
        import numpy as np
        ok = np.asarray([s for s in ok_raw if np.isfinite(s)], dtype=float)
        ng = np.asarray([s for s in ng_raw if np.isfinite(s)], dtype=float)
        if ok.size == 0:
            return float(ng.mean()) if ng.size else 1.0
        if ng.size:
            hi_ok = float(np.percentile(ok, 95))
            lo_ng = float(np.percentile(ng, 5))
            ref = ((hi_ok + lo_ng) / 2.0 if lo_ng > hi_ok
                   else float(ok.mean() + 3.0 * (ok.std() + 1e-6)))
        else:
            ref = float(ok.mean() + 3.0 * (ok.std() + 1e-6))
        return max(ref, 1e-6)

    # ---- Score Collection for Histogram ----

    def _collect_scores(self, model, ok_images, ng_images):
        """
        Run model on training images and collect anomaly scores.
        Returns (ok_scores, ng_scores) lists of floats.
        Silently returns empty lists on failure.
        """
        ok_scores = []
        ng_scores = []

        try:
            import cv2
            import torch
            model.eval()

            def _predict_scores(image_paths):
                scores = []
                device = next(model.parameters()).device
                for img_path in image_paths:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self._config.input_size, self._config.input_size))
                    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

                    with torch.no_grad():
                        output = model(img_tensor)

                    # Parse output (anomalib returns dict-like or tensor)
                    if isinstance(output, dict):
                        score = float(output.get("pred_score", output.get("anomaly_map", output.get("pred_scores", 0))))
                    elif hasattr(output, "item"):
                        score = float(output.item())
                    elif isinstance(output, (list, tuple)):
                        score = float(output[0]) if output else 0.0
                    else:
                        try:
                            score = float(output)
                        except (TypeError, ValueError):
                            score = 0.0

                    if isinstance(score, float) and score == float('inf'):
                        score = 1.0
                    scores.append(max(0.0, min(1.0, score)))
                return scores

            if ok_images:
                ok_scores = _predict_scores(ok_images)
            if ng_images:
                ng_scores = _predict_scores(ng_images)

            logger.info("Collected %d OK scores, %d NG scores", len(ok_scores), len(ng_scores))

        except Exception as e:
            logger.warning("Score collection failed: %s", e)

        return ok_scores, ng_scores

    # ---- Threshold Calibration ----

    def _calibrate_threshold(self, model, datamodule) -> float:
        """Calibrate threshold based on mode."""
        if self._config.threshold_mode == "manual":
            return self._config.manual_threshold

        # Adaptive calibration belum diimplementasikan penuh — pakai default
        # netral 0.5. Threshold final umumnya di-tuning manual lewat slider,
        # yang kini tersimpan permanen ke config template (lihat
        # MainWindow._on_threshold_released).
        return 0.5

    # ---- INT8 Quantization ----

    def _quantize_int8(self, xml_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Run INT8 PTQ quantization via NNCF.
        Requires representative dataset.
        """
        try:
            import nncf  # type: ignore[import-not-found]  # dependency opsional (INT8)
            import openvino as ov

            core = ov.Core()
            model = core.read_model(str(xml_path))

            # Simple quantization without calibration data
            # In production, use representative dataset
            quantized_model = nncf.quantize(
                model,
                nncf.Dataset([]),  # empty dataset = minimal calibration
                subset_size=10,
                preset=nncf.QuantizationPreset.PERFORMANCE,
            )

            int8_dir = output_dir / "openvino_int8"
            int8_dir.mkdir(parents=True, exist_ok=True)
            int8_xml = int8_dir / "model.xml"

            ov.serialize(quantized_model, str(int8_xml))
            logger.info("INT8 model saved: %s", int8_xml)
            return int8_xml

        except ImportError:
            logger.warning("NNCF not installed, skipping INT8 quantization")
            return None
        except Exception as e:
            logger.warning("INT8 quantization failed: %s", e)
            return None

    # ---- Internal ----

    def _report(self, percent: int, message: str):
        if self._progress_callback:
            self._progress_callback(percent, message)
