"""
VisionInspect - Inference Engine (OpenVINO)
Menangani inferensi model OpenVINO, hot-swap model, double-buffer.
ROI crop → resize → infer → heatmap overlay → score/judgement.
"""

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Full as QueueFull
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import cv2

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("inference")

try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    logger.warning("OpenVINO not installed. Inference will be unavailable.")


@dataclass
class InferenceResult:
    """Hasil inferensi untuk satu frame."""
    score: float           # anomaly score [0, 1]
    judgement: str         # "OK" or "NG"
    heatmap: Optional[npt.NDArray] = None  # anomaly heatmap (H, W)
    latency_ms: float = 0.0
    threshold: float = 0.5
    roi_cropped: Optional[npt.NDArray] = None  # ROI yang diproses


class InferenceEngineError(Exception):
    pass


class InferenceEngine:
    """
    OpenVINO inference engine dengan model hot-swap (double-buffer).
    Thread-safe untuk concurrent access.
    """

    def __init__(self, input_size: int = 256):
        self._input_size = input_size
        self._lock = threading.Lock()
        self._model: Optional[ov.CompiledModel] = None
        self._model_path: Optional[Path] = None
        self._threshold: float = 0.5
        self._use_ov = HAS_OPENVINO
        self._core: Optional[ov.Core] = None

        # Simple model data (fallback without PyTorch)
        self._simple_mean: Optional[npt.NDArray] = None
        self._simple_std: Optional[npt.NDArray] = None
        self._simple_loaded = False

        # Latency tracking
        self._latencies: list[float] = []
        self._max_latency_samples = 100

        if self._use_ov:
            try:
                self._core = ov.Core()
                logger.info("OpenVINO core initialized. Available devices: %s",
                            self._core.available_devices)
            except Exception as e:
                logger.warning("OpenVINO core init failed: %s", e)
                self._use_ov = False

    # ---- Properties ----

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None or self._simple_loaded

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, value))

    @property
    def latency_avg_ms(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    @property
    def latency_p95_ms(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            sorted_l = sorted(self._latencies)
            idx = int(len(sorted_l) * 0.95)
            return sorted_l[min(idx, len(sorted_l) - 1)]

    # ---- Model Loading / Hot-Swap ----

    def load_simple_model(self, model_dir: Path, threshold: float = 0.5) -> None:
        """Load simple statistical model (mean.npy + std.npy)."""
        mean_path = model_dir / "mean.npy"
        std_path = model_dir / "std.npy"
        if not mean_path.exists() or not std_path.exists():
            raise InferenceEngineError(f"Simple model not found in {model_dir}")

        mean = np.load(str(mean_path))
        std = np.load(str(std_path))
        self._input_size = mean.shape[0]

        with self._lock:
            self._simple_mean = mean
            self._simple_std = std
            self._simple_loaded = True
            self._threshold = threshold
            self._model = None  # clear any OpenVINO model

        logger.info("Simple model loaded: %s (threshold=%.3f, size=%d)",
                     model_dir, threshold, self._input_size)

    def load_model(self, model_path: Path, threshold: Optional[float] = None) -> None:
        """
        Load OpenVINO model from path. Thread-safe.
        Hot-swap: model baru dimuat dulu, lalu diganti secara atomik.
        """
        if not self._use_ov:
            raise InferenceEngineError("OpenVINO not available")

        if not model_path.exists():
            raise InferenceEngineError(f"Model not found: {model_path}")

        logger.info("Loading model: %s", model_path)

        # Cari file .xml (OpenVINO IR format)
        xml_path = model_path
        if xml_path.suffix.lower() not in (".xml",):
            # Maybe it's a directory or .bin
            if xml_path.is_dir():
                xml_files = list(xml_path.glob("*.xml"))
                if not xml_files:
                    raise InferenceEngineError(f"No OpenVINO IR (.xml) in {model_path}")
                xml_path = xml_files[0]
            else:
                # Try .xml with same stem
                xml_path = model_path.with_suffix(".xml")
                if not xml_path.exists():
                    raise InferenceEngineError(f"OpenVINO IR not found: {xml_path}")

        try:
            # Compile new model first (don't swap yet)
            model = self._core.read_model(str(xml_path))
            compiled = self._core.compile_model(model, "CPU")

            # Get input shape
            input_key = compiled.input(0)
            self._input_size = input_key.shape[-1]  # assume NCHW or NHWC square

            # Atomic swap
            with self._lock:
                old_model = self._model
                self._model = compiled
                self._model_path = model_path
                if threshold is not None:
                    self._threshold = threshold
                # Clean old model
                del old_model

            logger.info("Model loaded successfully: %s (input size: %d)", xml_path, self._input_size)
        except Exception as e:
            raise InferenceEngineError(f"Model load failed: {e}") from e

    def unload_model(self) -> None:
        """Unload current model (both OpenVINO and simple)."""
        with self._lock:
            self._model = None
            self._model_path = None
            self._simple_mean = None
            self._simple_std = None
            self._simple_loaded = False
        logger.info("Model unloaded")

    # ---- Inference ----

    def infer(self, frame: npt.NDArray, roi: Optional[dict] = None) -> InferenceResult:
        """
        Run inference on frame (or ROI-cropped region).
        Returns InferenceResult with score, judgement, heatmap.
        """
        start = time.perf_counter()

        # Crop ROI if specified — with bounds checking to prevent empty crops
        if roi:
            x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
            h_img, w_img = frame.shape[:2]
            # Clamp to valid image bounds, ensure min 1px size
            x = max(0, min(int(x), w_img - 1))
            y = max(0, min(int(y), h_img - 1))
            w = max(1, min(int(w), w_img - x))
            h = max(1, min(int(h), h_img - y))
            cropped = frame[y:y+h, x:x+w]
        else:
            cropped = frame

        # Resize to input size
        if cropped.shape[:2] != (self._input_size, self._input_size):
            resized = cv2.resize(cropped, (self._input_size, self._input_size))
        else:
            resized = cropped

        with self._lock:
            model = self._model
            threshold = self._threshold
            simple_mean = self._simple_mean
            simple_std = self._simple_std
            simple_loaded = self._simple_loaded

        if model is None and not simple_loaded:
            elapsed = (time.perf_counter() - start) * 1000
            return InferenceResult(
                score=0.0, judgement="OK", latency_ms=elapsed,
                threshold=threshold, roi_cropped=resized
            )

        # Simple model inference (z-score)
        if simple_loaded and simple_mean is not None and simple_std is not None:
            try:
                # Crop ROI if specified, with bounds checking
                if roi:
                    x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
                    h_img, w_img = frame.shape[:2]
                    x = max(0, min(int(x), w_img - 1))
                    y = max(0, min(int(y), h_img - 1))
                    w = max(1, min(int(w), w_img - x))
                    h = max(1, min(int(h), h_img - y))
                    cropped = frame[y:y+h, x:x+w]
                else:
                    cropped = frame
                resized = cv2.resize(cropped, (self._input_size, self._input_size))
                img_f = resized.astype(np.float32) / 255.0
                # Per-pixel z-score, then mean z-score as anomaly score
                z = np.abs(img_f - simple_mean) / simple_std
                score = float(np.mean(z))
                score = max(0.0, min(1.0, score))
                judgement = "OK" if score < threshold else "NG"
                elapsed = (time.perf_counter() - start) * 1000
                with self._lock:
                    self._latencies.append(elapsed)
                    if len(self._latencies) > self._max_latency_samples:
                        self._latencies.pop(0)
                return InferenceResult(
                    score=score, judgement=judgement, heatmap=None,
                    latency_ms=elapsed, threshold=threshold, roi_cropped=resized,
                )
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error("Simple inference error: %s", e)
                return InferenceResult(
                    score=1.0, judgement="NG", latency_ms=elapsed,
                    threshold=threshold, roi_cropped=resized,
                )

        try:
            # Preprocess: HWC to NCHW, normalize to [0,1]
            input_tensor = resized.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC → CHW
            input_tensor = np.expand_dims(input_tensor, axis=0)    # CHW → NCHW

            # Infer
            infer_request = model.create_infer_request()
            infer_request.set_input_tensor(ov.Tensor(input_tensor))
            infer_request.start_async()
            infer_request.wait()

            # Get outputs
            outputs = infer_request.get_output_tensor()
            output_data = outputs.data

            # Parse anomaly score & heatmap
            # Anomalib OpenVINO output: typically (1, 1, H, W) heatmap + scalar score
            if output_data.ndim == 4:
                # Heatmap output
                heatmap = output_data[0, 0]  # (H, W)
                score = float(np.max(heatmap))
                # Resize heatmap back to ROI size for overlay
                heatmap_resized = cv2.resize(heatmap, (resized.shape[1], resized.shape[0]))
            elif output_data.ndim == 1:
                score = float(output_data[0])
                heatmap_resized = None
            elif output_data.ndim == 2 and output_data.shape[1] == 1:
                score = float(output_data[0, 0])
                heatmap_resized = None
            else:
                score = float(np.max(output_data))
                heatmap_resized = None

            judgement = "OK" if score < threshold else "NG"
            elapsed = (time.perf_counter() - start) * 1000

            # Track latency
            with self._lock:
                self._latencies.append(elapsed)
                if len(self._latencies) > self._max_latency_samples:
                    self._latencies.pop(0)

            return InferenceResult(
                score=score,
                judgement=judgement,
                heatmap=heatmap_resized,
                latency_ms=elapsed,
                threshold=threshold,
                roi_cropped=resized,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Inference error: %s", e)
            # Fail-safe: return NG
            return InferenceResult(
                score=1.0, judgement="NG", latency_ms=elapsed,
                threshold=threshold, roi_cropped=resized
            )

    # ---- Hot-swap helper ----

    def hot_swap(self, new_model_path: Path, threshold: Optional[float] = None) -> None:
        """
        Hot-swap model atomically. Model lama dipakai sampai yang baru siap.
        """
        old_model_path = self._model_path
        try:
            self.load_model(new_model_path, threshold)
            if old_model_path and old_model_path != new_model_path:
                logger.info("Hot-swap: %s → %s", old_model_path, new_model_path)
        except Exception as e:
            logger.error("Hot-swap failed: %s", e)
            raise InferenceEngineError(f"Hot-swap failed: {e}") from e


def overlay_heatmap(
    image: npt.NDArray,
    heatmap: npt.NDArray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> npt.NDArray:
    """
    Overlay heatmap on image with transparency.
    Returns BGR image suitable for display.
    """
    if heatmap is None:
        return image

    # Ensure same size
    h, w = image.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))

    # Normalize to 0-255 uint8
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_norm, colormap)

    # Blend
    if image.shape[2] == 3:
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    else:
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    return overlay
