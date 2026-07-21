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
    # OpenVINO 2025+ API (Core moved to top-level)
    try:
        _OV_CORE = ov.Core
        HAS_OPENVINO = True
    except AttributeError:
        from openvino import Core as _OV_CORE
        HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    _OV_CORE = None
    logger.warning("OpenVINO not installed. Inference will be unavailable.")
except OSError as e:
    HAS_OPENVINO = False
    _OV_CORE = None
    logger.warning("OpenVINO DLL load failed (kemungkinan interpreter/arsitektur salah): %s", e)


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
        # Referensi normalisasi skor (raw PatchCore → [0,1]); score_ref → 0.5.
        # Dibaca dari norm.json di samping model.xml saat load_model.
        # _score_ref = fallback global; _score_ref_per_roi = {roi_uid: ref}
        # (multi-ROI: tiap ROI punya skala skor berbeda, perlu ref sendiri).
        self._score_ref: Optional[float] = None
        self._score_ref_per_roi: dict = {}
        self._use_ov = HAS_OPENVINO
        self._core: Optional[_OV_CORE] = None

        # Simple model data (fallback without PyTorch)
        self._simple_mean: Optional[npt.NDArray] = None
        self._simple_std: Optional[npt.NDArray] = None
        self._simple_loaded = False

        # Latency tracking
        self._latencies: list[float] = []
        self._max_latency_samples = 100

        if self._use_ov:
            try:
                self._core = _OV_CORE()
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
        std = np.clip(std, 0.05, None)  # safety floor — toleransi fluktuasi exposure
        self._input_size = mean.shape[0]

        with self._lock:
            self._simple_mean = mean
            self._simple_std = std
            self._simple_loaded = True
            self._threshold = threshold
            self._model = None  # clear any OpenVINO model
            self._score_ref = None  # simple model pakai z-score, bukan pred_score
            self._score_ref_per_roi = {}

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

        # Kalibrasi normalisasi skor (opsional). Ditulis saat training di
        # samping model.xml. Skor PatchCore mentah tak di [0,1]; score_ref → 0.5.
        score_ref, score_ref_per_roi = self._read_norm(xml_path)

        try:
            # Compile new model first (don't swap yet)
            model = self._core.read_model(str(xml_path))
            compiled = self._core.compile_model(model, "CPU")

            # Get input shape — guard terhadap dynamic shape (?,3,?,?)
            input_key = compiled.input(0)
            pshape = input_key.get_partial_shape()
            last_dim = pshape[-1]
            self._input_size = last_dim.get_length() if last_dim.is_static else self._input_size

            # Atomic swap
            with self._lock:
                old_model = self._model
                self._model = compiled
                self._model_path = model_path
                self._score_ref = score_ref
                self._score_ref_per_roi = score_ref_per_roi
                if threshold is not None:
                    self._threshold = threshold
                # Clean old model
                del old_model

            logger.info("Model loaded successfully: %s (input size: %d, score_ref: %s)",
                        xml_path, self._input_size,
                        f"{score_ref:.4f}" if score_ref else "none")
        except Exception as e:
            raise InferenceEngineError(f"Model load failed: {e}") from e

    @staticmethod
    def _read_norm(xml_path: Path):
        """Baca kalibrasi dari norm.json di samping model.xml (opsional).

        Returns (score_ref, per_roi) — score_ref = fallback global (float|None),
        per_roi = {roi_uid: ref}. Fallback ke folder 'openvino' bila model
        dimuat dari 'openvino_int8'.
        """
        import json
        candidates = [xml_path.parent / "norm.json"]
        if xml_path.parent.name == "openvino_int8":
            candidates.append(xml_path.parent.parent / "openvino" / "norm.json")
        for norm_path in candidates:
            if norm_path.exists():
                try:
                    with open(norm_path) as f:
                        data = json.load(f)
                    ref = float(data.get("score_ref", 0) or 0)
                    per_roi = {}
                    for uid, v in (data.get("per_roi") or {}).items():
                        try:
                            fv = float(v)
                            if fv > 0:
                                per_roi[uid] = fv
                        except (TypeError, ValueError):
                            pass
                    return (ref if ref > 0 else None), per_roi
                except Exception as e:
                    logger.warning("Gagal baca norm.json (%s): %s", norm_path, e)
        return None, {}

    def unload_model(self) -> None:
        """Unload current model (both OpenVINO and simple)."""
        with self._lock:
            self._model = None
            self._model_path = None
            self._score_ref = None
            self._score_ref_per_roi = {}
            self._simple_mean = None
            self._simple_std = None
            self._simple_loaded = False
        logger.info("Model unloaded")

    # ---- Inference ----

    def infer(self, frame: npt.NDArray, roi: Optional[dict] = None,
              track_latency: bool = True) -> InferenceResult:
        """
        Run inference on frame (or ROI-cropped region).
        Returns InferenceResult with score, judgement, heatmap.

        track_latency=False skips updating the shared latency_avg_ms/p95_ms
        rolling stats (used by the Diagnostics page for live RUN monitoring) —
        set this when calling infer() outside the live inspection loop (e.g.
        batch-testing static photos) so those runs don't skew production
        latency stats.
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
            score_ref = self._score_ref
            score_ref_per_roi = self._score_ref_per_roi
            simple_mean = self._simple_mean
            simple_std = self._simple_std
            simple_loaded = self._simple_loaded

        # Multi-ROI: tiap ROI punya skala skor berbeda → pakai ref khusus ROI
        # ini (by uid) bila ada, jika tidak fallback ke ref global.
        if roi and roi.get("uid") in score_ref_per_roi:
            score_ref = score_ref_per_roi[roi["uid"]]

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
                if track_latency:
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
            # Preprocess: BGR→RGB (model dilatih & dikalibrasi pd RGB — tanpa
            # konversi ini skor bergeser & OK bisa salah jadi NG), HWC→NCHW, /255.
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_tensor = rgb.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC → CHW
            input_tensor = np.expand_dims(input_tensor, axis=0)    # CHW → NCHW

            # Infer
            infer_request = model.create_infer_request()
            infer_request.set_input_tensor(ov.Tensor(input_tensor))
            infer_request.start_async()
            infer_request.wait()

            # PatchCore mengekspor 2 output: anomaly_map [1,1,H,W] + pred_score [1].
            # get_output_tensor() TANPA index gagal ("outputs.size() == 1"),
            # jadi baca tiap output by index & kenali dari bentuknya:
            # ndim>=3 → heatmap; selain itu → pred_score (diutamakan sbg skor).
            raw_score = None
            heatmap_resized = None
            for i in range(len(model.outputs)):
                data = infer_request.get_output_tensor(i).data
                if data.ndim >= 3:                       # anomaly_map
                    hm = data
                    while hm.ndim > 2:
                        hm = hm[0]
                    heatmap_resized = cv2.resize(
                        hm.astype(np.float32),
                        (resized.shape[1], resized.shape[0]))
                    if raw_score is None:                # fallback skor
                        raw_score = float(np.max(hm))
                else:                                    # pred_score
                    raw_score = float(np.asarray(data).reshape(-1)[0])
            if raw_score is None:
                raw_score = 0.0

            # Skor PatchCore mentah (jarak fitur, mis. ~20) tidak berada di [0,1].
            # Normalisasi pakai score_ref hasil kalibrasi training (score_ref → 0.5),
            # agar sebanding dgn threshold. Tanpa score_ref, pakai skor mentah apa adanya.
            if score_ref and score_ref > 0:
                score = min(1.0, max(0.0, 0.5 * raw_score / score_ref))
            else:
                score = raw_score

            judgement = "OK" if score < threshold else "NG"
            elapsed = (time.perf_counter() - start) * 1000

            # Track latency
            if track_latency:
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
