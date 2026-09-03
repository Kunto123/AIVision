"""
VisionInspect - Inference Engine
Jalankan inferensi per ROI + hot-swap model atomik (double-buffer, thread-safe).
Alur: ROI crop → resize → infer → score/judgement (+ heatmap untuk mode anomaly).
Mode dipilih otomatis dari file meta di samping model.xml:
  yolo (klasifikasi OK/NG) | anomaly (PatchCore/EfficientAd via OpenVINO) | simple (z-score, tanpa torch).
"""

import gc
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Full as QueueFull
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import cv2

from visioninspect.core.roi_mask import apply_polygon_mask
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
    score: float           # similarity score [0, 1] — 1.0 = mirip OK, 0.0 = anomali total
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

    def __init__(self, input_size: int = 256, device: str = "CPU",
                 cache_dir: Optional[Path] = None,
                 cpu_pcore_only: bool = False):
        self._input_size = input_size
        # Tugas 5: device inferensi bisa dipilih (CPU / GPU / AUTO). iGPU jauh
        # lebih cepat untuk model besar DAN membebaskan CPU untuk GUI —
        # terukur di PC dev (i5-7200U + HD 620): PatchCore 964→542 ms,
        # YOLO11l-cls 1058→160 ms. Default tetap CPU (paling aman/portabel).
        self._device = (device or "CPU").upper()
        self._active_device: Optional[str] = None   # device yg BENAR dipakai
        # CPU hybrid (P-core + E-core, mis. i3-1315U): OpenVINO membagi satu
        # inference ke semua thread lalu menunggu yang paling lambat — thread
        # di E-core menahan seluruh inference. PCORE_ONLY membuat latency
        # lebih stabil sekaligus menyisakan E-core untuk GUI + decode video.
        self._cpu_pcore_only = bool(cpu_pcore_only)
        self._lock = threading.Lock()
        self._model: Optional[ov.CompiledModel] = None
        self._model_path: Optional[Path] = None
        self._threshold: float = 0.5
        # Threshold PER ROI (kunci = roi uid). Tiap ROI melihat fitur yang
        # berbeda, jadi ambang yang pas untuk satu ROI belum tentu pas untuk
        # ROI lain. Kosong = semua ROI memakai `_threshold` global (perilaku
        # lama, dan tetap begitu untuk template yang belum menyetel per-ROI).
        self._threshold_per_roi: dict = {}
        # Referensi normalisasi skor (raw PatchCore → [0,1]); score_ref → 0.5.
        # Dibaca dari norm.json di samping model.xml saat load_model.
        # _score_ref = fallback global; _score_ref_per_roi = {roi_uid: ref}
        # (multi-ROI: tiap ROI punya skala skor berbeda, perlu ref sendiri).
        self._score_ref: Optional[float] = None
        self._score_ref_per_roi: dict = {}
        # Mode model: "anomaly" (PatchCore/EfficientAd — skor anomali → similarity)
        # atau "yolo" (klasifikasi OK/NG per crop — probabilitas kelas).
        # Terdeteksi otomatis dari yolo_meta.json di samping model.xml.
        self._algorithm: str = "anomaly"
        self._yolo_names: list = ["OK", "NG"]   # urutan kelas output model YOLO
        self._yolo_task: str = "classify"        # classify | detect
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
                # Tugas 5: model cache — WAJIB kalau device GPU dipakai.
                # Compile GPU tanpa cache ±18 detik tiap load (hot-swap jadi
                # tidak terpakai); dengan cache ±0,4 detik (terukur).
                if cache_dir:
                    try:
                        Path(cache_dir).mkdir(parents=True, exist_ok=True)
                        self._core.set_property({"CACHE_DIR": str(cache_dir)})
                        logger.info("OpenVINO CACHE_DIR: %s", cache_dir)
                    except Exception as e:
                        logger.warning("CACHE_DIR gagal diset (%s): %s",
                                       cache_dir, e)
                # Coba deteksi GPU device via OpenVINO
                try:
                    if self._core is not None:
                        gpu_devices = [d for d in self._core.available_devices if d.upper() in ("GPU", "IGPU")]
                        if gpu_devices:
                            for d in gpu_devices:
                                name = self._core.get_property(d, "FULL_DEVICE_NAME")
                                logger.info("OpenVINO GPU terdeteksi: %s (%s)", d, name)
                        else:
                            logger.info("OpenVINO GPU device tidak tersedia — hanya CPU: %s",
                                        self._core.available_devices)
                except Exception:
                    pass
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

    def set_roi_thresholds(self, per_roi: Optional[dict]) -> None:
        """Setel threshold per ROI ({uid: nilai}). None/kosong = pakai global.

        Nilai di luar [0,1] dan uid kosong diabaikan — supaya config yang
        rusak tidak diam-diam membuat satu ROI selalu lolos atau selalu NG.
        """
        clean = {}
        for uid, val in (per_roi or {}).items():
            if not uid:
                continue
            try:
                f = float(val)
            except (TypeError, ValueError):
                continue
            if 0.0 <= f <= 1.0:
                clean[str(uid)] = f
            else:
                logger.warning(
                    "Threshold ROI '%s' di luar [0,1] (%s) — diabaikan, "
                    "ROI ini memakai threshold global.", uid, val)
        with self._lock:
            self._threshold_per_roi = clean
        if clean:
            logger.info("Threshold per ROI aktif untuk %d ROI: %s",
                        len(clean), clean)

    def get_roi_threshold(self, uid: Optional[str]) -> float:
        """Threshold efektif untuk satu ROI (global bila belum disetel)."""
        if uid and uid in self._threshold_per_roi:
            return self._threshold_per_roi[uid]
        return self._threshold

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
            self._algorithm = "anomaly"
            self._yolo_names = ["OK", "NG"]
            self._yolo_task = "classify"

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

        # Mode YOLO: kalau ada yolo_meta.json di samping model.xml (ditulis
        # saat training/export YOLO), engine memakai jalur klasifikasi OK/NG
        # (probabilitas kelas) — bukan skor anomali PatchCore/EfficientAd.
        algorithm, yolo_names, yolo_task = self._read_yolo_meta(xml_path)

        try:
            # Retry compile_model — .bin bisa di-lock OpenVINO/Defender (WinError 32 / Errno 13)
            compiled = None
            for attempt in range(5):
                try:
                    model = self._core.read_model(str(xml_path))
                    compiled = self._compile_on_device(model)
                    break
                except (PermissionError, OSError) as e:
                    if attempt == 4:
                        raise InferenceEngineError(f"Model load failed (bin locked): {e}") from e
                    logger.warning("Model compile retry %d/5: %s", attempt + 1, e)
                    gc.collect()
                    time.sleep(0.5 * (attempt + 1))

            # Get input shape — guard terhadap dynamic shape (?,3,?,?)
            input_key = compiled.input(0)
            pshape = input_key.get_partial_shape()
            last_dim = pshape[-1]
            self._input_size = last_dim.get_length() if last_dim.is_static else self._input_size

            # Pengaman: apa yang dipilih di TEACH vs apa yang benar-benar
            # terpasang. `algorithm` dioper eksplisit karena atomic swap di
            # bawah belum jalan — self._algorithm masih milik model lama.
            self._warn_config_model_mismatch(xml_path, algorithm)

            # Atomic swap
            with self._lock:
                old_model = self._model
                self._model = compiled
                self._model_path = model_path
                self._score_ref = score_ref
                self._score_ref_per_roi = score_ref_per_roi
                # Threshold per ROI milik TEMPLATE, bukan model — dibersihkan
                # di sini supaya tidak bocor dari template sebelumnya, lalu
                # dipasang ulang oleh pemanggil (set_roi_thresholds).
                self._threshold_per_roi = {}
                self._algorithm = algorithm
                self._yolo_names = yolo_names
                self._yolo_task = yolo_task
                if threshold is not None:
                    self._threshold = threshold
                # Clean old model
                del old_model

            logger.info("Model loaded successfully: %s (input size: %d, score_ref: %s, "
                        "mode: %s, device: %s)",
                        xml_path, self._input_size,
                        f"{score_ref:.4f}" if score_ref else "none",
                        algorithm, self._active_device)
        except Exception as e:
            raise InferenceEngineError(f"Model load failed: {e}") from e

    def _compile_on_device(self, model):
        """Compile ke device pilihan; gagal → fallback CPU + WARNING.

        Device tidak tersedia TIDAK boleh membuat aplikasi crash — lini
        produksi harus tetap jalan meski iGPU/driver bermasalah.
        """
        want = self._device or "CPU"
        avail = []
        try:
            avail = [d.upper() for d in self._core.available_devices]
        except Exception:
            pass
        if want != "AUTO" and avail and want not in avail:
            logger.warning(
                "Device '%s' tidak tersedia (ada: %s) — fallback ke CPU",
                want, avail)
            want = "CPU"

        cfg = {}
        # Properti hybrid P/E-core hanya valid untuk plugin CPU — dikirim ke
        # GPU akan melempar exception.
        if want == "CPU" and self._cpu_pcore_only:
            cfg = {"SCHEDULING_CORE_TYPE": "PCORE_ONLY",
                   "ENABLE_HYPER_THREADING": "NO"}
        try:
            compiled = self._core.compile_model(model, want, cfg)
            self._active_device = want
            if cfg:
                logger.info("Compile di %s dengan %s", want, cfg)
            return compiled
        except (PermissionError, OSError):
            raise            # ditangani retry .bin-locked di load_model
        except Exception as e:
            if want == "CPU" and not cfg:
                raise
            logger.warning(
                "Compile di '%s' gagal (%s) — fallback ke CPU polos", want, e)
            compiled = self._core.compile_model(model, "CPU")
            self._active_device = "CPU"
            return compiled

    @property
    def active_device(self) -> str:
        """Device yang BENAR-BENAR dipakai (bisa beda dari yang diminta
        kalau terjadi fallback)."""
        return self._active_device or "-"

    def set_device(self, device: str, cpu_pcore_only: Optional[bool] = None
                   ) -> None:
        """Ganti device inferensi. Model yang sedang aktif di-compile ulang
        (hot-swap) supaya perubahan langsung berlaku tanpa restart.

        PERINGATAN: iGPU menghitung dengan presisi berbeda (FP16) sementara
        `score_ref` di norm.json dikalibrasi pada CPU FP32 — skor bisa
        bergeser. Validasi skor CPU vs GPU sebelum dipakai produksi.
        """
        new_dev = (device or "CPU").upper()
        changed = (new_dev != self._device
                   or (cpu_pcore_only is not None
                       and bool(cpu_pcore_only) != self._cpu_pcore_only))
        self._device = new_dev
        if cpu_pcore_only is not None:
            self._cpu_pcore_only = bool(cpu_pcore_only)
        if not changed:
            return
        path = self._model_path
        if path is not None:
            logger.info("Device diubah ke %s — compile ulang model %s",
                        new_dev, path)
            self.load_model(path)

    @staticmethod
    def _read_model_meta(xml_path: Path) -> dict:
        """Baca model_meta.json di samping model.xml (Tugas 8).

        Fallback ke folder 'openvino' bila model dimuat dari 'openvino_int8'
        (pola sama dengan _read_norm / _read_yolo_meta). Model lama tidak
        punya file ini → kembalikan {} tanpa keluhan.
        """
        import json
        candidates = [xml_path.parent / "model_meta.json"]
        if xml_path.parent.name == "openvino_int8":
            candidates.append(
                xml_path.parent.parent / "openvino" / "model_meta.json")
        for p in candidates:
            if p.exists():
                try:
                    with open(p, encoding="utf-8") as f:
                        return json.load(f) or {}
                except Exception as e:
                    logger.warning("Gagal baca model_meta.json (%s): %s", p, e)
        return {}

    def _warn_config_model_mismatch(self, xml_path: Path, algorithm: str):
        """Bandingkan model yang BENAR-BENAR terpasang dengan config template.

        Nama template = identitas PART, bukan nama arsitektur — jadi nama
        TIDAK pernah dipakai untuk memeriksa apa pun di sini. Yang dibandingkan
        adalah pengaturan yang dipilih operator di halaman TEACH (config)
        terhadap catatan independen hasil export (model_meta.json + shape IR).

        Field pembanding mengikuti algoritma, karena tiap algoritma memakai
        field yang berbeda di TEACH:
          yolo                  → `yolo_pretrained`  (field `backbone` tidak
                                  berlaku & memang disembunyikan di UI)
          patchcore/efficientad → `backbone`

        Hanya log WARNING — tidak mengubah perilaku inference.
        """
        import json
        cfg_path = xml_path.parent.parent.parent / "config.json"
        if not cfg_path.exists():
            return
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            cfg_size = cfg.get("input_size")
            if cfg_size and int(cfg_size) != int(self._input_size):
                logger.warning(
                    "INPUT SIZE MISMATCH: model IR ber-input %d tapi config "
                    "template %s tertulis input_size=%s. Skor tidak cocok — "
                    "wajib training ulang sebelum dipakai produksi.",
                    self._input_size, cfg_path, cfg_size)

            # backbone/algorithm tidak terbaca dari IR → dibandingkan dengan
            # model_meta.json (ditulis saat export). Model lama tidak punya
            # file ini; itu bukan error, cukup dilewati.
            meta = self._read_model_meta(xml_path)
            if not meta:
                return
            # Algoritma yang benar-benar terpasang: utamakan catatan export,
            # fallback ke deteksi dari yolo_meta.json.
            algo = str(meta.get("algorithm") or algorithm or "").lower()

            if algo == "yolo":
                want = cfg.get("yolo_pretrained")
                have = meta.get("yolo_pretrained") or meta.get("backbone")
                field = "yolo_pretrained"
            else:
                want = cfg.get("backbone")
                have = meta.get("backbone")
                field = "backbone"
            if want and have and str(want) != str(have):
                logger.warning(
                    "MODEL MISMATCH: model di disk dilatih dengan %s='%s' "
                    "tapi pengaturan TEACH template ini tertulis '%s'. "
                    "Yang berjalan adalah model di disk — training ulang "
                    "untuk menyamakan.", field, have, want)

            cfg_algo = str(cfg.get("algorithm", "") or "").lower()
            if cfg_algo and algo and cfg_algo != algo:
                logger.warning(
                    "ALGORITHM MISMATCH: model di disk = '%s' tapi pengaturan "
                    "TEACH template ini = '%s'. Training ulang untuk "
                    "menyamakan.", algo, cfg_algo)
        except Exception as e:  # config tak terbaca → bukan urusan kita
            logger.debug("Cek kecocokan config vs model dilewati: %s", e)

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

    @staticmethod
    def _read_yolo_meta(xml_path: Path):
        """Baca yolo_meta.json di samping model.xml → mode inferensi YOLO.

        Returns (algorithm, yolo_names, yolo_task). Tanpa meta file → mode
        anomaly biasa (PatchCore/EfficientAd). yolo_meta.json ditulis saat
        training/export YOLO: {"names": ["OK","NG"], "task": "classify"}.
        """
        import json
        candidates = [xml_path.parent / "yolo_meta.json"]
        if xml_path.parent.name == "openvino_int8":
            candidates.append(xml_path.parent.parent / "openvino" / "yolo_meta.json")
        for meta_path in candidates:
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        data = json.load(f)
                    names = list(data.get("names") or ["OK", "NG"])
                    task = str(data.get("task", "classify")).lower()
                    if task not in ("classify", "detect"):
                        task = "classify"
                    logger.info("YOLO meta ditemukan: %s (task=%s, classes=%s)",
                                meta_path, task, names)
                    return "yolo", names, task
                except Exception as e:
                    logger.warning("Gagal baca yolo_meta.json (%s): %s", meta_path, e)
        return "anomaly", ["OK", "NG"], "classify"

    def unload_model(self) -> None:
        """Unload current model (both OpenVINO and simple)."""
        with self._lock:
            self._model = None
            self._model_path = None
            self._score_ref = None
            self._score_ref_per_roi = {}
            self._threshold_per_roi = {}
            self._simple_mean = None
            self._simple_std = None
            self._simple_loaded = False
            self._algorithm = "anomaly"
            self._yolo_names = ["OK", "NG"]
            self._yolo_task = "classify"
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
            # Mask polygon (opsional) — HARUS identik dengan yang diterapkan
            # saat training (lihat training_worker.py::_crop_images_to_rois).
            # Tidak ada override per-gambar di sini: inference berjalan
            # real-time tanpa review manusia, jadi selalu pakai default ROI.
            cropped = apply_polygon_mask(cropped, roi.get("mask_polygon"))
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
            threshold_per_roi = self._threshold_per_roi
            score_ref = self._score_ref
            score_ref_per_roi = self._score_ref_per_roi
            simple_mean = self._simple_mean
            simple_std = self._simple_std
            simple_loaded = self._simple_loaded
            algorithm = self._algorithm
            yolo_names = self._yolo_names
            yolo_task = self._yolo_task

        # Multi-ROI: tiap ROI punya skala skor berbeda → pakai ref khusus ROI
        # ini (by uid) bila ada, jika tidak fallback ke ref global.
        roi_uid = roi.get("uid") if roi else None
        if roi_uid in score_ref_per_roi:
            score_ref = score_ref_per_roi[roi_uid]
        # Threshold per ROI — fallback ke global bila ROI ini belum disetel.
        if roi_uid in threshold_per_roi:
            threshold = threshold_per_roi[roi_uid]

        if model is None and not simple_loaded:
            elapsed = (time.perf_counter() - start) * 1000
            # No model = tidak bisa deteksi anomali → similarity=1.0 (lolos)
            return InferenceResult(
                score=1.0, judgement="OK", latency_ms=elapsed,
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
                    # Sama seperti jalur OpenVINO di atas — mask HARUS
                    # identik dengan training (lihat komentar di blok
                    # pertama fungsi ini).
                    cropped = apply_polygon_mask(cropped, roi.get("mask_polygon"))
                else:
                    cropped = frame
                resized = cv2.resize(cropped, (self._input_size, self._input_size))
                img_f = resized.astype(np.float32) / 255.0
                # Per-pixel z-score, then mean z-score as anomaly score
                z = np.abs(img_f - simple_mean) / simple_std
                score = float(np.mean(z))
                score = max(0.0, min(1.0, score))
                # Konversi anomaly → similarity (1.0 = mirip OK)
                score = 1.0 - score
                judgement = "OK" if score >= threshold else "NG"
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
                # Fail-safe: error → 0% similarity → NG
                return InferenceResult(
                    score=0.0, judgement="NG", latency_ms=elapsed,
                    threshold=threshold, roi_cropped=resized,
                )

        # ── YOLO mode: klasifikasi OK/NG per crop (probabilitas kelas) ──────
        # Model YOLO (hasil training/export ultralytics → OpenVINO) output
        # probabilitas per kelas, bukan anomaly score. Judgement = prob kelas
        # "OK" (atau 1 - prob "NG") >= threshold → OK.
        if algorithm == "yolo":
            try:
                # Preprocess sama dengan jalur anomaly: BGR→RGB, HWC→NCHW, /255.
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                input_tensor = rgb.astype(np.float32) / 255.0
                input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC → CHW
                input_tensor = np.expand_dims(input_tensor, axis=0)    # CHW → NCHW

                infer_request = model.create_infer_request()
                infer_request.set_input_tensor(ov.Tensor(input_tensor))
                infer_request.start_async()
                infer_request.wait()

                # Output classification: [1, C] logits/probs.
                # Output detect: [1, 4+C, N] anchors → fallback ambil prob kelas.
                probs = None
                for i in range(len(model.outputs)):
                    data = np.asarray(infer_request.get_output_tensor(i).data)
                    if data.ndim == 2 and data.shape[0] == 1:
                        probs = data[0]
                        break
                if probs is None:
                    data = np.asarray(infer_request.get_output_tensor(0).data)
                    if data.ndim == 3 and data.shape[1] > 4:
                        cls_rows = data[0, 4:, :]
                        probs = cls_rows.max(axis=1)
                    else:
                        probs = data.reshape(-1)

                # Softmax kalau masih logits (bisa negatif / sum != 1)
                if probs.shape[0] < 2:
                    score = float(probs.reshape(-1)[0])
                else:
                    if float(probs.min()) < 0 or not np.isclose(
                            float(probs.sum()), 1.0, atol=1e-2):
                        e = np.exp(probs - probs.max())
                        probs = e / e.sum()
                    names = [str(n).strip() for n in yolo_names]
                    ok_idx = names.index("OK") if "OK" in names else 0
                    ng_idx = names.index("NG") if "NG" in names else None
                    if ok_idx < probs.shape[0]:
                        score = float(probs[ok_idx])
                    elif ng_idx is not None and ng_idx < probs.shape[0]:
                        score = 1.0 - float(probs[ng_idx])
                    else:
                        score = float(probs[0])
                    score = max(0.0, min(1.0, score))

                judgement = "OK" if score >= threshold else "NG"
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
                logger.error("YOLO inference error: %s", e)
                # Fail-safe: error → 0% similarity → NG
                return InferenceResult(
                    score=0.0, judgement="NG", latency_ms=elapsed,
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
            # Konversi anomaly score → similarity score (1.0 = mirip OK)
            score = 1.0 - score
            score = max(0.0, min(1.0, score))

            judgement = "OK" if score >= threshold else "NG"
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
            # Fail-safe: error → 0% similarity → NG
            return InferenceResult(
                score=0.0, judgement="NG", latency_ms=elapsed,
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
