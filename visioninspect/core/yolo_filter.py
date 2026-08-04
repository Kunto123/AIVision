"""YOLO class pre-filter — filter part by class sebelum scoring Anomalib.

VisionInspect tetap memakai model anomali bawaan (OpenVINO EfficientAd/
PatchCore) untuk scoring defect. Layer YOLO di sini bersifat OPSIONAL: bila
aktif, frame/ROI dicek dulu apakah part kelas yang diharapkan terdeteksi.
Kalau tidak cocok → hasil NG (class mismatch), tanpa masuk scoring anomali.

Ultralytics tidak wajib — module ini bekerja tanpa instalasi (HAS_ULTRALYTICS
False → detektor tidak tersedia dan integrasi melewatinya).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO as _UltrYOLO
    HAS_ULTRALYTICS = True
except Exception:  # pragma: no cover - ultralytics belum terpasang
    HAS_ULTRALYTICS = False
    _UltrYOLO = None


def class_filter_matches(
    detections,
    expected_classes,
    min_conf: float = 0.25,
    roi: Optional[dict] = None,
) -> bool:
    """True bila ada deteksi kelas yang diharapkan (conf >= min_conf).

    Args:
        detections: [{class, conf, x1, y1, x2, y2}, ...] — dari YOLODetector.
        expected_classes: list nama kelas yang boleh diterima.
        min_conf: ambang confidence per deteksi.
        roi: dict {x,y,width,height} — kalau ada, deteksi harus memotong ROI.

    Kalau expected_classes kosong → True (tidak menyaring). Tanpa deteksi yang
    cocok → False (NG).
    """
    expected = {str(c).strip() for c in expected_classes if str(c).strip()}
    if not expected:
        return True  # tidak ada kriteria kelas → lolos (tidak menyaring)
    if not detections:
        return False
    for d in detections:
        cls_name = str(d.get("class", ""))
        if cls_name not in expected:
            continue
        if float(d.get("conf", 0.0)) < min_conf:
            continue
        if roi is not None and not _box_intersects(d, roi):
            continue
        return True
    return False


def _box_intersects(det: dict, roi: dict) -> bool:
    dx1, dy1, dx2, dy2 = det["x1"], det["y1"], det["x2"], det["y2"]
    rx1, ry1 = roi["x"], roi["y"]
    rx2, ry2 = rx1 + roi["width"], ry1 + roi["height"]
    return not (dx2 < rx1 or dx1 > rx2 or dy2 < ry1 or dy1 > ry2)


class YOLODetector:
    """Wrapper tipis di atas ultralytics.YOLO — mencerna model .pt/.onnx.

    Kalau ultralytics tidak terpasang / model gagal dimuat, `available=False`
    dan `error` memuat pesan — integrasi harus memeriksa status ini.
    """

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model = None
        self.error: Optional[str] = None
        if not HAS_ULTRALYTICS:
            self.error = "ultralytics belum terpasang — pasang via pip install ultralytics"
            return
        try:
            self._model = _UltrYOLO(model_path)
        except Exception as e:  # pragma: no cover - tergantung model user
            self.error = str(e)
            logger.warning("YOLO model load gagal (%s): %s", model_path, e)

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame) -> Optional[list]:
        """Deteksi objek. Return list[dict] atau None bila gagal / tak tersedia.

        Setiap dict: {"class": str, "conf": float, "x1","y1","x2","y2": float}
        """
        if self._model is None:
            return None
        try:
            results = self._model(frame, verbose=False)
            if not results:
                return []
            r = results[0]
            names = getattr(r, "names", {})
            if r.boxes is None or r.boxes.cls is None:
                return []
            out: list[dict] = []
            cls_list = r.boxes.cls.detach().cpu().tolist()
            conf_list = r.boxes.conf.detach().cpu().tolist()
            xyxy_list = r.boxes.xyxy.detach().cpu().tolist()
            for cls_id, conf, xyxy in zip(cls_list, conf_list, xyxy_list):
                out.append({
                    "class": str(names.get(int(cls_id), int(cls_id))),
                    "conf": float(conf),
                    "x1": float(xyxy[0]), "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]), "y2": float(xyxy[3]),
                })
            return out
        except Exception as e:  # pragma: no cover
            logger.warning("YOLO detect error: %s", e)
            return None