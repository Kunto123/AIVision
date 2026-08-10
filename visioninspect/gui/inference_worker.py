"""InferenceWorker — jalankan pipeline inference CPU-bound di thread terpisah.

Tugas 3 (perbaikan.md): sebelumnya ``_on_frame_for_inference`` dijalankan
langsung di GUI thread, sehingga UI membeku 0,65–1 dtk per frame (loop
infer per-ROI). Worker ini menjalankan bagian berat di thread sendiri:

  1) Part Presence Check (evaluate)          — murni OpenCV
  2) YOLO class pre-filter (kalau enabled)   — murni OpenVINO/numpy
  3) Loop infer per ROI                      — InferenceEngine (punya
                                               threading.Lock internal,
                                               aman dipakai lintas thread)

Lalu mengirim hasil via signal ``result_ready`` — GUI thread melakukan
SEMUA efek samping (update UI, PLC, history, counter, export, ack replay).
Worker TIDAK pernah menyentuh objek Qt widget; frame yang dikembalikan
adalah numpy array yang sama (tidak disalin, tidak diubah).

Catatan aman: ``submit`` → ``infer`` di-connect eksplisit dengan
QueuedConnection, jadi memanggil ``worker.submit.emit(...)`` dari thread
mana pun SELALU dieksekusi di thread worker ini.
"""

import logging
from typing import Optional

from PySide6.QtCore import QObject, Qt, Signal, Slot

from visioninspect.core import part_check as pc_module
from visioninspect.core.yolo_filter import YOLODetector, class_filter_matches

logger = logging.getLogger(__name__)


class InferenceWorker(QObject):
    """Worker inference — hidup di QThread terpisah."""

    # Request infer: (frame, pc_cfg, pc_state, rois, rois_uid, rois_label, yolo_cfg)
    submit = Signal(object, object, str, list, list, list, object)
    # Hasil lengkap berupa dict (lihat build_result di bawah)
    result_ready = Signal(object)

    def __init__(self, engine):
        super().__init__()
        self._engine = engine
        self._yolo_det = None  # lazy-load YOLODetector (filter kelas)
        # QueuedConnection EKSPLISIT — receiver & emitter sama objek, tanpa
        # ini PySide akan jalan direct di thread pemanggil (salah!).
        self.submit.connect(self.infer, Qt.QueuedConnection)

    # ---- Detektor YOLO (filter kelas) — lazy, dibangun di thread worker ----

    def _ensure_yolo_detector(self, yolo_cfg: dict):
        if self._yolo_det is not None:
            return self._yolo_det
        path = str((yolo_cfg or {}).get("model_path") or "").strip()
        if not (yolo_cfg or {}).get("enabled") or not path:
            return None
        try:
            det = YOLODetector(path)
            if not det.available:
                logger.warning("YOLO filter nonaktif: %s", det.error)
                return None
            self._yolo_det = det
        except Exception as e:
            logger.warning("YOLO filter init error: %s", e)
            self._yolo_det = None
        return self._yolo_det

    # ---- Pipeline inference murni ----

    def build_result(self, frame, pc_state: str) -> dict:
        return {
            "frame": frame,
            "pc_state": pc_state,
            "pc_blocked": False,
            "pc_result": None,
            "part_check_score": None,
            "class_ng": False,
            "overall_ng": False,
            "roi_results": [],
            "worst_score": 1.0,
            "avg_latency": 0.0,
            "heatmap": None,
            "raw_judgement": "OK",
            "error": None,
        }

    @Slot(object, object, str, list, list, list, object)
    def infer(self, frame, pc_cfg: Optional[dict], pc_state: str,
              rois: list, rois_uid: list, rois_label: list,
              yolo_cfg: Optional[dict]):
        """Jalankan bagian berat pipeline. Selalu emit result_ready — bahkan
        saat error — supaya token replay tidak pernah hilang."""
        res = self.build_result(frame, pc_state)
        try:
            # ── Step 1: Part Presence Check (evaluate — mahal) ──
            # State (active/disabled/incomplete) sudah dihitung GUI; di sini
            # hanya evaluate kalau aktif. pc_blocked=True → GUI stop QC.
            if pc_state == "active" and pc_cfg:
                try:
                    pc_result = pc_module.evaluate_part_presence(
                        frame, pc_cfg["gate_roi"], pc_cfg)
                except Exception as e:
                    logger.warning("Part check error: %s", e)
                    pc_result = None
                if pc_result is None or not pc_result.ready:
                    res["pc_blocked"] = True
                    self.result_ready.emit(res)
                    return
                res["pc_result"] = pc_result
                m = pc_result.method
                if m == 'color' and pc_result.color_score is not None:
                    res["part_check_score"] = pc_result.color_score
                elif m == 'edge' and pc_result.edge_score is not None:
                    res["part_check_score"] = pc_result.edge_score
                elif m == 'both':
                    cs = (pc_result.color_score
                          if pc_result.color_score is not None else 1.0)
                    es = (pc_result.edge_score
                          if pc_result.edge_score is not None else 1.0)
                    res["part_check_score"] = max(cs, es)
                else:
                    res["part_check_score"] = 0.0

            # ── Step 1.5: YOLO class pre-filter (opsional) ──
            # Kelas tidak cocok → NG langsung (tanpa scoring anomali).
            if yolo_cfg and yolo_cfg.get("enabled"):
                det = self._ensure_yolo_detector(yolo_cfg)
                if det is not None:
                    dets = det.detect(frame)
                    if dets is not None and not class_filter_matches(
                            dets, yolo_cfg.get("expected_classes", []),
                            min_conf=float(yolo_cfg.get("min_conf", 0.25))):
                        res["class_ng"] = True

            # ── Step 2: loop infer per ROI ──
            rois_to_check = [] if res["class_ng"] else rois
            total_latency = 0.0
            worst = 1.0
            results = []
            for idx, roi_rect in enumerate(rois_to_check):
                roi_dict = {
                    "x": roi_rect[0], "y": roi_rect[1],
                    "width": roi_rect[2], "height": roi_rect[3],
                    "uid": (rois_uid[idx] if idx < len(rois_uid) else None),
                }
                result = self._engine.infer(frame, roi=roi_dict)
                results.append({
                    "roi": roi_rect,
                    "label": (rois_label[idx] if idx < len(rois_label)
                              else f"ROI{idx + 1}"),
                    "score": result.score,
                    "judgement": result.judgement,
                    "latency": result.latency_ms,
                })
                total_latency += result.latency_ms
                if result.score < worst:
                    worst = result.score
                    res["heatmap"] = result.heatmap
                if result.judgement == "NG":
                    res["overall_ng"] = True

            res["roi_results"] = results
            res["worst_score"] = worst
            res["avg_latency"] = (total_latency / len(results)
                                  if results else 0.0)
            res["raw_judgement"] = (
                "NG" if (res["overall_ng"] or res["class_ng"]) else "OK")
        except Exception as e:
            logger.warning("Inference worker error: %s", e)
            res["error"] = str(e)
        self.result_ready.emit(res)
