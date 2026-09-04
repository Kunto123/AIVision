"""InferenceWorker — jalankan pipeline inference CPU-bound di thread terpisah.

Tugas 3 (perbaikan.md): sebelumnya ``_on_frame_for_inference`` dijalankan
langsung di GUI thread, sehingga UI membeku 0,65–1 dtk per frame (loop
infer per-ROI). Worker ini menjalankan bagian berat di thread sendiri:

  1) Part Presence Check (evaluate)  — murni OpenCV
  2) Loop infer per ROI              — InferenceEngine (punya threading.Lock
                                       internal, aman dipakai lintas thread)

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

logger = logging.getLogger(__name__)


class InferenceWorker(QObject):
    """Worker inference — hidup di QThread terpisah."""

    # Request: (seq, frame, pc_cfg, pc_state, rois, uid, label, mask_polygon).
    # `seq` dikembalikan apa adanya supaya GUI bisa buang hasil basi.
    submit = Signal(int, object, object, str, list, list, list, list)
    # Hasil lengkap berupa dict (lihat build_result di bawah)
    result_ready = Signal(object)

    def __init__(self, engine):
        super().__init__()
        self._engine = engine
        # QueuedConnection EKSPLISIT — receiver & emitter sama objek, tanpa
        # ini PySide akan jalan direct di thread pemanggil (salah!).
        self.submit.connect(self.infer, Qt.QueuedConnection)

    # ---- Pipeline inference murni ----

    def build_result(self, frame, pc_state: str, seq: int = -1) -> dict:
        return {
            "seq": seq,
            "frame": frame,
            "pc_state": pc_state,
            "pc_blocked": False,
            "pc_result": None,
            "part_check_score": None,
            "overall_ng": False,
            "roi_results": [],
            "worst_score": 1.0,
            # Margin = skor − threshold ROI itu. Inilah dasar pemilihan ROI
            # terburuk sejak threshold bisa berbeda per ROI.
            "worst_margin": 0.0,
            "worst_threshold": None,
            "worst_label": "",
            "avg_latency": 0.0,
            "heatmap": None,
            "raw_judgement": "OK",
            "error": None,
        }

    @Slot(int, object, object, str, list, list, list, list)
    def infer(self, seq: int, frame, pc_cfg: Optional[dict], pc_state: str,
              rois: list, rois_uid: list, rois_label: list,
              rois_mask_polygon: list):
        """Jalankan bagian berat pipeline. Selalu emit result_ready — bahkan
        saat error — supaya token replay tidak pernah hilang."""
        res = self.build_result(frame, pc_state, seq)
        try:
            # ── Step 1: Part Presence Check (evaluate — mahal) ──
            # State sudah dihitung GUI; pc_blocked=True → GUI stop QC.
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

            # ── Step 2: loop infer per ROI — ROI "terburuk" dipilih dari MARGIN
            # (skor − threshold ROI itu), bukan skor mentah yang tak sebanding.
            total_latency = 0.0
            worst_margin = None
            worst = 1.0
            results = []
            for idx, roi_rect in enumerate(rois):
                roi_dict = {
                    "x": roi_rect[0], "y": roi_rect[1],
                    "width": roi_rect[2], "height": roi_rect[3],
                    "uid": (rois_uid[idx] if idx < len(rois_uid) else None),
                    # WAJIB ada supaya infer() mask piksel sama persis dengan
                    # saat training (roi_mask.py). None = ROI tanpa mask.
                    "mask_polygon": (rois_mask_polygon[idx]
                                     if idx < len(rois_mask_polygon) else None),
                }
                result = self._engine.infer(frame, roi=roi_dict)
                margin = result.score - result.threshold
                results.append({
                    "roi": roi_rect,
                    "label": (rois_label[idx] if idx < len(rois_label)
                              else f"ROI{idx + 1}"),
                    "score": result.score,
                    "threshold": result.threshold,
                    "margin": margin,
                    "judgement": result.judgement,
                    "latency": result.latency_ms,
                })
                total_latency += result.latency_ms
                if worst_margin is None or margin < worst_margin:
                    worst_margin = margin
                    worst = result.score
                    res["worst_threshold"] = result.threshold
                    res["worst_label"] = results[-1]["label"]
                    res["heatmap"] = result.heatmap
                if result.judgement == "NG":
                    res["overall_ng"] = True

            res["roi_results"] = results
            res["worst_score"] = worst
            res["worst_margin"] = (worst_margin if worst_margin is not None
                                   else 0.0)
            res["avg_latency"] = (total_latency / len(results)
                                  if results else 0.0)
            res["raw_judgement"] = "NG" if res["overall_ng"] else "OK"
        except Exception as e:
            logger.warning("Inference worker error: %s", e)
            res["error"] = str(e)
        self.result_ready.emit(res)
