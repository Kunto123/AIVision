"""
VisionInspect - Polygon Mask Dialog
Dialog kecil buat gambar/adjust satu polygon mask di atas SATU crop
(bukan template-wide seperti ROIAdjustDialog). Dipakai dua tempat:

  - CaptureReviewDialog (_ROICropToggle) — adjust polygon khusus foto yang
    baru saja di-capture, SEBELUM disimpan (mask langsung dibakar ke file).
  - Galeri TEACH (thumbnail double-click) — adjust polygon khusus satu foto
    yang SUDAH tersimpan, disimpan sebagai override di config template.

Reuse penuh ROIEditor: ROI sintetis satu-satunya menutupi seluruh crop
(x=0,y=0), rectangle-nya dikunci (set_rect_locked) supaya operator tidak
tidak sengaja men-drag/resize-nya — cuma polygon yang bisa diubah.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from visioninspect.gui.widgets.roi_editor import ROIEditor, ROIData

_MIN_DISPLAY = 320  # crop kecil (mis. 100x100) diperbesar dulu biar bisa digambar presisi


class PolygonMaskDialog(QDialog):
    """Gambar/adjust polygon mask untuk SATU crop. `result_polygon()`
    mengembalikan polygon (koordinat crop asli, bukan koordinat tampilan
    yang diperbesar) atau None kalau tidak ada mask."""

    def __init__(self, crop_bgr: np.ndarray,
                 initial_polygon: Optional[List[Tuple[int, int]]],
                 title: str = "Adjust Mask", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

        h, w = crop_bgr.shape[:2]
        self._synthetic_roi = ROIData(0, 0, w, h)
        self._synthetic_roi.label = ""
        if initial_polygon:
            self._synthetic_roi.mask_polygon = list(initial_polygon)

        # Perbesar tampilan supaya crop kecil (mis. 100x100 thumbnail review)
        # tetap bisa digambar presisi — ROIEditor sendiri yang menangani
        # scaling tampilan vs koordinat gambar asli (_map_to_image), jadi
        # cukup atur ukuran widget-nya lebih besar dari pixmap aslinya.
        scale = max(1, int(_MIN_DISPLAY / max(w, h))) if max(w, h) else 1
        disp_w, disp_h = w * scale, h * scale

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        hint = QLabel(
            "Klik untuk tambah titik polygon, double-click atau Enter untuk "
            "menutup, Backspace batal titik terakhir, Esc batal semua.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9FB3C8; font-size: 12px;")
        layout.addWidget(hint)

        self._editor = ROIEditor(self)
        self._editor.setFixedSize(max(disp_w, _MIN_DISPLAY), max(disp_h, _MIN_DISPLAY))
        self._editor.set_rect_locked(True)
        self._editor.set_max_rois(1)

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        self._editor.set_pixmap(QPixmap.fromImage(qimg))
        self._editor.set_rois([self._synthetic_roi])
        layout.addWidget(self._editor)

        btn_row = QHBoxLayout()
        self._draw_btn = QPushButton()
        self._draw_btn.clicked.connect(self._on_draw_clicked)
        btn_row.addWidget(self._draw_btn)

        self._clear_btn = QPushButton("Hapus Mask")
        self._clear_btn.setObjectName("dangerButton")
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        btn_row.addWidget(self._clear_btn)
        btn_row.addStretch()

        cancel_btn = QPushButton("Batal")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        ok_btn = QPushButton("Simpan")
        ok_btn.setObjectName("primaryButton")
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self._refresh_buttons()
        self._editor.rois_changed.connect(self._refresh_buttons)

    def _refresh_buttons(self):
        has_mask = bool(self._synthetic_roi.mask_polygon)
        self._draw_btn.setText("Gambar Ulang" if has_mask else "Gambar Mask")
        self._clear_btn.setEnabled(has_mask)

    def _on_draw_clicked(self):
        self._editor.start_drawing_mask(self._synthetic_roi)

    def _on_clear_clicked(self):
        self._editor.clear_mask_polygon(self._synthetic_roi)

    def result_polygon(self) -> Optional[List[Tuple[int, int]]]:
        """Polygon final, koordinat crop ASLI (bukan tampilan diperbesar).
        None = tidak ada mask (dihapus atau memang tidak pernah digambar)."""
        return self._synthetic_roi.mask_polygon
