"""
VisionInspect - Capture Review Dialog
Review per-ROI OK/NG saat capture pertama kali. Satu foto bisa punya kondisi
campuran antar ROI (mis. ROI1 OK, ROI2 NG) — label tidak boleh diterapkan
rata ke semua ROI hanya dari satu tombol Capture OK/NG, karena semua ROI
dalam satu template berbagi satu memory bank/model yang sama; crop yang
salah label bisa mengajari model bahwa pola cacat itu normal.
"""

from typing import List, Tuple

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from visioninspect.gui.widgets.roi_editor import ROIData

GREEN = "#22C55E"
RED = "#EF4444"


class _ROICropToggle(QFrame):
    """Satu crop ROI dengan border warna yang bisa di-toggle OK/NG dengan klik."""

    toggled_label = Signal()

    def __init__(self, roi: ROIData, crop_bgr: np.ndarray, label: str, parent=None):
        super().__init__(parent)
        self.roi = roi
        self.crop = crop_bgr
        self.label = label  # "ok" | "ng" — state saat ini, bisa di-toggle

        self.setFixedWidth(120)
        self.setCursor(Qt.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        img_label = QLabel()
        img_label.setFixedSize(100, 100)
        img_label.setAlignment(Qt.AlignCenter)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_label.setPixmap(pixmap)
        layout.addWidget(img_label, alignment=Qt.AlignCenter)

        name_label = QLabel(roi.label or "ROI")
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("color: #E2E8F0; font-size: 11px; background: transparent;")
        layout.addWidget(name_label)

        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_label)

        self._refresh_style()

    def mousePressEvent(self, event):
        self.label = "ng" if self.label == "ok" else "ok"
        self._refresh_style()
        self.toggled_label.emit()
        super().mousePressEvent(event)

    def _refresh_style(self):
        color = GREEN if self.label == "ok" else RED
        self.setStyleSheet(
            f"QFrame {{ background: #111D30; border: 3px solid {color}; "
            f"border-radius: 6px; }}")
        self._status_label.setText("✅ OK" if self.label == "ok" else "❌ NG")
        self._status_label.setStyleSheet(
            f"font-weight: bold; font-size: 12px; color: {color}; background: transparent;")


class CaptureReviewDialog(QDialog):
    """
    Review per-ROI sebelum foto disimpan sebagai data training. Muncul
    hanya kalau template punya 2+ ROI aktif (lihat main_window._on_capture)
    — dengan 0-1 ROI tidak ada ambiguitas untuk direview.
    """

    def __init__(self, frame: np.ndarray, rois: List[ROIData], default_label: str,
                 parent=None):
        super().__init__(parent)
        self._frame = frame
        self._toggles: List[_ROICropToggle] = []

        self.setWindowTitle("📋 Review Per-ROI")
        self.setModal(True)
        self.resize(720, 320)

        self._setup_ui(rois, default_label)

    def _crop_roi(self, roi: ROIData) -> np.ndarray:
        h_img, w_img = self._frame.shape[:2]
        x = max(0, min(roi.x, w_img - 1))
        y = max(0, min(roi.y, h_img - 1))
        w = max(1, min(roi.width, w_img - x))
        h = max(1, min(roi.height, h_img - y))
        return self._frame[y:y + h, x:x + w].copy()

    def _setup_ui(self, rois: List[ROIData], default_label: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("📋 Review Per-ROI")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
        layout.addWidget(title)

        hint = QLabel(
            "Semua ROI dianggap sesuai tombol yang kamu klik. Klik ROI yang "
            "kondisinya beda (mis. ada cacat cuma di satu ROI) untuk membalik "
            "labelnya sebelum disimpan."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9FB3C8; font-size: 12px;")
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumHeight(160)
        strip_widget = QWidget()
        strip_layout = QHBoxLayout(strip_widget)
        strip_layout.setSpacing(8)
        for roi in rois:
            if not roi.enabled:
                continue
            crop = self._crop_roi(roi)
            toggle = _ROICropToggle(roi, crop, default_label, self)
            self._toggles.append(toggle)
            strip_layout.addWidget(toggle)
        strip_layout.addStretch()
        scroll.setWidget(strip_widget)
        layout.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("✕ Batal")
        cancel_btn.setMinimumHeight(36)
        cancel_btn.setMinimumWidth(100)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("✅ Simpan Semua")
        save_btn.setObjectName("primaryButton")
        save_btn.setMinimumHeight(36)
        save_btn.setMinimumWidth(140)
        save_btn.clicked.connect(self.accept)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def get_labeled_crops(self) -> List[Tuple[ROIData, np.ndarray, str]]:
        """Return list of (roi, crop_bgr, label) — label sudah final setelah toggle."""
        return [(t.roi, t.crop, t.label) for t in self._toggles]
