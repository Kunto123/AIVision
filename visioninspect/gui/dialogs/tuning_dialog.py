"""
VisionInspect - Tuning Mode Dialog
Mode koreksi per-ROI untuk hasil inspeksi yang salah.
User bisa memilih ROI tertentu, Register as OK/NG, lalu Additional Learning.
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QMessageBox,
)


# ── Tuning Data ────────────────────────────────────────────────────────

class TuningROI:
    """Data satu ROI dalam konteks tuning."""

    def __init__(self, roi_rect: tuple, score: float, judgement: str,
                 label: str = "", uid: str = ""):
        self.uid = uid or f"roi_{id(self)}"
        self.x, self.y, self.w, self.h = roi_rect
        self.score = score
        self.judgement = judgement          # "OK" / "NG" (asli dari inferensi)
        self.corrected_to: Optional[str] = None  # "OK" / "NG" setelah koreksi
        self.label = label or f"ROI"

    @property
    def current_judgement(self) -> str:
        """Judgement setelah koreksi (jika ada)."""
        return self.corrected_to or self.judgement

    @property
    def is_corrected(self) -> bool:
        return self.corrected_to is not None

    @property
    def color(self) -> str:
        j = self.current_judgement
        return "#22C55E" if j == "OK" else "#EF4444" if j == "NG" else "#9FB3C8"

    def rect(self) -> tuple:
        return (self.x, self.y, self.w, self.h)

    def to_dict(self) -> dict:
        return dict(x=self.x, y=self.y, width=self.w, height=self.h,
                    score=self.score, judgement=self.judgement,
                    corrected_to=self.corrected_to, label=self.label, uid=self.uid)


# ── Tuning Dialog ──────────────────────────────────────────────────────

class TuningDialog(QDialog):
    """
    Dialog tuning: lihat gambar hasil inspeksi dengan ROI berwarna,
    klik ROI untuk koreksi per-ROI (Register as OK / Register as NG).

    Returns:
        corrections: list of TuningROI.to_dict() for ROIs that were corrected.
    """

    def __init__(self, image_path: str, image: np.ndarray,
                 rois_data: List[dict], parent=None):
        """
        Args:
            image_path: Path ke file gambar (untuk referensi).
            image: Numpy array (BGR) — gambar full-frame.
            rois_data: List dict per-ROI dari history:
                [{x, y, width, height, score, judgement, label?}, ...]
            parent: Parent widget.
        """
        super().__init__(parent)
        self._image_path = image_path
        self._image = image
        self._rois: List[TuningROI] = []
        self._selected_idx: int = -1
        self._pixmap: Optional[QPixmap] = None

        self.setWindowTitle(f"🔧 Tuning — {Path(image_path).name}")
        self.setMinimumSize(960, 680)
        self.resize(1100, 780)
        self.setModal(True)

        self._init_rois(rois_data)
        self._build_pixmap()
        self._setup_ui()

    def _init_rois(self, rois_data: List[dict]):
        for i, r in enumerate(rois_data):
            rect = (r.get("x", 0), r.get("y", 0),
                    r.get("width", 64), r.get("height", 64))
            self._rois.append(TuningROI(
                roi_rect=rect,
                score=r.get("score", 0.0),
                judgement=r.get("judgement", "NG"),
                label=r.get("label", f"ROI {i+1}"),
                uid=r.get("uid", f"roi_{i}"),
            ))

    def _build_pixmap(self):
        """Convert image BGR → RGB → QPixmap."""
        if self._image is None or self._image.size == 0:
            self._pixmap = QPixmap()
            return
        rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title bar
        title_row = QHBoxLayout()
        title = QLabel(f"🔧 Tuning — {Path(self._image_path).name}")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        title_row.addWidget(title)
        title_row.addStretch()
        layout.addLayout(title_row)

        # Hint
        hint = QLabel(
            "Klik kotak ROI untuk memilih. "
            "Gunakan tombol Register as OK/NG untuk mengoreksi label per-ROI. "
            "Tekan 'Simpan & Additional Learning' untuk menyimpan perubahan dan memperbarui model."
        )
        hint.setStyleSheet("color: #9FB3C8; font-size: 12px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Main area: Image + Right panel
        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        # ── Left: Image with ROI overlays ──
        self._image_label = QLabel()
        self._image_label.setMinimumSize(540, 400)
        self._image_label.setStyleSheet(
            "background-color: #0A0F1A; border: 1px solid #233A57; border-radius: 4px;")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._update_image_display()
        main_row.addWidget(self._image_label, 3)

        # ── Right: ROI detail panel ──
        right_panel = QFrame()
        right_panel.setObjectName("cardPanel")
        right_panel.setMinimumWidth(260)
        right_panel.setMaximumWidth(320)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        right_layout.addWidget(QLabel("📍 Pilih ROI"))

        # ROI list
        self._roi_list_widget = QWidget()
        self._roi_list_layout = QVBoxLayout(self._roi_list_widget)
        self._roi_list_layout.setSpacing(4)
        self._roi_list_layout.setContentsMargins(0, 0, 0, 0)
        self._build_roi_list()

        roi_scroll = QScrollArea()
        roi_scroll.setWidgetResizable(True)
        roi_scroll.setWidget(self._roi_list_widget)
        roi_scroll.setFrameShape(QFrame.NoFrame)
        right_layout.addWidget(roi_scroll, 1)

        # Detail panel for selected ROI
        right_layout.addWidget(QLabel("📋 Detail"))
        self._detail_frame = QFrame()
        self._detail_frame.setObjectName("cardPanel")
        self._detail_layout = QVBoxLayout(self._detail_frame)
        self._detail_layout.setContentsMargins(8, 8, 8, 8)
        self._detail_layout.setSpacing(4)
        self._detail_label = QLabel("Klik ROI untuk melihat detail")
        self._detail_label.setStyleSheet("color: #9FB3C8;")
        self._detail_layout.addWidget(self._detail_label)
        right_layout.addWidget(self._detail_frame)

        # Register buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._register_ok_btn = QPushButton("✅ Register as OK")
        self._register_ok_btn.setObjectName("successButton")
        self._register_ok_btn.setMinimumHeight(36)
        self._register_ok_btn.setEnabled(False)
        self._register_ok_btn.clicked.connect(lambda: self._register("OK"))
        btn_row.addWidget(self._register_ok_btn)

        self._register_ng_btn = QPushButton("❌ Register as NG")
        self._register_ng_btn.setObjectName("dangerButton")
        self._register_ng_btn.setMinimumHeight(36)
        self._register_ng_btn.setEnabled(False)
        self._register_ng_btn.clicked.connect(lambda: self._register("NG"))
        btn_row.addWidget(self._register_ng_btn)
        right_layout.addLayout(btn_row)

        right_layout.addStretch()
        main_row.addWidget(right_panel, 1)
        layout.addLayout(main_row, 1)

        # Bottom bar: Save & Additional Learning
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(8)

        self._status_label = QLabel("💡 Pilih ROI untuk mulai koreksi")
        self._status_label.setStyleSheet("color: #F59E0B; font-weight: bold;")
        bottom_bar.addWidget(self._status_label, 1)

        self._save_btn = QPushButton("🧠 Simpan & Additional Learning")
        self._save_btn.setObjectName("primaryButton")
        self._save_btn.setMinimumHeight(40)
        self._save_btn.setMinimumWidth(220)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        bottom_bar.addWidget(self._save_btn)

        cancel_btn = QPushButton("✕ Batal")
        cancel_btn.setMinimumHeight(40)
        cancel_btn.setMinimumWidth(100)
        cancel_btn.setStyleSheet(
            "font-size: 13px; padding: 6px 16px; "
            "border: 1px solid #233A57; border-radius: 4px; "
            "background: #1A2A44; color: #9FB3C8;")
        cancel_btn.clicked.connect(self.reject)
        bottom_bar.addWidget(cancel_btn)

        layout.addLayout(bottom_bar)

    # ── ROI List ───────────────────────────────────────────────────

    def _build_roi_list(self):
        """Build clickable ROI buttons in the right panel."""
        # Clear existing
        while self._roi_list_layout.count():
            item = self._roi_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, roi in enumerate(self._rois):
            btn = QPushButton(
                f"{'✅' if roi.current_judgement == 'OK' else '❌'} "
                f"{roi.label}  ({roi.score:.3f})"
            )
            btn.setMinimumHeight(32)
            btn.setStyleSheet(
                f"text-align: left; padding: 4px 8px; font-size: 12px; "
                f"border: 2px solid {roi.color}; border-radius: 4px; "
                f"background: {'#1A2A44' if i != self._selected_idx else '#233A57'}; "
                f"color: #E2E8F0;"
            )
            btn.clicked.connect(lambda checked, idx=i: self._select_roi(idx))
            self._roi_list_layout.addWidget(btn)

    def _select_roi(self, idx: int):
        """Select ROI and update detail panel."""
        if idx < 0 or idx >= len(self._rois):
            return
        self._selected_idx = idx
        roi = self._rois[idx]

        # Enable register buttons
        self._register_ok_btn.setEnabled(True)
        self._register_ng_btn.setEnabled(True)

        # Update detail
        self._detail_label.setText(
            f"🆔 {roi.label}\n"
            f"📍 ({roi.x}, {roi.y}) {roi.w}×{roi.h}\n"
            f"📊 Score: {roi.score:.4f}\n"
            f"🏷 Asli: {'✅ OK' if roi.judgement == 'OK' else '❌ NG'}\n"
            f"{'✏️ Koreksi: → ' + ('✅ OK' if roi.corrected_to == 'OK' else '❌ NG') if roi.is_corrected else ''}"
        )

        self._update_image_display()
        self._build_roi_list()

    # ── Image Display ──────────────────────────────────────────────

    def _update_image_display(self):
        """Draw image with ROI overlays onto the label."""
        if self._pixmap is None or self._pixmap.isNull():
            self._image_label.setText("📷 Tidak ada gambar")
            return

        # Scale pixmap to label size
        label_size = self._image_label.size()
        scaled = self._pixmap.scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Paint ROIs onto a copy
        overlay = QPixmap(scaled)
        p = QPainter(overlay)
        p.setRenderHint(QPainter.Antialiasing)
        font = QFont("Segoe UI", 10, QFont.Bold)
        p.setFont(font)

        # Scale factors from original image → displayed
        sx = overlay.width() / self._pixmap.width()
        sy = overlay.height() / self._pixmap.height()

        for i, roi in enumerate(self._rois):
            is_selected = (i == self._selected_idx)
            color = QColor(roi.color)
            px = int(roi.x * sx)
            py = int(roi.y * sy)
            pw = max(1, int(roi.w * sx))
            ph = max(1, int(roi.h * sy))

            pen = QPen(color, 3 if is_selected else 2)
            if is_selected:
                pen.setStyle(Qt.SolidLine)
            else:
                pen.setStyle(Qt.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(px, py, pw, ph)

            # Label
            corrected_mark = ""
            if roi.is_corrected:
                corrected_mark = " ✓" if roi.corrected_to == "OK" else " ✗"
            label_text = f"{roi.label}{corrected_mark}"
            label_y = py - 6 if py >= 18 else py + ph + 14
            p.setPen(color)
            p.drawText(px + 4, label_y, label_text)

        p.end()
        self._image_label.setPixmap(overlay)

    # ── Register Actions ───────────────────────────────────────────

    def _register(self, new_judgement: str):
        """Register selected ROI as OK or NG."""
        if self._selected_idx < 0:
            return
        roi = self._rois[self._selected_idx]

        if roi.judgement == new_judgement and not roi.is_corrected:
            self._status_label.setText(f"ℹ️ {roi.label} sudah {new_judgement}, tidak perlu koreksi")
            return

        # Set correction
        roi.corrected_to = new_judgement
        self._status_label.setText(
            f"✏️ {roi.label}: {roi.judgement} → {new_judgement}")
        self._status_label.setStyleSheet(
            "color: #22C55E; font-weight: bold;")

        # Check if any corrections exist
        self._save_btn.setEnabled(self._has_corrections())

        # Refresh UI
        self._update_image_display()
        self._build_roi_list()
        self._select_roi(self._selected_idx)  # refresh detail

    def _has_corrections(self) -> bool:
        return any(r.is_corrected for r in self._rois)

    # ── Save ────────────────────────────────────────────────────────

    def _on_save(self):
        """Save corrections and accept dialog."""
        if not self._has_corrections():
            QMessageBox.information(self, "Tuning", "Tidak ada perubahan untuk disimpan.")
            return
        self.accept()

    # ── Public API ──────────────────────────────────────────────────

    def get_corrections(self) -> List[dict]:
        """Return list of corrected ROIs."""
        return [r.to_dict() for r in self._rois if r.is_corrected]

    def get_image(self) -> np.ndarray:
        """Return the original image (BGR)."""
        return self._image

    @property
    def image_path(self) -> str:
        return self._image_path
