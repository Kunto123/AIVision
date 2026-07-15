"""VisionInspect - ROI Adjust Dialog
Popup dialog untuk melihat full image dan menyesuaikan ROI.
Muncul saat user klik thumbnail di galeri TEACH page.
Auto-save ROI saat close atau tekan tombol centang.
"""

from pathlib import Path
from typing import List, Optional

import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from visioninspect.gui.widgets.roi_editor import ROIEditor, ROIData


class ROIAdjustDialog(QDialog):
    """Modal dialog untuk menyesuaikan ROI pada gambar.

    Args:
        image_path: Path ke file gambar.
        current_rois: List dict ROI saat ini (dari template config).
            Setiap dict: {uid, x, y, width, height, enabled, label}.
        parent: Parent widget.
    """

    def __init__(
        self,
        image_path: str,
        current_rois: Optional[List[dict]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._image_path = image_path
        self._rois: List[ROIData] = []
        self._setup_ui()

        # Load image
        self._load_image(image_path)

        # Load ROIs
        if current_rois:
            rois = [ROIData.from_dict(d) for d in current_rois]
            self._roi_editor.set_rois(rois)
            self._rois = rois

        self.setWindowTitle(f"Atur ROI — {Path(image_path).name}")
        self.setModal(True)

    def _setup_ui(self):
        self.setMinimumSize(800, 600)
        self.resize(960, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title
        title = QLabel("🔄 Atur Region of Interest")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        layout.addWidget(title)

        # Instruction
        hint = QLabel(
            "Geser & resize kotak ROI. Klik kanan untuk aktif/nonaktifkan. "
            "Tekan Delete untuk hapus."
        )
        hint.setStyleSheet("color: #9FB3C8; font-size: 12px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # ROI Editor (takes remaining space)
        self._roi_editor = ROIEditor()
        layout.addWidget(self._roi_editor, 1)

        # Button bar
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(8)
        btn_bar.addStretch()

        self._save_btn = QPushButton("✅ Simpan")
        self._save_btn.setObjectName("primaryButton")
        self._save_btn.setMinimumHeight(36)
        self._save_btn.setMinimumWidth(120)
        self._save_btn.clicked.connect(self._on_save)
        btn_bar.addWidget(self._save_btn)

        self._close_btn = QPushButton("✕ Tutup")
        self._close_btn.setMinimumHeight(36)
        self._close_btn.setMinimumWidth(100)
        self._close_btn.setStyleSheet(
            "font-size: 13px; padding: 6px 16px; "
            "border: 1px solid #233A57; border-radius: 4px; "
            "background: #1A2A44; color: #9FB3C8;"
        )
        self._close_btn.clicked.connect(self.reject)
        btn_bar.addWidget(self._close_btn)

        layout.addLayout(btn_bar)

    def _load_image(self, image_path: str):
        """Load image from disk and display in ROI editor."""
        img = cv2.imread(str(image_path))
        if img is None:
            self._roi_editor.set_pixmap(QPixmap())
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._roi_editor.set_pixmap(pixmap)

    def _on_save(self):
        """Save button clicked — accept dialog."""
        self._rois = self._roi_editor.get_rois()
        self.accept()

    def get_rois(self) -> List[dict]:
        """Return adjusted ROIs as list of dicts (for serialization)."""
        return [r.to_dict() for r in self._rois]

    def reject(self):
        """Override: auto-save ROIs even on close (✕)."""
        self._rois = self._roi_editor.get_rois()
        super().reject()
