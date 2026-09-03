"""
VisionInspect - Capture Review Dialog
Review per-ROI OK/NG saat capture pertama kali. Satu foto bisa punya kondisi
campuran antar ROI (mis. ROI1 OK, ROI2 NG) — label tidak boleh diterapkan
rata ke semua ROI hanya dari satu tombol Capture OK/NG, karena semua ROI
dalam satu template berbagi satu memory bank/model yang sama; crop yang
salah label bisa mengajari model bahwa pola cacat itu normal.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
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
from visioninspect.gui.dialogs.polygon_mask_dialog import PolygonMaskDialog

GREEN = "#22C55E"
RED = "#EF4444"
GREY = "#64748B"


class _ROICropToggle(QFrame):
    """Satu crop ROI: klik badan = toggle OK/NG, tombol ✕ = buang crop ini.

    "Buang" TIDAK menghapus file apa pun — pada tahap ini belum ada yang
    tersimpan. Yang dibuang hanya dikeluarkan dari daftar yang akan ditulis
    ke dataset, dan bisa dikembalikan lagi selama dialog masih terbuka.
    """

    toggled_label = Signal()

    def __init__(self, roi: ROIData, crop_bgr: np.ndarray, label: str, parent=None):
        super().__init__(parent)
        self.roi = roi
        self.crop = crop_bgr
        self.label = label      # "ok" | "ng" — state saat ini, bisa di-toggle
        self.excluded = False   # True = tidak ikut disimpan ke dataset
        # None = pakai mask default milik ROI (roi.mask_polygon) apa
        # adanya. Diisi HANYA kalau operator sengaja adjust polygon khusus
        # foto ini (outlier — part kebetulan geser lebih dari biasanya).
        self.mask_polygon_override: Optional[List[Tuple[int, int]]] = None

        self.setFixedWidth(120)
        self.setCursor(Qt.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Baris atas: nama ROI + tombol buang
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(2)
        name_label = QLabel(roi.label or "ROI")
        name_label.setStyleSheet(
            "color: #E2E8F0; font-size: 11px; background: transparent;")
        head.addWidget(name_label, 1)
        self._mask_btn = QPushButton("⬠")
        self._mask_btn.setFixedSize(20, 20)
        self._mask_btn.setCursor(Qt.PointingHandCursor)
        self._mask_btn.setToolTip(
            "Adjust mask polygon KHUSUS foto ini — cuma perlu kalau part\n"
            "kebetulan geser lebih dari biasanya di foto ini. Kosongkan\n"
            "(default) untuk mayoritas foto lain.")
        self._mask_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; "
            "color: #FBBF24; font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { color: #FFFFFF; }")
        self._mask_btn.clicked.connect(self._adjust_mask)
        head.addWidget(self._mask_btn, 0)
        self._drop_btn = QPushButton("✕")
        self._drop_btn.setFixedSize(20, 20)
        self._drop_btn.setCursor(Qt.PointingHandCursor)
        self._drop_btn.setToolTip(
            "Buang crop ini — tidak ikut disimpan ke dataset.\n"
            "Klik lagi untuk mengembalikan.")
        self._drop_btn.clicked.connect(self._toggle_excluded)
        head.addWidget(self._drop_btn, 0)
        layout.addLayout(head)

        self._img_label = QLabel()
        self._img_label.setFixedSize(100, 100)
        self._img_label.setAlignment(Qt.AlignCenter)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        self._pix_normal = QPixmap.fromImage(qimg).scaled(
            96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Versi redup untuk keadaan "dibuang" — supaya sekali lihat langsung
        # ketahuan mana yang tidak ikut, tanpa harus membaca teks.
        faded = QPixmap(self._pix_normal.size())
        faded.fill(Qt.transparent)
        _p = QPainter(faded)
        _p.setOpacity(0.25)
        _p.drawPixmap(0, 0, self._pix_normal)
        _p.end()
        self._pix_faded = faded
        self._img_label.setPixmap(self._pix_normal)
        layout.addWidget(self._img_label, alignment=Qt.AlignCenter)

        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_label)

        self._refresh_style()

    def _adjust_mask(self):
        """Buka dialog kecil buat adjust polygon mask khusus crop ini —
        seed dari override yang sudah ada (kalau sudah pernah di-adjust),
        atau dari mask default milik ROI."""
        seed = (self.mask_polygon_override if self.mask_polygon_override
                else getattr(self.roi, "mask_polygon", None))
        dlg = PolygonMaskDialog(self.crop, seed,
                                f"Adjust Mask — {self.roi.label or 'ROI'}", self)
        if dlg.exec():
            self.mask_polygon_override = dlg.result_polygon()
            self._refresh_style()

    def _toggle_excluded(self):
        self.excluded = not self.excluded
        self._refresh_style()
        self.toggled_label.emit()

    def mousePressEvent(self, event):
        # Crop yang sudah dibuang tidak ikut di-toggle OK/NG — labelnya tidak
        # bermakna lagi. Klik badan mengembalikannya dulu.
        if self.excluded:
            self._toggle_excluded()
        else:
            self.label = "ng" if self.label == "ok" else "ok"
            self._refresh_style()
            self.toggled_label.emit()
        super().mousePressEvent(event)

    def _refresh_style(self):
        if self.excluded:
            color, text, style = GREY, "DIBUANG", "dashed"
        else:
            color = GREEN if self.label == "ok" else RED
            text, style = ("OK" if self.label == "ok" else "NG"), "solid"
        self.setStyleSheet(
            f"QFrame {{ background: #111D30; border: 3px {style} {color}; "
            f"border-radius: 6px; }}")
        self._img_label.setPixmap(
            self._pix_faded if self.excluded else self._pix_normal)
        self._drop_btn.setText("↺" if self.excluded else "✕")
        self._drop_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"color: {color}; font-weight: bold; font-size: 13px; }}"
            f"QPushButton:hover {{ color: #FFFFFF; }}")
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"font-weight: bold; font-size: 12px; color: {color}; "
            f"background: transparent;")
        # Isi penuh (bukan outline) = foto ini punya mask khusus sendiri,
        # beda dari default ROI — penanda visual supaya operator tahu
        # tanpa perlu buka dialog lagi.
        mask_color = "#FBBF24" if self.mask_polygon_override else "#94A3B8"
        self._mask_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"color: {mask_color}; font-weight: bold; font-size: 13px; }}"
            f"QPushButton:hover {{ color: #FFFFFF; }}")


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

        self.setWindowTitle("Review Per-ROI")
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

        title = QLabel("Review Per-ROI")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
        layout.addWidget(title)

        hint = QLabel(
            "Klik gambar untuk membalik label OK ⇄ NG. Klik tombol ✕ di pojok "
            "untuk membuang crop itu — crop yang dibuang TIDAK masuk dataset "
            "(klik ↺ untuk mengembalikan). Tidak ada file yang dihapus; belum "
            "ada yang tersimpan sampai kamu menekan tombol di bawah."
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
            toggle.toggled_label.connect(self._refresh_summary)
            self._toggles.append(toggle)
            strip_layout.addWidget(toggle)
        strip_layout.addStretch()
        scroll.setWidget(strip_widget)
        layout.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        self._summary = QLabel()
        self._summary.setStyleSheet("color: #9FB3C8; font-size: 12px;")
        btn_row.addWidget(self._summary)
        btn_row.addStretch()
        cancel_btn = QPushButton("✕ Batal")
        cancel_btn.setMinimumHeight(36)
        cancel_btn.setMinimumWidth(100)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._save_btn = QPushButton("Simpan Semua")
        self._save_btn.setObjectName("primaryButton")
        self._save_btn.setMinimumHeight(36)
        self._save_btn.setMinimumWidth(150)
        self._save_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._save_btn)
        layout.addLayout(btn_row)
        self._refresh_summary()

    def _refresh_summary(self):
        """Tampilkan apa yang AKAN terjadi, bukan sekadar jumlah kotak."""
        kept = [t for t in self._toggles if not t.excluded]
        n_ok = sum(1 for t in kept if t.label == "ok")
        n_ng = len(kept) - n_ok
        n_drop = len(self._toggles) - len(kept)
        parts = [f"Disimpan: {len(kept)} ({n_ok} OK, {n_ng} NG)"]
        if n_drop:
            parts.append(f"dibuang: {n_drop}")
        self._summary.setText("  ·  ".join(parts))
        # Kalau semua dibuang, tombolnya harus jujur: tidak ada yang disimpan,
        # gambar ini dilewati.
        if not kept:
            self._save_btn.setText("Lewati gambar ini")
            self._summary.setText(
                "Semua crop dibuang — tidak ada yang masuk dataset")
        else:
            self._save_btn.setText(f"Simpan {len(kept)} crop")

    def get_labeled_crops(self) -> List[Tuple[ROIData, np.ndarray, str, Optional[list]]]:
        """(roi, crop_bgr, label, mask_polygon) untuk crop yang AKAN disimpan.

        Crop yang ditandai dibuang tidak ikut — inilah satu-satunya sumber
        kebenaran yang dipakai pemanggil untuk menulis ke dataset.

        `mask_polygon` = override khusus foto ini kalau operator sempat
        adjust (lihat _adjust_mask), else mask default milik ROI, else
        None. Pemanggil (main_window._maybe_review_and_save_per_roi) yang
        menerapkannya via apply_polygon_mask SEBELUM menyimpan — supaya
        hasil di disk sudah ter-mask (folder *_per_roi tidak di-crop ulang
        saat training, lihat training_worker.py).
        """
        return [(t.roi, t.crop, t.label,
                 t.mask_polygon_override or getattr(t.roi, "mask_polygon", None))
                for t in self._toggles if not t.excluded]

    def get_dropped_count(self) -> int:
        """Jumlah crop yang sengaja dibuang operator (untuk status/log)."""
        return sum(1 for t in self._toggles if t.excluded)
