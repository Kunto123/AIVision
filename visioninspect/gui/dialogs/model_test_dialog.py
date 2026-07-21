"""
VisionInspect - Model Test Dialog
Uji model terhadap batch foto statis dari disk (read-only sanity check,
tidak melalui Part Presence gate/RUN page, tidak disimpan ke riwayat).
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from visioninspect.gui.widgets.histogram_widget import HistogramWidget
from visioninspect.gui.widgets.thumbnail import ThumbnailWidget

AMBER = "#F59E0B"
GREEN = "#22C55E"
RED = "#EF4444"

GRID_COLUMNS = 6


def _border_color(photo_result: dict) -> str:
    if photo_result.get("unreadable") or photo_result.get("resolution_mismatch"):
        return AMBER
    return GREEN if photo_result.get("overall_judgement") == "OK" else RED


def _bgr_to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ── Per-photo detail (read-only) ────────────────────────────────────────

class PhotoDetailDialog(QDialog):
    """
    Tampilkan satu foto uji dengan overlay kotak ROI + breakdown per-ROI.
    Read-only — tidak ada tombol koreksi/simpan (beda dengan TuningDialog,
    yang punya alur Register-as-OK/NG + Additional Learning yang tidak
    relevan untuk sekadar sanity-check foto statis).
    """

    def __init__(self, photo_result: dict, parent=None):
        super().__init__(parent)
        self._result = photo_result
        self._selected_idx = -1
        self._pixmap: Optional[QPixmap] = None

        name = Path(photo_result["path"]).name
        self.setWindowTitle(f"🧪 Detail — {name}")
        self.setMinimumSize(880, 620)
        self.resize(1000, 700)
        self.setModal(True)

        self._build_pixmap()
        self._setup_ui()

    def _build_pixmap(self):
        image = self._result.get("image")
        if image is None or image.size == 0:
            self._pixmap = QPixmap()
            return
        self._pixmap = _bgr_to_pixmap(image)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel(f"🧪 {Path(self._result['path']).name}")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        layout.addWidget(title)

        if self._result.get("resolution_mismatch"):
            ref = self._result.get("reference_dims")
            actual = self._result.get("actual_dims")
            msg = (
                f"⚠ Resolusi foto ini ({actual[0]}×{actual[1]}px) berbeda dari "
                f"resolusi acuan template ({ref[0]}×{ref[1]}px). Posisi ROI mungkin "
                "tidak akurat pada foto ini — hasil di bawah tetap ditampilkan "
                "untuk referensi."
                if ref and actual else
                "⚠ Resolusi foto ini berbeda dari resolusi acuan template — "
                "posisi ROI mungkin tidak akurat."
            )
            banner = QLabel(msg)
            banner.setWordWrap(True)
            banner.setStyleSheet(
                f"background: #1A2A44; border: 1px solid {AMBER}; border-radius: 4px;"
                f" color: {AMBER}; padding: 6px;")
            layout.addWidget(banner)

        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        self._image_label = QLabel()
        self._image_label.setMinimumSize(540, 400)
        self._image_label.setStyleSheet(
            "background-color: #0A0F1A; border: 1px solid #233A57; border-radius: 4px;")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._update_image_display()
        main_row.addWidget(self._image_label, 3)

        right_panel = QFrame()
        right_panel.setObjectName("cardPanel")
        right_panel.setMinimumWidth(260)
        right_panel.setMaximumWidth(320)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        overall = self._result.get("overall_judgement")
        overall_label = QLabel(
            f"{'✅ OK' if overall == 'OK' else '❌ NG'}  (skor tertinggi: "
            f"{self._result.get('worst_score', 0.0):.3f})")
        overall_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; "
            f"color: {GREEN if overall == 'OK' else RED};")
        right_layout.addWidget(overall_label)

        right_layout.addWidget(QLabel("📍 Per-ROI"))

        roi_scroll = QScrollArea()
        roi_scroll.setWidgetResizable(True)
        roi_scroll.setFrameShape(QFrame.NoFrame)
        roi_list_widget = QWidget()
        roi_list_layout = QVBoxLayout(roi_list_widget)
        roi_list_layout.setSpacing(4)
        roi_list_layout.setContentsMargins(0, 0, 0, 0)

        for i, r in enumerate(self._result.get("roi_results", [])):
            color = GREEN if r["judgement"] == "OK" else RED
            btn = QPushButton(
                f"{'✅' if r['judgement'] == 'OK' else '❌'} "
                f"{r['label']}  ({r['score']:.3f})")
            btn.setMinimumHeight(32)
            btn.setStyleSheet(
                f"text-align: left; padding: 4px 8px; font-size: 12px; "
                f"border: 2px solid {color}; border-radius: 4px; "
                f"background: {'#233A57' if i == self._selected_idx else '#1A2A44'}; "
                f"color: #E2E8F0;")
            btn.clicked.connect(lambda checked=False, idx=i: self._select_roi(idx))
            roi_list_layout.addWidget(btn)

        roi_scroll.setWidget(roi_list_widget)
        right_layout.addWidget(roi_scroll, 1)

        right_layout.addWidget(QLabel("📋 Detail"))
        self._detail_label = QLabel("Klik ROI untuk melihat detail")
        self._detail_label.setStyleSheet("color: #9FB3C8;")
        self._detail_label.setWordWrap(True)
        right_layout.addWidget(self._detail_label)

        main_row.addWidget(right_panel, 1)
        layout.addLayout(main_row, 1)

        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        close_btn = QPushButton("✕ Tutup")
        close_btn.setMinimumHeight(36)
        close_btn.setMinimumWidth(100)
        close_btn.clicked.connect(self.reject)
        bottom_bar.addWidget(close_btn)
        layout.addLayout(bottom_bar)

    def _select_roi(self, idx: int):
        roi_results = self._result.get("roi_results", [])
        if idx < 0 or idx >= len(roi_results):
            return
        self._selected_idx = idx
        r = roi_results[idx]
        x, y, w, h = r["roi"]
        self._detail_label.setText(
            f"🆔 {r['label']}\n"
            f"📍 ({x}, {y}) {w}×{h}\n"
            f"📊 Score: {r['score']:.4f}\n"
            f"🏷 {'✅ OK' if r['judgement'] == 'OK' else '❌ NG'}\n"
            f"⏱ {r['latency_ms']:.1f} ms"
        )
        self._update_image_display()

    def _update_image_display(self):
        if self._pixmap is None or self._pixmap.isNull():
            self._image_label.setText("📷 Tidak ada gambar")
            return

        label_size = self._image_label.size()
        scaled = self._pixmap.scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        overlay = QPixmap(scaled)
        p = QPainter(overlay)
        p.setRenderHint(QPainter.Antialiasing)
        font = QFont("Segoe UI", 10, QFont.Bold)
        p.setFont(font)

        sx = overlay.width() / self._pixmap.width()
        sy = overlay.height() / self._pixmap.height()

        for i, r in enumerate(self._result.get("roi_results", [])):
            is_selected = (i == self._selected_idx)
            x, y, w, h = r["roi"]
            color = QColor(GREEN if r["judgement"] == "OK" else RED)
            px, py = int(x * sx), int(y * sy)
            pw, ph = max(1, int(w * sx)), max(1, int(h * sy))

            pen = QPen(color, 3 if is_selected else 2)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(px, py, pw, ph)

            label_y = py - 6 if py >= 18 else py + ph + 14
            p.setPen(color)
            p.drawText(px + 4, label_y, r["label"])

        p.end()
        self._image_label.setPixmap(overlay)


# ── Batch summary dialog ────────────────────────────────────────────────

class ModelTestDialog(QDialog):
    """
    Grid hasil uji model untuk satu batch foto statis + laporan agregat.
    Read-only — hasil tidak disimpan ke inspection_history/counter/PLC.
    """

    def __init__(self, template_label: str, results: List[dict], aggregate: dict,
                 threshold: float, parent=None):
        super().__init__(parent)
        self._template_label = template_label
        self._results = results
        self._aggregate = aggregate
        self._threshold = threshold
        self._path_to_result = {r["path"]: r for r in results}

        self.setWindowTitle(f"🧪 Uji Model — {template_label} ({len(results)} foto)")
        self.setMinimumSize(980, 680)
        self.resize(1100, 760)
        self.setModal(True)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel(f"🧪 Uji Model — {self._template_label}")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        layout.addWidget(title)

        hint = QLabel(
            "Klik salah satu foto untuk melihat detail per-ROI. Hasil ini bersifat "
            "sementara dan tidak disimpan ke riwayat inspeksi."
        )
        hint.setStyleSheet("color: #9FB3C8; font-size: 12px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        # ── Kiri: grid thumbnail ──
        grid_scroll = QScrollArea()
        grid_scroll.setWidgetResizable(True)
        grid_scroll.setFrameShape(QFrame.NoFrame)
        grid_scroll.setStyleSheet(
            "background: #111D30; border: 1px solid #233A57; border-radius: 4px;")

        grid_widget = QWidget()
        self._grid_layout = QGridLayout(grid_widget)
        self._grid_layout.setContentsMargins(8, 8, 8, 8)
        self._grid_layout.setSpacing(6)
        self._rebuild_grid()
        grid_scroll.setWidget(grid_widget)
        main_row.addWidget(grid_scroll, 2)

        # ── Kanan: ringkasan ──
        right_panel = QFrame()
        right_panel.setObjectName("cardPanel")
        right_panel.setMinimumWidth(280)
        right_panel.setMaximumWidth(340)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(6)

        right_layout.addWidget(QLabel("📊 Ringkasan"))
        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        right_layout.addWidget(self._summary_label)

        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet(f"color: {AMBER};")
        right_layout.addWidget(self._warning_label)

        self._histogram = HistogramWidget()
        self._histogram.setMinimumHeight(140)
        right_layout.addWidget(self._histogram)
        right_layout.addStretch()

        self._refresh_summary()

        main_row.addWidget(right_panel, 1)
        layout.addLayout(main_row, 1)

        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        close_btn = QPushButton("✕ Tutup")
        close_btn.setMinimumHeight(36)
        close_btn.setMinimumWidth(100)
        close_btn.clicked.connect(self.reject)
        bottom_bar.addWidget(close_btn)
        layout.addLayout(bottom_bar)

    # ---- Grid ----

    def _rebuild_grid(self):
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for idx, result in enumerate(self._results):
            if result.get("unreadable") or result.get("image") is None:
                pixmap = QPixmap()
            else:
                pixmap = _bgr_to_pixmap(result["image"])
            thumb = ThumbnailWidget(pixmap, result["path"], _border_color(result))
            thumb.clicked.connect(self._on_thumbnail_clicked)
            thumb.deleted.connect(self._on_thumbnail_deleted)
            row, col = divmod(idx, GRID_COLUMNS)
            self._grid_layout.addWidget(thumb, row, col)

    def _on_thumbnail_clicked(self, path: str):
        result = self._path_to_result.get(path)
        if result is None:
            return
        if result.get("unreadable"):
            QMessageBox.warning(self, "Uji Model", f"File gagal dibaca:\n{path}")
            return
        dialog = PhotoDetailDialog(result, self)
        dialog.exec()

    def _on_thumbnail_deleted(self, path: str):
        self._results = [r for r in self._results if r["path"] != path]
        self._path_to_result.pop(path, None)
        self._recompute_aggregate()
        self._rebuild_grid()
        self._refresh_summary()

    # ---- Aggregate ----

    def _recompute_aggregate(self):
        valid = [r for r in self._results if not r.get("unreadable")]
        ok_count = sum(1 for r in valid if r.get("overall_judgement") == "OK")
        ng_count = len(valid) - ok_count
        self._aggregate = {
            "total": len(valid),
            "ok_count": ok_count,
            "ng_count": ng_count,
            "pass_rate": (ok_count / len(valid) * 100.0) if valid else 0.0,
            "unreadable_count": sum(1 for r in self._results if r.get("unreadable")),
            "mismatch_count": sum(1 for r in valid if r.get("resolution_mismatch")),
        }

    def _refresh_summary(self):
        agg = self._aggregate
        self._summary_label.setText(
            f"Total foto: {agg['total']}\n"
            f"✅ OK: {agg['ok_count']}\n"
            f"❌ NG: {agg['ng_count']}\n"
            f"Pass rate: {agg['pass_rate']:.1f}%"
        )
        warnings = []
        if agg.get("mismatch_count"):
            warnings.append(f"⚠ {agg['mismatch_count']} foto resolusi tidak cocok")
        if agg.get("unreadable_count"):
            warnings.append(f"⚠ {agg['unreadable_count']} foto gagal dibaca")
        self._warning_label.setText("\n".join(warnings))

        valid = [r for r in self._results if not r.get("unreadable")]
        ok_scores = [r["worst_score"] for r in valid if r["overall_judgement"] == "OK"]
        ng_scores = [r["worst_score"] for r in valid if r["overall_judgement"] == "NG"]
        self._histogram.set_data(ok_scores, ng_scores, self._threshold)
