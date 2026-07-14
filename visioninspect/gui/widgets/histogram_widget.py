"""
VisionInspect - Histogram Widget
Custom QPainter widget untuk menampilkan distribusi skor anomaly setelah training.
Ringan, tanpa matplotlib. Dua warna: OK (hijau) dan NG (merah).
"""

import math
from typing import List, Optional

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QLinearGradient
from PySide6.QtWidgets import QWidget


class HistogramWidget(QWidget):
    """
    Widget histogram distribusi skor anomaly.
    - OK scores: bar hijau
    - NG scores: bar merah  
    - Garis threshold vertikal putih
    """

    # Colors
    COLOR_OK = QColor("#22C55E")
    COLOR_NG = QColor("#EF4444")
    COLOR_BG = QColor("#0A0F1A")
    COLOR_GRID = QColor("#1A2A40")
    COLOR_TEXT = QColor("#9FB3C8")
    COLOR_THRESHOLD = QColor("#FFFFFF")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ok_scores: List[float] = []
        self._ng_scores: List[float] = []
        self._threshold: float = 0.5
        self._bins = 20
        self.setMinimumHeight(120)

    def set_data(self, ok_scores: List[float], ng_scores: List[float],
                 threshold: float = 0.5):
        """Set data and redraw."""
        self._ok_scores = ok_scores or []
        self._ng_scores = ng_scores or []
        self._threshold = threshold
        self.update()

    def clear_data(self):
        """Clear all data."""
        self._ok_scores = []
        self._ng_scores = []
        self.update()

    def has_data(self) -> bool:
        return len(self._ok_scores) > 0 or len(self._ng_scores) > 0

    def paintEvent(self, event):
        """Draw the histogram."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, self.COLOR_BG)

        if not self.has_data():
            painter.setPen(self.COLOR_TEXT)
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Belum ada data\nTraining dulu untuk melihat histogram")
            painter.end()
            return

        # Margins
        margin_left = 50
        margin_right = 16
        margin_top = 20
        margin_bottom = 32
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        if plot_w < 20 or plot_h < 20:
            painter.end()
            return

        # Calculate bins
        all_scores = self._ok_scores + self._ng_scores
        if not all_scores:
            painter.end()
            return

        max_score = max(1.0, max(all_scores))
        min_score = 0.0
        bin_width = (max_score - min_score) / self._bins

        # Count per bin
        ok_bins = [0] * self._bins
        ng_bins = [0] * self._bins

        for s in self._ok_scores:
            idx = min(int(s / bin_width) if bin_width > 0 else 0, self._bins - 1)
            ok_bins[idx] += 1

        for s in self._ng_scores:
            idx = min(int(s / bin_width) if bin_width > 0 else 0, self._bins - 1)
            ng_bins[idx] += 1

        max_count = max(max(ok_bins), max(ng_bins), 1)

        # Grid lines (horizontal)
        painter.setPen(QPen(self.COLOR_GRID, 1))
        num_grid = 4
        for i in range(num_grid + 1):
            y = margin_top + plot_h - (plot_h * i / num_grid)
            painter.drawLine(margin_left, int(y), margin_left + plot_w, int(y))

            # Y-axis labels
            count_val = int(max_count * i / num_grid)
            painter.setPen(self.COLOR_TEXT)
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(QRectF(0, y - 8, margin_left - 4, 16),
                             Qt.AlignRight | Qt.AlignVCenter, str(count_val))
            painter.setPen(QPen(self.COLOR_GRID, 1))

        # X-axis labels (score values)
        painter.setPen(self.COLOR_TEXT)
        painter.setFont(QFont("Segoe UI", 8))
        num_x_labels = 5
        for i in range(num_x_labels + 1):
            score_val = min_score + (max_score * i / num_x_labels)
            x = margin_left + (plot_w * i / num_x_labels)
            painter.drawText(QRectF(x - 20, h - margin_bottom + 4, 40, 16),
                             Qt.AlignCenter, f"{score_val:.1f}")

        # Draw OK bars (green)
        bar_w = plot_w / self._bins
        painter.setBrush(QBrush(self.COLOR_OK))
        painter.setPen(Qt.NoPen)
        for i in range(self._bins):
            bar_h = (ok_bins[i] / max_count) * plot_h
            if bar_h > 0:
                x = margin_left + i * bar_w + 1
                y = margin_top + plot_h - bar_h
                painter.drawRect(QRectF(x, y, bar_w - 2, bar_h))

        # Draw NG bars (red, slightly narrower on top)
        painter.setBrush(QBrush(self.COLOR_NG))
        for i in range(self._bins):
            bar_h = (ng_bins[i] / max_count) * plot_h
            if bar_h > 0:
                x = margin_left + i * bar_w + 1
                y = margin_top + plot_h - bar_h
                painter.drawRect(QRectF(x, y, bar_w - 2, bar_h))

        # Threshold line
        thresh_x = margin_left + (self._threshold / max_score) * plot_w
        pen = QPen(self.COLOR_THRESHOLD, 2, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(thresh_x), margin_top,
                         int(thresh_x), margin_top + plot_h)

        # Threshold label
        painter.setPen(self.COLOR_THRESHOLD)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(QRectF(thresh_x - 30, margin_top - 16, 60, 14),
                         Qt.AlignCenter, f"Threshold: {self._threshold:.2f}")

        # Legend
        legend_y = margin_top
        legend_x = margin_left + plot_w - 120

        # OK legend
        painter.fillRect(legend_x, legend_y, 12, 12, self.COLOR_OK)
        painter.setPen(self.COLOR_TEXT)
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(legend_x + 16, legend_y + 10, f"OK ({len(self._ok_scores)})")

        # NG legend
        painter.fillRect(legend_x, legend_y + 18, 12, 12, self.COLOR_NG)
        painter.drawText(legend_x + 16, legend_y + 28, f"NG ({len(self._ng_scores)})")

        # Axis labels
        painter.setPen(self.COLOR_TEXT)
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(0, 0, 20, h), Qt.AlignCenter, "Count")
        painter.drawText(QRectF(0, h - 14, w, 14), Qt.AlignCenter, "Score")

        painter.end()
