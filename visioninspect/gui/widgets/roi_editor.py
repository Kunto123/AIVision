"""
VisionInspect - Multi-ROI Editor Widget
Widget untuk menggambar banyak Region of Interest (ROI) di atas live preview.
Fitur: add, select, resize, move, delete, toggle, zoom (Ctrl+±0), pan (drag+scroll).
"""

import math
from typing import List, Optional
from uuid import uuid4

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QPixmap, QMouseEvent, QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QWidget


ROI_MIN_SIZE = 32
PAN_CLICK_THRESHOLD = 5  # px — batas antara klik vs drag untuk pan


class ROIData:
    """Data satu ROI."""

    def __init__(self, x: int = 0, y: int = 0, width: int = 256, height: int = 256,
                 enabled: bool = True, label: str = ""):
        self.uid = uuid4().hex[:8]
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.enabled = enabled
        self.label = label or f"ROI"

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "enabled": self.enabled,
            "label": self.label,
        }

    @staticmethod
    def from_dict(d: dict) -> "ROIData":
        roi = ROIData(
            x=d.get("x", 0), y=d.get("y", 0),
            width=d.get("width", 256), height=d.get("height", 256),
            enabled=d.get("enabled", True),
            label=d.get("label", "ROI"),
        )
        roi.uid = d.get("uid", roi.uid)
        return roi

    def rect(self) -> tuple:
        return (self.x, self.y, self.width, self.height)


class ROIEditor(QWidget):
    """
    Editor multi-ROI. Drag untuk move/resize rectangle.
    Click untuk select. Delete untuk hapus. Toggle enabled/disabled.
    Zoom: Ctrl++ / Ctrl+- / Ctrl+0.  Pan: drag di area kosong + scroll wheel.
    """

    rois_changed = Signal()  # emitted when ROIs are modified

    # Colors
    COLORS = [
        ("#FFFFFF", "#22C55E"),  # putih + hijau
        ("#FFD700", "#FFA500"),  # emas + orange
        ("#00BFFF", "#1E90FF"),  # cyan + biru
        ("#FF69B4", "#FF1493"),  # pink + hotpink
        ("#7B68EE", "#4B0082"),  # mediumslate + indigo
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._rois: List[ROIData] = []
        self._selected_idx: int = -1
        self._max_rois: Optional[int] = None  # None = unlimited

        # Drag state
        self._dragging = False
        self._resizing = False
        self._resize_handle = -1
        self._drag_start_img = None
        self._roi_drag_start = None

        # Image geometry (recomputed each paintEvent)
        self._img_rect = QRectF()
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._offset_x = 0
        self._offset_y = 0

        # — Zoom + Pan —
        self._zoom = 1.0
        self._pan_dx = 0.0   # pan offset in widget pixels (applied on top of centered image)
        self._pan_dy = 0.0
        self._panning = False
        self._pan_start_pos = None   # QPointF — mouse press position for pan
        self._pan_start_dx = 0.0
        self._pan_start_dy = 0.0
        self._click_start = None     # QPointF — for click-vs-drag detection

        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ---- Public API ----

    def set_pixmap(self, pixmap: QPixmap):
        # Only reset zoom/pan when image dimensions actually change
        # (camera frames at same resolution keep current zoom/pan)
        old_w = self._pixmap.width() if self._pixmap and not self._pixmap.isNull() else 0
        old_h = self._pixmap.height() if self._pixmap and not self._pixmap.isNull() else 0
        new_w = pixmap.width() if pixmap and not pixmap.isNull() else 0
        new_h = pixmap.height() if pixmap and not pixmap.isNull() else 0
        self._pixmap = pixmap
        if old_w != new_w or old_h != new_h:
            self._reset_view()
        self.update()

    def set_rois(self, rois: List[ROIData]):
        """Set ROIs from list."""
        self._rois = list(rois)
        if self._selected_idx >= len(self._rois):
            self._selected_idx = len(self._rois) - 1
        self.update()

    def set_max_rois(self, n: Optional[int]) -> None:
        """Limit number of ROIs. None = unlimited.
        When max is reached, clicking empty replaces existing ROI."""
        self._max_rois = n

    def get_rois(self) -> List[ROIData]:
        return list(self._rois)

    def add_roi(self, x: int = 100, y: int = 100, w: int = 256, h: int = 256) -> ROIData:
        """Add a new ROI and select it."""
        roi = ROIData(x, y, w, h)
        roi.label = f"ROI {len(self._rois) + 1}"
        self._rois.append(roi)
        self._selected_idx = len(self._rois) - 1
        self.rois_changed.emit()
        self.update()
        return roi

    def delete_selected_roi(self):
        """Delete the currently selected ROI."""
        if 0 <= self._selected_idx < len(self._rois):
            self._rois.pop(self._selected_idx)
            self._selected_idx = min(self._selected_idx, len(self._rois) - 1)
            self.rois_changed.emit()
            self.update()

    def toggle_selected_roi(self):
        """Toggle enabled/disabled for selected ROI."""
        if 0 <= self._selected_idx < len(self._rois):
            self._rois[self._selected_idx].enabled = not self._rois[self._selected_idx].enabled
            self.rois_changed.emit()
            self.update()

    def select_roi(self, index: int):
        if 0 <= index < len(self._rois):
            self._selected_idx = index
            self.update()

    @property
    def selected_roi(self) -> Optional[ROIData]:
        if 0 <= self._selected_idx < len(self._rois):
            return self._rois[self._selected_idx]
        return None

    def clear_all(self):
        self._rois.clear()
        self._selected_idx = -1
        self.rois_changed.emit()
        self.update()

    def _reset_view(self):
        """Reset zoom and pan to default (called when image changes)."""
        self._zoom = 1.0
        self._pan_dx = 0.0
        self._pan_dy = 0.0

    # ---- Mapping (image <-> widget coords) ----

    def _map_to_image(self, pos) -> tuple:
        """Map widget position to image pixel coordinates (accounting for zoom+pan)."""
        if self._pixmap is None or self._pixmap.isNull():
            return (0, 0)
        x = int((pos.x() - self._offset_x) / self._scale_x)
        y = int((pos.y() - self._offset_y) / self._scale_y)
        return (x, y)

    def _roi_to_widget(self, roi: ROIData) -> QRectF:
        """Map ROI image coordinates to widget rectangle (accounting for zoom+pan)."""
        rx = roi.x * self._scale_x + self._offset_x
        ry = roi.y * self._scale_y + self._offset_y
        rw = roi.width * self._scale_x
        rh = roi.height * self._scale_y
        return QRectF(rx, ry, rw, rh)

    def _get_handles(self, roi: ROIData) -> List[QRectF]:
        r = self._roi_to_widget(roi)
        hs = 8
        return [
            QRectF(r.x() - hs, r.y() - hs, hs * 2, hs * 2),       # TL
            QRectF(r.x() + r.width() - hs, r.y() - hs, hs * 2, hs * 2),  # TR
            QRectF(r.x() - hs, r.y() + r.height() - hs, hs * 2, hs * 2), # BL
            QRectF(r.x() + r.width() - hs, r.y() + r.height() - hs, hs * 2, hs * 2), # BR
        ]

    def _roi_color(self, idx: int) -> tuple:
        c = self.COLORS[idx % len(self.COLORS)]
        return (QColor(c[0]), QColor(c[1]))

    def _hit_test_handles(self, pos) -> tuple:
        """Returns (roi_idx, handle_idx) or (-1, -1)."""
        p = pos
        for ri, roi in enumerate(self._rois):
            for hi, h in enumerate(self._get_handles(roi)):
                if h.contains(p):
                    return (ri, hi)
        return (-1, -1)

    def _hit_test_roi(self, pos) -> int:
        """Returns index of ROI at position, or -1."""
        for ri in range(len(self._rois) - 1, -1, -1):  # reverse for topmost
            r = self._roi_to_widget(self._rois[ri])
            if r.contains(pos):
                return ri
        return -1

    def _clamp_pan(self):
        """Clamp pan offset so image doesn't go too far off-screen."""
        if not self._pixmap or self._pixmap.isNull():
            self._pan_dx = 0.0
            self._pan_dy = 0.0
            return
        display_w = self._compute_display_size()[0]
        display_h = self._compute_display_size()[1]
        margin_x = max(display_w * 0.5, 50)
        margin_y = max(display_h * 0.5, 50)
        self._pan_dx = max(-margin_x, min(margin_x, self._pan_dx))
        self._pan_dy = max(-margin_y, min(margin_y, self._pan_dy))

    def _compute_display_size(self) -> tuple:
        """Return (display_width, display_height) in widget pixels at current zoom."""
        if not self._pixmap or self._pixmap.isNull():
            return (0, 0)
        base_w = self._pixmap.width()
        base_h = self._pixmap.height()
        fit = min(self.width() / base_w, self.height() / base_h)
        return (int(base_w * fit * self._zoom), int(base_h * fit * self._zoom))

    # ---- Events ----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Draw image with zoom + pan support
        if self._pixmap and not self._pixmap.isNull():
            base_w = self._pixmap.width()
            base_h = self._pixmap.height()
            fit_scale = min(w / base_w, h / base_h)
            display_w = int(base_w * fit_scale * self._zoom)
            display_h = int(base_h * fit_scale * self._zoom)

            # Center image, then apply pan offset
            self._clamp_pan()
            ox = (w - display_w) // 2 + int(self._pan_dx)
            oy = (h - display_h) // 2 + int(self._pan_dy)

            self._img_rect = QRectF(ox, oy, display_w, display_h)
            scaled = self._pixmap.scaled(
                display_w, display_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(ox, oy, scaled)

            self._scale_x = display_w / base_w
            self._scale_y = display_h / base_h
            self._offset_x = ox
            self._offset_y = oy
        else:
            painter.fillRect(0, 0, w, h, QColor("#0A0F1A"))
            painter.setPen(QColor("#9FB3C8"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Aktifkan kamera\nuntuk mengatur ROI")
            painter.end()
            return

        # Draw ROIs
        for i, roi in enumerate(self._rois):
            is_selected = (i == self._selected_idx)
            border_c, handle_c = self._roi_color(i)

            rect = self._roi_to_widget(roi)
            alpha = 60 if roi.enabled else 20
            fill = QColor(border_c.red(), border_c.green(), border_c.blue(), alpha)

            pen = QPen(border_c if roi.enabled else QColor("#555555"), 2 if is_selected else 1.5)
            pen.setStyle(Qt.SolidLine if roi.enabled else Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(fill)
            painter.drawRect(rect)

            # Label
            painter.setPen(border_c if roi.enabled else QColor("#555555"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            label = f"{roi.label} {'✓' if roi.enabled else '✗'}"
            painter.drawText(int(rect.x()), int(rect.y()) - 6, label)

            # Grid inside (selected only)
            if is_selected and rect.width() > 30 and rect.height() > 30:
                pen_g = QPen(QColor(255, 255, 255, 50), 1, Qt.DashLine)
                painter.setPen(pen_g)
                for gi in range(1, 3):
                    gx = rect.x() + rect.width() * gi / 3
                    painter.drawLine(int(gx), int(rect.y()),
                                     int(gx), int(rect.bottom()))
                    gy = rect.y() + rect.height() * gi / 3
                    painter.drawLine(int(rect.x()), int(gy),
                                     int(rect.right()), int(gy))

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        if self._pixmap is None:
            return
        pos = event.position()
        btn = event.button()

        # Check handles first (resize)
        ri, hi = self._hit_test_handles(pos)
        if ri >= 0:
            self._selected_idx = ri
            self._resizing = True
            self._resize_handle = hi
            self._drag_start_img = self._map_to_image(pos)
            r = self._rois[ri]
            self._roi_drag_start = (r.x, r.y, r.width, r.height)
            self.update()
            return

        # Check ROI body (drag-move or toggle)
        ri = self._hit_test_roi(pos)
        if ri >= 0:
            self._selected_idx = ri
            if btn == Qt.RightButton:
                # Toggle on right click
                self._rois[ri].enabled = not self._rois[ri].enabled
                self.rois_changed.emit()
            else:
                self._dragging = True
                self._drag_start_img = self._map_to_image(pos)
                self._roi_drag_start = (self._rois[ri].x, self._rois[ri].y)
            self.update()
            return

        # Click empty area — start pan (drag) or add ROI (click)
        if btn == Qt.LeftButton:
            self._click_start = pos
            self._panning = True
            self._pan_start_pos = pos
            self._pan_start_dx = self._pan_dx
            self._pan_start_dy = self._pan_dy

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()

        if self._panning:
            # Pan the view (drag on empty area)
            dx = pos.x() - self._pan_start_pos.x()
            dy = pos.y() - self._pan_start_pos.y()
            self._pan_dx = self._pan_start_dx - dx
            self._pan_dy = self._pan_start_dy - dy
            self.update()
            return

        if self._resizing and self._selected_idx >= 0:
            roi = self._rois[self._selected_idx]
            img_pos = self._map_to_image(pos)
            sx, sy, sw, sh = self._roi_drag_start
            dx = img_pos[0] - self._drag_start_img[0]
            dy = img_pos[1] - self._drag_start_img[1]
            h = self._resize_handle

            if h == 0:  # TL
                nx = min(sx + dx, sx + sw - ROI_MIN_SIZE)
                ny = min(sy + dy, sy + sh - ROI_MIN_SIZE)
                roi.x, roi.y, roi.width, roi.height = nx, ny, sx + sw - nx, sy + sh - ny
            elif h == 1:  # TR
                nw = max(sw + dx, ROI_MIN_SIZE)
                ny = min(sy + dy, sy + sh - ROI_MIN_SIZE)
                roi.x, roi.y, roi.width, roi.height = sx, ny, int(nw), sy + sh - ny
            elif h == 2:  # BL
                nx = min(sx + dx, sx + sw - ROI_MIN_SIZE)
                nh = max(sh + dy, ROI_MIN_SIZE)
                roi.x, roi.y, roi.width, roi.height = nx, sy, sx + sw - nx, int(nh)
            elif h == 3:  # BR
                roi.x, roi.y, roi.width, roi.height = sx, sy, max(sw + dx, ROI_MIN_SIZE), max(sh + dy, ROI_MIN_SIZE)
            self.rois_changed.emit()
            self.update()

        elif self._dragging and self._selected_idx >= 0:
            roi = self._rois[self._selected_idx]
            img_pos = self._map_to_image(pos)
            dx = img_pos[0] - self._drag_start_img[0]
            dy = img_pos[1] - self._drag_start_img[1]
            ox, oy = self._roi_drag_start
            roi.x, roi.y = ox + dx, oy + dy
            self.rois_changed.emit()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._panning:
            self._panning = False
            # If it was a short click (no significant drag) — add ROI at click pos
            if self._click_start is not None:
                dist = (event.position() - self._click_start).manhattanLength()
                if dist < PAN_CLICK_THRESHOLD:
                    pos = self._click_start
                    # If max_rois is set and reached, replace (clear all first)
                    if self._max_rois is not None and len(self._rois) >= self._max_rois:
                        self._rois.clear()
                        self._selected_idx = -1
                    img_pos = self._map_to_image(pos)
                    self.add_roi(img_pos[0], img_pos[1], ROI_MIN_SIZE, ROI_MIN_SIZE)
            self._click_start = None
            return

        if self._dragging or self._resizing:
            self._dragging = False
            self._resizing = False
            self.rois_changed.emit()

    def wheelEvent(self, event: QWheelEvent):
        """Scroll wheel / touchpad panning when zoomed."""
        if self._pixmap is None or self._pixmap.isNull():
            return
        # Use angle delta for smooth scrolling
        delta = event.angleDelta()
        if event.modifiers() & Qt.ShiftModifier:
            # Shift + scroll = horizontal pan
            self._pan_dx -= delta.y() * 0.5
        else:
            # Vertical pan
            self._pan_dy += delta.y() * 0.5
        # Horizontal from touchpad (angleDelta.x())
        self._pan_dx -= delta.x() * 0.5
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_selected_roi()
        elif event.key() == Qt.Key_Space:
            self.toggle_selected_roi()
        elif event.key() == Qt.Key_Plus and (event.modifiers() & Qt.ControlModifier):
            self._zoom = min(5.0, self._zoom * 1.25)
            self.update()
        elif event.key() == Qt.Key_Minus and (event.modifiers() & Qt.ControlModifier):
            self._zoom = max(0.2, self._zoom / 1.25)
            self.update()
        elif event.key() == Qt.Key_0 and (event.modifiers() & Qt.ControlModifier):
            self._zoom = 1.0
            self._pan_dx = 0.0
            self._pan_dy = 0.0
            self.update()
        else:
            super().keyPressEvent(event)
