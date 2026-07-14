"""
VisionInspect - ROI List Panel
Panel kontrol untuk daftar ROI: add, delete, toggle, select all/none.
"""

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget, QAbstractItemView,
)

from visioninspect.gui.widgets.roi_editor import ROIData


class ROIListPanel(QFrame):
    """
    Panel daftar ROI dengan kontrol.
    Sync selection dengan ROIEditor.
    """

    roi_selected = Signal(int)      # index
    roi_added = Signal()
    roi_delete_requested = Signal(int)  # index
    roi_toggle_all = Signal(bool)   # True=enable, False=disable

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cardPanel")
        self._setup_ui()
        self._rois: List[ROIData] = []
        self._updating = False

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("📍 ROI List")
        title.setStyleSheet("font-weight: bold; color: #FFFFFF;")
        layout.addWidget(title)

        # ROI list
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setMaximumHeight(200)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self._add_btn = QPushButton("➕ Add")
        self._add_btn.setObjectName("successButton")
        self._add_btn.clicked.connect(self.roi_added.emit)
        btn_layout.addWidget(self._add_btn)

        self._del_btn = QPushButton("🗑 Del")
        self._del_btn.setObjectName("dangerButton")
        self._del_btn.clicked.connect(self._request_delete)
        btn_layout.addWidget(self._del_btn)

        layout.addLayout(btn_layout)

        # Enable All / Disable All
        toggle_layout = QHBoxLayout()
        self._enable_all_btn = QPushButton("✅ All OK")
        self._enable_all_btn.clicked.connect(lambda: self.roi_toggle_all.emit(True))
        toggle_layout.addWidget(self._enable_all_btn)

        self._disable_all_btn = QPushButton("❌ All NG")
        self._disable_all_btn.clicked.connect(lambda: self.roi_toggle_all.emit(False))
        toggle_layout.addWidget(self._disable_all_btn)

        layout.addLayout(toggle_layout)

        # Info
        self._info_label = QLabel("0 ROI")
        self._info_label.setStyleSheet("color: #9FB3C8; font-size: 11px;")
        layout.addWidget(self._info_label)

    def set_rois(self, rois: List[ROIData], selected: int = -1):
        """Refresh list from ROI data."""
        self._rois = list(rois)
        self._updating = True
        self._list.clear()

        colors = ["#FFFFFF", "#FFD700", "#00BFFF", "#FF69B4", "#7B68EE"]
        for i, roi in enumerate(rois):
            c = colors[i % len(colors)]
            icon = "✓" if roi.enabled else "✗"
            text = f"{icon} {roi.label}  ({roi.x},{roi.y} {roi.width}x{roi.height})"
            item = QListWidgetItem(text)
            item.setForeground(Qt.GlobalColor.white)
            if roi.enabled:
                item.setBackground(QColor(f"{c}20"))
            self._list.addItem(item)

        if 0 <= selected < len(rois):
            self._list.setCurrentRow(selected)

        self._info_label.setText(f"{len(rois)} ROI ({sum(1 for r in rois if r.enabled)} aktif)")
        self._updating = False

    def select_row(self, index: int):
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)

    def _on_item_clicked(self, item):
        if self._updating:
            return
        row = self._list.row(item)
        self.roi_selected.emit(row)

    def _request_delete(self):
        row = self._list.currentRow()
        if row >= 0:
            self.roi_delete_requested.emit(row)
