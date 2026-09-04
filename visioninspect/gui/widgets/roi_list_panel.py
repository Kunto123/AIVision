"""
VisionInspect - ROI List Panel
Panel kontrol untuk daftar ROI: add, delete, toggle, select all/none.
"""

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDoubleSpinBox, QFrame, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
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
    roi_rename_requested = Signal(int)  # index
    roi_toggle_all = Signal(bool)   # True=enable, False=disable
    # (index, threshold) — threshold < 0 berarti "ikut global"
    roi_threshold_changed = Signal(int, float)
    roi_threshold_apply_all = Signal(float)
    # index ROI terpilih — parent yang memanggil roi_editor.start_drawing_mask()
    mask_polygon_requested = Signal(int)
    mask_polygon_clear_requested = Signal(int)

    #: Highlight baris terpilih — SATU-SATUNYA pembeda warna di daftar ini.
    #: Garis kiri tebal supaya tetap terbaca di tema gelap.
    _LIST_QSS = """
    QListWidget {
        background: #0E1A2B; border: 1px solid #233A57; border-radius: 4px;
        outline: none;
    }
    QListWidget::item {
        padding: 5px 6px; border-left: 3px solid transparent;
    }
    QListWidget::item:hover {
        background: #16263D;
    }
    QListWidget::item:selected {
        background: #1D4ED8; color: #FFFFFF;
        border-left: 3px solid #93C5FD; font-weight: bold;
    }
    QListWidget::item:selected:!active {
        background: #1D4ED8; color: #FFFFFF;
    }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cardPanel")
        self._setup_ui()
        self._rois: List[ROIData] = []
        self._updating = False
        # Threshold template (untuk ROI yang "ikut global") — ditampilkan di
        # spin supaya operator lihat angka yang BENAR berlaku.
        self._global_threshold = 0.5

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("ROI List")
        title.setStyleSheet("font-weight: bold; color: #FFFFFF;")
        layout.addWidget(title)

        # ROI list
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setStyleSheet(self._LIST_QSS)
        self._list.setMaximumHeight(200)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        # Nonaktifkan edit in-place bawaan — double-click dipakai untuk dialog rename
        self._list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self._list)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self._add_btn = QPushButton("+ Add")
        self._add_btn.setObjectName("successButton")
        self._add_btn.clicked.connect(self.roi_added.emit)
        btn_layout.addWidget(self._add_btn)

        self._rename_btn = QPushButton("Rename")
        self._rename_btn.setObjectName("primaryButton")
        self._rename_btn.clicked.connect(self._request_rename)
        btn_layout.addWidget(self._rename_btn)

        self._del_btn = QPushButton("Hapus")
        self._del_btn.setObjectName("dangerButton")
        self._del_btn.clicked.connect(self._request_delete)
        btn_layout.addWidget(self._del_btn)

        layout.addLayout(btn_layout)

        # Enable All / Disable All
        toggle_layout = QHBoxLayout()
        self._enable_all_btn = QPushButton("All OK")
        self._enable_all_btn.clicked.connect(lambda: self.roi_toggle_all.emit(True))
        toggle_layout.addWidget(self._enable_all_btn)

        self._disable_all_btn = QPushButton("All NG")
        self._disable_all_btn.clicked.connect(lambda: self.roi_toggle_all.emit(False))
        toggle_layout.addWidget(self._disable_all_btn)

        layout.addLayout(toggle_layout)

        # ── Threshold ROI terpilih ──────────────────────────────────────
        # Tiap ROI lihat fitur beda; centang "ikut global" = pakai threshold template.
        thr_box = QVBoxLayout()
        thr_box.setSpacing(3)
        thr_title = QLabel("Threshold ROI terpilih")
        thr_title.setStyleSheet("font-weight: bold; color: #FFFFFF; "
                                "font-size: 11px;")
        thr_box.addWidget(thr_title)

        thr_row = QHBoxLayout()
        thr_row.setSpacing(4)
        self._thr_global_cb = QCheckBox("Ikut global")
        self._thr_global_cb.setChecked(True)
        self._thr_global_cb.setToolTip(
            "Tercentang: ROI ini memakai threshold template (perilaku lama).\n"
            "Hilangkan centang untuk memberi ROI ini ambang sendiri.")
        self._thr_global_cb.toggled.connect(self._on_thr_global_toggled)
        thr_row.addWidget(self._thr_global_cb)

        self._thr_spin = QDoubleSpinBox()
        self._thr_spin.setRange(0.0, 1.0)
        self._thr_spin.setSingleStep(0.005)
        self._thr_spin.setDecimals(3)
        self._thr_spin.setFixedWidth(80)
        self._thr_spin.setEnabled(False)
        self._thr_spin.valueChanged.connect(self._on_thr_value_changed)
        thr_row.addWidget(self._thr_spin)

        self._thr_all_btn = QPushButton("→ semua")
        self._thr_all_btn.setToolTip(
            "Terapkan nilai ini ke SEMUA ROI. Berguna kalau pembedaan "
            "per-ROI ternyata tidak dibutuhkan.")
        self._thr_all_btn.setEnabled(False)
        self._thr_all_btn.clicked.connect(self._on_thr_apply_all)
        thr_row.addWidget(self._thr_all_btn)
        thr_row.addStretch()
        thr_box.addLayout(thr_row)
        layout.addLayout(thr_box)

        # ── Mask polygon ROI terpilih (opsional) ─────────────────────────
        # Piksel di luar polygon dinolkan saat training & inference.
        mask_box = QVBoxLayout()
        mask_box.setSpacing(3)
        mask_title = QLabel("Mask polygon ROI terpilih")
        mask_title.setStyleSheet("font-weight: bold; color: #FFFFFF; "
                                 "font-size: 11px;")
        mask_box.addWidget(mask_title)

        mask_row = QHBoxLayout()
        mask_row.setSpacing(4)
        self._mask_status_label = QLabel("Tidak ada mask")
        self._mask_status_label.setStyleSheet("color: #9FB3C8; font-size: 11px;")
        mask_row.addWidget(self._mask_status_label, 1)

        self._mask_draw_btn = QPushButton("Gambar Mask")
        self._mask_draw_btn.setToolTip(
            "Gambar polygon mengikuti kontur part di dalam ROI ini.\n"
            "Klik untuk tambah titik, double-click/Enter untuk menutup.\n"
            "Piksel di luar polygon dinolkan saat training & inference.")
        self._mask_draw_btn.setEnabled(False)
        self._mask_draw_btn.clicked.connect(self._on_mask_draw_clicked)
        mask_row.addWidget(self._mask_draw_btn)

        self._mask_clear_btn = QPushButton("Hapus")
        self._mask_clear_btn.setObjectName("dangerButton")
        self._mask_clear_btn.setEnabled(False)
        self._mask_clear_btn.clicked.connect(self._on_mask_clear_clicked)
        mask_row.addWidget(self._mask_clear_btn)
        mask_box.addLayout(mask_row)
        layout.addLayout(mask_box)

        # Info
        self._info_label = QLabel("0 ROI")
        self._info_label.setStyleSheet("color: #9FB3C8; font-size: 11px;")
        layout.addWidget(self._info_label)

    def set_rois(self, rois: List[ROIData], selected: int = -1):
        """Refresh list from ROI data."""
        self._rois = list(rois)
        self._updating = True
        self._list.clear()

        # Warna HANYA untuk menandai baris terpilih (lihat _LIST_QSS) — warna
        # latar per-indeks bikin baris aktif tenggelam.
        for roi in rois:
            icon = "✓" if roi.enabled else "✗"
            text = f"{icon} {roi.label}  ({roi.x},{roi.y} {roi.width}x{roi.height})"
            if getattr(roi, "threshold", None) is not None:
                text += f"  ⌁{roi.threshold:.3f}"   # punya ambang sendiri
            item = QListWidgetItem(text)
            # Pembeda selain seleksi: aktif vs nonaktif. Redupnya tidak terlalu
            # gelap — harus tetap terbaca di atas latar biru saat terpilih.
            item.setForeground(QColor("#E2E8F0" if roi.enabled else "#94A3B8"))
            self._list.addItem(item)

        if 0 <= selected < len(rois):
            self._list.setCurrentRow(selected)

        n_custom = sum(1 for r in rois
                       if getattr(r, "threshold", None) is not None)
        extra = f", {n_custom} threshold sendiri" if n_custom else ""
        self._info_label.setText(
            f"{len(rois)} ROI ({sum(1 for r in rois if r.enabled)} aktif{extra})")
        self._updating = False
        self._sync_threshold_widgets(self._list.currentRow())
        self._sync_mask_widgets(self._list.currentRow())

    # ---- Threshold per ROI ----

    def set_global_threshold(self, value: float):
        """Threshold template — ditampilkan untuk ROI yang 'ikut global'."""
        self._global_threshold = float(value)
        if self._thr_global_cb.isChecked():
            self._sync_threshold_widgets(self._list.currentRow())

    def _sync_threshold_widgets(self, row: int):
        """Tampilkan threshold ROI terpilih tanpa memicu sinyal simpan."""
        self._updating = True
        try:
            roi = self._rois[row] if 0 <= row < len(self._rois) else None
            thr = getattr(roi, "threshold", None) if roi else None
            has_sel = roi is not None
            custom = has_sel and thr is not None
            self._thr_global_cb.setEnabled(has_sel)
            self._thr_global_cb.setChecked(not custom)
            self._thr_spin.setEnabled(custom)
            self._thr_all_btn.setEnabled(custom)
            # Ikut global → tampilkan nilai global (abu, tidak bisa diedit).
            # Punya sendiri → tampilkan nilainya.
            self._thr_spin.setValue(
                float(thr) if custom else float(self._global_threshold))
            self._thr_global_cb.setText(
                "Ikut global" if custom
                else f"Ikut global ({self._global_threshold:.3f})")
        finally:
            self._updating = False

    def _on_thr_global_toggled(self, checked: bool):
        if self._updating:
            return
        row = self._list.currentRow()
        if row < 0:
            return
        self._thr_spin.setEnabled(not checked)
        self._thr_all_btn.setEnabled(not checked)
        # < 0 = kembali ikut global (field threshold dibuang dari config).
        # Titik awal ambang sendiri = nilai global, bukan sisa ROI sebelumnya.
        self.roi_threshold_changed.emit(
            row, -1.0 if checked else float(self._thr_spin.value()))

    def _on_thr_value_changed(self, value: float):
        if self._updating or self._thr_global_cb.isChecked():
            return
        row = self._list.currentRow()
        if row >= 0:
            self.roi_threshold_changed.emit(row, float(value))

    def _on_thr_apply_all(self):
        if self._thr_global_cb.isChecked():
            return
        self.roi_threshold_apply_all.emit(float(self._thr_spin.value()))

    # ---- Mask polygon per ROI ----

    def _sync_mask_widgets(self, row: int):
        """Tampilkan status mask ROI terpilih (ada/tidak) di panel."""
        roi = self._rois[row] if 0 <= row < len(self._rois) else None
        has_sel = roi is not None
        has_mask = has_sel and bool(getattr(roi, "mask_polygon", None))
        self._mask_draw_btn.setEnabled(has_sel)
        self._mask_draw_btn.setText("Gambar Ulang" if has_mask else "Gambar Mask")
        self._mask_clear_btn.setEnabled(has_mask)
        if not has_sel:
            self._mask_status_label.setText("Pilih ROI dulu")
        elif has_mask:
            n = len(roi.mask_polygon)
            self._mask_status_label.setText(f"Mask aktif ({n} titik)")
        else:
            self._mask_status_label.setText("Tidak ada mask — ROI dipakai penuh")

    def _on_mask_draw_clicked(self):
        row = self._list.currentRow()
        if row >= 0:
            self.mask_polygon_requested.emit(row)

    def _on_mask_clear_clicked(self):
        row = self._list.currentRow()
        if row >= 0:
            self.mask_polygon_clear_requested.emit(row)

    def select_row(self, index: int):
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)
            # Seleksi bisa datang dari editor ROI (klik kotak di gambar) —
            # panel threshold harus ikut menunjuk ROI yang sama.
            self._sync_threshold_widgets(index)
            self._sync_mask_widgets(index)

    def _on_item_clicked(self, item):
        if self._updating:
            return
        row = self._list.row(item)
        self._sync_threshold_widgets(row)
        self._sync_mask_widgets(row)
        self.roi_selected.emit(row)

    def _on_item_double_clicked(self, item):
        if self._updating:
            return
        row = self._list.row(item)
        if row >= 0:
            self.roi_rename_requested.emit(row)

    def _request_rename(self):
        row = self._list.currentRow()
        if row >= 0:
            self.roi_rename_requested.emit(row)

    def _request_delete(self):
        row = self._list.currentRow()
        if row >= 0:
            self.roi_delete_requested.emit(row)
