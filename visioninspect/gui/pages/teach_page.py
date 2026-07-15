"""
VisionInspect - Teach Page
Teaching & Training: capture OK/NG, gallery, train button, threshold slider, histogram.
Plus Part Presence Check configuration.
Layout: Left panel = dual ROI editors (QC + Gate), Right panel = controls.
"""

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from visioninspect.gui.widgets.histogram_widget import HistogramWidget
from visioninspect.gui.widgets.roi_editor import ROIEditor, ROIData
from visioninspect.gui.widgets.roi_list_panel import ROIListPanel
from visioninspect.gui.widgets.thumbnail import ThumbnailWidget
from visioninspect.utils.i18n import Translator


class TeachPage(QWidget):
    """Halaman TEACH — teaching dan training model dengan multi-template."""

    image_deleted = Signal(str)

    def __init__(self, translator: Translator, parent=None):
        super().__init__(parent)
        self._tr = translator
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # ── Left Panel: Preview + Gate + Gallery ──
        # Wrapped in a QScrollArea (same pattern as the right panel below) so the
        # QC preview + Gate preview + gallery stack never overlaps on shorter
        # screens (e.g. 1366x768 panel PCs) — it scrolls instead of squeezing.
        left_panel = QFrame()
        left_panel.setObjectName("cardPanel")
        left_outer = QVBoxLayout(left_panel)
        left_outer.setContentsMargins(0, 0, 0, 0)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_outer.addWidget(left_scroll)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)
        left_scroll.setWidget(left_content)

        # Title row
        title_row = QHBoxLayout()
        title = QLabel("📋 " + self._tr.tr("teach_title"))
        title.setObjectName("sectionTitle")
        title_row.addWidget(title)
        title_row.addStretch()
        left_layout.addLayout(title_row)

        # Template selector
        tmpl_bar = QFrame()
        tmpl_bar.setObjectName("cardPanel")
        tmpl_bar.setMaximumHeight(44)
        tmpl_layout = QHBoxLayout(tmpl_bar)
        tmpl_layout.setContentsMargins(8, 4, 8, 4)
        tmpl_layout.setSpacing(6)

        tmpl_layout.addWidget(QLabel("Template:"))
        self._template_combo = QComboBox()
        self._template_combo.setMinimumWidth(160)
        self._template_combo.setToolTip("Pilih template aktif")
        tmpl_layout.addWidget(self._template_combo, 1)

        self._add_template_btn = QPushButton("➕")
        self._add_template_btn.setFixedWidth(36)
        self._add_template_btn.setObjectName("successButton")
        self._add_template_btn.setToolTip("Buat template baru")
        tmpl_layout.addWidget(self._add_template_btn)

        self._clear_btn = QPushButton("🗑")
        self._clear_btn.setFixedWidth(36)
        self._clear_btn.setToolTip("Hapus template aktif")
        tmpl_layout.addWidget(self._clear_btn)

        left_layout.addWidget(tmpl_bar)

        # Capture buttons
        capture_row = QHBoxLayout()
        capture_row.setSpacing(6)
        self._capture_ok_btn = QPushButton("✅ Capture OK")
        self._capture_ok_btn.setObjectName("successButton")
        self._capture_ok_btn.setMinimumHeight(38)
        capture_row.addWidget(self._capture_ok_btn)

        self._capture_ng_btn = QPushButton("❌ Capture NG")
        self._capture_ng_btn.setObjectName("dangerButton")
        self._capture_ng_btn.setMinimumHeight(38)
        capture_row.addWidget(self._capture_ng_btn)

        self._import_btn = QPushButton("📁 Import")
        self._import_btn.setMinimumHeight(38)
        capture_row.addWidget(self._import_btn)
        left_layout.addLayout(capture_row)

        # Import status
        self._import_status_label = QLabel("")
        self._import_status_label.setObjectName("secondaryText")
        self._import_status_label.setAlignment(Qt.AlignCenter)
        self._import_status_label.setStyleSheet(
            "color: #F59E0B; font-weight: bold; padding: 4px; "
            "background-color: #1A2A44; border-radius: 4px;")
        self._import_status_label.hide()
        left_layout.addWidget(self._import_status_label)

        # ── Preview area: 3 columns — QC Region | Daftar ROI | Gate Part ──
        preview_row = QHBoxLayout()
        preview_row.setSpacing(8)

        # Col 1: QC Region
        qc_box = QVBoxLayout()
        qc_box.setSpacing(2)
        qc_label = QLabel("🔍 QC Region")
        qc_label.setObjectName("secondaryText")
        qc_box.addWidget(qc_label)
        self._roi_editor = ROIEditor()
        self._roi_editor.setMinimumSize(240, 200)
        qc_box.addWidget(self._roi_editor, 1)
        preview_row.addLayout(qc_box, 3)

        # Col 2: Daftar ROI
        rp_box = QVBoxLayout()
        rp_box.setSpacing(2)
        rp_label = QLabel("📍 Daftar ROI")
        rp_label.setObjectName("secondaryText")
        rp_box.addWidget(rp_label)
        self._roi_panel = ROIListPanel()
        self._roi_panel.setMinimumWidth(140)
        rp_box.addWidget(self._roi_panel, 1)
        preview_row.addLayout(rp_box, 1)

        # Col 3: Gate Part
        gate_box = QVBoxLayout()
        gate_box.setSpacing(2)
        gate_label = QLabel("🧩 Gate Part 1")
        gate_label.setObjectName("secondaryText")
        gate_box.addWidget(gate_label)
        self._gate_roi_editor = ROIEditor()
        self._gate_roi_editor.setMinimumSize(180, 200)
        self._gate_roi_editor.set_max_rois(1)
        self._gate_roi_editor.setToolTip(
            "Gambar 1 kotak di area yang harus terisi part")
        gate_box.addWidget(self._gate_roi_editor, 1)
        gate_hint = QLabel("🎯 Klik & tarik (maks 1)")
        gate_hint.setObjectName("secondaryText")
        gate_hint.setAlignment(Qt.AlignCenter)
        gate_box.addWidget(gate_hint)
        preview_row.addLayout(gate_box, 2)

        left_layout.addLayout(preview_row, 1)

        # Gallery
        gallery_layout = QHBoxLayout()
        gallery_layout.setSpacing(8)

        # OK
        ok_group = QGroupBox("✅ " + self._tr.tr("teach_gallery_ok"))
        ok_g = QVBoxLayout(ok_group)
        ok_g.setContentsMargins(6, 6, 6, 6)
        self._ok_count_label = QLabel(self._tr.tr("teach_count_ok", count=0))
        self._ok_count_label.setStyleSheet("color: #22C55E; font-weight: bold;")
        ok_g.addWidget(self._ok_count_label)
        ok_scroll = QScrollArea()
        ok_scroll.setWidgetResizable(True)
        ok_scroll.setMinimumHeight(80)
        ok_scroll.setStyleSheet("background: #111D30; border: 1px solid #233A57; border-radius: 4px;")
        self._ok_gallery_widget = QWidget()
        self._ok_gallery_layout = QHBoxLayout(self._ok_gallery_widget)
        self._ok_gallery_layout.setContentsMargins(4, 4, 4, 4)
        self._ok_gallery_layout.addStretch()
        ok_scroll.setWidget(self._ok_gallery_widget)
        ok_g.addWidget(ok_scroll)
        gallery_layout.addWidget(ok_group)

        # NG
        ng_group = QGroupBox("❌ " + self._tr.tr("teach_gallery_ng"))
        ng_g = QVBoxLayout(ng_group)
        ng_g.setContentsMargins(6, 6, 6, 6)
        self._ng_count_label = QLabel(self._tr.tr("teach_count_ng", count=0))
        self._ng_count_label.setStyleSheet("color: #EF4444; font-weight: bold;")
        ng_g.addWidget(self._ng_count_label)
        ng_scroll = QScrollArea()
        ng_scroll.setWidgetResizable(True)
        ng_scroll.setMinimumHeight(80)
        ng_scroll.setStyleSheet("background: #111D30; border: 1px solid #233A57; border-radius: 4px;")
        self._ng_gallery_widget = QWidget()
        self._ng_gallery_layout = QHBoxLayout(self._ng_gallery_widget)
        self._ng_gallery_layout.setContentsMargins(4, 4, 4, 4)
        self._ng_gallery_layout.addStretch()
        ng_scroll.setWidget(self._ng_gallery_widget)
        ng_g.addWidget(ng_scroll)
        gallery_layout.addWidget(ng_group)

        left_layout.addLayout(gallery_layout)

        layout.addWidget(left_panel, 3)

        # ── Right Panel: Training + Part Check Controls ──
        # Wrapped in a QScrollArea (same pattern as SettingsPage) so the stack of
        # cards below never overlaps on shorter screens (e.g. 1366x768 panel PCs) —
        # it scrolls instead of squeezing widgets into negative space.
        right_panel = QFrame()
        right_panel.setObjectName("cardPanel")
        right_outer = QVBoxLayout(right_panel)
        right_outer.setContentsMargins(0, 0, 0, 0)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_outer.addWidget(right_scroll)

        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)
        right_scroll.setWidget(right_content)

        # Model version
        self._version_label = QLabel("💾 Model: —")
        self._version_label.setObjectName("secondaryText")
        right_layout.addWidget(self._version_label)

        # Train button + progress stacked
        train_box = QVBoxLayout()
        train_box.setSpacing(4)
        self._train_btn = QPushButton("🎯 TRAIN")
        self._train_btn.setObjectName("primaryButton")
        self._train_btn.setMinimumHeight(44)
        self._train_btn.setEnabled(False)
        train_box.addWidget(self._train_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("")
        self._progress_bar.setMaximumHeight(18)
        train_box.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setObjectName("secondaryText")
        self._progress_label.setMaximumHeight(16)
        train_box.addWidget(self._progress_label)
        right_layout.addLayout(train_box)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #233A57;")
        right_layout.addWidget(sep)

        # ── Part Presence Check Controls ──
        # Step 1 of the 2-step flow (gate before QC) — form-aligned rows, with
        # method-specific rows hidden via setRowVisible so the panel only ever
        # shows the 2-3 controls relevant to the chosen method, not all 6 at once.
        pc_group = QGroupBox("🧩 Part Presence (Step 1)")
        pc_outer = QVBoxLayout(pc_group)
        pc_outer.setSpacing(8)
        pc_outer.setContentsMargins(10, 8, 10, 10)

        self._pc_enabled_cb = QCheckBox("Aktifkan Part Presence Check")
        self._pc_enabled_cb.setStyleSheet("font-weight: bold;")
        pc_outer.addWidget(self._pc_enabled_cb)

        self._pc_form = QFormLayout()
        self._pc_form.setSpacing(6)
        self._pc_form.setLabelAlignment(Qt.AlignLeft)
        self._pc_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._pc_method_combo = QComboBox()
        self._pc_method_combo.addItem("Warna (mean/std)", "color")
        self._pc_method_combo.addItem("Tepi (Canny)", "edge")
        self._pc_method_combo.addItem("Keduanya (AND)", "both")
        self._pc_method_combo.setToolTip(
            "Warna: cocok bila part punya warna kontras dengan latar. "
            "Tepi: hanya hitung JUMLAH tepi — tidak deteksi posisi. "
            "Keduanya: AND — part harus lolos kedua metode."
        )
        self._pc_method_combo.currentIndexChanged.connect(self._update_pc_field_visibility)
        self._pc_form.addRow("Metode:", self._pc_method_combo)

        self._pc_color_th_spin = QDoubleSpinBox()
        self._pc_color_th_spin.setRange(0.01, 5.0)
        self._pc_color_th_spin.setValue(0.35)
        self._pc_color_th_spin.setSingleStep(0.05)
        self._pc_color_th_spin.setDecimals(3)
        self._pc_form.addRow("Toleransi warna:", self._pc_color_th_spin)

        self._pc_edge_th_spin = QDoubleSpinBox()
        self._pc_edge_th_spin.setRange(0.001, 10.0)
        self._pc_edge_th_spin.setValue(0.5)
        self._pc_edge_th_spin.setSingleStep(0.1)
        self._pc_edge_th_spin.setDecimals(3)
        self._pc_edge_th_spin.setToolTip(
            "Ambang perubahan edge relatif (rasio). "
            "Makin kecil = makin sensitif. "
            "0.5 = part dianggap berbeda bila edge berubah >50% "
            "dari master."
        )
        self._pc_form.addRow("Toleransi tepi:", self._pc_edge_th_spin)

        canny_row = QHBoxLayout()
        canny_row.setSpacing(4)
        self._pc_canny_low_spin = QSpinBox()
        self._pc_canny_low_spin.setRange(0, 500)
        self._pc_canny_low_spin.setValue(50)
        canny_row.addWidget(self._pc_canny_low_spin, 1)
        canny_row.addWidget(QLabel("→"))
        self._pc_canny_high_spin = QSpinBox()
        self._pc_canny_high_spin.setRange(0, 500)
        self._pc_canny_high_spin.setValue(150)
        canny_row.addWidget(self._pc_canny_high_spin, 1)
        self._pc_canny_row_widget = QWidget()
        self._pc_canny_row_widget.setLayout(canny_row)
        self._pc_form.addRow("Ambang Canny:", self._pc_canny_row_widget)

        pc_outer.addLayout(self._pc_form)

        # Master photo — capture + status on one row, compact thumbnail below
        master_row = QHBoxLayout()
        master_row.setSpacing(8)
        self._capture_master_btn = QPushButton("📸 Ambil Master")
        self._capture_master_btn.setObjectName("successButton")
        self._capture_master_btn.setMinimumHeight(32)
        master_row.addWidget(self._capture_master_btn)

        self._master_thumbnail = QLabel("—")
        self._master_thumbnail.setAlignment(Qt.AlignCenter)
        self._master_thumbnail.setFixedSize(56, 40)
        self._master_thumbnail.setObjectName("secondaryText")
        self._master_thumbnail.setStyleSheet(
            "background: #0A0F1A; border: 1px solid #233A57; border-radius: 4px;")
        master_row.addWidget(self._master_thumbnail)

        self._master_status_label = QLabel("⏹ Belum ada master")
        self._master_status_label.setObjectName("secondaryText")
        self._master_status_label.setWordWrap(True)
        master_row.addWidget(self._master_status_label, 1)
        pc_outer.addLayout(master_row)

        right_layout.addWidget(pc_group)

        self._update_pc_field_visibility()

        # Threshold
        threshold_group = QGroupBox("🎚️ " + self._tr.tr("teach_threshold"))
        th_layout = QVBoxLayout(threshold_group)
        th_layout.setSpacing(4)
        self._threshold_slider = QSlider(Qt.Horizontal)
        self._threshold_slider.setRange(0, 1000)
        self._threshold_slider.setValue(500)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        th_layout.addWidget(self._threshold_slider)

        tv = QHBoxLayout()
        tv.addWidget(QLabel("0.0"))
        self._threshold_value_label = QLabel("0.500")
        self._threshold_value_label.setAlignment(Qt.AlignCenter)
        self._threshold_value_label.setObjectName("bigCounter")
        tv.addWidget(self._threshold_value_label)
        tv.addWidget(QLabel("1.0"))
        th_layout.addLayout(tv)
        right_layout.addWidget(threshold_group)

        # Histogram
        histogram_group = QGroupBox("📊 Histogram")
        hist_layout = QVBoxLayout(histogram_group)
        self._histogram = HistogramWidget()
        self._histogram.setMinimumHeight(100)
        hist_layout.addWidget(self._histogram)
        right_layout.addWidget(histogram_group)

        # Warning
        self._warning_label = QLabel("")
        self._warning_label.setObjectName("secondaryText")
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet("color: #F59E0B;")
        right_layout.addWidget(self._warning_label)

        right_layout.addStretch()

        layout.addWidget(right_panel, 2)

    # ---- Gallery ----

    def add_ok_thumbnail(self, pixmap, path=""):
        t = ThumbnailWidget(pixmap, path, "#22C55E")
        t.deleted.connect(lambda p: self._on_delete_image(p, "ok"))
        self._ok_gallery_layout.insertWidget(self._ok_gallery_layout.count() - 1, t)

    def add_ng_thumbnail(self, pixmap, path=""):
        t = ThumbnailWidget(pixmap, path, "#EF4444")
        t.deleted.connect(lambda p: self._on_delete_image(p, "ng"))
        self._ng_gallery_layout.insertWidget(self._ng_gallery_layout.count() - 1, t)

    def _on_delete_image(self, path, label):
        import os
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        self.image_deleted.emit(label)

    def clear_galleries(self):
        for layout in [self._ok_gallery_layout, self._ng_gallery_layout]:
            while layout.count() > 1:
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    # ---- Slots ----

    @Slot()
    def set_ok_count(self, count: int):
        self._ok_count_label.setText(f"✅ {self._tr.tr('teach_count_ok', count=count)}")
        self._train_btn.setEnabled(count > 0)

    @Slot()
    def set_ng_count(self, count: int):
        self._ng_count_label.setText(f"❌ {self._tr.tr('teach_count_ng', count=count)}")

    @Slot()
    def set_preview(self, pixmap: QPixmap):
        self._roi_editor.set_pixmap(pixmap)
        self._gate_roi_editor.set_pixmap(pixmap)

    @Slot()
    def set_preview_text(self, text: str):
        empty = QPixmap()
        self._roi_editor.set_pixmap(empty)
        self._gate_roi_editor.set_pixmap(empty)

    @Slot()
    def set_version(self, version: int):
        self._version_label.setText(f"💾 Model: v{version}" if version else "💾 Model: —")

    @Slot()
    def set_training_progress(self, percent: int, message: str = ""):
        self._progress_bar.setValue(percent)
        self._progress_bar.setFormat(f"{percent}%")
        self._progress_label.setText(message or f"{percent}%")

    @Slot()
    def set_training_done(self):
        self._progress_bar.setValue(100)
        self._progress_bar.setFormat("✅ Selesai!")
        self._progress_label.setText(self._tr.tr("teach_training_done"))

    @Slot()
    def set_training_failed(self, error: str):
        self._progress_bar.setFormat("❌ Gagal")
        self._progress_label.setText(self._tr.tr("teach_training_failed", error=error))

    @Slot()
    def set_threshold(self, value: float):
        self._threshold_value_label.setText(f"{value:.3f}")
        self._threshold_slider.blockSignals(True)
        self._threshold_slider.setValue(int(value * 1000))
        self._threshold_slider.blockSignals(False)

    def _on_threshold_changed(self, value: int):
        self._threshold_value_label.setText(f"{value / 1000.0:.3f}")

    @Slot()
    def set_warning(self, message: str):
        self._warning_label.setText(message)

    @Slot()
    def set_histogram_data(self, ok_scores, ng_scores, threshold=0.5):
        self._histogram.set_data(ok_scores, ng_scores, threshold)

    @Slot()
    def clear_histogram(self):
        self._histogram.clear_data()

    # ---- Import Mode ----

    @Slot()
    def show_import_mode(self, active: bool):
        if active:
            self._capture_ok_btn.setText("✅ Simpan OK")
            self._capture_ng_btn.setText("❌ Simpan NG")
            self._import_btn.hide()
            self._import_status_label.show()
        else:
            self._capture_ok_btn.setText("✅ Capture OK")
            self._capture_ng_btn.setText("❌ Capture NG")
            self._import_btn.show()
            self._import_status_label.hide()

    @Slot()
    def set_import_status(self, current: int, total: int):
        self._import_status_label.setText(f"📁 Import: {current}/{total} — pilih OK atau NG")

    # ---- Part Check ----

    def set_part_check_config(self, cfg: dict):
        self._pc_enabled_cb.setChecked(cfg.get("enabled", False))
        m = cfg.get("method", "both")
        idx = self._pc_method_combo.findData(m)
        if idx >= 0:
            self._pc_method_combo.setCurrentIndex(idx)
        self._pc_color_th_spin.setValue(cfg.get("color_threshold", 0.35))
        self._pc_edge_th_spin.setValue(cfg.get("edge_threshold", 0.5))
        self._pc_canny_low_spin.setValue(cfg.get("canny_low", 50))
        self._pc_canny_high_spin.setValue(cfg.get("canny_high", 150))

    def set_gate_roi(self, rois: list):
        objs = [ROIData.from_dict(r) for r in rois] if rois else []
        self._gate_roi_editor.set_rois(objs)

    def get_gate_roi(self) -> list:
        return [r.to_dict() for r in self._gate_roi_editor.get_rois()]

    def set_master_status(self, has_master: bool, timestamp="", thumbnail=None):
        if has_master and timestamp:
            self._master_status_label.setText(f"✅ {timestamp}")
            self._master_status_label.setStyleSheet("color: #22C55E; font-weight: bold;")
        else:
            self._master_status_label.setText("⏹ Belum ada master")
            self._master_status_label.setStyleSheet("color: #EF4444;")
        if thumbnail and not thumbnail.isNull():
            self._master_thumbnail.setPixmap(
                thumbnail.scaled(56, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self._master_thumbnail.clear()
            self._master_thumbnail.setText("—")

    def _update_pc_field_visibility(self, *_):
        """Show only the threshold/canny rows relevant to the selected method."""
        method = self._pc_method_combo.currentData()
        show_color = method in ("color", "both")
        show_edge = method in ("edge", "both")
        self._pc_form.setRowVisible(self._pc_color_th_spin, show_color)
        self._pc_form.setRowVisible(self._pc_edge_th_spin, show_edge)
        self._pc_form.setRowVisible(self._pc_canny_row_widget, show_edge)

    # ---- Accessors ----

    def get_capture_ok_button(self): return self._capture_ok_btn
    def get_capture_ng_button(self): return self._capture_ng_btn
    def get_import_button(self): return self._import_btn
    def get_train_button(self): return self._train_btn
    def get_threshold_slider(self): return self._threshold_slider
    def get_progress_bar(self): return self._progress_bar
    def get_template_combo(self): return self._template_combo
    def get_add_template_button(self): return self._add_template_btn
    def get_clear_button(self): return self._clear_btn
    def get_roi_editor(self): return self._roi_editor
    def get_roi_panel(self): return self._roi_panel
    def get_pc_enabled_cb(self): return self._pc_enabled_cb
    def get_pc_method_combo(self): return self._pc_method_combo
    def get_pc_color_th_spin(self): return self._pc_color_th_spin
    def get_pc_edge_th_spin(self): return self._pc_edge_th_spin
    def get_pc_canny_low_spin(self): return self._pc_canny_low_spin
    def get_pc_canny_high_spin(self): return self._pc_canny_high_spin
    def get_gate_roi_editor(self): return self._gate_roi_editor
    def get_capture_master_button(self): return self._capture_master_btn
