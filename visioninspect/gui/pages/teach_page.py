"""
VisionInspect - Teach Page
Teaching & Training: capture OK/NG, gallery, train button, threshold slider, histogram.
Support multiple templates.
"""

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
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

    image_deleted = Signal(str)  # label: "ok" or "ng"

    def __init__(self, translator: Translator, parent=None):
        super().__init__(parent)
        self._tr = translator
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # === Left Panel: Template selector + Capture + Preview + Gallery ===
        left_panel = QFrame()
        left_panel.setObjectName("cardPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("📋 " + self._tr.tr("teach_title"))
        title.setObjectName("sectionTitle")
        left_layout.addWidget(title)

        # === Template selector bar ===
        tmpl_bar = QFrame()
        tmpl_bar.setObjectName("cardPanel")
        tmpl_bar.setMaximumHeight(48)
        tmpl_layout = QHBoxLayout(tmpl_bar)
        tmpl_layout.setContentsMargins(8, 4, 8, 4)

        tmpl_layout.addWidget(QLabel("Template:"))
        self._template_combo = QComboBox()
        self._template_combo.setMinimumWidth(180)
        self._template_combo.setToolTip("Pilih template aktif")
        tmpl_layout.addWidget(self._template_combo)

        self._add_template_btn = QPushButton("➕ Baru")
        self._add_template_btn.setObjectName("successButton")
        self._add_template_btn.setToolTip("Buat template baru")
        tmpl_layout.addWidget(self._add_template_btn)

        self._clear_btn = QPushButton("🗑 Hapus")
        self._clear_btn.setToolTip("Hapus template aktif")
        tmpl_layout.addWidget(self._clear_btn)

        left_layout.addWidget(tmpl_bar)

        # Capture buttons
        capture_layout = QHBoxLayout()
        self._capture_ok_btn = QPushButton("✅ Capture OK")
        self._capture_ok_btn.setObjectName("successButton")
        self._capture_ok_btn.setMinimumHeight(44)
        self._capture_ok_btn.setToolTip("Ambil gambar dari live view untuk sampel OK")
        capture_layout.addWidget(self._capture_ok_btn)

        self._capture_ng_btn = QPushButton("❌ Capture NG")
        self._capture_ng_btn.setObjectName("dangerButton")
        self._capture_ng_btn.setMinimumHeight(44)
        self._capture_ng_btn.setToolTip("Ambil gambar dari live view untuk sampel NG")
        capture_layout.addWidget(self._capture_ng_btn)

        self._import_btn = QPushButton("📁 Import")
        self._import_btn.setMinimumHeight(44)
        self._import_btn.setToolTip("Import gambar dari file")
        capture_layout.addWidget(self._import_btn)
        left_layout.addLayout(capture_layout)

        # Import review status — hidden by default, shown during import review
        self._import_status_label = QLabel("")
        self._import_status_label.setObjectName("secondaryText")
        self._import_status_label.setAlignment(Qt.AlignCenter)
        self._import_status_label.setStyleSheet(
            "color: #F59E0B; font-weight: bold; padding: 4px; "
            "background-color: #1A2A44; border-radius: 4px;")
        self._import_status_label.hide()
        left_layout.addWidget(self._import_status_label)

        # Live preview with ROI editor + ROI list panel
        preview_row = QHBoxLayout()
        preview_row.setSpacing(8)

        self._roi_editor = ROIEditor()
        self._roi_editor.setMinimumSize(320, 240)
        preview_row.addWidget(self._roi_editor, 3)

        self._roi_panel = ROIListPanel()
        self._roi_panel.setMinimumWidth(220)
        preview_row.addWidget(self._roi_panel, 1)

        left_layout.addLayout(preview_row, 1)

        # Gallery thumbnails (scrollable)
        gallery_layout = QHBoxLayout()

        # OK Gallery
        ok_group = QGroupBox("✅ " + self._tr.tr("teach_gallery_ok"))
        ok_group_layout = QVBoxLayout(ok_group)
        self._ok_count_label = QLabel(self._tr.tr("teach_count_ok", count=0))
        self._ok_count_label.setStyleSheet("color: #22C55E; font-weight: bold;")
        ok_group_layout.addWidget(self._ok_count_label)

        ok_scroll = QScrollArea()
        ok_scroll.setWidgetResizable(True)
        ok_scroll.setMinimumHeight(100)
        ok_scroll.setStyleSheet("background-color: #111D30; border: 1px solid #233A57; border-radius: 4px;")
        self._ok_gallery_widget = QWidget()
        self._ok_gallery_layout = QHBoxLayout(self._ok_gallery_widget)
        self._ok_gallery_layout.setContentsMargins(4, 4, 4, 4)
        self._ok_gallery_layout.addStretch()
        ok_scroll.setWidget(self._ok_gallery_widget)
        ok_group_layout.addWidget(ok_scroll)
        gallery_layout.addWidget(ok_group)

        # NG Gallery
        ng_group = QGroupBox("❌ " + self._tr.tr("teach_gallery_ng"))
        ng_group_layout = QVBoxLayout(ng_group)
        self._ng_count_label = QLabel(self._tr.tr("teach_count_ng", count=0))
        self._ng_count_label.setStyleSheet("color: #EF4444; font-weight: bold;")
        ng_group_layout.addWidget(self._ng_count_label)

        ng_scroll = QScrollArea()
        ng_scroll.setWidgetResizable(True)
        ng_scroll.setMinimumHeight(100)
        ng_scroll.setStyleSheet("background-color: #111D30; border: 1px solid #233A57; border-radius: 4px;")
        self._ng_gallery_widget = QWidget()
        self._ng_gallery_layout = QHBoxLayout(self._ng_gallery_widget)
        self._ng_gallery_layout.setContentsMargins(4, 4, 4, 4)
        self._ng_gallery_layout.addStretch()
        ng_scroll.setWidget(self._ng_gallery_widget)
        ng_group_layout.addWidget(ng_scroll)
        gallery_layout.addWidget(ng_group)

        left_layout.addLayout(gallery_layout)

        layout.addWidget(left_panel, 2)

        # === Right Panel: Training controls ===
        right_panel = QFrame()
        right_panel.setObjectName("cardPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        # Model version
        self._version_label = QLabel("💾 Model: —")
        self._version_label.setObjectName("secondaryText")
        right_layout.addWidget(self._version_label)

        # Train button
        self._train_btn = QPushButton("🎯 TRAIN")
        self._train_btn.setObjectName("primaryButton")
        self._train_btn.setMinimumHeight(48)
        self._train_btn.setEnabled(False)
        self._train_btn.setToolTip("Latih model AI dengan gambar OK/NG template ini")
        right_layout.addWidget(self._train_btn)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("")
        right_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setObjectName("secondaryText")
        right_layout.addWidget(self._progress_label)

        right_layout.addSpacing(12)

        # Threshold controls
        threshold_group = QGroupBox("🎚️ " + self._tr.tr("teach_threshold"))
        thresh_layout = QVBoxLayout(threshold_group)

        self._threshold_slider = QSlider(Qt.Horizontal)
        self._threshold_slider.setRange(0, 1000)
        self._threshold_slider.setValue(500)
        self._threshold_slider.setToolTip("Geser untuk mengatur sensitivitas deteksi anomaly")
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        thresh_layout.addWidget(self._threshold_slider)

        thresh_value_layout = QHBoxLayout()
        thresh_value_layout.addWidget(QLabel("Sensitif (0.0)"))
        self._threshold_value_label = QLabel("0.500")
        self._threshold_value_label.setAlignment(Qt.AlignCenter)
        self._threshold_value_label.setObjectName("bigCounter")
        thresh_value_layout.addWidget(self._threshold_value_label)
        thresh_value_layout.addWidget(QLabel("Selektif (1.0)"))
        thresh_layout.addLayout(thresh_value_layout)

        thresh_help = QLabel("⬅ Lebih longgar (lebih sedikit NG) | Lebih ketat ➡")
        thresh_help.setObjectName("secondaryText")
        thresh_help.setAlignment(Qt.AlignCenter)
        thresh_layout.addWidget(thresh_help)

        right_layout.addWidget(threshold_group)

        # Histogram area (real widget)
        histogram_group = QGroupBox("📊 " + self._tr.tr("teach_histogram"))
        hist_layout = QVBoxLayout(histogram_group)
        self._histogram = HistogramWidget()
        self._histogram.setMinimumHeight(120)
        hist_layout.addWidget(self._histogram)
        right_layout.addWidget(histogram_group)

        right_layout.addStretch()

        # Warning
        self._warning_label = QLabel("")
        self._warning_label.setObjectName("secondaryText")
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet("color: #F59E0B;")
        right_layout.addWidget(self._warning_label)

        layout.addWidget(right_panel, 1)

    # ---- Gallery thumbnails ----

    def add_ok_thumbnail(self, pixmap: QPixmap, path: str = ""):
        thumb = ThumbnailWidget(pixmap, path, "#22C55E")
        thumb.deleted.connect(lambda p: self._on_delete_image(p, "ok"))
        self._ok_gallery_layout.insertWidget(self._ok_gallery_layout.count() - 1, thumb)

    def add_ng_thumbnail(self, pixmap: QPixmap, path: str = ""):
        thumb = ThumbnailWidget(pixmap, path, "#EF4444")
        thumb.deleted.connect(lambda p: self._on_delete_image(p, "ng"))
        self._ng_gallery_layout.insertWidget(self._ng_gallery_layout.count() - 1, thumb)

    def _on_delete_image(self, path: str, label: str):
        """Delete image file and refresh gallery."""
        import os
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Delete error: {e}")
        # Signal parent to refresh
        self.image_deleted.emit(label)

    def clear_galleries(self):
        """Clear all thumbnails."""
        while self._ok_gallery_layout.count() > 1:
            item = self._ok_gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        while self._ng_gallery_layout.count() > 1:
            item = self._ng_gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # ---- Public API slots ----

    @Slot()
    def set_ok_count(self, count: int):
        self._ok_count_label.setText(f"✅ {self._tr.tr('teach_count_ok', count=count)}")
        self._train_btn.setEnabled(count > 0)

    @Slot()
    def set_ng_count(self, count: int):
        self._ng_count_label.setText(f"❌ {self._tr.tr('teach_count_ng', count=count)}")

    @Slot()
    def set_preview(self, pixmap: QPixmap):
        """Update ROI editor with camera frame."""
        self._roi_editor.set_pixmap(pixmap)

    @Slot()
    def set_preview_text(self, text: str):
        """Clear preview (pasang text saat kamera mati)."""
        if text:
            # ROI editor akan menampilkan text sendiri saat pixmap null
            self._roi_editor.set_pixmap(QPixmap())

    @Slot()
    def set_version(self, version: int):
        self._version_label.setText(f"💾 Model: v{version}" if version else "💾 Model: —")

    @Slot()
    def set_training_progress(self, percent: int, message: str = ""):
        self._progress_bar.setValue(percent)
        self._progress_bar.setFormat(f"{percent}%")
        self._progress_label.setText(message or f"Training: {percent}%")

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
        """Set threshold from external (e.g. after training)."""
        self._threshold_value_label.setText(f"{value:.3f}")
        self._threshold_slider.blockSignals(True)
        self._threshold_slider.setValue(int(value * 1000))
        self._threshold_slider.blockSignals(False)

    def _on_threshold_changed(self, value: int):
        """Slider moved — update label."""
        val = value / 1000.0
        self._threshold_value_label.setText(f"{val:.3f}")

    @Slot()
    def set_warning(self, message: str):
        self._warning_label.setText(message)

    @Slot()
    def set_histogram_data(self, ok_scores: list, ng_scores: list, threshold: float = 0.5):
        """Update histogram widget with real score data."""
        self._histogram.set_data(ok_scores, ng_scores, threshold)

    @Slot()
    def clear_histogram(self):
        self._histogram.clear_data()

    # ---- Import Review Mode ----

    @Slot()
    def show_import_mode(self, active: bool):
        """Toggle import review UI: show/hide status bar, adjust capture buttons."""
        if active:
            self._capture_ok_btn.setText("✅ Simpan OK")
            self._capture_ok_btn.setToolTip("Simpan gambar import ini sebagai OK → lanjut")
            self._capture_ng_btn.setText("❌ Simpan NG")
            self._capture_ng_btn.setToolTip("Simpan gambar import ini sebagai NG → lanjut")
            self._import_btn.hide()
            self._import_status_label.show()
        else:
            self._capture_ok_btn.setText("✅ Capture OK")
            self._capture_ok_btn.setToolTip("Ambil gambar dari live view untuk sampel OK")
            self._capture_ng_btn.setText("❌ Capture NG")
            self._capture_ng_btn.setToolTip("Ambil gambar dari live view untuk sampel NG")
            self._import_btn.show()
            self._import_status_label.hide()

    @Slot()
    def set_import_status(self, current: int, total: int):
        """Update import progress label, e.g. '📁 Import: 3/12'."""
        self._import_status_label.setText(f"📁 Import: {current}/{total} — pilih OK atau NG")

    # ---- Widget accessors ----

    def get_capture_ok_button(self) -> QPushButton:
        return self._capture_ok_btn

    def get_capture_ng_button(self) -> QPushButton:
        return self._capture_ng_btn

    def get_import_button(self) -> QPushButton:
        return self._import_btn

    def get_train_button(self) -> QPushButton:
        return self._train_btn

    def get_threshold_slider(self) -> QSlider:
        return self._threshold_slider

    def get_progress_bar(self) -> QProgressBar:
        return self._progress_bar

    def get_template_combo(self) -> QComboBox:
        return self._template_combo

    def get_add_template_button(self) -> QPushButton:
        return self._add_template_btn

    def get_clear_button(self) -> QPushButton:
        return self._clear_btn

    # ---- ROI Controls ----

    def get_roi_editor(self) -> ROIEditor:
        return self._roi_editor

    def get_roi_panel(self) -> ROIListPanel:
        return self._roi_panel
