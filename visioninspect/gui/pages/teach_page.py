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

    # Preset backbone+coreset combos for PatchCore — pre-tested combinations so
    # most users never have to reason about raw hyperparameters. "custom" is a
    # separate, implicit 4th option (not listed here) for hand-tuned values via
    # Mode Lanjutan.
    TRAINING_PROFILES = {
        "fast": {"label": "⚡ Cepat", "backbone": "resnet18",
                 "coreset_sampling_ratio": 0.1},
        "balanced": {"label": "⚖️ Seimbang", "backbone": "resnet18",
                     "coreset_sampling_ratio": 0.25},
        "accurate": {"label": "🎯 Detail Tinggi", "backbone": "wide_resnet50_2",
                     "coreset_sampling_ratio": 0.25},
    }

    image_deleted = Signal(str)
    thumbnail_clicked = Signal(str)
    import_cancelled = Signal()
    training_config_changed = Signal()
    augmentation_config_changed = Signal()

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

        self._test_model_btn = QPushButton("🧪 Uji Model")
        self._test_model_btn.setMinimumHeight(38)
        self._test_model_btn.setToolTip(
            "Uji model terhadap batch foto statis dari disk — sanity check "
            "tanpa kamera live, hasil tidak disimpan ke riwayat inspeksi.")
        capture_row.addWidget(self._test_model_btn)
        left_layout.addLayout(capture_row)

        # Import status — baris dengan progress + cancel
        import_row = QHBoxLayout()
        import_row.setSpacing(6)

        self._import_status_label = QLabel("")
        self._import_status_label.setObjectName("secondaryText")
        self._import_status_label.setAlignment(Qt.AlignCenter)
        self._import_status_label.setStyleSheet(
            "color: #F59E0B; font-weight: bold; padding: 4px; "
            "background-color: #1A2A44; border-radius: 4px;")
        self._import_status_label.hide()
        import_row.addWidget(self._import_status_label, 1)

        self._import_progress_bar = QProgressBar()
        self._import_progress_bar.setRange(0, 100)
        self._import_progress_bar.setValue(0)
        self._import_progress_bar.setTextVisible(True)
        self._import_progress_bar.setMaximumHeight(18)
        self._import_progress_bar.setFormat("")
        self._import_progress_bar.hide()
        import_row.addWidget(self._import_progress_bar)

        self._cancel_import_btn = QPushButton("✕ Batal")
        self._cancel_import_btn.setFixedWidth(70)
        self._cancel_import_btn.setFixedHeight(26)
        self._cancel_import_btn.setStyleSheet(
            "font-size: 11px; padding: 0 6px; border: 1px solid #EF4444;"
            " border-radius: 3px; background: #1A2A44; color: #EF4444;")
        self._cancel_import_btn.setToolTip("Batalkan import dan kembali ke mode normal")
        self._cancel_import_btn.hide()
        self._cancel_import_btn.clicked.connect(self.import_cancelled.emit)
        import_row.addWidget(self._cancel_import_btn)

        left_layout.addLayout(import_row)

        # ── Preview area: QC + Gate (top row) | Daftar ROI (bottom, full width) ──
        preview_outer = QVBoxLayout()
        preview_outer.setSpacing(8)

        # Top row: QC Region (left) | Gate Part 1 (right)
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # QC Region
        qc_box = QVBoxLayout()
        qc_box.setSpacing(2)
        qc_label = QLabel("🔍 QC Region")
        qc_label.setObjectName("secondaryText")
        qc_box.addWidget(qc_label)
        self._roi_editor = ROIEditor()
        self._roi_editor.setMinimumSize(240, 200)
        qc_box.addWidget(self._roi_editor, 1)
        top_row.addLayout(qc_box, 1)

        # Gate Part 1
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
        top_row.addLayout(gate_box, 1)

        preview_outer.addLayout(top_row, 3)

        # Bottom row: Daftar ROI (full width)
        rp_box = QVBoxLayout()
        rp_box.setSpacing(2)
        rp_label = QLabel("📍 Daftar ROI")
        rp_label.setObjectName("secondaryText")
        rp_box.addWidget(rp_label)
        self._roi_panel = ROIListPanel()
        self._roi_panel.setMinimumWidth(140)
        rp_box.addWidget(self._roi_panel, 1)
        preview_outer.addLayout(rp_box, 2)

        left_layout.addLayout(preview_outer, 1)

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

        # ── Training Profile ──
        # Preset-driven so most users never touch raw hyperparameters directly;
        # "Mode Lanjutan" reveals the underlying fields for full manual control.
        profile_group = QGroupBox("🧠 Training Profile")
        profile_outer = QVBoxLayout(profile_group)
        profile_outer.setSpacing(6)
        profile_outer.setContentsMargins(10, 8, 10, 10)

        self._profile_combo = QComboBox()
        for key, prof in self.TRAINING_PROFILES.items():
            self._profile_combo.addItem(prof["label"], key)
        self._profile_combo.addItem("🔧 Custom", "custom")
        self._profile_combo.setToolTip(
            "⚡ Cepat: resnet18, ringan & responsif — cocok kebanyakan part.\n"
            "⚖️ Seimbang: resnet18 dengan memory bank lebih lengkap.\n"
            "🎯 Detail Tinggi: wide_resnet50_2 — lebih akurat untuk detail halus, "
            "tapi ~3-4x lebih berat/lambat di CPU.\n"
            "🔧 Custom: atur manual lewat Mode Lanjutan."
        )
        self._profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        profile_outer.addWidget(self._profile_combo)

        self._advanced_cb = QCheckBox("🔧 Mode Lanjutan")
        self._advanced_cb.toggled.connect(self._on_advanced_toggled)
        profile_outer.addWidget(self._advanced_cb)

        self._advanced_widget = QWidget()
        self._adv_form = QFormLayout(self._advanced_widget)
        self._adv_form.setSpacing(6)
        self._adv_form.setLabelAlignment(Qt.AlignLeft)
        self._adv_form.setContentsMargins(0, 4, 0, 0)

        self._algo_combo = QComboBox()
        self._algo_combo.addItem("PatchCore", "patchcore")
        self._algo_combo.addItem("EfficientAd", "efficientad")
        self._algo_combo.currentIndexChanged.connect(self._on_advanced_field_changed)
        self._algo_combo.currentIndexChanged.connect(self._update_algorithm_field_visibility)
        self._adv_form.addRow("Algorithm:", self._algo_combo)

        # Backbone & Coreset Ratio cuma dipakai PatchCore (lihat training.py:
        # Patchcore(backbone=..., coreset_sampling_ratio=...) vs
        # EfficientAd() yang tidak menerima parameter itu sama sekali — nilai
        # yang tersimpan tetap ada di config tapi diabaikan total oleh
        # training). Disembunyikan untuk EfficientAd lewat setRowVisible,
        # sama seperti Epochs disembunyikan untuk PatchCore, biar tidak
        # menyesatkan seolah-olah field itu berlaku untuk kedua algorithm.
        self._backbone_combo = QComboBox()
        self._backbone_combo.addItems(["resnet18", "wide_resnet50_2"])
        self._backbone_combo.currentIndexChanged.connect(self._on_advanced_field_changed)
        self._adv_form.addRow("Backbone:", self._backbone_combo)

        self._coreset_spin = QDoubleSpinBox()
        self._coreset_spin.setRange(0.01, 1.0)
        self._coreset_spin.setSingleStep(0.05)
        self._coreset_spin.setDecimals(2)
        self._coreset_spin.setToolTip(
            "Porsi fitur OK yang disimpan sebagai memory bank. Lebih besar = "
            "lebih lengkap menangkap variasi (lighting, posisi) tapi lebih "
            "berat/lambat. Default 0.1. Hanya berlaku untuk PatchCore."
        )
        self._coreset_spin.valueChanged.connect(self._on_advanced_field_changed)
        self._adv_form.addRow("Coreset Ratio:", self._coreset_spin)

        # Epoch cuma bermakna untuk EfficientAd (network beneran di-training
        # via backprop) — disembunyikan untuk PatchCore (one-shot, selalu 1
        # epoch) lewat setRowVisible, bukan dihapus, biar gampang di-toggle.
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 1000)
        self._epochs_spin.setValue(100)
        self._epochs_spin.setToolTip(
            "Jumlah epoch training. Hanya berlaku untuk EfficientAd — "
            "PatchCore selalu 1 epoch (one-shot, tidak butuh iterasi)."
        )
        self._epochs_spin.valueChanged.connect(self._on_advanced_field_changed)
        self._adv_form.addRow("Epochs:", self._epochs_spin)

        self._advanced_widget.setVisible(False)
        profile_outer.addWidget(self._advanced_widget)
        self._update_algorithm_field_visibility()

        right_layout.addWidget(profile_group)

        # ── Augmentasi Data ──
        # Sengaja tidak menyediakan cutout/distorsi berat/blur berat — jenis
        # augmentasi itu bisa menyerupai defect asli (scratch/kontaminasi)
        # dan justru mengajari model salah. Cuma transformasi klasik yang
        # aman: rotasi, flip, translasi, brightness, contrast.
        aug_group = QGroupBox("🧬 Augmentasi Data")
        aug_outer = QVBoxLayout(aug_group)
        aug_outer.setSpacing(6)
        aug_outer.setContentsMargins(10, 8, 10, 10)

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Jumlah gambar per jenis:"))
        self._aug_count_spin = QSpinBox()
        self._aug_count_spin.setRange(1, 50)
        self._aug_count_spin.setValue(5)
        self._aug_count_spin.valueChanged.connect(self._on_augmentation_field_changed)
        count_row.addWidget(self._aug_count_spin)
        count_row.addStretch()
        aug_outer.addLayout(count_row)

        # Baris tanpa parameter rentang (flip) — cuma checkbox.
        self._aug_flip_h_cb = QCheckBox("Flip Horizontal")
        self._aug_flip_h_cb.toggled.connect(self._on_augmentation_field_changed)
        aug_outer.addWidget(self._aug_flip_h_cb)

        self._aug_flip_v_cb = QCheckBox("Flip Vertical")
        self._aug_flip_v_cb.toggled.connect(self._on_augmentation_field_changed)
        aug_outer.addWidget(self._aug_flip_v_cb)

        # Baris dengan parameter rentang: checkbox aktif + spin max + "Acak".
        self._aug_rotation_cb, self._aug_rotation_spin, self._aug_rotation_random_cb = \
            self._build_augmentation_range_row(
                aug_outer, "Rotasi", "°", 1, 180, 15)
        self._aug_translation_cb, self._aug_translation_spin, self._aug_translation_random_cb = \
            self._build_augmentation_range_row(
                aug_outer, "Translasi", "%", 1, 50, 10)
        self._aug_brightness_cb, self._aug_brightness_spin, self._aug_brightness_random_cb = \
            self._build_augmentation_range_row(
                aug_outer, "Brightness", "%", 1, 100, 20)
        self._aug_contrast_cb, self._aug_contrast_spin, self._aug_contrast_random_cb = \
            self._build_augmentation_range_row(
                aug_outer, "Contrast", "%", 1, 100, 20)

        self._aug_regenerate_btn = QPushButton("🔄 Regenerate")
        self._aug_regenerate_btn.setToolTip(
            "Paksa generate ulang augmentasi walau setting tidak berubah "
            "(misal cuma ingin nilai Acak yang baru).")
        aug_outer.addWidget(self._aug_regenerate_btn)

        right_layout.addWidget(aug_group)

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

    def _build_augmentation_range_row(self, parent_layout, label: str, unit: str,
                                       min_val: int, max_val: int, default_val: int):
        """Baris augmentasi bertipe rentang: checkbox aktif + spin max + 'Acak'.
        Qt spinbox tidak punya state kosong native, jadi 'kosong = Acak' di
        spek fitur direalisasikan lewat checkbox terpisah yang men-disable
        spin-nya — nilai tersimpan jadi None (bukan angka yang kebetulan
        masih ada di spinbox yang di-disable) saat Acak dicentang."""
        row = QHBoxLayout()
        row.setSpacing(6)

        enable_cb = QCheckBox(label)
        enable_cb.toggled.connect(self._on_augmentation_field_changed)
        row.addWidget(enable_cb)
        row.addStretch()

        row.addWidget(QLabel(f"Max ({unit}):"))
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default_val)
        spin.valueChanged.connect(self._on_augmentation_field_changed)
        row.addWidget(spin)

        random_cb = QCheckBox("Acak")
        random_cb.toggled.connect(lambda checked, s=spin: s.setEnabled(not checked))
        random_cb.toggled.connect(self._on_augmentation_field_changed)
        row.addWidget(random_cb)

        parent_layout.addLayout(row)
        return enable_cb, spin, random_cb

    # ---- Gallery ----

    def add_ok_thumbnail(self, pixmap, path=""):
        t = ThumbnailWidget(pixmap, path, "#22C55E")
        t.deleted.connect(lambda p: self._on_delete_image(p, "ok"))
        t.clicked.connect(self.thumbnail_clicked.emit)
        self._ok_gallery_layout.insertWidget(self._ok_gallery_layout.count() - 1, t)

    def add_ng_thumbnail(self, pixmap, path=""):
        t = ThumbnailWidget(pixmap, path, "#EF4444")
        t.deleted.connect(lambda p: self._on_delete_image(p, "ng"))
        t.clicked.connect(self.thumbnail_clicked.emit)
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
            self._import_progress_bar.show()
            self._import_progress_bar.setValue(0)
            self._cancel_import_btn.show()
        else:
            self._capture_ok_btn.setText("✅ Capture OK")
            self._capture_ng_btn.setText("❌ Capture NG")
            self._import_btn.show()
            self._import_status_label.hide()
            self._import_progress_bar.hide()
            self._cancel_import_btn.hide()
        self._test_model_btn.setEnabled(not active)

    @Slot()
    def set_import_status(self, current: int, total: int):
        self._import_status_label.setText(f"📁 Import: {current}/{total} — pilih OK atau NG")

    @Slot()
    def set_import_progress(self, value: int):
        """Set progress bar percentage (0-100)."""
        self._import_progress_bar.setValue(value)
        self._import_progress_bar.setFormat(f"{value}%")

    # ---- Part Check ----

    def set_part_check_config(self, cfg: dict):
        # Blokir sinyal selama load programatik. Tanpa ini, setChecked/setCurrentIndex
        # memicu slot penyimpan config (di main_window) yang membaca nilai spinbox yang
        # BELUM di-load → menimpa config di disk dgn default → setting kereset saat rerun.
        _widgets = [
            self._pc_enabled_cb, self._pc_method_combo,
            self._pc_color_th_spin, self._pc_edge_th_spin,
            self._pc_canny_low_spin, self._pc_canny_high_spin,
        ]
        for _w in _widgets:
            _w.blockSignals(True)
        try:
            self._pc_enabled_cb.setChecked(cfg.get("enabled", False))
            m = cfg.get("method", "both")
            idx = self._pc_method_combo.findData(m)
            if idx >= 0:
                self._pc_method_combo.setCurrentIndex(idx)
            self._pc_color_th_spin.setValue(cfg.get("color_threshold", 0.35))
            self._pc_edge_th_spin.setValue(cfg.get("edge_threshold", 0.5))
            self._pc_canny_low_spin.setValue(cfg.get("canny_low", 50))
            self._pc_canny_high_spin.setValue(cfg.get("canny_high", 150))
        finally:
            for _w in _widgets:
                _w.blockSignals(False)
        # Sinyal diblokir di atas, jadi update visibilitas (biasanya dipicu
        # currentIndexChanged combo) dipanggil manual di sini.
        self._update_pc_field_visibility()

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

    # ---- Training Profile ----

    def get_training_config(self) -> dict:
        cfg = {
            "algorithm": self._algo_combo.currentData(),
            "backbone": self._backbone_combo.currentText(),
            "coreset_sampling_ratio": round(self._coreset_spin.value(), 2),
            "training_profile": self._profile_combo.currentData(),
        }
        # Cuma simpan override eksplisit untuk EfficientAd — PatchCore biar
        # tetap pakai default 1 epoch dari TrainingConfig, bukan nilai yang
        # kebetulan tersisa di spinbox yang sedang disembunyikan.
        if self._algo_combo.currentData() == "efficientad":
            cfg["max_epochs"] = self._epochs_spin.value()
        return cfg

    def set_training_config(self, cfg: dict):
        # Blokir sinyal selama load programatik — sama alasannya dengan
        # set_part_check_config di atas (hindari re-trigger handler penyimpan
        # sebelum semua field selesai di-load).
        _widgets = [self._profile_combo, self._algo_combo,
                    self._backbone_combo, self._coreset_spin, self._epochs_spin]
        for w in _widgets:
            w.blockSignals(True)
        try:
            algorithm = cfg.get("algorithm", "patchcore")
            backbone = cfg.get("backbone", "resnet18")
            coreset = cfg.get("coreset_sampling_ratio", 0.1)

            idx = self._algo_combo.findData(algorithm)
            self._algo_combo.setCurrentIndex(idx if idx >= 0 else 0)
            idx = self._backbone_combo.findText(backbone)
            self._backbone_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self._coreset_spin.setValue(coreset)
            self._epochs_spin.setValue(cfg.get("max_epochs", 100))

            profile = cfg.get("training_profile") or self._infer_profile(
                algorithm, backbone, coreset)
            idx = self._profile_combo.findData(profile)
            self._profile_combo.setCurrentIndex(
                idx if idx >= 0 else self._profile_combo.findData("custom"))
        finally:
            for w in _widgets:
                w.blockSignals(False)
        self._update_algorithm_field_visibility()

    def _infer_profile(self, algorithm: str, backbone: str, coreset_ratio: float) -> str:
        """Guess which preset (if any) matches values loaded from an older
        template config that predates the training_profile field."""
        if algorithm == "patchcore":
            for key, prof in self.TRAINING_PROFILES.items():
                if (prof["backbone"] == backbone
                        and abs(prof["coreset_sampling_ratio"] - coreset_ratio) < 1e-6):
                    return key
        return "custom"

    def _on_profile_selected(self, _index: int):
        key = self._profile_combo.currentData()
        prof = self.TRAINING_PROFILES.get(key)
        if not prof:
            return  # "custom" selected — leave advanced fields as-is
        _widgets = [self._algo_combo, self._backbone_combo, self._coreset_spin]
        for w in _widgets:
            w.blockSignals(True)
        try:
            self._algo_combo.setCurrentIndex(self._algo_combo.findData("patchcore"))
            idx = self._backbone_combo.findText(prof["backbone"])
            if idx >= 0:
                self._backbone_combo.setCurrentIndex(idx)
            self._coreset_spin.setValue(prof["coreset_sampling_ratio"])
        finally:
            for w in _widgets:
                w.blockSignals(False)
        self._update_algorithm_field_visibility()
        self.training_config_changed.emit()

    def _on_advanced_field_changed(self, *_):
        """User hand-edited a raw field — the current selection no longer
        matches any preset, so reflect that in the profile combo."""
        self._profile_combo.blockSignals(True)
        idx = self._profile_combo.findData("custom")
        if idx >= 0:
            self._profile_combo.setCurrentIndex(idx)
        self._profile_combo.blockSignals(False)
        self.training_config_changed.emit()

    def _on_advanced_toggled(self, checked: bool):
        self._advanced_widget.setVisible(checked)

    def _update_algorithm_field_visibility(self, *_):
        """Tampilkan cuma field yang benar-benar dipakai algorithm terpilih:
        Backbone & Coreset Ratio khusus PatchCore, Epochs khusus EfficientAd."""
        is_efficientad = self._algo_combo.currentData() == "efficientad"
        self._adv_form.setRowVisible(self._epochs_spin, is_efficientad)
        self._adv_form.setRowVisible(self._backbone_combo, not is_efficientad)
        self._adv_form.setRowVisible(self._coreset_spin, not is_efficientad)

    # ---- Augmentasi Data ----

    def get_augmentation_config(self) -> dict:
        def _range_val(spin: QSpinBox, random_cb: QCheckBox):
            return None if random_cb.isChecked() else spin.value()

        return {
            "count_per_type": self._aug_count_spin.value(),
            "rotation": {"enabled": self._aug_rotation_cb.isChecked(),
                         "max_degrees": _range_val(self._aug_rotation_spin,
                                                    self._aug_rotation_random_cb)},
            "flip_horizontal": {"enabled": self._aug_flip_h_cb.isChecked()},
            "flip_vertical": {"enabled": self._aug_flip_v_cb.isChecked()},
            "translation": {"enabled": self._aug_translation_cb.isChecked(),
                             "max_percent": _range_val(self._aug_translation_spin,
                                                        self._aug_translation_random_cb)},
            "brightness": {"enabled": self._aug_brightness_cb.isChecked(),
                           "max_percent": _range_val(self._aug_brightness_spin,
                                                      self._aug_brightness_random_cb)},
            "contrast": {"enabled": self._aug_contrast_cb.isChecked(),
                         "max_percent": _range_val(self._aug_contrast_spin,
                                                    self._aug_contrast_random_cb)},
        }

    def set_augmentation_config(self, cfg: dict):
        _widgets = [
            self._aug_count_spin, self._aug_flip_h_cb, self._aug_flip_v_cb,
            self._aug_rotation_cb, self._aug_rotation_spin, self._aug_rotation_random_cb,
            self._aug_translation_cb, self._aug_translation_spin, self._aug_translation_random_cb,
            self._aug_brightness_cb, self._aug_brightness_spin, self._aug_brightness_random_cb,
            self._aug_contrast_cb, self._aug_contrast_spin, self._aug_contrast_random_cb,
        ]
        for w in _widgets:
            w.blockSignals(True)
        try:
            self._aug_count_spin.setValue(cfg.get("count_per_type", 5))
            self._aug_flip_h_cb.setChecked(cfg.get("flip_horizontal", {}).get("enabled", False))
            self._aug_flip_v_cb.setChecked(cfg.get("flip_vertical", {}).get("enabled", False))

            def _load_range(enable_cb, spin, random_cb, sub_cfg, key, default):
                enable_cb.setChecked(sub_cfg.get("enabled", False))
                val = sub_cfg.get(key)
                random_cb.setChecked(val is None)
                spin.setValue(val if val is not None else default)
                spin.setEnabled(val is not None)

            _load_range(self._aug_rotation_cb, self._aug_rotation_spin,
                        self._aug_rotation_random_cb,
                        cfg.get("rotation", {}), "max_degrees", 15)
            _load_range(self._aug_translation_cb, self._aug_translation_spin,
                        self._aug_translation_random_cb,
                        cfg.get("translation", {}), "max_percent", 10)
            _load_range(self._aug_brightness_cb, self._aug_brightness_spin,
                        self._aug_brightness_random_cb,
                        cfg.get("brightness", {}), "max_percent", 20)
            _load_range(self._aug_contrast_cb, self._aug_contrast_spin,
                        self._aug_contrast_random_cb,
                        cfg.get("contrast", {}), "max_percent", 20)
        finally:
            for w in _widgets:
                w.blockSignals(False)

    def _on_augmentation_field_changed(self, *_):
        self.augmentation_config_changed.emit()

    def get_augmentation_regenerate_button(self) -> QPushButton:
        return self._aug_regenerate_btn

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
    def get_test_model_button(self): return self._test_model_btn
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
