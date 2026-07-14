"""
VisionInspect - Settings Page
Pengaturan kamera, ROI, PLC, model, retensi, Flask API.
"""

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from visioninspect.utils.i18n import Translator


class SettingsPage(QWidget):
    """Halaman SETTINGS — semua pengaturan aplikasi."""

    def __init__(self, translator: Translator, parent=None):
        super().__init__(parent)
        self._tr = translator
        self._setup_ui()

    def _setup_ui(self):
        # Scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        title = QLabel(self._tr.tr("settings_title"))
        title.setObjectName("sectionTitle")
        main_layout.addWidget(title)

        # === Camera Settings ===
        cam_group = QGroupBox(self._tr.tr("settings_camera"))
        cam_layout = QVBoxLayout(cam_group)

        def _add_combo_row(parent_layout, label, items, default_idx=0):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            combo = QComboBox()
            combo.addItems(items)
            combo.setCurrentIndex(default_idx)
            row.addWidget(combo)
            row.addStretch()
            parent_layout.addLayout(row)
            return combo

        def _add_spin_row(parent_layout, label, min_v, max_v, default_v):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QSpinBox()
            spin.setRange(min_v, max_v)
            spin.setValue(default_v)
            row.addWidget(spin)
            row.addStretch()
            parent_layout.addLayout(row)
            return spin

        self._cam_device = _add_spin_row(cam_layout, "Device Index", 0, 10, 0)
        self._cam_width = _add_spin_row(cam_layout, "Resolution Width", 320, 4096, 1920)
        self._cam_height = _add_spin_row(cam_layout, "Resolution Height", 240, 3072, 1080)
        self._cam_fps = _add_spin_row(cam_layout, "Target FPS", 1, 120, 30)
        self._cam_exposure = _add_spin_row(cam_layout, "Exposure (-1=auto)", -1, 100000, -1)

        main_layout.addWidget(cam_group)

        # === ROI Settings ===
        roi_group = QGroupBox(self._tr.tr("settings_roi"))
        roi_layout = QVBoxLayout(roi_group)

        self._roi_enabled = QCheckBox(self._tr.tr("settings_roi"))
        self._roi_enabled.setChecked(True)
        roi_layout.addWidget(self._roi_enabled)

        roi_coords = QHBoxLayout()
        roi_coords.addWidget(QLabel("X, Y:"))
        self._roi_x = QSpinBox()
        self._roi_x.setRange(0, 10000)
        roi_coords.addWidget(self._roi_x)
        self._roi_y = QSpinBox()
        self._roi_y.setRange(0, 10000)
        roi_coords.addWidget(self._roi_y)
        roi_coords.addWidget(QLabel("W, H:"))
        self._roi_w = QSpinBox()
        self._roi_w.setRange(16, 4096)
        self._roi_w.setValue(256)
        roi_coords.addWidget(self._roi_w)
        self._roi_h = QSpinBox()
        self._roi_h.setRange(16, 4096)
        self._roi_h.setValue(256)
        roi_coords.addWidget(self._roi_h)
        roi_coords.addStretch()
        roi_layout.addLayout(roi_coords)

        main_layout.addWidget(roi_group)

        # === PLC Settings ===
        plc_group = QGroupBox(self._tr.tr("settings_plc"))
        plc_layout = QVBoxLayout(plc_group)

        self._plc_enabled = QCheckBox("Enable PLC")
        plc_layout.addWidget(self._plc_enabled)

        plc_mode_row = QHBoxLayout()
        plc_mode_row.addWidget(QLabel("Mode:"))
        self._plc_mode = QComboBox()
        self._plc_mode.addItems([self._tr.tr("plc_rs232"), self._tr.tr("plc_rs485")])
        plc_mode_row.addWidget(self._plc_mode)

        plc_mode_row.addWidget(QLabel("Protocol:"))
        self._plc_protocol = QComboBox()
        self._plc_protocol.addItems([self._tr.tr("plc_modbus"), self._tr.tr("plc_ascii")])
        plc_mode_row.addWidget(self._plc_protocol)
        plc_mode_row.addStretch()
        plc_layout.addLayout(plc_mode_row)

        self._plc_port = QLineEdit("COM1")
        plc_layout.addWidget(QLabel("Port:"))
        plc_layout.addWidget(self._plc_port)

        plc_params = QHBoxLayout()
        plc_params.addWidget(QLabel("Baudrate:"))
        self._plc_baud = QComboBox()
        self._plc_baud.addItems(["9600", "19200", "38400", "57600", "115200"])
        plc_params.addWidget(self._plc_baud)
        plc_params.addWidget(QLabel("Parity:"))
        self._plc_parity = QComboBox()
        self._plc_parity.addItems(["N", "E", "O"])
        plc_params.addWidget(self._plc_parity)
        plc_params.addStretch()
        plc_layout.addLayout(plc_params)

        main_layout.addWidget(plc_group)

        # === Model Settings ===
        model_group = QGroupBox(self._tr.tr("settings_model"))
        model_layout = QVBoxLayout(model_group)

        self._model_algo = _add_combo_row(model_layout, "Algorithm:", ["PatchCore", "EfficientAd"])
        self._model_backbone = _add_combo_row(model_layout, "Backbone:", ["resnet18", "wide_resnet50_2"])
        self._model_input_size = _add_spin_row(model_layout, "Input Size:", 64, 512, 256)

        main_layout.addWidget(model_group)

        # === History / Retention ===
        hist_group = QGroupBox(self._tr.tr("settings_history"))
        hist_layout = QVBoxLayout(hist_group)

        self._retention_days = _add_spin_row(hist_layout, "Auto-purge (days):", 0, 365, 30)
        self._max_entries = _add_spin_row(hist_layout, "Max history entries:", 100, 100000, 10000)
        self._ok_sample_pct = _add_spin_row(hist_layout, "Save OK sample (%):", 0, 100, 10)

        main_layout.addWidget(hist_group)

        # === Flask API ===
        flask_group = QGroupBox(self._tr.tr("settings_flask"))
        flask_layout = QVBoxLayout(flask_group)

        self._flask_enabled = QCheckBox("Enable Flask API")
        flask_layout.addWidget(self._flask_enabled)

        self._flask_port = _add_spin_row(flask_layout, "Port:", 1024, 65535, 5000)

        main_layout.addWidget(flask_group)

        # === Language ===
        lang_group = QGroupBox(self._tr.tr("settings_language"))
        lang_layout = QHBoxLayout(lang_group)
        self._lang_combo = QComboBox()
        self._lang_combo.addItems(["Bahasa Indonesia", "English"])
        lang_layout.addWidget(self._lang_combo)
        lang_layout.addStretch()
        main_layout.addWidget(lang_group)

        main_layout.addStretch()

        # Save button
        self._save_btn = QPushButton(self._tr.tr("settings_save"))
        self._save_btn.setObjectName("primaryButton")
        self._save_btn.setMinimumHeight(40)
        main_layout.addWidget(self._save_btn)

        scroll.setWidget(scroll_content)

        # Main layout
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    # ---- Public API ----

    @Slot()
    def get_settings_dict(self) -> dict:
        """Return dict of current settings values."""
        return {
            "camera": {
                "device_index": self._cam_device.value(),
                "resolution_width": self._cam_width.value(),
                "resolution_height": self._cam_height.value(),
                "fps_target": self._cam_fps.value(),
                "exposure": self._cam_exposure.value(),
            },
            "roi": {
                "enabled": self._roi_enabled.isChecked(),
                "x": self._roi_x.value(),
                "y": self._roi_y.value(),
                "width": self._roi_w.value(),
                "height": self._roi_h.value(),
            },
            "plc": {
                "enabled": self._plc_enabled.isChecked(),
                "mode": "rs232" if self._plc_mode.currentIndex() == 0 else "rs485",
                "protocol": "modbus" if self._plc_protocol.currentIndex() == 0 else "ascii",
                "port": self._plc_port.text(),
                "baudrate": int(self._plc_baud.currentText()),
                "parity": self._plc_parity.currentText(),
            },
            "model": {
                "algorithm": self._model_algo.currentText().lower(),
                "backbone": self._model_backbone.currentText(),
                "input_size": self._model_input_size.value(),
            },
            "history": {
                "auto_purge_days": self._retention_days.value(),
                "max_history_entries": self._max_entries.value(),
                "save_ok_sample_percent": self._ok_sample_pct.value(),
            },
            "flask_api": {
                "enabled": self._flask_enabled.isChecked(),
                "port": self._flask_port.value(),
            },
            "language": "id" if self._lang_combo.currentIndex() == 0 else "en",
        }

    def get_save_button(self) -> QPushButton:
        return self._save_btn
