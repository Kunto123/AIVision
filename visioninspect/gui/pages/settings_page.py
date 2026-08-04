"""
VisionInspect - Settings Page
Pengaturan kamera, ROI, PLC, model, retensi, Flask API.
"""

from PySide6.QtCore import Slot
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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

    def __init__(self, translator: Translator, config, parent=None):
        super().__init__(parent)
        self._tr = translator
        self._config = config
        self._setup_ui()
        self._load_settings()

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

        self._plc_status_label = QLabel("Tidak aktif")
        self._plc_status_label.setStyleSheet(
            "font-weight: bold; padding: 2px 8px; border-radius: 3px; "
            "color: #9FB3C8; background-color: #1A2A44;")
        plc_layout.addWidget(self._plc_status_label)

        plc_mode_row = QHBoxLayout()
        plc_mode_row.addWidget(QLabel("Mode:"))
        self._plc_mode = QComboBox()
        self._plc_mode.addItems([self._tr.tr("plc_rs232"), self._tr.tr("plc_rs485")])
        plc_mode_row.addWidget(self._plc_mode)

        plc_mode_row.addWidget(QLabel("Protocol:"))
        self._plc_protocol = QComboBox()
        self._plc_protocol.addItems([self._tr.tr("plc_modbus"), self._tr.tr("plc_ascii")])
        # ASCII deprecated — hanya Modbus RTU yang didukung saat ini
        _proto_model = self._plc_protocol.model()
        if isinstance(_proto_model, QStandardItemModel):
            _item = _proto_model.item(1)
            if _item is not None:
                _item.setEnabled(False)
        self._plc_protocol.setToolTip(
            "Protocol ASCII tidak lagi didukung — hanya Modbus RTU.")
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

        plc_addr_row = QHBoxLayout()
        plc_addr_row.addWidget(QLabel("Slave ID:"))
        self._plc_slave_id = QSpinBox()
        self._plc_slave_id.setRange(1, 247)
        self._plc_slave_id.setValue(1)
        self._plc_slave_id.setToolTip(
            "Modbus slave ID (device address) PLC — default 1 untuk Mitsubishi.")
        plc_addr_row.addWidget(self._plc_slave_id)
        plc_addr_row.addWidget(QLabel("Scan range:"))
        self._plc_scan_range = QSpinBox()
        self._plc_scan_range.setRange(0, 9999)
        self._plc_scan_range.setValue(127)
        self._plc_scan_range.setToolTip(
            "Batas atas probe alamat coil saat Scan/Deteksi Aktif.")
        plc_addr_row.addWidget(self._plc_scan_range)
        plc_addr_row.addStretch()
        plc_layout.addLayout(plc_addr_row)

        # --- IO Mapping (Modbus) — alamat bisa diganti tanpa edit kode ---
        io_hint = QLabel(
            "IO Mapping (Modbus RTU) — coil yang sistem tulis/ baca.\n"
            "Klik 'Scan Coils' untuk cari alamat valid, atau ketik manual.")
        io_hint.setWordWrap(True)
        io_hint.setObjectName("secondaryText")
        plc_layout.addWidget(io_hint)

        self._plc_pulse_ms = _add_spin_row(plc_layout, "Pulse durasi (ms):", 0, 5000, 300)

        self._io_combos: dict = {}
        io_rows = [
            ("result_ok",     "Output OK coil:",         1),
            ("result_ng",     "Output NG coil:",         2),
            ("part_ready",    "Output Part Ready coil:", 3),
            ("busy",          "Output Busy coil:",       4),
            ("trigger",       "Input Trigger coil:",     0),
            ("reset_result",  "Input Reset coil:",       5),
            ("switch_program","Input Ganti Prog coil:",  6),
        ]
        for key, label, default_addr in io_rows:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            combo = QComboBox()
            combo.setEditable(True)
            combo.setCurrentText(str(default_addr))
            row.addWidget(combo)
            row.addStretch()
            plc_layout.addLayout(row)
            self._io_combos[key] = combo

        self._plc_program_register = _add_spin_row(
            plc_layout, "Program Register:", 0, 9999, 10)

        scan_row = QHBoxLayout()
        self._scan_btn = QPushButton("Scan Coils...")
        self._scan_btn.setMinimumHeight(32)
        self._detect_btn = QPushButton("Deteksi Aktif")
        self._detect_btn.setMinimumHeight(32)
        scan_row.addWidget(self._scan_btn)
        scan_row.addWidget(self._detect_btn)
        scan_row.addStretch()
        plc_layout.addLayout(scan_row)
        self._scan_result_label = QLabel("")
        self._scan_result_label.setObjectName("secondaryText")
        self._scan_result_label.setWordWrap(True)
        plc_layout.addWidget(self._scan_result_label)

        main_layout.addWidget(plc_group)

        # === Inference Settings ===
        infer_group = QGroupBox("Inference")
        infer_layout = QVBoxLayout(infer_group)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Start Cycle Mode:"))
        self._inference_mode = QComboBox()
        self._inference_mode.addItems(
            ["Auto Sequence (jalan terus)", "PLC Trigger", "Start Cycle (tombol)"])
        self._inference_mode.setToolTip(
            "Auto Sequence: inspeksi tiap frame tanpa trigger (jalan terus).\n"
            "PLC Trigger: inspeksi hanya saat coil trigger PLC ON — timing "
            "antar part di ladder PLC (filosofi Keyence IV3).\n"
            "Start Cycle: inspeksi hanya saat tombol Start Cycle ditekan.")
        mode_row.addWidget(self._inference_mode)
        mode_row.addStretch()
        infer_layout.addLayout(mode_row)

        mode_help = QLabel(
            "PLC Trigger membutuhkan PLC aktif (Settings → PLC → Enable PLC).")
        mode_help.setObjectName("secondaryText")
        mode_help.setWordWrap(True)
        infer_layout.addWidget(mode_help)

        main_layout.addWidget(infer_group)

        # === YOLO Class Filter (opsional) ===
        # Form lengkap hanya muncul saat checkbox diaktifkan ("setting lengkap
        # namun hanya muncul ketika dipilih model yolo").
        yolo_group = QGroupBox("Filter Kelas (YOLO)")
        yolo_layout = QVBoxLayout(yolo_group)

        self._yolo_enabled = QCheckBox("Aktifkan filter kelas YOLO")
        self._yolo_enabled.setToolTip(
            "Pre-filter: frame dicek dulu dengan model YOLO custom (.pt/.onnx).\n"
            "Kalau kelas yang diharapkan tidak terdeteksi → NG langsung, tanpa\n"
            "scoring anomali. Butuh: pip install ultralytics + model terlatih.")
        yolo_layout.addWidget(self._yolo_enabled)

        self._yolo_form = QWidget()
        yf = QVBoxLayout(self._yolo_form)
        yf.setContentsMargins(0, 0, 0, 0)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Model path (.pt/.onnx):"))
        self._yolo_model_path = QLineEdit("")
        self._yolo_model_path.setPlaceholderText("mis. data/yolo/best.pt")
        path_row.addWidget(self._yolo_model_path, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_yolo_browse)
        path_row.addWidget(browse_btn)
        yf.addLayout(path_row)

        yf.addWidget(QLabel("Kelas yang diizinkan (pisah koma):"))
        self._yolo_classes = QLineEdit("")
        self._yolo_classes.setPlaceholderText("mis. coca_cola_bottle, cap")
        self._yolo_classes.setToolTip(
            "Nama kelas harus sama persis dengan nama kelas di model YOLO.\n"
            "Kosong = tidak menyaring (semua kelas lolos filter).")
        yf.addWidget(self._yolo_classes)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Min confidence:"))
        self._yolo_min_conf = QDoubleSpinBox()
        self._yolo_min_conf.setRange(0.05, 0.95)
        self._yolo_min_conf.setSingleStep(0.05)
        self._yolo_min_conf.setDecimals(2)
        self._yolo_min_conf.setValue(0.25)
        conf_row.addWidget(self._yolo_min_conf)
        conf_row.addStretch()
        yf.addLayout(conf_row)

        yolo_layout.addWidget(self._yolo_form)
        self._yolo_enabled.toggled.connect(self._update_yolo_form_visibility)

        main_layout.addWidget(yolo_group)

        # === Model Settings ===
        model_group = QGroupBox(self._tr.tr("settings_model"))
        model_layout = QVBoxLayout(model_group)

        model_hint = QLabel(
            "Default untuk template baru saja — tidak mengubah template yang "
            "sudah ada. Untuk template yang sedang aktif, atur lewat "
            "Training Profile di tab TEACH.")
        model_hint.setWordWrap(True)
        model_hint.setObjectName("secondaryText")
        model_layout.addWidget(model_hint)

        self._model_algo = _add_combo_row(model_layout, "Algorithm:", ["PatchCore", "EfficientAd"])
        self._model_backbone = _add_combo_row(model_layout, "Backbone:", ["resnet18", "wide_resnet50_2"])
        self._model_input_size = _add_spin_row(model_layout, "Input Size:", 64, 512, 256)

        # Inference runtime indicator
        runtime_row = QHBoxLayout()
        runtime_row.addWidget(QLabel("Inference Runtime:"))
        self._runtime_label = QLabel("—")
        self._runtime_label.setStyleSheet("font-weight: bold; padding: 2px 8px; border-radius: 3px;")
        runtime_row.addWidget(self._runtime_label)
        runtime_row.addStretch()
        model_layout.addLayout(runtime_row)

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

        self._flask_status_label = QLabel("Tidak aktif")
        self._flask_status_label.setStyleSheet(
            "font-weight: bold; padding: 2px 8px; border-radius: 3px; "
            "color: #9FB3C8; background-color: #1A2A44;")
        flask_layout.addWidget(self._flask_status_label)

        self._flask_port = _add_spin_row(flask_layout, "Port:", 1024, 65535, 5000)

        main_layout.addWidget(flask_group)

        # === PostgreSQL Settings ===
        pg_group = QGroupBox("PostgreSQL")
        pg_layout = QVBoxLayout(pg_group)

        self._pg_enabled = QCheckBox("Enable PostgreSQL")
        self._pg_enabled.setToolTip(
            "Aktifkan koneksi ke PostgreSQL untuk autentikasi dan push inspeksi.\n"
            "Nonaktifkan untuk tetap pakai SQLite lokal.")
        pg_layout.addWidget(self._pg_enabled)

        pg_host_row = QHBoxLayout()
        pg_host_row.addWidget(QLabel("Host:"))
        self._pg_host = QLineEdit("localhost")
        self._pg_host.setMinimumHeight(28)
        pg_host_row.addWidget(self._pg_host, 1)
        pg_host_row.addWidget(QLabel("Port:"))
        self._pg_port = QSpinBox()
        self._pg_port.setRange(1, 65535)
        self._pg_port.setValue(5432)
        self._pg_port.setFixedWidth(80)
        pg_host_row.addWidget(self._pg_port)
        pg_layout.addLayout(pg_host_row)

        dbname_row = QHBoxLayout()
        dbname_row.addWidget(QLabel("Database:"))
        self._pg_dbname = QLineEdit("visioninspect")
        self._pg_dbname.setMinimumHeight(28)
        dbname_row.addWidget(self._pg_dbname, 1)
        pg_layout.addLayout(dbname_row)

        user_row = QHBoxLayout()
        user_row.addWidget(QLabel("User:"))
        self._pg_user = QLineEdit("postgres")
        self._pg_user.setMinimumHeight(28)
        user_row.addWidget(self._pg_user, 1)
        user_row.addWidget(QLabel("Password:"))
        self._pg_password = QLineEdit()
        self._pg_password.setEchoMode(QLineEdit.Password)
        self._pg_password.setMinimumHeight(28)
        user_row.addWidget(self._pg_password, 1)
        pg_layout.addLayout(user_row)

        # Connection status + test button
        status_row = QHBoxLayout()
        self._pg_status_label = QLabel("Tidak aktif")
        self._pg_status_label.setStyleSheet(
            "font-weight: bold; padding: 2px 8px; border-radius: 3px; "
            "color: #9FB3C8; background-color: #1A2A44;")
        status_row.addWidget(self._pg_status_label, 1)

        self._pg_test_btn = QPushButton("Test Koneksi")
        self._pg_test_btn.setFixedHeight(28)
        self._pg_test_btn.setStyleSheet(
            "font-size: 11px; padding: 0 10px; border: 1px solid #233A57; "
            "border-radius: 3px; background: #1A2A44; color: #E2E8F0;")
        self._pg_test_btn.clicked.connect(self._on_test_pg_connection)
        status_row.addWidget(self._pg_test_btn)
        pg_layout.addLayout(status_row)

        pg_help = QLabel(
            "Password hash menggunakan SHA-256 + pepper (sama dengan SQLite).\n"
            "RFID UID di-hash sebelum disimpan. Aktifkan setelah config diisi.")
        pg_help.setObjectName("secondaryText")
        pg_help.setWordWrap(True)
        pg_layout.addWidget(pg_help)

        main_layout.addWidget(pg_group)

        # === NG Detection Settings ===
        ng_group = QGroupBox("NG Detection")
        ng_form = QVBoxLayout(ng_group)

        ng_delay_row = QHBoxLayout()
        ng_delay_row.addWidget(QLabel("NG Timeout:"))
        self._ng_delay_spin = QSpinBox()
        self._ng_delay_spin.setRange(0, 999999)
        self._ng_delay_spin.setValue(500)
        self._ng_delay_spin.setSuffix(" ms")
        self._ng_delay_spin.setSingleStep(100)
        self._ng_delay_spin.setFixedWidth(130)
        ng_delay_row.addWidget(self._ng_delay_spin)
        ng_delay_row.addStretch()
        ng_form.addLayout(ng_delay_row)

        ng_help = QLabel(
            "Jeda minimum sebelum NG dihitung.\n"
            "Timer mulai saat anomali terdeteksi.\n"
            "Setiap kali timer habis → NG +1, lalu timer reset.\n"
            "0 = instant (setiap frame NG dihitung).")
        ng_help.setObjectName("secondaryText")
        ng_help.setWordWrap(True)
        ng_form.addWidget(ng_help)

        main_layout.addWidget(ng_group)

        # === Cycle Delay ===
        cycle_group = QGroupBox("Cycle Delay")
        cycle_form = QVBoxLayout(cycle_group)

        cycle_delay_row = QHBoxLayout()
        cycle_delay_row.addWidget(QLabel("Jeda antar siklus:"))
        self._cycle_delay_spin = QSpinBox()
        self._cycle_delay_spin.setRange(0, 30000)
        self._cycle_delay_spin.setValue(1000)
        self._cycle_delay_spin.setSuffix(" ms")
        self._cycle_delay_spin.setSingleStep(100)
        self._cycle_delay_spin.setFixedWidth(130)
        cycle_delay_row.addWidget(self._cycle_delay_spin)
        cycle_delay_row.addStretch()
        cycle_form.addLayout(cycle_delay_row)

        cycle_help = QLabel(
            "Jeda setelah hasil inspeksi ditampilkan, sebelum siklus berikutnya dimulai.\n"
            "0 = langsung lanjut ke siklus berikutnya.\n"
            "Berguna untuk memberi waktu part diganti.")
        cycle_help.setObjectName("secondaryText")
        cycle_help.setWordWrap(True)
        cycle_form.addWidget(cycle_help)

        main_layout.addWidget(cycle_group)

        # === Logging Settings ===
        log_group = QGroupBox("Logging")
        log_layout = QVBoxLayout(log_group)
        self._show_debug_cb = QCheckBox("Tampilkan log debug di terminal")
        self._show_debug_cb.setToolTip(
            "Menampilkan log DEBUG (PC_EVAL, dll) di console.\n"
            "Matikan untuk produksi agar terminal tidak penuh.")
        log_layout.addWidget(self._show_debug_cb)
        main_layout.addWidget(log_group)

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

    def _on_yolo_browse(self):
        """Pilih file model YOLO (.pt/.onnx) lewat dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih model YOLO", "",
            "YOLO models (*.pt *.onnx);;All files (*)")
        if path:
            self._yolo_model_path.setText(path)

    def _update_yolo_form_visibility(self):
        """Form setting YOLO hanya tampil saat filter diaktifkan."""
        self._yolo_form.setVisible(self._yolo_enabled.isChecked())

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
                "modbus_slave_id": self._plc_slave_id.value(),
                "scan_range": self._plc_scan_range.value(),
                "pulse_ms": self._plc_pulse_ms.value(),
                "io_map": self.get_plc_io_map(),
            },
            "inference": {
                "mode": ["continuous", "plc_trigger", "manual"][self._inference_mode.currentIndex()],
                "cycle_delay_ms": self._cycle_delay_spin.value(),
            },
            "yolo": {
                "enabled": self._yolo_enabled.isChecked(),
                "model_path": self._yolo_model_path.text().strip(),
                "expected_classes": [
                    c.strip() for c in self._yolo_classes.text().split(",") if c.strip()],
                "min_conf": self._yolo_min_conf.value(),
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
            'flask_api': {
                'enabled': self._flask_enabled.isChecked(),
                'port': self._flask_port.value(),
            },
            'postgresql': {
                'enabled': self._pg_enabled.isChecked(),
                'host': self._pg_host.text(),
                'port': self._pg_port.value(),
                'dbname': self._pg_dbname.text(),
                'user': self._pg_user.text(),
                'password': self._pg_password.text(),
                'sslmode': 'prefer',
                'connect_timeout': 10,
            },
            "language": "id" if self._lang_combo.currentIndex() == 0 else "en",
            "ng_debounce_ms": self._ng_delay_spin.value(),
            "show_debug": self._show_debug_cb.isChecked(),
        }

    def get_save_button(self) -> QPushButton:
        return self._save_btn

    def get_camera_device_spin(self) -> QSpinBox:
        return self._cam_device

    def get_ng_debounce_ms(self) -> int:
        """Get current NG debounce delay from config."""
        return self._config.get("ng_debounce_ms", 500)

    def get_cycle_delay_ms(self) -> int:
        """Get cycle delay from config (ms). 0 = no delay."""
        return self._config.get("inference.cycle_delay_ms", 1000)

    def get_yolo_enabled_checkbox(self) -> QCheckBox:
        return self._yolo_enabled

    def get_yolo_form(self) -> QWidget:
        return self._yolo_form

    # ---- PLC IO Mapping API ----

    def _set_io_combo(self, key: str, addr) -> None:
        """Set combo address value (internal). Non-int → fallback 0."""
        combo = self._io_combos.get(key)
        if combo is None:
            return
        try:
            combo.setCurrentText(str(int(addr)))
        except (ValueError, TypeError):
            combo.setCurrentText("0")

    def _parse_io_addr(self, combo) -> int:
        """Parse combo text ke int alamat (aman: non-numeric → 0)."""
        text = combo.currentText().strip()
        try:
            return max(0, int(text))
        except ValueError:
            return 0

    def get_plc_io_map(self) -> dict:
        """Read current IO mapping from UI (untuk disimpan ke config)."""
        return {
            "outputs": {
                "result_ok": self._parse_io_addr(self._io_combos["result_ok"]),
                "result_ng": self._parse_io_addr(self._io_combos["result_ng"]),
                "part_ready": self._parse_io_addr(self._io_combos["part_ready"]),
                "busy": self._parse_io_addr(self._io_combos["busy"]),
            },
            "inputs": {
                "trigger": self._parse_io_addr(self._io_combos["trigger"]),
                "reset_result": self._parse_io_addr(self._io_combos["reset_result"]),
                "switch_program": self._parse_io_addr(self._io_combos["switch_program"]),
            },
            "program_register": self._plc_program_register.value(),
        }

    def get_plc_pulse_ms(self) -> int:
        return self._plc_pulse_ms.value()

    def get_scan_button(self) -> QPushButton:
        return self._scan_btn

    def get_detect_button(self) -> QPushButton:
        return self._detect_btn

    def set_scan_coil_options(self, valid_coils: list) -> None:
        """Tambahkan alamat coil valid hasil scan ke semua combo (tanpa
        mengubah pilihan yang sedang aktif)."""
        for combo in self._io_combos.values():
            current = combo.currentText()
            for addr in valid_coils:
                s = str(int(addr))
                if combo.findText(s) < 0:
                    combo.addItem(s)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def set_scan_result_label(self, text: str) -> None:
        self._scan_result_label.setText(text)

    def set_plc_status(self, connected: bool, detail: str = "") -> None:
        """Update PLC connection status indicator di group PLC."""
        if connected:
            self._plc_status_label.setText(f"Terhubung{(' — ' + detail) if detail else ''}")
            self._plc_status_label.setStyleSheet(
                "font-weight: bold; padding: 2px 8px; border-radius: 3px; "
                "color: #22C55E; background-color: #1A2A44;")
        elif detail == "Tidak diaktifkan":
            # PLC sengaja dimatikan — status netral, bukan error
            self._plc_status_label.setText(detail)
            self._plc_status_label.setStyleSheet(
                "font-weight: bold; padding: 2px 8px; border-radius: 3px; "
                "color: #9FB3C8; background-color: #1A2A44;")
        else:
            text = detail or "Tidak terhubung"
            self._plc_status_label.setText(f"Gagal: {text}")
            self._plc_status_label.setStyleSheet(
                "font-weight: bold; padding: 2px 8px; border-radius: 3px; "
                "color: #EF4444; background-color: #1A2A44;")

    def set_flask_status(self, running: bool, detail: str = "") -> None:
        """Update Flask API status indicator di group Flask."""
        if running:
            self._flask_status_label.setText(f"Jalan{(' — ' + detail) if detail else ''}")
            self._flask_status_label.setStyleSheet(
                "font-weight: bold; padding: 2px 8px; border-radius: 3px; "
                "color: #22C55E; background-color: #1A2A44;")
        else:
            text = detail or "Tidak aktif"
            self._flask_status_label.setText(text)
            self._flask_status_label.setStyleSheet(
                "font-weight: bold; padding: 2px 8px; border-radius: 3px; "
                "color: #9FB3C8; background-color: #1A2A44;")

    def get_inference_mode(self) -> str:
        """Get selected inference mode (continuous|plc_trigger|manual)."""
        return ["continuous", "plc_trigger", "manual"][self._inference_mode.currentIndex()]

    def get_program_register_spin(self) -> QSpinBox:
        return self._plc_program_register

    def set_runtime_status(self, has_openvino: bool, has_torch: bool,
                           active_runtime: str = "",
                           gpu_available: bool = False,
                           gpu_name: str = ""):
        """Update inference runtime indicator in Model settings."""
        parts = []
        color = "#9FB3C8"
        if has_openvino:
            parts.append("OpenVINO OK")
        else:
            parts.append("OpenVINO error")
        if has_torch:
            parts.append("PyTorch OK")
        else:
            parts.append("PyTorch error")
        text = " | ".join(parts)

        if active_runtime == "openvino":
            text += " | Active: OpenVINO"
            color = "#22C55E"
        elif active_runtime == "simple":
            text += " | Active: SimpleThreshold"
            color = "#F59E0B"
        elif active_runtime == "anomalib":
            text += " | Active: Anomalib"
            color = "#22C55E"

        # GPU indicator
        if gpu_available:
            gpu_label = gpu_name if gpu_name else "GPU"
            text += f" | GPU: {gpu_label}"
            color = "#22C55E"
        else:
            text += " | GPU: None (CPU)"

        self._runtime_label.setText(text)
        self._runtime_label.setStyleSheet(
            f"font-weight: bold; padding: 2px 8px; border-radius: 3px; color: {color};"
            f"background-color: #1A2A44;")

    # ---- PostgreSQL Status ----

    def set_pg_status(self, connected: bool, detail: str = ""):
        """Update PostgreSQL connection status indicator."""
        if connected:
            self._pg_status_label.setText(f"Terhubung{(' — ' + detail) if detail else ''}")
            self._pg_status_label.setStyleSheet(
                "font-weight: bold; padding: 2px 8px; border-radius: 3px; "
                "color: #22C55E; background-color: #1A2A44;")
        else:
            text = detail or "Tidak terhubung"
            self._pg_status_label.setText(f"Gagal: {text}")
            self._pg_status_label.setStyleSheet(
                "font-weight: bold; padding: 2px 8px; border-radius: 3px; "
                "color: #EF4444; background-color: #1A2A44;")

    def _on_test_pg_connection(self):
        """Test PostgreSQL connection with current form values."""
        self._pg_status_label.setText("Menguji koneksi...")
        self._pg_status_label.setStyleSheet(
            "font-weight: bold; padding: 2px 8px; border-radius: 3px; "
            "color: #F59E0B; background-color: #1A2A44;")

        cfg = {
            "enabled": self._pg_enabled.isChecked(),
            "host": self._pg_host.text(),
            "port": self._pg_port.value(),
            "dbname": self._pg_dbname.text(),
            "user": self._pg_user.text(),
            "password": self._pg_password.text(),
            "sslmode": "prefer",
            "connect_timeout": 5,
        }
        try:
            from visioninspect.storage.postgres_db import PostgresDB
            pg = PostgresDB(cfg)
            if not pg.is_enabled:
                self.set_pg_status(False, "PostgreSQL tidak diaktifkan")
                return
            # Quick connect test
            conn = pg._connect()
            conn.close()
            self.set_pg_status(True, cfg["host"])
        except Exception as e:
            self.set_pg_status(False, str(e).split(":")[-1].strip()[:60])

    def _load_settings(self) -> None:
        """Load settings from config into UI widgets."""
        # Camera
        self._cam_device.setValue(self._config.get("camera.device_index", 0))
        self._cam_width.setValue(self._config.get("camera.resolution_width", 1920))
        self._cam_height.setValue(self._config.get("camera.resolution_height", 1080))
        self._cam_fps.setValue(self._config.get("camera.fps_target", 30))
        self._cam_exposure.setValue(self._config.get("camera.exposure", -1))

        # ROI
        self._roi_enabled.setChecked(self._config.get("roi.enabled", True))
        self._roi_x.setValue(self._config.get("roi.x", 0))
        self._roi_y.setValue(self._config.get("roi.y", 0))
        self._roi_w.setValue(self._config.get("roi.width", 256))
        self._roi_h.setValue(self._config.get("roi.height", 256))

        # PLC
        self._plc_enabled.setChecked(self._config.get("plc.enabled", False))
        plc_mode = self._config.get("plc.mode", "rs232")
        self._plc_mode.setCurrentIndex(0 if plc_mode == "rs232" else 1)
        plc_protocol = self._config.get("plc.protocol", "modbus")
        self._plc_protocol.setCurrentIndex(0 if plc_protocol == "modbus" else 1)
        self._plc_port.setText(self._config.get("plc.port", "COM1"))
        baudrate = str(self._config.get("plc.baudrate", 9600))
        idx = self._plc_baud.findText(baudrate)
        if idx >= 0:
            self._plc_baud.setCurrentIndex(idx)
        parity = self._config.get("plc.parity", "N")
        idx = self._plc_parity.findText(parity)
        if idx >= 0:
            self._plc_parity.setCurrentIndex(idx)

        # PLC IO Mapping
        self._plc_pulse_ms.setValue(self._config.get("plc.pulse_ms", 300))
        io_map = self._config.get("plc.io_map", {})
        if not isinstance(io_map, dict):
            io_map = {}
        outputs = io_map.get("outputs", {})
        inputs = io_map.get("inputs", {})
        if not isinstance(outputs, dict):
            outputs = {}
        if not isinstance(inputs, dict):
            inputs = {}
        self._set_io_combo("result_ok", outputs.get("result_ok", 1))
        self._set_io_combo("result_ng", outputs.get("result_ng", 2))
        self._set_io_combo("part_ready", outputs.get("part_ready", 3))
        self._set_io_combo("busy", outputs.get("busy", 4))
        self._set_io_combo("trigger", inputs.get("trigger", 0))
        self._set_io_combo("reset_result", inputs.get("reset_result", 5))
        self._set_io_combo("switch_program", inputs.get("switch_program", 6))
        self._plc_program_register.setValue(io_map.get("program_register", 10))

        # PLC Slave ID + Scan range
        self._plc_slave_id.setValue(self._config.get("plc.modbus_slave_id", 1))
        self._plc_scan_range.setValue(self._config.get("plc.scan_range", 127))

        # Inference mode
        infer_mode = self._config.get("inference.mode", "continuous")
        self._inference_mode.setCurrentIndex(
            {"continuous": 0, "plc_trigger": 1, "manual": 2}.get(infer_mode, 0))

        # Model
        algo = self._config.get("model.algorithm", "patchcore")
        self._model_algo.setCurrentIndex(0 if algo == "patchcore" else 1)
        backbone = self._config.get("model.backbone", "resnet18")
        idx = self._model_backbone.findText(backbone)
        if idx >= 0:
            self._model_backbone.setCurrentIndex(idx)
        self._model_input_size.setValue(self._config.get("model.input_size", 256))

        # History
        self._retention_days.setValue(self._config.get("history.auto_purge_days", 30))
        self._max_entries.setValue(self._config.get("history.max_history_entries", 10000))
        self._ok_sample_pct.setValue(self._config.get("history.save_ok_sample_percent", 10))

        # Flask
        self._flask_enabled.setChecked(self._config.get("flask_api.enabled", False))
        self._flask_port.setValue(self._config.get("flask_api.port", 5000))

        # PostgreSQL
        self._pg_enabled.setChecked(self._config.get("postgresql.enabled", False))
        self._pg_host.setText(self._config.get("postgresql.host", "localhost"))
        self._pg_port.setValue(self._config.get("postgresql.port", 5432))
        self._pg_dbname.setText(self._config.get("postgresql.dbname", "visioninspect"))
        self._pg_user.setText(self._config.get("postgresql.user", "postgres"))
        self._pg_password.setText(self._config.get("postgresql.password", ""))

        # Language
        lang = self._config.get("language", "id")
        self._lang_combo.setCurrentIndex(0 if lang == "id" else 1)

        # NG Timeout
        self._ng_delay_spin.setValue(self._config.get("ng_debounce_ms", 500))

        # Cycle Delay
        self._cycle_delay_spin.setValue(self._config.get("inference.cycle_delay_ms", 1000))

        # YOLO class filter
        self._yolo_enabled.setChecked(self._config.get("yolo.enabled", False))
        self._yolo_model_path.setText(self._config.get("yolo.model_path", ""))
        yolo_classes = self._config.get("yolo.expected_classes", [])
        self._yolo_classes.setText(
            ", ".join(yolo_classes) if isinstance(yolo_classes, list) else "")
        self._yolo_min_conf.setValue(self._config.get("yolo.min_conf", 0.25))
        self._update_yolo_form_visibility()

        # Logging
        self._show_debug_cb.setChecked(self._config.get("show_debug", False))
