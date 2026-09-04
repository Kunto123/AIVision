"""
VisionInspect - Settings Page
Pengaturan global: kamera, PLC, inferensi (mode/timing/device), retensi,
Flask API, PostgreSQL, bahasa. ROI & model diatur per-template di tab TEACH.
"""

from PySide6.QtCore import Slot
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
        self._cam_gain = _add_spin_row(cam_layout, "Gain (-1=auto)", -1, 100000, -1)
        self._cam_wb = _add_spin_row(cam_layout, "White Balance (-1=auto, Kelvin)", -1, 20000, -1)

        main_layout.addWidget(cam_group)

        # ROI diatur per-template di tab TEACH (editor multi-ROI), bukan di sini.

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

        lbl_proto = QLabel(
            "Protokol: FX Computer Link — jalur yang sama dengan GX Works2. "
            "Format serial dikunci 7E1 oleh protokolnya; hanya baudrate yang "
            "bisa diatur.")
        lbl_proto.setWordWrap(True)
        lbl_proto.setStyleSheet("color: #94A3B8;")
        plc_layout.addWidget(lbl_proto)

        self._plc_port = QLineEdit("COM1")
        plc_layout.addWidget(QLabel("Port:"))
        plc_layout.addWidget(self._plc_port)

        plc_params = QHBoxLayout()
        plc_params.addWidget(QLabel("Baudrate:"))
        self._plc_baud = QComboBox()
        self._plc_baud.addItems(["9600", "19200", "38400", "57600", "115200"])
        plc_params.addWidget(self._plc_baud)
        plc_params.addStretch()
        plc_layout.addLayout(plc_params)

        plc_addr_row = QHBoxLayout()
        plc_addr_row.addWidget(QLabel("Scan range:"))
        self._plc_scan_range = QSpinBox()
        self._plc_scan_range.setRange(0, 9999)
        self._plc_scan_range.setValue(127)
        self._plc_scan_range.setToolTip(
            "Batas atas probe alamat coil saat Scan/Deteksi Aktif.")
        plc_addr_row.addWidget(self._plc_scan_range)
        plc_addr_row.addStretch()
        plc_layout.addLayout(plc_addr_row)

        # Pulse durasi (ms) untuk coil part_ready. Mapping coil ada di tab
        # I/O Settings — jangan dobel sumber config plc.io_map.
        self._plc_pulse_ms = _add_spin_row(plc_layout, "Pulse durasi (ms):", 0, 5000, 300)

        main_layout.addWidget(plc_group)

        # === Inference Settings ===
        infer_group = QGroupBox("Inference")
        infer_layout = QVBoxLayout(infer_group)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Start Cycle Mode:"))
        self._inference_mode = QComboBox()
        self._inference_mode.addItems(
            ["Auto Sequence (jalan terus)", "PLC Trigger"])
        self._inference_mode.setToolTip(
            "Auto Sequence: inspeksi tiap frame tanpa trigger (jalan terus).\n"
            "Dipakai untuk self-trigger lewat part-check: begitu tahap 1\n"
            "mengenali part, coil part_ready dikirim dan PLC yang memulai\n"
            "timer. Nyalakan 'Tulis coil part_ready' di I/O Settings.\n\n"
            "PLC Trigger: inspeksi hanya saat ada trigger — timing antar part\n"
            "di ladder PLC (filosofi Keyence IV3). Sumber trigger setara:\n"
            "coil PLC, tombol Trigger Now, atau POST /trigger.")
        mode_row.addWidget(self._inference_mode)
        mode_row.addStretch()
        infer_layout.addLayout(mode_row)

        # === Opsi khusus mode PLC Trigger ===
        self._infer_when_idle = QCheckBox(
            "Tetap infer di antara trigger (skor live terlihat)")
        self._infer_when_idle.setToolTip(
            "MATI (disarankan): tanpa trigger tidak ada inferensi sama sekali.\n"
            "Paling ringan, dan hasil resmi paling cepat keluar.\n\n"
            "NYALA: operator melihat skor live sepanjang waktu. Konsekuensi\n"
            "terukur di CPU 2 core — saat trigger masuk hampir selalu ada\n"
            "inferensi lain yang sedang jalan dan tidak bisa dibatalkan di\n"
            "tengah jalan, sehingga hasil resmi bisa tertunda hingga ~1 detik.\n"
            "Watchdog di ladder PLC harus dinaikkan (±2,5 → ±3,5 detik).\n\n"
            "Hanya berlaku di mode PLC Trigger.")
        infer_layout.addWidget(self._infer_when_idle)

        trig_row = QHBoxLayout()
        trig_row.addWidget(QLabel("Batas waktu siklus trigger:"))
        self._trigger_timeout_spin = QSpinBox()
        self._trigger_timeout_spin.setRange(200, 30000)
        self._trigger_timeout_spin.setValue(2000)
        self._trigger_timeout_spin.setSuffix(" ms")
        self._trigger_timeout_spin.setSingleStep(100)
        self._trigger_timeout_spin.setFixedWidth(130)
        self._trigger_timeout_spin.setToolTip(
            "Lewat batas ini, siklus trigger dianggap gagal: peringatan muncul\n"
            "di layar, tampilan tidak lagi beku, dan TIDAK ada sinyal ke PLC.\n"
            "Setel LEBIH PENDEK dari watchdog ladder supaya operator melihat\n"
            "penyebabnya sebelum lini berhenti.")
        trig_row.addWidget(self._trigger_timeout_spin)
        trig_row.addStretch()
        infer_layout.addLayout(trig_row)

        mode_help = QLabel(
            "PLC Trigger membutuhkan PLC aktif (Settings → PLC → Enable PLC).\n"
            "Kontrak sinyal: pulse OK/NG HANYA dikirim kalau model benar-benar "
            "selesai menilai. Part-check menolak, error, atau timeout → tidak "
            "ada sinyal sama sekali (diam = gagal), lini dihentikan oleh "
            "watchdog ladder.")
        mode_help.setObjectName("secondaryText")
        mode_help.setWordWrap(True)
        infer_layout.addWidget(mode_help)

        main_layout.addWidget(infer_group)

        # Pemilihan engine (YOLO / PatchCore / EfficientAd), backbone, input size,
        # dan threshold semuanya PER-TEMPLATE di tab TEACH → Training Profile.

        # === Runtime Inferensi (status, bukan setelan) ===
        runtime_group = QGroupBox("Runtime Inferensi")
        runtime_layout = QVBoxLayout(runtime_group)
        runtime_row = QHBoxLayout()
        runtime_row.addWidget(QLabel("Status:"))
        self._runtime_label = QLabel("—")
        self._runtime_label.setStyleSheet("font-weight: bold; padding: 2px 8px; border-radius: 3px;")
        runtime_row.addWidget(self._runtime_label)
        runtime_row.addStretch()
        runtime_layout.addLayout(runtime_row)
        main_layout.addWidget(runtime_group)

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

        # === Penghitungan Part ===
        # Pengganti "NG Timeout" lama yang menghitung DURASI, bukan jumlah part.
        ng_group = QGroupBox("Penghitungan Part")
        ng_form = QVBoxLayout(ng_group)

        ng_delay_row = QHBoxLayout()
        ng_delay_row.addWidget(QLabel("Jarak minimum antar hitungan:"))
        self._count_cooldown_spin = QSpinBox()
        self._count_cooldown_spin.setRange(0, 60000)
        self._count_cooldown_spin.setValue(1500)
        self._count_cooldown_spin.setSuffix(" ms")
        self._count_cooldown_spin.setSingleStep(100)
        self._count_cooldown_spin.setFixedWidth(130)
        self._count_cooldown_spin.setToolTip(
            "Hanya dipakai kalau tidak ada sumber 'satu part' yang pasti.\n"
            "Trigger PLC dan gate part-check lebih akurat dan dipakai duluan.")
        ng_delay_row.addWidget(self._count_cooldown_spin)
        ng_delay_row.addStretch()
        ng_form.addLayout(ng_delay_row)

        confirm_row = QHBoxLayout()
        confirm_row.addWidget(QLabel("Konfirmasi OK — N frame berturut:"))
        self._confirm_ok_frames_spin = QSpinBox()
        self._confirm_ok_frames_spin.setRange(1, 50)
        self._confirm_ok_frames_spin.setValue(1)
        self._confirm_ok_frames_spin.setFixedWidth(130)
        self._confirm_ok_frames_spin.setToolTip(
            "Gate part-check DAN judgement QC baru mengeluarkan OK setelah "
            "menerima N hasil infer OK berturut-turut.\n"
            "NG apa pun langsung mereset hitungan (fail-safe).\n"
            "Nilai yang sama juga jadi toleransi: gate yang sudah lolos butuh "
            "N frame 'tidak terbaca' berturut sebelum part dianggap hilang — "
            "tangan/bayangan lewat sekejap tidak memicu NG palsu.\n"
            "1 = tanpa konfirmasi (perilaku lama).\n"
            "Hanya mode Auto Sequence (continuous) — mode PLC Trigger selalu 1.\n"
            "Tiap frame konfirmasi tetap kena Cycle Delay.")
        confirm_row.addWidget(self._confirm_ok_frames_spin)
        confirm_row.addStretch()
        ng_form.addLayout(confirm_row)

        ng_help = QLabel(
            "Satu part harus terhitung SEKALI, berapa pun frame yang sempat "
            "diperiksa selagi part itu ada di depan kamera. Berlaku sama "
            "untuk OK dan NG.\n\n"
            "Urutan sumber 'satu part':\n"
            "  1. Trigger PLC — 1 trigger = 1 part (paling tepat)\n"
            "  2. Gate part-check — sekali per part yang lewat; butuh ada "
            "celah kosong antar part\n"
            "  3. Jarak waktu di atas — dipakai kalau dua-duanya tidak ada\n\n"
            "Setel LEBIH PENDEK dari jarak kedatangan part tercepat, kalau "
            "tidak ada part yang tidak terhitung. 0 = hitung tiap inspeksi "
            "(perilaku lama, satu part bisa terhitung berkali-kali).")
        ng_help.setObjectName("secondaryText")
        ng_help.setWordWrap(True)
        ng_form.addWidget(ng_help)

        main_layout.addWidget(ng_group)

        # === Inference Engine (Tugas 5) ===
        dev_group = QGroupBox("Inference Engine")
        dev_form = QVBoxLayout(dev_group)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Device:"))
        self._ov_device = QComboBox()
        self._ov_device.addItems(["CPU", "GPU", "AUTO"])
        self._ov_device.setFixedWidth(130)
        dev_row.addWidget(self._ov_device)
        dev_row.addStretch()
        dev_form.addLayout(dev_row)

        self._cpu_pcore_only = QCheckBox(
            "Batasi inference ke P-core (CPU hybrid)")
        self._cpu_pcore_only.setToolTip(
            "Untuk CPU dengan P-core + E-core (mis. Intel 12th gen ke atas).\n"
            "OpenVINO membagi satu inference ke semua thread lalu menunggu\n"
            "yang paling lambat — thread di E-core menahan seluruh inference.\n"
            "Aktif = latency lebih stabil, dan E-core bebas untuk GUI/video.\n"
            "Tidak berpengaruh di CPU non-hybrid. Diabaikan bila device GPU.")
        dev_form.addWidget(self._cpu_pcore_only)

        dev_help = QLabel(
            "GPU (iGPU Intel) biasanya jauh lebih cepat DAN membebaskan CPU "
            "untuk tampilan.\n"
            "PERINGATAN: GPU menghitung dengan presisi berbeda (FP16) "
            "sementara kalibrasi skor (norm.json) dibuat di CPU FP32 — skor "
            "bisa bergeser. Bandingkan hasil OK/NG CPU vs GPU pada gambar "
            "yang sama sebelum dipakai produksi.\n"
            "Compile pertama di GPU lambat (±18 dtk); berikutnya cepat "
            "karena memakai cache.")
        dev_help.setObjectName("secondaryText")
        dev_help.setWordWrap(True)
        dev_form.addWidget(dev_help)

        main_layout.addWidget(dev_group)

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
                "gain": self._cam_gain.value(),
                "white_balance": self._cam_wb.value(),
            },
            "plc": {
                "enabled": self._plc_enabled.isChecked(),
                "port": self._plc_port.text(),
                "baudrate": int(self._plc_baud.currentText()),
                "scan_range": self._plc_scan_range.value(),
                "pulse_ms": self._plc_pulse_ms.value(),
            },
            "inference": {
                "mode": ["continuous", "plc_trigger"][self._inference_mode.currentIndex()],
                "cycle_delay_ms": self._cycle_delay_spin.value(),
                "openvino_device": self._ov_device.currentText(),
                "cpu_pcore_only": self._cpu_pcore_only.isChecked(),
                "infer_when_idle": self._infer_when_idle.isChecked(),
                "trigger_timeout_ms": self._trigger_timeout_spin.value(),
                "count_cooldown_ms": self._count_cooldown_spin.value(),
                "confirm_ok_frames": self._confirm_ok_frames_spin.value(),
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
            "show_debug": self._show_debug_cb.isChecked(),
        }

    def get_save_button(self) -> QPushButton:
        return self._save_btn

    def get_camera_device_spin(self) -> QSpinBox:
        return self._cam_device

    def get_count_cooldown_ms(self) -> int:
        """Jarak minimum antar hitungan part (dipakai bila tidak ada trigger
        PLC / gate part-check sebagai sumber 'satu part')."""
        return self._config.get("inference.count_cooldown_ms", 1500)

    def get_confirm_ok_frames(self) -> int:
        """N hasil infer OK berturut sebelum verdict OK (gate + QC).
        1 = tanpa konfirmasi (perilaku lama)."""
        return self._config.get("inference.confirm_ok_frames", 1)

    def get_cycle_delay_ms(self) -> int:
        """Get cycle delay from config (ms). 0 = no delay."""
        return self._config.get("inference.cycle_delay_ms", 1000)

    # ---- PLC status (koneksi) — mapping coil ada di tab I/O Settings ----

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
        """Get selected inference mode (continuous|plc_trigger)."""
        return ["continuous", "plc_trigger"][self._inference_mode.currentIndex()]

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
        self._cam_gain.setValue(self._config.get("camera.gain", -1))
        self._cam_wb.setValue(self._config.get("camera.white_balance", -1))

        # PLC
        self._plc_enabled.setChecked(self._config.get("plc.enabled", False))
        self._plc_port.setText(self._config.get("plc.port", "COM1"))
        baudrate = str(self._config.get("plc.baudrate", 9600))
        idx = self._plc_baud.findText(baudrate)
        if idx >= 0:
            self._plc_baud.setCurrentIndex(idx)

        # PLC IO Mapping ada di tab I/O Settings → Apply, bukan di sini
        # (supaya tidak ada dua sumber config plc.io_map).
        self._plc_pulse_ms.setValue(self._config.get("plc.pulse_ms", 300))

        self._plc_scan_range.setValue(self._config.get("plc.scan_range", 127))

        # Inference mode
        infer_mode = self._config.get("inference.mode", "continuous")
        # "manual" tetap dipetakan sebagai jaring pengaman kalau ada config
        # yang lolos dari Config._migrate() — jangan sampai jatuh ke index 0.
        self._inference_mode.setCurrentIndex(
            {"continuous": 0, "plc_trigger": 1, "manual": 1}.get(infer_mode, 0))
        self._infer_when_idle.setChecked(
            self._config.get("inference.infer_when_idle", False))
        self._trigger_timeout_spin.setValue(
            self._config.get("inference.trigger_timeout_ms", 2000))

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
        self._count_cooldown_spin.setValue(
            self._config.get("inference.count_cooldown_ms", 1500))
        self._confirm_ok_frames_spin.setValue(
            self._config.get("inference.confirm_ok_frames", 1))

        # Cycle Delay
        self._cycle_delay_spin.setValue(self._config.get("inference.cycle_delay_ms", 1000))

        # Inference Engine (Tugas 5)
        dev = str(self._config.get("inference.openvino_device", "CPU")).upper()
        idx_dev = self._ov_device.findText(dev)
        self._ov_device.setCurrentIndex(idx_dev if idx_dev >= 0 else 0)
        self._cpu_pcore_only.setChecked(
            self._config.get("inference.cpu_pcore_only", False))

        # Logging
        self._show_debug_cb.setChecked(self._config.get("show_debug", False))
