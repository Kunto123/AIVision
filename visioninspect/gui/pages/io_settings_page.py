"""I/O Settings page — mirip "I/O Settings" di sensor Keyence IV3.

Fitur:
- Output Settings: mode hasil Latching / One-Shot (+ One-Shot ON Time & Delay),
  opsi coil BUSY / part_ready (default hanya coil OK — NG diputuskan PLC).
- Assign I/O: nomor coil output (OK/BUSY/part_ready) & input
  (trigger/reset/switch_template/ng_from_plc) + program_register.
- I/O Monitor: status coil real-time (live, polling 1 detik).

Perubahan di sini TIDAK menyentuh koneksi serial — itu tetap di Settings → PLC.
Halaman ini hanya mengatur pemetaan coil & perilaku hasil.
"""
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox,
    QSpinBox, QCheckBox, QPushButton, QGridLayout, QRadioButton,
    QButtonGroup, QScrollArea, QFrame,
)

from visioninspect.plc.io_map import build_io_mode

# (nama internal, label UI) — urutan tampil di tabel assign
OUTPUT_ROWS = [
    ("result_ok", "OK (hasil OK)"),
    # NG tidak lagi dikirim sistem — PLC yang memutuskannya dari ketiadaan OK.
    ("heartbeat", "Heartbeat (sistem sehat)"),
    ("busy", "BUSY (sibuk sensing)"),
    ("part_ready", "part_ready (part terdeteksi)"),
]
INPUT_ROWS = [
    ("trigger", "Trigger (minta 1x sensing)"),
    ("reset_result", "Reset counter produksi"),
    ("switch_template", "Ganti template (nomor dari register)"),
    ("ng_from_plc", "NG dari PLC (vonis + siap part berikutnya)"),
]
MONITOR_ROWS = [
    ("result_ok", "OK", "output"),
    ("heartbeat", "Heartbeat", "output"),
    ("part_ready", "part_ready", "output"),
    ("busy", "BUSY", "output"),
    ("trigger", "Trigger", "input"),
    ("reset_result", "Reset", "input"),
    ("switch_template", "Ganti Template", "input"),
    ("ng_from_plc", "NG dari PLC", "input"),
]

_QSS = """
QGroupBox {
    font-weight: 600; color: #A5B4FC;
    border: 1px solid #2A3A55; border-radius: 6px;
    margin-top: 10px; padding-top: 6px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton { background-color: #2563EB; color: white; border: none;
    border-radius: 5px; padding: 7px 16px; font-weight: 600; }
QPushButton:hover { background-color: #3B82F6; }
QTableWidget { background-color: #0F172A; alternate-background-color: #16213A;
    gridline-color: #2A3A55; color: #E2E8F0; }
"""


class IOSettingsPage(QWidget):
    # (io_map, io_mode) — dipancarkan saat tombol Apply; main_window yang simpan.
    apply_requested = Signal(dict, dict)
    # Tombol scan coil — main_window yang menjalankan thread + isi hasil.
    scan_requested = Signal()
    detect_requested = Signal()
    # Tombol "Test Koneksi" — main_window memanggil FXPLCManager.diagnose()
    test_connection_requested = Signal()
    # Uji tulis coil dari I/O Monitor — (nama_coil, nilai).
    test_coil_requested = Signal(str, bool)

    def __init__(self, translator, config, parent=None):
        super().__init__(parent)
        self._tr = translator
        self._config = config
        self._io_map: dict = {}
        self._monitor_source = None  # FXPLCManager — di-set main_window
        self._monitor_labels: dict = {}
        self._monitor_timer = QTimer(self)
        self._monitor_timer.setInterval(1000)
        self._monitor_timer.timeout.connect(self._poll_monitor)
        self._build_ui()
        self.load_from_config(config.get("plc") or {})

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self.setStyleSheet(_QSS)
        root = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)

        # ---- Status Koneksi ----
        # Ditaruh PALING ATAS karena ini pertanyaan pertama saat sesuatu tidak
        # jalan. Tanpa panel ini, monitor di bawah hanya bisa bilang "unknown"
        # untuk tiga sebab yang tindakannya berbeda-beda.
        grp_conn = QGroupBox("Status Koneksi PLC")
        gc = QVBoxLayout(grp_conn)

        self._conn_badge = QLabel("BELUM DIUJI")
        self._conn_badge.setAlignment(Qt.AlignCenter)
        self._conn_badge.setStyleSheet(self._conn_qss("unknown"))
        gc.addWidget(self._conn_badge)

        self._conn_detail = QLabel(
            "Tekan “Test Koneksi” untuk memeriksa port, slave ID, dan apakah "
            "PLC benar-benar menjawab.")
        self._conn_detail.setWordWrap(True)
        self._conn_detail.setStyleSheet("color: #94A3B8;")
        gc.addWidget(self._conn_detail)

        row_conn = QHBoxLayout()
        self._btn_test_conn = QPushButton("Test Koneksi")
        self._btn_test_conn.setToolTip(
            "Uji bertahap: port bisa dibuka? slave menjawab? alamat coil benar?\n"
            "Hasilnya menyebut tahap mana yang gagal, bukan sekadar gagal.")
        self._btn_test_conn.clicked.connect(self.test_connection_requested)
        row_conn.addWidget(self._btn_test_conn)
        row_conn.addStretch()
        gc.addLayout(row_conn)
        layout.addWidget(grp_conn)

        # ---- Output Settings (mirip IV3: Latching / One-Shot) ----
        grp_mode = QGroupBox("Output Settings (mode hasil ke PLC)")
        gm = QVBoxLayout(grp_mode)

        self._radio_latching = QRadioButton(
            "Latching — hasil di-hold (LEVEL) sampai trigger berikutnya / PLC "
            "reset (paling persis Keyence IV3)")
        self._radio_one_shot = QRadioButton(
            "One-Shot — pulse singkat (delay lalu ON selama durasi)")
        self._radio_latching.setChecked(True)
        self._btn_mode = QButtonGroup(self)
        self._btn_mode.addButton(self._radio_latching, 0)
        self._btn_mode.addButton(self._radio_one_shot, 1)
        gm.addWidget(self._radio_latching)
        gm.addWidget(self._radio_one_shot)

        row_os = QHBoxLayout()
        row_os.addWidget(QLabel("One-Shot ON Time (ms):"))
        self._spin_on_time = QSpinBox()
        self._spin_on_time.setRange(0, 10000)
        self._spin_on_time.setValue(300)
        self._spin_on_time.setToolTip("Durasi coil ON saat mode One-Shot")
        row_os.addWidget(self._spin_on_time)
        row_os.addWidget(QLabel("Delay (ms):"))
        self._spin_delay = QSpinBox()
        self._spin_delay.setRange(0, 10000)
        self._spin_delay.setValue(0)
        self._spin_delay.setToolTip("Tunda sebelum coil ON (mode One-Shot)")
        row_os.addWidget(self._spin_delay)
        row_os.addStretch()
        gm.addLayout(row_os)

        row_opt = QHBoxLayout()
        self._cb_busy = QCheckBox("Tulis coil BUSY ke PLC")
        self._cb_busy.setToolTip(
            "Default nonaktif (hanya OK/NG). Nyalakan bila ladder PLC butuh "
            "sinyal busy dari aplikasi.")
        self._cb_part_ready = QCheckBox("Tulis coil part_ready (transisi part tiba)")
        self._cb_part_ready.setToolTip(
            "Default nonaktif (hanya OK/NG). Nyalakan bila PLC ingin tahu part "
            "terdeteksi (Part Presence Check).")
        row_opt.addWidget(self._cb_busy)
        row_opt.addWidget(self._cb_part_ready)
        row_opt.addStretch()
        gm.addLayout(row_opt)

        lbl_timing = QLabel(
            "ℹ️ Sequence delay antar part = tanggung jawab ladder PLC "
            "(filosofi Keyence IV3). Aplikasi hanya memublikasikan hasil.")
        lbl_timing.setStyleSheet("color: #94A3B8;")
        gm.addWidget(lbl_timing)
        layout.addWidget(grp_mode)

        # ---- Assign Output ----
        grp_out = QGroupBox("Output Assign (coil yang aplikasi tulis)")
        g = QGridLayout(grp_out)
        g.addWidget(QLabel("Signal"), 0, 0)
        g.addWidget(QLabel("Coil #"), 0, 1)
        self._out_spins: dict = {}
        for i, (key, label) in enumerate(OUTPUT_ROWS, start=1):
            g.addWidget(QLabel(label), i, 0)
            spin = QSpinBox()
            spin.setRange(0, 9999)
            spin.setToolTip(f"Nomor coil untuk {key} — dipakai langsung sebagai nomor relay M di PLC (coil 1 = M1)")
            g.addWidget(spin, i, 1)
            self._out_spins[key] = spin
        g.setColumnStretch(0, 1)
        layout.addWidget(grp_out)

        # ---- Assign Input ----
        grp_in = QGroupBox("Input Assign (coil yang PLC tulis / aplikasi baca)")
        gi = QGridLayout(grp_in)
        gi.addWidget(QLabel("Signal"), 0, 0)
        gi.addWidget(QLabel("Coil #"), 0, 1)
        self._in_spins: dict = {}
        for i, (key, label) in enumerate(INPUT_ROWS, start=1):
            gi.addWidget(QLabel(label), i, 0)
            spin = QSpinBox()
            spin.setRange(0, 9999)
            spin.setToolTip(f"Nomor coil untuk {key} — dipakai langsung sebagai nomor relay M di PLC (coil 1 = M1)")
            gi.addWidget(spin, i, 1)
            self._in_spins[key] = spin
        # Baris DIHITUNG dari jumlah input, jangan ditulis keras: dulu angka
        # 4 benar saat input masih 3 baris, lalu ng_from_plc ditambahkan dan
        # baris ini menimpanya.
        _row_prog = len(INPUT_ROWS) + 1
        gi.addWidget(QLabel("Program Register # (D)"), _row_prog, 0)
        self._spin_prog_reg = QSpinBox()
        self._spin_prog_reg.setRange(0, 9999)
        self._spin_prog_reg.setToolTip(
            "Data register D berisi nomor template tujuan.\n"
            "Hanya dipakai bila plc.template_switch_mode = \"register\".\n"
            "Pada mode \"cycle\" (default) register ini tidak dibaca.")
        gi.addWidget(self._spin_prog_reg, _row_prog, 1)
        gi.setColumnStretch(0, 1)
        layout.addWidget(grp_in)

        # ---- Scan Coils (pindah dari Settings → PLC) ----
        grp_scan = QGroupBox("Scan Coils (cari alamat coil yang valid/aktif)")
        gs = QVBoxLayout(grp_scan)
        row_scan = QHBoxLayout()
        self._btn_scan = QPushButton("Scan Coils...")
        self._btn_scan.setToolTip("Probe alamat coil 0..scan_range — cari coil yang merespon")
        self._btn_scan.clicked.connect(self.scan_requested)
        row_scan.addWidget(self._btn_scan)
        self._btn_detect = QPushButton("Deteksi Aktif")
        self._btn_detect.setToolTip("Cari coil yang sedang ON — tekan tombol fisik di PLC saat scan")
        self._btn_detect.clicked.connect(self.detect_requested)
        row_scan.addWidget(self._btn_detect)
        row_scan.addStretch()
        gs.addLayout(row_scan)
        self._scan_result_label = QLabel("")
        self._scan_result_label.setWordWrap(True)
        self._scan_result_label.setStyleSheet("color: #94A3B8;")
        gs.addWidget(self._scan_result_label)
        layout.addWidget(grp_scan)

        # ---- I/O Monitor ----
        grp_mon = QGroupBox("I/O Monitor (status coil real-time — 1 detik)")
        gmon = QGridLayout(grp_mon)
        gmon.addWidget(QLabel("Signal"), 0, 0)
        gmon.addWidget(QLabel("Arah"), 0, 1)
        gmon.addWidget(QLabel("Status"), 0, 2)
        gmon.addWidget(QLabel("Uji tulis"), 0, 3)
        self._monitor_labels = {}
        self._test_buttons: dict = {}
        for i, (key, label, direction) in enumerate(MONITOR_ROWS, start=1):
            gmon.addWidget(QLabel(label), i, 0)
            dir_label = QLabel("→ PLC" if direction == "output" else "← PLC")
            dir_label.setStyleSheet("color: #94A3B8;")
            gmon.addWidget(dir_label, i, 1)
            st = QLabel("—")
            st.setAlignment(Qt.AlignCenter)
            st.setStyleSheet(self._monitor_qss("unknown"))
            gmon.addWidget(st, i, 2)
            self._monitor_labels[key] = st

            # Tombol uji HANYA untuk output. Input ditulis oleh ladder PLC —
            # menulisnya dari sini berarti melawan ladder, dan hasilnya akan
            # tertimpa di scan berikutnya sehingga terlihat "tidak berfungsi".
            if direction == "output":
                cell = QWidget()
                row = QHBoxLayout(cell)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(4)
                b_on = QPushButton("ON")
                b_off = QPushButton("OFF")
                for b in (b_on, b_off):
                    b.setFixedWidth(52)
                    b.setEnabled(False)
                    row.addWidget(b)
                b_on.clicked.connect(
                    lambda _=False, k=key: self.test_coil_requested.emit(k, True))
                b_off.clicked.connect(
                    lambda _=False, k=key: self.test_coil_requested.emit(k, False))
                gmon.addWidget(cell, i, 3)
                self._test_buttons[key] = (b_on, b_off)
        gmon.setColumnStretch(0, 1)

        self._cb_allow_test = QCheckBox(
            "Izinkan uji tulis coil — MEMATIKAN pengaman")
        self._cb_allow_test.setToolTip(
            "Menyalakan tombol ON/OFF di atas.\n"
            "Coil yang ditulis di sini masuk ke PLC SUNGGUHAN: ladder akan\n"
            "bereaksi, counter bisa bertambah, lampu bisa menyala.\n"
            "Jangan dinyalakan saat lini sedang produksi.")
        self._cb_allow_test.toggled.connect(self._on_allow_test_toggled)
        gmon.addWidget(self._cb_allow_test, len(MONITOR_ROWS) + 1, 0, 1, 4)

        self._lbl_test_warn = QLabel(self._TEST_HINT_OFF)
        self._lbl_test_warn.setWordWrap(True)
        self._lbl_test_warn.setStyleSheet("color: #94A3B8;")
        gmon.addWidget(self._lbl_test_warn, len(MONITOR_ROWS) + 2, 0, 1, 4)
        layout.addWidget(grp_mon)

        # ---- Apply ----
        row_apply = QHBoxLayout()
        row_apply.addStretch()
        self._btn_apply = QPushButton("Apply I/O Settings")
        self._btn_apply.clicked.connect(self._on_apply)
        row_apply.addWidget(self._btn_apply)
        layout.addLayout(row_apply)

        layout.addStretch()
        scroll.setWidget(body)
        root.addWidget(scroll)

    @staticmethod
    def _monitor_qss(state: str) -> str:
        colors = {
            "on": "#16A34A", "off": "#334155", "unknown": "#475569",
        }
        c = colors.get(state, colors["unknown"])
        return (f"font-weight: bold; color: white; background-color: {c}; "
                "border-radius: 4px; padding: 2px 10px;")

    # ------------------------------------------------------------ load/save

    def load_from_config(self, plc_cfg: dict):
        """Isi widget dari config plc (io_map + io_mode)."""
        plc_cfg = plc_cfg or {}
        io = plc_cfg.get("io_map") or {}
        outputs = io.get("outputs") or {}
        inputs = io.get("inputs") or {}

        for key, spin in self._out_spins.items():
            spin.setValue(int(outputs.get(key, 0)))
        for key, spin in self._in_spins.items():
            spin.setValue(int(inputs.get(key, 0)))
        self._spin_prog_reg.setValue(int(io.get("program_register", 10)))

        mode = build_io_mode(plc_cfg)
        if mode["output_mode"] == "one_shot":
            self._radio_one_shot.setChecked(True)
        else:
            self._radio_latching.setChecked(True)
        self._spin_on_time.setValue(int(mode["one_shot_on_time_ms"]))
        self._spin_delay.setValue(int(mode["one_shot_delay_ms"]))
        self._cb_busy.setChecked(bool(mode["busy_output"]))
        self._cb_part_ready.setChecked(bool(mode["part_ready_output"]))

    def get_io_map(self) -> dict:
        return {
            "outputs": {k: s.value() for k, s in self._out_spins.items()},
            "inputs": {k: s.value() for k, s in self._in_spins.items()},
            "program_register": self._spin_prog_reg.value(),
        }

    def get_io_mode(self) -> dict:
        mode = "one_shot" if self._radio_one_shot.isChecked() else "latching"
        return {
            "output_mode": mode,
            "one_shot_on_time_ms": self._spin_on_time.value(),
            "one_shot_delay_ms": self._spin_delay.value(),
            "busy_output": self._cb_busy.isChecked(),
            "part_ready_output": self._cb_part_ready.isChecked(),
        }

    def _on_apply(self):
        self.apply_requested.emit(self.get_io_map(), self.get_io_mode())

    # ------------------------------------------------------------ scan

    def set_scan_result(self, text: str) -> None:
        """Tampilkan hasil Scan Coils / Deteksi Aktif di label group scan."""
        self._scan_result_label.setText(text)

    def set_scan_busy(self, active: bool) -> None:
        """Enable/disable tombol scan saat thread worker sedang jalan."""
        self._btn_scan.setEnabled(not active)
        self._btn_detect.setEnabled(not active)

    # ------------------------------------------------------------ monitor

    def set_monitor_source(self, link) -> None:
        """Pasang sumber monitor (FXPLCManager) & mulai polling."""
        self._monitor_source = link
        if link is not None and getattr(link, "is_connected", False):
            self._monitor_timer.start()
        else:
            self._monitor_timer.stop()
            self._set_all_monitor("unknown")

    def stop_monitor(self) -> None:
        self._monitor_timer.stop()

    #: Tiga keadaan panel uji tulis. Dipisah sebagai konstanta supaya tidak
    #: ada cabang yang lupa memperbarui label.
    _TEST_HINT_OFF = (
        "Uji tulis berguna saat commissioning: nyalakan coil, lalu lihat "
        "apakah ladder bereaksi sesuai harapan. Aplikasi juga menulis coil "
        "yang sama saat berjalan — kalau keduanya aktif bersamaan, nilai "
        "Anda akan tertimpa.")
    _TEST_HINT_ARMED = (
        "Uji tulis AKTIF. Coil yang Anda tekan masuk ke PLC sungguhan — "
        "ladder akan bereaksi, counter bisa bertambah, lampu bisa menyala.")
    _TEST_HINT_NO_LINK = (
        "PLC belum terhubung — tombol uji tetap mati. "
        "Tekan “Test Koneksi” di atas dulu.")

    def _on_allow_test_toggled(self, on: bool) -> None:
        """Aktifkan tombol uji, tapi hanya kalau PLC memang terhubung.

        Tanpa syarat kedua, tombol akan tampak siap padahal setiap penekanan
        gagal diam-diam — dan operator akan mengira ladder-nya yang salah.
        """
        live = (self._monitor_source is not None
                and getattr(self._monitor_source, "is_connected", False))
        for b_on, b_off in self._test_buttons.values():
            b_on.setEnabled(on and live)
            b_off.setEnabled(on and live)
        # Label harus ditulis di SETIAP cabang. Sebelumnya hanya cabang gagal
        # yang mengisinya, sehingga pesan "PLC belum terhubung" tetap terpampang
        # setelah koneksi berhasil — bertabrakan dengan badge hijau di atasnya.
        if not on:
            self._lbl_test_warn.setText(self._TEST_HINT_OFF)
        elif live:
            self._lbl_test_warn.setText(self._TEST_HINT_ARMED)
        else:
            self._lbl_test_warn.setText(self._TEST_HINT_NO_LINK)

    def set_test_result(self, name: str, value: bool, ok_: bool) -> None:
        """Umpan balik satu penulisan uji, dipanggil main_window."""
        self._lbl_test_warn.setText(
            f"{'BERHASIL' if ok_ else 'GAGAL'} menulis {name} = "
            f"{'ON' if value else 'OFF'}."
            + ("" if ok_ else " Cek status koneksi di atas."))

    @staticmethod
    def _conn_qss(state: str) -> str:
        base = ("font-weight: 700; font-size: 15px; letter-spacing: 1px; "
                "padding: 9px; border-radius: 5px; ")
        return base + {
            "ok": "background-color: #14532D; color: #86EFAC;",
            "fail": "background-color: #4C1D24; color: #FCA5A5;",
        }.get(state, "background-color: #1E293B; color: #94A3B8;")

    def set_connection_status(self, info: dict) -> None:
        """Tampilkan hasil FXPLCManager.diagnose() (atau status kasar).

        `info` minimal berisi: ok (bool), stage (str), detail (str).
        """
        ok = bool(info.get("ok"))
        stage = str(info.get("stage") or "")
        labels = {
            "ok": "TERHUBUNG",
            "port": "GAGAL — PORT SERIAL",
            "serial": "GAGAL — TRANSAKSI SERIAL",
            "slave": "GAGAL — PLC TIDAK MENJAWAB",
            "library": "GAGAL — FXPLC TIDAK TERPASANG",
            "disabled": "PLC BELUM DIAKTIFKAN",
        }
        self._conn_badge.setText(labels.get(stage, "GAGAL" if not ok else "TERHUBUNG"))
        self._conn_badge.setStyleSheet(
            self._conn_qss("ok" if ok else "fail" if stage else "unknown"))
        self._conn_detail.setText(str(info.get("detail") or ""))

    def refresh_monitor_connection(self) -> None:
        """Panggil saat status koneksi PLC berubah (start/stop polling)."""
        # Tombol uji ikut status koneksi — kalau tidak, ia tetap terlihat
        # aktif setelah PLC putus dan setiap penekanan gagal tanpa penjelasan.
        self._on_allow_test_toggled(self._cb_allow_test.isChecked())
        if self._monitor_source is not None and \
                getattr(self._monitor_source, "is_connected", False):
            self._monitor_timer.start()
        else:
            self._monitor_timer.stop()
            self._set_all_monitor("unknown")
            # Jangan menimpa hasil diagnose yang lebih rinci dengan pesan
            # generik — hanya isi kalau memang belum pernah diuji.
            if self._conn_badge.text() == "BELUM DIUJI":
                self.set_connection_status({
                    "ok": False, "stage": "",
                    "detail": "PLC tidak terhubung. Tekan “Test Koneksi” "
                              "untuk tahu tahap mana yang gagal.",
                })

    def _poll_monitor(self):
        src = self._monitor_source
        if src is None or not getattr(src, "is_connected", False):
            self._set_all_monitor("unknown")
            return
        try:
            states = src.read_all_coil_states()
        except Exception:
            self._set_all_monitor("unknown")
            return
        for key, label in self._monitor_labels.items():
            val = states.get(key)
            if val is None:
                label.setText("?")
                label.setStyleSheet(self._monitor_qss("unknown"))
            else:
                label.setText("ON" if val else "OFF")
                label.setStyleSheet(self._monitor_qss("on" if val else "off"))

    def _set_all_monitor(self, state: str):
        for key, label in self._monitor_labels.items():
            label.setText("—" if state == "unknown" else "OFF")
            label.setStyleSheet(self._monitor_qss(state))
