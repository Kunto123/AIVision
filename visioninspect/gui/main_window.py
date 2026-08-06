"""
VisionInspect - Main Window
Window utama dengan tab navigasi: RUN, TEACH, HISTORY, SETTINGS, DIAGNOSTICS.
Mengelola CameraWorker, inferensi, ProgramManager, dan komponen global.
"""

import os
import sys
import time
import json
import uuid
import threading
import re
from pathlib import Path
from typing import Optional

import cv2
import psutil

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence, QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from visioninspect.utils import normalize_wsl_path, is_wsl
from visioninspect.utils.config import Config, ConfigError
from visioninspect.utils.i18n import Translator
from visioninspect.utils.logging_setup import setup_logging, get_logger

from visioninspect.core.program import ProgramManager
from visioninspect.core.inference import InferenceEngine, overlay_heatmap
from visioninspect.core.yolo_filter import YOLODetector, class_filter_matches
from visioninspect.core import part_check as pc_module
from visioninspect.gui.camera_worker import CameraThread, CameraWorker
from visioninspect.gui.training_worker import TrainingThread, TrainingWorker
from visioninspect.gui.widgets.roi_editor import ROIData
from visioninspect.gui.pages.run_page import RunPage
from visioninspect.plc.modbus_rtu import (
    ModbusRTUManager, HAS_MODBUS, build_io_mode,
)
from visioninspect.api.flask_app import FlaskAPI, HAS_FLASK
from visioninspect.gui.pages.teach_page import TeachPage
from visioninspect.gui.pages.history_page import HistoryPage
from visioninspect.gui.pages.settings_page import SettingsPage
from visioninspect.gui.pages.io_settings_page import IOSettingsPage
from visioninspect.gui.pages.diagnostics_page import DiagnosticsPage
from visioninspect.gui.pages.account_page import AccountPage
from visioninspect.gui.dialogs.login_dialog import LoginDialog

logger = get_logger("app")


class MainWindow(QMainWindow):
    """Main application window dengan 5 tab halaman."""

    # Signal untuk invoke training di QThread worker
    start_training_signal = Signal(str, str, bool)

    # Signals untuk kirim hasil training WSL dari background thread biasa
    # (bukan QThread) balik ke GUI thread. QTimer.singleShot yang dipanggil
    # dari thread tanpa Qt event loop tidak aman (bisa warning "Timers can
    # only be used with threads started with QThread") — Signal.emit() lintas
    # thread otomatis di-queue oleh Qt ke thread pemilik receiver, jadi ini
    # cara yang benar.
    _wsl_train_done_signal = Signal()
    _wsl_train_error_signal = Signal(str)
    _wsl_train_progress_signal = Signal(str)
    _plc_scan_done_signal = Signal(list)
    _plc_detect_done_signal = Signal(list)

    def __init__(self, config: Config, translator: Translator):
        super().__init__()
        self._config = config
        self._tr = translator

        # Camera
        self._camera_thread: Optional[CameraThread] = None
        self._camera_worker: Optional[CameraWorker] = None

        # Program Manager — path normalisasi khusus WSL
        raw_data_dir = config.get("data_dir", "")
        if raw_data_dir and is_wsl():
            data_dir = normalize_wsl_path(raw_data_dir).resolve()
        elif raw_data_dir:
            data_dir = Path(raw_data_dir).resolve()
        else:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir
        logger.info("Data directory: %s", self._data_dir)
        self._pm = ProgramManager(data_dir / "programs")

        # Database (shared instance for history, counters, corrections, users)
        from visioninspect.storage.db import Database
        from visioninspect.storage.postgres_db import PostgresDB
        from visioninspect.storage import secret_store
        self._db = Database(data_dir / "database.db")
        # PostgreSQL connection (optional — enabled via config).
        # WAJIB sub-dict "postgresql" (PostgresDB baca config["enabled"]/["host"]/dst);
        # get_all() meneruskan config penuh -> "enabled" tak ketemu -> keliru nonaktif.
        self._pg = PostgresDB(self._config.get("postgresql", {}))
        if self._pg.is_enabled:
            # Pastikan DB siap pakai (tabel ada, admin ter-seed) begitu terhubung
            self._pg.ensure_ready()

        # C4: migrasi kredensial — password PG plaintext lama dienkripsi sekali
        pg_cfg = self._config.get("postgresql", {})
        pg_pw = pg_cfg.get("password", "")
        if pg_pw and not secret_store.is_encrypted(pg_pw):
            try:
                self._config.set("postgresql.password", secret_store.encrypt(pg_pw))
                self._config.save()
                logger.info("Migrasi C4: password PostgreSQL dienkripsi (bukan plaintext)")
            except Exception as e:
                logger.warning("Migrasi enkripsi password PG gagal: %s", e)

        # C3: flush sisa outbox saat startup + tick berkala 30 detik
        self._pg_flush_timer = QTimer(self)
        self._pg_flush_timer.timeout.connect(self._flush_pg_outbox)
        self._pg_flush_timer.start(30000)
        QTimer.singleShot(1500, self._flush_pg_outbox)

        # Authentication state
        self._current_user: Optional[dict] = None
        self._user_role: str = "operator"

        # State
        self._active_program = ""
        self._active_template = ""
        self._force_regenerate_augmentation = False

        # Performance monitoring
        self._perf_timer = QTimer(self)
        self._perf_timer.timeout.connect(self._update_performance)
        self._process = psutil.Process()

        # Training worker
        self._training_thread = TrainingThread(self._pm, self)
        self._training_thread.start()
        self._training_worker = self._training_thread.worker

        # Inference engine
        self._inference_engine = InferenceEngine(input_size=256)
        # Detektor YOLO (filter kelas) — lazy load, None bila nonaktif/gagal
        self._yolo_det = None
        self._last_class_filter_ng = False
        self._current_roi: Optional[tuple] = None
        self._current_all_rois: list = []
        self._current_all_roi_uids: list = []
        self._current_all_roi_labels: list = []
        self._heatmap_enabled = False
        self._last_frame: Optional[object] = None
        self._last_heatmap: Optional[object] = None

        # Import review mode
        self._import_files: list = []
        self._import_index = 0
        self._is_import_mode = False
        self._import_current_image = None  # numpy array, cache untuk hindari double-read
        self._import_cancelled = False
        self._import_ok_count = 0  # accumulated counts untuk batch config update
        self._import_ng_count = 0

        # Counters
        self._inspection_count = 0
        self._inspection_ok = 0
        self._inspection_ng = 0
        self._inference_save_counter = 0  # throttle DB saves (~1/sec)

        # NG interval timer (fires every N ms during sustained NG)
        self._ng_interval_timer = QTimer(self)
        self._ng_interval_timer.timeout.connect(self._on_ng_interval_tick)
        self._ng_interval_active = False

        # Cycle delay timer (jeda antar siklus inspeksi)
        self._cycle_delay_timer = QTimer(self)
        self._cycle_delay_timer.setSingleShot(True)
        self._cycle_delay_timer.timeout.connect(self._on_cycle_delay_tick)
        self._cycle_delay_active = False

        # Part Presence Check (cached config — read from disk only on template switch)
        self._current_part_check_cfg: dict = {}
        # Overlay/gating state (updated every frame in _on_frame_for_inference)
        self._last_part_ready = False
        self._pc_active_for_overlay = False
        self._last_gate_roi: Optional[dict] = None
        # Part check score untuk push ke PG
        self._last_part_check_score = 1.0
        # Worst score terakhir untuk NG tick
        self._last_worst_score = 0.0
        # ROI color state untuk live inference: orange default, berubah hijau/merah
        # sesuai judgement, lalu balik ke orange setelah TIME_OUT detik
        self._roi_col_judgement: dict = {}        # {idx: "OK"/"NG"}
        self._roi_col_timestamp: float = 0.0       # time.monotonic() terakhir update
        self._roi_col_duration: float = 3.0        # detik sebelum balik orange
        # Part name untuk push ke PG (di-set saat ganti template)
        self._active_partname = ""

        # ── PLC (Modbus master) ──
        self._plc_modbus: Optional[ModbusRTUManager] = None
        self._plc_poll_timer: Optional[QTimer] = None
        self._plc_trigger_pending = False

        # ── Flask API (opsional, bind 127.0.0.1) ──
        self._flask_api: Optional[FlaskAPI] = None

        # Judgement terakhir untuk endpoint /last_result
        self._last_judgement = "—"

        self._setup_window()
        self._setup_tabs()
        self._setup_statusbar()
        self._setup_menu()
        self._apply_theme()
        self._connect_signals()
        self._init_camera()
        self._start_perf_monitor()
        self._init_programs()
        self._init_plc()
        self._init_flask()

        # Sync label Trigger mode + status PLC di Settings saat startup
        self._run_page.set_trigger_mode(
            self._config.get("inference.mode", "continuous"))
        if not self._config.get("plc.enabled", False):
            self._settings_page.set_plc_status(False, "Tidak diaktifkan")

        # Apply saved debug logging setting on startup
        import logging
        show_debug = self._config.get("show_debug", False)
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(logging.DEBUG if show_debug else logging.INFO)

        # Update runtime status indicator
        QTimer.singleShot(500, self._update_runtime_status)

        # Initial history load
        QTimer.singleShot(1000, self._refresh_history)

        # Show login dialog segera (full screen, blocks until login/cancel).
        # Kamera & inferensi baru mulai setelah login (lihat _show_login).
        QTimer.singleShot(0, self._show_login)

        logger.info("MainWindow initialized")

    # ---- Setup ----

    def _setup_window(self):
        self.setWindowTitle(self._tr.tr("app_title"))
        self.setMinimumSize(1280, 720)
        self.resize(1600, 1000)

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        self._main_layout = QVBoxLayout(central)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

    def _setup_tabs(self):
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._run_page = RunPage(self._tr, self._config)
        self._teach_page = TeachPage(self._tr)
        self._history_page = HistoryPage(self._tr)
        self._settings_page = SettingsPage(self._tr, self._config)
        self._diagnostics_page = DiagnosticsPage(self._tr)
        auth_db = self._pg if self._pg.is_enabled else self._db
        self._account_page = AccountPage(auth_db)
        self._io_page = IOSettingsPage(self._tr, self._config)

        self._tabs.addTab(self._run_page, self._tr.tr("nav_run"))
        self._tabs.addTab(self._teach_page, self._tr.tr("nav_teach"))
        self._tabs.addTab(self._history_page, self._tr.tr("nav_history"))
        self._tabs.addTab(self._settings_page, self._tr.tr("nav_settings"))
        self._tabs.addTab(self._diagnostics_page, self._tr.tr("nav_diagnostics"))
        self._tabs.addTab(self._account_page, "👥 Akun")
        self._tabs.addTab(self._io_page, "I/O Settings")
        self._io_page.apply_requested.connect(self._on_io_settings_apply)

        # By default hide admin-only tabs; shown after login if role=admin
        for idx in range(1, self._tabs.count()):
            self._tabs.setTabVisible(idx, False)
        self._tabs.setTabVisible(5, False)  # account page

        self._main_layout.addWidget(self._tabs)

    def _setup_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

        # User info + Logout (right side)
        self._user_label = QLabel("👤 —")
        self._user_label.setStyleSheet("font-weight: bold; color: #22C55E; padding: 0 8px;")
        self._statusbar.addPermanentWidget(self._user_label)

        self._logout_btn = QPushButton("Logout")
        self._logout_btn.setFixedHeight(24)
        self._logout_btn.setStyleSheet(
            "font-size: 11px; padding: 0 8px; border: 1px solid #233A57;"
            " border-radius: 3px; background: #1A2A44; color: #EF4444;")
        self._logout_btn.setVisible(False)
        self._logout_btn.clicked.connect(self._on_logout)
        self._statusbar.addPermanentWidget(self._logout_btn)

        self._program_label = QLabel("Program: —")
        self._statusbar.addPermanentWidget(self._program_label)

        self._cam_status_label = QLabel("Camera —")
        self._statusbar.addPermanentWidget(self._cam_status_label)

        self._fps_status_label = QLabel("FPS: —")
        self._statusbar.addPermanentWidget(self._fps_status_label)

        self._statusbar.showMessage(self._tr.tr("ready"))

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        cam_menu = menubar.addMenu("Kamera")
        self._start_cam_action = QAction("Start Kamera", self)
        self._start_cam_action.triggered.connect(self._toggle_camera_menu)
        cam_menu.addAction(self._start_cam_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction(f"About {self._tr.tr('app_name')}", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        # ── Tools menu ──
        tools_menu = menubar.addMenu("Tools")

        self._export_model_action = QAction("Export Model", self)
        self._export_model_action.setShortcut(QKeySequence("Ctrl+E"))
        self._export_model_action.triggered.connect(self._export_model_dialog)
        tools_menu.addAction(self._export_model_action)

        self._import_model_action = QAction("Import Model", self)
        self._import_model_action.setShortcut(QKeySequence("Ctrl+I"))
        self._import_model_action.triggered.connect(self._import_model_dialog)
        tools_menu.addAction(self._import_model_action)

    def _apply_theme(self):
        theme_path = Path(__file__).parent / "theme.qss"
        if theme_path.exists():
            with open(theme_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    # ---- Authentication ----

    def _show_login(self):
        """Show login dialog, apply role visibility after success."""
        # C2: is_enabled hanya flag config — server bisa mati. Cek koneksi
        # hidup; kalau PG tidak terjangkau, fallback ke autentikasi SQLite
        # lokal supaya lini tidak berhenti (server DB mati ≠ nobody can login).
        if self._pg.is_enabled and self._pg.is_alive(timeout=2.0):
            auth_db = self._pg
        else:
            if self._pg.is_enabled:
                logger.warning(
                    "PostgreSQL tidak terjangkau — fallback autentikasi SQLite lokal (C2)")
            auth_db = self._db
        dialog = LoginDialog(auth_db, self)
        if dialog.exec():
            self._current_user = dialog.user
            self._user_role = dialog.role
            self._apply_role_visibility()
            # Mulai kamera & inferensi SETELAH login berhasil (view hanya
            # berjalan setelah user login).
            if self._camera_worker and not self._camera_worker.is_running:
                dev = self._config.get("camera.device_index", 0)
                QTimer.singleShot(300, lambda: self._camera_worker.start_camera(dev))
            self.set_status(
                f"Selamat datang, {dialog.display_name} ({dialog.role})", 3000)
            logger.info("Login: %s (role=%s)", dialog.username, dialog.role)
        else:
            # Login cancelled — exit app
            logger.info("Login dibatalkan, keluar aplikasi")
            QTimer.singleShot(200, self.close)

    def _on_logout(self):
        """Logout current user and show login dialog again."""
        reply = QMessageBox.question(
            self, "Logout", "Yakin ingin logout?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        # Reset state
        self._current_user = None
        self._user_role = "operator"
        self._logout_btn.setVisible(False)
        self._user_label.setText("👤 —")

        # Hentikan kamera & inferensi selama logout — view berhenti berjalan
        # sampai user login lagi.
        if self._camera_worker and self._camera_worker.is_running:
            self._camera_worker.stop_camera()

        # Show login dialog again
        self._show_login()

    def _apply_role_visibility(self):
        """Show/hide tabs based on user role.
        Admin sees all tabs EXCEPT RUN. Operator sees only RUN."""
        is_admin = self._user_role == "admin"

        # Tab indices: 0=RUN, 1=TEACH, 2=HISTORY, 3=SETTINGS, 4=DIAGNOSTICS,
        #              5=AKUN, 6=GLOBAL SETTINGS
        self._tabs.setTabVisible(0, not is_admin)  # RUN: operator only
        self._tabs.setTabVisible(1, is_admin)      # TEACH
        self._tabs.setTabVisible(2, is_admin)      # HISTORY
        self._tabs.setTabVisible(3, is_admin)      # SETTINGS
        self._tabs.setTabVisible(4, is_admin)      # DIAGNOSTICS
        self._tabs.setTabVisible(5, is_admin)      # AKUN
        self._tabs.setTabVisible(6, is_admin)      # GLOBAL SETTINGS
        # View operator = 1 tab saja → sembunyikan tab bar (hilangkan
        # tulisan "RUN"); admin butuh tab bar untuk navigasi.
        self._tabs.tabBar().setVisible(is_admin)

        if is_admin:
            self._account_page.refresh()
            self._tabs.setCurrentIndex(1)  # Start on TEACH for admin
            QTimer.singleShot(0, self._go_windowed)
        else:
            self._tabs.setCurrentIndex(0)  # Start on RUN for operator
            self._reset_counters()          # Fresh counters for operator
            QTimer.singleShot(0, self._go_fullscreen)

        # Update user display in status bar
        uname = self._current_user.get("display_name", self._current_user.get("username", ""))
        self._user_label.setText(f"👤 {uname} ({self._user_role})")
        self._logout_btn.setVisible(True)
        logger.info("Role applied: %s (admin=%s)", self._user_role, is_admin)

    def _connect_signals(self):
        # Settings
        self._settings_page.get_save_button().clicked.connect(self._on_settings_save)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # PLC scan (I/O Settings → Scan Coils) — hasil dari thread worker
        self._io_page.scan_requested.connect(self._on_plc_scan)
        self._io_page.detect_requested.connect(self._on_plc_detect_active)
        self._plc_scan_done_signal.connect(self._on_plc_scan_done)
        self._plc_detect_done_signal.connect(self._on_plc_detect_done)

        # Manual trigger (Run page 'Trigger Now' — juga dipakai mode manual)
        self._run_page.get_trigger_button().clicked.connect(self._on_trigger_now)

        # WSL training results (emitted from a plain background thread —
        # see _train_via_wsl)
        self._wsl_train_done_signal.connect(self._on_wsl_train_done)
        self._wsl_train_error_signal.connect(self._on_training_error)
        self._wsl_train_progress_signal.connect(self._on_wsl_train_progress)

        # Camera toggle
        self._run_page.get_camera_toggle_button().clicked.connect(self._toggle_camera)
        self._run_page.get_device_spin().valueChanged.connect(self._on_camera_device_change)
        self._settings_page.get_camera_device_spin().valueChanged.connect(self._on_camera_device_change)
        self._run_page.get_heatmap_button().toggled.connect(self._on_heatmap_toggle)

        # TEACH: Capture buttons
        self._teach_page.get_capture_ok_button().clicked.connect(
            lambda: self._on_capture("ok"))
        self._teach_page.get_capture_ng_button().clicked.connect(
            lambda: self._on_capture("ng"))
        self._teach_page.get_import_button().clicked.connect(self._on_import_images)
        self._teach_page.import_cancelled.connect(self._on_cancel_import)
        self._teach_page.get_test_model_button().clicked.connect(self._on_test_model)

        # TEACH: Train button
        self._teach_page.get_train_button().clicked.connect(self._on_train)

        # TEACH: Template buttons
        self._teach_page.get_add_template_button().clicked.connect(self._on_add_template)
        self._teach_page.get_rename_template_button().clicked.connect(self._on_rename_template)
        self._teach_page.get_template_combo().currentIndexChanged.connect(
            self._on_template_changed)

        # RUN: Template selector (syncs with TEACH)
        self._run_page.get_template_combo().currentIndexChanged.connect(
            self._on_template_changed)

        # TEACH: Clear gallery button
        self._teach_page.get_clear_button().clicked.connect(self._on_clear_template)

        # TEACH: ROI controls
        self._teach_page.get_roi_editor().rois_changed.connect(self._on_rois_changed)
        self._teach_page.get_roi_panel().roi_added.connect(self._on_roi_add)
        self._teach_page.get_roi_panel().roi_selected.connect(self._on_roi_select)
        self._teach_page.get_roi_panel().roi_delete_requested.connect(self._on_roi_delete)
        self._teach_page.get_roi_panel().roi_rename_requested.connect(self._on_roi_rename)
        self._teach_page.get_roi_panel().roi_toggle_all.connect(self._on_roi_toggle_all)

        # TEACH: Threshold slider → live update inference threshold,
        # lalu simpan permanen ke config template saat slider dilepas
        # (sliderReleased, bukan tiap tick, agar tidak menulis file terus-menerus)
        self._teach_page.get_threshold_slider().valueChanged.connect(self._on_threshold_slider)
        self._teach_page.get_threshold_slider().sliderReleased.connect(self._on_threshold_released)
        # Commit threshold yang diketik manual — editingFinished = Enter/focus-out,
        # setara sliderReleased untuk jalur keyboard
        self._teach_page.get_threshold_spin().editingFinished.connect(self._on_threshold_released)

        # TEACH: Image deleted from gallery
        self._teach_page.image_deleted.connect(self._on_gallery_image_deleted)
        # TEACH: Thumbnail clicked → popup ROI adjust
        self._teach_page.thumbnail_clicked.connect(self._on_thumbnail_clicked)

        # ACCOUNT: User changes
        self._account_page.roles_changed.connect(self._refresh_history)

        # TEACH: Part Presence Check signals
        pc = self._teach_page
        pc.get_pc_enabled_cb().toggled.connect(self._on_part_check_config_changed)
        pc.get_pc_method_combo().currentIndexChanged.connect(
            self._on_part_check_config_changed)
        pc.get_pc_color_th_spin().editingFinished.connect(
            self._on_part_check_config_changed)
        pc.get_pc_edge_th_spin().editingFinished.connect(
            self._on_part_check_config_changed)
        pc.get_pc_canny_low_spin().editingFinished.connect(
            self._on_part_check_config_changed)
        pc.get_pc_canny_high_spin().editingFinished.connect(
            self._on_part_check_config_changed)
        pc.get_gate_roi_editor().rois_changed.connect(self._on_gate_roi_changed)
        pc.get_capture_master_button().clicked.connect(self._on_capture_master)

        # TEACH: Training Profile signal
        self._teach_page.training_config_changed.connect(
            self._on_training_config_changed)

        # TEACH: Augmentasi Data signals
        self._teach_page.augmentation_config_changed.connect(
            self._on_augmentation_config_changed)

        # HISTORY: Filter
        self._history_page.get_filter_combo().currentIndexChanged.connect(
            self._on_history_filter_changed)

        # HISTORY: Correction buttons
        self._history_page.get_correct_ok_button().clicked.connect(
            lambda: self._on_correct_history("OK"))
        self._history_page.get_correct_ng_button().clicked.connect(
            lambda: self._on_correct_history("NG"))
        self._history_page.get_rebuild_button().clicked.connect(self._on_rebuild_from_history)

        # HISTORY: Tuning
        self._history_page.tuning_requested.connect(self._on_tuning_requested)

        # HISTORY: Rollback
        self._history_page.get_rollback_button().clicked.connect(self._on_rollback)

        # HISTORY: Selection changed
        self._history_page.get_table().itemSelectionChanged.connect(
            self._on_history_selection_changed)

        # Training worker signals
        self.start_training_signal.connect(self._training_worker.start_training)
        self._training_worker.progress.connect(self._on_training_progress)
        self._training_worker.finished.connect(self._on_training_finished)
        self._training_worker.error.connect(self._on_training_error)
        self._training_worker.done.connect(self._on_training_done)

    # ---- Camera ----

    def _init_camera(self):
        self._camera_thread = CameraThread(self)
        self._camera_thread.init_worker()
        self._camera_worker = self._camera_thread.worker
        # F2: terusan config kamera (exposure/gain/WB) — sebelumnya hanya
        # device_index, sehingga exposure di Settings tidak pernah berlaku
        self._camera_worker.set_camera_config(self._config.get("camera", {}))
        self._camera_thread.start()

        self._camera_worker.frame_ready.connect(self._on_frame_received)
        self._camera_worker.camera_started.connect(self._on_camera_started)
        self._camera_worker.camera_stopped.connect(self._on_camera_stopped)
        self._camera_worker.camera_error.connect(self._on_camera_error)
        self._camera_worker.fps_updated.connect(self._on_fps_updated)
        self._camera_worker.status_message.connect(self._on_camera_status)
        self._camera_worker.frame_raw.connect(self._on_frame_for_inference)

        # Kamera TIDAK auto-start di sini — baru dijalankan setelah login
        # berhasil (lihat _show_login), agar view tidak berjalan sebelum login.

    def _toggle_camera(self):
        if self._camera_worker:
            self._camera_worker.toggle_camera()

    def _toggle_camera_menu(self):
        self._toggle_camera()
        if self._camera_worker and self._camera_worker.is_running:
            self._start_cam_action.setText("Stop Kamera")
        else:
            self._start_cam_action.setText("Start Kamera")

    def _on_camera_device_change(self, device_index: int):
        if self._camera_worker:
            self._camera_worker.set_device(device_index)
            self._config.set("camera.device_index", device_index)
            self._config.save()
        # Sync both spinboxes (RunPage and SettingsPage)
        self._run_page.get_device_spin().blockSignals(True)
        self._run_page.get_device_spin().setValue(device_index)
        self._run_page.get_device_spin().blockSignals(False)
        self._settings_page._cam_device.blockSignals(True)
        self._settings_page._cam_device.setValue(device_index)
        self._settings_page._cam_device.blockSignals(False)

    # ---- Camera Slots ----

    def _on_frame_received(self, pixmap):
        """Frame baru dari kamera — update display."""
        if self._heatmap_enabled and self._last_heatmap is not None and self._last_frame is not None:
            # Overlay heatmap on frame
            try:
                overlaid = overlay_heatmap(self._last_frame, self._last_heatmap, alpha=0.4)
                rgb = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
                if not qimg.isNull():
                    pixmap = QPixmap.fromImage(qimg)
            except Exception as e:
                logger.warning("Heatmap overlay error: %s", e)

        # Draw ROI rectangles on RUN page live view BEFORE sending to display
        if self._tabs.currentIndex() == 0 and self._current_all_rois:
            try:
                qp = QPainter(pixmap)
                qp.setRenderHint(QPainter.Antialiasing)
                font = QFont("Segoe UI", 9)
                qp.setFont(font)

                # Tentukan warna per-ROI berdasarkan hasil inferensi
                elapsed = time.monotonic() - self._roi_col_timestamp
                fresh = elapsed < self._roi_col_duration

                for i, roi_rect in enumerate(self._current_all_rois):
                    x, y, w, h = roi_rect

                    if fresh and i in self._roi_col_judgement:
                        jdg = self._roi_col_judgement[i]
                        color = "#22C55E" if jdg == "OK" else "#EF4444"
                    else:
                        color = "#F59E0B"  # orange — default / timeout

                    pen = QPen(QColor(color), 2)
                    qp.setPen(pen)
                    qp.setBrush(Qt.NoBrush)
                    qp.drawRect(x, y, w, h)
                    # Label text
                    label_x = x + 3
                    label_y = y - 4 if y >= 16 else y + 13
                    qp.setPen(QColor(color))
                    if i < len(self._current_all_roi_labels):
                        label = self._current_all_roi_labels[i]
                    else:
                        label = f"ROI{i+1}"  # fallback template lama tanpa label custom
                    qp.drawText(label_x, label_y, label)
                qp.end()
            except Exception as e:
                logger.warning("ROI overlay draw error: %s", e)

        # Draw gate ROI on RUN page when part check is active
        if self._tabs.currentIndex() == 0 and self._pc_active_for_overlay and self._last_gate_roi:
            try:
                qp_gate = QPainter(pixmap)
                qp_gate.setRenderHint(QPainter.Antialiasing)
                font = QFont("Segoe UI", 9)
                qp_gate.setFont(font)
                gx = int(self._last_gate_roi.get("x", 0))
                gy = int(self._last_gate_roi.get("y", 0))
                gw = int(self._last_gate_roi.get("width", 64))
                gh = int(self._last_gate_roi.get("height", 64))
                # Dynamic color: blue when waiting, green when ready
                gate_color = "#22C55E" if self._last_part_ready else "#3B82F6"
                pen = QPen(QColor(gate_color), 2, Qt.DashLine)
                qp_gate.setPen(pen)
                qp_gate.drawRect(gx, gy, gw, gh)
                # Label badge
                qp_gate.setPen(Qt.NoPen)
                qp_gate.setBrush(QColor(gate_color))
                label_w = 44
                label_y = gy - 16 if gy >= 16 else gy
                qp_gate.drawRect(gx, label_y, label_w, 16)
                qp_gate.setPen(QColor("#FFFFFF"))
                qp_gate.drawText(gx + 3, label_y + 12, "GATE")
                qp_gate.end()
            except Exception as e:
                logger.warning("Gate ROI overlay draw error: %s", e)

        self._run_page.set_frame(pixmap)
        # During import review mode, camera frames must NOT overwrite the ROI editor
        if self._is_import_mode:
            return
        if self._tabs.currentIndex() == 1:
            self._teach_page.set_preview(pixmap)

    def _on_camera_started(self):
        self._cam_status_label.setText("Camera Aktif")
        self._cam_status_label.setStyleSheet("color: #22C55E;")
        self._run_page.set_camera_status(True)
        self._start_cam_action.setText("Stop Kamera")
        self._teach_page.set_preview_text("")
        self.set_status("Kamera aktif", 3000)

    def _on_camera_stopped(self):
        self._cam_status_label.setText("Camera Mati")
        self._cam_status_label.setStyleSheet("color: #EF4444;")
        self._run_page.set_camera_status(False)
        self._start_cam_action.setText("Start Kamera")
        self._teach_page.set_preview_text("Kamera dimatikan")

    def _on_camera_error(self, msg: str):
        self._cam_status_label.setText("Camera Error")
        self._cam_status_label.setStyleSheet("color: #F59E0B;")
        self._run_page.set_camera_status(False)
        self._run_page.set_status_message(
            "Kamera tidak terdeteksi. Cek koneksi atau ganti device index di SETTINGS.")
        self.set_status(f"Kamera: {msg}. Coba device index lain di SETTINGS.", 5000)

    def _on_fps_updated(self, fps: float):
        self._fps_status_label.setText(f"FPS: {fps:.1f}")
        self._run_page.update_fps(fps)

    def _on_camera_status(self, msg: str):
        self._run_page.set_status_message(msg)

    # ---- Heatmap ----

    def _on_heatmap_toggle(self, enabled: bool):
        """Toggle heatmap overlay on/off."""
        self._heatmap_enabled = enabled
        if enabled:
            self._run_page.get_heatmap_button().setText("Heatmap ON")
        else:
            self._run_page.get_heatmap_button().setText("Heatmap")
        logger.info("Heatmap overlay: %s", "ON" if enabled else "OFF")

    # ---- Inference ----

    def _on_trigger_now(self):
        """Trigger inspeksi manual — tombol 'Trigger Now' / POST /trigger.

        Di mode manual/plc_trigger: frame berikutnya di-inspeksi sekali.
        Di mode continuous: hanya log (inspeksi sudah jalan terus).
        """
        self._plc_trigger_pending = True
        mode = self._config.get("inference.mode", "continuous")
        logger.info("Manual trigger dikirim (mode=%s)", mode)
        self.statusBar().showMessage(
            f"Trigger dikirim ({mode})", 3000)

    def _on_frame_for_inference(self, frame):
        """Run inference on frame from camera — per-ROI + aggregate.
        Only runs when on RUN tab. Respects cycle delay between inspections."""
        if not self._inference_engine.is_loaded:
            return
        if not self._current_all_rois:
            return
        # Only infer on RUN tab — other tabs don't show inference results
        if self._tabs.currentIndex() != 0:
            return
        # Skip frame if in cycle delay (jeda antar siklus)
        if self._cycle_delay_active:
            return
        # Mode plc_trigger/manual: inspeksi hanya saat ada trigger
        # (coil trigger PLC ON, tombol Trigger Now, atau POST /trigger).
        infer_mode = self._config.get("inference.mode", "continuous")
        if infer_mode in ("plc_trigger", "manual"):
            if not self._plc_trigger_pending:
                return
            self._plc_trigger_pending = False

        # ── Step 1: Part Presence Check (fail-safe gating) ──
        pc_cfg = self._current_part_check_cfg
        pc_state = pc_module.part_check_state(pc_cfg)
        if pc_state == "active":
            self._pc_active_for_overlay = True
            self._last_gate_roi = pc_cfg.get("gate_roi")
            try:
                pc_result = pc_module.evaluate_part_presence(
                    frame, pc_cfg["gate_roi"], pc_cfg)
            except Exception as e:
                logger.warning("Part check error: %s", e)
                pc_result = None
            if pc_result is None:
                # Exception during evaluation — fail-safe: block QC
                self._last_part_ready = False
                if self._ng_interval_timer.isActive():
                    self._ng_interval_timer.stop()
                    self._ng_interval_active = False
                self._run_page.set_waiting_for_part()
                return
            if not pc_result.ready:
                self._last_part_ready = False
                # Stop NG interval timer so phantom NG doesn't count
                if self._ng_interval_timer.isActive():
                    self._ng_interval_timer.stop()
                    self._ng_interval_active = False
                self._run_page.set_waiting_for_part()
                return
            _prev_ready = self._last_part_ready
            self._last_part_ready = True
            if not _prev_ready and self._get_io_mode()["part_ready_output"]:
                # Transisi part belum-ready → ready: pulse coil part_ready ke PLC
                # (opsional — default hanya OK/NG, sesuai konfigurasi io_mode)
                self._plc_pulse("part_ready")
            # Capture part check score untuk PG push
            m = pc_result.method
            if m == 'color' and pc_result.color_score is not None:
                self._last_part_check_score = pc_result.color_score
            elif m == 'edge' and pc_result.edge_score is not None:
                self._last_part_check_score = pc_result.edge_score
            elif m == 'both':
                cs = pc_result.color_score if pc_result.color_score is not None else 1.0
                es = pc_result.edge_score if pc_result.edge_score is not None else 1.0
                self._last_part_check_score = max(cs, es)
            else:
                self._last_part_check_score = 0.0
        elif pc_state == "incomplete":
            # Fail-safe: part check enabled but not fully configured
            # Block QC to prevent false NG 1.000 on empty scene
            self._pc_active_for_overlay = False
            self._last_gate_roi = None
            self._last_part_ready = False
            if self._ng_interval_timer.isActive():
                self._ng_interval_timer.stop()
                self._ng_interval_active = False
            self._run_page.set_part_check_incomplete(
                "Part-check aktif tapi belum lengkap: "
                "foto master / gate ROI belum diset")
            return
        else:  # "disabled"
            self._pc_active_for_overlay = False
            self._last_gate_roi = None
            self._last_part_ready = False
            # Fall through to QC (backward compat)
        # ── Step 2: QC Inference ──

        try:
            overall_ng = False
            worst_score = 1.0  # similarity: 1.0=terbaik, dicari terendah (paling anomali)
            total_latency = 0.0
            roi_results = []

            # ── Step 1.5: YOLO class pre-filter (opsional) ──
            # Kalau aktif: cek dulu part kelas yang diharapkan terdeteksi.
            # Kelas tidak cocok → NG langsung (tanpa scoring anomali).
            class_ng = False
            yc = self._yolo_cfg()
            if yc.get("enabled"):
                det = self._ensure_yolo_detector()
                if det is not None:
                    dets = det.detect(frame)
                    if dets is not None and not class_filter_matches(
                            dets, yc.get("expected_classes", []),
                            min_conf=float(yc.get("min_conf", 0.25))):
                        class_ng = True

            # ROI hanya dicek bila lolos filter kelas
            rois_to_check = [] if class_ng else self._current_all_rois
            for idx, roi_rect in enumerate(rois_to_check):
                roi_dict = {
                    "x": roi_rect[0], "y": roi_rect[1],
                    "width": roi_rect[2], "height": roi_rect[3],
                    "uid": (self._current_all_roi_uids[idx]
                            if idx < len(self._current_all_roi_uids) else None),
                }
                result = self._inference_engine.infer(frame, roi=roi_dict)
                roi_results.append({
                    "roi": roi_rect,
                    "label": (self._current_all_roi_labels[idx]
                              if idx < len(self._current_all_roi_labels)
                              else f"ROI{idx + 1}"),
                    "score": result.score,
                    "judgement": result.judgement,
                    "latency": result.latency_ms,
                })
                total_latency += result.latency_ms
                if result.score < worst_score:
                    worst_score = result.score
                    self._last_heatmap = result.heatmap
                    self._last_frame = frame
                if result.judgement == "NG":
                    overall_ng = True

            raw_judgement = "NG" if (overall_ng or class_ng) else "OK"
            self._last_worst_score = worst_score
            # Catat alasan NG karena filter kelas (untuk tampilan/status)
            self._last_class_filter_ng = bool(class_ng)

            # Simpan judgement per-ROI beserta timestamp untuk warna live
            self._roi_col_judgement = {}
            for idx, r in enumerate(roi_results):
                self._roi_col_judgement[idx] = r.get("judgement", "OK")
            self._roi_col_timestamp = time.monotonic()

            # Push ke PostgreSQL TIDAK di sini (per-frame = boros + tanpa
            # backpressure, lihat PRD R4). Push dilakukan per verdict-event
            # di blok add_inspection (NG pertama / OK throttled) via outbox.


            avg_latency = total_latency / len(roi_results) if roi_results else 0.0
            self._run_page.update_latency(avg_latency)
            self._run_page.update_roi_results(roi_results)

            # ---- NG Interval Timer ----
            self._last_judgement = raw_judgement
            self._last_worst_score = worst_score
            if raw_judgement == "OK":
                # Stop interval timer — anomaly cleared
                if self._ng_interval_timer.isActive():
                    self._ng_interval_timer.stop()
                    self._ng_interval_active = False
                # Show OK immediately, increment OK counter
                self._run_page.update_judgement("OK", worst_score)
                self._inspection_ok += 1
                self._run_page.update_counters(
                    self._inspection_ok, self._inspection_ng)
                # Feedback ke PLC: publikasi hasil OK (latching/one-shot sesuai io_mode)
                self._publish_result("OK")

            else:  # raw_judgement == "NG"
                if not self._ng_interval_timer.isActive():
                    # First NG frame — start interval timer
                    delay = self._settings_page.get_ng_debounce_ms()
                    effective_delay = max(delay, 50)  # minimum 50ms agar timer tetap jalan
                    self._ng_interval_timer.start(effective_delay)
                    self._ng_interval_active = True
                    # Show NG immediately on display (counter hanya bertambah via timer tick)
                    self._run_page.update_judgement("NG", worst_score)
                    # Feedback ke PLC: publikasi hasil NG (latching/one-shot sesuai io_mode)
                    self._publish_result("NG")
                    # Save frame untuk tuning
                    img_path = self._save_inspection_frame(
                        frame, "NG", worst_score, roi_results, avg_latency)
                    # Simpan ke SQLite agar entry bisa di-tuning
                    roi_region = json.dumps([{
                        "x": r["roi"][0], "y": r["roi"][1],
                        "width": r["roi"][2], "height": r["roi"][3],
                        "label": r.get("label", f"ROI{i + 1}"),
                        "score": r["score"], "judgement": r["judgement"],
                    } for i, r in enumerate(roi_results)])
                    ng_entry_id = self._db.add_inspection({
                        "program": self._active_program,
                        "score": worst_score,
                        "judgement": "NG",
                        "threshold": self._inference_engine.threshold,
                        "latency_ms": avg_latency,
                        "image_path": img_path or "",
                        "roi_region": roi_region,
                        "metadata": {
                            "num_rois": len(roi_results),
                            "template": self._active_template,
                            "template_name": self._active_partname,
                        },
                    })
                    # Sink ke PostgreSQL via outbox (C1/C3) — local_id = id SQLite
                    # agar koreksi operator bisa di-propagasi (C0).
                    self._push_inspection_async(self._build_push_entry(
                        "NG", worst_score, img_path, avg_latency, ng_entry_id))
                else:
                    # Timer already running — update display (worse score)
                    self._run_page.update_judgement("NG", worst_score)

            # ── Cycle delay: jeda antar siklus inspeksi ──
            # Hanya berlaku di mode continuous (auto sequence). Di mode
            # plc_trigger/manual, timing antar part dipegang ladder PLC —
            # aplikasi tidak boleh menahan siklus (filosofi Keyence IV3).
            if self._config.get("inference.mode", "continuous") == "continuous":
                cycle_delay = self._settings_page.get_cycle_delay_ms()
                if cycle_delay > 0:
                    # Stop NG interval timer so it doesn't phantom-count during delay
                    if self._ng_interval_timer.isActive():
                        self._ng_interval_timer.stop()
                        self._ng_interval_active = False
                    self._cycle_delay_timer.start(cycle_delay)
                    self._cycle_delay_active = True
                    self._run_page.set_status_message(
                        f"Cycle delay {cycle_delay} ms...")
                else:
                    self._cycle_delay_active = False
            else:
                # plc_trigger/manual: timing antar siklus dari PLC — tanpa jeda
                self._cycle_delay_active = False

            # Diagnostics latency
            self._diagnostics_page.update_performance(
                0, 0, 0,
                self._inference_engine.latency_avg_ms,
                self._inference_engine.latency_p95_ms,
            )

            # Simpan gambar + riwayat SQLite di-throttle (mahal) tiap 30 frame OK
            if raw_judgement == "OK":
                self._inference_save_counter += 1
                if self._inference_save_counter % 30 == 0:
                    img_path = self._save_inspection_frame(
                        frame, "OK", worst_score, roi_results, avg_latency)
                    roi_region = json.dumps([{
                        "x": r["roi"][0], "y": r["roi"][1],
                        "width": r["roi"][2], "height": r["roi"][3],
                        "label": r.get("label", f"ROI{i + 1}"),
                        "score": r["score"], "judgement": r["judgement"],
                    } for i, r in enumerate(roi_results)])
                    ok_entry_id = self._db.add_inspection({
                        "program": self._active_program,
                        "score": worst_score,
                        "judgement": "OK",
                        "threshold": self._inference_engine.threshold,
                        "latency_ms": avg_latency,
                        "image_path": img_path or "",
                        "roi_region": roi_region,
                        'metadata': {'num_rois': len(roi_results),
                                      'template': self._active_template,
                                      'template_name': self._active_partname},
                    })
                    self._push_inspection_async(self._build_push_entry(
                        "OK", worst_score, img_path, avg_latency, ok_entry_id))

        except Exception as e:
            logger.warning("Inference error: %s", e)

    def _build_push_entry(self, judgement: str, score: float,
                          img_path: str, latency: float, local_id: int) -> dict:
        """Bangun dict kwargs untuk ``PostgresDB.push_inspection`` (C1).

        Perbaikan mapping lama: ``mpcheck`` = verdict OK/NG (sebelumnya
        salah diisi nama operator). Nama operator pindah ke kolom ``operator``;
        ``line`` diambil dari config (sebelumnya tidak pernah ditulis).
        """
        operator = ""
        if self._current_user:
            operator = (self._current_user.get("display_name")
                        or self._current_user.get("username", ""))
        return {
            "partname": self._active_partname or self._active_program,
            "mpcheck": judgement,                                   # verdict OK/NG
            "operator": operator,
            "data1": self._last_part_check_score or 0.0,
            "data2": float(score),
            "image_path": img_path or "",
            "threshold": getattr(self._inference_engine, "threshold", None),
            "latency_ms": float(latency) if latency is not None else None,
            "local_id": int(local_id) if local_id is not None else None,
            "line": (self._config.get("postgresql.line", "")
                     or self._config.get("line_name", "") or ""),
        }

    def _push_inspection_async(self, entry: dict) -> None:
        """Enqueue hasil inspeksi ke outbox SQLite, lalu flush ke PostgreSQL (C3).

        Outbox tahan-restart: bila PG/jaringan bermasalah, entry tetap
        tersimpan dan di-retry oleh ``_flush_pg_outbox`` (dipanggil dari
        thread ini, QTimer 30 detik, dan saat startup). Hasil tidak pernah
        hilang diam-diam.
        """
        if not self._pg.is_enabled:
            return
        try:
            self._db.add_outbox(entry)
        except Exception as e:
            logger.warning("Outbox enqueue error: %s", e)
            return
        threading.Thread(target=self._flush_pg_outbox, daemon=True).start()

    def _flush_pg_outbox(self) -> None:
        """Kirim batch outbox ke PostgreSQL; sukses → hapus (nol duplikat).

        Satu worker + batch berbatas (PRD R4): tidak ada satu koneksi per
        frame, tidak ada thread menumpuk. Antrian bounded: bila membengkak,
        entry tertua dibuang dan dicatat.
        """
        if not self._pg.is_enabled:
            return
        try:
            batch = self._db.get_outbox(limit=200)
            if not batch:
                return
            ok_ids = []
            for item in batch:
                try:
                    rid = self._pg.push_inspection(**item["entry"])
                    if rid is not None:
                        ok_ids.append(item["id"])
                except Exception as e:
                    logger.warning("Push outbox item %s gagal: %s", item["id"], e)
            if ok_ids:
                self._db.delete_outbox(ok_ids)
            failed = [i["id"] for i in batch if i["id"] not in ok_ids]
            if failed:
                self._db.bump_outbox_attempts(failed)
                logger.warning(
                    "Outbox: %d entry tertunda (PostgreSQL tidak terjangkau?)",
                    len(failed))
                total = self._db.count_outbox()
                if total > 5000:
                    self._db.drop_oldest_outbox(total - 5000)
        except Exception as e:
            logger.warning("Flush outbox error: %s", e)

    # ---- Save Inspection Frame (untuk Tuning) ----

    def _save_inspection_frame(self, frame, judgement: str, score: float,
                                roi_results: list, latency: float) -> str:
        """Save frame + per-ROI data to disk for Tuning mode.

        Returns image_path string, or empty string on failure.
        """
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            img_dir = self._data_dir / "inspection_images"
            img_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{ts}_{uuid.uuid4().hex[:8]}.png"
            dest = img_dir / fname
            cv2.imwrite(str(dest), frame)

            # Save per-ROI metadata alongside
            meta = {
                "timestamp": ts,
                "program": self._active_program,
                "template": self._active_template,
                "template_name": self._active_partname,
                "judgement": judgement,
                "score": score,
                "threshold": self._inference_engine.threshold,
                "latency_ms": latency,
                "operator": (self._current_user.get("display_name")
                             or self._current_user.get("username", "")
                            ) if self._current_user else "",
                "rois": [{
                    "x": r["roi"][0], "y": r["roi"][1],
                    "width": r["roi"][2], "height": r["roi"][3],
                    "score": r["score"], "judgement": r["judgement"],
                } for r in roi_results],
            }
            meta_path = dest.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            return str(dest)
        except Exception as e:
            logger.warning("Save inspection frame error: %s", e)
            return ""

    # ---- NG Interval Timer ----

    def _on_ng_interval_tick(self):
        """Interval timer tick — NG still ongoing, count another NG."""
        self._inspection_ng += 1
        self._run_page.update_counters(self._inspection_ok, self._inspection_ng)
        # Catatan: push ke PostgreSQL sekarang dilakukan per inferensi di
        # _on_frame_for_inference (setiap hasil OK & NG), jadi tick ini hanya
        # menambah counter agar tidak dobel-push.
        # Timer auto-restarts (QTimer with interval keeps firing)

    def _on_cycle_delay_tick(self):
        """Cycle delay timer tick — ready for next inspection cycle."""
        self._cycle_delay_active = False
        self._run_page.set_status_message("Siap")
 
     # ---- Part Presence Check ----

    def _on_part_check_config_changed(self):
        """Save part check UI state to template config and refresh cache."""
        if not self._active_template:
            return
        pc = self._teach_page
        updates = {
            "enabled": pc.get_pc_enabled_cb().isChecked(),
            "method": pc.get_pc_method_combo().currentData(),
            "color_threshold": pc.get_pc_color_th_spin().value(),
            "edge_threshold": pc.get_pc_edge_th_spin().value(),
            "canny_low": pc.get_pc_canny_low_spin().value(),
            "canny_high": pc.get_pc_canny_high_spin().value(),
        }
        try:
            self._pm.update_part_check_config(
                self._active_program, self._active_template, updates)
            self._refresh_part_check_gate_cache()
        except Exception as e:
            logger.warning("Part check config save error: %s", e)

    def _on_training_config_changed(self):
        """Save Training Profile UI state (algorithm/backbone/coreset) to
        template config. Training itself always rebuilds the model from all
        images in the template's folder (see TrainingPipeline.train) — so
        there's no incremental-learning corruption risk from switching
        backbone. The only thing worth flagging is that the model file
        currently on disk still reflects the OLD setting until the user
        clicks TRAIN again."""
        if not self._active_template:
            return
        updates = self._teach_page.get_training_config()
        try:
            old_cfg = self._pm.get_template_config(
                self._active_program, self._active_template)
            changed_deploy_relevant = (
                old_cfg.get("trained")
                and (old_cfg.get("backbone") != updates.get("backbone")
                     or old_cfg.get("algorithm") != updates.get("algorithm")))
            self._pm.update_template_config(
                self._active_program, self._active_template, updates)
            if changed_deploy_relevant:
                self.set_status(
                    "Algorithm/Backbone diubah — klik TRAIN untuk "
                    "menerapkan ke model (model saat ini masih pakai setting lama).",
                    5000)
        except Exception as e:
            logger.warning("Training config save error: %s", e)

    def _on_augmentation_config_changed(self):
        """Save Augmentasi Data UI state to template config. Actual generation
        happens on the next TRAIN click (training_worker._do_training) — this
        just persists what should be generated, mirroring _on_training_config_changed."""
        if not self._active_template:
            return
        updates = self._teach_page.get_augmentation_config()
        try:
            self._pm.update_augmentation_config(
                self._active_program, self._active_template, updates)
        except Exception as e:
            logger.warning("Augmentation config save error: %s", e)

    def _on_gate_roi_changed(self):
        """Save gate ROI from editor to template config."""
        if not self._active_template:
            return
        gate_rois = self._teach_page.get_gate_roi()
        gate_roi = gate_rois[0] if gate_rois else None
        try:
            self._pm.update_part_check_config(
                self._active_program, self._active_template,
                {"gate_roi": gate_roi})
            self._refresh_part_check_gate_cache()
        except Exception as e:
            logger.warning("Gate ROI save error: %s", e)

    def _on_capture_master(self):
        """Capture current frame as master photo for part check."""
        if not self._active_template:
            self.set_status("Tidak ada template aktif!", 3000)
            return
        if not self._camera_worker or not self._camera_worker.is_running:
            self.set_status("Kamera tidak aktif!", 3000)
            return
        gate_rois = self._teach_page.get_gate_roi()
        if not gate_rois:
            self.set_status("Gambar gate ROI dulu!", 3000)
            return
        gate_roi = gate_rois[0]
        frame = self._camera_worker.get_frame()
        if frame is None:
            self.set_status("Gagal ambil frame!", 3000)
            return
        try:
            pc_updates = self._pm.save_part_check_master(
                self._active_program, self._active_template,
                frame, gate_roi,
                canny_low=self._teach_page.get_pc_canny_low_spin().value(),
                canny_high=self._teach_page.get_pc_canny_high_spin().value(),
            )
            self._refresh_part_check_ui()
            self._refresh_part_check_gate_cache()
            self.set_status("Foto master part tersimpan!", 3000)
            # Peringatan bila metode edge — deteksi pergeseran posisi terbatas
            method = self._teach_page.get_pc_method_combo().currentData()
            if method == "edge":
                logger.warning(
                    "Metode Tepi (Canny) hanya membandingkan jumlah edge pixel, "
                    "bukan posisinya. Part yang bergeser dalam ROI masih bisa "
                    "dianggap 'ready'. Jika part sering bergeser, "
                    "coba metode Warna atau Keduanya."
                )
        except Exception as e:
            self.set_status(f"Gagal simpan master: {e}", 5000)

    def _refresh_part_check_ui(self):
        """Load part check config + gate ROI + master thumbnail into TEACH UI."""
        if not self._active_template:
            return
        try:
            pc_cfg = self._pm.get_part_check_config(
                self._active_program, self._active_template)
        except Exception:
            pc_cfg = {}
        self._teach_page.set_part_check_config(pc_cfg)

        # Restore gate ROI from config
        gate_roi_dict = pc_cfg.get("gate_roi")
        if gate_roi_dict:
            self._teach_page.set_gate_roi([gate_roi_dict])

        # Master thumbnail — Canny edge preview hanya untuk metode edge/both
        master_path = self._pm.get_part_check_master_image_path(
            self._active_program, self._active_template)
        if master_path and master_path.exists():
            method = pc_cfg.get("method", "both")
            show_edge = method in ("edge", "both")
            from PySide6.QtGui import QPixmap, QImage

            if show_edge:
                import cv2
                import numpy as np
                master_bgr = cv2.imread(str(master_path))
                if master_bgr is not None and master_bgr.size > 0:
                    gray = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2GRAY)
                    canny_low = self._teach_page.get_pc_canny_low_spin().value()
                    canny_high = self._teach_page.get_pc_canny_high_spin().value()
                    edges = cv2.Canny(gray, canny_low, canny_high)
                    # Green edges on dark background
                    preview = np.zeros((*edges.shape, 3), dtype=np.uint8)
                    preview[edges > 0] = [34, 197, 94]  # BGR = hijau #22C55E
                    h, w = preview.shape[:2]
                    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                    qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
                    pix = QPixmap.fromImage(qimg)
                else:
                    pix = QPixmap(str(master_path))
            else:
                pix = QPixmap(str(master_path))
            self._teach_page.set_master_status(
                pc_cfg.get("has_master", False),
                pc_cfg.get("master_captured_at", ""), pix)
        else:
            self._teach_page.set_master_status(False)

    def _refresh_part_check_gate_cache(self):
        """Reload part check config from disk into in-memory cache."""
        if not self._active_template:
            self._current_part_check_cfg = {}
            return
        try:
            self._current_part_check_cfg = self._pm.get_part_check_config(
                self._active_program, self._active_template)
        except Exception:
            self._current_part_check_cfg = {}

    # ---- Programs / Templates ----

    def _new_template_defaults(self) -> dict:
        """Config overrides seeded into a newly created template, sourced
        from the global Settings 'Model' section. Settings no longer edits
        existing templates directly — per-template tuning lives in the
        Training Profile panel on the TEACH tab."""
        return {
            "algorithm": self._config.get("model.algorithm", "patchcore"),
            "backbone": self._config.get("model.backbone", "resnet18"),
            "input_size": self._config.get("model.input_size", 256),
        }

    def _init_programs(self):
        """Load programs and create default if none exist."""
        programs = self._pm.list_programs()
        if not programs:
            prog = self._pm.create_program("Default")
            self._active_program = prog["name"]
            # Create default template
            tmpl = self._pm.create_template(
                self._active_program, "Template 1",
                config=self._new_template_defaults())
            self._active_template = tmpl["id"]
            self._pm.set_active_template(self._active_program, self._active_template)
        else:
            self._active_program = programs[0]["name"]
            templates = self._pm.list_templates(self._active_program)
            if templates:
                active_id = self._pm.get_active_template(self._active_program)
                if active_id and any(t["id"] == active_id for t in templates):
                    self._active_template = active_id
                else:
                    self._active_template = templates[0]["id"]
                    self._pm.set_active_template(self._active_program, self._active_template)

        # Cache nama template untuk push PostgreSQL (partname). Saat startup
        # _active_template di-set langsung tanpa lewat _activate_template, jadi
        # _active_partname harus di-isi di sini — kalau tidak, push memakai
        # fallback nama program ("Default") alih-alih nama template.
        if self._active_template:
            _tc = self._pm.get_template_config(
                self._active_program, self._active_template)
            self._active_partname = _tc.get("name", self._active_template)

        self._refresh_template_ui()
        self._load_template_model()
        self._program_label.setText(f"Program: {self._active_program}")
        logger.info("Active program: %s, template: %s", self._active_program, self._active_template)

    # ═══════════════════════════ PLC — Modbus Master ═══════════════════════════
    # Sistem = MASTER, PLC = slave. Semua alamat coil/register dari
    # config "plc.io_map" (Settings → PLC → IO Mapping) — ganti PLC tinggal
    # ganti angka di config, tanpa edit kode.

    def _build_plc_config(self) -> dict:
        """Kumpulkan konfigurasi PLC dari Config → dict untuk ModbusRTUManager."""
        return {
            "port": self._config.get("plc.port", "COM1"),
            "baudrate": self._config.get("plc.baudrate", 9600),
            "parity": self._config.get("plc.parity", "N"),
            "bytesize": 8,
            "stopbits": 1,
            "timeout": 1.0,
            "modbus_slave_id": self._config.get("plc.modbus_slave_id", 1),
            "pulse_ms": self._config.get("plc.pulse_ms", 300),
            "io_map": self._config.get("plc.io_map", {}),
        }

    def _init_plc(self):
        """Inisialisasi ModbusRTUManager dari config + connect + start poll."""
        if not self._config.get("plc.enabled", False):
            return
        if not HAS_MODBUS:
            logger.warning("PLC enabled tapi pymodbus tidak terinstall")
            return
        try:
            self._plc_modbus = ModbusRTUManager(self._build_plc_config())
        except Exception as e:
            logger.error("PLC init error: %s", e)
            return
        self._plc_modbus.set_on_status_change(self._on_plc_status_change)
        if not self._plc_modbus.connect():
            # Gagal connect — ModbusRTUManager akan auto-retry saat
            # user tekan tombol Scan/Deteksi atau restart app.
            logger.warning("PLC connect gagal saat startup")
            return
        self._start_plc_polling()

    def _start_plc_polling(self):
        if self._plc_poll_timer is None:
            self._plc_poll_timer = QTimer(self)
            self._plc_poll_timer.setInterval(200)  # 5 Hz — poll input PLC
            self._plc_poll_timer.timeout.connect(self._on_plc_poll_tick)
        self._plc_poll_timer.start()

    def _stop_plc_polling(self):
        if self._plc_poll_timer is not None:
            self._plc_poll_timer.stop()

    def _on_plc_status_change(self, connected: bool):
        """Status PLC berubah — update label di RUN page + mulai/henti poll."""
        try:
            self._run_page.set_plc_status(connected)
        except Exception:
            pass
        try:
            if connected:
                self._settings_page.set_plc_status(
                    True, str(self._config.get("plc.port", "")))
            else:
                self._settings_page.set_plc_status(False)
        except Exception:
            pass
        if connected:
            self._start_plc_polling()
        else:
            self._stop_plc_polling()
        # Sinkronkan I/O Monitor (halaman I/O Settings) dgn status koneksi
        try:
            self._io_page.refresh_monitor_connection()
        except Exception:
            pass

    def _on_io_settings_apply(self, io_map: dict, io_mode: dict):
        """Terapkan pemetaan coil & perilaku hasil dari halaman I/O Settings."""
        try:
            self._config.set("plc.io_map", io_map)
            self._config.set("plc.io_mode", io_mode)
            self._config.save()
        except Exception as e:
            logger.warning("I/O settings save error: %s", e)
            self.set_status("I/O Settings gagal disimpan", 3000)
            return
        # Re-init PLC agar io_map baru berlaku di ModbusRTUManager
        plc_cfg = self._config.get("plc") or {}
        self._shutdown_plc()
        if plc_cfg.get("enabled"):
            self._init_plc()
        try:
            self._io_page.refresh_monitor_connection()
        except Exception:
            pass
        self.set_status("I/O Settings tersimpan & diterapkan", 3000)

    def _on_plc_poll_tick(self):
        """Poll input PLC tiap 200ms — deteksi trigger/reset/switch program."""
        if not self._plc_modbus or not self._plc_modbus.is_connected:
            return
        # Sync coil busy dengan state kamera (level, bukan pulse) — hanya
        # bila konfigurasi io_mode menyalakan busy_output (default hanya OK/NG).
        if self._get_io_mode()["busy_output"]:
            busy = bool(self._camera_worker and self._camera_worker.is_running)
            self._plc_modbus.set_output("busy", busy)
        try:
            events = self._plc_modbus.read_inputs()
        except Exception as e:
            logger.warning("PLC read inputs error: %s", e)
            return
        if events.get("trigger"):
            self._on_plc_trigger()
        if events.get("reset_result"):
            self._on_plc_reset()
        if events.get("switch_program"):
            prog = self._plc_modbus.read_program_register()
            if prog is not None:
                self._on_plc_switch_program(prog)

    def _on_plc_trigger(self):
        """PLC minta 1 siklus inspeksi (coil trigger ON)."""
        mode = self._config.get("inference.mode", "continuous")
        if mode == "plc_trigger":
            self._plc_trigger_pending = True
            self.set_status("PLC: trigger inspeksi", 2000)
        # Mode continuous: trigger hanya melewati cycle delay
        self._cycle_delay_active = False

    def _on_plc_reset(self):
        """IN reset: matikan semua coil hasil + reset counter."""
        if self._plc_modbus:
            self._plc_modbus.reset_outputs()
        self._inspection_ok = 0
        self._inspection_ng = 0
        try:
            self._run_page.update_counters(0, 0)
        except Exception:
            pass
        self.set_status("PLC: reset OK", 3000)

    def _on_plc_switch_program(self, program_number: int):
        """IN switch program: ganti template aktif dari nomor register PLC."""
        try:
            programs = self._pm.list_programs()
            if not programs:
                return
            idx = program_number - 1  # PLC: 1 = program pertama
            if not (0 <= idx < len(programs)):
                self.set_status(f"PLC: program #{program_number} tidak ada", 3000)
                return
            prog = programs[idx]
            if prog["name"] == self._active_program:
                return
            self._active_program = prog["name"]
            templates = self._pm.list_templates(self._active_program)
            if templates:
                active_id = self._pm.get_active_template(self._active_program)
                if active_id and any(t["id"] == active_id for t in templates):
                    self._active_template = active_id
                else:
                    self._active_template = templates[0]["id"]
                    self._pm.set_active_template(self._active_program,
                                                 self._active_template)
            self._refresh_template_ui()
            self._load_template_model()
            self._program_label.setText(f"Program: {self._active_program}")
            self.set_status(f"PLC: program → {self._active_program}", 3000)
        except Exception as e:
            logger.warning("PLC switch program error: %s", e)

    def _get_io_mode(self) -> dict:
        """I/O behaviour mode aktif (dari config plc.io_mode, selalu lengkap)."""
        return build_io_mode(self._config.get("plc"))

    # ---- YOLO class filter (Fase D) ----

    def _yolo_cfg(self) -> dict:
        cfg = self._config.get("yolo")
        return cfg if isinstance(cfg, dict) else {}

    def _ensure_yolo_detector(self):
        """Lazy-load YOLODetector dari config yolo.model_path (None bila gagal)."""
        if self._yolo_det is not None:
            return self._yolo_det
        cfg = self._yolo_cfg()
        path = str(cfg.get("model_path") or "").strip()
        if not cfg.get("enabled") or not path:
            return None
        try:
            det = YOLODetector(path)
            if not det.available:
                logger.warning("YOLO filter nonaktif: %s", det.error)
                return None
            self._yolo_det = det
            logger.info(
                "YOLO filter aktif: %s | kelas: %s | min_conf: %s",
                path, cfg.get("expected_classes"), cfg.get("min_conf"))
        except Exception as e:
            logger.warning("YOLO load error: %s", e)
            return None
        return self._yolo_det

    def _publish_result(self, judgement: str):
        """Publikasi hasil OK/NG ke PLC sesuai `plc.io_mode.output_mode`.

        - latching: coil OK/NG di-hold sebagai LEVEL sampai hasil berikutnya
          ditulis / PLC reset (persis "kept until next status result" IV3).
        - one_shot: pulse — tunda `one_shot_delay_ms`, lalu ON selama
          `one_shot_on_time_ms` (mirip "One-Shot" IV3).
        PLC yang memegang timing antar part; aplikasi hanya memublikasikan hasil.
        """
        if not self._plc_modbus or not self._plc_modbus.is_connected:
            return
        # Rollout shadow mode: NG hanya ditampilkan & dicatat, coil result_ng
        # TIDAK ditulis — lini tidak berhenti sebelum akurasi terbukti.
        # OK tetap dipublikasikan normal (coil result_ok).
        if judgement == "NG" and self._config.get("rollout.shadow_mode", False):
            logger.warning(
                "SHADOW MODE: NG ditekan dari coil (tidak diteruskan ke PLC)")
            return
        io = self._get_io_mode()
        if io["output_mode"] == "one_shot":
            self._publish_one_shot(judgement, io)
        else:
            self._publish_latching(judgement)

    def _publish_latching(self, judgement: str):
        """Tulis hasil sebagai LEVEL; coil lawan di-OFF-kan agar PLC bisa
        deteksi rising/falling edge (pola ladder KV Keyence: INC pd edge OK/NG)."""
        ok = (judgement == "OK")
        self._plc_modbus.set_output("result_ok", ok)
        self._plc_modbus.set_output("result_ng", not ok)

    def _publish_one_shot(self, judgement: str, io: dict):
        """Pulse singkat sesuai io_mode (delay + ON time) tanpa blokir UI."""
        name = "result_ok" if judgement == "OK" else "result_ng"
        duration = max(0, int(io.get("one_shot_on_time_ms", 300)))
        delay = max(0, int(io.get("one_shot_delay_ms", 0)))

        def _fire():
            if self._plc_modbus and self._plc_modbus.set_output(name, True):
                if duration > 0:
                    QTimer.singleShot(duration,
                                      lambda: self._safe_plc_output_off(name))

        if delay > 0:
            QTimer.singleShot(delay, _fire)
        else:
            _fire()

    def _plc_pulse(self, name: str):
        """Pulse coil output tanpa memblokir UI: ON → QTimer singleShot → OFF."""
        if not self._plc_modbus or not self._plc_modbus.is_connected:
            return
        ms = max(0, int(self._config.get("plc.pulse_ms", 300)))
        if not self._plc_modbus.set_output(name, True):
            return
        if ms > 0:
            QTimer.singleShot(ms, lambda: self._safe_plc_output_off(name))

    def _safe_plc_output_off(self, name: str):
        if self._plc_modbus:
            self._plc_modbus.set_output(name, False)

    def _on_plc_scan(self):
        """Tombol 'Scan Coils' (I/O Settings): probe alamat valid di background."""
        if not self._plc_modbus or not self._plc_modbus.is_connected:
            self._io_page.set_scan_result(
                "⚠️ PLC belum connect — cek Settings → PLC → Enable + Save")
            return
        max_addr = max(0, int(self._config.get("plc.scan_range", 127)))
        self._io_page.set_scan_result(
            f"Scan coil 0-{max_addr}... (bisa ±15-30 detik)")
        self._io_page.set_scan_busy(True)

        def _worker():
            valid = self._plc_modbus.scan_coils(max_addr)
            self._plc_scan_done_signal.emit(valid)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_plc_detect_active(self):
        """Tombol 'Deteksi Aktif' (I/O Settings): cari coil yang sedang ON."""
        if not self._plc_modbus or not self._plc_modbus.is_connected:
            self._io_page.set_scan_result(
                "⚠️ PLC belum connect — cek Settings → PLC → Enable + Save")
            return
        max_addr = max(0, int(self._config.get("plc.scan_range", 127)))
        self._io_page.set_scan_result(
            f"Deteksi coil aktif 0-{max_addr}... tekan tombol fisik di PLC")
        self._io_page.set_scan_busy(True)

        def _worker():
            active = self._plc_modbus.find_active_coils(max_addr)
            self._plc_detect_done_signal.emit(active)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_plc_scan_done(self, valid_coils: list):
        self._io_page.set_scan_busy(False)
        if valid_coils:
            n = len(valid_coils)
            shown = ", ".join(str(c) for c in valid_coils[:20])
            more = f" ... (+{n - 20})" if n > 20 else ""
            self._io_page.set_scan_result(
                f"✓ {n} coil valid: {shown}{more} — ketik alamatnya "
                "di Output/Input Assign di atas")
        else:
            self._io_page.set_scan_result(
                "⚠️ Tidak ada coil valid — cek port/ID slave/kabel")

    def _on_plc_detect_done(self, active_coils: list):
        self._io_page.set_scan_busy(False)
        if active_coils:
            self._io_page.set_scan_result(
                f"⚡ {len(active_coils)} coil aktif: "
                + ", ".join(str(c) for c in active_coils)
                + " — cocokkan dengan input fisik PLC, lalu ketik di "
                  "Input Assign di atas")
        else:
            self._io_page.set_scan_result(
                "Tidak ada coil aktif — pastikan tombol fisik ditekan saat scan")

    def _shutdown_plc(self):
        """Matikan semua coil output + tutup port (dipanggil saat keluar)."""
        self._stop_plc_polling()
        if self._plc_modbus:
            self._plc_modbus.reset_outputs()
            self._plc_modbus.disconnect()
            self._plc_modbus = None

    # ═══════════════════════════ Flask API (opsional) ═══════════════════════════
    # REST API lokal di 127.0.0.1 untuk integrasi eksternal.
    # Aktif hanya jika "Enable Flask API" di Settings dicentang.

    def _init_flask(self):
        """Init FlaskAPI dari config saat startup. Bind HANYA 127.0.0.1."""
        if not HAS_FLASK:
            self._settings_page.set_flask_status(False, "Flask belum terinstall")
            return
        cfg = self._config.get("flask_api", {})
        if not cfg.get("enabled", False):
            self._settings_page.set_flask_status(False)
            return
        self._start_flask(
            port=int(cfg.get("port", 5000)),
            api_key=str(cfg.get("api_key", "")),
        )

    def _start_flask(self, port: int, api_key: str) -> None:
        if not HAS_FLASK:
            self._settings_page.set_flask_status(False, "Flask belum terinstall")
            return
        if self._flask_api and self._flask_api.is_running:
            # Server tidak bisa pindah port tanpa restart — kasih tahu user.
            if self._flask_api._port != port:
                self._settings_page.set_flask_status(
                    False, f"Port berubah — restart aplikasi (masih jalan di {self._flask_api._port})")
            else:
                self._settings_page.set_flask_status(True, f"127.0.0.1:{port}")
            return
        self._shutdown_flask()
        self._flask_api = FlaskAPI(
            port=port,
            api_key=api_key,
            get_status_fn=self._api_get_status,
            get_last_result_fn=self._api_get_last_result,
            trigger_inspection_fn=self._on_trigger_now,
            get_history_fn=self._api_get_history,
            activate_program_fn=self._api_activate_program,
        )
        self._flask_api.start()
        if self._flask_api.is_running:
            self._settings_page.set_flask_status(True, f"127.0.0.1:{port}")
        else:
            self._settings_page.set_flask_status(False, "gagal start — cek log")

    def _shutdown_flask(self):
        if self._flask_api:
            self._flask_api.stop()
            self._flask_api = None
        self._settings_page.set_flask_status(False)

    def _apply_flask_settings(self, settings: dict):
        """Terapkan config Flask dari Settings saat save."""
        if not HAS_FLASK:
            self._settings_page.set_flask_status(False, "Flask belum terinstall")
            return
        fl = settings.get("flask_api", {})
        if not fl.get("enabled", False):
            self._shutdown_flask()
            return
        # api_key tidak ada di UI — ambil dari config tersimpan agar tidak
        # berganti tiap save (FlaskAPI akan generate key baru kalau kosong)
        api_key = str(self._config.get("flask_api.api_key", ""))
        self._start_flask(
            port=int(fl.get("port", 5000)),
            api_key=api_key,
        )

    # ---- Callbacks untuk endpoint Flask ----

    def _api_get_status(self) -> dict:
        return {
            "app": "VisionInspect",
            "program": self._active_program,
            "template": self._active_template,
            "camera_running": bool(self._camera_thread and self._camera_thread.isRunning()),
            "plc_enabled": bool(self._plc_modbus and self._plc_modbus.is_connected),
            "inference_mode": self._config.get("inference.mode", "continuous"),
        }

    def _api_get_last_result(self) -> dict:
        return {
            "judgement": self._last_judgement,
            "score": round(self._last_worst_score, 4),
            "program": self._active_program,
            "template": self._active_template,
        }

    def _api_get_history(self, limit: int = 100) -> list:
        try:
            return self._db.get_history(limit=max(1, min(int(limit), 500)))
        except Exception as e:
            logger.warning("Flask /history error: %s", e)
            return []

    def _api_activate_program(self, name: str) -> None:
        # Cari template (di program aktif) yang id atau nama-nya cocok
        for t in self._pm.list_templates(self._active_program):
            cfg = self._pm.get_template_config(self._active_program, t["id"])
            if t["id"] == name or cfg.get("name") == name:
                self._activate_template(t["id"])
                return
        raise ValueError(f"Template tidak ditemukan: {name}")

    def _refresh_template_ui(self):
        """Sync template selector (TEACH + RUN) + counts from disk."""
        templates = self._pm.list_templates(self._active_program)
        current_id = self._active_template

        teach_combo = self._teach_page.get_template_combo()
        run_combo = self._run_page.get_template_combo()

        # Block signals on both combos to avoid recursive template_changed
        teach_combo.blockSignals(True)
        run_combo.blockSignals(True)

        teach_combo.clear()
        run_combo.clear()
        for t in templates:
            name = t["config"].get("name", t["id"])
            trained = "✓" if t["config"].get("trained") else "○"
            display = f"{trained} {name}"
            teach_combo.addItem(display, t["id"])
            run_combo.addItem(display, t["id"])

        # Set current index on both combos
        for i in range(teach_combo.count()):
            if teach_combo.itemData(i) == current_id:
                teach_combo.setCurrentIndex(i)
                run_combo.setCurrentIndex(i)
                break

        teach_combo.blockSignals(False)
        run_combo.blockSignals(False)

        # Counts
        if self._active_template:
            ok_count = self._count_all_images("ok")
            ng_count = self._count_all_images("ng")
            self._teach_page.set_ok_count(ok_count)
            self._teach_page.set_ng_count(ng_count)

            # Load ROIs from template config
            tmpl_cfg = self._pm.get_template_config(
                self._active_program, self._active_template)
            # Label nama template aktif — besar di tengah view operator
            self._run_page.set_active_template(
                tmpl_cfg.get("name", self._active_template))
            roi_dicts = tmpl_cfg.get("rois", [])
            # Support legacy single ROI format
            if not roi_dicts and "roi" in tmpl_cfg:
                old = tmpl_cfg["roi"]
                roi_dicts = [{"uid": "default", "x": old.get("x",0), "y": old.get("y",0),
                              "width": old.get("width",256), "height": old.get("height",256),
                              "enabled": True, "label": "ROI 1"}]
            rois = [ROIData.from_dict(d) for d in roi_dicts]
            if not rois:
                # Default single ROI
                rois = [ROIData(0, 0, 256, 256)]
                rois[0].label = "ROI 1"
            self._teach_page.get_roi_editor().set_rois(rois)
            self._teach_page.get_roi_panel().set_rois(rois)

            # Sync current ROIs for inference
            enabled = [r for r in rois if r.enabled]
            if enabled:
                self._current_roi = enabled[0].rect()
                self._current_all_rois = [r.rect() for r in enabled]
                self._current_all_roi_uids = [r.uid for r in enabled]
                self._current_all_roi_labels = [r.label for r in enabled]
            else:
                self._current_roi = None
                self._current_all_rois = []
                self._current_all_roi_uids = []
                self._current_all_roi_labels = []

            # Gallery thumbnails
            self._teach_page.clear_galleries()
            self._load_gallery_thumbnails("ok")
            self._load_gallery_thumbnails("ng")

            # Part Presence Check UI
            self._refresh_part_check_ui()
            self._refresh_part_check_gate_cache()

            # Training Profile UI
            self._teach_page.set_training_config(tmpl_cfg)

            # Augmentasi Data UI
            self._teach_page.set_augmentation_config(
                self._pm.get_augmentation_config(
                    self._active_program, self._active_template))

    def _count_all_images(self, label: str) -> int:
        """Total gambar training untuk label ini — gabungan foto legacy
        (images/<label>/, di-crop rata ke semua ROI saat training) + crop
        per-ROI dari CaptureReviewDialog (images/<label>_per_roi/, sudah
        benar per-ROI). Dipakai di mana pun butuh "berapa banyak data
        OK/NG yang sebenarnya ada", bukan cuma folder lama — supaya
        template yang datanya semua lewat review per-ROI (2+ ROI) tidak
        salah dianggap "belum ada gambar OK" padahal folder legacy-nya
        memang sengaja kosong."""
        if not self._active_template:
            return 0
        base = self._pm.count_template_images(
            self._active_program, self._active_template, label)
        per_roi = self._pm.count_template_images(
            self._active_program, self._active_template, f"{label}_per_roi")
        return base + per_roi

    def _list_all_images(self, label: str) -> list:
        """Sama seperti _count_all_images tapi mengembalikan daftar path."""
        if not self._active_template:
            return []
        base = self._pm.list_template_images(
            self._active_program, self._active_template, label)
        per_roi = self._pm.list_template_images(
            self._active_program, self._active_template, f"{label}_per_roi")
        return base + per_roi

    def _load_gallery_thumbnails(self, label: str):
        """Load thumbnail images from disk into gallery."""
        images = self._list_all_images(label)
        for img_path in images[-30:]:
            pixmap = QPixmap(str(img_path))
            if not pixmap.isNull():
                if label == "ok":
                    self._teach_page.add_ok_thumbnail(pixmap, str(img_path))
                else:
                    self._teach_page.add_ng_thumbnail(pixmap, str(img_path))

    # ---- Capture ----

    def _maybe_review_and_save_per_roi(self, frame, default_label: str) -> bool:
        """Kalau template ini punya 2+ ROI aktif, buka dialog review per-ROI
        sebelum menyimpan — satu foto bisa punya kondisi campuran antar ROI
        (mis. ROI1 OK, ROI2 NG), dan semua ROI berbagi satu memory
        bank/model yang sama, jadi label yang salah pada satu ROI bisa
        mencemari model yang dipakai ROI lain juga. Dengan 0-1 ROI tidak
        ada ambiguitas untuk direview (foto = ROI itu sendiri).

        Return True kalau ditangani di sini (baik tersimpan maupun
        dibatalkan user) — caller harus langsung return. Return False
        kalau caller harus lanjut pakai alur simpan foto-utuh yang lama.
        """
        if len(self._current_all_rois) < 2:
            return False

        from visioninspect.gui.dialogs.capture_review_dialog import CaptureReviewDialog
        rois = self._teach_page.get_roi_editor().get_rois()
        dialog = CaptureReviewDialog(frame, rois, default_label, parent=self)
        if not dialog.exec():
            self.set_status("Review dibatalkan, tidak ada yang disimpan.", 3000)
            return True

        saved_ok = saved_ng = 0
        for roi, crop, lbl in dialog.get_labeled_crops():
            self._pm.save_template_image(
                self._active_program, self._active_template,
                crop, f"{lbl}_per_roi", update_count=False)
            if lbl == "ok":
                saved_ok += 1
            else:
                saved_ng += 1
        logger.info("Capture per-ROI: %d OK, %d NG (template=%s)",
                     saved_ok, saved_ng, self._active_template)
        self._refresh_template_ui()
        self.set_status(f"Tersimpan per-ROI: {saved_ok} OK, {saved_ng} NG", 3000)
        return True

    def _on_capture(self, label: str):
        """Capture frame from camera — or in import mode, save current import image.

        Full frame disimpan untuk ditampilkan di galeri (kecuali template
        punya 2+ ROI, lihat _maybe_review_and_save_per_roi).
        ROI cropping dilakukan saat training (lihat TrainingWorker).
        """
        # === IMPORT REVIEW MODE ===
        if self._is_import_mode:
            self._save_current_import_image(label)
            return

        # === NORMAL CAPTURE FROM CAMERA ===
        if not self._camera_worker or not self._camera_worker.is_running:
            self.set_status("Kamera tidak aktif!", 3000)
            return
        if not self._active_template:
            self.set_status("Tidak ada template aktif!", 3000)
            return

        frame = self._camera_worker.get_frame()
        if frame is None:
            self.set_status("Gagal ambil frame!", 3000)
            return

        if self._maybe_review_and_save_per_roi(frame, label):
            return

        # Save full frame (cropping ke ROI dilakukan saat training)
        dest = self._pm.save_template_image(
            self._active_program, self._active_template, frame, label)

        logger.info("Captured %s: %s", label, dest)

        # Refresh UI
        self._refresh_template_ui()
        self.set_status(f"Gambar {label} tersimpan ({dest.name})", 3000)

    def _on_import_images(self):
        """Import images from disk — show in ROI editor one-by-one for OK/NG decision."""
        if not self._active_template:
            self.set_status("Tidak ada template aktif!", 3000)
            return
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self, "Pilih gambar untuk import", "",
            "Images (*.png *.jpg *.jpeg *.bmp)")
        if not files:
            return

        # Enter import review mode — show images one-by-one in ROI editor
        self._import_files = files
        self._import_index = 0
        self._is_import_mode = True
        self._import_cancelled = False
        self._import_current_image = None
        self._import_ok_count = 0
        self._import_ng_count = 0

        self._teach_page.show_import_mode(True)
        self._show_import_image()

    # ---- Import Review Helpers ----

    def _show_import_image(self):
        """Show current import image in the ROI editor and cache it to avoid double-read."""
        if self._import_cancelled:
            self._exit_import_mode()
            return
        if self._import_index >= len(self._import_files):
            self._exit_import_mode()
            return

        path = self._import_files[self._import_index]
        img = cv2.imread(path)
        if img is None:
            logger.warning("Import: skipping unreadable file %s", path)
            self._import_index += 1
            self._show_import_image()
            return

        # Cache the image to avoid re-reading from disk on save
        self._import_current_image = img

        # Update progress bar
        total = len(self._import_files)
        progress = int((self._import_index * 100) / total) if total > 0 else 0
        self._teach_page.set_import_progress(progress)

        # Convert BGR → RGB → QPixmap and show in ROI editor
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._teach_page.set_preview(pixmap)

        # Update import progress info
        current = self._import_index + 1
        total = len(self._import_files)
        self._teach_page.set_import_status(current, total)
        self.set_status(f"Import: {current}/{total}", 2000)

    def _save_current_import_image(self, label: str):
        """Save the current import image under 'ok' or 'ng' and advance.
        Uses cached numpy array from _show_import_image to avoid re-reading disk.
        Accumulates counts in memory for batch config update."""
        if self._import_cancelled:
            self._exit_import_mode()
            return
        if self._import_index >= len(self._import_files):
            self._exit_import_mode()
            return

        img = self._import_current_image
        if img is not None:
            path = self._import_files[self._import_index]
            if self._maybe_review_and_save_per_roi(img, label):
                # Tersimpan (atau dibatalkan) lewat review per-ROI — tidak
                # pakai counter num_ok/num_ng batch import (per-ROI save
                # selalu update_count=False, dihitung via glob folder saja).
                pass
            else:
                # Save to disk (batch mode — count diakumulasi dan ditulis sekali di akhir)
                self._pm.save_template_image(
                    self._active_program, self._active_template, img, label,
                    update_count=False)
                # Accumulate count in memory
                if label == "ok":
                    self._import_ok_count += 1
                else:
                    self._import_ng_count += 1
            logger.info("Import saved %s as %s", Path(path).name, label)
        else:
            logger.warning("Import: cached image missing for index %d", self._import_index)

        # Clear cache for this image
        self._import_current_image = None

        # Advance to next image
        self._import_index += 1
        self._show_import_image()

    def _on_cancel_import(self):
        """Cancel the current import session."""
        logger.info("Import cancelled by user at index %d/%d",
                     self._import_index, len(self._import_files))
        self._import_cancelled = True
        self._import_current_image = None
        self._exit_import_mode()

    def _exit_import_mode(self):
        """Exit import review mode, write batched config, restore normal UI."""
        # Write accumulated counts to config (batch update)
        if self._import_ok_count > 0 or self._import_ng_count > 0:
            try:
                tmpl_cfg = self._pm.get_template_config(
                    self._active_program, self._active_template)
                if self._import_ok_count > 0:
                    tmpl_cfg["num_ok"] = tmpl_cfg.get("num_ok", 0) + self._import_ok_count
                if self._import_ng_count > 0:
                    tmpl_cfg["num_ng"] = tmpl_cfg.get("num_ng", 0) + self._import_ng_count
                self._pm.update_template_config(
                    self._active_program, self._active_template, tmpl_cfg)
                logger.info("Batch config update: +%d OK, +%d NG",
                            self._import_ok_count, self._import_ng_count)
            except Exception as e:
                logger.warning("Batch config update error: %s", e)

        total = len(self._import_files)
        self._is_import_mode = False
        self._import_files = []
        self._import_index = 0
        self._import_current_image = None
        self._import_cancelled = False
        self._import_ok_count = 0
        self._import_ng_count = 0

        self._teach_page.show_import_mode(False)
        self._refresh_template_ui()

        # If camera is still active, its frame_ready signal will refresh the preview.
        # If camera is off, show placeholder text.
        if not self._camera_worker or not self._camera_worker.is_running:
            self._teach_page.set_preview_text("Import selesai")

        if self._import_cancelled:
            self.set_status("Import dibatalkan", 3000)
        else:
            self.set_status(f"Import selesai ({total} gambar diproses)", 3000)

    # ---- Test Model (batch foto statis) ----

    def _get_reference_resolution(self):
        """Dimensi acuan (w, h) untuk cek resolusi foto uji — diambil dari
        gambar OK asli template (dijamin sama dengan resolusi ROI digambar),
        fallback ke config kamera kalau tidak ada gambar OK sama sekali."""
        if self._active_template:
            ok_images = self._pm.list_template_images(
                self._active_program, self._active_template, "ok")
            if ok_images:
                img = cv2.imread(str(ok_images[0]))
                if img is not None:
                    h, w = img.shape[:2]
                    return (w, h)
        w = self._config.get("camera.resolution_width", 0)
        h = self._config.get("camera.resolution_height", 0)
        return (w, h) if w and h else None

    def _on_test_model(self):
        """Uji model terhadap batch foto statis dari disk — sanity check
        read-only. Bypass Part Presence gate/NG-debounce/cycle-delay (semua
        itu soal kamera live), dan TIDAK ditulis ke inspection_history/
        counter/PLC — murni untuk admin melihat apakah model bekerja baik."""
        if not self._inference_engine.is_loaded:
            self.set_status("Model belum dimuat. Latih atau load model dulu.", 3000)
            return
        if not self._current_all_rois:
            self.set_status("Tidak ada ROI aktif pada template ini.", 3000)
            return

        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self, "Pilih foto untuk diuji", "",
            "Images (*.png *.jpg *.jpeg *.bmp)")
        if not files:
            return

        ref_dims = self._get_reference_resolution()

        results = []
        unreadable_count = 0
        mismatch_count = 0
        for path in files:
            img = cv2.imread(path)
            if img is None:
                logger.warning("Test model: gagal baca file %s", path)
                unreadable_count += 1
                results.append({
                    "path": path, "image": None, "overall_judgement": None,
                    "worst_score": 0.0, "roi_results": [],
                    "resolution_mismatch": False, "unreadable": True,
                })
                continue

            h_img, w_img = img.shape[:2]
            mismatch = bool(ref_dims and (w_img, h_img) != ref_dims)
            if mismatch:
                mismatch_count += 1

            overall_ng = False
            worst_score = 0.0
            roi_results = []
            for idx, roi_rect in enumerate(self._current_all_rois):
                roi_dict = {
                    "x": roi_rect[0], "y": roi_rect[1],
                    "width": roi_rect[2], "height": roi_rect[3],
                    "uid": (self._current_all_roi_uids[idx]
                            if idx < len(self._current_all_roi_uids) else None),
                }
                result = self._inference_engine.infer(
                    img, roi=roi_dict, track_latency=False)
                roi_results.append({
                    "roi": roi_rect,
                    "uid": roi_dict["uid"],
                    "label": f"ROI {idx + 1}",
                    "score": result.score,
                    "judgement": result.judgement,
                    "latency_ms": result.latency_ms,
                })
                if result.score > worst_score:
                    worst_score = result.score
                if result.judgement == "NG":
                    overall_ng = True

            results.append({
                "path": path, "image": img,
                "overall_judgement": "NG" if overall_ng else "OK",
                "worst_score": worst_score, "roi_results": roi_results,
                "resolution_mismatch": mismatch, "unreadable": False,
                "reference_dims": ref_dims, "actual_dims": (w_img, h_img),
            })

        valid = [r for r in results if not r["unreadable"]]
        if not valid:
            QMessageBox.warning(
                self, "Uji Model",
                f"Semua {len(files)} file gagal dibaca. Periksa format file.")
            return

        ok_count = sum(1 for r in valid if r["overall_judgement"] == "OK")
        aggregate = {
            "total": len(valid),
            "ok_count": ok_count,
            "ng_count": len(valid) - ok_count,
            "pass_rate": (ok_count / len(valid) * 100.0) if valid else 0.0,
            "unreadable_count": unreadable_count,
            "mismatch_count": mismatch_count,
        }

        from visioninspect.gui.dialogs.model_test_dialog import ModelTestDialog
        dialog = ModelTestDialog(
            template_label=self._active_template, results=results,
            aggregate=aggregate, threshold=self._inference_engine.threshold,
            parent=self)
        dialog.exec()

    def _on_add_template(self):
        """Create a new template with default ROI."""
        name, ok = QInputDialog.getText(self, "Template Baru",
                                         "Nama template:")
        if ok and name.strip():
            try:
                tmpl = self._pm.create_template(
                    self._active_program, name.strip(),
                    config=self._new_template_defaults())
                self._active_template = tmpl["id"]
                self._pm.set_active_template(self._active_program, self._active_template)
                self._refresh_template_ui()
                # Add default ROI to the new template
                self._teach_page.get_roi_editor().add_roi(0, 0, 256, 256)
                self._save_rois(self._teach_page.get_roi_editor().get_rois())
                # Unload old model — new template has no trained model yet
                self._load_template_model()
                self._reset_counters()
                self.set_status(f"Template '{name.strip()}' dibuat", 3000)
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def _on_rename_template(self):
        """Rename active template: folder + config.

        Model OpenVINO di-unload dulu: folder berisi model.bin yang sedang
        di-mmap — Windows menolak rename folder berisi file yang di-lock
        proses sendiri (WinError 5 Access is denied, terutama di Windows
        native). Setelah rename, model di-load ulang dari folder baru.
        """
        if not self._active_program or not self._active_template:
            QMessageBox.warning(self, "Rename Template",
                                "Pilih template terlebih dahulu.")
            return
        old_id = self._active_template
        current_name = self._pm.get_template_config(
            self._active_program, old_id).get("name", old_id)
        new_name, ok = QInputDialog.getText(
            self, "Rename Template",
            "Nama baru:", text=current_name)
        if ok and new_name.strip() and new_name.strip() != current_name:
            # Lepas handle model.bin agar folder bisa di-rename di Windows
            try:
                self._inference_engine.unload_model()
            except Exception:
                pass
            try:
                result = self._pm.rename_template(
                    self._active_program, old_id, new_name.strip())
                self._active_template = result["id"]
                self._pm.set_active_template(
                    self._active_program, self._active_template)
                self._refresh_template_ui()
                # Muat ulang model dari folder baru (path berubah)
                try:
                    self._load_template_model()
                except Exception:
                    logger.warning("Gagal reload model setelah rename template",
                                   exc_info=True)
                self.set_status(
                    f"Template diganti: '{current_name}' → '{result['name']}'", 3000)
            except Exception as e:
                # Rename gagal (mis. antivirus masih memindai) — kembalikan
                # model lama supaya aplikasi tetap jalan.
                try:
                    self._load_template_model()
                except Exception:
                    pass
                hint = ""
                if getattr(e, "winerror", None) == 5:
                    hint = ("\n\nCoba: tutup jendela Explorer yang membuka folder "
                            "template ini, atau tunggu antivirus selesai memindai.")
                QMessageBox.warning(self, "Error", f"{e}{hint}")

    def _on_template_changed(self, index: int):
        """Switch active template — triggered by either TEACH or RUN combo."""
        combo = self.sender()
        if not combo or index < 0:
            return
        tmpl_id = combo.itemData(index)
        if tmpl_id and tmpl_id != self._active_template:
            self._activate_template(tmpl_id)

    def _activate_template(self, tmpl_id: str):
        """Core logic: switch to a template, load model + ROIs, sync combos."""
        self._active_template = tmpl_id
        self._pm.set_active_template(self._active_program, self._active_template)
        # Clear RUN page display immediately before loading new template data
        self._run_page.clear_results()
        # Reset part check overlay state (will refresh on next frame)
        self._last_part_ready = False
        self._pc_active_for_overlay = False
        self._last_gate_roi = None
        self._last_part_check_score = 1.0
        self._last_worst_score = 0.0
        # Cache part name for PG push
        tmpl_cfg = self._pm.get_template_config(
            self._active_program, self._active_template)
        self._active_partname = tmpl_cfg.get("name", self._active_template)
        self._refresh_template_ui()
        self._load_template_model()
        self._reset_counters()

        # Reset ROI color state agar tidak pakai warna hasil inference template lama
        self._roi_col_judgement = {}
        self._roi_col_timestamp = 0.0

        logger.info("Switched to template: %s", tmpl_id)

    def _load_template_model(self):
        """Load active template's model into inference engine."""
        if not self._active_template:
            return
        model_path = self._pm.get_template_model_path(
            self._active_program, self._active_template)
        tmpl_cfg = self._pm.get_template_config(
            self._active_program, self._active_template)
        threshold = tmpl_cfg.get("threshold", 0.5)
        tmpl_name = tmpl_cfg.get("name", self._active_template)
        trained = tmpl_cfg.get("trained", False)

        if trained and model_path and model_path.exists():
            if model_path.suffix == ".npy":
                # Simple model (no PyTorch needed)
                model_dir = model_path.parent
                try:
                    self._inference_engine.load_simple_model(model_dir, threshold=threshold)
                    self._teach_page.set_threshold(threshold)
                    self._teach_page.set_version(tmpl_cfg.get("model_version", 0))
                    self._run_page.set_model_info(tmpl_name, True, threshold)
                    self.set_status(f"Model {tmpl_name} siap", 3000)
                    logger.info("Simple model loaded: %s", tmpl_name)
                except Exception as e:
                    logger.warning("Gagal load simple model %s: %s", tmpl_name, e)
                    self._run_page.set_model_info(tmpl_name, False)
            else:
                # OpenVINO model — load into inference engine
                try:
                    self._inference_engine.load_model(model_path, threshold=threshold)
                    self._teach_page.set_threshold(threshold)
                    self._run_page.set_model_info(tmpl_name, True, threshold)
                    self.set_status(f"Model {tmpl_name} dimuat", 3000)
                    logger.info("Model loaded: %s (threshold=%.3f)",
                                self._active_template, threshold)
                except Exception as e:
                    logger.warning("Gagal load model %s: %s", tmpl_name, e)
                    self._run_page.set_model_info(tmpl_name, False)
        else:
            self._inference_engine.unload_model()
            self._run_page.set_model_info(tmpl_name, False)
            logger.info("No model for template: %s", self._active_template)

        self._update_runtime_status()

    def _on_clear_template(self):
        """Hapus template aktif dengan konfirmasi."""
        if not self._active_template:
            return

        templates = self._pm.list_templates(self._active_program)
        if len(templates) <= 1:
            QMessageBox.warning(self, "Hapus Template",
                                "Tidak bisa menghapus satu-satunya template.")
            return

        tmpl_cfg = self._pm.get_template_config(
            self._active_program, self._active_template)
        tmpl_name = tmpl_cfg.get("name", self._active_template)

        reply = QMessageBox.question(
            self, "Hapus Template",
            f"Hapus template '{tmpl_name}'?\nSemua gambar dan model akan dihapus.",
            QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                self._pm.delete_template(self._active_program, self._active_template)
                # Pindah ke template pertama yang tersisa
                templates = self._pm.list_templates(self._active_program)
                if templates:
                    self._active_template = templates[0]["id"]
                    self._pm.set_active_template(
                        self._active_program, self._active_template)
                self._refresh_template_ui()
                # FIX: Load model for the new active template
                self._load_template_model()
                self._reset_counters()
                self.set_status(f"Template '{tmpl_name}' dihapus", 3000)
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    # ---- ROI ----

    def _on_rois_changed(self):
        """ROIs changed in editor — sync panel + save."""
        rois = self._teach_page.get_roi_editor().get_rois()
        sel = self._teach_page.get_roi_editor().selected_roi
        sel_idx = -1
        if sel:
            for i, r in enumerate(rois):
                if r.uid == sel.uid:
                    sel_idx = i
                    break
        self._teach_page.get_roi_panel().set_rois(rois, sel_idx)
        self._save_rois(rois)

    def _on_roi_add(self):
        """Add new ROI at default position."""
        editor = self._teach_page.get_roi_editor()
        editor.add_roi(120, 120, 256, 256)
        self._on_rois_changed()

    def _on_roi_select(self, index: int):
        """Select ROI in editor from panel."""
        self._teach_page.get_roi_editor().select_roi(index)

    def _on_roi_delete(self, index: int):
        """Delete ROI by index."""
        rois = self._teach_page.get_roi_editor().get_rois()
        if 0 <= index < len(rois):
            self._teach_page.get_roi_editor().delete_selected_roi()
            self._on_rois_changed()

    # Karakter yang diizinkan di label ROI: huruf/angka + - _ $ | (tanpa spasi).
    # Label dipakai sebagai nama folder per-ROI ({label}_per_roi), jadi karakter
    # ilegal Windows otomatis dibuang.
    _ROI_LABEL_RE = re.compile(r"[^A-Za-z0-9_$\-|]")

    def _sanitize_roi_label(self, raw: str) -> str:
        """Bersihkan label ROI: buang spasi & karakter selain - _ $ |."""
        return self._ROI_LABEL_RE.sub("", raw.strip())

    def _on_roi_rename(self, index: int):
        """Rename label ROI (visual only — geometri/logika tidak berubah).

        Aturan: tanpa spasi, karakter spesial hanya -, _, $, |; label harus unik.
        """
        rois = self._teach_page.get_roi_editor().get_rois()
        if not (0 <= index < len(rois)):
            return
        roi = rois[index]
        new_label, ok = QInputDialog.getText(
            self, "Rename ROI",
            "Nama baru (tanpa spasi; hanya huruf/angka dan - _ $ |):",
            text=roi.label)
        if not ok:
            return
        cleaned = self._sanitize_roi_label(new_label)
        if not cleaned:
            QMessageBox.warning(
                self, "Rename ROI",
                "Nama tidak valid: kosong setelah karakter ilegal dibuang.")
            return
        if cleaned != new_label:
            self.set_status(
                f"Karakter ilegal/spasi dibuang: '{new_label}' → '{cleaned}'", 4000)
        if cleaned == roi.label:
            return  # tidak berubah
        if any(r.uid != roi.uid and r.label == cleaned for r in rois):
            QMessageBox.warning(
                self, "Rename ROI",
                f"Nama '{cleaned}' sudah dipakai ROI lain. Gunakan nama unik.")
            return
        roi.label = cleaned
        self._teach_page.get_roi_editor().set_rois(rois)
        self._teach_page.get_roi_panel().set_rois(rois, selected=index)
        self._save_rois(rois)
        self.set_status(f"ROI diganti: '{roi.label}'", 3000)

    def _on_roi_toggle_all(self, enabled: bool):
        """Enable or disable all ROIs."""
        rois = self._teach_page.get_roi_editor().get_rois()
        for r in rois:
            r.enabled = enabled
        self._teach_page.get_roi_editor().set_rois(rois)
        self._teach_page.get_roi_panel().set_rois(rois)
        self._save_rois(rois)

    def _save_rois(self, rois):
        """Save ROIs to template config."""
        if not self._active_template:
            return
        roi_dicts = [r.to_dict() for r in rois]
        self._pm.update_template_config(
            self._active_program, self._active_template,
            {"rois": roi_dicts})
        # Update current ROI untuk inference (hanya yang enabled)
        enabled = [r for r in rois if r.enabled]
        if enabled:
            self._current_roi = enabled[0].rect()
            self._current_all_rois = [r.rect() for r in enabled]
            self._current_all_roi_uids = [r.uid for r in enabled]
            self._current_all_roi_labels = [r.label for r in enabled]
        else:
            self._current_roi = None
            self._current_all_rois = []
            self._current_all_roi_uids = []
            self._current_all_roi_labels = []
        self.set_status(f"{len(rois)} ROI ({len(enabled)} aktif)", 3000)

    def _reset_counters(self):
        """Reset inspection counters (called on template switch/delete)."""
        self._inspection_count = 0
        self._inspection_ok = 0
        self._inspection_ng = 0
        self._run_page.update_counters(0, 0)
        # Cancel any pending NG interval timer
        if self._ng_interval_timer.isActive():
            self._ng_interval_timer.stop()
        self._ng_interval_active = False
        # Cancel any pending cycle delay
        if self._cycle_delay_timer.isActive():
            self._cycle_delay_timer.stop()
        self._cycle_delay_active = False
        self._run_page.set_status_message("Siap")

    def _on_gallery_image_deleted(self, label: str):
        """Refresh gallery after image deletion."""
        self._refresh_template_ui()

    def _on_thumbnail_clicked(self, image_path: str):
        """Open popup to adjust ROIs on a gallery image."""
        if not self._active_template:
            return
        from visioninspect.gui.dialogs.roi_adjust_dialog import ROIAdjustDialog
        from visioninspect.gui.widgets.roi_editor import ROIData

        current_rois = self._pm.get_template_config(
            self._active_program, self._active_template).get("rois", [])
        dialog = ROIAdjustDialog(image_path, current_rois, self)
        dialog.exec()
        # Auto-save on any close (accept = Save, reject = ✕ also saves)
        updated_rois = [ROIData.from_dict(d) for d in dialog.get_rois()]
        self._save_rois(updated_rois)
        self._refresh_template_ui()

    def _on_threshold_slider(self, value: int):
        """Update inference engine threshold when slider is moved."""
        threshold = value / 1000.0
        self._inference_engine.threshold = threshold
        self.set_status(f"Threshold: {threshold:.3f}", 2000)

    def _on_threshold_released(self):
        """Persist manually-tuned threshold to template config on slider release.

        Tanpa ini, geseran slider hanya mengubah engine live dan hilang saat
        restart (config hanya di-set saat training). Disimpan on-release agar
        tidak menulis file tiap tick geseran.
        """
        if not self._active_template:
            return
        threshold = self._teach_page.get_threshold_slider().value() / 1000.0
        try:
            self._pm.update_template_config(
                self._active_program, self._active_template,
                {"threshold": threshold})
            self._inference_engine.threshold = threshold
            self.set_status(f"Threshold {threshold:.3f} tersimpan", 3000)
            logger.info("Threshold manual disimpan: %.3f (template=%s)",
                        threshold, self._active_template)
        except Exception as e:
            logger.warning("Gagal simpan threshold: %s", e)

    # ---- Training ----

    def _on_train(self):
        """Start training for active template.

        Auto-routing:
          - PyTorch available → TrainingWorker (QThread, normal flow)
          - PyTorch blocked (Windows policy) + WSL available → training via WSL
          - Neither → SimpleThreshold fallback (existing)
        """
        if not self._active_template:
            self.set_status("Pilih template dulu!", 3000)
            return

        ok_count = self._count_all_images("ok")
        if ok_count < 1:
            QMessageBox.warning(self, "Training",
                                "Minimal 1 gambar OK diperlukan untuk training.")
            return

        # Cek torch
        import importlib
        torch_ok = True
        try:
            import torch  # noqa: F401
        except Exception:
            torch_ok = False

        # ── Jika torch diblokir Windows policy, coba WSL ──
        if not torch_ok:
            import shutil
            wsl_path = shutil.which("wsl.exe")
            if wsl_path:
                self._train_via_wsl()
                return
            logger.warning("PyTorch not available — using simple training mode")

        self._teach_page.set_training_progress(0, "Memulai training...")
        self._teach_page.get_train_button().setEnabled(False)
        logger.info("Training dimulai: program=%s, template=%s",
                     self._active_program, self._active_template)

        # Lepaskan model yang sedang dimuat agar file-nya (model.bin di-mmap
        # OpenVINO) tidak terkunci saat ditimpa hasil training baru
        # (WinError 32). Model di-reload otomatis di _on_training_finished.
        import gc
        self._inference_engine.unload_model()
        gc.collect()

        # Emit signal — worker di QThread akan menjalankan training
        force_regen = self._force_regenerate_augmentation
        self._force_regenerate_augmentation = False
        self.start_training_signal.emit(
            self._active_program, self._active_template, force_regen)

    # ---- Training via WSL (fallback saat PyTorch diblokir Windows) ----

    # Batas waktu total pipeline WSL (venv setup + pip install requirements.txt
    # + download dataset/pretrained-weights EfficientAd + training). 30 menit
    # — jauh lebih longgar dari 600s sebelumnya, karena semua langkah itu bisa
    # perlu unduhan besar (torch/anomalib/openvino + imagenette ~1.3GB +
    # pretrained teacher weights) terutama di venv/percobaan pertama atau
    # koneksi yang lambat/dibatasi kebijakan jaringan korporat.
    _WSL_TRAIN_TIMEOUT_SEC = 1800

    def _train_via_wsl(self):
        """Launch training in WSL where PyTorch can load, then reload model."""
        import subprocess
        import threading

        # Path conversion: C:\Proj → /mnt/c/Proj
        proj = Path(__file__).resolve().parent.parent.parent
        drive = proj.drive[0].lower()
        wsl_proj = f"/mnt/{drive}{str(proj)[2:]}".replace("\\", "/")

        prog = self._active_program
        tmpl = self._active_template

        self._teach_page.set_training_progress(5, "Meluncurkan WSL...")
        self._teach_page.get_train_button().setEnabled(False)
        self.set_status("Training via WSL (PyTorch di Linux)...", 0)
        logger.info("Training via WSL: %s %s (path=%s)", prog, tmpl, wsl_proj)

        def _run_wsl():
            proc = None
            timer = None
            timed_out = {"flag": False}
            try:
                cmd = [
                    "wsl.exe", "-e", "bash", "-c",
                    f"cd '{wsl_proj}' && "
                    f"if ! python3 -m venv --help >/dev/null 2>&1; then echo 'NEED_PYTHON3_VENV'; exit 1; fi && "
                    f"if [ ! -f .venv/bin/pip ]; then rm -rf .venv && python3 -m venv .venv; fi && "
                    f".venv/bin/pip install -q -r requirements.txt && "
                    f".venv/bin/python tools/train_cli.py "
                    f"--program '{prog}' --template '{tmpl}'"
                ]
                # Popen + baca stdout baris-per-baris (bukan subprocess.run yang
                # menunggu diam sampai proses selesai total) — supaya tiap baris
                # output WSL/pip/train_cli bisa langsung ditampilkan sebagai
                # progress feedback, bukan layar diam sampai akhir.
                # encoding eksplisit UTF-8 (bukan ikut locale default Windows
                # yang kadang cp1252) — tanpa ini decoding bisa crash begitu
                # ada karakter non-ASCII di output pip/apt.
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1)

                def _kill_on_timeout():
                    timed_out["flag"] = True
                    try:
                        proc.kill()
                    except Exception:
                        pass

                # Timer terpisah dari loop baca output — sengaja begitu, karena
                # kalau cuma cek elapsed-time di dalam loop `for line in
                # proc.stdout`, proses yang diam lama TANPA output (mis. lagi
                # download file besar tanpa progress bar per-baris) tidak akan
                # pernah memicu pengecekan itu sampai ada baris baru masuk —
                # timeout jadi tidak pernah kena walau sudah lama sekali diam.
                timer = threading.Timer(self._WSL_TRAIN_TIMEOUT_SEC, _kill_on_timeout)
                timer.daemon = True
                timer.start()

                lines = []
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\n")
                    if line:
                        lines.append(line)
                        logger.info("[WSL] %s", line)
                        self._wsl_train_progress_signal.emit(line)
                returncode = proc.wait()
                timer.cancel()

                if timed_out["flag"]:
                    self._wsl_train_error_signal.emit(
                        f"WSL training timeout ({self._WSL_TRAIN_TIMEOUT_SEC}s)")
                    return

                out = "\n".join(lines)

                if returncode == 0:
                    logger.info("WSL training selesai")
                    self._wsl_train_done_signal.emit()
                elif "NEED_PYTHON3_VENV" in out or "ensurepip" in out:
                    self._wsl_train_error_signal.emit(
                        "WSL butuh python3-venv.\n\n"
                        "Jalankan di WSL:\n"
                        "  sudo apt install python3-venv\n\n"
                        "Lalu coba TRAIN lagi.")
                else:
                    # Log lengkap ke file (lihat logs/app.log) untuk diagnosa
                    # penuh. Untuk pesan di UI, ambil bagian EKOR output, bukan
                    # awal — traceback/pesan error sebenarnya nyaris selalu di
                    # baris-baris terakhir sebelum proses exit, sedangkan awal
                    # output sering cuma notice rutin pip/apt (mis. "versi pip
                    # baru tersedia") yang menutupi error aslinya kalau dipotong
                    # dari depan.
                    logger.error("WSL training gagal (output lengkap):\n%s", out)
                    err = out[-600:] if out else f"exit code {returncode}"
                    self._wsl_train_error_signal.emit(f"WSL training gagal: {err}")
            except FileNotFoundError:
                self._wsl_train_error_signal.emit(
                    "wsl.exe tidak ditemukan. Install WSL dulu.")
            except Exception as e:
                self._wsl_train_error_signal.emit(str(e))
            finally:
                if timer is not None:
                    timer.cancel()
                if proc is not None and proc.poll() is None:
                    proc.kill()

        thread = threading.Thread(target=_run_wsl, daemon=True, name="wsl-train")
        thread.start()

    def _on_wsl_train_progress(self, line: str):
        """Tampilkan baris output WSL terbaru sebagai progress feedback, biar
        pipeline yang panjang (venv setup / pip install / download dataset /
        training) tidak kelihatan diam selama itu."""
        display = line if len(line) <= 120 else line[:117] + "..."
        self._teach_page.set_training_progress(10, f"WSL: {display}")

    def _on_wsl_train_done(self):
        """WSL training berhasil — reload model + refresh UI."""
        self._teach_page.set_training_done()
        self._teach_page.get_train_button().setEnabled(True)
        self._refresh_template_ui()
        self._load_template_model()
        self.set_status("Training via WSL selesai! Model dimuat.", 5000)
        logger.info("WSL training selesai, model reloaded")

    def _on_training_progress(self, percent: int, message: str):
        """Update progress bar."""
        self._teach_page.set_training_progress(percent, message)

    def _on_training_finished(self, result: dict):
        """Training completed successfully. Load model into inference engine."""
        self._teach_page.set_training_done()
        threshold = result.get("threshold", 0.5)
        self._teach_page.set_threshold(threshold)
        self._teach_page.set_version(result.get("version", 0))
        self._teach_page.get_train_button().setEnabled(True)
        self._refresh_template_ui()
        self.set_status(f"Training selesai! Threshold: {threshold:.3f}", 5000)
        logger.info("Training selesai: threshold=%.4f", threshold)

        # Update histogram dengan score real
        ok_scores = result.get("ok_scores", [])
        ng_scores = result.get("ng_scores", [])
        if ok_scores or ng_scores:
            self._teach_page.set_histogram_data(ok_scores, ng_scores, threshold)
        else:
            self._teach_page.clear_histogram()

        # Load model into inference engine (via shared method)
        self._load_template_model()

        # Update threshold slider after model load
        self._teach_page.set_threshold(threshold)

    def _on_training_error(self, error_msg: str):
        """Training failed."""
        self._teach_page.set_training_failed(error_msg)
        self._teach_page.get_train_button().setEnabled(True)
        self.set_status(f"Training gagal: {error_msg}", 5000)
        QMessageBox.warning(self, "Training Gagal", error_msg)

    def _on_training_done(self):
        """Training finished (success or failure)."""
        self._teach_page.get_train_button().setEnabled(True)
        self._refresh_template_ui()

    # ---- Redefinition (History Corrections) ----

    def _on_history_selection_changed(self):
        """Enable/disable buttons based on selection + corrected state."""
        data = self._history_page.get_selected_row_data()
        has_selection = data is not None
        self._history_page.get_correct_ok_button().setEnabled(has_selection)
        self._history_page.get_correct_ng_button().setEnabled(has_selection)
        self._history_page.get_tuning_button().setEnabled(has_selection)
        # Rollback hanya aktif jika entry sudah dikoreksi
        self._history_page.get_rollback_button().setEnabled(
            bool(data and data.get("corrected")))

    def _on_correct_history(self, correct_judgement: str):
        """Mark selected history entry as correction."""
        data = self._history_page.get_selected_row_data()
        if not data:
            return

        entry_id = data["id"]
        original = data["judgement"]

        if original == correct_judgement:
            self.set_status(f"Sudah {correct_judgement}, tidak perlu koreksi", 3000)
            return

        logger.info("Correction: entry=%d, %s → %s", entry_id, original, correct_judgement)

        # Mark in DB using shared database instance
        try:
            self._db.mark_correction(entry_id, correct_judgement)
            self._db.add_audit(self._active_program, "correction",
                         {"entry_id": entry_id, "from": original, "to": correct_judgement})
            # C0: propagasi koreksi ke PostgreSQL (local_id = id SQLite).
            # Non-blocking — PG boleh lambat/mati; sink ulang via outbox tick.
            if self._pg.is_enabled:
                threading.Thread(
                    target=self._pg.mark_correction_pg,
                    args=(entry_id, correct_judgement), daemon=True).start()
            self._refresh_history()
            self.set_status(f"Entry #{entry_id} dikoreksi ke {correct_judgement}", 3000)
        except Exception as e:
            logger.error("Correction DB error: %s", e)
            self.set_status(f"Gagal menyimpan koreksi: {e}", 5000)

    def _on_rollback(self):
        """Rollback koreksi pada entry yang dipilih."""
        data = self._history_page.get_selected_row_data()
        if not data:
            return
        entry_id = data["id"]
        if not data.get("corrected"):
            self.set_status(f"Entry #{entry_id} belum dikoreksi", 3000)
            return

        try:
            self._db.rollback_correction(entry_id)
            self._db.add_audit(self._active_program, "rollback",
                         {"entry_id": entry_id})
            # C0: batalkan koreksi di PostgreSQL juga
            if self._pg.is_enabled:
                threading.Thread(
                    target=self._pg.rollback_correction_pg,
                    args=(entry_id,), daemon=True).start()
            self._refresh_history()
            self.set_status(f"Koreksi entry #{entry_id} dibatalkan", 3000)
        except Exception as e:
            logger.error("Rollback DB error: %s", e)
            self.set_status(f"Gagal rollback: {e}", 5000)

    def _on_history_filter_changed(self):
        """Filter history berdasarkan pilihan combo (All/OK/NG)."""
        idx = self._history_page.get_filter_combo().currentIndex()
        judgement = {0: None, 1: "OK", 2: "NG"}.get(idx)
        self._refresh_history(judgement=judgement)

    # ---- Tuning (Per-ROI Correction + Additional Learning) ----

    @staticmethod
    def _parse_roi_region(value) -> list:
        """Parse kolom `roi_region` DB → list dict per-ROI.

        Robust terhadap double-encode (entry lama tersimpan sebagai JSON
        string di dalam JSON string — `'"[{...}]"'`) dan tipe tak terduga:
        None/"" → [], dict → [dict], list → list-of-dict, garbage → [].
        """
        if not value:
            return []
        data = value
        for _ in range(3):
            if not isinstance(data, str):
                break
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError, ValueError):
                return []
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            return [data]
        return []

    @staticmethod
    def _format_roi_detail(rois: list) -> str:
        """Format ringkas per-ROI untuk kolom history: 'Label1:OK · Label2:NG'."""
        parts = []
        for i, r in enumerate(rois):
            label = str(r.get("label") or f"ROI{i + 1}")
            judgement = str(r.get("judgement") or "?")
            parts.append(f"{label}:{judgement}")
        return " · ".join(parts)

    def _on_tuning_requested(self, entry_id: int):
        """Open Tuning dialog for a history entry, apply per-ROI corrections.

        CRITICAL: Tuning always runs on the SAME template that produced the
        inference result — NOT the currently active template. This prevents
        mixing training data between models (model A trained on ROI crops
        from model B's inference).

        Flow:
          1. Load entry from DB → extract template_id from metadata
          2. Activate that template (switches active model if needed)
          3. Load saved image + per-ROI data from disk
          4. Show TuningDialog — user clicks ROI, registers OK/NG
          5. On save: crop corrected ROIs → save to template → retrain
        """
        # Fetch full entry from DB (need metadata + image_path + roi_region)
        entry = self._db.get_history_entry(entry_id)
        if not entry:
            self.set_status(f"Entry #{entry_id} tidak ditemukan", 3000)
            return

        img_path = entry.get("image_path", "")
        roi_region_str = entry.get("roi_region", "")
        if not img_path or not Path(img_path).exists():
            self.set_status(f"Gambar untuk entry #{entry_id} tidak tersimpan. "
                            "Gunakan Capture di TEACH untuk menyimpan gambar.", 4000)
            return

        # ── Extract & activate the ORIGINAL template ──
        metadata = entry.get("metadata", "")
        tmpl_id = ""
        if metadata:
            try:
                meta = json.loads(metadata) if isinstance(metadata, str) else metadata
                if isinstance(meta, dict):
                    tmpl_id = str(meta.get("template", "") or "")
            except (json.JSONDecodeError, TypeError, ValueError):
                tmpl_id = ""

        if tmpl_id and tmpl_id != self._active_template:
            # Verify template still exists
            templates = self._pm.list_templates(self._active_program)
            if any(t["id"] == tmpl_id for t in templates):
                logger.info("Tuning: switching to original template %s", tmpl_id)
                self._activate_template(tmpl_id)
            else:
                logger.warning("Tuning: original template %s not found, using current", tmpl_id)

        # Parse per-ROI data (robust terhadap double-encode entry lama)
        rois_data = self._parse_roi_region(roi_region_str)
        if not rois_data:
            self.set_status(f"Tidak ada data ROI untuk entry #{entry_id}", 3000)
            return

        # Load image
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            self.set_status(f"Gagal memuat gambar: {img_path}", 3000)
            return

        # Open TuningDialog
        from visioninspect.gui.dialogs.tuning_dialog import TuningDialog
        dialog = TuningDialog(img_path, img, rois_data, self)
        if not dialog.exec():
            self.set_status("Tuning dibatalkan", 3000)
            return

        # ── Process corrections ──
        corrections = dialog.get_corrections()
        if not corrections:
            self.set_status("Tidak ada koreksi yang dilakukan", 3000)
            return

        logger.info("Tuning: %d ROI correction(s) for entry #%d on template '%s'",
                     len(corrections), entry_id, self._active_template)

        # Save each corrected ROI crop to the (now-active) template's training dir
        saved_count = 0
        for corr in corrections:
            roi_rect = (corr["x"], corr["y"], corr["width"], corr["height"])
            new_label = corr["corrected_to"].lower()  # "ok" or "ng"
            if new_label not in ("ok", "ng"):
                continue

            # Crop ROI from image
            x, y, w, h = roi_rect
            h_img, w_img = img.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            crop = img[y:y + h, x:x + w].copy()

            # Save to template's training images
            try:
                self._pm.save_template_image(
                    self._active_program, self._active_template,
                    crop, new_label, update_count=True)
                saved_count += 1
                logger.info("Tuning: cropped ROI %s → %s on template %s (%d×%d)",
                            corr.get("label", "?"), new_label,
                            self._active_template, w, h)
            except Exception as e:
                logger.warning("Tuning: gagal simpan ROI crop: %s", e)

        # ── Update history entry: judgement terkoreksi + per-ROI terbaru ──
        # (sebelumnya TIDAK ada — entry di tabel tidak pernah berubah setelah
        #  koreksi; mark_correction hanya menyimpan di kolom terpisah, kolom
        #  judgement asli tetap utuh dan tampilan memakai COALESCE)
        if corrections:
            new_overall = "NG" if any(
                roi.current_judgement == "NG" for roi in dialog._rois
            ) else "OK"
            original = entry.get("judgement", "")
            try:
                self._db.mark_correction(entry_id, new_overall)
                self._db.add_audit(
                    self._active_program, "correction",
                    {"entry_id": entry_id, "from": original,
                     "to": new_overall, "source": "tuning"})
                logger.info("Tuning: entry #%d dikoreksi %s → %s",
                            entry_id, original, new_overall)
            except Exception as e:
                logger.error("Tuning: gagal mark correction: %s", e)

            # Per-ROI breakdown terbaru (hasil terkoreksi) untuk kolom Per-ROI
            try:
                new_roi_region = json.dumps([{
                    "x": roi.x, "y": roi.y,
                    "width": roi.w, "height": roi.h,
                    "label": roi.label, "score": roi.score,
                    "judgement": roi.current_judgement,
                } for roi in dialog._rois])
                self._db.update_roi_region(entry_id, new_roi_region)
            except Exception as e:
                logger.error("Tuning: gagal update roi_region: %s", e)

            self._refresh_history()

        # ── Additional Learning: retrain on this specific template ──
        if saved_count > 0:
            ok_count = self._count_all_images("ok")
            if ok_count >= 1:
                self.set_status(f"Additional Learning: {saved_count} ROI(s) + {ok_count} OK total",
                                2000)
                # Trigger training for this specific template
                self._on_train()
            else:
                self.set_status(f"{saved_count} ROI(s) disimpan. Butuh minimal 1 OK untuk training.",
                                4000)
        else:
            self.set_status("Tidak ada ROI yang berhasil disimpan", 3000)

    def _on_rebuild_from_history(self):
        """Rebuild model using corrections data."""
        if not self._active_template:
            QMessageBox.warning(self, "Rebuild", "Tidak ada template aktif.")
            return

        # Get corrected entries from DB
        try:
            corrected = self._db.get_history(
                program=self._active_program, judgement="OK", limit=500)
            corrected += self._db.get_history(
                program=self._active_program, judgement="NG", limit=500)
            corrected = [e for e in corrected if e.get("corrected")]
        except Exception as ex:
            logger.warning("Rebuild: error fetching corrections: %s", ex)
            corrected = []

        # Copy correction images into template's corrections directory
        import shutil
        tmpl_dir = self._pm._get_template_dir(self._active_program) / self._active_template
        corr_ok_dir = tmpl_dir / "images" / "corrections" / "ok"
        corr_ng_dir = tmpl_dir / "images" / "corrections" / "ng"
        copied = 0

        for entry in corrected:
            img_path = entry.get("image_path", "")
            if not img_path or not Path(img_path).exists():
                continue
            correct_judgement = entry.get("correct_judgement", "")
            if correct_judgement == "OK":
                dest_dir = corr_ok_dir
            elif correct_judgement == "NG":
                dest_dir = corr_ng_dir
            else:
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            fname = f"corr_{entry['id']}_{Path(img_path).name}"
            shutil.copy2(img_path, dest_dir / fname)
            copied += 1

        if copied > 0:
            logger.info("Rebuild: copied %d correction images", copied)

        reply = QMessageBox.question(
            self, "Rebuild Model",
            f"Rebuild dengan {copied} gambar koreksi + data asli.\nLanjutkan?",
            QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            self._on_train()

    def _refresh_history(self, judgement: Optional[str] = None):
        """Refresh history page from local SQLite (PG hanya untuk push eksternal).

        Args:
            judgement: Filter — None=tampil semua, "OK"=hanya OK, "NG"=hanya NG.
        """
        try:
            entries = self._db.get_history(
                program=self._active_program, judgement=judgement, limit=100)
            self._history_page.clear()
            for e in entries:
                # Tampilkan hasil TERKOREKSI (kalau ada) — judgement asli
                # tetap utuh di DB, hanya tampilan yang mengikuti koreksi
                corrected = bool(e.get("corrected", 0))
                if corrected and e.get("correct_judgement"):
                    display_judgement = str(e["correct_judgement"])
                else:
                    display_judgement = str(e.get("judgement", ""))
                roi_detail = self._format_roi_detail(
                    self._parse_roi_region(e.get("roi_region")))
                self._history_page.add_entry(
                    entry_id=int(e["id"]),
                    timestamp=str(e.get("timestamp", "")),
                    program=str(e.get("program", "")),
                    score=float(e.get("score", 0.0)),
                    judgement=display_judgement,
                    image_path=str(e.get("image_path", "")),
                    corrected=corrected,
                    roi_detail=roi_detail,
                )
            self._history_page.set_status(f"{len(entries)} entries")
        except Exception as ex:
            logger.warning("History refresh error: %s", ex)

    # ---- Performance ----

    def _start_perf_monitor(self):
        self._perf_timer.start(2000)

    def _update_performance(self):
        try:
            ram_mb = self._process.memory_info().rss / 1024 / 1024
            cpu_percent = self._process.cpu_percent()
            fps = self._camera_worker.fps if self._camera_worker else 0.0
            self._diagnostics_page.update_performance(ram_mb, cpu_percent, fps, 0.0, 0.0)
        except Exception as e:
            logger.warning("Perf monitor error: %s", e)

    # ---- Slots ----

    def _on_settings_save(self):
        settings = self._settings_page.get_settings_dict()
        # C4: kredensial tidak plaintext — enkripsi password PG sebelum disimpan
        pg_settings = settings.get("postgresql", {})
        pg_pass = pg_settings.get("password", "")
        if pg_pass and not secret_store.is_encrypted(pg_pass):
            try:
                settings["postgresql"]["password"] = secret_store.encrypt(pg_pass)
            except Exception as e:
                logger.error("Enkripsi password PG gagal: %s", e)
                self.set_status("Gagal mengenkripsi password PostgreSQL", 5000)
                return
        for key, value in self._flatten_dict(settings):
            self._config.set(key, value)
        self._config.save()

        # F2: camera settings berubah → restart kamera supaya exposure/gain/WB
        # yang baru benar-benar diterapkan (sebelumnya tidak pernah berlaku).
        if settings.get("camera") is not None and self._camera_worker:
            old_cam = {k: self._config.get(f"camera.{k}")
                       for k in ("resolution_width", "resolution_height",
                                 "fps_target", "exposure", "gain",
                                 "white_balance")}
            new_cam = settings.get("camera", {})
            changed = any(new_cam.get(k) != old_cam.get(k) for k in old_cam)
            if changed:
                self._camera_worker.set_camera_config(new_cam)
                dev = self._config.get("camera.device_index", 0)
                QTimer.singleShot(
                    400, lambda: self._camera_worker.restart_camera(dev))
                self.set_status("Kamera di-restart (setting exposure/gain/WB)",
                                4000)
        # YOLO config bisa berubah di Settings → muat ulang detektor (lazy)
        # pada frame berikutnya, tanpa perlu restart aplikasi
        self._yolo_det = None
        self._last_class_filter_ng = False
        self._statusbar.showMessage(self._tr.tr("settings_saved"), 3000)

        lang = settings.get("language", "id")
        if lang != self._tr.language:
            self._tr.language = lang
            self._retranslate_ui()

        # Toggle debug logging
        show_debug = settings.get("show_debug", False)
        import logging
        root = logging.getLogger()
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(logging.DEBUG if show_debug else logging.INFO)
        logger.info("Log debug: %s", "AKTIF" if show_debug else "NONAKTIF")

        # Re-init PostgreSQL dengan config dari UI (bukan read-back dari file)
        pg_cfg = settings.get("postgresql", {})
        self._pg = self._pg.__class__(pg_cfg)
        if pg_cfg.get("enabled"):
            try:
                conn = self._pg._connect()
                conn.close()
                # Pastikan tabel siap pakai setelah koneksi berhasil
                self._pg.ensure_ready()
                self._settings_page.set_pg_status(True, pg_cfg.get("host", ""))
                logger.info("PostgreSQL terhubung: %s@%s:%d/%s",
                            pg_cfg.get("user"), pg_cfg.get("host"),
                            pg_cfg.get("port"), pg_cfg.get("dbname"))
            except Exception as e:
                err = str(e).split(":")[-1].strip()[:60]
                self._settings_page.set_pg_status(False, err)
                logger.warning("PostgreSQL connection failed: %s", e)
        else:
            self._settings_page.set_pg_status(False, "Tidak diaktifkan")

        # Re-init PLC dengan config terbaru dari UI (io_map/pulse/port)
        self._shutdown_plc()
        plc_cfg = settings.get("plc", {})
        if plc_cfg.get("enabled"):
            self._config.set("plc.io_map", plc_cfg.get("io_map", {}))
            self._config.set("plc.pulse_ms", plc_cfg.get("pulse_ms", 300))
            self._init_plc()
        else:
            try:
                self._run_page.set_plc_status(False)
            except Exception:
                pass
            self._settings_page.set_plc_status(False, "Tidak diaktifkan")

        # Inference mode → label Trigger di Run page
        infer_mode = settings.get("inference", {}).get("mode", "continuous")
        self._run_page.set_trigger_mode(infer_mode)

        # Flask API — start/stop sesuai config UI (api_key tetap dari config
        # tersimpan agar tidak berganti tiap save)
        self._apply_flask_settings(settings)

    def _on_tab_changed(self, index: int):
        page_names = ["Run", "Teach", "History", "Settings", "Diagnostics",
                      "Akun", "I/O Settings"]
        name = page_names[index] if index < len(page_names) else f"Tab {index}"
        logger.debug("Switched to %s tab", name)

        # Full screen on RUN tab, windowed on others
        if index == 0:
            QTimer.singleShot(0, self._go_fullscreen)
        else:
            QTimer.singleShot(0, self._go_windowed)

        # Refresh teach preview
        if index == 1:
            self._refresh_template_ui()
        # Refresh history when switching to HISTORY tab
        elif index == 2:
            self._refresh_history()
        # Refresh account page when switching to AKUN tab
        elif index == 5:
            self._account_page.refresh()
        # Refresh I/O Monitor saat beralih ke tab I/O Settings
        elif index == 6:
            self._io_page.refresh_monitor_connection()

    def _show_about(self):
        QMessageBox.about(
            self,
            f"About {self._tr.tr('app_name')}",
            f"<h2>{self._tr.tr('app_name')} v1.0.0</h2>"
            f"<p>{self._tr.tr('app_title')}</p>"
            "<p>Built with Anomalib + OpenVINO + PySide6</p>"
            "<p>100% lokal, CPU-only, offline.</p>"
        )

    # ── Model Export / Import ───────────────────────────────────────────

    def _export_model_dialog(self):
        """Dialog untuk export model ke file .zip."""
        program = self._active_program
        template = self._active_template
        if not program or not template:
            QMessageBox.warning(self, "Export Model",
                                "Pilih program dan template terlebih dahulu.")
            return

        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path

        try:
            tmpl_cfg = self._pm.get_template_config(program, template)
            if not tmpl_cfg.get("trained", False):
                QMessageBox.warning(self, "Export Model",
                                    f"Template '{template}' belum pernah di-train.\n"
                                    "Latih model terlebih dahulu sebelum export.")
                return

            default_name = f"model_{program}_{template}_{tmpl_cfg.get('model_version', 0)}.zip"
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Export Model", default_name,
                "ZIP files (*.zip)")
            if not save_path:
                return

            self.set_status("Exporting model...", 0)
            result = self._pm.export_model_to_zip(
                program, template, Path(save_path))
            self.set_status(f"Model exported: {save_path}", 3000)
            QMessageBox.information(self, "Export Model",
                                    f"Model berhasil diexport:\n{result}")
        except Exception as e:
            logger.error("Export model gagal: %s", e)
            QMessageBox.critical(self, "Export Model Error", str(e))

    def _import_model_dialog(self):
        """Dialog untuk import model dari file .zip."""
        program = self._active_program
        template = self._active_template
        if not program or not template:
            QMessageBox.warning(self, "Import Model",
                                "Pilih program dan template terlebih dahulu.")
            return

        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path

        try:
            zip_path, _ = QFileDialog.getOpenFileName(
                self, "Import Model", "",
                "ZIP files (*.zip)")
            if not zip_path:
                return

            # Konfirmasi
            reply = QMessageBox.question(
                self, "Import Model",
                f"Import model ke template '{template}'?\n"
                "Model yang ada akan diganti.",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

            self.set_status("Importing model...", 0)

            # Unload current model dulu — lepaskan handle OpenVINO biar model.bin gak di-lock
            if hasattr(self, "_inference_engine") and self._inference_engine is not None:
                self._inference_engine.unload_model()
                import gc
                gc.collect()
                gc.collect()

            result = self._pm.import_model_from_zip(
                Path(zip_path), program, template)
            self.set_status(f"Model imported (v{result['model_version']})", 3000)

            # Refresh teach page — reload template + model
            if hasattr(self, "_teach_page"):
                self._refresh_template_ui()
                self._load_template_model()
            QMessageBox.information(self, "Import Model",
                                    f"Model berhasil diimport!\n"
                                    f"  Versi: {result['model_version']}\n"
                                    f"  Files: {result['files_restored']}\n"
                                    f"  Diexport: {result.get('source_exported_at', '-')}")
        except Exception as e:
            logger.error("Import model gagal: %s", e)
            QMessageBox.critical(self, "Import Model Error", str(e))

    def keyPressEvent(self, event):
        """Esc untuk konfirmasi exit saat borderless full screen."""
        if event.key() == Qt.Key_Escape:
            reply = QMessageBox.question(
                self, "Keluar", "Yakin ingin keluar aplikasi?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.close()
        else:
            super().keyPressEvent(event)

    def _go_fullscreen(self):
        """Borderless full screen — no title bar, no taskbar."""
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.showFullScreen()

    def _go_windowed(self):
        """Borderless full screen juga — untuk admin tabs."""
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.showFullScreen()

    def _update_runtime_status(self):
        """Update inference runtime indicator in Settings page."""
        has_ov = self._inference_engine._use_ov if hasattr(self._inference_engine, '_use_ov') else False
        has_torch = False
        try:
            import torch  # noqa: F401
            has_torch = True
        except Exception:
            pass

        # Deteksi GPU/CUDA
        gpu_info = ""
        gpu_available = False
        if has_torch:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_info = torch.cuda.get_device_name(0)
            except Exception:
                pass

        if has_ov and self._inference_engine._model is not None:
            active = "openvino"
        elif self._inference_engine._simple_loaded:
            active = "simple"
        elif has_torch:
            active = "anomalib"
        else:
            active = ""

        self._settings_page.set_runtime_status(has_ov, has_torch, active,
                                                gpu_available, gpu_info)

        # Update PostgreSQL connection status (gunakan self._pg langsung)
        if self._pg.is_enabled:
            try:
                conn = self._pg._connect()
                conn.close()
                pg_cfg = self._config.get("postgresql", {})
                self._settings_page.set_pg_status(True, pg_cfg.get("host", ""))
            except Exception as e:
                self._settings_page.set_pg_status(False, str(e).split(":")[-1].strip()[:60])
        else:
            self._settings_page.set_pg_status(False, "Tidak diaktifkan")

    def _retranslate_ui(self):
        self._tabs.setTabText(0, self._tr.tr("nav_run"))
        self._tabs.setTabText(1, self._tr.tr("nav_teach"))
        self._tabs.setTabText(2, self._tr.tr("nav_history"))
        self._tabs.setTabText(3, self._tr.tr("nav_settings"))
        self._tabs.setTabText(4, self._tr.tr("nav_diagnostics"))
        self.setWindowTitle(self._tr.tr("app_title"))

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "") -> list[tuple[str, any]]:
        items = []
        for key, value in d.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(MainWindow._flatten_dict(value, new_key))
            else:
                items.append((new_key, value))
        return items

    # ---- Public API ----

    def get_tabs(self) -> QTabWidget:
        return self._tabs

    def get_run_page(self) -> RunPage:
        return self._run_page

    def get_teach_page(self) -> TeachPage:
        return self._teach_page

    def get_history_page(self) -> HistoryPage:
        return self._history_page

    def get_settings_page(self) -> SettingsPage:
        return self._settings_page

    def get_diagnostics_page(self) -> DiagnosticsPage:
        return self._diagnostics_page

    def get_camera_worker(self):
        return self._camera_worker

    def get_program_manager(self):
        return self._pm

    def set_status(self, message: str, timeout: int = 0):
        self._statusbar.showMessage(message, timeout)

    def closeEvent(self, event):
        logger.info("Application closing...")
        if self._camera_worker:
            self._camera_worker.stop_camera()
        self._perf_timer.stop()
        # Tutup port PLC + matikan coil output (biar PLC tidak menerima
        # sinyal OK/NG dari sistem yang sudah mati)
        self._shutdown_plc()
        # Matikan Flask API (thread daemon + server shutdown)
        self._shutdown_flask()
        for t, name in [(self._camera_thread, "camera"),
                        (self._training_thread, "training")]:
            if t and t.isRunning():
                t.quit()
                if not t.wait(2000):
                    logger.warning("%s thread did not stop, terminating", name)
                    t.terminate()
                    t.wait(1000)
        event.accept()
