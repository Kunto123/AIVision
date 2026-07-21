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

from visioninspect.utils.config import Config, ConfigError
from visioninspect.utils.i18n import Translator
from visioninspect.utils.logging_setup import setup_logging, get_logger

from visioninspect.core.program import ProgramManager
from visioninspect.core.inference import InferenceEngine, overlay_heatmap
from visioninspect.core import part_check as pc_module
from visioninspect.gui.camera_worker import CameraThread, CameraWorker
from visioninspect.gui.training_worker import TrainingThread, TrainingWorker
from visioninspect.gui.widgets.roi_editor import ROIData
from visioninspect.gui.pages.run_page import RunPage
from visioninspect.gui.pages.teach_page import TeachPage
from visioninspect.gui.pages.history_page import HistoryPage
from visioninspect.gui.pages.settings_page import SettingsPage
from visioninspect.gui.pages.diagnostics_page import DiagnosticsPage
from visioninspect.gui.pages.account_page import AccountPage
from visioninspect.gui.dialogs.login_dialog import LoginDialog

logger = get_logger("app")


class MainWindow(QMainWindow):
    """Main application window dengan 5 tab halaman."""

    # Signal untuk invoke training di QThread worker
    start_training_signal = Signal(str, str)

    def __init__(self, config: Config, translator: Translator):
        super().__init__()
        self._config = config
        self._tr = translator

        # Camera
        self._camera_thread: Optional[CameraThread] = None
        self._camera_worker: Optional[CameraWorker] = None

        # Program Manager — use project-relative path for WSL/Windows sharing
        data_dir = Path(config.get("data_dir", "")).resolve()
        if not data_dir.is_absolute():
            data_dir = Path(__file__).resolve().parent.parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir
        self._pm = ProgramManager(data_dir / "programs")

        # Database (shared instance for history, counters, corrections, users)
        from visioninspect.storage.db import Database
        from visioninspect.storage.postgres_db import PostgresDB
        self._db = Database(data_dir / "database.db")
        # PostgreSQL connection (optional — enabled via config).
        # WAJIB sub-dict "postgresql" (PostgresDB baca config["enabled"]/["host"]/dst);
        # get_all() meneruskan config penuh -> "enabled" tak ketemu -> keliru nonaktif.
        self._pg = PostgresDB(self._config.get("postgresql", {}))
        if self._pg.is_enabled:
            # Pastikan DB siap pakai (tabel ada, admin ter-seed) begitu terhubung
            self._pg.ensure_ready()

        # Authentication state
        self._current_user: Optional[dict] = None
        self._user_role: str = "operator"

        # State
        self._active_program = ""
        self._active_template = ""

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
        self._current_roi: Optional[tuple] = None
        self._current_all_rois: list = []
        self._current_all_roi_uids: list = []
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
        # Part name untuk push ke PG (di-set saat ganti template)
        self._active_partname = ""

        self._setup_window()
        self._setup_tabs()
        self._setup_statusbar()
        self._setup_menu()
        self._apply_theme()
        self._connect_signals()
        self._init_camera()
        self._start_perf_monitor()
        self._init_programs()

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

        self._tabs.addTab(self._run_page, self._tr.tr("nav_run"))
        self._tabs.addTab(self._teach_page, self._tr.tr("nav_teach"))
        self._tabs.addTab(self._history_page, self._tr.tr("nav_history"))
        self._tabs.addTab(self._settings_page, self._tr.tr("nav_settings"))
        self._tabs.addTab(self._diagnostics_page, self._tr.tr("nav_diagnostics"))
        self._tabs.addTab(self._account_page, "👥 Akun")

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

        self._logout_btn = QPushButton("🔓 Logout")
        self._logout_btn.setFixedHeight(24)
        self._logout_btn.setStyleSheet(
            "font-size: 11px; padding: 0 8px; border: 1px solid #233A57;"
            " border-radius: 3px; background: #1A2A44; color: #EF4444;")
        self._logout_btn.setVisible(False)
        self._logout_btn.clicked.connect(self._on_logout)
        self._statusbar.addPermanentWidget(self._logout_btn)

        self._program_label = QLabel("Program: —")
        self._statusbar.addPermanentWidget(self._program_label)

        self._cam_status_label = QLabel("📷 —")
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

    def _apply_theme(self):
        theme_path = Path(__file__).parent / "theme.qss"
        if theme_path.exists():
            with open(theme_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    # ---- Authentication ----

    def _show_login(self):
        """Show login dialog, apply role visibility after success."""
        auth_db = self._pg if self._pg.is_enabled else self._db
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

        # TEACH: Train button
        self._teach_page.get_train_button().clicked.connect(self._on_train)

        # TEACH: Template buttons
        self._teach_page.get_add_template_button().clicked.connect(self._on_add_template)
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
        self._teach_page.get_roi_panel().roi_toggle_all.connect(self._on_roi_toggle_all)

        # TEACH: Threshold slider → live update inference threshold,
        # lalu simpan permanen ke config template saat slider dilepas
        # (sliderReleased, bukan tiap tick, agar tidak menulis file terus-menerus)
        self._teach_page.get_threshold_slider().valueChanged.connect(self._on_threshold_slider)
        self._teach_page.get_threshold_slider().sliderReleased.connect(self._on_threshold_released)

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

        # HISTORY: Correction buttons
        self._history_page.get_correct_ok_button().clicked.connect(
            lambda: self._on_correct_history("OK"))
        self._history_page.get_correct_ng_button().clicked.connect(
            lambda: self._on_correct_history("NG"))
        self._history_page.get_rebuild_button().clicked.connect(self._on_rebuild_from_history)

        # HISTORY: Tuning
        self._history_page.tuning_requested.connect(self._on_tuning_requested)

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
                for i, roi_rect in enumerate(self._current_all_rois):
                    x, y, w, h = roi_rect
                    # Green border only — no fill, no badge
                    pen = QPen(QColor("#22C55E"), 2)
                    qp.setPen(pen)
                    qp.setBrush(Qt.NoBrush)
                    qp.drawRect(x, y, w, h)
                    # Label text langsung di atas gambar, tanpa background fill
                    label_x = x + 3
                    label_y = y - 4 if y >= 16 else y + 13
                    qp.setPen(QColor("#22C55E"))
                    qp.drawText(label_x, label_y, f"ROI{i+1}")
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
        self._cam_status_label.setText("📷 Aktif")
        self._cam_status_label.setStyleSheet("color: #22C55E;")
        self._run_page.set_camera_status(True)
        self._start_cam_action.setText("Stop Kamera")
        self._teach_page.set_preview_text("")
        self.set_status("Kamera aktif", 3000)

    def _on_camera_stopped(self):
        self._cam_status_label.setText("📷 Mati")
        self._cam_status_label.setStyleSheet("color: #EF4444;")
        self._run_page.set_camera_status(False)
        self._start_cam_action.setText("Start Kamera")
        self._teach_page.set_preview_text("Kamera dimatikan")

    def _on_camera_error(self, msg: str):
        self._cam_status_label.setText("📷 Error")
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
            self._run_page.get_heatmap_button().setText("🔥 Heatmap ON")
        else:
            self._run_page.get_heatmap_button().setText("🔥 Heatmap")
        logger.info("Heatmap overlay: %s", "ON" if enabled else "OFF")

    # ---- Inference ----

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
            self._last_part_ready = True
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
            worst_score = 0.0
            total_latency = 0.0
            roi_results = []

            for idx, roi_rect in enumerate(self._current_all_rois):
                roi_dict = {
                    "x": roi_rect[0], "y": roi_rect[1],
                    "width": roi_rect[2], "height": roi_rect[3],
                    "uid": (self._current_all_roi_uids[idx]
                            if idx < len(self._current_all_roi_uids) else None),
                }
                result = self._inference_engine.infer(frame, roi=roi_dict)
                roi_results.append({
                    "roi": roi_rect,
                    "score": result.score,
                    "judgement": result.judgement,
                    "latency": result.latency_ms,
                })
                total_latency += result.latency_ms
                if result.score > worst_score:
                    worst_score = result.score
                    self._last_heatmap = result.heatmap
                    self._last_frame = frame
                if result.judgement == "NG":
                    overall_ng = True

            raw_judgement = "NG" if overall_ng else "OK"
            self._last_worst_score = worst_score

            # Push SETIAP hasil inferensi ke PostgreSQL (OK & NG).
            # partname = nama template, mpcheck = nama operator (lihat helper).
            # Non-blocking (daemon thread) agar koneksi DB tidak membekukan
            # thread GUI saat frame-rate tinggi.
            self._push_inspection_async(worst_score)

            # Always update latency and ROI info (informational)
            avg_latency = total_latency / len(roi_results) if roi_results else 0.0
            self._run_page.update_latency(avg_latency)
            self._run_page.update_roi_results(roi_results)

            # ---- NG Interval Timer ----
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

            else:  # raw_judgement == "NG"
                if not self._ng_interval_timer.isActive():
                    # First NG frame — start interval timer
                    delay = self._settings_page.get_ng_debounce_ms()
                    effective_delay = max(delay, 50)  # minimum 50ms agar timer tetap jalan
                    self._ng_interval_timer.start(effective_delay)
                    self._ng_interval_active = True
                    # Show NG immediately on display (counter hanya bertambah via timer tick)
                    self._run_page.update_judgement("NG", worst_score)
                    # Save frame untuk tuning
                    self._save_inspection_frame(frame, "NG", worst_score,
                                                 roi_results, avg_latency)
                else:
                    # Timer already running — update display (worse score)
                    self._run_page.update_judgement("NG", worst_score)

            # ── Cycle delay: jeda antar siklus inspeksi ──
            cycle_delay = self._settings_page.get_cycle_delay_ms()
            if cycle_delay > 0:
                # Stop NG interval timer so it doesn't phantom-count during delay
                if self._ng_interval_timer.isActive():
                    self._ng_interval_timer.stop()
                    self._ng_interval_active = False
                self._cycle_delay_timer.start(cycle_delay)
                self._cycle_delay_active = True
                self._run_page.set_status_message(
                    f"⏳ Cycle delay {cycle_delay} ms...")
            else:
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
                        "score": r["score"], "judgement": r["judgement"],
                    } for r in roi_results])
                    self._db.add_inspection({
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

        except Exception as e:
            logger.warning("Inference error: %s", e)

    def _push_inspection_async(self, score: float) -> None:
        """Push satu hasil inferensi ke PostgreSQL tanpa blok thread GUI.

        Mapping kolom qc_inspection_push:
          partname = nama template (bukan id)
          mpcheck  = nama akun operator yang login (bukan OK/NG)
          data1    = part-ready confidence
          data2    = anomaly score

        push_inspection membuka koneksi baru tiap panggil; menjalankannya
        langsung di thread GUI (per frame) bisa membekukan UI. Jadi dijalankan
        fire-and-forget di daemon thread. Aman karena tiap push memakai koneksi
        sendiri (tanpa shared state).
        """
        if not self._pg.is_enabled:
            return
        partname = self._active_partname or self._active_program  # nama template
        operator = ""
        if self._current_user:
            operator = (self._current_user.get("display_name")
                        or self._current_user.get("username", ""))
        data1 = self._last_part_check_score
        threading.Thread(
            target=self._pg.push_inspection,
            kwargs=dict(partname=partname, mpcheck=operator,
                        data1=data1, data2=score),
            daemon=True,
        ).start()

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
        self._run_page.set_status_message("⏳ Siap")

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
                    "⚠️ Algorithm/Backbone diubah — klik TRAIN untuk "
                    "menerapkan ke model (model saat ini masih pakai setting lama).",
                    5000)
        except Exception as e:
            logger.warning("Training config save error: %s", e)

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
            ok_count = self._pm.count_template_images(
                self._active_program, self._active_template, "ok")
            ng_count = self._pm.count_template_images(
                self._active_program, self._active_template, "ng")
            self._teach_page.set_ok_count(ok_count)
            self._teach_page.set_ng_count(ng_count)

            # Load ROIs from template config
            tmpl_cfg = self._pm.get_template_config(
                self._active_program, self._active_template)
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
            else:
                self._current_roi = None
                self._current_all_rois = []
                self._current_all_roi_uids = []

            # Gallery thumbnails
            self._teach_page.clear_galleries()
            self._load_gallery_thumbnails("ok")
            self._load_gallery_thumbnails("ng")

            # Part Presence Check UI
            self._refresh_part_check_ui()
            self._refresh_part_check_gate_cache()

            # Training Profile UI
            self._teach_page.set_training_config(tmpl_cfg)

    def _load_gallery_thumbnails(self, label: str):
        """Load thumbnail images from disk into gallery."""
        images = self._pm.list_template_images(
            self._active_program, self._active_template, label)
        for img_path in images[-30:]:
            pixmap = QPixmap(str(img_path))
            if not pixmap.isNull():
                if label == "ok":
                    self._teach_page.add_ok_thumbnail(pixmap, str(img_path))
                else:
                    self._teach_page.add_ng_thumbnail(pixmap, str(img_path))

    # ---- Capture ----

    def _on_capture(self, label: str):
        """Capture frame from camera — or in import mode, save current import image.

        Full frame disimpan untuk ditampilkan di galeri.
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
        else:
            self._current_roi = None
            self._current_all_rois = []
            self._current_all_roi_uids = []
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
        self._run_page.set_status_message("⏳ Siap")

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

        ok_count = self._pm.count_template_images(
            self._active_program, self._active_template, "ok")
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
        self.start_training_signal.emit(
            self._active_program, self._active_template)

    # ---- Training via WSL (fallback saat PyTorch diblokir Windows) ----

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

        self._teach_page.set_training_progress(5, "🧠 Meluncurkan WSL...")
        self._teach_page.get_train_button().setEnabled(False)
        self.set_status("Training via WSL (PyTorch di Linux)...", 0)
        logger.info("Training via WSL: %s %s (path=%s)", prog, tmpl, wsl_proj)

        def _run_wsl():
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
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                out = result.stdout + result.stderr

                if result.returncode == 0:
                    logger.info("WSL training selesai")
                    QTimer.singleShot(0, self._on_wsl_train_done)
                elif "NEED_PYTHON3_VENV" in out:
                    QTimer.singleShot(0, lambda: self._on_training_error(
                        "WSL butuh python3-venv.\n\n"
                        "Jalankan di WSL:\n"
                        "  sudo apt install python3-venv\n\n"
                        "Lalu coba TRAIN lagi."))
                elif "ensurepip" in out or "python3-venv" in out:
                    QTimer.singleShot(0, lambda: self._on_training_error(
                        "WSL butuh python3-venv.\n\n"
                        "Jalankan di WSL:\n"
                        "  sudo apt install python3-venv\n\n"
                        "Lalu coba TRAIN lagi."))
                else:
                    err = (result.stderr.strip()[:200]
                           or f"exit code {result.returncode}")
                    QTimer.singleShot(0, lambda: self._on_training_error(
                        f"WSL training gagal: {err}"))
            except subprocess.TimeoutExpired:
                QTimer.singleShot(0, lambda: self._on_training_error(
                    "WSL training timeout (600s)"))
            except FileNotFoundError:
                QTimer.singleShot(0, lambda: self._on_training_error(
                    "wsl.exe tidak ditemukan. Install WSL dulu."))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._on_training_error(str(e)))

        thread = threading.Thread(target=_run_wsl, daemon=True, name="wsl-train")
        thread.start()

    def _on_wsl_train_done(self):
        """WSL training berhasil — reload model + refresh UI."""
        self._teach_page.set_training_done()
        self._teach_page.get_train_button().setEnabled(True)
        self._refresh_template_ui()
        self._load_template_model()
        self.set_status("✅ Training via WSL selesai! Model dimuat.", 5000)
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
        """Enable/disable correction + tuning buttons based on selection."""
        data = self._history_page.get_selected_row_data()
        has_selection = data is not None
        self._history_page.get_correct_ok_button().setEnabled(has_selection)
        self._history_page.get_correct_ng_button().setEnabled(has_selection)
        self._history_page.get_tuning_button().setEnabled(has_selection)

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
            self._refresh_history()
            self.set_status(f"Entry #{entry_id} dikoreksi ke {correct_judgement}", 3000)
        except Exception as e:
            logger.error("Correction DB error: %s", e)
            self.set_status(f"Gagal menyimpan koreksi: {e}", 5000)

    # ---- Tuning (Per-ROI Correction + Additional Learning) ----

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
                tmpl_id = meta.get("template", "")
            except (json.JSONDecodeError, TypeError):
                tmpl_id = ""

        if tmpl_id and tmpl_id != self._active_template:
            # Verify template still exists
            templates = self._pm.list_templates(self._active_program)
            if any(t["id"] == tmpl_id for t in templates):
                logger.info("Tuning: switching to original template %s", tmpl_id)
                self._activate_template(tmpl_id)
            else:
                logger.warning("Tuning: original template %s not found, using current", tmpl_id)

        # Parse per-ROI data
        rois_data = []
        if roi_region_str:
            try:
                rois_data = json.loads(roi_region_str) if isinstance(roi_region_str, str) else roi_region_str
            except (json.JSONDecodeError, TypeError):
                rois_data = []

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

        # ── Additional Learning: retrain on this specific template ──
        if saved_count > 0:
            ok_count = self._pm.count_template_images(
                self._active_program, self._active_template, "ok")
            if ok_count >= 1:
                self.set_status(f"🧠 Additional Learning: {saved_count} ROI(s) + {ok_count} OK total",
                                2000)
                # Trigger training for this specific template
                self._on_train()
            else:
                self.set_status(f"✅ {saved_count} ROI(s) disimpan. Butuh minimal 1 OK untuk training.",
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

    def _refresh_history(self):
        """Refresh history page from database (PostgreSQL first if enabled)."""
        try:
            if self._pg.is_enabled:
                entries = self._pg.get_history(limit=100)
            else:
                entries = self._db.get_history(program=self._active_program, limit=100)
            self._history_page.clear()
            for e in entries:
                self._history_page.add_entry(
                    entry_id=int(e["id"]),
                    timestamp=str(e.get("timestamp", "")),
                    program=str(e.get("program", "")),
                    score=float(e.get("score", 0.0)),
                    judgement=str(e.get("judgement", "")),
                    image_path=str(e.get("image_path", "")),
                    corrected=bool(e.get("corrected", 0)),
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
        for key, value in self._flatten_dict(settings):
            self._config.set(key, value)
        self._config.save()
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

    def _on_tab_changed(self, index: int):
        page_names = ["Run", "Teach", "History", "Settings", "Diagnostics", "Akun"]
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

    def _show_about(self):
        QMessageBox.about(
            self,
            f"About {self._tr.tr('app_name')}",
            f"<h2>{self._tr.tr('app_name')} v1.0.0</h2>"
            f"<p>{self._tr.tr('app_title')}</p>"
            "<p>Built with Anomalib + OpenVINO + PySide6</p>"
            "<p>100% lokal, CPU-only, offline.</p>"
        )

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

        if has_ov and self._inference_engine._model is not None:
            active = "openvino"
        elif self._inference_engine._simple_loaded:
            active = "simple"
        elif has_torch:
            active = "anomalib"
        else:
            active = ""

        self._settings_page.set_runtime_status(has_ov, has_torch, active)

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
        for t, name in [(self._camera_thread, "camera"),
                        (self._training_thread, "training")]:
            if t and t.isRunning():
                t.quit()
                if not t.wait(2000):
                    logger.warning("%s thread did not stop, terminating", name)
                    t.terminate()
                    t.wait(1000)
        event.accept()
