"""
VisionInspect - Main Window
Window utama dengan tab navigasi: RUN, TEACH, HISTORY, SETTINGS, DIAGNOSTICS.
Mengelola CameraWorker, inferensi, ProgramManager, dan komponen global.
"""

import os
import sys
import time
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
from visioninspect.gui.pages.global_settings_page import GlobalSettingsPage
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
        self._db = Database(data_dir / "database.db")

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
        self._heatmap_enabled = False
        self._last_frame: Optional[object] = None
        self._last_heatmap: Optional[object] = None

        # Import review mode
        self._import_files: list = []
        self._import_index = 0
        self._is_import_mode = False

        # Counters
        self._inspection_count = 0
        self._inspection_ok = 0
        self._inspection_ng = 0
        self._inference_save_counter = 0  # throttle DB saves (~1/sec)

        # NG interval timer (fires every N ms during sustained NG)
        self._ng_interval_timer = QTimer(self)
        self._ng_interval_timer.timeout.connect(self._on_ng_interval_tick)
        self._ng_interval_active = False

        # Part Presence Check (cached config — read from disk only on template switch)
        self._current_part_check_cfg: dict = {}

        self._setup_window()
        self._setup_tabs()
        self._setup_statusbar()
        self._setup_menu()
        self._apply_theme()
        self._connect_signals()
        self._init_camera()
        self._start_perf_monitor()
        self._init_programs()

        # Initial history load
        QTimer.singleShot(1000, self._refresh_history)

        # Show login dialog (blocks until login or cancel)
        QTimer.singleShot(100, self._show_login)

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
        self._account_page = AccountPage(self._db)
        self._global_settings_page = GlobalSettingsPage(self._config)

        self._tabs.addTab(self._run_page, self._tr.tr("nav_run"))
        self._tabs.addTab(self._teach_page, self._tr.tr("nav_teach"))
        self._tabs.addTab(self._history_page, self._tr.tr("nav_history"))
        self._tabs.addTab(self._settings_page, self._tr.tr("nav_settings"))
        self._tabs.addTab(self._diagnostics_page, self._tr.tr("nav_diagnostics"))
        self._tabs.addTab(self._account_page, "👥 Akun")
        self._tabs.addTab(self._global_settings_page, "⚙️ Global")

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
        dialog = LoginDialog(self._db, self)
        if dialog.exec():
            self._current_user = dialog.user
            self._user_role = dialog.role
            self._apply_role_visibility()
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
        else:
            self._tabs.setCurrentIndex(0)  # Start on RUN for operator
            self._reset_counters()          # Fresh counters for operator

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

        # TEACH: Threshold slider → live update inference threshold
        self._teach_page.get_threshold_slider().valueChanged.connect(self._on_threshold_slider)

        # TEACH: Image deleted from gallery
        self._teach_page.image_deleted.connect(self._on_gallery_image_deleted)

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

        # HISTORY: Correction buttons
        self._history_page.get_correct_ok_button().clicked.connect(
            lambda: self._on_correct_history("OK"))
        self._history_page.get_correct_ng_button().clicked.connect(
            lambda: self._on_correct_history("NG"))
        self._history_page.get_rebuild_button().clicked.connect(self._on_rebuild_from_history)

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

        device = self._config.get("camera.device_index", 0)
        QTimer.singleShot(500, lambda: self._camera_worker.start_camera(device))

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
                    # Green border for ROIs (same style as ROI editor)
                    pen = QPen(QColor("#22C55E"), 2)
                    qp.setPen(pen)
                    qp.drawRect(x, y, w, h)
                    # Small semi-transparent label
                    qp.setPen(Qt.NoPen)
                    qp.setBrush(QColor(34, 197, 94, 180))
                    qp.drawRect(x, y - 16 if y >= 16 else y, 50, 16)
                    qp.setPen(QColor("#FFFFFF"))
                    qp.drawText(x + 3, y - 4 if y >= 16 else y + 11, f"ROI{i+1}")
                qp.end()
            except Exception as e:
                logger.warning("ROI overlay draw error: %s", e)

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
        Only runs when on RUN tab to avoid CPU waste and counter drift."""
        if not self._inference_engine.is_loaded:
            return
        if not self._current_all_rois:
            return
        # Only infer on RUN tab — other tabs don't show inference results
        if self._tabs.currentIndex() != 0:
            return

        # ── Step 1: Part Presence Check ──
        pc_cfg = self._current_part_check_cfg
        if pc_cfg.get("enabled") and pc_cfg.get("has_master") and pc_cfg.get("gate_roi"):
            try:
                pc_result = pc_module.evaluate_part_presence(
                    frame, pc_cfg["gate_roi"], pc_cfg)
            except Exception as e:
                logger.warning("Part check error: %s", e)
                pc_result = None
            if pc_result is not None and not pc_result.ready:
                # Stop NG interval timer so phantom NG doesn't count
                if self._ng_interval_timer.isActive():
                    self._ng_interval_timer.stop()
                    self._ng_interval_active = False
                self._run_page.set_waiting_for_part()
                return
        # ── Step 2: QC Inference ──

        try:
            overall_ng = False
            worst_score = 0.0
            total_latency = 0.0
            roi_results = []

            for roi_rect in self._current_all_rois:
                roi_dict = {
                    "x": roi_rect[0], "y": roi_rect[1],
                    "width": roi_rect[2], "height": roi_rect[3],
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
                    delay = self._global_settings_page.get_ng_debounce_ms()
                    if delay > 0:
                        self._ng_interval_timer.start(delay)
                        self._ng_interval_active = True
                    # Show NG immediately on display
                    self._inspection_ng += 1  # count first NG
                    self._run_page.update_counters(
                        self._inspection_ok, self._inspection_ng)
                    self._run_page.update_judgement("NG", worst_score)
                else:
                    # Timer already running — update display (worse score)
                    self._run_page.update_judgement("NG", worst_score)

            # Diagnostics latency
            self._diagnostics_page.update_performance(
                0, 0, 0,
                self._inference_engine.latency_avg_ms,
                self._inference_engine.latency_p95_ms,
            )

            # Save OK to history (throttled)
            if raw_judgement == "OK":
                self._inference_save_counter += 1
                if self._inference_save_counter % 30 == 0:
                    self._db.add_inspection({
                        "program": self._active_program,
                        "score": worst_score,
                        "judgement": "OK",
                        "threshold": self._inference_engine.threshold,
                        "latency_ms": avg_latency,
                        "image_path": "",
                        "metadata": {"num_rois": len(roi_results)},
                    })

        except Exception as e:
            logger.warning("Inference error: %s", e)

    # ---- NG Interval Timer ----

    def _on_ng_interval_tick(self):
        """Interval timer tick — NG still ongoing, count another NG."""
        self._inspection_ng += 1
        self._run_page.update_counters(self._inspection_ok, self._inspection_ng)
        # Timer auto-restarts (QTimer with interval keeps firing)

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

        # Master thumbnail
        master_path = self._pm.get_part_check_master_image_path(
            self._active_program, self._active_template)
        if master_path and master_path.exists():
            from PySide6.QtGui import QPixmap
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

    def _init_programs(self):
        """Load programs and create default if none exist."""
        programs = self._pm.list_programs()
        if not programs:
            prog = self._pm.create_program("Default")
            self._active_program = prog["name"]
            # Create default template
            tmpl = self._pm.create_template(self._active_program, "Template 1")
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
            else:
                self._current_roi = None
                self._current_all_rois = []

            # Gallery thumbnails
            self._teach_page.clear_galleries()
            self._load_gallery_thumbnails("ok")
            self._load_gallery_thumbnails("ng")

            # Part Presence Check UI
            self._refresh_part_check_ui()
            self._refresh_part_check_gate_cache()

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
        """Capture frame from camera — or in import mode, save current import image."""
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

        # Save to template
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

        self._teach_page.show_import_mode(True)
        self._show_import_image()

    # ---- Import Review Helpers ----

    def _show_import_image(self):
        """Show current import image in the ROI editor."""
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

        # Convert BGR → RGB → QPixmap and show in ROI editor
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._teach_page.set_preview(pixmap)

        # Update import progress
        current = self._import_index + 1
        total = len(self._import_files)
        self._teach_page.set_import_status(current, total)
        self.set_status(f"Import: {current}/{total}", 2000)

    def _save_current_import_image(self, label: str):
        """Save the current import image under 'ok' or 'ng' and advance."""
        if self._import_index >= len(self._import_files):
            self._exit_import_mode()
            return

        path = self._import_files[self._import_index]
        img = cv2.imread(path)
        if img is not None:
            self._pm.save_template_image(
                self._active_program, self._active_template, img, label)
            logger.info("Import saved %s as %s", Path(path).name, label)
        else:
            logger.warning("Import: could not read %s", path)

        # Advance to next image
        self._import_index += 1
        self._show_import_image()

    def _exit_import_mode(self):
        """Exit import review mode and restore normal UI."""
        total = len(self._import_files)
        self._is_import_mode = False
        self._import_files = []
        self._import_index = 0

        self._teach_page.show_import_mode(False)
        self._refresh_template_ui()

        # If camera is still active, its frame_ready signal will refresh the preview.
        # If camera is off, show placeholder text.
        if not self._camera_worker or not self._camera_worker.is_running:
            self._teach_page.set_preview_text("Import selesai")

        self.set_status(f"Import selesai ({total} gambar diproses)", 3000)

    def _on_add_template(self):
        """Create a new template with default ROI."""
        name, ok = QInputDialog.getText(self, "Template Baru",
                                         "Nama template:")
        if ok and name.strip():
            try:
                tmpl = self._pm.create_template(self._active_program, name.strip())
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
        else:
            self._current_roi = None
            self._current_all_rois = []
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

    def _on_gallery_image_deleted(self, label: str):
        """Refresh gallery after image deletion."""
        self._refresh_template_ui()

    def _on_threshold_slider(self, value: int):
        """Update inference engine threshold when slider is moved."""
        threshold = value / 1000.0
        self._inference_engine.threshold = threshold
        self.set_status(f"Threshold: {threshold:.3f}", 2000)

    # ---- Training ----

    def _on_train(self):
        """Start training for active template via TrainingWorker."""
        if not self._active_template:
            self.set_status("Pilih template dulu!", 3000)
            return

        ok_count = self._pm.count_template_images(
            self._active_program, self._active_template, "ok")
        if ok_count < 1:
            QMessageBox.warning(self, "Training",
                                "Minimal 1 gambar OK diperlukan untuk training.")
            return

        # Pre-check torch availability (Windows DLL issue) — warning only
        torch_ok = True
        try:
            import torch  # noqa: F401
        except Exception:
            torch_ok = False

        if not torch_ok:
            logger.warning("PyTorch not available — using simple training mode")

        self._teach_page.set_training_progress(0, "Memulai training...")
        self._teach_page.get_train_button().setEnabled(False)
        logger.info("Training dimulai: program=%s, template=%s",
                     self._active_program, self._active_template)

        # Emit signal — worker di QThread akan menjalankan training
        self.start_training_signal.emit(
            self._active_program, self._active_template)

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
        """Enable/disable correction buttons based on selection."""
        data = self._history_page.get_selected_row_data()
        has_selection = data is not None
        self._history_page.get_correct_ok_button().setEnabled(has_selection)
        self._history_page.get_correct_ng_button().setEnabled(has_selection)

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
        """Refresh history page from database."""
        try:
            entries = self._db.get_history(program=self._active_program, limit=100)
            self._history_page.clear()
            for e in entries:
                self._history_page.add_entry(
                    entry_id=e["id"],
                    timestamp=e["timestamp"],
                    program=e["program"],
                    score=e["score"],
                    judgement=e["judgement"],
                    image_path=e.get("image_path", ""),
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

    def _on_tab_changed(self, index: int):
        page_names = ["Run", "Teach", "History", "Settings", "Diagnostics", "Akun", "Global"]
        name = page_names[index] if index < len(page_names) else f"Tab {index}"
        logger.debug("Switched to %s tab", name)

        # Refresh teach preview
        if index == 1:
            self._refresh_template_ui()
        # Refresh history when switching to HISTORY tab
        elif index == 2:
            self._refresh_history()
        # Refresh account page when switching to AKUN tab
        elif index == 5:
            self._account_page.refresh()
        # Load settings when switching to GLOBAL tab
        elif index == 6:
            self._global_settings_page._load_settings()

    def _show_about(self):
        QMessageBox.about(
            self,
            f"About {self._tr.tr('app_name')}",
            f"<h2>{self._tr.tr('app_name')} v1.0.0</h2>"
            f"<p>{self._tr.tr('app_title')}</p>"
            "<p>Built with Anomalib + OpenVINO + PySide6</p>"
            "<p>100% lokal, CPU-only, offline.</p>"
        )

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
