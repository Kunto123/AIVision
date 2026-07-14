"""
VisionInspect - Run Page
Layar utama operator: live view, judgement OK/NG, counter, status PLC.
"""

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from visioninspect.utils.i18n import Translator


class RunPage(QWidget):
    """Halaman RUN — mode inspeksi utama untuk operator."""

    def __init__(self, translator: Translator, parent=None):
        super().__init__(parent)
        self._tr = translator
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # === Camera Control Bar ===
        cam_ctrl = QFrame()
        cam_ctrl.setObjectName("cardPanel")
        cam_ctrl.setMaximumHeight(56)
        ctrl_layout = QHBoxLayout(cam_ctrl)
        ctrl_layout.setContentsMargins(12, 6, 12, 6)

        # Template selector (syncs with TEACH page)
        ctrl_layout.addWidget(QLabel("Template:"))
        self._template_combo = QComboBox()
        self._template_combo.setMinimumWidth(160)
        self._template_combo.setToolTip("Pilih template aktif untuk inspeksi")
        ctrl_layout.addWidget(self._template_combo)

        ctrl_layout.addSpacing(12)

        ctrl_layout.addWidget(QLabel("Kamera:"))
        self._device_spin = QSpinBox()
        self._device_spin.setRange(0, 10)
        self._device_spin.setValue(0)
        self._device_spin.setToolTip("Pilih index device kamera (0=default)")
        ctrl_layout.addWidget(self._device_spin)

        self._cam_toggle_btn = QPushButton("🔴 Start Kamera")
        self._cam_toggle_btn.setObjectName("primaryButton")
        self._cam_toggle_btn.setMinimumWidth(140)
        ctrl_layout.addWidget(self._cam_toggle_btn)

        self._heatmap_btn = QPushButton("🔥 Heatmap")
        self._heatmap_btn.setCheckable(True)
        self._heatmap_btn.setChecked(False)
        self._heatmap_btn.setToolTip("Tampilkan overlay heatmap anomaly di live view")
        ctrl_layout.addWidget(self._heatmap_btn)

        self._cam_status_icon = QLabel("⏹ Mati")
        self._cam_status_icon.setStyleSheet("color: #EF4444; font-weight: bold;")
        ctrl_layout.addWidget(self._cam_status_icon)

        ctrl_layout.addStretch()

        ctrl_layout.addWidget(QLabel("Trigger:"))
        self._trigger_mode_label = QLabel(self._tr.tr("run_trigger_continuous"))
        self._trigger_mode_label.setObjectName("secondaryText")
        ctrl_layout.addWidget(self._trigger_mode_label)

        self._trigger_btn = QPushButton(self._tr.tr("run_trigger_now"))
        self._trigger_btn.setObjectName("primaryButton")
        ctrl_layout.addWidget(self._trigger_btn)

        layout.addWidget(cam_ctrl)

        # === Main: Live View + Right Panel ===
        main_layout = QHBoxLayout()
        main_layout.setSpacing(16)

        # === Left: Live View (big) ===
        left_panel = QFrame()
        left_panel.setObjectName("cardPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)

        # Live view
        self._live_view = QLabel("Kamera belum aktif\n\nTekan 'Start Kamera' untuk memulai")
        self._live_view.setAlignment(Qt.AlignCenter)
        self._live_view.setMinimumSize(640, 480)
        self._live_view.setStyleSheet(
            "background-color: #0A0F1A; border: 1px solid #233A57; border-radius: 4px;"
            "color: #9FB3C8; font-size: 16px;"
        )
        self._live_view.setScaledContents(False)
        left_layout.addWidget(self._live_view, 1)

        left_layout.addWidget(QLabel("Live View"))

        main_layout.addWidget(left_panel, 3)

        # === Right Panel: Judgement + Counters ===
        right_panel = QFrame()
        right_panel.setObjectName("cardPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(16)

        # Judgement big display
        self._judgement_label = QLabel("—")
        self._judgement_label.setObjectName("judgementOK")
        self._judgement_label.setAlignment(Qt.AlignCenter)
        self._judgement_label.setMinimumHeight(100)
        self._judgement_label.setStyleSheet("color: #9FB3C8; font-size: 48px; font-weight: bold;")
        right_layout.addWidget(self._judgement_label)

        # Score
        score_frame = QFrame()
        score_frame.setObjectName("cardPanel")
        score_layout = QHBoxLayout(score_frame)
        score_layout.setContentsMargins(12, 8, 12, 8)
        score_layout.addWidget(QLabel(self._tr.tr("run_score") + ":"))
        self._score_label = QLabel("0.000")
        self._score_label.setObjectName("bigCounter")
        score_layout.addWidget(self._score_label)
        score_layout.addStretch()
        right_layout.addWidget(score_frame)

        # Latency + FPS side by side
        perf_frame = QFrame()
        perf_frame.setObjectName("cardPanel")
        perf_layout = QHBoxLayout(perf_frame)
        perf_layout.setContentsMargins(12, 8, 12, 8)

        perf_layout.addWidget(QLabel(self._tr.tr("run_latency") + ":"))
        self._latency_label = QLabel("— ms")
        self._latency_label.setObjectName("secondaryText")
        perf_layout.addWidget(self._latency_label)

        perf_layout.addSpacing(20)

        perf_layout.addWidget(QLabel(self._tr.tr("run_fps") + ":"))
        self._fps_label = QLabel("—")
        self._fps_label.setObjectName("secondaryText")
        perf_layout.addWidget(self._fps_label)

        perf_layout.addStretch()
        right_layout.addWidget(perf_frame)

        # Counters
        self._counter_frame = QFrame()
        self._counter_frame.setObjectName("cardPanel")
        counter_layout = QGridLayout(self._counter_frame)
        counter_layout.setContentsMargins(12, 8, 12, 8)

        # Header
        counter_layout.addWidget(QLabel(""), 0, 0)
        counter_layout.addWidget(QLabel("Total"), 0, 1)
        counter_layout.addWidget(QLabel("OK"), 0, 2)
        counter_layout.addWidget(QLabel("NG"), 0, 3)

        # Nilai
        counter_layout.addWidget(QLabel(self._tr.tr("run_counter_total")), 1, 0)
        self._total_counter = QLabel("0")
        self._total_counter.setObjectName("bigCounter")
        self._total_counter.setAlignment(Qt.AlignCenter)
        counter_layout.addWidget(self._total_counter, 1, 1)

        self._ok_counter = QLabel("0")
        self._ok_counter.setObjectName("bigCounter")
        self._ok_counter.setAlignment(Qt.AlignCenter)
        self._ok_counter.setStyleSheet("color: #22C55E; font-size: 28px; font-weight: bold;")
        counter_layout.addWidget(self._ok_counter, 1, 2)

        self._ng_counter = QLabel("0")
        self._ng_counter.setObjectName("bigCounter")
        self._ng_counter.setAlignment(Qt.AlignCenter)
        self._ng_counter.setStyleSheet("color: #EF4444; font-size: 28px; font-weight: bold;")
        counter_layout.addWidget(self._ng_counter, 1, 3)

        right_layout.addWidget(self._counter_frame)

        right_layout.addStretch()

        # Status + PLC + Per-ROI Results + Active Model
        status_frame = QFrame()
        status_frame.setObjectName("cardPanel")
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(12, 8, 12, 8)

        # Active model indicator
        self._model_indicator = QLabel("🔴 Model: —")
        self._model_indicator.setStyleSheet("font-weight: bold; color: #9FB3C8;")
        status_layout.addWidget(self._model_indicator)

        # PLC status
        plc_row = QHBoxLayout()
        plc_row.addWidget(QLabel(self._tr.tr("run_plc_status") + ":"))
        self._plc_status_label = QLabel("⏹ " + self._tr.tr("disconnected"))
        self._plc_status_label.setObjectName("plcDisconnected")
        plc_row.addWidget(self._plc_status_label)
        plc_row.addStretch()
        status_layout.addLayout(plc_row)

        # Per-ROI Results
        self._roi_results_label = QLabel("ROI: —")
        self._roi_results_label.setObjectName("secondaryText")
        self._roi_results_label.setWordWrap(True)
        status_layout.addWidget(self._roi_results_label)

        # Status message
        self._status_msg = QLabel(self._tr.tr("ready"))
        self._status_msg.setObjectName("secondaryText")
        status_layout.addWidget(self._status_msg)

        right_layout.addWidget(status_frame)

        main_layout.addWidget(right_panel, 1)
        layout.addLayout(main_layout, 1)

    # ---- Public API ----

    @Slot()
    def update_judgement(self, judgement: str, score: float):
        """Update judgement display."""
        if judgement == "OK":
            self._judgement_label.setText("✅ " + self._tr.tr("ok"))
            self._judgement_label.setStyleSheet("color: #22C55E; font-size: 56px; font-weight: bold;")
        else:
            self._judgement_label.setText("❌ " + self._tr.tr("ng"))
            self._judgement_label.setStyleSheet("color: #EF4444; font-size: 56px; font-weight: bold;")
        self._score_label.setText(f"{score:.4f}")

    @Slot()
    def update_latency(self, ms: float):
        self._latency_label.setText(f"{ms:.1f} ms")

    @Slot()
    def update_fps(self, fps: float):
        self._fps_label.setText(f"{fps:.1f}")

    @Slot()
    def update_counters(self, total: int, ok_count: int, ng_count: int):
        self._total_counter.setText(str(total))
        self._ok_counter.setText(str(ok_count))
        self._ng_counter.setText(str(ng_count))

    @Slot()
    def set_plc_status(self, connected: bool):
        if connected:
            self._plc_status_label.setText("✅ " + self._tr.tr("connected"))
            self._plc_status_label.setStyleSheet("color: #22C55E; font-weight: bold;")
        else:
            self._plc_status_label.setText("⏹ " + self._tr.tr("disconnected"))
            self._plc_status_label.setStyleSheet("color: #EF4444; font-weight: bold;")

    @Slot()
    def set_model_info(self, template_name: str, model_loaded: bool, threshold: float = 0.5):
        """Show active template + model info on RUN page."""
        if model_loaded:
            self._model_indicator.setText(
                f"🟢 [{template_name}] Model aktif | Threshold: {threshold:.3f}")
            self._model_indicator.setStyleSheet("font-weight: bold; color: #22C55E;")
        else:
            self._model_indicator.setText(f"🟡 [{template_name}] Belum ditraining")
            self._model_indicator.setStyleSheet("font-weight: bold; color: #F59E0B;")

    @Slot()
    def set_frame(self, pixmap: QPixmap):
        """Set live view pixmap (dari kamera)."""
        self._live_view.setPixmap(pixmap.scaled(
            self._live_view.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        ))

    @Slot()
    def set_status_message(self, msg: str):
        self._status_msg.setText(msg)

    @Slot()
    def update_roi_results(self, results: list):
        """Update per-ROI results display."""
        lines = []
        colors = ["#FFFFFF", "#FFD700", "#00BFFF", "#FF69B4", "#7B68EE"]
        for i, r in enumerate(results):
            roi = r["roi"]
            label = f"ROI{i+1}"
            judge = r["judgement"]
            icon = "✅" if judge == "OK" else "❌"
            color = colors[i % len(colors)]
            lines.append(
                f'<span style="color:{color}">{icon} {label}:'
                f' score={r["score"]:.3f} ({judge})</span>'
            )
        self._roi_results_label.setText("<br>".join(lines))

    @Slot()
    def clear_results(self):
        """Clear judgement, score, latency and ROI results when switching template."""
        self._judgement_label.setText("—")
        self._judgement_label.setStyleSheet("color: #9FB3C8; font-size: 48px; font-weight: bold;")
        self._score_label.setText("0.000")
        self._latency_label.setText("— ms")
        self._roi_results_label.setText("ROI: —")

    @Slot()
    def set_camera_status(self, active: bool):
        """Update camera status indicator."""
        if active:
            self._cam_status_icon.setText("▶ Aktif")
            self._cam_status_icon.setStyleSheet("color: #22C55E; font-weight: bold;")
            self._cam_toggle_btn.setText("⏹ Stop Kamera")
        else:
            self._cam_status_icon.setText("⏹ Mati")
            self._cam_status_icon.setStyleSheet("color: #EF4444; font-weight: bold;")
            self._cam_toggle_btn.setText("▶ Start Kamera")

    # ---- Widget accessors ----

    def get_camera_toggle_button(self) -> QPushButton:
        return self._cam_toggle_btn

    def get_device_spin(self) -> QSpinBox:
        return self._device_spin

    def get_trigger_button(self) -> QPushButton:
        return self._trigger_btn

    def get_heatmap_button(self) -> QPushButton:
        return self._heatmap_btn

    def get_template_combo(self) -> QComboBox:
        return self._template_combo
