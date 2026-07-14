"""
VisionInspect - Diagnostics Page
Log live, metrik performa, status thread, tes PLC.
Auto-refresh setiap 2 detik.
"""

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from visioninspect.utils.i18n import Translator


class DiagnosticsPage(QWidget):
    """Halaman DIAGNOSTICS — monitoring performa dan troubleshooting."""

    def __init__(self, translator: Translator, parent=None):
        super().__init__(parent)
        self._tr = translator
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # === Left: Live Logs ===
        left_panel = QFrame()
        left_panel.setObjectName("cardPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("📋 " + self._tr.tr("diagnostics_logs"))
        title.setObjectName("sectionTitle")
        left_layout.addWidget(title)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(1000)
        self._log_view.setStyleSheet(
            "background-color: #0A0F1A; color: #9FB3C8; "
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 11px;"
        )
        self._log_view.appendPlainText("[System] VisionInspect Diagnostics started")
        left_layout.addWidget(self._log_view, 1)

        log_btn_layout = QHBoxLayout()
        self._clear_log_btn = QPushButton("🗑 Clear")
        log_btn_layout.addWidget(self._clear_log_btn)
        log_btn_layout.addStretch()
        left_layout.addLayout(log_btn_layout)

        layout.addWidget(left_panel, 2)

        # === Right: Performance + PLC Test ===
        # Wrapped in a QScrollArea (defensive — matches TEACH/SETTINGS/RUN pattern)
        # so this card stack can't reintroduce the overlap bug if content grows.
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
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(16)
        right_scroll.setWidget(right_content)

        # Performance group
        perf_group = QGroupBox("📊 " + self._tr.tr("diagnostics_performance"))
        perf_layout = QVBoxLayout(perf_group)
        perf_layout.setSpacing(8)

        # RAM
        ram_frame = QFrame()
        ram_frame.setObjectName("cardPanel")
        ram_layout = QHBoxLayout(ram_frame)
        ram_layout.setContentsMargins(8, 4, 8, 4)
        ram_layout.addWidget(QLabel("🧠 RAM:"))
        self._ram_label = QLabel(self._tr.tr("diagnostics_ram_usage", mb="—"))
        self._ram_label.setStyleSheet("font-weight: bold;")
        ram_layout.addWidget(self._ram_label)
        ram_layout.addStretch()
        perf_layout.addWidget(ram_frame)

        # CPU
        cpu_frame = QFrame()
        cpu_frame.setObjectName("cardPanel")
        cpu_layout = QHBoxLayout(cpu_frame)
        cpu_layout.setContentsMargins(8, 4, 8, 4)
        cpu_layout.addWidget(QLabel("⚡ CPU:"))
        self._cpu_label = QLabel(self._tr.tr("diagnostics_cpu_usage", percent="—"))
        self._cpu_label.setStyleSheet("font-weight: bold;")
        cpu_layout.addWidget(self._cpu_label)
        cpu_layout.addStretch()
        perf_layout.addWidget(cpu_frame)

        # Camera FPS
        fps_frame = QFrame()
        fps_frame.setObjectName("cardPanel")
        fps_layout = QHBoxLayout(fps_frame)
        fps_layout.setContentsMargins(8, 4, 8, 4)
        fps_layout.addWidget(QLabel("📷 FPS Kamera:"))
        self._fps_label = QLabel(self._tr.tr("diagnostics_fps", fps="—"))
        self._fps_label.setStyleSheet("font-weight: bold;")
        fps_layout.addWidget(self._fps_label)
        fps_layout.addStretch()
        perf_layout.addWidget(fps_frame)

        # Latency
        lat_frame = QFrame()
        lat_frame.setObjectName("cardPanel")
        lat_layout = QHBoxLayout(lat_frame)
        lat_layout.setContentsMargins(8, 4, 8, 4)
        lat_layout.addWidget(QLabel("⏱ Inferensi:"))
        self._latency_label = QLabel(self._tr.tr("diagnostics_inference_latency", ms="—"))
        self._latency_label.setStyleSheet("font-weight: bold;")
        lat_layout.addWidget(self._latency_label)
        lat_layout.addStretch()
        perf_layout.addWidget(lat_frame)

        # P95
        p95_frame = QFrame()
        p95_frame.setObjectName("cardPanel")
        p95_layout = QHBoxLayout(p95_frame)
        p95_layout.setContentsMargins(8, 4, 8, 4)
        p95_layout.addWidget(QLabel("📈 Latensi P95:"))
        self._latency_p95_label = QLabel(self._tr.tr("diagnostics_latency_p95", ms="—"))
        self._latency_p95_label.setStyleSheet("font-weight: bold;")
        p95_layout.addWidget(self._latency_p95_label)
        p95_layout.addStretch()
        perf_layout.addWidget(p95_frame)

        right_layout.addWidget(perf_group)

        # Thread status
        thread_group = QGroupBox("🧵 " + self._tr.tr("diagnostics_threads"))
        thread_layout = QVBoxLayout(thread_group)
        self._thread_status_label = QLabel(
            "📷 Camera: ⏹ —\n"
            "🧠 Inference: ⏹ —\n"
            "🔌 PLC: ⏹ —\n"
            "🎓 Training: ⏹ —"
        )
        self._thread_status_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        thread_layout.addWidget(self._thread_status_label)
        right_layout.addWidget(thread_group)

        # PLC Test
        plc_test_group = QGroupBox("🔌 " + self._tr.tr("diagnostics_plc_test"))
        plc_test_layout = QVBoxLayout(plc_test_group)

        self._send_test_btn = QPushButton("📤 " + self._tr.tr("diagnostics_plc_send_test"))
        self._send_test_btn.setMinimumHeight(36)
        plc_test_layout.addWidget(self._send_test_btn)

        self._plc_test_result = QLabel("")
        self._plc_test_result.setObjectName("secondaryText")
        plc_test_layout.addWidget(self._plc_test_result)

        right_layout.addWidget(plc_test_group)

        right_layout.addStretch()
        layout.addWidget(right_panel, 1)

    # ---- Public API ----

    @Slot()
    def append_log(self, message: str):
        self._log_view.appendPlainText(message)
        # Auto-scroll to bottom
        scrollbar = self._log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot()
    def update_performance(self, ram_mb: float, cpu_percent: float,
                           fps: float, latency_avg: float, latency_p95: float):
        self._ram_label.setText(f"{ram_mb:.0f} MB")
        self._cpu_label.setText(f"{cpu_percent:.1f}%")
        self._fps_label.setText(f"{fps:.1f}")
        self._latency_label.setText(f"{latency_avg:.1f} ms")
        self._latency_p95_label.setText(f"{latency_p95:.1f} ms")

    @Slot()
    def update_thread_status(self, camera: str, inference: str, plc: str, training: str):
        self._thread_status_label.setText(
            f"📷 Camera: {camera}\n"
            f"🧠 Inference: {inference}\n"
            f"🔌 PLC: {plc}\n"
            f"🎓 Training: {training}"
        )

    @Slot()
    def set_plc_test_result(self, result: str):
        self._plc_test_result.setText(result)

    def get_send_test_button(self) -> QPushButton:
        return self._send_test_btn

    def get_clear_log_button(self) -> QPushButton:
        return self._clear_log_btn
