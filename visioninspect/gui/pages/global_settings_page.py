"""
VisionInspect - Global Settings Page (Admin only)
Pengaturan global aplikasi: NG debounce, dll.
"""

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")


class GlobalSettingsPage(QWidget):
    """Halaman pengaturan global — hanya untuk admin."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("⚙️ Pengaturan Global")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # === NG Settings ===
        ng_group = QGroupBox("🔴 NG Detection")
        ng_form = QFormLayout(ng_group)
        ng_form.setSpacing(8)

        self._ng_delay_spin = QSpinBox()
        self._ng_delay_spin.setRange(0, 999999)
        self._ng_delay_spin.setValue(500)
        self._ng_delay_spin.setSuffix(" ms")
        self._ng_delay_spin.setSingleStep(100)
        self._ng_delay_spin.setFixedWidth(130)
        ng_form.addRow("NG Timeout:", self._ng_delay_spin)

        ng_help = QLabel(
            "Jeda minimum sebelum NG dihitung.\n"
            "Timer mulai saat anomali terdeteksi.\n"
            "Setiap kali timer habis → NG +1, lalu timer reset.\n"
            "0 = instant (setiap frame NG dihitung).")
        ng_help.setObjectName("secondaryText")
        ng_help.setWordWrap(True)
        ng_form.addRow(ng_help)

        layout.addWidget(ng_group)
        layout.addStretch()

        # Save
        self._save_btn = QPushButton("💾 Simpan Pengaturan")
        self._save_btn.setObjectName("primaryButton")
        self._save_btn.setMinimumHeight(40)
        self._save_btn.clicked.connect(self._on_save)
        layout.addWidget(self._save_btn)

        # Status
        self._status_label = QLabel("")
        self._status_label.setObjectName("secondaryText")
        layout.addWidget(self._status_label)

    def _load_settings(self):
        """Load settings from config."""
        ng_delay = self._config.get("ng_debounce_ms", 500)
        self._ng_delay_spin.setValue(ng_delay)

    def _on_save(self):
        """Save settings to config."""
        self._config.set("ng_debounce_ms", self._ng_delay_spin.value())
        self._status_label.setText("✅ Pengaturan tersimpan!")
        logger.info("Global settings saved: ng_debounce_ms=%d",
                     self._ng_delay_spin.value())

    # ---- Widget accessors ----

    def get_ng_delay_spin(self) -> QSpinBox:
        return self._ng_delay_spin

    def get_save_button(self) -> QPushButton:
        return self._save_btn

    def get_ng_debounce_ms(self) -> int:
        """Get current NG debounce delay from config (not UI)."""
        return self._config.get("ng_debounce_ms", 500)

    def get_status_label(self) -> QLabel:
        return self._status_label
