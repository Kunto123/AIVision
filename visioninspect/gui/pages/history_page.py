"""
VisionInspect - History Page
Riwayat inspeksi, filter, koreksi (redefinition), rebuild, rollback.
"""

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from visioninspect.utils.i18n import Translator


class HistoryPage(QWidget):
    """Halaman HISTORY — riwayat hasil inspeksi dengan fitur koreksi dan tuning."""

    tuning_requested = Signal(int)  # entry_id — emitted when Tuning button clicked

    def __init__(self, translator: Translator, parent=None):
        super().__init__(parent)
        self._tr = translator
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel(self._tr.tr("history_title"))
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # Filter bar
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel(self._tr.tr("history_filter") + ":"))

        self._filter_combo = QComboBox()
        self._filter_combo.addItems([
            self._tr.tr("history_all"),
            self._tr.tr("history_filter_ok"),
            self._tr.tr("history_filter_ng"),
        ])
        filter_layout.addWidget(self._filter_combo)
        filter_layout.addStretch()

        # Action buttons
        self._correct_ok_btn = QPushButton(self._tr.tr("history_mark_ok"))
        self._correct_ok_btn.setObjectName("successButton")
        self._correct_ok_btn.setEnabled(False)
        filter_layout.addWidget(self._correct_ok_btn)

        self._correct_ng_btn = QPushButton(self._tr.tr("history_mark_ng"))
        self._correct_ng_btn.setObjectName("dangerButton")
        self._correct_ng_btn.setEnabled(False)
        filter_layout.addWidget(self._correct_ng_btn)

        self._rebuild_btn = QPushButton(self._tr.tr("history_rebuild"))
        self._rebuild_btn.setObjectName("primaryButton")
        filter_layout.addWidget(self._rebuild_btn)

        self._tuning_btn = QPushButton("🔧 Tuning")
        self._tuning_btn.setObjectName("primaryButton")
        self._tuning_btn.setEnabled(False)
        self._tuning_btn.setToolTip("Buka mode tuning untuk koreksi per-ROI + additional learning")
        self._tuning_btn.clicked.connect(self._on_tuning)
        filter_layout.addWidget(self._tuning_btn)

        self._rollback_btn = QPushButton(self._tr.tr("history_rollback", version="?"))
        self._rollback_btn.setEnabled(False)
        filter_layout.addWidget(self._rollback_btn)

        layout.addLayout(filter_layout)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "ID", self._tr.tr("history_date"), self._tr.tr("history_program"),
            self._tr.tr("history_score"), self._tr.tr("history_judgement"),
            self._tr.tr("history_image"), self._tr.tr("history_correct")
        ])
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(False)

        # Empty state
        self._table.setRowCount(0)
        layout.addWidget(self._table, 1)

        # Status
        self._status_label = QLabel(self._tr.tr("history_no_data"))
        self._status_label.setObjectName("secondaryText")
        layout.addWidget(self._status_label)

    # ---- Public API ----

    @Slot()
    def add_entry(self, entry_id: int, timestamp: str, program: str,
                  score: float, judgement: str, image_path: str,
                  corrected: bool = False):
        """Add a row to the history table."""
        row = self._table.rowCount()
        self._table.insertRow(row)

        self._table.setItem(row, 0, QTableWidgetItem(str(entry_id)))
        self._table.setItem(row, 1, QTableWidgetItem(timestamp))
        self._table.setItem(row, 2, QTableWidgetItem(program))
        self._table.setItem(row, 3, QTableWidgetItem(f"{score:.4f}"))

        judgement_item = QTableWidgetItem(judgement)
        if judgement == "OK":
            judgement_item.setForeground(Qt.green)
        else:
            judgement_item.setForeground(Qt.red)
        self._table.setItem(row, 4, judgement_item)

        self._table.setItem(row, 5, QTableWidgetItem(image_path))
        self._table.setItem(row, 6, QTableWidgetItem(
            "✓" if corrected else ""
        ))

        self._status_label.setText(f"{self._table.rowCount()} entries")

    @Slot()
    def clear(self):
        self._table.setRowCount(0)
        self._status_label.setText(self._tr.tr("history_no_data"))

    @Slot()
    def set_status(self, text: str):
        self._status_label.setText(text)

    def get_selected_row_data(self) -> dict | None:
        """Get data from selected row, or None.

        Returns dict with id, judgement, image_path (from table columns).
        """
        row = self._table.currentRow()
        if row < 0:
            return None
        return {
            "id": int(self._table.item(row, 0).text()),
            "judgement": self._table.item(row, 4).text(),
            "image_path": self._table.item(row, 5).text() if self._table.item(row, 5) else "",
        }

    def _on_tuning(self):
        """Emit tuning signal with selected entry id."""
        data = self.get_selected_row_data()
        if data:
            self.tuning_requested.emit(data["id"])

    def get_filter_combo(self) -> QComboBox:
        return self._filter_combo

    def get_correct_ok_button(self) -> QPushButton:
        return self._correct_ok_btn

    def get_correct_ng_button(self) -> QPushButton:
        return self._correct_ng_btn

    def get_rebuild_button(self) -> QPushButton:
        return self._rebuild_btn

    def get_tuning_button(self) -> QPushButton:
        return self._tuning_btn

    def get_rollback_button(self) -> QPushButton:
        return self._rollback_btn

    def get_table(self):
        return self._table
