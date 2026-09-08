"""
VisionInspect - Account Management Page (Admin only)
CRUD user + bind RFID (keyboard-wedge). db.txt aktif → list gabungan
akun lokal (users.json) + akun DB (DB_USER_TABLE), kolom Lokasi menandai asal.
"""

from PySide6.QtCore import Qt, QTimer, Signal, Slot, QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")

RFID_MIN_LEN = 4
RFID_MAX_LEN = 24
RFID_TIMEOUT_MS = 200


class AccountPage(QWidget):
    """Halaman ACCOUNT — manajemen user (admin only)."""

    roles_changed = Signal()

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self._db = db
        self._setup_ui()

        # RFID bind mode state
        self._bind_rfid_mode = False
        self._bind_target = None          # dict {id, username, source} baris terpilih
        self._rfid_buffer = ""
        self._rfid_timer = QTimer(self)
        self._rfid_timer.setSingleShot(True)
        self._rfid_timer.timeout.connect(self._flush_rfid_buffer)

    def _has_external(self) -> bool:
        return bool(getattr(self._db, "has_external", False))

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("👥 Manajemen Akun")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # Toolbar
        toolbar = QHBoxLayout()
        self._add_btn = QPushButton("+ Tambah User")
        self._add_btn.setObjectName("successButton")
        self._add_btn.clicked.connect(self._on_add_user)
        toolbar.addWidget(self._add_btn)

        self._edit_btn = QPushButton("Edit")
        self._edit_btn.setEnabled(False)
        self._edit_btn.clicked.connect(self._on_edit_user)
        toolbar.addWidget(self._edit_btn)

        self._delete_btn = QPushButton("Hapus")
        self._delete_btn.setObjectName("dangerButton")
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._on_delete_user)
        toolbar.addWidget(self._delete_btn)

        toolbar.addStretch()

        self._bind_rfid_btn = QPushButton("Bind RFID")
        self._bind_rfid_btn.setEnabled(False)
        self._bind_rfid_btn.setCheckable(True)
        self._bind_rfid_btn.clicked.connect(self._on_toggle_bind_rfid)
        toolbar.addWidget(self._bind_rfid_btn)

        self._unbind_rfid_btn = QPushButton("Unbind RFID")
        self._unbind_rfid_btn.setEnabled(False)
        self._unbind_rfid_btn.clicked.connect(self._on_unbind_rfid)
        toolbar.addWidget(self._unbind_rfid_btn)

        layout.addLayout(toolbar)

        # RFID binding status
        self._rfid_status = QLabel("")
        self._rfid_status.setStyleSheet("color: #F59E0B; font-weight: bold; padding: 4px;")
        self._rfid_status.hide()
        layout.addWidget(self._rfid_status)

        # Table — kolom Lokasi (Lokal/DB) di ujung
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "ID", "Username", "Nama", "Role", "RFID", "Tgl Buat", "Lokasi"
        ])
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table, 1)

        # Help text
        help_text = QLabel(
            "Pilih user, klik 'Bind RFID', lalu tap kartu RFID.\n"
            "Kolom Lokasi: 'Lokal' = users.json di PC ini, 'DB' = tabel DB_USER_TABLE.\n"
            "Login berhasil kalau akun cocok di salah satunya."
        )
        help_text.setObjectName("secondaryText")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

    # ---- Event Filter: RFID Wedge Capture (only active during bind mode) ----

    def eventFilter(self, obj, event):
        """Intercept key events during RFID bind mode."""
        if not self._bind_rfid_mode:
            return super().eventFilter(obj, event)

        if event.type() == QEvent.KeyPress:
            key_event = QKeyEvent(event)
            key = key_event.key()
            text = key_event.text()

            if text and text.isprintable() and len(text) == 1:
                self._rfid_buffer += text
                self._rfid_timer.start(RFID_TIMEOUT_MS)
                return True  # Consume event — don't interfere with table/widgets

            elif key in (Qt.Key_Return, Qt.Key_Enter):
                self._flush_rfid_buffer()
                return True

        return super().eventFilter(obj, event)

    def _flush_rfid_buffer(self):
        """Process buffered RFID UID and bind to target user."""
        self._rfid_timer.stop()
        uid = self._rfid_buffer.strip()

        if not uid or not self._bind_rfid_mode or not self._bind_target:
            self._rfid_buffer = ""
            return

        if RFID_MIN_LEN <= len(uid) <= RFID_MAX_LEN:
            t = self._bind_target
            logger.info("RFID bind: UID=%s user=%s (%s)", uid, t["username"], t["source"])
            if self._db.bind_rfid(t["id"], uid, **self._route(t)):
                self._rfid_status.setText(f"RFID {uid[:12]}... berhasil di-bind!")
                self._rfid_status.setStyleSheet("color: #22C55E; font-weight: bold; padding: 4px;")
                self.refresh()
            else:
                self._rfid_status.setText(f"RFID {uid[:12]}... sudah dipakai user lain!")
                self._rfid_status.setStyleSheet("color: #EF4444; font-weight: bold; padding: 4px;")
        else:
            self._rfid_status.setText(f"UID tidak valid ({len(uid)} chars)")
            self._rfid_status.setStyleSheet("color: #EF4444; font-weight: bold; padding: 4px;")

        self._rfid_buffer = ""
        self._exit_bind_mode()
        QTimer.singleShot(3000, self._rfid_status.hide)

    def _exit_bind_mode(self):
        """Exit RFID bind mode and remove event filter."""
        self._bind_rfid_mode = False
        self._bind_target = None
        self._bind_rfid_btn.setChecked(False)
        self.removeEventFilter(self)

    # ---- Public API ----

    def _route(self, sel: dict) -> dict:
        """kwargs source/username untuk MultiAuth; kosong kalau store tunggal."""
        if not self._has_external():
            return {}
        return {"source": sel["source"], "username": sel["username"]}

    @Slot()
    def refresh(self):
        """Reload user list (lokal + DB kalau db.txt aktif)."""
        try:
            users = self._db.list_users()
        except Exception:
            users = []
        self._table.setRowCount(0)
        for u in users:
            row = self._table.rowCount()
            self._table.insertRow(row)
            src = u.get("source", "lokal")
            id_item = QTableWidgetItem(str(u["id"]))
            id_item.setData(Qt.UserRole,
                            {"id": u["id"], "username": u["username"], "source": src})
            self._table.setItem(row, 0, id_item)
            self._table.setItem(row, 1, QTableWidgetItem(u["username"]))
            self._table.setItem(row, 2, QTableWidgetItem(u.get("display_name", "")))
            self._table.setItem(row, 3, QTableWidgetItem(u["role"]))
            rfid = u.get("rfid_uid", "")
            if rfid == "Bound":
                rfid_display = "Bound"
            elif rfid:
                rfid_display = f"{rfid[:8]}..."
            else:
                rfid_display = "—"
            self._table.setItem(row, 4, QTableWidgetItem(rfid_display))
            self._table.setItem(row, 5, QTableWidgetItem(str(u.get("created_at", ""))))
            self._table.setItem(row, 6, QTableWidgetItem("DB" if src == "db" else "Lokal"))
        self._table.resizeColumnsToContents()

    # ---- Handlers ----

    def _on_selection_changed(self):
        has = self._table.currentRow() >= 0
        self._edit_btn.setEnabled(has)
        self._delete_btn.setEnabled(has)
        self._unbind_rfid_btn.setEnabled(has)
        self._bind_rfid_btn.setEnabled(has)

    def _get_selected(self):
        """dict {id, username, source, display_name, role} atau None."""
        row = self._table.currentRow()
        if row < 0:
            return None
        meta = self._table.item(row, 0).data(Qt.UserRole) or {}
        return {
            "id": meta.get("id", int(self._table.item(row, 0).text())),
            "username": meta.get("username", self._table.item(row, 1).text()),
            "source": meta.get("source", "lokal"),
            "display_name": self._table.item(row, 2).text(),
            "role": self._table.item(row, 3).text(),
        }

    def _on_add_user(self):
        dialog = UserEditDialog(self._db, has_external=self._has_external(), parent=self)
        if dialog.exec():
            self.refresh()
            self.roles_changed.emit()

    def _on_edit_user(self):
        sel = self._get_selected()
        if not sel:
            return
        dialog = UserEditDialog(self._db, sel["id"], sel,
                                has_external=self._has_external(), parent=self)
        if dialog.exec():
            self.refresh()
            self.roles_changed.emit()

    def _on_delete_user(self):
        sel = self._get_selected()
        if not sel:
            return
        loc = "DB" if sel["source"] == "db" else "lokal"
        reply = QMessageBox.question(
            self, "Hapus User",
            f"Hapus user '{sel['username']}' ({loc})?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self._db.delete_user(sel["id"], **self._route(sel)):
                self.refresh()
                self.roles_changed.emit()
            else:
                QMessageBox.warning(self, "Gagal",
                                    "Tidak bisa menghapus admin terakhir!")

    def _on_toggle_bind_rfid(self, checked: bool):
        if checked:
            sel = self._get_selected()
            if not sel:
                self._bind_rfid_btn.setChecked(False)
                return
            self._bind_rfid_mode = True
            self._bind_target = sel
            self._rfid_buffer = ""
            self.installEventFilter(self)
            self._rfid_status.setText("Tap kartu RFID sekarang...")
            self._rfid_status.setStyleSheet("color: #F59E0B; font-weight: bold; padding: 4px;")
            self._rfid_status.show()
            self.setFocus()  # Ensure this widget can receive key events
        else:
            self._exit_bind_mode()
            self._rfid_status.hide()

    def _on_unbind_rfid(self):
        sel = self._get_selected()
        if not sel:
            return
        self._db.unbind_rfid(sel["id"], **self._route(sel))
        self.refresh()
        self._rfid_status.setText("RFID unbind berhasil")
        self._rfid_status.setStyleSheet("color: #22C55E; font-weight: bold; padding: 4px;")
        self._rfid_status.show()
        QTimer.singleShot(3000, self._rfid_status.hide)


class UserEditDialog(QDialog):
    """Dialog tambah / edit user. `source` = 'lokal' | 'db' untuk baris edit."""

    def __init__(self, db, user_id: int = None, current_data: dict = None,
                 has_external: bool = False, parent=None):
        super().__init__(parent)
        self._db = db
        self._user_id = user_id
        self._is_edit = user_id is not None
        self._has_external = has_external
        self._source = (current_data or {}).get("source", "lokal")
        self._username = (current_data or {}).get("username", "")

        self.setWindowTitle("Edit User" if self._is_edit else "Tambah User")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        title = QLabel("Edit User" if self._is_edit else "+ Tambah User")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(8)

        self._username_input = QLineEdit()
        if current_data:
            self._username_input.setText(current_data.get("username", ""))
        self._username_input.setMinimumHeight(32)
        if self._is_edit:
            self._username_input.setReadOnly(True)   # ganti username tidak didukung
        form.addRow("Username:", self._username_input)

        # "Simpan ke" — hanya saat tambah user & DB eksternal aktif
        self._target_combo = None
        if not self._is_edit and has_external:
            self._target_combo = QComboBox()
            self._target_combo.addItems(["Lokal", "DB"])
            form.addRow("Simpan ke:", self._target_combo)

        self._display_input = QLineEdit()
        if current_data:
            self._display_input.setText(current_data.get("display_name", ""))
        self._display_input.setPlaceholderText("Nama tampilan")
        self._display_input.setMinimumHeight(32)
        # Tabel DB user tidak punya kolom nama tampilan
        if self._is_edit and self._source == "db":
            self._display_input.setEnabled(False)
            self._display_input.setPlaceholderText("(akun DB tidak menyimpan nama)")
        form.addRow("Nama:", self._display_input)

        self._password_input = QLineEdit()
        self._password_input.setEchoMode(QLineEdit.Password)
        self._password_input.setPlaceholderText(
            "Kosongkan jika tidak ingin ganti" if self._is_edit else "Password")
        self._password_input.setMinimumHeight(32)
        form.addRow("Password:", self._password_input)

        self._role_combo = QComboBox()
        self._role_combo.addItems(["operator", "admin"])
        if current_data:
            idx = self._role_combo.findText(current_data.get("role", "operator"))
            if idx >= 0:
                self._role_combo.setCurrentIndex(idx)
        form.addRow("Role:", self._role_combo)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Simpan")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Batal")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)
        self.setMinimumWidth(380)

    def _on_save(self):
        username = self._username_input.text().strip()
        display_name = self._display_input.text().strip()
        password = self._password_input.text()
        role = self._role_combo.currentText()

        if not username:
            QMessageBox.warning(self, "Validasi", "Username harus diisi!")
            return

        try:
            if self._is_edit:
                if self._has_external:
                    kwargs = {"role": role, "source": self._source,
                              "username": self._username}
                    if self._source == "lokal":
                        kwargs["display_name"] = display_name
                    if password:
                        kwargs["password"] = password
                    self._db.update_user(self._user_id, **kwargs)
                else:
                    kwargs = {"display_name": display_name, "role": role}
                    if password:
                        kwargs["password"] = password
                    self._db.update_user(self._user_id, **kwargs)
            else:
                if not password:
                    QMessageBox.warning(self, "Validasi", "Password harus diisi!")
                    return
                if self._target_combo is not None:
                    src = self._target_combo.currentText().lower()  # "lokal" | "db"
                    self._db.add_user(username, password, display_name, role,
                                      source=src)
                else:
                    self._db.add_user(username, password, display_name, role)
        except Exception as e:
            logger.warning("Simpan user gagal: %s", e)
            QMessageBox.critical(self, "Gagal", f"Gagal menyimpan user:\n{e}")
            return

        self.accept()
