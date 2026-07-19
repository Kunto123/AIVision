"""
VisionInspect - Login Dialog
Modal login dengan support username/password dan RFID tap card.
Event filter untuk keyboard wedge RFID reader.
"""

from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QKeyEvent, QPixmap, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QMessageBox,
    QFrame,
    QGraphicsDropShadowEffect,
)

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")

RFID_MIN_LEN, RFID_MAX_LEN, RFID_TIMEOUT_MS = 4, 24, 200


class LoginDialog(QDialog):
    """Login dialog with manual login + RFID card tap support."""

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self._db = db
        self._user = None
        self._rfid_buffer = ""
        self._rfid_timer = QTimer(self)
        self._rfid_timer.setSingleShot(True)
        self._rfid_timer.timeout.connect(self._flush_rfid_buffer)

        self.setWindowTitle("VisionInspect — Login")
        self.setModal(True)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
            | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        # Full screen: tutupi seluruh layar agar view aplikasi tidak
        # terlihat/dipakai sebelum login berhasil.
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        self.setStyleSheet("QDialog { background: #0A0F1A; }")

        self._setup_ui()
        self.installEventFilter(self)

    def showEvent(self, event):
        """Pastikan tampil full screen saat dialog dibuka."""
        super().showEvent(event)
        self.showFullScreen()

    def _setup_ui(self):
        # Halaman full-screen: card login di-tengah-kan via layout + stretch
        page = QVBoxLayout(self)
        page.setContentsMargins(0, 0, 0, 0)
        page.addStretch(1)
        center_row = QHBoxLayout()
        center_row.addStretch(1)

        # Outer container (card) with shadow — ukuran tetap
        outer = QFrame()
        outer.setObjectName("cardPanel")
        outer.setStyleSheet("""
            #cardPanel { background: #0F1A2E; border: 1px solid #233A57;
                         border-radius: 12px; }
        """)
        outer.setFixedSize(400, 430)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(Qt.black)
        shadow.setOffset(0, 4)
        outer.setGraphicsEffect(shadow)

        center_row.addWidget(outer)
        center_row.addStretch(1)
        page.addLayout(center_row)
        page.addStretch(1)

        layout = QVBoxLayout(outer)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(12)

        # Logo / Title
        title = QLabel("🔐 VisionInspect")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #22C55E;")
        layout.addWidget(title)

        sub = QLabel("Masuk dengan akun atau tap kartu RFID")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #9FB3C8; font-size: 12px; margin-bottom: 4px;")
        layout.addWidget(sub)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #233A57; margin: 4px 0;")
        layout.addWidget(sep)

        # Username
        self._username_input = QLineEdit()
        self._username_input.setPlaceholderText("Username")
        self._username_input.setMinimumHeight(38)
        self._username_input.setStyleSheet(
            "background: #1A2A44; border: 1px solid #233A57; border-radius: 6px;"
            " color: #E2E8F0; padding: 0 10px; font-size: 14px;")
        layout.addWidget(self._username_input)

        # Password
        self._password_input = QLineEdit()
        self._password_input.setPlaceholderText("Password")
        self._password_input.setEchoMode(QLineEdit.Password)
        self._password_input.setMinimumHeight(38)
        self._password_input.setStyleSheet(
            "background: #1A2A44; border: 1px solid #233A57; border-radius: 6px;"
            " color: #E2E8F0; padding: 0 10px; font-size: 14px;")
        self._password_input.returnPressed.connect(self._on_login)
        layout.addWidget(self._password_input)

        # Login button
        self._login_btn = QPushButton("🔑 Login")
        self._login_btn.setObjectName("primaryButton")
        self._login_btn.setMinimumHeight(42)
        self._login_btn.setStyleSheet(
            "#primaryButton { background: #22C55E; color: #0A0F1A; font-weight: bold;"
            " border-radius: 8px; font-size: 15px; }"
            "#primaryButton:hover { background: #16A34A; }"
            "#primaryButton:pressed { background: #15803D; }")
        self._login_btn.clicked.connect(self._on_login)
        layout.addWidget(self._login_btn)

        # RFID hint
        hint_frame = QFrame()
        hint_frame.setStyleSheet(
            "background: #1A2A44; border: 1px dashed #F59E0B; border-radius: 6px;")
        hl = QHBoxLayout(hint_frame)
        hl.setContentsMargins(8, 6, 8, 6)
        self._rfid_hint = QLabel("💳 Tap kartu RFID untuk masuk otomatis")
        self._rfid_hint.setStyleSheet("color: #F59E0B; font-size: 11px; background: transparent;")
        self._rfid_hint.setAlignment(Qt.AlignCenter)
        hl.addWidget(self._rfid_hint)
        layout.addWidget(hint_frame)

        layout.addStretch()

    # ---- Event Filter ----

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            ke = QKeyEvent(event)
            t = ke.text()
            if t and t.isprintable() and len(t) == 1:
                self._rfid_buffer += t
                self._rfid_timer.start(RFID_TIMEOUT_MS)
            elif ke.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._flush_rfid_buffer()
        return super().eventFilter(obj, event)

    def _flush_rfid_buffer(self):
        self._rfid_timer.stop()
        uid = self._rfid_buffer.strip()
        if not uid:
            return
        if RFID_MIN_LEN <= len(uid) <= RFID_MAX_LEN:
            self._process_rfid(uid)
        self._rfid_buffer = ""

    def _process_rfid(self, uid: str):
        user = self._db.get_user_by_rfid(uid)
        if user:
            self._user = user
            logger.info("RFID login: %s (role=%s)", user["username"], user["role"])
            self.accept()
        else:
            self._rfid_hint.setText(f"❌ Kartu tidak terdaftar ({uid[:12]}...)")
            self._rfid_hint.setStyleSheet("color: #EF4444; font-size: 11px; background: transparent;")
            QTimer.singleShot(3000, self._reset_hint)

    def _reset_hint(self):
        self._rfid_hint.setText("💳 Tap kartu RFID untuk masuk otomatis")
        self._rfid_hint.setStyleSheet("color: #F59E0B; font-size: 11px; background: transparent;")

    def _on_login(self):
        u = self._username_input.text().strip()
        p = self._password_input.text()
        if not u or not p:
            QMessageBox.warning(self, "Login", "Masukkan username dan password!")
            return
        user = self._db.authenticate(u, p)
        if user:
            self._user = user
            self.accept()
        else:
            # Bedakan koneksi DB gagal vs kredensial salah. Backend PostgreSQL
            # mengembalikan None untuk KEDUANYA, sehingga koneksi gagal dulu
            # tampil sbg "password salah" yang membingungkan.
            msg = "Username atau password salah!"
            connect = getattr(self._db, "_connect", None)
            if callable(connect):
                try:
                    conn = connect()
                    conn.close()
                except Exception:
                    msg = ("Koneksi database (PostgreSQL) gagal.\n"
                           "Periksa pengaturan di tab Settings.")
            QMessageBox.warning(self, "Login Gagal", msg)
            self._password_input.clear()
            self._password_input.setFocus()

    @property
    def user(self): return self._user
    @property
    def username(self): return self._user["username"] if self._user else ""
    @property
    def role(self): return self._user["role"] if self._user else "operator"
    @property
    def display_name(self):
        return self._user.get("display_name", self._user.get("username", ""))
