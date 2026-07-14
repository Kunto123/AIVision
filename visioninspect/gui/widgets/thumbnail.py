"""
VisionInspect - Thumbnail Widget
Clickable thumbnail with small X overlay button at top-right corner.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout


class ThumbnailWidget(QWidget):
    """A thumbnail image with X delete button overlaid at top-right."""

    clicked = Signal(str)    # path
    deleted = Signal(str)    # path

    def __init__(self, pixmap: QPixmap, path: str = "",
                 border_color: str = "#22C55E", parent=None):
        super().__init__(parent)
        self._path = path
        self.setFixedSize(78, 82)

        # Main layout - stacked
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Container for image + X overlay
        img_container = QWidget()
        img_container.setFixedSize(74, 74)
        img_container.setStyleSheet(f"""
            background: #111D30;
            border: 2px solid {border_color};
            border-radius: 4px;
        """)

        img_container.setLayout(QVBoxLayout())
        img_container.layout().setContentsMargins(0, 0, 0, 0)

        # Clickable image label
        self._img = QLabel(img_container)
        thumb = pixmap.scaled(70, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._img.setPixmap(thumb)
        self._img.setGeometry(2, 2, 70, 70)
        self._img.setAlignment(Qt.AlignCenter)
        self._img.mousePressEvent = lambda e: self.clicked.emit(self._path)

        # X delete button — inside image, top-right corner
        self._del_btn = QPushButton("✕", img_container)
        self._del_btn.setGeometry(52, 2, 20, 20)
        self._del_btn.setStyleSheet("""
            QPushButton {
                background: rgba(239, 68, 68, 200); color: white;
                border: 1px solid rgba(255,255,255,180);
                border-radius: 10px;
                font-size: 11px; font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover { background: #DC2626; }
        """)
        self._del_btn.clicked.connect(lambda: self.deleted.emit(self._path))

        layout.addWidget(img_container, alignment=Qt.AlignCenter)

        # Label below
        name = QLabel(path.split("/")[-1].split("\\")[-1][:10] if path else "img")
        name.setStyleSheet("color: #9FB3C8; font-size: 8px; border: none;")
        name.setAlignment(Qt.AlignCenter)
        layout.addWidget(name, alignment=Qt.AlignCenter)

    def path(self) -> str:
        return self._path
