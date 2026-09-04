"""VisionInspect - Flow Gallery

Galeri thumbnail yang MENGALIR ke bawah (wrap) lalu di-scroll vertikal.

Sebelumnya galeri memakai satu QHBoxLayout: 136 thumbnail berarti satu baris
selebar ±11.000 px di dalam QScrollArea. Akibatnya scrollbar horizontal
panjang, lebar widget dalam meledak, dan tata letak panel kiri ikut rusak.

Di sini item ditata dalam grid yang jumlah kolomnya dihitung dari lebar
viewport, jadi lebarnya tidak pernah melebihi container dan yang tersisa
hanyalah scroll ke bawah. Ukuran tiap thumbnail tidak diubah (tetap seragam).
"""

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QScrollArea, QWidget


class FlowGallery(QScrollArea):
    """Container thumbnail: wrap otomatis + scroll vertikal saja."""

    def __init__(self, item_width: int = 78, item_height: int = 82,
                 spacing: int = 6, parent=None):
        super().__init__(parent)
        self._item_w = max(1, int(item_width))
        self._item_h = max(1, int(item_height))
        self._spacing = max(0, int(spacing))
        self._items: List[QWidget] = []
        self._cols = 0
        self._stretch_row = -1

        self.setWidgetResizable(True)
        # Kunci: tidak pernah ada scroll horizontal — isi wajib muat selebar
        # container, sisanya mengalir ke baris berikutnya.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._inner = QWidget()
        self._grid = QGridLayout(self._inner)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(self._spacing)
        # JANGAN setAlignment() di sini — bikin QScrollArea mengira isinya
        # selalu muat (scrollbar tak muncul). Pakai stretch di _relayout().
        self.setWidget(self._inner)

    # ---- API ----

    def add_widget(self, widget: QWidget) -> None:
        """Tambah 1 thumbnail di sel berikutnya — BUKAN menata ulang grid
        (menata ulang tiap penambahan = O(n²), berat di PC edge)."""
        if self._cols <= 0:
            self._cols = self._column_count()
        idx = len(self._items)
        self._items.append(widget)
        self._grid.addWidget(widget, idx // self._cols, idx % self._cols)
        self._apply_stretch()

    def clear(self) -> None:
        """Buang semua thumbnail (widget benar-benar dihapus)."""
        for w in self._items:
            self._grid.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self._items.clear()
        self._cols = 0
        self._stretch_row = -1
        self._inner.updateGeometry()

    def count(self) -> int:
        return len(self._items)

    # ---- Internal ----

    def _column_count(self) -> int:
        avail = self.viewport().width() - 8          # dikurangi margin grid
        step = self._item_w + self._spacing
        return max(1, (avail + self._spacing) // step)

    def _apply_stretch(self) -> None:
        """Dorong isi ke kiri-atas + segarkan jangkauan scroll. `updateGeometry()`
        WAJIB, dan stretch baris lama harus dinolkan dulu."""
        cols = max(1, self._cols)
        self._grid.setColumnStretch(cols, 1)
        rows = (len(self._items) + cols - 1) // cols
        if self._stretch_row >= 0 and self._stretch_row != rows:
            self._grid.setRowStretch(self._stretch_row, 0)
        self._grid.setRowStretch(rows, 1)
        self._stretch_row = rows
        self._inner.updateGeometry()

    def _relayout(self, force: bool = False) -> None:
        cols = self._column_count()
        if cols == self._cols and not force:
            return                                    # jumlah kolom tetap
        self._cols = cols
        # Lepas semua dulu supaya penempatan ulang tidak menumpuk
        while self._grid.count():
            self._grid.takeAt(0)
        for i, w in enumerate(self._items):
            self._grid.addWidget(w, i // cols, i % cols)
        self._apply_stretch()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Hanya menata ulang bila jumlah kolom benar-benar berubah — resize
        # kecil tidak memicu penataan 100+ widget.
        self._relayout()
