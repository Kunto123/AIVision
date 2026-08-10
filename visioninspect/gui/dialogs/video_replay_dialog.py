"""
VisionInspect - Video Replay Dialog
Dialog kontrol untuk uji model via file video (replay lewat jalur live).
NON-MODAL: Run page tetap terlihat di belakang sehingga operator bisa melihat
video + overlay ROI + judgement persis seperti kondisi kamera live.

Dialog hanya KONTROL + RINGKASAN. Frame/overlay ditampilkan oleh Run page
(jalur live yang sama). Stats di-update live dari main_window tiap frame
(ringkas: int + list kecil — murah).
"""

import os
import time
from pathlib import Path

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout,
    QFrame, QGroupBox, QGridLayout, QProgressBar, QSpinBox,
)


class VideoReplayDialog(QDialog):
    play_requested = Signal()
    pause_requested = Signal()
    seek_requested = Signal(int)
    stop_requested = Signal()
    frame_step_changed = Signal(int)   # Tugas 6a: "Periksa tiap N frame"
    closed = Signal()

    def __init__(self, video_path: str, total_frames: int, video_fps: float,
                 video_size: tuple, ref_dims, export_dir: str, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._total = max(0, int(total_frames))
        self._video_fps = video_fps
        self._video_size = video_size
        self._ref_dims = ref_dims
        self._export_dir = export_dir
        self._finished = False
        self._playing = False
        self._last_ng_len = -1
        self._proc_start = None
        self._proc_start_idx = 0

        self.setWindowTitle("Uji Model via Video — Replay")
        self.setMinimumWidth(460)
        self._build_ui()
        self._update_video_info()

    # ---- UI ----

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # Info video
        info_box = QGroupBox("Video")
        grid = QGridLayout(info_box)
        self._lbl_name = QLabel()
        self._lbl_info = QLabel()
        self._lbl_export = QLabel()
        self._lbl_name.setWordWrap(True)
        self._lbl_export.setWordWrap(True)
        self._lbl_export.setStyleSheet("color: #6B7280; font-size: 11px;")
        grid.addWidget(self._lbl_name, 0, 0, 1, 2)
        grid.addWidget(self._lbl_info, 1, 0, 1, 2)
        grid.addWidget(self._lbl_export, 2, 0, 1, 2)
        root.addWidget(info_box)

        # Progress + slider
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, max(1, self._total - 1))
        self._slider.setValue(0)
        self._slider.sliderMoved.connect(self._on_seek)
        self._lbl_progress = QLabel("Frame 0 / %d" % self._total)
        self._lbl_progress.setAlignment(Qt.AlignCenter)
        root.addWidget(self._lbl_progress)
        root.addWidget(self._slider)

        # Tombol kontrol
        btns = QHBoxLayout()
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setMinimumHeight(34)
        self._play_btn.clicked.connect(self._on_play_clicked)
        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.setMinimumHeight(34)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._open_btn = QPushButton("📁 Buka Folder Export")
        self._open_btn.setMinimumHeight(34)
        self._open_btn.clicked.connect(self._open_export_dir)
        btns.addWidget(self._play_btn)
        btns.addWidget(self._stop_btn)
        btns.addWidget(self._open_btn)
        root.addLayout(btns)

        # Cakupan uji (Tugas 6a). Di replay, cycle_delay produksi TIDAK
        # berlaku — kalau berlaku, video 30 dtk hanya terperiksa ±30 frame
        # dari 900 (3%) dan kejadian NG < 1 dtk bisa terlewat total. Uji
        # offline tidak butuh real-time; yang dibutuhkan cakupan.
        step_box = QGroupBox("Cakupan uji")
        step_layout = QVBoxLayout(step_box)
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Periksa tiap"))
        self._step_spin = QSpinBox()
        self._step_spin.setRange(1, 100)
        self._step_spin.setValue(1)
        self._step_spin.setSuffix(" frame")
        self._step_spin.setFixedWidth(120)
        self._step_spin.setToolTip(
            "1 = semua frame diperiksa (cakupan penuh, paling lambat).\n"
            "N > 1 = periksa 1 dari tiap N frame; frame yang dilewati tidak\n"
            "di-decode penuh sehingga jauh lebih ringan.\n\n"
            "Naikkan hanya kalau cacat yang dicari bertahan beberapa frame.\n"
            "Kejadian NG yang lebih pendek dari N frame bisa TERLEWAT.")
        self._step_spin.valueChanged.connect(self._on_step_changed)
        step_row.addWidget(self._step_spin)
        step_row.addStretch()
        step_layout.addLayout(step_row)
        self._lbl_step_note = QLabel()
        self._lbl_step_note.setWordWrap(True)
        self._lbl_step_note.setStyleSheet("color: #6B7280; font-size: 11px;")
        step_layout.addWidget(self._lbl_step_note)
        root.addWidget(step_box)
        self._update_step_note()

        # Ringkasan live
        stats_box = QGroupBox("Ringkasan (live)")
        stats_grid = QGridLayout(stats_box)
        self._lbl_total = self._stat_label()
        self._lbl_ok = self._stat_label()
        self._lbl_ng = self._stat_label()
        self._lbl_rate = self._stat_label()
        self._lbl_speed = self._stat_label()
        stats_grid.addWidget(QLabel("Frame diproses:"), 0, 0)
        stats_grid.addWidget(self._lbl_total, 0, 1)
        stats_grid.addWidget(QLabel("OK:"), 0, 2)
        stats_grid.addWidget(self._lbl_ok, 0, 3)
        stats_grid.addWidget(QLabel("NG:"), 1, 0)
        stats_grid.addWidget(self._lbl_ng, 1, 1)
        stats_grid.addWidget(QLabel("Pass rate:"), 1, 2)
        stats_grid.addWidget(self._lbl_rate, 1, 3)
        stats_grid.addWidget(QLabel("Kecepatan proses:"), 2, 0)
        stats_grid.addWidget(self._lbl_speed, 2, 1, 1, 3)
        root.addWidget(stats_box)

        # Daftar frame NG pertama (untuk lokasi cepat saat koreksi dataset)
        self._ng_box = QGroupBox("Kejadian NG (untuk koreksi dataset)")
        ng_layout = QVBoxLayout(self._ng_box)
        self._lbl_ng_list = QLabel("—")
        self._lbl_ng_list.setWordWrap(True)
        self._lbl_ng_list.setStyleSheet(
            "color: #EF4444; font-family: Consolas, monospace; font-size: 12px;")
        ng_layout.addWidget(self._lbl_ng_list)
        root.addWidget(self._ng_box)

        # Catatan mode uji
        note = QLabel(
            "⚠ Mode uji: PLC / counter produksi / history NONAKTIF. "
            "Frame OK & NG disimpan ke folder export saat judgement berubah "
            "dan sampling tiap 10 frame (koreksi dataset).")
        note.setWordWrap(True)
        note.setStyleSheet("color: #B45309; font-size: 11px;")
        root.addWidget(note)

    def _stat_label(self):
        lbl = QLabel("0")
        f = QFont("Segoe UI", 11, QFont.Bold)
        lbl.setFont(f)
        return lbl

    def _update_video_info(self):
        name = Path(self._video_path).name
        w, h = self._video_size
        ref_txt = (f"{self._ref_dims[0]}x{self._ref_dims[1]}"
                   if self._ref_dims else "tidak diketahui")
        fps_txt = f"{self._video_fps:.0f} fps" if self._video_fps > 0 else "fps?"
        self._lbl_name.setText(f"<b>{name}</b>")
        self._lbl_info.setText(
            f"Resolusi: {w}x{h} | {fps_txt} | Total frame: {self._total} | "
            f"Referensi template: {ref_txt}")
        self._lbl_export.setText(
            f"Export frame: {self._export_dir}/ok & /ng")

    # ---- Public API (dipanggil main_window) ----

    def update_progress(self, idx: int, total: int, video_fps: float):
        if self._finished:
            return
        self._total = max(0, int(total))
        self._lbl_progress.setText(f"Frame {idx + 1} / {self._total}")
        if self._slider.maximum() != max(1, self._total - 1):
            self._slider.setRange(0, max(1, self._total - 1))
            self._update_step_note()   # total berubah → hitung ulang cakupan
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)

        # Kecepatan proses (bukan fps video asli)
        now = time.monotonic()
        if self._proc_start is None:
            self._proc_start = now
            self._proc_start_idx = idx
        else:
            dt = now - self._proc_start
            if dt >= 1.0:
                proc_fps = (idx - self._proc_start_idx) / dt
                self._lbl_speed.setText(
                    f"{proc_fps:.1f} fps (video asli "
                    f"{self._video_fps:.0f} fps)" if self._video_fps > 0
                    else f"{proc_fps:.1f} fps")
                self._proc_start = now
                self._proc_start_idx = idx

    def update_stats(self, stats: dict):
        if self._finished:
            return
        total = stats.get("total", 0)
        ok = stats.get("ok", 0)
        ng = stats.get("ng", 0)
        rate = (ok / total * 100.0) if total else 0.0
        self._lbl_total.setText(str(total))
        self._lbl_ok.setText(str(ok))
        self._lbl_ng.setText(str(ng))
        self._lbl_rate.setText(f"{rate:.1f}%")

        # List NG — update hanya kalau ada kejadian baru (hemat refresh)
        ng_list = stats.get("ng_frames", [])
        if len(ng_list) != self._last_ng_len:
            self._last_ng_len = len(ng_list)
            if not ng_list:
                self._lbl_ng_list.setText("—")
            else:
                shown = ng_list[:8]
                txt = "\n".join(
                    f"  • Frame {idx + 1}  (score {score:.3f})"
                    for idx, score in shown)
                if len(ng_list) > 8:
                    txt += f"\n  … dan {len(ng_list) - 8} lagi"
                self._lbl_ng_list.setText(txt)

    def set_finished(self):
        """Replay berhenti (selesai/stop/error) — kontrol di-disable, tombol
        stop berubah jadi Tutup. Dialog tetap terbuka untuk melihat hasil."""
        if self._finished:
            return
        self._finished = True
        self._playing = False
        self._play_btn.setEnabled(False)
        self._slider.setEnabled(False)
        self._step_spin.setEnabled(False)
        self._stop_btn.setText("Tutup")

    # ---- Handlers ----

    def _on_play_clicked(self):
        if self._finished:
            return
        if self._playing:
            self._playing = False
            self._play_btn.setText("▶ Play")
            self.pause_requested.emit()
        else:
            self._playing = True
            self._play_btn.setText("⏸ Pause")
            self.play_requested.emit()

    def _on_stop_clicked(self):
        if self._finished:
            self.close()
        else:
            self._playing = False
            self._play_btn.setText("▶ Play")
            self.stop_requested.emit()

    def _on_seek(self, value: int):
        if self._finished:
            return
        self.seek_requested.emit(value)

    def _on_step_changed(self, value: int):
        """Tugas 6a: ubah cakupan uji saat replay berjalan (boleh live)."""
        self._update_step_note()
        if not self._finished:
            self.frame_step_changed.emit(int(value))

    def _update_step_note(self):
        """Terjemahkan N ke bahasa yang bisa dinilai operator: berapa frame
        yang benar-benar diperiksa, dan celah waktu yang tidak terlihat."""
        n = self._step_spin.value()
        checked = self._total // n if self._total else 0
        if n == 1:
            self._lbl_step_note.setText(
                f"Cakupan penuh — semua {self._total} frame diperiksa.")
            return
        pct = (checked / self._total * 100.0) if self._total else 0.0
        gap = (n / self._video_fps * 1000.0) if self._video_fps > 0 else 0.0
        gap_txt = (f" Celah antar pemeriksaan ≈ {gap:.0f} ms — kejadian NG "
                   f"lebih pendek dari itu bisa terlewat." if gap else "")
        self._lbl_step_note.setText(
            f"⚠ {checked} dari {self._total} frame diperiksa ({pct:.0f}%)."
            f"{gap_txt}")

    def _open_export_dir(self):
        try:
            p = Path(self._export_dir)
            if p.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        except Exception:
            pass

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
