"""
VisionInspect - Video Replay Worker (QThread)
Membaca frame dari file video (cv2.VideoCapture) untuk uji model — "kamera
virtual". Desain SEQUENTIAL TOKEN: worker hanya membaca SATU frame per
panggilan `_next_frame()`, lalu berhenti dan menunggu sinyal ack dari GUI
sebelum membaca frame berikutnya. Ini memberi backpressure alami: kecepatan
replay = kecepatan proses frame (decode + infer di GUI thread), TANPA
penumpukan antrian signal yang bisa terjadi kalau pakai QTimer polling bebas
(kamera asli boleh polling bebas karena frame boleh di-drop; replay harus
memproses SEMUA frame supaya momen NG tidak terlewat).

Pola thread mengikuti camera_worker.py: QObject di QThread terpisah; method
public aman dipanggil dari thread mana pun (self-dispatch via
QMetaObject.invokeMethod). Untuk keperluan unit test, method juga bisa
dipanggil langsung (tanpa thread) — sama seperti CameraWorker.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import QMetaObject, QObject, QThread, Qt, Signal, Slot, Q_ARG
from PySide6.QtGui import QImage, QPixmap

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("video_replay")


class VideoReplayWorker(QObject):
    """Baca frame video untuk uji model — sequential token pattern."""

    # Signals
    frame_raw = Signal(object)      # np.ndarray (BGR) — untuk inference
    frame_ready = Signal(object)    # QPixmap — untuk display (+ROI overlay)
    progress = Signal(int, int, float)   # frame_idx, total_frames, video_fps
    opened = Signal(int, int, int, float)  # total_frames, width, height, fps
    finished = Signal()             # video habis (natural end)
    stopped = Signal()              # di-stop manual
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap: Optional[cv2.VideoCapture] = None
        self._path = ""
        self._running = False
        self._stop_flag = False
        self._total_frames = 0
        self._video_fps = 0.0
        self._w = 0
        self._h = 0
        self._last_frame_t = None
        # Pacing: batasi laju replay ≈ kecepatan live (30 fps) supaya CPU
        # tidak 100% terus selama uji — decode tetap di worker thread, GUI
        # tetap responsif. Sleep terjadi di thread worker, bukan GUI.
        self._target_fps = 30.0

    # ---- Public API (thread-safe, self-dispatch) ----

    def open(self, path: str) -> Tuple[int, int, int, float]:
        """Buka file video. Return (total_frames, width, height, fps).
        Raise ValueError kalau tidak bisa dibaca (codec/format)."""
        cap = cv2.VideoCapture(path)
        if cap is None or not cap.isOpened():
            raise ValueError(
                f"Tidak bisa membaca video: {path}\n"
                "Periksa format/codec (mp4/avi umumnya didukung). "
                "Video dari kamera yang sama dengan deploy seharusnya terbaca.")
        self._cap = cap
        self._path = path
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        self._video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        logger.info("Video opened: %s (%d frames, %dx%d, %.2f fps)",
                    path, self._total_frames, self._w, self._h, self._video_fps)
        return (self._total_frames, self._w, self._h, self._video_fps)

    @Slot()
    def start(self):
        """Mulai replay dari posisi saat ini (di-thread sendiri kalau perlu)."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(self, "start", Qt.QueuedConnection)
            return
        if self._cap is None:
            return
        self._stop_flag = False
        self._running = True
        self._next_frame()

    @Slot()
    def pause(self):
        """Jeda: berhenti meminta frame berikutnya (frame terakhir tetap tampil)."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(self, "pause", Qt.QueuedConnection)
            return
        self._running = False

    @Slot()
    def resume(self):
        """Lanjut replay dari posisi terakhir."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(self, "resume", Qt.QueuedConnection)
            return
        if not self._running and self._cap is not None:
            self._stop_flag = False
            self._running = True
            self._next_frame()

    @Slot(int)
    def seek_to(self, frame_idx: int):
        """Lompat ke frame index (posisi dipakai saat start/resume berikutnya)."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(
                self, "seek_to", Qt.QueuedConnection, Q_ARG(int, frame_idx))
            return
        if self._cap is None:
            return
        idx = max(0, int(frame_idx))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self.progress.emit(idx, self._total_frames, self._video_fps)

    @Slot()
    def stop(self):
        """Hentikan replay total: release capture + emit stopped."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(self, "stop", Qt.BlockingQueuedConnection)
            return
        self._stop_flag = True
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.stopped.emit()

    # ---- Token: baca SATU frame, emit, lalu TUNGGU ack ----

    @Slot()
    def _next_frame(self):
        """Slot token — dipanggil dari GUI thread via invokeMethod (ack).
        Membaca satu frame, emit signal, lalu berhenti sampai dipanggil lagi."""
        if not self._running or self._stop_flag or self._cap is None:
            return
        # Pacing: jaga laju ≤ target fps (default 30) — replay lebih halus,
        # CPU tidak tersiksa, dan debounce/cycle-delay berperilaku seperti
        # produksi nyata (yang justru tujuannya uji jalur live).
        if self._target_fps > 0:
            target_dt = 1.0 / self._target_fps
            now = time.monotonic()
            if self._last_frame_t is not None:
                dt = now - self._last_frame_t
                if dt < target_dt:
                    time.sleep(target_dt - dt)
            self._last_frame_t = time.monotonic()
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._running = False
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            logger.info("Video replay finished: %s", self._path)
            self.finished.emit()
            return

        idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if idx < 0:
            idx = 0

        self.frame_raw.emit(frame)
        self.progress.emit(idx, self._total_frames, self._video_fps)

        # Convert BGR → QPixmap untuk display (sama seperti camera_worker)
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
            h, w, ch = rgb.shape
            qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
            if not qimg.isNull():
                self.frame_ready.emit(QPixmap.fromImage(qimg))
        except Exception as e:
            logger.warning("Replay frame conversion error: %s", e)

    # ---- Properties ----

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def video_path(self) -> str:
        return self._path

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def video_fps(self) -> float:
        return self._video_fps

    @property
    def video_size(self) -> Tuple[int, int]:
        return (self._w, self._h)


class VideoReplayThread(QThread):
    """QThread khusus untuk VideoReplayWorker (pola CameraThread)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[VideoReplayWorker] = None

    def init_worker(self):
        self.worker = VideoReplayWorker()
        self.worker.moveToThread(self)

    def run(self):
        self.exec()
