"""
VisionInspect - Camera Worker (QThread)
Menjembatani CameraDevice ke GUI via Qt signals menggunakan QTimer polling.
QTimer dibuat LAZY di _do_start() agar thread affinity-nya benar (CameraThread).
"""

from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

from PySide6.QtCore import QMetaObject, QObject, QThread, QTimer, Qt, Signal, Slot, Q_ARG
from PySide6.QtGui import QImage, QPixmap

from visioninspect.core.camera import CameraDevice, CameraConfig, CameraError, CameraState
from visioninspect.utils.logging_setup import get_logger

logger = get_logger("camera")


class CameraWorker(QObject):
    """
    Worker untuk kamera yang berjalan di QThread terpisah.
    QTimer dibuat LAZY di _do_start() agar thread affinity-nya benar.
    """

    # Signals
    frame_ready = Signal(object)   # QPixmap untuk display
    frame_raw = Signal(object)     # np.ndarray untuk inference
    camera_started = Signal()
    camera_stopped = Signal()
    camera_error = Signal(str)
    fps_updated = Signal(float)
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._camera: Optional[CameraDevice] = None
        # Timer created in __init__ BEFORE moveToThread — child akan ikut
        # ke CameraThread saat moveToThread dipanggil.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._grab_frame)
        self._target_fps = 30
        self._running = False
        self._device_index = 0

    # ---- Private: start/stop timer (timer sudah dibuat di __init__) ----

    def _ensure_timer_running(self):
        """Start timer at target interval. Aman dipanggil dari thread mana pun
        karena QTimer sudah dibuat bersama parent di __init__ (sebelum moveToThread)."""
        if not self._timer.isActive():
            interval = max(16, int(1000 / self._target_fps))
            self._timer.start(interval)

    # ---- Public API ----

    @Slot(int)
    def start_camera(self, device_index: int = 0):
        """Open kamera dan mulai polling frame.
        Aman dipanggil dari thread mana pun — self-dispatch ke CameraThread."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(
                self, "start_camera", Qt.QueuedConnection,
                Q_ARG(int, device_index))
            return
        if self._running:
            self.stop_camera()
            QTimer.singleShot(200, lambda: self._do_start(device_index))
            return
        self._do_start(device_index)

    def _do_start(self, device_index: int):
        """Internal: benar-benar start kamera. HARUS di CameraThread."""
        self._device_index = device_index

        try:
            config = CameraConfig(device_index=device_index)
            self._camera = CameraDevice(config)
            self._camera.open()

            # Timer sudah dibuat di __init__, tinggal start
            self._ensure_timer_running()
            self._running = True

            self.camera_started.emit()
            self.status_message.emit(f"Kamera {device_index} aktif")
            logger.info("Camera worker started: device=%d", device_index)

        except CameraError as e:
            self.camera_error.emit(str(e))
            self.status_message.emit(f"Gagal buka kamera {device_index}")
            logger.error("Camera start failed: %s", e)
        except Exception as e:
            self.camera_error.emit(str(e))
            self.status_message.emit(f"Error: {e}")

    @Slot()
    def stop_camera(self):
        """Hentikan kamera dan polling. Aman dipanggil dari thread mana pun."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(self, "stop_camera", Qt.BlockingQueuedConnection)
            return
        if self._timer:
            self._timer.stop()
        self._running = False
        if self._camera:
            self._camera.close()
            self._camera = None
        self.camera_stopped.emit()
        self.status_message.emit("Kamera dimatikan")
        logger.info("Camera worker stopped")

    @Slot(int)
    def set_device(self, device_index: int):
        """Ganti device kamera (restart jika sedang running).
        Aman dipanggil dari thread mana pun."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(
                self, "set_device", Qt.QueuedConnection,
                Q_ARG(int, device_index))
            return
        was_running = self._running
        if was_running:
            self.stop_camera()
            QTimer.singleShot(300, lambda: self.start_camera(device_index))
        else:
            self.start_camera(device_index)

    @Slot()
    def toggle_camera(self):
        """Start/stop toggle. Aman dipanggil dari thread mana pun."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(
                self, "toggle_camera", Qt.QueuedConnection)
            return
        if self._running:
            self.stop_camera()
        else:
            self.start_camera(self._device_index)

    def get_frame(self) -> Optional[npt.NDArray]:
        if self._camera:
            return self._camera.get_frame()
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        if self._camera:
            return self._camera.fps
        return 0.0

    # ---- Internal: grab frame ----

    def _grab_frame(self):
        """Polling: ambil frame dari kamera dan emit signal."""
        if not self._camera:
            return

        # read() = one-shot cap.read(), update _latest_frame + FPS counter
        frame = self._camera.read()
        if frame is None:
            return

        # Emit raw frame untuk inference
        self.frame_raw.emit(frame)

        # Convert numpy array (BGR) ke QPixmap (RGB) untuk display
        try:
            # .copy() → contiguous + thread-safe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
            h, w, ch = rgb.shape
            qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
            if qimg.isNull():
                return

            pixmap = QPixmap.fromImage(qimg)
            self.frame_ready.emit(pixmap)

            fps = self._camera.fps
            if fps > 0:
                self.fps_updated.emit(fps)

        except Exception as e:
            logger.warning("Frame conversion error: %s", e)


class CameraThread(QThread):
    """QThread khusus untuk CameraWorker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[CameraWorker] = None

    def init_worker(self):
        """Buat worker di thread ini."""
        self.worker = CameraWorker()
        self.worker.moveToThread(self)

    def run(self):
        """Event loop thread."""
        self.exec()
