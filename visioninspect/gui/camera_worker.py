"""
VisionInspect - Camera Worker (QThread)
Menjembatani CameraDevice ke GUI via Qt signals menggunakan QTimer polling.

QTimer dibuat di __init__ sebagai CHILD worker, SEBELUM moveToThread — jadi
affinity-nya ikut pindah ke CameraThread. Konsekuensinya: start/stop timer
WAJIB terjadi di CameraThread. Semua jalur yang menyentuh timer
(_ensure_timer_running / _ensure_timer_stopped / stop_camera / set_polling)
melakukan self-dispatch, jangan dilewati.
"""

from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

from PySide6.QtCore import QMetaObject, QObject, QThread, QTimer, Qt, Signal, Slot, Q_ARG
from PySide6.QtGui import QImage

from visioninspect.core.camera import CameraDevice, CameraConfig, CameraError, CameraState
from visioninspect.utils.logging_setup import get_logger

logger = get_logger("camera")


class CameraWorker(QObject):
    #: Frame gagal beruntun sebelum operator diberi tahu. Di 30 fps ini
    #: ~0,5 detik — cukup lama untuk melewati kedipan sesaat.
    _BAD_FRAME_WARN = 15

    #: Frame gagal beruntun sebelum kamera dibuka ulang sendiri (~1,5 detik).
    #: Sengaja jauh lebih longgar dari ambang peringatan: membuka ulang device
    #: memakan waktu, jadi jangan dilakukan untuk gangguan sekejap.
    _BAD_FRAME_RECOVER = 45

    """
    Worker untuk kamera yang berjalan di QThread terpisah.
    QTimer dibuat LAZY di _do_start() agar thread affinity-nya benar.
    """

    # Signals
    frame_ready = Signal(object)   # QImage untuk display (konversi di GUI)
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
        # F2: param kamera dari Settings (exposure/gain/white_balance)
        self._camera_params: dict = {}
        # Deteksi kamera lepas — lihat _note_capture_failure().
        self._consec_bad = 0
        self._recovering = False

    # ---- F2: kamera config dari Settings ----

    def set_camera_config(self, cfg: dict) -> None:
        """Terapkan config kamera (exposure/gain/WB) dari Settings (F2).

        Sebelum fix ini, CameraConfig hanya dibuat dari device_index —
        exposure/gain/white_balance di Settings TIDAK pernah sampai ke
        kamera (semua auto). Param ini diteruskan ke CameraConfig saat
        start; dipanggil ulang setelah save settings (restart kamera).
        """
        self._camera_params = {}
        # Key config kamera: resolution_width/resolution_height — petakan ke
        # param CameraConfig (width/height). Key lain langsung cocok.
        if cfg.get("resolution_width") is not None:
            self._camera_params["width"] = int(cfg["resolution_width"])
        if cfg.get("resolution_height") is not None:
            self._camera_params["height"] = int(cfg["resolution_height"])
        for k in ("fps_target", "exposure", "gain", "white_balance"):
            v = cfg.get(k)
            if v is not None:
                self._camera_params[k] = v
        if "fps_target" in self._camera_params:
            self._target_fps = int(self._camera_params["fps_target"])

    # ---- Private: start/stop timer (timer sudah dibuat di __init__) ----

    @Slot()
    def _ensure_timer_running(self):
        """Start timer at target interval.

        KOREKSI: komentar lama ("aman dipanggil dari thread mana pun karena
        QTimer dibuat bersama parent di __init__") SALAH. Yang ikut pindah
        lewat moveToThread adalah kepemilikan/affinity-nya; start/stop QTimer
        tetap WAJIB terjadi di thread pemilik timer. Memanggilnya dari GUI
        thread menghasilkan "QObject::killTimer: Timers cannot be stopped
        from another thread" dan timer bisa gagal berhenti (polling terus
        jalan walau kamera sudah stop). Karena itu di-dispatch sendiri."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(
                self, "_ensure_timer_running", Qt.QueuedConnection)
            return
        if not self._timer.isActive():
            interval = max(16, int(1000 / self._target_fps))
            self._timer.start(interval)

    @Slot()
    def _ensure_timer_stopped(self):
        """Stop timer — sama seperti start, WAJIB di thread pemilik timer."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(
                self, "_ensure_timer_stopped", Qt.QueuedConnection)
            return
        if self._timer is not None and self._timer.isActive():
            self._timer.stop()

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

    def restart_camera(self, device_index: int = 0):
        """Stop lalu start ulang kamera (dipakai saat settings kamera berubah,
        supaya exposure/gain/WB yang baru benar-benar diterapkan — F2)."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(
                self, "restart_camera", Qt.QueuedConnection,
                Q_ARG(int, device_index))
            return
        self.stop_camera()
        QTimer.singleShot(250, lambda: self._do_start(device_index))

    def _do_start(self, device_index: int):
        """Internal: benar-benar start kamera. HARUS di CameraThread."""
        self._device_index = device_index
        self._consec_bad = 0

        try:
            config = CameraConfig(device_index=device_index,
                                  **self._camera_params)
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
        self._ensure_timer_stopped()
        self._running = False
        self._consec_bad = 0
        # Batalkan pemulihan yang mungkin sedang tertunda. Tanpa ini,
        # QTimer.singleShot dari _note_capture_failure akan menghidupkan
        # kembali kamera yang baru saja sengaja dimatikan operator.
        self._recovering = False
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

    @Slot(bool)
    def set_polling(self, enabled: bool):
        """Jeda/lanjut polling frame TANPA menutup kamera (Tugas 2 — hemat
        CPU saat tidak ada konsumen frame: tab non-RUN/TEACH & bukan replay).
        Kamera tetap terbuka, jadi kembali ke RUN langsung jalan tanpa
        restart. Aman dipanggil dari thread mana pun (self-dispatch)."""
        if self.thread() is not QThread.currentThread():
            QMetaObject.invokeMethod(
                self, "set_polling", Qt.QueuedConnection,
                Q_ARG(bool, enabled))
            return
        if enabled:
            self._ensure_timer_running()
        else:
            self._ensure_timer_stopped()
        logger.info("Camera polling %s", "ON" if enabled else "OFF")

    def get_frame(self) -> Optional[npt.NDArray]:
        if self._camera:
            return self._camera.get_frame()
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_polling_frames(self) -> bool:
        """True bila timer grab frame benar-benar berjalan.

        `is_running` hanya berarti kamera terbuka — polling bisa sengaja
        dihentikan (`set_polling(False)` saat tab non-RUN/replay, Tugas 2)
        tanpa menutup device. Heartbeat PLC harus mengikuti polling: kalau
        frame tidak lagi diambil, tidak ada yang bisa dinilai, jadi "sehat"
        juga bohon.
        """
        return bool(self._timer is not None and self._timer.isActive())

    @property
    def fps(self) -> float:
        if self._camera:
            return self._camera.fps
        return 0.0

    # ---- Internal: grab frame ----

    def _frame_is_blank(self, frame) -> bool:
        """True bila frame benar-benar kosong (semua piksel nol).

        Kegagalan kamera USB yang paling menipu: device tetap "terbuka",
        cap.read() tetap mengembalikan ret=True, tapi isinya nol semua.
        Layar jadi hitam sementara ROI tetap tergambar dengan rasio benar —
        persis gejala yang dilaporkan.

        Sensor sungguhan selalu punya derau, jadi frame yang BENAR-BENAR nol
        tidak pernah sah. Diperiksa dengan subsampel supaya murah: 1080p
        jadi ~500 piksel, bukan 2 juta.
        """
        try:
            return not frame[::48, ::48].any()
        except Exception:
            return False

    def _note_capture_failure(self, sebab: str) -> None:
        """Hitung kegagalan beruntun; pulihkan sendiri bila melewati batas.

        Sebelum ini `_grab_frame` hanya `return` tanpa jejak: kamera bisa
        lepas dan aplikasi diam saja sampai operator sadar dan menekan
        stop/start sendiri.
        """
        self._consec_bad += 1
        if self._consec_bad == self._BAD_FRAME_WARN:
            self.status_message.emit(
                f"Kamera bermasalah ({sebab}) — mencoba memulihkan...")
            logger.warning("Kamera bermasalah (%s), %d frame beruntun",
                           sebab, self._consec_bad)
        if self._consec_bad < self._BAD_FRAME_RECOVER:
            return
        if self._recovering:
            return
        self._recovering = True
        self._consec_bad = 0
        idx = self._device_index
        logger.error("Kamera tidak memberi frame sah (%s) — membuka ulang "
                     "device %d", sebab, idx)
        self.camera_error.emit(
            f"Kamera terputus ({sebab}). Membuka ulang device {idx}...")
        try:
            self._ensure_timer_stopped()
            self._running = False
            if self._camera:
                self._camera.close()
                self._camera = None
        except Exception as e:
            logger.warning("Menutup kamera saat pemulihan gagal: %s", e)
        # Beri jeda supaya driver melepas handle sebelum dibuka lagi.
        QTimer.singleShot(400, self._recover_reopen)

    def _recover_reopen(self) -> None:
        # stop_camera() membatalkan pemulihan dengan mengosongkan flag ini —
        # jangan menghidupkan kamera yang sengaja dimatikan operator.
        if not self._recovering:
            return
        try:
            self._do_start(self._device_index)
            if self._running:
                self.status_message.emit("Kamera pulih")
                logger.info("Kamera pulih di device %d", self._device_index)
        finally:
            self._recovering = False

    def _grab_frame(self):
        """Polling: ambil frame dari kamera dan emit signal."""
        if not self._camera or self._recovering:
            return

        # read() = one-shot cap.read(), update _latest_frame + FPS counter
        frame = self._camera.read()
        if frame is None:
            self._note_capture_failure("tidak ada frame")
            return
        if self._frame_is_blank(frame):
            self._note_capture_failure("frame kosong")
            return
        if self._consec_bad:
            self._consec_bad = 0

        # Emit raw frame untuk inference
        self.frame_raw.emit(frame)

        # Tugas 7: worker emit QImage (aman lintas thread) — konversi ke
        # QPixmap dilakukan di GUI thread (_on_frame_received).
        try:
            # .copy() → contiguous + thread-safe (QImage tidak punya referensi
            # ke buffer worker; buffer QImage dipakai GUI thread nanti)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
            h, w, ch = rgb.shape
            qimg = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
            if qimg.isNull():
                return

            self.frame_ready.emit(qimg)

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
