"""
VisionInspect - Camera Thread
Abstraksi driver kamera untuk frame grabbing di thread terpisah.
Saat ini: OpenCV cv2.VideoCapture. Dirancang agar mudah ditambah GigE/GenICam.
"""

import time
from enum import Enum
from threading import Event, Lock
from queue import Queue, Full as QueueFull
from typing import Callable, Optional

import cv2
import numpy as np
import numpy.typing as npt

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("camera")


class CameraError(Exception):
    """Base camera exception."""
    pass


class CameraState(Enum):
    CLOSED = "closed"
    OPENING = "opening"
    RUNNING = "running"
    ERROR = "error"
    CLOSING = "closing"


class CameraConfig:
    """Camera configuration data class."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps_target: int = 30,
        exposure: int = -1,  # -1 = auto
        backend: Optional[int] = None,
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps_target = fps_target
        self.exposure = exposure
        # Auto-detect Windows backend if not specified
        if backend is None:
            import platform
            if platform.system() == "Windows":
                self.backend = cv2.CAP_DSHOW
            else:
                self.backend = None
        else:
            self.backend = backend

    def apply_to(self, cap: cv2.VideoCapture) -> None:
        """Apply configuration to an opened VideoCapture."""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        if self.exposure >= 0:
            cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        else:
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # auto
            except AttributeError:
                pass  # Some OpenCV builds don't have this constant


class CameraDevice:
    """
    Thread-safe wrapper around a single camera device.
    Mengelola frame grabbing di thread terpisah.
    """

    def __init__(self, config: CameraConfig):
        self._config = config
        self._state = CameraState.CLOSED
        self._state_lock = Lock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._stop_event = Event()
        self._frame_queue: Queue = Queue(maxsize=2)  # bounded, drop-oldest
        self._frame_callback: Optional[Callable[[npt.NDArray], None]] = None
        self._latest_frame: Optional[npt.NDArray] = None
        self._latest_frame_lock = Lock()
        self._fps_counter = _FPSCounter()

    # ---- Properties ----

    @property
    def state(self) -> CameraState:
        with self._state_lock:
            return self._state

    @property
    def fps(self) -> float:
        return self._fps_counter.fps

    # ---- Lifecycle ----

    def open(self) -> None:
        """Open camera device."""
        with self._state_lock:
            if self._state == CameraState.RUNNING:
                logger.warning("Camera already running")
                return
            self._state = CameraState.OPENING

        try:
            if self._config.backend is not None:
                self._cap = cv2.VideoCapture(self._config.device_index, self._config.backend)
            else:
                self._cap = cv2.VideoCapture(self._config.device_index)

            if not self._cap.isOpened():
                raise CameraError(f"Failed to open camera device {self._config.device_index}")

            self._config.apply_to(self._cap)
            logger.info(
                "Camera opened: device=%d resolution=%dx%d",
                self._config.device_index,
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )

            with self._state_lock:
                self._state = CameraState.RUNNING
        except Exception as e:
            with self._state_lock:
                self._state = CameraState.ERROR
            logger.error("Camera open failed: %s", e)
            raise CameraError(str(e)) from e

    def start(self) -> None:
        """Start frame grabbing in a loop (call from worker thread)."""
        self._stop_event.clear()
        self._fps_counter.reset()

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                logger.warning("Camera not opened, stopping grab loop")
                break

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame grab failed")
                time.sleep(0.01)
                continue

            self._fps_counter.tick()

            # Store latest frame
            with self._latest_frame_lock:
                self._latest_frame = frame.copy()

            # Push to queue (drop oldest if full)
            try:
                self._frame_queue.put_nowait(frame)
            except QueueFull:
                try:
                    self._frame_queue.get_nowait()  # drop oldest
                    self._frame_queue.put_nowait(frame)
                except QueueFull:
                    pass  # shouldn't happen

            # Callback
            if self._frame_callback:
                try:
                    self._frame_callback(frame)
                except Exception as e:
                    logger.error("Frame callback error: %s", e)

        # Cleanup on stop
        with self._state_lock:
            self._state = CameraState.CLOSED
        logger.info("Camera grab loop ended")

    def stop(self) -> None:
        """Signal the grab loop to stop."""
        self._stop_event.set()

    def close(self) -> None:
        """Close camera and release resources."""
        with self._state_lock:
            self._state = CameraState.CLOSING
        self.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._state_lock:
            self._state = CameraState.CLOSED
        logger.info("Camera closed")

    # ---- One-shot Read (untuk QTimer polling) ----

    def read(self) -> Optional[npt.NDArray]:
        """
        Read a single frame (non-blocking, one-shot).
        Returns frame or None on failure.
        Cocok untuk QTimer polling: panggil read() tiap tick timer.
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        self._fps_counter.tick()

        # Update latest frame
        with self._latest_frame_lock:
            self._latest_frame = frame.copy()

        return frame

    # ---- Frame Access ----

    def get_frame(self) -> Optional[npt.NDArray]:
        """Get the latest frame (non-blocking)."""
        with self._latest_frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def wait_for_frame(self, timeout: float = 1.0) -> Optional[npt.NDArray]:
        """Get a frame from queue (blocking up to timeout)."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except Exception:
            return None

    def set_frame_callback(self, callback: Optional[Callable[[npt.NDArray], None]]) -> None:
        """Set callback for every frame."""
        self._frame_callback = callback

    # ---- Configuration ----

    def update_config(self, **kwargs) -> None:
        """Update camera config parameters."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        # Re-apply if running
        if self._cap and self._cap.isOpened():
            self._config.apply_to(self._cap)


class _FPSCounter:
    """Simple FPS counter using rolling window."""

    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._times: list[float] = []

    def reset(self) -> None:
        self._times.clear()

    def tick(self) -> None:
        now = time.monotonic()
        self._times.append(now)
        # Keep only window
        while len(self._times) > self._window_size:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed
