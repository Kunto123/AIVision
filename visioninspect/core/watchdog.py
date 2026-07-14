"""
VisionInspect - Watchdog Monitor
Memantau thread-thread aplikasi dan merestart jika hang.
"""

import time
import threading
from typing import Callable, Dict, Optional

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("app")


class Watchdog:
    """
    Watchdog internal untuk memonitor thread.
    Jika thread hang > N detik, restart dan catat event.
    """

    def __init__(
        self,
        check_interval: float = 2.0,
        inference_timeout: float = 10.0,
        camera_timeout: float = 5.0,
    ):
        self._check_interval = check_interval
        self._inference_timeout = inference_timeout
        self._camera_timeout = camera_timeout

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Registered components with last-seen timestamps
        self._components: Dict[str, WatchdogComponent] = {}

        # Callbacks
        self._on_restart: Optional[Callable[[str], None]] = None

    def register(self, name: str, timeout: float,
                 on_restart: Optional[Callable] = None) -> None:
        """Register a component to watch."""
        self._components[name] = WatchdogComponent(name, timeout, on_restart)

    def unregister(self, name: str) -> None:
        """Unregister a component."""
        self._components.pop(name, None)

    def ping(self, name: str) -> None:
        """Ping a component to update its last-seen time."""
        if name in self._components:
            self._components[name].last_seen = time.monotonic()

    def start(self) -> None:
        """Start watchdog thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="watchdog")
        self._thread.start()
        logger.info("Watchdog started (check=%ss, inference=%ss, camera=%ss)",
                     self._check_interval, self._inference_timeout, self._camera_timeout)

    def stop(self) -> None:
        """Stop watchdog thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        while not self._stop_event.is_set():
            now = time.monotonic()
            for name, comp in list(self._components.items()):
                elapsed = now - comp.last_seen
                if elapsed > comp.timeout:
                    logger.warning(
                        "Watchdog timeout: %s (%.1fs > %.1fs)",
                        name, elapsed, comp.timeout
                    )
                    if comp.on_restart:
                        try:
                            comp.on_restart()
                        except Exception as e:
                            logger.error("Watchdog restart failed for %s: %s", name, e)
                    # Reset timer after restart attempt
                    comp.last_seen = now

            self._stop_event.wait(self._check_interval)


class WatchdogComponent:
    """A watched component."""

    def __init__(self, name: str, timeout: float,
                 on_restart: Optional[Callable] = None):
        self.name = name
        self.timeout = timeout
        self.last_seen = time.monotonic()
        self.on_restart = on_restart
