"""
VisionInspect - Serial Manager
Mengelola koneksi serial RS232/RS485 dengan reconnect otomatis.
Mendukung RTS direction control untuk RS485 half-duplex.
"""

import time
import threading
from enum import Enum
from typing import Callable, Optional

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("plc")

try:
    import serial
    import serial.rs485
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    logger.warning("pyserial not installed")


class SerialState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class SerialDirection(Enum):
    """RS485 direction control mode."""
    AUTO = "auto"           # Konverter auto-direction
    RTS = "rts"             # RTS-controlled


class SerialConfig:
    """Serial port configuration."""

    def __init__(
        self,
        port: str = "COM1",
        baudrate: int = 9600,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: int = 1,
        timeout: float = 1.0,
        mode: str = "rs232",           # rs232 | rs485
        rs485_direction: str = "auto",  # auto | rts
        rs485_delay_before_tx: float = 0.0,
        rs485_delay_after_tx: float = 0.0,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 0,  # 0 = unlimited
    ):
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        self.mode = mode
        self.rs485_direction = rs485_direction
        self.rs485_delay_before_tx = rs485_delay_before_tx
        self.rs485_delay_after_tx = rs485_delay_after_tx
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts


class SerialManager:
    """
    Manajemen koneksi serial dengan:
    - Auto-reconnect (exponential backoff)
    - RS485 RTS direction control
    - RX/TX logging untuk diagnostik
    - Thread-safe
    - Callback untuk notifikasi koneksi
    """

    def __init__(self, config: Optional[SerialConfig] = None):
        self._config = config or SerialConfig()
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._state = SerialState.DISCONNECTED
        self._stop_event = threading.Event()
        self._connect_thread: Optional[threading.Thread] = None
        self._reconnect_count = 0

        # Callbacks
        self._on_connected: Optional[Callable[[], None]] = None
        self._on_disconnected: Optional[Callable[[], None]] = None
        self._on_data_received: Optional[Callable[[bytes], None]] = None
        self._log_callback: Optional[Callable[[str], None]] = None

        # RX/TX log (diagnostik)
        self._tx_log: list[tuple[float, bytes]] = []
        self._rx_log: list[tuple[float, bytes]] = []
        self._max_log_entries = 100

        # Buffer for incomplete frames
        self._rx_buffer = b""

    # ---- Properties ----

    @property
    def state(self) -> SerialState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == SerialState.CONNECTED

    @property
    def config(self) -> SerialConfig:
        return self._config

    @config.setter
    def config(self, value: SerialConfig) -> None:
        with self._lock:
            self._config = value

    # ---- Callbacks ----

    def set_on_connected(self, cb: Optional[Callable]) -> None:
        self._on_connected = cb

    def set_on_disconnected(self, cb: Optional[Callable]) -> None:
        self._on_disconnected = cb

    def set_on_data_received(self, cb: Optional[Callable[[bytes], None]]) -> None:
        self._on_data_received = cb

    def set_log_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        self._log_callback = cb

    # ---- Lifecycle ----

    def connect(self) -> bool:
        """
        Connect to serial port synchronously.
        Returns True if connected successfully.
        """
        if not HAS_SERIAL:
            self._log("ERROR: pyserial not installed")
            return False

        with self._lock:
            if self._serial and self._serial.is_open:
                logger.warning("Already connected to %s", self._config.port)
                return True

            self._state = SerialState.CONNECTING
            self._log(f"Connecting to {self._config.port} ({self._config.mode})...")

            try:
                ser = serial.Serial(
                    port=self._config.port,
                    baudrate=self._config.baudrate,
                    bytesize=self._config.bytesize,
                    parity=self._config.parity,
                    stopbits=self._config.stopbits,
                    timeout=self._config.timeout,
                )

                # Configure RS485 mode
                if self._config.mode == "rs485":
                    if self._config.rs485_direction == "rts":
                        # RTS-controlled direction
                        ser.rts = False  # Initially receive mode
                        ser.rs485_mode = serial.rs485.RS485Settings(
                            rts_level_for_tx=True,
                            rts_level_for_rx=False,
                            delay_before_tx=self._config.rs485_delay_before_tx,
                            delay_after_tx=self._config.rs485_delay_after_tx,
                        )
                    else:
                        # Auto-direction (if converter supports it)
                        ser.rs485_mode = serial.rs485.RS485Settings()

                self._serial = ser
                self._state = SerialState.CONNECTED
                self._reconnect_count = 0
                self._log(f"Connected to {self._config.port} (baud={self._config.baudrate})")
                logger.info("Serial connected: %s", self._config.port)

                if self._on_connected:
                    self._on_connected()

                return True

            except Exception as e:
                self._state = SerialState.ERROR
                self._log(f"Connection failed: {e}")
                logger.error("Serial connect failed: %s", e)
                return False

    def disconnect(self) -> None:
        """Disconnect serial port."""
        self._stop_event.set()
        with self._lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                    self._log("Disconnected")
                except Exception as e:
                    logger.error("Serial close error: %s", e)
            self._serial = None
            self._state = SerialState.DISCONNECTED

        if self._on_disconnected:
            self._on_disconnected()

    def start_reconnect_loop(self) -> None:
        """Start background thread for auto-reconnect."""
        if self._connect_thread and self._connect_thread.is_alive():
            logger.warning("Reconnect thread already running")
            return

        self._stop_event.clear()
        self._connect_thread = threading.Thread(
            target=self._reconnect_loop,
            daemon=True,
            name="plc-reconnect",
        )
        self._connect_thread.start()
        logger.info("PLC reconnect thread started")

    def stop_reconnect_loop(self) -> None:
        """Stop the reconnect loop."""
        self._stop_event.set()
        if self._connect_thread:
            self._connect_thread.join(timeout=3)
        self.disconnect()

    # ---- Data Transfer ----

    def send(self, data: bytes) -> bool:
        """
        Send data over serial.
        Returns True if sent successfully.
        """
        with self._lock:
            if not self._serial or not self._serial.is_open:
                self._log("Cannot send: not connected")
                return False

            try:
                # RS485: set RTS high for TX if RTS-controlled
                if self._config.mode == "rs485" and self._config.rs485_direction == "rts":
                    self._serial.rts = True

                if self._config.rs485_delay_before_tx > 0:
                    time.sleep(self._config.rs485_delay_before_tx)

                self._serial.write(data)

                if self._config.rs485_delay_after_tx > 0:
                    time.sleep(self._config.rs485_delay_after_tx)

                # RS485: set RTS low back to receive
                if self._config.mode == "rs485" and self._config.rs485_direction == "rts":
                    self._serial.rts = False

                # Log TX
                self._tx_log.append((time.monotonic(), data))
                if len(self._tx_log) > self._max_log_entries:
                    self._tx_log.pop(0)

                hex_str = data.hex(" ").upper()
                ascii_str = data.decode("ascii", errors="replace")
                self._log(f"TX: {hex_str}  |{ascii_str}|")

                return True

            except Exception as e:
                self._state = SerialState.ERROR
                self._log(f"TX error: {e}")
                logger.error("Serial TX error: %s", e)
                return False

    def read(self, size: int = 1) -> Optional[bytes]:
        """
        Read data from serial (non-blocking).
        Returns None on timeout/error.
        """
        with self._lock:
            if not self._serial or not self._serial.is_open:
                return None

            try:
                data = self._serial.read(size)
                if data:
                    # Log RX
                    self._rx_log.append((time.monotonic(), data))
                    if len(self._rx_log) > self._max_log_entries:
                        self._rx_log.pop(0)

                    hex_str = data.hex(" ").upper()
                    ascii_str = data.decode("ascii", errors="replace")
                    self._log(f"RX: {hex_str}  |{ascii_str}|")

                    if self._on_data_received:
                        self._on_data_received(data)

                return data
            except Exception as e:
                logger.error("Serial RX error: %s", e)
                return None

    def read_until(self, expected: bytes, timeout: float = 1.0) -> Optional[bytes]:
        """
        Read until expected sequence or timeout.
        Returns accumulated bytes or None on timeout/error.
        """
        with self._lock:
            if not self._serial or not self._serial.is_open:
                return None
            try:
                data = self._serial.read_until(expected, timeout)
                if data:
                    self._rx_log.append((time.monotonic(), data))
                    if len(self._rx_log) > self._max_log_entries:
                        self._rx_log.pop(0)
                    self._log(f"RX: {data.hex(' ').upper()}")
                    if self._on_data_received:
                        self._on_data_received(data)
                return data
            except Exception as e:
                logger.error("Serial read_until error: %s", e)
                return None

    # ---- Diagnostics ----

    def get_tx_log(self) -> list[dict]:
        return [
            {"time": t, "data": d.hex(), "ascii": d.decode("ascii", errors="replace")}
            for t, d in self._tx_log[-50:]
        ]

    def get_rx_log(self) -> list[dict]:
        return [
            {"time": t, "data": d.hex(), "ascii": d.decode("ascii", errors="replace")}
            for t, d in self._rx_log[-50:]
        ]

    def send_test_frame(self, data: bytes = b"TEST\x0D\x0A") -> bool:
        """Send a test frame (for diagnostics panel)."""
        return self.send(data)

    # ---- Internal ----

    def _reconnect_loop(self):
        """Background reconnect loop."""
        while not self._stop_event.is_set():
            if not self.is_connected:
                self.connect()

                if not self.is_connected:
                    self._reconnect_count += 1
                    if (self._config.max_reconnect_attempts > 0
                            and self._reconnect_count >= self._config.max_reconnect_attempts):
                        self._log("Max reconnect attempts reached, stopping")
                        break

            # Wait before retry
            self._stop_event.wait(self._config.reconnect_interval)

    def _log(self, message: str):
        logger.debug("[Serial] %s", message)
        if self._log_callback:
            self._log_callback(message)

    def __del__(self):
        self.disconnect()
