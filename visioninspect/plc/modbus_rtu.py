"""
VisionInspect - Modbus RTU Protocol
Register map untuk komunikasi PLC via Modbus RTU.
VisionInspect sebagai slave, PLC sebagai master.
"""

import struct
import time
from typing import Callable, Optional

from visioninspect.plc.serial_manager import SerialManager
from visioninspect.utils.logging_setup import get_logger

logger = get_logger("plc")

try:
    from pymodbus.client import ModbusSerialClient
    from pymodbus.exceptions import ModbusException
    HAS_MODBUS = True
except ImportError:
    HAS_MODBUS = False
    logger.warning("pymodbus not installed")


# Register Map (sesuai spesifikasi)
HOLDING_REGISTER_MAP = {
    "system_status": 0,      # 0=idle, 1=running, 2=training, 3=error
    "last_result": 1,        # 0=none, 1=OK, 2=NG
    "last_score": 2,         # score × 100 (integer)
    "total_counter": 3,      # 16-bit rolling
    "ng_counter": 4,         # 16-bit rolling
    "active_program": 10,    # program number to switch
}

COIL_MAP = {
    "trigger_inspection": 0,  # PLC set → read → reset
    "reset_counter": 1,       # PLC set → reset counters
}


class ModbusRTUManager:
    """
    Modbus RTU komunikasi.
    Register map:
    - Holding Register 0: system status
    - Holding Register 1: last result (0=none, 1=OK, 2=NG)
    - Holding Register 2: last score × 100
    - Holding Register 3: total counter (16-bit rolling)
    - Holding Register 4: NG counter
    - Coil 0: trigger inspection
    - Holding Register 10: active program number
    - Coil 1: reset counter
    """

    def __init__(self, serial_manager: SerialManager, slave_id: int = 1):
        self._serial = serial_manager
        self._slave_id = slave_id
        self._client: Optional[ModbusSerialClient] = None
        self._last_trigger = False
        self._last_reset = False
        self._on_trigger: Optional[Callable[[], None]] = None
        self._on_program_change: Optional[Callable[[int], None]] = None
        self._on_reset_counter: Optional[Callable[[], None]] = None
        self._log_callback: Optional[Callable[[str], None]] = None

        if HAS_MODBUS:
            self._init_client()

    def _init_client(self):
        """Initialize pymodbus client."""
        if not HAS_MODBUS:
            return
        try:
            self._client = ModbusSerialClient(
                port=self._serial.config.port,
                baudrate=self._serial.config.baudrate,
                bytesize=self._serial.config.bytesize,
                parity=self._serial.config.parity,
                stopbits=self._serial.config.stopbits,
                timeout=self._serial.config.timeout,
            )
        except Exception as e:
            logger.error("Modbus client init failed: %s", e)

    # ---- Callbacks ----

    def set_on_trigger(self, cb: Optional[Callable]) -> None:
        self._on_trigger = cb

    def set_on_program_change(self, cb: Optional[Callable[[int], None]]) -> None:
        self._on_program_change = cb

    def set_on_reset_counter(self, cb: Optional[Callable]) -> None:
        self._on_reset_counter = cb

    def set_log_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        self._log_callback = cb

    # ---- Status Update (VisionInspect → PLC) ----

    def update_status(self, status: int) -> bool:
        """Update system status register."""
        return self._write_holding_register(
            HOLDING_REGISTER_MAP["system_status"], status
        )

    def update_result(self, result: int, score: float) -> bool:
        """Update result and score registers."""
        ok = self._write_holding_register(
            HOLDING_REGISTER_MAP["last_result"], result
        )
        score_int = max(0, min(65535, int(score * 100)))
        ok &= self._write_holding_register(
            HOLDING_REGISTER_MAP["last_score"], score_int
        )
        return ok

    def update_counters(self, total: int, ng: int) -> bool:
        """Update counter registers (16-bit rolling)."""
        ok = self._write_holding_register(
            HOLDING_REGISTER_MAP["total_counter"], total % 65536
        )
        ok &= self._write_holding_register(
            HOLDING_REGISTER_MAP["ng_counter"], ng % 65536
        )
        return ok

    # ---- PLC Commands Check (PLC → VisionInspect) ----

    def poll_commands(self) -> dict:
        """
        Poll PLC for pending commands.
        Returns dict with any commands found.
        """
        if not self._client or not HAS_MODBUS:
            return {}

        commands = {}

        try:
            # Check trigger coil
            trigger = self._read_coil(COIL_MAP["trigger_inspection"])
            if trigger and not self._last_trigger:
                commands["trigger"] = True
                # Reset coil
                self._write_coil(COIL_MAP["trigger_inspection"], False)
            self._last_trigger = bool(trigger)

            # Check reset coil
            reset = self._read_coil(COIL_MAP["reset_counter"])
            if reset and not self._last_reset:
                commands["reset_counter"] = True
                self._write_coil(COIL_MAP["reset_counter"], False)
            self._last_reset = bool(reset)

            # Check program change
            prog = self._read_holding_register(HOLDING_REGISTER_MAP["active_program"])
            if prog is not None and prog > 0:
                commands["program_change"] = prog
                # Reset after reading
                self._write_holding_register(HOLDING_REGISTER_MAP["active_program"], 0)

            if commands:
                self._log(f"PLC commands: {commands}")

        except Exception as e:
            logger.warning("Modbus poll error: %s", e)

        return commands

    # ---- Low-level Modbus Operations ----

    def _write_holding_register(self, address: int, value: int) -> bool:
        if not self._client or not HAS_MODBUS:
            return False
        try:
            result = self._client.write_register(address, value, slave=self._slave_id)
            if result.isError():
                logger.warning("Modbus write error at %d = %d", address, value)
                return False
            return True
        except Exception as e:
            logger.error("Modbus write exception: %s", e)
            return False

    def _read_holding_register(self, address: int) -> Optional[int]:
        if not self._client or not HAS_MODBUS:
            return None
        try:
            result = self._client.read_holding_registers(address, 1, slave=self._slave_id)
            if result.isError():
                return None
            return result.registers[0]
        except Exception:
            return None

    def _write_coil(self, address: int, value: bool) -> bool:
        if not self._client or not HAS_MODBUS:
            return False
        try:
            result = self._client.write_coil(address, value, slave=self._slave_id)
            if result.isError():
                return False
            return True
        except Exception:
            return False

    def _read_coil(self, address: int) -> Optional[bool]:
        if not self._client or not HAS_MODBUS:
            return None
        try:
            result = self._client.read_coils(address, 1, slave=self._slave_id)
            if result.isError():
                return None
            return result.bits[0]
        except Exception:
            return None

    def _log(self, message: str):
        if self._log_callback:
            self._log_callback(message)
