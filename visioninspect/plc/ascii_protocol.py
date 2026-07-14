"""
VisionInspect - ASCII Protocol Sederhana
Protokol komunikasi serial dengan format frame:
    STX <CMD> [DATA] ETX <CHECKSUM>

Checksum: XOR semua byte antara STX dan ETX (inklusif).
Frame contoh:
    - Trigger:   0x02 TRG 0x03 <chk>
    - Result OK: 0x02 RES,OK,<score> 0x03 <chk>
    - Result NG: 0x02 RES,NG,<score> 0x03 <chk>
    - Program:   0x02 PRG,<n> 0x03 <chk>
    - Status:    0x02 STA 0x03 <chk>
    - Response:  0x02 ACK 0x03 <chk> or 0x02 NAK,<reason> 0x03 <chk>
"""

import time
from typing import Callable, Optional

from visioninspect.plc.serial_manager import SerialManager
from visioninspect.utils.logging_setup import get_logger

logger = get_logger("plc")

# ASCII Protocol constants
STX = b"\x02"
ETX = b"\x03"
ACK = b"ACK"
NAK = b"NAK"


class ASCIIProtocolError(Exception):
    pass


class ASCIIProtocolManager:
    """
    Protokol ASCII sederhana untuk PLC lama yang tidak mendukung Modbus.
    Format frame: STX <CMD> [DATA] ETX <checksum>
    Checksum: XOR semua byte STX..ETX inklusif.
    """

    def __init__(self, serial_manager: SerialManager):
        self._serial = serial_manager
        self._on_trigger: Optional[Callable[[], None]] = None
        self._on_program_change: Optional[Callable[[int], None]] = None
        self._on_status_request: Optional[Callable[[], dict]] = None
        self._log_callback: Optional[Callable[[str], None]] = None

    # ---- Callbacks ----

    def set_on_trigger(self, cb: Optional[Callable]) -> None:
        self._on_trigger = cb

    def set_on_program_change(self, cb: Optional[Callable[[int], None]]) -> None:
        self._on_program_change = cb

    def set_on_status_request(self, cb: Optional[Callable[[], dict]]) -> None:
        self._on_status_request = cb

    def set_log_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        self._log_callback = cb

    # ---- Frame Construction ----

    @staticmethod
    def make_frame(command: str, data: str = "") -> bytes:
        """Build an ASCII protocol frame with checksum."""
        payload = f"{command},{data}" if data else command
        frame = STX + payload.encode("ascii") + ETX
        checksum = ASCIIProtocolManager._calc_checksum(frame)
        return frame + bytes([checksum])

    @staticmethod
    def _calc_checksum(data: bytes) -> int:
        """XOR checksum of all bytes."""
        checksum = 0
        for b in data:
            checksum ^= b
        return checksum

    @staticmethod
    def parse_frame(frame: bytes) -> Optional[dict]:
        """
        Parse an ASCII protocol frame.
        Returns dict with command, data, or None if invalid.
        """
        frame = frame.strip()

        # Check minimum length
        if len(frame) < 5:
            return None

        # Find STX and ETX
        stx_idx = frame.find(STX)
        etx_idx = frame.find(ETX, stx_idx + 1)

        if stx_idx < 0 or etx_idx < 0:
            return None

        # Extract payload and checksum
        payload = frame[stx_idx + 1:etx_idx]
        full_frame = frame[stx_idx:etx_idx + 1]

        # Expected checksum
        if len(frame) > etx_idx + 1:
            received_cs = frame[etx_idx + 1]
            calculated_cs = ASCIIProtocolManager._calc_checksum(full_frame)
            if received_cs != calculated_cs:
                return {"error": "checksum_mismatch", "raw": frame.hex()}

        # Parse command and data
        payload_str = payload.decode("ascii", errors="replace").strip()
        if "," in payload_str:
            command, data_str = payload_str.split(",", 1)
        else:
            command = payload_str
            data_str = ""

        return {
            "command": command.strip(),
            "data": data_str.strip(),
            "raw": frame.hex(),
        }

    # ---- Sending Commands (VisionInspect → PLC) ----

    def send_result(self, judgement: str, score: float) -> bool:
        """Send inspection result to PLC."""
        frame = self.make_frame("RES", f"{judgement},{score:.4f}")
        return self._serial.send(frame)

    def send_status(self, status_code: int, status_text: str = "") -> bool:
        """Send status to PLC."""
        data = f"{status_code},{status_text}" if status_text else str(status_code)
        frame = self.make_frame("STA", data)
        return self._serial.send(frame)

    def send_ack(self) -> bool:
        """Send ACK."""
        frame = self.make_frame("ACK")
        return self._serial.send(frame)

    def send_nak(self, reason: str = "ERROR") -> bool:
        """Send NAK."""
        frame = self.make_frame("NAK", reason)
        return self._serial.send(frame)

    # ---- Receiving & Processing ----

    def process_received_data(self, data: bytes) -> Optional[dict]:
        """
        Process received data (from serial callback).
        Returns command dict if a complete frame was parsed.
        """
        parsed = self.parse_frame(data)
        if parsed is None:
            return None

        if "error" in parsed:
            self._log(f"Frame error: {parsed['error']}")
            return parsed

        cmd = parsed["command"]
        data_val = parsed["data"]

        # Handle commands
        if cmd == "TRG":
            self._log("PLC trigger received")
            if self._on_trigger:
                self._on_trigger()
            self.send_ack()
            return {"command": "trigger"}

        elif cmd == "PRG":
            try:
                prog_num = int(data_val)
                self._log(f"Program change request: {prog_num}")
                if self._on_program_change:
                    self._on_program_change(prog_num)
                self.send_ack()
                return {"command": "program_change", "program": prog_num}
            except ValueError:
                self.send_nak("INVALID_PROG")
                return {"command": "error", "message": f"Invalid program: {data_val}"}

        elif cmd == "STA":
            self._log("Status request received")
            if self._on_status_request:
                status_info = self._on_status_request()
                self.send_status(
                    status_info.get("code", 0),
                    status_info.get("text", ""),
                )
            else:
                self.send_status(0, "OK")
            return {"command": "status_request"}

        elif cmd == "ACK":
            self._log("ACK received")
            return {"command": "ack"}

        elif cmd == "NAK":
            self._log(f"NAK received: {data_val}")
            return {"command": "nak", "reason": data_val}

        else:
            self._log(f"Unknown command: {cmd}")
            self.send_nak("UNKNOWN_CMD")
            return {"command": "unknown", "raw": cmd}

    # ---- Helper ----

    def _log(self, message: str):
        if self._log_callback:
            self._log_callback(message)
