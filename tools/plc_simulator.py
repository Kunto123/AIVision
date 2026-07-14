#!/usr/bin/env python3
"""
PLC Simulator — VisionInspect
Simulasi PLC untuk menguji komunikasi serial (Modbus RTU atau ASCII).
Menggunakan virtual serial pair (socat atau com0com).

Cara pakai:
    1. Install socat (Linux/WSL): sudo apt-get install socat
    2. Buat virtual serial pair:
        socat -d -d PTY,link=/tmp/ttyV0 PTY,link=/tmp/ttyV1
       Ini akan membuat /tmp/ttyV0 dan /tmp/ttyV1 (serial pair)
    3. Jalankan simulator: python tools/plc_simulator.py --port /tmp/ttyV0 --protocol modbus
    4. Konfigurasi VisionInspect ke port /tmp/ttyV1 dengan protokol yang sama
"""

import argparse
import json
import struct
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioninspect.plc.serial_manager import SerialManager, SerialConfig
from visioninspect.plc.ascii_protocol import ASCIIProtocolManager


class PLCSimulator:
    """
    Simulator PLC untuk testing.
    - Mode ASCII: mengirim trigger periodik, menerima hasil
    - Mode Modbus: membaca register, menulis trigger coil
    """

    def __init__(self, port: str, protocol: str = "ascii", slave_id: int = 1):
        self._port = port
        self._protocol = protocol
        self._slave_id = slave_id

        # Serial config
        config = SerialConfig(
            port=port,
            baudrate=9600,
            timeout=1.0,
        )
        self._serial = SerialManager(config)

        # Protocol
        self._ascii = ASCIIProtocolManager(self._serial)

        # Statistics
        self._inspections_requested = 0
        self._results_received = 0
        self._last_result = None

        # Callbacks
        self._ascii.set_on_trigger(self._on_trigger)

    def run(self):
        """Run simulator main loop."""
        print(f"\n{'='*60}")
        print(f"  PLC SIMULATOR")
        print(f"  Port: {self._port}")
        print(f"  Protocol: {self._protocol.upper()}")
        print(f"{'='*60}\n")

        # Connect
        if not self._serial.connect():
            print("ERROR: Gagal connect ke serial port")
            return

        print(f"  Terhubung ke {self._port}")
        print("  Tekan Ctrl+C untuk berhenti\n")

        try:
            if self._protocol == "ascii":
                self._run_ascii()
            elif self._protocol == "modbus":
                self._run_modbus()
            else:
                print(f"ERROR: Protokol tidak dikenal: {self._protocol}")
        except KeyboardInterrupt:
            print("\n  Simulator dihentikan oleh user")
        finally:
            self._serial.disconnect()
            self._print_stats()

    def _run_ascii(self):
        """ASCII protocol mode - polling dan kirim trigger."""
        trigger_count = 0
        last_trigger_time = 0

        while True:
            # Kirim trigger setiap 3 detik
            now = time.monotonic()
            if now - last_trigger_time >= 3.0:
                trigger_count += 1
                self._inspections_requested += 1
                frame = ASCIIProtocolManager.make_frame("TRG")
                print(f"\n[Trigger #{trigger_count}] Kirim TRG...")
                self._serial.send(frame)
                last_trigger_time = now

            # Baca response
            data = self._serial.read(64)
            if data:
                parsed = self._ascii.process_received_data(data)
                if parsed:
                    print(f"  Response: {parsed}")
                    if parsed.get("command") == "ack":
                        self._results_received += 1
                        self._print_status()

            time.sleep(0.1)

    def _run_modbus(self):
        """Modbus RTU mode - baca register periodik."""
        try:
            from pymodbus.client import ModbusSerialClient
            from pymodbus.exceptions import ModbusException

            client = ModbusSerialClient(
                port=self._port,
                baudrate=9600,
                timeout=1.0,
            )
            client.connect()
            print("  Modbus client connected")

            while True:
                # Baca holding register
                try:
                    # Register 0: system status
                    rr = client.read_holding_registers(0, 6, slave=self._slave_id)
                    if not rr.isError():
                        status = rr.registers[0]
                        result = rr.registers[1]
                        score = rr.registers[2] / 100.0
                        total = rr.registers[3]
                        ng = rr.registers[4]
                        prog = rr.registers[5]

                        print(f"  Status={status} | Result={result} | Score={score:.2f} | "
                              f"Total={total} | NG={ng} | Program={prog}")

                except ModbusException as e:
                    print(f"  Read error: {e}")

                time.sleep(2.0)

        except ImportError:
            print("ERROR: pymodbus tidak terinstall")
            print("Install: pip install pymodbus")

    def _on_trigger(self):
        """Callback saat trigger diterima dari VisionInspect (ACK)."""
        self._inspections_requested += 1
        self._print_status()

    def _print_status(self):
        print(f"  >> Stats: {self._inspections_requested} triggered, "
              f"{self._results_received} results")

    def _print_stats(self):
        print(f"\n{'='*60}")
        print(f"  STATISTIK SIMULATOR")
        print(f"  Trigger dikirim: {self._inspections_requested}")
        print(f"  Hasil diterima: {self._results_received}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="VisionInspect PLC Simulator")
    parser.add_argument("--port", default="COM3",
                        help="Serial port (default: COM3 for Windows). "
                             "Linux: /tmp/ttyV0 for virtual pair")
    parser.add_argument("--protocol", choices=["ascii", "modbus"], default="ascii",
                        help="Protocol (default: ascii)")
    parser.add_argument("--slave-id", type=int, default=1,
                        help="Modbus slave ID (default: 1)")
    args = parser.parse_args()

    sim = PLCSimulator(
        port=args.port,
        protocol=args.protocol,
        slave_id=args.slave_id,
    )
    sim.run()


if __name__ == "__main__":
    main()
