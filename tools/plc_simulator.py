#!/usr/bin/env python3
"""
PLC Simulator — VisionInspect
Simulasi PLC (Modbus RTU SLAVE) untuk menguji komunikasi.
Sekarang sistem = MASTER, simulator = SLAVE (sesuai arsitektur baru).

Cara pakai:
    1. Install socat (Linux/WSL): sudo apt-get install socat
    2. Buat virtual serial pair:
        socat -d -d PTY,link=/tmp/ttyV0 PTY,link=/tmp/ttyV1
       → /tmp/ttyV0 dan /tmp/ttyV1 (dua ujung kabel virtual)
    3. Jalankan simulator (slave):  python tools/plc_simulator.py --port /tmp/ttyV0 --protocol modbus
    4. Konfigurasi VisionInspect ke /tmp/ttyV1, protocol modbus, enable PLC.
       (Settings → PLC → port /tmp/ttyV1 → Save)

Simulator meniru PLC FX3U:
    - Coil output yang DITULIS sistem:  1=OK, 2=NG, 3=part_ready, 4=busy
    - Coil input yang DIBACA sistem:    0=trigger, 5=reset, 6=switch_program
    - Holding register: 10 = nomor program (untuk switch_program)

Keyboard simulator:
    t  → trigger inspeksi (coil 0 ON sesaat)
    r  → reset (coil 5 ON sesaat)
    s  → switch program (coil 6 ON sesaat — gunakan p dulu untuk set nomor)
    p  → set nomor program (default 2)
    q  → keluar

⚠️ pymodbus 3.13: API lama (ModbusDeviceContext) rusak — pakai SimDevice/SimData.
"""

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Peta alamat — ⚠️ samakan dengan io_map di config VisionInspect
OUT_COIL_OK = 1
OUT_COIL_NG = 2
OUT_COIL_PART_READY = 3
OUT_COIL_BUSY = 4
IN_COIL_TRIGGER = 0
IN_COIL_RESET = 5
IN_COIL_SWITCH_PROGRAM = 6
PROGRAM_REGISTER = 10


class PLCSimulator:
    """Modbus RTU slave simulator (menggantikan PLC Mitsubishi FX3U)."""

    def __init__(self, port: str, slave_id: int = 1):
        self._port = port
        self._slave_id = slave_id
        self._server = None
        self._running = False
        self._print_lock = threading.Lock()

    # ---- Akses store lewat server.context (SimCore milik server) ----

    def _set_coil(self, addr: int, value: bool):
        try:
            asyncio.run(self._server.context.async_setValues(
                self._slave_id, 1, addr, [1 if value else 0]))
        except Exception:
            pass

    def _get_coil(self, addr: int) -> int:
        try:
            v = asyncio.run(self._server.context.async_getValues(
                self._slave_id, 1, addr, 1))
            return 1 if (isinstance(v, list) and v and v[0]) else 0
        except Exception:
            return 0

    def _get_register(self, addr: int) -> int:
        try:
            v = asyncio.run(self._server.context.async_getValues(
                self._slave_id, 3, addr, 1))
            return v[0] if isinstance(v, list) and v else 0
        except Exception:
            return 0

    def _set_register(self, addr: int, value: int):
        try:
            asyncio.run(self._server.context.async_setValues(
                self._slave_id, 3, addr, [value]))
        except Exception:
            pass

    # ---- Input simulasi (dari keyboard) ----

    def pulse_coil(self, addr: int, ms: float = 0.4):
        self._set_coil(addr, True)

        def _off():
            time.sleep(ms)
            self._set_coil(addr, False)

        threading.Thread(target=_off, daemon=True).start()

    # ---- Display ----

    def _display_loop(self):
        last_out = None
        prog_set = False
        while self._running:
            if not prog_set:
                # Set program awal (register 10 = 1) dengan retry sampai server siap
                self._set_register(PROGRAM_REGISTER, 1)
                if self._get_register(PROGRAM_REGISTER) == 1:
                    prog_set = True
            ok = self._get_coil(OUT_COIL_OK)
            ng = self._get_coil(OUT_COIL_NG)
            pr = self._get_coil(OUT_COIL_PART_READY)
            busy = self._get_coil(OUT_COIL_BUSY)
            prog = self._get_register(PROGRAM_REGISTER)
            out = (ok, ng, pr, busy, prog)
            if out != last_out:
                with self._print_lock:
                    print(f"  [PLC] OK={ok} NG={ng} PartReady={pr} Busy={busy} "
                          f"Program={prog}")
                last_out = out
            time.sleep(0.3)

    def _keyboard_loop(self):
        with self._print_lock:
            print("  Keyboard: t=trigger  r=reset  s=switch_program  "
                  "p=set_program  q=quit")
        while self._running:
            try:
                cmd = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd == "q":
                self._running = False
                break
            elif cmd == "t":
                self.pulse_coil(IN_COIL_TRIGGER)
                with self._print_lock:
                    print("  → trigger ON (coil 0) sesaat")
            elif cmd == "r":
                self.pulse_coil(IN_COIL_RESET)
                with self._print_lock:
                    print("  → reset ON (coil 5) sesaat")
            elif cmd == "s":
                self.pulse_coil(IN_COIL_SWITCH_PROGRAM)
                with self._print_lock:
                    print("  → switch_program ON (coil 6) sesaat")
            elif cmd == "p":
                self._set_register(PROGRAM_REGISTER, 2)
                with self._print_lock:
                    print("  → program register = 2")

    # ---- Main ----

    def run(self):
        from pymodbus.framer import FramerType
        from pymodbus.server import ModbusSerialServer
        from pymodbus.simulator import DataType, SimData, SimDevice

        print(f"\n{'='*60}")
        print("  PLC SIMULATOR (SLAVE — Modbus RTU)")
        print(f"  Port: {self._port}  Slave ID: {self._slave_id}")
        print(f"  Output coil: OK={OUT_COIL_OK} NG={OUT_COIL_NG} "
              f"PartReady={OUT_COIL_PART_READY} Busy={OUT_COIL_BUSY}")
        print(f"  Input coil:  Trigger={IN_COIL_TRIGGER} Reset={IN_COIL_RESET} "
              f"SwitchProg={IN_COIL_SWITCH_PROGRAM}")
        print(f"  Register:    Program={PROGRAM_REGISTER}")
        print("="*60)

        coil_sim = SimData(address=0, count=128, values=[0] * 128,
                           datatype=DataType.BITS)
        di_sim = SimData(address=0, count=128, values=[0] * 128,
                         datatype=DataType.BITS)
        hr_sim = SimData(address=0, count=64, values=[0] * 64,
                         datatype=DataType.REGISTERS)
        ir_sim = SimData(address=0, count=64, values=[0] * 64,
                         datatype=DataType.REGISTERS)
        device = SimDevice(id=self._slave_id,
                           simdata=([coil_sim], [di_sim], [hr_sim], [ir_sim]))

        # Program awal = 1 (register 10) — di-set setelah server siap

        # Server jalan di thread (event loop sendiri), main thread untuk
        # keyboard + display
        def _serve():
            async def _start():
                srv = ModbusSerialServer(
                    [device], framer=FramerType.RTU,
                    device_address=self._slave_id,
                    port=self._port,
                    baudrate=9600, bytesize=8, parity="N",
                    stopbits=1, timeout=1.0,
                )
                self._server = srv
                await srv.serve_forever()
            asyncio.run(_start())

        server_thread = threading.Thread(target=_serve, daemon=True)
        server_thread.start()
        time.sleep(0.5)

        self._running = True
        display_thread = threading.Thread(target=self._display_loop, daemon=True)
        display_thread.start()

        try:
            self._keyboard_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            print("\n  Simulator dihentikan")
            try:
                asyncio.run(self._server.shutdown())
            except Exception:
                pass
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="VisionInspect PLC Simulator (SLAVE)")
    parser.add_argument("--port", default="COM3",
                        help="Serial port (Windows: COM3, Linux: /tmp/ttyV0)")
    parser.add_argument("--protocol", choices=["ascii", "modbus"], default="modbus",
                        help="Protocol (default: modbus; ascii belum di-support slave mode)")
    parser.add_argument("--slave-id", type=int, default=1,
                        help="Modbus slave ID (default: 1)")
    args = parser.parse_args()

    if args.protocol != "modbus":
        print("⚠️  Mode ASCII di simulator lama (master) sudah diganti: "
              "sistem sekarang MASTER, jadi simulator harus SLAVE (modbus).")
        print("    Pakai --protocol modbus.")
        sys.exit(1)

    sim = PLCSimulator(port=args.port, slave_id=args.slave_id)
    sim.run()


if __name__ == "__main__":
    main()
