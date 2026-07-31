"""
VisionInspect - Modbus RTU Protocol (Sistem = MASTER, PLC = SLAVE)
=================================================================
Komunikasi serial Modbus RTU: VisionInspect bertindak sebagai MASTER,
PLC sebagai slave. Sistem aktif menulis coil hasil (OK/NG/part_ready)
dan membaca coil input (trigger/reset/switch program) dari PLC.

Register/Coil map diambil dari config.json → "plc" → "io_map",
JADI SEMUA ALAMAT BISA DIGANTI TANPA EDIT KODE.
Cukup ubah angka di config (menu Settings → tab PLC → IO Mapping).

⚠️ GANTI DI SINI kalau PLC berubah (lihat PLC_PROFILES di bawah):
  - Mitsubishi FX3U + 485ADP-MB : M0-M7679 → coil 0-7679, D0-D7999 → register 0-7999
  - Omron CP1W-CIF11           : 0.00-15.15 → coil 0-255 (bergantung setting)
  - Siemens S7-1200/1500        : butuh gateway Modbus (CM1241 / ET200SP), alamat beda
  → Sesuaikan PLC_PROFILES + io_map di config.
"""

import time
from typing import Callable, Optional

from visioninspect.utils.logging_setup import get_logger

logger = get_logger("plc")

try:
    from pymodbus.client import ModbusSerialClient
    HAS_MODBUS = True
except ImportError:
    HAS_MODBUS = False
    logger.warning("pymodbus not installed")

# ---------------------------------------------------------------------------
# ⚠️ GANTI DI SINI — profil PLC. Pilih profil saat init, atau tambah profil baru.
# ---------------------------------------------------------------------------
PLC_PROFILES = {
    "mitsubishi_fx3u": {
        "note": "FX3U + 485ADP-MB: M0-M7679 → coil 0-7679, D0-D7999 → register 0-7999",
        "coil_base": 0,
        "register_base": 0,
        "max_coil": 7679,
        "max_register": 7999,
    },
    "generic": {
        "note": "Profil generik — alamat persis mengikuti io_map di config",
        "coil_base": 0,
        "register_base": 0,
        "max_coil": 9999,
        "max_register": 9999,
    },
}


def build_io_map(plc_config: Optional[dict]) -> dict:
    """IO map default (coil/register) — override dari config.

    outputs  : coil yang SISTEM tulis → PLC baca
               result_ok / result_ng / part_ready / busy
    inputs   : coil yang PLC tulis → sistem baca
               trigger (minta inspeksi) / reset_result / switch_program
    program_register : holding register berisi nomor program untuk switch_program
    """
    default = {
        "outputs": {
            "result_ok": 1,       # coil 1: ON = part OK (pulse)
            "result_ng": 2,       # coil 2: ON = part NG (pulse)
            "part_ready": 3,      # coil 3: ON = part terdeteksi (pulse saat transisi)
            "busy": 4,            # coil 4: ON = sistem sedang inspeksi
        },
        "inputs": {
            "trigger": 0,         # coil 0: PLC minta 1 siklus inspeksi
            "reset_result": 5,    # IN1: reset kondisi coil hasil + counter
            "switch_program": 6,  # IN2: ganti template aktif (baca program_register)
        },
        "program_register": 10,   # holding register nomor program tujuan
    }
    io = plc_config.get("io_map") if plc_config else None
    if isinstance(io, dict):
        # Deep-merge: isi key yang ada, sisanya default
        for section in ("outputs", "inputs"):
            if isinstance(io.get(section), dict):
                default[section].update(io[section])
        if isinstance(io.get("program_register"), int):
            default["program_register"] = io["program_register"]
    return default


class ModbusRTUManager:
    """
    Modbus RTU komunikasi — VisionInspect = MASTER, PLC = slave.

    Alamat coil/register sepenuhnya dari io_map (config), bukan hardcode.
    """

    def __init__(self, plc_config: Optional[dict] = None):
        plc_config = plc_config or {}
        self._config = plc_config
        self._device_id = int(plc_config.get("modbus_slave_id", 1))
        self._pulse_ms = int(plc_config.get("pulse_ms", 300))
        self._io_map = build_io_map(plc_config)

        self._client: Optional[ModbusSerialClient] = None
        self._connected = False

        # Edge-detection untuk input PLC (agar tidak retrigger)
        self._last_inputs: dict[str, bool] = {}

        # Callback hasil poll input (dipanggil dari thread poll)
        self._on_trigger: Optional[Callable[[], None]] = None
        self._on_reset: Optional[Callable[[], None]] = None
        self._on_switch_program: Optional[Callable[[int], None]] = None
        self._on_status_change: Optional[Callable[[bool], None]] = None
        self._log_callback: Optional[Callable[[str], None]] = None

        if HAS_MODBUS:
            self._init_client()

    # ---- Init ----

    def _init_client(self):
        try:
            self._client = ModbusSerialClient(
                port=self._config.get("port", "COM1"),
                framer="rtu",
                baudrate=int(self._config.get("baudrate", 9600)),
                bytesize=int(self._config.get("bytesize", 8)),
                parity=str(self._config.get("parity", "N")),
                stopbits=int(self._config.get("stopbits", 1)),
                timeout=float(self._config.get("timeout", 1.0)),
                retries=2,
            )
        except Exception as e:
            logger.error("Modbus client init failed: %s", e)
            self._client = None

    # ---- Callbacks ----

    def set_on_trigger(self, cb: Optional[Callable]) -> None:
        self._on_trigger = cb

    def set_on_reset(self, cb: Optional[Callable]) -> None:
        self._on_reset = cb

    def set_on_switch_program(self, cb: Optional[Callable[[int], None]]) -> None:
        self._on_switch_program = cb

    def set_on_status_change(self, cb: Optional[Callable[[bool], None]]) -> None:
        self._on_status_change = cb

    def set_log_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        self._log_callback = cb

    # ---- Lifecycle ----

    def connect(self) -> bool:
        """Connect ke PLC (slave) sebagai master Modbus RTU."""
        if not HAS_MODBUS or self._client is None:
            self._log("ERROR: pymodbus tidak tersedia")
            return False
        if self._connected:
            return True
        try:
            ok = self._client.connect()
            self._connected = bool(ok)
            if ok:
                self._log(f"Modbus RTU connected ({self._config.get('port')}, device={self._device_id})")
            else:
                self._log("Modbus connect gagal (port tidak merespons)")
            if self._on_status_change:
                self._on_status_change(self._connected)
            return self._connected
        except Exception as e:
            logger.error("Modbus connect exception: %s", e)
            self._connected = False
            if self._on_status_change:
                self._on_status_change(False)
            return False

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning("Modbus close error: %s", e)
        self._connected = False
        if self._on_status_change:
            self._on_status_change(False)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def io_map(self) -> dict:
        return self._io_map

    # ---- OUTPUT (sistem → PLC) ----
    # Coil yang sistem tulis; PLC (slave) membaca coil ini di ladder-nya.
    # Contoh ladder Mitsubishi: M1 → Y0 (reject) = coil result_ng.

    def set_output(self, name: str, value: bool) -> bool:
        """Set coil output by nama (result_ok/result_ng/part_ready/busy)."""
        addr = self._io_map["outputs"].get(name)
        if addr is None:
            return False
        return self._write_coil(addr, value)

    def pulse_output(self, name: str, pulse_ms: Optional[int] = None) -> bool:
        """Pulse coil output: ON → tunggu → OFF (blocking).

        Untuk dipanggil dari thread worker, bukan UI thread.
        UI thread → pakai set_output + QTimer di main_window.
        """
        ms = pulse_ms if pulse_ms is not None else self._pulse_ms
        ok = self.set_output(name, True)
        if ok and ms > 0:
            time.sleep(ms / 1000.0)
            self.set_output(name, False)
        return ok

    def reset_outputs(self) -> None:
        """Matikan semua coil output (dipakai saat input reset aktif)."""
        for name in self._io_map["outputs"]:
            self._write_coil(self._io_map["outputs"][name], False)

    # ---- INPUT (PLC → sistem) ----

    def read_inputs(self) -> dict:
        """Baca semua coil input + edge detection.

        Returns dict dengan nama input yang BERUBAH False→True:
        {"trigger": True, "reset_result": False, "switch_program": False}
        """
        events: dict[str, bool] = {}
        for name, addr in self._io_map["inputs"].items():
            val = self._read_coil(addr)
            if val is None:
                continue  # baca gagal → skip
            prev = self._last_inputs.get(name, False)
            self._last_inputs[name] = val
            if val and not prev:
                events[name] = True
        return events

    def read_program_register(self) -> Optional[int]:
        """Baca holding register nomor program tujuan (untuk switch_program)."""
        addr = self._io_map.get("program_register", 10)
        return self._read_holding_register(addr)

    # ---- Scan coil (untuk dropdown di Settings) ----

    def scan_coils(self, max_addr: int = 127) -> list[int]:
        """Probe alamat coil 0..max_addr, kembalikan daftar yang VALID.

        Alamat valid = PLC membalas (tanpa exception). Alamat kosong/invalid
        → PLC balas exception code 2 (Illegal Data Address) → di-skip.
        Catatan: ini probing baca-only, TIDAK mengubah state coil.
        """
        if not self._connected or self._client is None:
            return []
        valid: list[int] = []
        # Timeout kecil agar scan cepat (128 probe × 0.15s ≈ 20s worst case)
        orig_timeout = getattr(self._client.comm_params, "timeout_connect", 1.0)
        try:
            self._client.comm_params.timeout_connect = 0.15
            for addr in range(max_addr + 1):
                try:
                    res = self._client.read_coils(addr, count=1, device_id=self._device_id)
                    if not res.isError():
                        valid.append(addr)
                except Exception:
                    continue
        finally:
            try:
                self._client.comm_params.timeout_connect = orig_timeout
            except Exception:
                pass
        return valid

    def find_active_coils(self, max_addr: int = 127) -> list[int]:
        """Cari coil yang sedang ON saat ini (input PLC aktif / tombol fisik).

        Dipakai tombol "Deteksi Aktif" di Settings — tekan tombol fisik di
        PLC, hasilnya coil yang nyala akan muncul di sini.
        """
        if not self._connected or self._client is None:
            return []
        active: list[int] = []
        orig_timeout = getattr(self._client.comm_params, "timeout_connect", 1.0)
        try:
            self._client.comm_params.timeout_connect = 0.15
            for addr in range(max_addr + 1):
                try:
                    res = self._client.read_coils(addr, count=1, device_id=self._device_id)
                    if not res.isError() and res.bits and res.bits[0]:
                        active.append(addr)
                except Exception:
                    continue
        finally:
            try:
                self._client.comm_params.timeout_connect = orig_timeout
            except Exception:
                pass
        return active

    # ---- Status registers (backup, untuk PLC yang polling register) ----

    def update_status(self, status: int) -> bool:
        """Update system status register 0 (0=idle,1=running,2=training,3=error)."""
        return self._write_holding_register(0, status)

    def update_result(self, result: int, score: float) -> bool:
        """Update result register 1 (0=none,1=OK,2=NG) + score register 2 (×100)."""
        ok = self._write_holding_register(1, result)
        score_int = max(0, min(65535, int(score * 100)))
        ok &= self._write_holding_register(2, score_int)
        return ok

    def update_counters(self, total: int, ng: int) -> bool:
        """Update counter register 3 (total) dan 4 (NG), 16-bit rolling."""
        ok = self._write_holding_register(3, total % 65536)
        ok &= self._write_holding_register(4, ng % 65536)
        return ok

    # ---- Low-level Modbus (device_id = pymodbus 3.13) ----

    def _write_coil(self, address: int, value: bool) -> bool:
        if not self._connected or self._client is None:
            return False
        try:
            res = self._client.write_coil(address, value, device_id=self._device_id)
            if res.isError():
                logger.warning("Modbus write coil error @%d=%s", address, value)
                return False
            return True
        except Exception as e:
            logger.error("Modbus write coil exception @%d: %s", address, e)
            return False

    def _read_coil(self, address: int) -> Optional[bool]:
        if not self._connected or self._client is None:
            return None
        try:
            res = self._client.read_coils(address, count=1, device_id=self._device_id)
            if res.isError():
                return None
            return bool(res.bits[0])
        except Exception:
            return None

    def _write_holding_register(self, address: int, value: int) -> bool:
        if not self._connected or self._client is None:
            return False
        try:
            res = self._client.write_register(address, value, device_id=self._device_id)
            if res.isError():
                logger.warning("Modbus write register error @%d=%d", address, value)
                return False
            return True
        except Exception as e:
            logger.error("Modbus write register exception @%d: %s", address, e)
            return False

    def _read_holding_register(self, address: int) -> Optional[int]:
        if not self._connected or self._client is None:
            return None
        try:
            res = self._client.read_holding_registers(address, count=1, device_id=self._device_id)
            if res.isError():
                return None
            return int(res.registers[0])
        except Exception:
            return None

    def _log(self, message: str):
        logger.debug("[Modbus] %s", message)
        if self._log_callback:
            self._log_callback(message)
