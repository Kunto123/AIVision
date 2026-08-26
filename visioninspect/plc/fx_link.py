"""
VisionInspect — FX Computer Link (protokol port pemrograman Mitsubishi)
=======================================================================
Alternatif Modbus RTU untuk PLC FX yang TIDAK menyediakan MODBUS slave.

Latar belakang: MODBUS di FX3U asli menuntut adaptor khusus (-MB) dan blok
setup `M8411`. Pada unit yang tidak mendukungnya, `D8400`/`D8401` tetap nol
berapa kali pun di-power-cycle. Protokol ini memakai jalur yang SAMA dengan
GX Works2, jadi ia bekerja di port yang sudah terbukti hidup.

Kontraknya identik dengan `ModbusRTUManager` supaya `main_window` tidak perlu
tahu transport mana yang dipakai. Nomor coil dipetakan langsung ke relay M
(coil 1 → M1) — sengaja, karena io_map memang sudah memakai penomoran itu.

Catatan format: `TransportSerial` milik fxplc mengunci 7E1; hanya baudrate
yang bisa diatur. Itu memang format port pemrograman FX.
"""

import asyncio
import threading
import time
from typing import Callable, Optional

from visioninspect.plc.io_map import build_io_map
from visioninspect.utils.logging_setup import get_logger

logger = get_logger("plc")

try:
    from fxplc.client.FXPLCClient import FXPLCClient
    from fxplc.transports.TransportSerial import TransportSerial
    HAS_FXPLC = True
except ImportError:  # pragma: no cover - tergantung environment
    FXPLCClient = None      # type: ignore[assignment]
    TransportSerial = None  # type: ignore[assignment]
    HAS_FXPLC = False
    logger.warning("fxplc tidak terpasang — transport FX tidak tersedia")


#: Alamat basis bit-image relay M, diambil dari
#: fxplc.client.FXPLCClient.registers_map_bit_images["M"] = (0x0100, 8).
#: byte = 0x0100 + nomor//8, bit = nomor%8. Terverifikasi di lapangan:
#: menyalakan M1+M3 menghasilkan byte0 = 0x0A, bukan 0x14.
M_BIT_IMAGE_BASE = 0x0100
M_BITS_PER_BYTE = 8

#: read_bytes mengemas jumlah byte dalam satu oktet (struct ">HB").
_MAX_BATCH_BYTES = 255

# ⚠️ PETA ALAMAT fxplc ITU LINEAR (FXPLCClient.py:32):
#   registers_map_bit_images["M"] = (0x0100, 8)  →  0x0100 + nomor//8
# Artinya M8000 dihitung ke 0x0100 + 1000 = 0x04E8 — blok memori yang TIDAK
# ADA isinya, padahal special relay FX M8000+ sesungguhnya duduk di 0x01E0
# (terverifikasi lapangan 2026-08-26: read_bytes(0x01E0,1)=0x09 saat RUN,
# sementara read_bit("M8000") selalu NoResponseError). Konsekuensinya:
# JANGAN PERNAH memakai relay M ≥ M1000 (special relay) sebagai probe atau
# pembacaan lewat library ini — hanya relay M biasa (M0–M3071 area umum)
# yang alamatnya benar di bawah pemetaan linear itu.
CONNECT_PROBE_COIL = None   # probe = coil result_ok dari io_map (lihat connect())


class FXPLCManager:
    """FX Computer Link — antarmuka sama persis dengan ModbusRTUManager."""

    #: Transaksi gagal beruntun sebelum breaker terbuka.
    _FAIL_THRESHOLD = 3
    _RECONNECT_BACKOFF_MAX = 60.0
    _RECONNECT_GIVE_UP = 5

    def __init__(self, plc_config: Optional[dict] = None):
        plc_config = plc_config or {}
        self._config = plc_config
        self._io_map = build_io_map(plc_config)
        self._pulse_ms = int(plc_config.get("pulse_ms", 300))
        self._port = str(plc_config.get("port", "COM1"))
        self._baudrate = int(plc_config.get("baudrate", 9600))
        self._timeout = float(plc_config.get("timeout", 1.0))

        self._client = None
        self._transport = None
        self._connected = False
        # Tahap kegagalan connect() terakhir: None | "port" | "slave".
        # connect() menutup transport di semua jalur gagal, jadi pembeda
        # "port gagal dibuka" vs "PLC tidak menjawab" tidak bisa diambil
        # dari port_open setelahnya — diagnose() membaca flag ini.
        self._last_connect_fail: Optional[str] = None
        self._lock = threading.RLock()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        self._last_inputs: dict[str, bool] = {}

        self._on_trigger: Optional[Callable[[], None]] = None
        self._on_reset: Optional[Callable[[], None]] = None
        self._on_switch_program: Optional[Callable[[int], None]] = None
        self._on_status_change: Optional[Callable[[bool], None]] = None
        self._log_callback: Optional[Callable[[str], None]] = None

        self._reconnect_interval = max(
            1.0, float(plc_config.get("reconnect_interval", 5.0)))
        self._consec_fail = 0
        self._breaker_open = False
        self._breaker_until = 0.0
        self._breaker_logged = False
        self._reconnect_tries = 0
        self._reconnect_gave_up = False

    # ---- Status koneksi (2 lapis) ------------------------------------------

    @property
    def port_open(self) -> bool:
        """Port serial terbuka — TIDAK membuktikan PLC menjawab.

        Kabel dicabut setelah open() tetap membuat flag ini True; itulah
        sebabnya badge di UI wajib membedakannya dari `is_connected`
        (blueprint §T6: dulu keduanya tidak dibedakan, badge hijau bohong).
        """
        return self._transport is not None

    def _seed_input_state(self) -> None:
        """Isi `_last_inputs` dari kondisi coil saat ini SEBELUM poll pertama.

        `read_inputs()` mendeteksi event dari tepi-naik (False → True).
        Tanpa seeding, coil input yang kebetulan sedang ON saat aplikasi
        connect (mis. reset_result ditahan operator) langsung terhitung
        sebagai event palsu pada tick pertama.
        """
        inputs = self._io_map.get("inputs") or {}
        if not inputs:
            return
        names = list(inputs.keys())
        values = self._read_coils_batch([inputs[n] for n in names])
        for name in names:
            val = values.get(inputs[name])
            if val is not None:
                self._last_inputs[name] = val

    # ---- Callbacks ----

    def set_on_trigger(self, cb): self._on_trigger = cb
    def set_on_reset(self, cb): self._on_reset = cb
    def set_on_switch_program(self, cb): self._on_switch_program = cb
    def set_on_status_change(self, cb): self._on_status_change = cb
    def set_log_callback(self, cb): self._log_callback = cb

    def _log(self, message: str):
        logger.debug("[FX] %s", message)
        if self._log_callback:
            self._log_callback(message)

    @property
    def io_map(self) -> dict:
        return self._io_map

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    # ---- Jembatan async → sync -------------------------------------------
    #
    # fxplc benar-benar async: pembacaan serial lewat ThreadPoolExecutor di
    # dalam transport. Jadi ia butuh event loop yang hidup, bukan
    # asyncio.run() sekali pakai per panggilan. Loop diletakkan di thread
    # sendiri; semua panggilan diserialkan lewat satu loop itu sehingga
    # tidak ada dua transaksi menabrak port yang sama.

    def _ensure_loop(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            return
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="fxplc-loop", daemon=True)
        self._loop_thread.start()

    def _run(self, coro, timeout: Optional[float] = None):
        """Jalankan coroutine di loop khusus, tunggu hasilnya."""
        self._ensure_loop()
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout or (self._timeout + 1.0))

    def _stop_loop(self) -> None:
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=3.0)
            self._loop_thread = None
        try:
            self._loop.close()
        except Exception:
            pass
        self._loop = None

    # ---- Lifecycle ----

    def connect(self) -> bool:
        if not HAS_FXPLC:
            self._log("ERROR: fxplc tidak terpasang")
            return False
        with self._lock:
            if self._connected and self._client is not None:
                return True
            try:
                self._ensure_loop()
                self._transport = TransportSerial(
                    self._port, baudrate=self._baudrate, timeout=self._timeout)
                self._client = FXPLCClient(self._transport)
            except Exception as e:
                self._close_transport()
                self._client = None
                self._connected = False
                self._last_connect_fail = "port"
                # WARNING, bukan debug — insiden 2026-08-26 butuh berjam-jam
                # karena satu-satunya jejak ("PLC connect gagal saat startup"
                # di app.log) tidak punya sebab di plc.log.
                logger.warning("Port %s tidak bisa dibuka: %r", self._port, e)
                self._log(f"FX connect gagal: {e}")
                if self._on_status_change:
                    self._on_status_change(False)
                return False
            # Port terbuka ≠ PLC menjawab (blueprint §T6 / Fase 4). Dulu
            # status langsung hijau di sini — badge "Terhubung" tampil walau
            # kabel dicabut. Probe = coil result_ok dari io_map (relay M
            # BIASA). BUKAN M8000: peta alamat fxplc linear (0x0100+n//8)
            # tidak pernah sampai ke special relay M8000+ yang duduk di
            # 0x01E0 — read_bit("M8000") tidak akan pernah dijawab PLC.
            probe_label = self._label(
                self._io_map.get("outputs", {}).get("result_ok", 1))
            try:
                bool(self._run(self._client.read_bit(probe_label)))
            except Exception as e:
                logger.warning(
                    "Probe %s gagal saat connect (%r) — port %s terbuka "
                    "tapi PLC tidak menjawab", probe_label, e, self._port)
                self._close_transport()
                self._client = None
                self._connected = False
                self._last_connect_fail = "slave"
                self._log(
                    f"Port {self._port} terbuka tapi PLC tidak menjawab "
                    f"(probe {probe_label}) — status tetap putus")
                if self._on_status_change:
                    self._on_status_change(False)
                return False
            self._last_connect_fail = None
            self._connected = True
            # Seed state input SEBELUM poll pertama: coil input yang kebetulan
            # ON saat connect tidak boleh dihitung sebagai tepi-naik palsu.
            try:
                self._seed_input_state()
            except Exception as e:
                logger.debug("Seed input state gagal (diabaikan): %r", e)
            self._log(f"FX terhubung ({self._port} @ {self._baudrate}, 7E1)")
            if self._on_status_change:
                self._on_status_change(True)
            return True

    def disconnect(self) -> None:
        with self._lock:
            self._client = None
            self._close_transport()
            self._stop_loop()
            was = self._connected
            self._connected = False
            if was and self._on_status_change:
                self._on_status_change(False)

    def _close_transport(self) -> None:
        t = self._transport
        self._transport = None
        if t is None:
            return
        try:
            t.close()
        except Exception:
            pass

    # ---- Circuit breaker (semantik sama dengan ModbusRTUManager) ----------

    def _io_ok(self) -> None:
        self._consec_fail = 0

    def _io_failed(self) -> None:
        self._consec_fail += 1
        if self._consec_fail >= self._FAIL_THRESHOLD and not self._breaker_open:
            self._trip_breaker()

    def _next_backoff(self) -> float:
        return min(self._reconnect_interval * (2 ** self._reconnect_tries),
                   self._RECONNECT_BACKOFF_MAX)

    def _trip_breaker(self, why: str = "") -> None:
        self._breaker_open = True
        delay = self._next_backoff()
        self._breaker_until = time.monotonic() + delay
        sebab = why or f"{self._consec_fail} transaksi gagal beruntun"
        if not self._breaker_logged:
            self._breaker_logged = True
            logger.error(
                "PLC tidak menjawab (%s) — komunikasi dihentikan sementara "
                "supaya UI tidak membeku. Percobaan berikutnya %.0f dtk lagi.",
                sebab, delay)
        else:
            logger.debug("PLC masih tidak menjawab (%s)", sebab)
        self._log("PLC tidak menjawab — komunikasi dijeda")
        with self._lock:
            self._client = None
            self._close_transport()
            self._connected = False
        if self._on_status_change:
            self._on_status_change(False)

    def clear_breaker(self) -> None:
        self._breaker_open = False
        self._breaker_until = 0.0
        self._consec_fail = 0
        self._breaker_logged = False
        self._reconnect_tries = 0
        self._reconnect_gave_up = False

    def _give_up(self) -> None:
        self._reconnect_gave_up = True
        self._breaker_open = True
        self.disconnect()
        logger.error(
            "PLC tetap tidak menjawab setelah %d percobaan — pemulihan "
            "otomatis DIHENTIKAN dan port %s dilepas. Tekan Test Koneksi "
            "untuk mencoba lagi.", self._reconnect_tries, self._port)

    def try_reconnect(self) -> bool:
        if self._reconnect_gave_up:
            return False
        # Dulu: `if not self._breaker_open: return self.is_connected`.
        # Lubangnya: kegagalan connect SAAT STARTUP (port dipakai aplikasi
        # lain / PLC belum menyala sehingga probe gagal) tidak membuka
        # breaker, jadi timer pemulihan melewati jalur ini dan tidak pernah
        # mencoba ulang — diam sampai operator menekan Test Koneksi.
        if self.is_connected:
            return True
        if time.monotonic() < self._breaker_until:
            return False

        self._breaker_open = False
        self._consec_fail = 0
        if self.connect() and self._read_bit_raw("M1") is not None:
            self._reconnect_tries = 0
            self._breaker_logged = False
            logger.info("PLC menjawab lagi — komunikasi dilanjutkan")
            return True

        self._reconnect_tries += 1
        if self._reconnect_tries >= self._RECONNECT_GIVE_UP:
            self._give_up()
            return False
        self._trip_breaker("percobaan pemulihan gagal")
        return False

    # ---- Primitif ---------------------------------------------------------

    @staticmethod
    def _label(addr: int) -> str:
        """Coil N → relay M N. Penomoran io_map memang sudah disamakan."""
        return f"M{int(addr)}"

    def _blocked(self) -> bool:
        return self._breaker_open or not self.is_connected

    def _read_bit_raw(self, label: str) -> Optional[bool]:
        if self._blocked():
            return None
        try:
            v = bool(self._run(self._client.read_bit(label)))
            self._io_ok()
            return v
        except Exception as e:
            logger.debug("FX read_bit %s gagal: %r", label, e)
            self._io_failed()
            return None

    def _write_coil(self, address: int, value: bool) -> bool:
        if self._blocked():
            return False
        label = self._label(address)
        try:
            self._run(self._client.write_bit(label, bool(value)))
            self._io_ok()
            return True
        except Exception as e:
            logger.warning("FX write %s gagal: %r", label, e)
            self._io_failed()
            return False

    def _read_coil(self, address: int) -> Optional[bool]:
        return self._read_bit_raw(self._label(address))

    def _read_coils_batch(self, addrs: list[int]) -> dict[int, Optional[bool]]:
        """Baca banyak relay M dalam SATU transaksi.

        Terukur di lapangan: 4 read_bit terpisah ≈ 80 ms, satu read_bytes
        yang mencakup semuanya ≈ 20 ms. Karena semua coil kontrak berada di
        M0-M15, borongan ini memangkas siklus poll hampir tiga kali lipat.
        """
        if not addrs:
            return {}
        if self._blocked():
            return {a: None for a in addrs}

        lo = M_BIT_IMAGE_BASE + min(addrs) // M_BITS_PER_BYTE
        hi = M_BIT_IMAGE_BASE + max(addrs) // M_BITS_PER_BYTE
        count = hi - lo + 1
        if count > _MAX_BATCH_BYTES:
            # Alamat tersebar terlalu jauh — borongan malah boros. Jatuh ke
            # pembacaan satuan supaya tetap benar.
            return {a: self._read_coil(a) for a in addrs}

        try:
            raw = self._run(self._client.read_bytes(lo, count))
            self._io_ok()
        except Exception as e:
            logger.debug("FX read_bytes(0x%04X, %d) gagal: %r", lo, count, e)
            self._io_failed()
            return {a: None for a in addrs}

        out: dict[int, Optional[bool]] = {}
        for a in addrs:
            idx = (M_BIT_IMAGE_BASE + a // M_BITS_PER_BYTE) - lo
            if 0 <= idx < len(raw):
                out[a] = bool(raw[idx] & (1 << (a % M_BITS_PER_BYTE)))
            else:
                out[a] = None
        return out

    # ---- API yang dipakai main_window ------------------------------------

    def set_output(self, name: str, value: bool) -> bool:
        addr = self._io_map["outputs"].get(name)
        if addr is None:
            logger.warning("Output '%s' tidak ada di io_map", name)
            return False
        return self._write_coil(addr, value)

    def pulse_output(self, name: str, pulse_ms: Optional[int] = None) -> bool:
        ms = pulse_ms if pulse_ms is not None else self._pulse_ms
        ok = self.set_output(name, True)
        if ok and ms > 0:
            time.sleep(ms / 1000.0)
            self.set_output(name, False)
        return ok

    def reset_outputs(self) -> None:
        for name in self._io_map["outputs"]:
            self._write_coil(self._io_map["outputs"][name], False)

    def read_inputs(self) -> dict:
        """Baca semua coil input + deteksi tepi naik — satu transaksi."""
        inputs = self._io_map.get("inputs") or {}
        if not inputs:
            return {}
        names = list(inputs.keys())
        values = self._read_coils_batch([inputs[n] for n in names])
        events: dict[str, bool] = {}
        for name in names:
            val = values.get(inputs[name])
            if val is None:
                continue                      # baca gagal → jangan tebak
            prev = self._last_inputs.get(name, False)
            self._last_inputs[name] = val
            if val and not prev:
                events[name] = True
        return events

    def read_program_register(self) -> Optional[int]:
        """Nomor template dari data register D."""
        addr = self._io_map.get("program_register", 10)
        if self._blocked():
            return None
        try:
            v = int(self._run(self._client.read_int(f"D{int(addr)}")))
            self._io_ok()
            return v
        except Exception as e:
            logger.debug("FX read D%s gagal: %r", addr, e)
            self._io_failed()
            return None

    def read_coil_state(self, name: str) -> Optional[bool]:
        outputs = self._io_map.get("outputs", {})
        inputs = self._io_map.get("inputs", {})
        if name in outputs:
            return self._read_coil(outputs[name])
        if name in inputs:
            return self._read_coil(inputs[name])
        return None

    def read_all_coil_states(self) -> dict:
        """Semua coil untuk I/O Monitor — satu transaksi borongan."""
        outputs = self._io_map.get("outputs", {})
        inputs = self._io_map.get("inputs", {})
        pairs = list(outputs.items()) + list(inputs.items())
        values = self._read_coils_batch([a for _, a in pairs])
        return {name: values.get(addr) for name, addr in pairs}

    def scan_coils(self, max_addr: int = 127) -> list[int]:
        """Relay M yang bisa dibaca. Dibaca borongan supaya tidak lama."""
        if not self.is_connected:
            return []
        addrs = list(range(max(0, max_addr) + 1))
        values = self._read_coils_batch(addrs)
        return [a for a in addrs if values.get(a) is not None]

    def find_active_coils(self, max_addr: int = 127) -> list[int]:
        """Relay M yang sedang ON — untuk tombol 'Deteksi Aktif'."""
        if not self.is_connected:
            return []
        addrs = list(range(max(0, max_addr) + 1))
        values = self._read_coils_batch(addrs)
        return [a for a in addrs if values.get(a)]

    def diagnose(self) -> dict:
        """Uji bertahap, formatnya sama dengan ModbusRTUManager.diagnose()."""
        info = {
            "ok": False, "stage": "", "detail": "",
            "port": self._port, "baudrate": self._baudrate,
            "parity": "E (dikunci fxplc)", "slave_id": "-", "probe_addr": None,
        }
        if not HAS_FXPLC:
            info["stage"] = "library"
            info["detail"] = ("fxplc tidak terpasang. Pasang dengan:\n"
                              "pip install git+https://github.com/KrystianD/fxplc.git")
            return info

        self.clear_breaker()
        was_connected = self.is_connected
        if not was_connected and not self.connect():
            # connect() menutup transport di setiap jalur gagal, jadi
            # port_open saja TIDAK cukup untuk membedakan penyebab — tahap
            # gagalnya dicatat di `_last_connect_fail`. Sebelum 2026-08-26
            # semua kegagalan dilempar ke stage="port" ("Port tidak bisa
            # dibuka") walau port terbuka sempurna dan yang gagal coil
            # probe-nya — bohong yang mengarahkan debugging ke kabel.
            if self._last_connect_fail == "slave":
                probe = self._io_map.get("outputs", {}).get("result_ok", 1)
                info["probe_addr"] = probe
                info["stage"] = "slave"
                info["detail"] = (
                    f"Port {self._port} terbuka, tapi PLC tidak menjawab "
                    f"(probe {self._label(probe)}). Cek baudrate (sekarang "
                    f"{self._baudrate}) dan pastikan PLC menyala.")
            else:
                info["stage"] = "port"
                info["detail"] = (
                    f"Port {self._port} tidak bisa dibuka. Cek nomor COM, "
                    "dan pastikan GX Works2 tidak sedang memakai port yang "
                    "sama.")
            return info

        probe = self._io_map.get("outputs", {}).get("result_ok", 1)
        info["probe_addr"] = probe
        # Jalur ini hanya tersisa untuk link yang SUDAH terhubung sebelumnya
        # (was_connected): cek ulang cepat tanpa connect() lagi. Untuk link
        # yang baru saja connect() berhasil, probe yang sama sudah lulus.
        if not self.is_connected or self._read_coil(probe) is None:
            info["stage"] = "slave"
            info["detail"] = (
                f"PLC berhenti menjawab di {self._label(probe)}. Cek "
                f"baudrate ({self._baudrate}) dan pastikan PLC menyala.")
            return info

        info["ok"] = True
        info["stage"] = "ok"
        info["detail"] = (
            f"PLC menjawab di {self._label(probe)} lewat {self._port} @ "
            f"{self._baudrate} 7E1 (FX Computer Link).")
        return info
