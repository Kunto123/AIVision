"""
VisionInspect - Configuration Manager
Membaca dan menyimpan konfigurasi aplikasi dalam format JSON.
Menyediakan default untuk semua parameter.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from visioninspect.utils import normalize_wsl_path, is_wsl

APP_NAME = "VisionInspect"
APP_VERSION = "1.0.0"

# Normalisasi path WSL: C:\foo → /mnt/c/foo (khusus WSL)
_raw_data_dir = os.getenv("VISIONINSPECT_DATA", "")
if _raw_data_dir and is_wsl():
    DEFAULT_DATA_DIR = normalize_wsl_path(_raw_data_dir)
elif _raw_data_dir:
    DEFAULT_DATA_DIR = Path(_raw_data_dir)
else:
    DEFAULT_DATA_DIR = Path.home() / ".visioninspect"
if not DEFAULT_DATA_DIR.is_absolute():
    DEFAULT_DATA_DIR = Path.home() / ".visioninspect"


class ConfigError(Exception):
    """Base exception for config errors."""
    pass


class Config:
    """
    Manajemen konfigurasi aplikasi.
    Menyimpan di JSON dengan atomic write (write-to-temp lalu rename).
    """

    DEFAULTS: Dict[str, Any] = {
        # General
        "language": "id",
        "theme": "dark",
        "data_dir": str(DEFAULT_DATA_DIR),
        "show_debug": False,

        # Camera
        "camera": {
            "device_index": 0,
            "resolution_width": 1920,
            "resolution_height": 1080,
            "fps_target": 30,
            "exposure": -1,  # -1 = auto; >=0 = kunci nilai tetap (F2)
            "gain": -1,  # -1 = auto; >=0 = kunci gain (F2)
            "white_balance": -1,  # -1 = auto; >=0 = kunci Kelvin (F2)
        },

        # Rollout / deploy bertahap
        "rollout": {
            "shadow_mode": False,
            # True: hasil hanya ditampilkan, coil result_ok TIDAK ditulis.
            # AWAS: tanpa sinyal OK, PLC memvonis SEMUA part NG (= tolak semua).
        },

        # ROI & model config = PER-TEMPLATE (diatur di tab TEACH), bukan di sini.
        # Template baru di-seed oleh MainWindow._new_template_defaults().

        # Inference
        "inference": {
            # continuous  — infer terus, self-trigger lewat gate part-check.
            # plc_trigger — infer saat ada trigger (coil PLC / tombol / POST).
            "mode": "continuous",  # continuous | plc_trigger
            "openvino_device": "CPU",   # CPU | GPU | AUTO (Tugas 5)
            # CPU hybrid (P+E core): batasi ke P-core → latency stabil.
            # Tidak berpengaruh di CPU non-hybrid / device=GPU.
            "cpu_pcore_only": False,
            "enable_int8": True,
            # Batas waktu 1 siklus trigger; lewat ini = peringatan + TIDAK ada
            # pulse. Setel LEBIH PENDEK dari watchdog ladder.
            "trigger_timeout_ms": 2000,
            # Tetap infer di antara trigger (skor live terlihat); hasil resmi
            # tetap dari frame trigger. Default MATI — di CPU 2 core ~2x lambat.
            "infer_when_idle": False,
            # Jarak minimum antar hitungan part (cegah 1 part terhitung berkali).
            # Hanya dipakai kalau tidak ada trigger PLC / gate. 0 = tiap inspeksi.
            "count_cooldown_ms": 1500,
            # Verdict OK (gate & QC) baru keluar setelah N infer OK berturut;
            # NG mereset. 1 = tanpa konfirmasi. Continuous saja (trigger = 1).
            "confirm_ok_frames": 1,
            "cycle_delay_ms": 1000,  # jeda antar siklus inspeksi (ms), 0=langsung
        },

        # PLC
        "plc": {
            # Transport FX Computer Link (jalur GX Works2); MODBUS dihapus.
            # Format serial dikunci 7E1 — hanya port & baudrate bisa diatur.
            "enabled": False,
            "port": "COM1",
            "baudrate": 9600,
            "timeout": 1.0,
            # Input `switch_template`: "cycle" = tiap sinyal maju 1 template
            # (memutar); "register" = lompat ke nomor di program_register.
            "template_switch_mode": "cycle",
            "pulse_ms": 300,          # durasi coil hasil nyala (OK/NG), ms
            "scan_range": 127,        # range probe scan coil (0..N)
            # IO mapping — outputs = coil yang sistem tulis, inputs = yang PLC tulis.
            # Ganti di sini atau lewat tab I/O Settings.
            "io_map": {
                "outputs": {
                    "result_ok": 1,
                    "part_ready": 3,
                    "busy": 4,
                    # Di-toggle ±1 Hz selama sistem sehat. Ladder memantau
                    # PERUBAHANnya: diam > N detik = sistem rusak (bukan NG).
                    "heartbeat": 7,
                    # Pulse saat operator masuk RUN — lihat plc.reset_on_run_entry.
                    "session_reset": 9,
                },
                "inputs": {
                    "trigger": 0,
                    "reset_result": 5,
                    # Ganti TEMPLATE aktif (nomornya dari program_register).
                    # Config lama dengan key "switch_program" tetap dikenali.
                    "switch_template": 6,
                    # PLC memvonis NG → sistem menambah counter NG dan
                    # membersihkan state siklus (siap part berikutnya).
                    "ng_from_plc": 8,
                },
                "program_register": 10,
            },
            # Reconnect
            "reconnect_interval": 5.0,
            "max_reconnect_attempts": 0,  # 0 = unlimited
            # Opt-in: pulse coil session_reset (M9) saat masuk RUN → ladder
            # bersihkan state basi. DEFAULT MATI (butuh rung M9·/M100 dulu).
            "reset_on_run_entry": False,
        },

        # Flask API
        "flask_api": {
            "enabled": False,
            "port": 5000,
            "api_key": "",
        },

        # History / Retention
        "history": {
            "save_all_ng": True,
            "save_ok_sample_percent": 10,
            "auto_purge_days": 30,
            "max_history_entries": 10000,
        },

        # Logging
        "logging": {
            "level": "INFO",
            "max_bytes": 10 * 1024 * 1024,  # 10 MB
            "backup_count": 5,
        },

        # Watchdog
        "watchdog": {
            "inference_timeout_sec": 10.0,
            "camera_timeout_sec": 5.0,
            "check_interval_sec": 2.0,
        },

        # `ng_debounce_ms` TIDAK DIPAKAI (mengukur durasi NG, bukan jumlah part).
        # Digantikan inference.count_cooldown_ms; key disimpan agar config lama kebaca.
        "ng_debounce_ms": 500,   # [TIDAK DIPAKAI]

        # Active program
        "active_program": "",

        # PostgreSQL
        "postgresql": {
            "enabled": False,
            "host": "localhost",
            "port": 5432,
            "dbname": "visioninspect",
            "user": "postgres",
            "password": "",
            "sslmode": "prefer",
            "connect_timeout": 10,
        },
    }

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or (DEFAULT_DATA_DIR / "config.json")
        self._data: Dict[str, Any] = {}
        self._load()

    # ---- Public API ----

    @property
    def path(self) -> Path:
        return self._config_path

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by dot-separated key, e.g. 'camera.device_index'."""
        return self._get_nested(self._data, key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value by dot-separated key."""
        keys = key.split(".")
        d = self._data
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def save(self) -> None:
        """Atomic write to config file."""
        self._ensure_data_dir()
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._config_path.parent,
            suffix=".tmp",
            delete=False,
        )
        try:
            json.dump(self._data, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            try:
                os.fsync(tmp.fileno())
            except (OSError, AttributeError):
                pass
            tmp.close()
            os.replace(tmp.name, str(self._config_path))
        except Exception:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            raise

    def reset_to_defaults(self) -> None:
        """Reset all config to defaults."""
        self._data = self._deep_copy(self.DEFAULTS)

    def get_all(self) -> Dict[str, Any]:
        """Return full config dict."""
        return self._deep_copy(self._data)

    # ---- Internal ----

    def _load(self) -> None:
        if self._config_path.exists() and self._config_path.stat().st_size > 0:
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                raise ConfigError(f"Failed to load config: {e}")
        else:
            self.reset_to_defaults()
            self.save()

        # Merge with defaults to ensure new keys exist
        self._data = self._deep_merge(self._deep_copy(self.DEFAULTS), self._data)
        self._migrate()

    #: Nilai config lama → penggantinya. Tanpa ini, mode yang sudah dihapus
    #: jatuh ke "continuous" — stasiun bertrigger diam-diam menulis coil OK.
    MODE_ALIASES: Dict[str, str] = {
        "manual": "plc_trigger",
    }

    def _migrate(self) -> None:
        """Normalisasi config lama supaya tidak jatuh ke default diam-diam.
        Sengaja hanya mutasi di memori — file terkoreksi saat save berikutnya."""
        mode = self._get_nested(self._data, "inference.mode")
        if isinstance(mode, str) and mode in self.MODE_ALIASES:
            self.set("inference.mode", self.MODE_ALIASES[mode])

    def _ensure_data_dir(self) -> None:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_nested(d: dict, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    @staticmethod
    def _deep_copy(obj: Any) -> Any:
        return json.loads(json.dumps(obj))

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursive merge: override values into base."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = Config._deep_copy(value)
        return result
