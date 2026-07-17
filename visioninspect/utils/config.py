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


APP_NAME = "VisionInspect"
APP_VERSION = "1.0.0"
DEFAULT_DATA_DIR = Path(os.getenv("VISIONINSPECT_DATA", default=""))
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
            "exposure": -1,  # -1 = auto
        },

        # ROI
        "roi": {
            "enabled": True,
            "x": 0,
            "y": 0,
            "width": 256,
            "height": 256,
            "multi_roi_enabled": False,
            "roi_list": [],  # list of {x, y, width, height, model_id}
        },

        # Model / Training
        "model": {
            "algorithm": "patchcore",  # patchcore | efficientad
            "backbone": "resnet18",    # resnet18 | wide_resnet50_2
            "coreset_sampling_ratio": 0.1,
            "input_size": 256,
            "threshold_mode": "adaptive",  # adaptive | manual
            "manual_threshold": 0.5,
            "threshold_margin_sigma": 3.0,
        },

        # Inference
        "inference": {
            "mode": "continuous",  # continuous | plc_trigger | manual
            "openvino_device": "CPU",
            "enable_int8": True,
            "cycle_delay_ms": 1000,  # jeda antar siklus inspeksi (ms), 0=langsung
        },

        # PLC
        "plc": {
            "enabled": False,
            "mode": "rs232",        # rs232 | rs485
            "port": "COM1",
            "baudrate": 9600,
            "bytesize": 8,
            "parity": "N",
            "stopbits": 1,
            "timeout": 1.0,
            "protocol": "modbus",    # modbus | ascii
            # RS485 specific
            "rs485_direction": "auto",  # auto | rts
            "rs485_delay_before_tx": 0.0,
            "rs485_delay_after_tx": 0.0,
            # Modbus
            "modbus_slave_id": 1,
            # Reconnect
            "reconnect_interval": 5.0,
            "max_reconnect_attempts": 0,  # 0 = unlimited
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

        # Global settings
        "ng_debounce_ms": 500,

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
