"""
VisionInspect — Entry Point
Sistem Inspeksi Visual Industri Berbasis AI (CPU-only, full local).
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

# 1. Pastikan package root ada di sys.path
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


def _is_edge_mode() -> bool:
    """Baca flag `edge_mode` dari config SEBELUM import apa pun.
    True → torch tidak dimuat (hemat ±250 MB / ±5 dtk startup di PC edge)."""
    try:
        data_dir = os.environ.get("VISIONINSPECT_DATA", "")
        candidates = []
        if data_dir:
            candidates.append(Path(data_dir) / "config.json")
        candidates.append(Path.home() / ".visioninspect" / "config.json")
        candidates.append(Path(__file__).resolve().parent / "data" / "config.json")
        for cand in candidates:
            if cand and cand.exists():
                with open(cand, encoding="utf-8") as f:
                    import json
                    return bool(json.load(f).get("edge_mode", False))
    except Exception:
        pass
    return False


# 2. Import torch lebih awal (sebelum PySide6/cv2/openvino) — urutan ini
#    menghindari TLS-slot exhaustion di Windows (WinError 1114).
_EDGE_MODE = _is_edge_mode()

if not _EDGE_MODE:
    try:
        import torch  # noqa: F401
    except Exception as _e:
        # Kegagalan JANGAN senyap — tanpa warning, training diam-diam jatuh
        # ke WSL/simple mode dan user tidak tahu.
        print(f"WARN: Gagal import torch di startup ({_e}) — training akan "
              "menggunakan mode fallback", file=sys.stderr)
else:
    print("INFO: edge_mode=true — torch tidak dimuat "
          "(PC edge inference-only)", file=sys.stderr)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from visioninspect.utils.config import Config, ConfigError, APP_NAME, APP_VERSION
from visioninspect.utils.i18n import Translator
from visioninspect.utils.logging_setup import setup_logging, get_logger
from visioninspect.gui.main_window import MainWindow


def _setup_qt_message_handler():
    """Saring warning Qt yang tidak berbahaya (QPainter::end dll)."""
    from PySide6.QtCore import qInstallMessageHandler, QtMsgType
    import sys

    def handler(mode, context, message):
        # Abaikan QPainter::end warnings yang tidak berbahaya
        if "QPainter::end" in message and "saved states" in message:
            return
        # Cetak warning/error lainnya ke stderr (default Qt behavior)
        if mode == QtMsgType.QtCriticalMsg or mode == QtMsgType.QtFatalMsg:
            print(f"QT CRITICAL: {message}", file=sys.stderr)
        elif mode == QtMsgType.QtWarningMsg:
            print(f"QT WARN: {message}", file=sys.stderr)

    qInstallMessageHandler(handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} v{APP_VERSION} — Industrial Visual Inspection System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path file config (default: <proyek>/data/config.json)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Override folder data (default: <proyek>/data)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="",
        help="Logging level override",
    )
    parser.add_argument(
        "--version", action="version", version=f"{APP_NAME} v{APP_VERSION}"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Config
    try:
        config = Config()
        if args.config:
            config = Config(Path(args.config))
            config.save()
        if args.data_dir:
            config.set("data_dir", args.data_dir)
            config.save()
    except ConfigError as e:
        print(f"FATAL: Configuration error: {e}", file=sys.stderr)
        return 1

    data_dir = Path(config.get("data_dir", ""))
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 2. Logging (rotating file handler per subsistem, di <data>/logs/)
    log_level = args.log_level or config.get("logging.level", "INFO")
    log_dir = data_dir / "logs"
    setup_logging(
        log_dir=log_dir,
        level=log_level,
        max_bytes=config.get("logging.max_bytes", 10 * 1024 * 1024),
        backup_count=config.get("logging.backup_count", 5),
    )
    logger = get_logger("app")

    logger.info("Python interpreter: %s", sys.executable)

    # 3. Translator (bahasa UI: "id" | "en")
    translator = Translator(language=config.get("language", "id"))

    # 4. Qt application
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("VisionInspect")
    app.setStyle("Fusion")            # base style
    _setup_qt_message_handler()       # saring QPainter warning tak berbahaya

    # 5. Main window
    try:
        window = MainWindow(config, translator)
        window.show()
        logger.info("Application started (data dir: %s)", data_dir)
    except Exception as e:
        logger.critical("Failed to initialize main window: %s", traceback.format_exc())
        print(f"FATAL: {e}", file=sys.stderr)
        return 1

    # 6. Event loop
    exit_code = app.exec()

    logger.info("Application exited with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
