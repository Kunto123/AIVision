"""
VisionInspect — Entry Point
Sistem Inspeksi Visual Industri Berbasis AI (CPU-only, full local).
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

# Ensure package root is in path
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


def _is_edge_mode() -> bool:
    """Baca flag `edge_mode` dari config SEBELUM import apa pun (Tugas 4).

    PC edge (i3-1315U, inference-only) tidak pernah perlu memuat torch
    (±250 MB RSS / ±5 s startup). Torch hanya boleh dimuat kalau memang
    dibutuhkan training, dan hanya di PC dev.
    """
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


_EDGE_MODE = _is_edge_mode()

if not _EDGE_MODE:
    # Early import torch (sebelum PySide6/cv2/openvino) untuk menghindari
    # konflik TLS-slot exhaustion di Windows (WinError 1114). Lihat debug:
    # https://github.com/pytorch/pytorch/issues/110488
    try:
        import torch  # noqa: F401
    except Exception as _e:
        # Kegagalan 1114 TIDAK boleh senyap — efeknya training diam-diam
        # jatuh ke WSL/simple mode tanpa user tahu.
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
        help="Path to config file (default: ~/.visioninspect/config.json)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Data directory override (default: ~/.visioninspect)",
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

    # --- Configuration ---
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

    # --- Logging ---
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

    # --- Translator ---
    translator = Translator(language=config.get("language", "id"))

    # --- Qt Application ---
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("VisionInspect")

    # High-DPI support
    app.setStyle("Fusion")  # Use Fusion style as base

    # Saring QPainter warnings yang tidak berbahaya
    _setup_qt_message_handler()

    # --- Main Window ---
    try:
        window = MainWindow(config, translator)
        window.show()
        logger.info("Application started (data dir: %s)", data_dir)
    except Exception as e:
        logger.critical("Failed to initialize main window: %s", traceback.format_exc())
        print(f"FATAL: {e}", file=sys.stderr)
        return 1

    # --- Run event loop ---
    exit_code = app.exec()

    logger.info("Application exited with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
