"""
VisionInspect - Logging Setup
Logging terstruktur dengan rotating file handler.
Tiga log file terpisah: app, plc, inference.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_loggers: dict[str, logging.Logger] = {}
_initialized = False


def setup_logging(
    log_dir: Path,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Setup rotating file handlers untuk app, plc, dan inference.
    Juga setup console handler untuk development.
    """
    global _initialized
    if _initialized:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Format
    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)

    # --- File handler: app.log ---
    _add_file_handler(root, log_dir / "app.log", formatter, numeric_level, max_bytes, backup_count)

    # --- Console handler (stderr) ---
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # --- Named loggers ---
    for name in ("plc", "inference", "camera", "training", "api"):
        logger = logging.getLogger(name)
        logger.setLevel(numeric_level)
        logger.propagate = True  # Also goes to root
        _add_file_handler(
            logger, log_dir / f"{name}.log", formatter, numeric_level, max_bytes, backup_count
        )
        _loggers[name] = logger

    _initialized = True
    logging.info("Logging initialized: level=%s, dir=%s", level, log_dir)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Falls back to root logger."""
    return _loggers.get(name) or logging.getLogger(name)


def _add_file_handler(
    logger: logging.Logger,
    path: Path,
    formatter: logging.Formatter,
    level: int,
    max_bytes: int,
    backup_count: int,
) -> None:
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
