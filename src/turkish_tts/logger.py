"""Shared logger, writing to the console and optionally to a rotating file."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "turkish_tts"
FORMAT = "[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s:%(funcName)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5


def setup_logger() -> logging.Logger:
    """Configures the logger once, honouring TTS_LOG_LEVEL and TTS_LOG_DIR."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    level = os.getenv("TTS_LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_dir = os.getenv("TTS_LOG_DIR")
    if log_dir:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        log_file = directory / "turkish_tts.log"
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log dosyasi: {log_file}")

    return logger


def get_logger() -> logging.Logger:
    """Returns the shared logger, configuring it on first use."""
    return setup_logger()
