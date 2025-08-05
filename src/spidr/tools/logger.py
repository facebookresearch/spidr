# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import ClassVar


class ColoredFormatter(logging.Formatter):
    COLORS: ClassVar[dict[str, str]] = {
        "NOTSET": "\033[0m",  # No color
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"  # Reset color code

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return f"{self.COLORS[record.levelname]}{message}{self.RESET}"


def init_logger(file: str | Path | None = None, *, with_colors: bool = False) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    template = "[spidr] %(asctime)s - %(levelname)s - %(name)s - %(message)s"
    formatter = (ColoredFormatter if with_colors else logging.Formatter)(template)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.DEBUG)
    stdout.setFormatter(formatter)
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(stdout)
    logger.addHandler(stderr)

    if file is not None:
        handler = logging.FileHandler(file, "a")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
