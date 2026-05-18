"""Logging setup."""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from typing import Optional


def _running_under_systemd() -> bool:
    """Return True when running as a systemd unit.

    systemd sets JOURNAL_STREAM and INVOCATION_ID in the unit environment.
    When either is present the journal already prepends a timestamp to every
    line, so we skip adding one ourselves.
    """
    return bool(os.environ.get("JOURNAL_STREAM") or os.environ.get("INVOCATION_ID"))


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure root logger with console and optional file handler."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Under systemd/journald the journal already prepends a timestamp;
    # omit it here to avoid double-timestamps on Linux.
    if _running_under_systemd():
        fmt = logging.Formatter("%(levelname)-8s | %(name)s | %(message)s")
    else:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        file_handler = TimedRotatingFileHandler(
            log_file, when="midnight", backupCount=90, encoding="utf-8", utc=False,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    return logging.getLogger("router")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"router.{name}")
