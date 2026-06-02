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


class _StreamToLogger:
    """Redirect file-like writes (sys.stdout / sys.stderr) into a logger.

    Every non-empty line is emitted as a log record at *level*.  Used under
    launchd so that *all* process output — uvicorn startup, tracebacks, stray
    prints — flows through the same ``TimedRotatingFileHandler`` and inherits
    its daily rotation, rather than accumulating in a raw ``stdout.log``.
    """

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level

    def write(self, message: str) -> None:
        stripped = message.strip()
        if stripped:
            self._logger.log(self._level, stripped)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


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

    in_tty = sys.stdout.isatty()
    under_systemd = _running_under_systemd()

    # Console handler: always add it unless we are redirecting stdout to the
    # rotated file below (launchd with a log_file).  systemd always keeps it
    # because journald has its own rotation.
    if in_tty or under_systemd or not log_file:
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

        # Under launchd (macOS) — not a TTY and not systemd — stdout is
        # captured to ~/Library/Logs/.../stdout.log with no rotation.
        # Redirect sys.stdout / sys.stderr into the logger so that *all*
        # process output inherits the daily rotation above.
        if not in_tty and not under_systemd:
            sys.stdout = _StreamToLogger(root, logging.INFO)       # type: ignore[assignment]
            sys.stderr = _StreamToLogger(root, logging.ERROR)      # type: ignore[assignment]

    return logging.getLogger("router")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"router.{name}")
