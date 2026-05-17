"""Per-request usage logging to a daily-rotated JSONL file.

Each request to /v1/messages produces one JSON line:
  {"ts": "2026-05-17T12:34:56Z", "model": "anthropic/claude-opus-4-5",
   "provider": "anthropic", "backend_model": "claude-opus-4-5",
   "input_tokens": 1234, "output_tokens": 456,
   "cache_read_tokens": 0, "cache_write_tokens": 0,
   "streaming": false, "status": 200, "duration_ms": 312}

The file is written to <log_dir>/router.usage.jsonl and rotated nightly to
router.usage.jsonl.YYYY-MM-DD (90 days of history kept by default).
"""
from __future__ import annotations

import json
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

_usage_logger: Optional[logging.Logger] = None
_usage_log_path: Optional[str] = None


def setup_usage_logging(main_log_file: str, backup_count: int = 90) -> None:
    """Set up the JSONL usage logger alongside *main_log_file*.

    Creates ``<dir>/router.usage.jsonl`` in the same directory as the main
    log file, rotated daily.  Safe to call multiple times (idempotent).
    """
    global _usage_logger, _usage_log_path

    p = Path(main_log_file).expanduser()
    usage_path = p.parent / "router.usage.jsonl"
    _usage_log_path = str(usage_path)

    handler = TimedRotatingFileHandler(
        _usage_log_path,
        when="midnight",
        backupCount=backup_count,
        encoding="utf-8",
        utc=False,
    )
    # Each record is already a complete JSON string — no extra formatting needed.
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("router.usage")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't bleed into the main log
    logger.handlers.clear()
    logger.addHandler(handler)
    _usage_logger = logger


def log_usage(record: dict) -> None:
    """Append one usage record as a JSON line.  No-op when not configured."""
    if _usage_logger is None:
        return
    try:
        _usage_logger.info(json.dumps(record, ensure_ascii=False))
    except Exception:
        pass


def get_usage_log_path() -> Optional[str]:
    """Return the active usage log path, or None if not configured."""
    return _usage_log_path
