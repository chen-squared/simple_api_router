"""Retry and cooldown tracking per API."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Dict

from router.config import RetryConfig
from router.logger import get_logger

log = get_logger("retry")


class RetryTracker:
    """
    Tracks per-request error counts and global cooldown state.

    - error_counts: per HTTP error code counts in the current request attempt
    - total_failures: rolling count of failures (resets on success)
    - cooldown_until: timestamp when the API is available again
    """

    def __init__(self, api_id: str, cfg: RetryConfig) -> None:
        self.api_id = api_id
        self.cfg = cfg
        self._lock = asyncio.Lock()
        self._total_failures: int = 0  # consecutive failures
        self._cooldown_until: float = 0.0

    async def is_in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    async def cooldown_remaining(self) -> float:
        return max(0.0, self._cooldown_until - time.time())

    async def on_success(self) -> None:
        """Reset consecutive failure count on success."""
        async with self._lock:
            self._total_failures = 0
        log.debug("[%s] retry tracker: success, failures reset", self.api_id)

    async def on_failure(self, status_code: int) -> None:
        """
        Record a failure. If total failures exceed cooldown_after, enter cooldown.
        Returns whether cooldown was triggered.
        """
        async with self._lock:
            self._total_failures += 1
            count = self._total_failures
            threshold = self.cfg.cooldown_after

        log.debug(
            "[%s] failure status=%d consecutive=%d threshold=%d",
            self.api_id, status_code, count, threshold,
        )

        if count >= threshold:
            async with self._lock:
                self._cooldown_until = time.time() + self.cfg.cooldown_duration
                self._total_failures = 0  # reset after entering cooldown
            log.warning(
                "[%s] entering cooldown for %ds after %d failures",
                self.api_id, self.cfg.cooldown_duration, count,
            )

    def should_retry_error(self, status_code: int, error_count: Dict[int, int]) -> bool:
        """
        Given how many times this error code has occurred in the current request,
        decide whether to retry.
        """
        limit = self.cfg.error_limits.get(status_code, self.cfg.max_retries)
        current = error_count.get(status_code, 0)
        if current >= limit:
            log.debug(
                "[%s] error %d reached limit %d, not retrying",
                self.api_id, status_code, limit,
            )
            return False
        return True

    def should_retry_general(self, total_retries: int) -> bool:
        """Check if we've hit the overall per-request retry max."""
        return total_retries < self.cfg.max_retries

    def stats(self) -> dict:
        return {
            "api_id": self.api_id,
            "consecutive_failures": self._total_failures,
            "cooldown_until": self._cooldown_until,
            "in_cooldown": time.time() < self._cooldown_until,
        }
