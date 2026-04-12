"""Usage tracking with sliding windows and hard limits."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Optional

from router.config import UsageConfig
from router.logger import get_logger

log = get_logger("usage")


class UsageTracker:
    """
    Tracks RPM (sliding window) and period-based token usage.
    Also tracks total request count (success + failure).
    """

    def __init__(self, api_id: str, cfg: UsageConfig) -> None:
        self.api_id = api_id
        self.cfg = cfg
        self._lock = asyncio.Lock()

        # Sliding window for RPM: deque of request timestamps
        self._rpm_window: deque[float] = deque()

        # Period counters: (tokens, reset_timestamp)
        self._daily_tokens: int = 0
        self._daily_reset: float = self._next_reset(86400)
        self._per5h_tokens: int = 0
        self._per5h_reset: float = self._next_reset(18000)
        self._weekly_tokens: int = 0
        self._weekly_reset: float = self._next_reset(604800)

        # Total request counters (stats only, never reset)
        self.total_requests: int = 0
        self.total_success: int = 0
        self.total_failure: int = 0

        # No-retry cooldown after usage exceeded + 429
        self._no_retry_until: float = 0.0

    @staticmethod
    def _next_reset(period_seconds: float) -> float:
        return time.time() + period_seconds

    def _refresh_periods(self) -> None:
        now = time.time()
        if now >= self._daily_reset:
            self._daily_tokens = 0
            self._daily_reset = now + 86400
        if now >= self._per5h_reset:
            self._per5h_tokens = 0
            self._per5h_reset = now + 18000
        if now >= self._weekly_reset:
            self._weekly_tokens = 0
            self._weekly_reset = now + 604800

    def _prune_rpm_window(self) -> None:
        now = time.time()
        while self._rpm_window and now - self._rpm_window[0] > 60.0:
            self._rpm_window.popleft()

    async def is_rate_limited(self) -> bool:
        """Returns True if the RPM limit is currently exceeded."""
        if self.cfg.rpm is None:
            return False
        async with self._lock:
            self._prune_rpm_window()
            return len(self._rpm_window) >= self.cfg.rpm

    async def is_usage_exceeded(self) -> bool:
        """Returns True if any hard usage limit is exceeded."""
        async with self._lock:
            self._refresh_periods()
            if self.cfg.daily and self._daily_tokens >= self.cfg.daily:
                return True
            if self.cfg.per_5h and self._per5h_tokens >= self.cfg.per_5h:
                return True
            if self.cfg.weekly and self._weekly_tokens >= self.cfg.weekly:
                return True
            return False

    async def is_blocked(self) -> bool:
        """Returns True if in the no-retry cooldown after usage+429."""
        return time.time() < self._no_retry_until

    async def check_available(self) -> tuple[bool, str]:
        """Returns (available, reason). Checks all usage limits."""
        if await self.is_blocked():
            remaining = self._no_retry_until - time.time()
            return False, f"usage cooldown ({remaining:.0f}s remaining)"
        if await self.is_rate_limited():
            return False, "RPM limit exceeded"
        if await self.is_usage_exceeded():
            return False, "usage quota exceeded"
        return True, ""

    async def record_request(self, success: bool, tokens_used: int = 0) -> None:
        """Record a completed request."""
        async with self._lock:
            now = time.time()
            self.total_requests += 1
            if success:
                self.total_success += 1
            else:
                self.total_failure += 1

            # RPM sliding window
            self._rpm_window.append(now)

            # Add tokens to period counters
            if tokens_used > 0:
                self._refresh_periods()
                self._daily_tokens += tokens_used
                self._per5h_tokens += tokens_used
                self._weekly_tokens += tokens_used

        log.debug(
            "[%s] request recorded success=%s tokens=%d total=%d",
            self.api_id, success, tokens_used, self.total_requests,
        )

    async def on_usage_exceeded_429(self) -> None:
        """Called when a 429 is received while usage is exceeded."""
        duration = self.cfg.no_retry_duration
        self._no_retry_until = time.time() + duration
        log.warning(
            "[%s] usage exceeded + 429: blocking for %ds",
            self.api_id, duration,
        )

    def stats(self) -> dict:
        return {
            "api_id": self.api_id,
            "total_requests": self.total_requests,
            "total_success": self.total_success,
            "total_failure": self.total_failure,
            "daily_tokens": self._daily_tokens,
            "per_5h_tokens": self._per5h_tokens,
            "weekly_tokens": self._weekly_tokens,
            "rpm_current": len(self._rpm_window),
            "no_retry_until": self._no_retry_until,
        }
