"""Usage tracking: request-count limits (rpm/daily/per_5h/weekly) and token-based budget."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Optional, Tuple

from router.config import BudgetConfig, UsageConfig
from router.logger import get_logger

log = get_logger("usage")


class UsageTracker:
    """
    Tracks:
    - RPM  : sliding-window request rate
    - Request-count limits: daily / per-5h / weekly (not tokens)
    - Token-based budget:   daily / weekly / monthly spend in USD
    - Total request stats (success + failure, never reset)
    """

    def __init__(self, api_id: str, cfg: UsageConfig) -> None:
        self.api_id = api_id
        self.cfg = cfg
        self._lock = asyncio.Lock()

        # ── RPM sliding window ──────────────────────────────────────────
        self._rpm_window: deque[float] = deque()

        # ── Request-count period counters ───────────────────────────────
        self._daily_reqs: int = 0
        self._daily_req_reset: float = _next_reset(86_400)

        self._per5h_reqs: int = 0
        self._per5h_req_reset: float = _next_reset(18_000)

        self._weekly_reqs: int = 0
        self._weekly_req_reset: float = _next_reset(604_800)

        # No-retry cooldown after request-quota exceeded + 429
        self._req_quota_blocked_until: float = 0.0

        # ── Token counters for budget ───────────────────────────────────
        self._daily_input_tokens: int = 0
        self._daily_output_tokens: int = 0
        self._daily_token_reset: float = _next_reset(86_400)

        self._weekly_input_tokens: int = 0
        self._weekly_output_tokens: int = 0
        self._weekly_token_reset: float = _next_reset(604_800)

        self._monthly_input_tokens: int = 0
        self._monthly_output_tokens: int = 0
        self._monthly_token_reset: float = _next_reset(2_592_000)

        # No-retry cooldown after budget exceeded + 429
        self._budget_blocked_until: float = 0.0

        # ── Lifetime stats (never reset) ────────────────────────────────
        self.total_requests: int = 0
        self.total_success: int = 0
        self.total_failure: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    # ────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────

    def _refresh_req_periods(self) -> None:
        now = time.time()
        if now >= self._daily_req_reset:
            self._daily_reqs = 0
            self._daily_req_reset = now + 86_400
        if now >= self._per5h_req_reset:
            self._per5h_reqs = 0
            self._per5h_req_reset = now + 18_000
        if now >= self._weekly_req_reset:
            self._weekly_reqs = 0
            self._weekly_req_reset = now + 604_800

    def _refresh_token_periods(self) -> None:
        now = time.time()
        if now >= self._daily_token_reset:
            self._daily_input_tokens = 0
            self._daily_output_tokens = 0
            self._daily_token_reset = now + 86_400
        if now >= self._weekly_token_reset:
            self._weekly_input_tokens = 0
            self._weekly_output_tokens = 0
            self._weekly_token_reset = now + 604_800
        if now >= self._monthly_token_reset:
            self._monthly_input_tokens = 0
            self._monthly_output_tokens = 0
            self._monthly_token_reset = now + 2_592_000

    def _prune_rpm_window(self) -> None:
        now = time.time()
        while self._rpm_window and now - self._rpm_window[0] > 60.0:
            self._rpm_window.popleft()

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute USD cost for the given token counts."""
        b = self.cfg.budget
        if b is None:
            return 0.0
        return (input_tokens / 1_000_000) * b.input_price_per_1m + \
               (output_tokens / 1_000_000) * b.output_price_per_1m

    # ────────────────────────────────────────────────────────────────────
    # Public checks
    # ────────────────────────────────────────────────────────────────────

    async def is_rate_limited(self) -> bool:
        """True if the RPM limit is currently exceeded (read-only, non-atomic check)."""
        if self.cfg.rpm is None:
            return False
        async with self._lock:
            self._prune_rpm_window()
            return len(self._rpm_window) >= self.cfg.rpm

    async def try_claim_rpm_slot(self) -> bool:
        """
        Atomically check the RPM window AND claim a slot if available.

        This is the only authoritative gate for RPM limiting. Unlike
        ``is_rate_limited()`` (which is a read-only hint), this method both
        checks and records in a single lock acquisition, preventing the
        check-then-act race when many coroutines make concurrent requests.

        Returns True if the slot was successfully claimed (request may proceed),
        False if the RPM limit is already reached.

        Callers must NOT also call ``record_request()``'s RPM path; the slot
        is recorded here. ``record_request()`` still handles period counters
        and budget tracking.
        """
        if self.cfg.rpm is None:
            return True
        async with self._lock:
            self._prune_rpm_window()
            if len(self._rpm_window) >= self.cfg.rpm:
                return False
            self._rpm_window.append(time.time())
            return True

    async def is_request_quota_exceeded(self) -> bool:
        """True if any request-count limit (daily/per_5h/weekly) is exceeded."""
        async with self._lock:
            self._refresh_req_periods()
            c = self.cfg
            if c.daily_requests and self._daily_reqs >= c.daily_requests:
                return True
            if c.per_5h_requests and self._per5h_reqs >= c.per_5h_requests:
                return True
            if c.weekly_requests and self._weekly_reqs >= c.weekly_requests:
                return True
            return False

    async def is_budget_exceeded(self) -> bool:
        """True if any token-cost budget (daily/weekly/monthly) is exceeded."""
        b = self.cfg.budget
        if b is None:
            return False
        async with self._lock:
            self._refresh_token_periods()
            if b.daily is not None:
                cost = self._estimate_cost(self._daily_input_tokens, self._daily_output_tokens)
                if cost >= b.daily:
                    return True
            if b.weekly is not None:
                cost = self._estimate_cost(self._weekly_input_tokens, self._weekly_output_tokens)
                if cost >= b.weekly:
                    return True
            if b.monthly is not None:
                cost = self._estimate_cost(self._monthly_input_tokens, self._monthly_output_tokens)
                if cost >= b.monthly:
                    return True
            return False

    async def is_req_quota_blocked(self) -> bool:
        return time.time() < self._req_quota_blocked_until

    async def is_budget_blocked(self) -> bool:
        return time.time() < self._budget_blocked_until

    async def check_available(self) -> Tuple[bool, str]:
        """Returns (available, reason). Checks all limits."""
        if await self.is_req_quota_blocked():
            remaining = self._req_quota_blocked_until - time.time()
            return False, f"request-quota cooldown ({remaining:.0f}s remaining)"
        if await self.is_budget_blocked():
            remaining = self._budget_blocked_until - time.time()
            return False, f"budget cooldown ({remaining:.0f}s remaining)"
        if await self.is_rate_limited():
            return False, "RPM limit exceeded"
        if await self.is_request_quota_exceeded():
            return False, "request quota exceeded"
        if await self.is_budget_exceeded():
            return False, "budget exceeded"
        return True, ""

    # Keep backward-compat alias
    async def is_usage_exceeded(self) -> bool:
        return await self.is_request_quota_exceeded()

    async def is_blocked(self) -> bool:
        return await self.is_req_quota_blocked() or await self.is_budget_blocked()

    # ────────────────────────────────────────────────────────────────────
    # Recording
    # ────────────────────────────────────────────────────────────────────

    async def record_request(
        self,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        # Legacy arg kept for backward compat (treated as output tokens)
        tokens_used: int = 0,
    ) -> None:
        """Record a completed request (success or failure).

        Note: RPM slot is NOT recorded here.  It is claimed atomically by
        ``try_claim_rpm_slot()`` before the HTTP request is dispatched.
        """
        if tokens_used and not output_tokens:
            output_tokens = tokens_used  # backward compat

        async with self._lock:
            now = time.time()  # noqa: F841  (kept for readability)
            self.total_requests += 1
            if success:
                self.total_success += 1
            else:
                self.total_failure += 1

            # Request-count period tracking (both success and failure count)
            self._refresh_req_periods()
            self._daily_reqs += 1
            self._per5h_reqs += 1
            self._weekly_reqs += 1

            # Token tracking for budget (only on success with known token counts)
            if success and (input_tokens or output_tokens):
                self._refresh_token_periods()
                self._daily_input_tokens += input_tokens
                self._daily_output_tokens += output_tokens
                self._weekly_input_tokens += input_tokens
                self._weekly_output_tokens += output_tokens
                self._monthly_input_tokens += input_tokens
                self._monthly_output_tokens += output_tokens
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

        log.debug(
            "[%s] request recorded success=%s in=%d out=%d total=%d",
            self.api_id, success, input_tokens, output_tokens, self.total_requests,
        )

    # ────────────────────────────────────────────────────────────────────
    # Cooldown triggers
    # ────────────────────────────────────────────────────────────────────

    async def on_request_quota_exceeded_429(self) -> None:
        """Block after request-quota exceeded + 429."""
        duration = self.cfg.no_retry_duration
        self._req_quota_blocked_until = time.time() + duration
        log.warning(
            "[%s] request quota exceeded + 429: blocking for %ds",
            self.api_id, duration,
        )

    async def on_usage_exceeded_429(self) -> None:
        """Alias for backward compat."""
        await self.on_request_quota_exceeded_429()

    async def on_budget_exceeded_429(self) -> None:
        """Block after budget exceeded + 429."""
        b = self.cfg.budget
        duration = b.no_retry_duration if b else self.cfg.no_retry_duration
        self._budget_blocked_until = time.time() + duration
        log.warning(
            "[%s] budget exceeded + 429: blocking for %ds",
            self.api_id, duration,
        )

    # ────────────────────────────────────────────────────────────────────
    # Stats
    # ────────────────────────────────────────────────────────────────────

    def current_budget_spend(self) -> dict:
        """Current estimated spend in USD per period."""
        if self.cfg.budget is None:
            return {}
        return {
            "daily_usd": round(self._estimate_cost(
                self._daily_input_tokens, self._daily_output_tokens), 6),
            "weekly_usd": round(self._estimate_cost(
                self._weekly_input_tokens, self._weekly_output_tokens), 6),
            "monthly_usd": round(self._estimate_cost(
                self._monthly_input_tokens, self._monthly_output_tokens), 6),
            "total_usd": round(self._estimate_cost(
                self.total_input_tokens, self.total_output_tokens), 6),
        }

    def stats(self) -> dict:
        return {
            "api_id": self.api_id,
            "total_requests": self.total_requests,
            "total_success": self.total_success,
            "total_failure": self.total_failure,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            # Request-count periods
            "daily_requests": self._daily_reqs,
            "per_5h_requests": self._per5h_reqs,
            "weekly_requests": self._weekly_reqs,
            # RPM
            "rpm_current": len(self._rpm_window),
            # Budget
            "budget": self.current_budget_spend(),
            # Cooldown timestamps
            "req_quota_blocked_until": self._req_quota_blocked_until,
            "budget_blocked_until": self._budget_blocked_until,
        }


def _next_reset(period_seconds: float) -> float:
    return time.time() + period_seconds
