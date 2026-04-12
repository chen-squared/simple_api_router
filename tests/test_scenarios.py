"""
Scenario-based integration tests for simple_api_router.

Simulates realistic user flows exercising:
1. Daily-request quota exhaustion → sequential fallback chain (api1→api2→api3)
2. RPM limit → fallback to next API in group
3. Dollar budget exhaustion → fallback to next API
4. Concurrent requests with RPM ceiling
5. Per-5h request quota fallback
6. Weekly request quota fallback
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict

import httpx
import uvicorn
import yaml

ROOT = Path(__file__).parent.parent
MOCK_PORT = 19999
ROUTER_PORT = 18080

# -----------------------------------------------------------------------
# Server management (same pattern as test_router.py)
# -----------------------------------------------------------------------

def _run_mock_server():
    sys.path.insert(0, str(ROOT))
    os.environ["MOCK_PORT"] = str(MOCK_PORT)
    from mock_api.server import app
    uvicorn.run(app, host="127.0.0.1", port=MOCK_PORT, log_level="error")


def _run_router_server(config_path: str):
    sys.path.insert(0, str(ROOT))
    from router.config import load_config
    from router.app import create_app
    from router.logger import setup_logging
    config = load_config(config_path)
    setup_logging("WARNING")
    app = create_app(config)
    uvicorn.run(app, host="127.0.0.1", port=ROUTER_PORT, log_level="error")


def _wait_for_server(url: str, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Server at {url} did not start within {timeout}s")


def _write_test_config(config: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(config, f)


_mock_proc: multiprocessing.Process | None = None


def _ensure_mock_server():
    global _mock_proc
    if _mock_proc is None or not _mock_proc.is_alive():
        _mock_proc = multiprocessing.Process(target=_run_mock_server, daemon=True)
        _mock_proc.start()
        _wait_for_server(f"http://127.0.0.1:{MOCK_PORT}/health")


MOCK_BASE_V1 = f"http://127.0.0.1:{MOCK_PORT}/v1"
MOCK_BASE = f"http://127.0.0.1:{MOCK_PORT}"

# -----------------------------------------------------------------------
# Base test class
# -----------------------------------------------------------------------

class RouterTestCase(unittest.TestCase):
    router_proc: multiprocessing.Process
    config_path: Path

    @classmethod
    def setUpClass(cls):
        _ensure_mock_server()
        cls.config_path = ROOT / "tests" / f"test_config_{cls.__name__}.yaml"
        cls._write_config()
        cls.router_proc = multiprocessing.Process(
            target=_run_router_server, args=(str(cls.config_path),), daemon=True
        )
        cls.router_proc.start()
        _wait_for_server(f"http://127.0.0.1:{ROUTER_PORT}/health")

    @classmethod
    def tearDownClass(cls):
        if cls.router_proc.is_alive():
            cls.router_proc.terminate()
            cls.router_proc.join(timeout=3)
        if cls.config_path.exists():
            cls.config_path.unlink()
        time.sleep(0.5)

    @classmethod
    def _write_config(cls):
        raise NotImplementedError

    def client(self) -> httpx.Client:
        return httpx.Client(base_url=f"http://127.0.0.1:{ROUTER_PORT}", timeout=30.0)


# -----------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------

def _openai_api(base_v1: str, extra: dict | None = None) -> dict:
    cfg = {
        "base_url": base_v1,
        "api_key": "test-key",
        "type": "openai",
        # No retries so availability-based skipping is clean and fast
        "retry": {"max_retries": 0, "cooldown_after": 100, "cooldown_duration": 5,
                  "error_limits": {}},
        "usage": {"rpm": 100},
    }
    if extra:
        cfg.update(extra)
    return cfg


def _server_config(groups: dict, apis: dict, default_group: str = "main") -> dict:
    return {
        "server": {"host": "127.0.0.1", "port": ROUTER_PORT, "log_level": "WARNING",
                   "log_file": None},
        "default_group": default_group,
        "apis": apis,
        "groups": groups,
    }


CHAT_REQ = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
}

ANTHROPIC_REQ = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}],
}

# -----------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------

def _find_api_usage(node: dict, api_id: str) -> dict | None:
    """Recursively search the stats tree for a given api_id's usage stats."""
    if node.get("api_id") == api_id:
        return node.get("usage", {})
    for member in node.get("members", []):
        result = _find_api_usage(member, api_id)
        if result is not None:
            return result
    return None


def _api_usage(client: httpx.Client, api_id: str) -> dict:
    stats = client.get("/stats").json()
    usage = _find_api_usage(stats.get("tree", {}), api_id)
    return usage or {}


# =======================================================================
# Scenario 1: Daily-request quota exhaustion → fallback chain
# =======================================================================

class TestDailyRequestsFallback(RouterTestCase):
    """
    api1 has daily_requests=1, api2 has daily_requests=1, api3 is unlimited.
    Three sequential requests should be spread across api1 → api2 → api3.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {"usage": {"daily_requests": 1, "rpm": 100}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"daily_requests": 1, "rpm": 100}}),
                "api3": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [
                        {"api": "api1"}, {"api": "api2"}, {"api": "api3"},
                    ],
                },
            },
        ), cls.config_path)

    def test_fallback_chain_daily_exhaustion(self):
        with self.client() as c:
            # Request 1 → api1 (0 < 1 → available)
            r1 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r1.status_code, 200, r1.text)

            # Request 2 → api2 (api1 at daily limit: 1 >= 1)
            r2 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r2.status_code, 200, r2.text)

            # Request 3 → api3 (api1 & api2 both at daily limit)
            r3 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r3.status_code, 200, r3.text)

            # Verify distribution via stats
            u1 = _api_usage(c, "api1")
            u2 = _api_usage(c, "api2")
            u3 = _api_usage(c, "api3")

        self.assertEqual(u1["total_requests"], 1,
                         "api1 should have handled exactly 1 request")
        self.assertEqual(u2["total_requests"], 1,
                         "api2 should have handled exactly 1 request")
        self.assertEqual(u3["total_requests"], 1,
                         "api3 should have handled exactly 1 request")
        self.assertEqual(u1["daily_requests"], 1,
                         "api1 daily counter should be 1")
        self.assertEqual(u2["daily_requests"], 1,
                         "api2 daily counter should be 1")

    def test_stats_reflect_quota_state(self):
        """Stats endpoint should show daily request counters accurately."""
        with self.client() as c:
            # Make requests to exhaust api1 and api2 again (independent router process)
            c.post("/v1/chat/completions", json=CHAT_REQ)
            c.post("/v1/chat/completions", json=CHAT_REQ)
            stats_raw = c.get("/stats").json()

        # Tree structure sanity check
        self.assertIn("tree", stats_raw)
        self.assertIn("members", stats_raw["tree"])
        member_ids = [m["api_id"] for m in stats_raw["tree"]["members"]]
        self.assertIn("api1", member_ids)
        self.assertIn("api2", member_ids)


# =======================================================================
# Scenario 2: RPM limit → fallback
# =======================================================================

class TestRPMLimitFallback(RouterTestCase):
    """
    api1 has rpm=2. After 2 requests api1's sliding window is full.
    The 3rd sequential request should fall back to api2.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 2}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def test_rpm_limit_triggers_fallback(self):
        with self.client() as c:
            # Requests 1 & 2 → api1 (within rpm=2)
            r1 = c.post("/v1/chat/completions", json=CHAT_REQ)
            r2 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r2.status_code, 200)

            # Request 3 → api2 (api1 RPM window full: 2 >= 2)
            r3 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r3.status_code, 200)

            u1 = _api_usage(c, "api1")
            u2 = _api_usage(c, "api2")

        self.assertEqual(u1["total_requests"], 2,
                         "api1 should have handled exactly 2 requests (rpm=2)")
        self.assertEqual(u2["total_requests"], 1,
                         "api2 should have handled exactly 1 request (overflow from rpm limit)")

    def test_rpm_sliding_window_tracks_correctly(self):
        """rpm_current in stats should reflect window size."""
        with self.client() as c:
            c.post("/v1/chat/completions", json=CHAT_REQ)
            u1 = _api_usage(c, "api1")
        # After 1 request, rpm_current should be 1
        self.assertGreaterEqual(u1["rpm_current"], 1)


# =======================================================================
# Scenario 3: Budget exhaustion → fallback
# =======================================================================

class TestBudgetFallback(RouterTestCase):
    """
    api1 has a tiny daily dollar budget ($0.000015).
    Mock API returns prompt_tokens=10, completion_tokens=8.
    At $1/1M each: cost = $0.000018 per request.
    After the first request the budget is exceeded ($0.000018 >= $0.000015).
    The second request should fall back to api2.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {
                    "usage": {
                        "rpm": 100,
                        "budget": {
                            "input_price_per_1m": 1.0,
                            "output_price_per_1m": 1.0,
                            "daily": 0.000015,   # $0.000015 limit
                        },
                    },
                }),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def test_budget_exhaustion_triggers_fallback(self):
        with self.client() as c:
            # Request 1 → api1 (budget not yet exceeded: 0 < $0.000015)
            r1 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r1.status_code, 200, r1.text)

            # After request 1, tokens recorded:
            #   cost = (10+8)/1M * $1 = $0.000018 >= $0.000015 → budget exceeded
            # Request 2 → api2 (api1 budget exceeded)
            r2 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r2.status_code, 200, r2.text)

            u1 = _api_usage(c, "api1")
            u2 = _api_usage(c, "api2")

        self.assertEqual(u1["total_requests"], 1,
                         "api1 should only handle 1 request before budget exceeded")
        self.assertEqual(u2["total_requests"], 1,
                         "api2 should handle request after api1 budget exhausted")
        # Verify budget spend was recorded
        budget = u1.get("budget", {})
        self.assertGreater(budget.get("daily_usd", 0), 0,
                           "api1 should have recorded non-zero daily budget spend")

    def test_budget_spend_reported_in_stats(self):
        """Stats endpoint reports budget spend per period."""
        with self.client() as c:
            c.post("/v1/chat/completions", json=CHAT_REQ)
            u1 = _api_usage(c, "api1")
        budget = u1.get("budget", {})
        self.assertIn("daily_usd", budget)
        self.assertIn("weekly_usd", budget)
        self.assertIn("monthly_usd", budget)
        self.assertGreater(budget["daily_usd"], 0)


# =======================================================================
# Scenario 4: Concurrent requests with RPM ceiling
# =======================================================================

class TestConcurrentRPM(RouterTestCase):
    """
    api1 has rpm=5, api2 is unlimited.
    Sending 10 concurrent requests: api1 should serve at most 5,
    the remainder spill over to api2.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 5}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def _send_one(self) -> int:
        with httpx.Client(base_url=f"http://127.0.0.1:{ROUTER_PORT}", timeout=30.0) as c:
            r = c.post("/v1/chat/completions", json=CHAT_REQ)
            return r.status_code

    def test_concurrent_requests_all_succeed_and_distribute(self):
        """
        All 10 concurrent requests must succeed (200).
        api1 (rpm=5) handles at most 5; overflow must go to api2.
        """
        with self.client() as c:
            u1_before = _api_usage(c, "api1")
            u2_before = _api_usage(c, "api2")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(self._send_one) for _ in range(10)]
            results = [f.result() for f in futures]

        self.assertEqual(results.count(200), 10,
                         "All 10 concurrent requests should succeed")

        with self.client() as c:
            u1_after = _api_usage(c, "api1")
            u2_after = _api_usage(c, "api2")

        delta1 = u1_after["total_requests"] - u1_before["total_requests"]
        delta2 = u2_after["total_requests"] - u2_before["total_requests"]
        total = delta1 + delta2

        self.assertEqual(total, 10,
                         "All 10 requests must be accounted for across both APIs")
        self.assertLessEqual(delta1, 5,
                             "api1 must not exceed rpm=5 requests in this batch")
        self.assertGreaterEqual(delta2, 5,
                                "api2 should receive the overflow (at least 5 requests)")


# =======================================================================
# Scenario 5: Per-5h request quota fallback
# =======================================================================

class TestPer5hRequestsFallback(RouterTestCase):
    """api1 has per_5h_requests=1 → second request goes to api2."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1,
                                    {"usage": {"rpm": 100, "per_5h_requests": 1}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def test_per5h_quota_triggers_fallback(self):
        with self.client() as c:
            u1_before = _api_usage(c, "api1")
            u2_before = _api_usage(c, "api2")

            r1 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r1.status_code, 200)

            r2 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r2.status_code, 200)

            u1_after = _api_usage(c, "api1")
            u2_after = _api_usage(c, "api2")

        delta1 = u1_after["total_requests"] - u1_before["total_requests"]
        delta2 = u2_after["total_requests"] - u2_before["total_requests"]

        self.assertGreaterEqual(delta1, 0)
        self.assertEqual(delta1 + delta2, 2, "Both requests should be accounted for")
        # api1 can handle at most per_5h_requests=1 in the current window
        used_already = u1_before["per_5h_requests"]
        self.assertLessEqual(u1_after["per_5h_requests"], 1,
                             "api1 per_5h counter should not exceed 1")

    def test_per5h_counter_in_stats(self):
        with self.client() as c:
            c.post("/v1/chat/completions", json=CHAT_REQ)
            u1 = _api_usage(c, "api1")
        self.assertEqual(u1["per_5h_requests"], 1)


# =======================================================================
# Scenario 6: Weekly request quota fallback
# =======================================================================

class TestWeeklyRequestsFallback(RouterTestCase):
    """api1 has weekly_requests=1 → second request goes to api2."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1,
                                    {"usage": {"rpm": 100, "weekly_requests": 1}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def test_weekly_quota_triggers_fallback(self):
        with self.client() as c:
            u1_before = _api_usage(c, "api1")
            u2_before = _api_usage(c, "api2")

            r1 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r1.status_code, 200)

            r2 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r2.status_code, 200)

            u1_after = _api_usage(c, "api1")
            u2_after = _api_usage(c, "api2")

        delta1 = u1_after["total_requests"] - u1_before["total_requests"]
        delta2 = u2_after["total_requests"] - u2_before["total_requests"]

        self.assertEqual(delta1 + delta2, 2, "Both requests should be accounted for")
        self.assertLessEqual(u1_after["weekly_requests"], 1,
                             "api1 weekly counter should not exceed 1")

    def test_weekly_counter_in_stats(self):
        with self.client() as c:
            c.post("/v1/chat/completions", json=CHAT_REQ)
            u1 = _api_usage(c, "api1")
        self.assertEqual(u1["weekly_requests"], 1)


# =======================================================================
# Scenario 7: Budget set to $0.0 → immediately unavailable
# =======================================================================

class TestBudgetZeroImmediate(RouterTestCase):
    """
    api1 has daily budget=$0.0. Since cost(0 tokens) = 0.0 >= 0.0,
    api1 is considered over-budget immediately and every request falls to api2.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {
                    "usage": {
                        "rpm": 100,
                        "budget": {
                            "input_price_per_1m": 1.0,
                            "output_price_per_1m": 1.0,
                            "daily": 0.0,
                        },
                    },
                }),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def test_zero_budget_skips_api1_immediately(self):
        with self.client() as c:
            r = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r.status_code, 200)

            u1 = _api_usage(c, "api1")
            u2 = _api_usage(c, "api2")

        self.assertEqual(u1["total_requests"], 0,
                         "api1 with $0 budget should never be called")
        self.assertEqual(u2["total_requests"], 1,
                         "api2 should handle the request when api1 has zero budget")


# =======================================================================
# Scenario 8: All APIs exhausted → 503
# =======================================================================

class TestAllAPIsExhausted(RouterTestCase):
    """All APIs have daily_requests=1; after each is used once, further requests → 503."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100, "daily_requests": 1}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100, "daily_requests": 1}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def test_503_when_all_exhausted(self):
        with self.client() as c:
            # Use up both APIs
            r1 = c.post("/v1/chat/completions", json=CHAT_REQ)
            r2 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r2.status_code, 200)

            # Third request: both exhausted → 503
            r3 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r3.status_code, 503,
                             "Router should return 503 when all APIs are exhausted")

            # 503 response must be JSON
            self.assertEqual(r3.headers["content-type"], "application/json")
            data = r3.json()
            self.assertIn("detail", data)


# =======================================================================
# Scenario 9: Anthropic-format requests with daily fallback
# =======================================================================

class TestAnthropicFormatDailyFallback(RouterTestCase):
    """
    Same as Scenario 1 but using Anthropic-format (/v1/messages) requests.
    api1 daily_requests=1, api2 daily_requests=1, api3 unlimited.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {"usage": {"daily_requests": 1, "rpm": 100}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"daily_requests": 1, "rpm": 100}}),
                "api3": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [
                        {"api": "api1"}, {"api": "api2"}, {"api": "api3"},
                    ],
                },
            },
        ), cls.config_path)

    def test_anthropic_format_fallback_chain(self):
        """Anthropic-format requests trigger the same quota-based fallback."""
        with self.client() as c:
            # router converts Anthropic → OpenAI for the openai-type backends
            r1 = c.post("/v1/messages", json=ANTHROPIC_REQ)
            r2 = c.post("/v1/messages", json=ANTHROPIC_REQ)
            r3 = c.post("/v1/messages", json=ANTHROPIC_REQ)

        self.assertEqual(r1.status_code, 200, r1.text)
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r3.status_code, 200, r3.text)

        with self.client() as c:
            u1 = _api_usage(c, "api1")
            u2 = _api_usage(c, "api2")
            u3 = _api_usage(c, "api3")

        self.assertEqual(u1["total_requests"], 1)
        self.assertEqual(u2["total_requests"], 1)
        self.assertEqual(u3["total_requests"], 1)


# =======================================================================
# Scenario 10: Combined RPM + daily_requests limits
# =======================================================================

class TestCombinedLimits(RouterTestCase):
    """
    api1 has rpm=3 AND daily_requests=2.
    After 2 requests, daily limit is hit even though rpm would allow more.
    The 3rd request should go to api2.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1,
                                    {"usage": {"rpm": 3, "daily_requests": 2}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def test_daily_limit_takes_precedence_over_rpm(self):
        with self.client() as c:
            r1 = c.post("/v1/chat/completions", json=CHAT_REQ)
            r2 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r2.status_code, 200)

            # 3rd request: daily_requests=2 exhausted (even though rpm=3 would allow it)
            r3 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r3.status_code, 200)

            u1 = _api_usage(c, "api1")
            u2 = _api_usage(c, "api2")

        self.assertEqual(u1["total_requests"], 2,
                         "api1 should handle exactly 2 requests (daily_requests=2)")
        self.assertEqual(u2["total_requests"], 1,
                         "api2 handles 3rd request after api1 daily limit exhausted")


# =======================================================================
# Round 2 – Scenario 11: Streaming requests with quota fallback
# =======================================================================

class TestStreamingQuotaFallback(RouterTestCase):
    """
    api1 has daily_requests=1.
    A non-streaming request uses up api1's quota.
    The subsequent STREAMING request must fall back to api2.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {"usage": {"daily_requests": 1, "rpm": 100}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def test_streaming_request_respects_quota_and_increments_counter(self):
        """
        Validates streaming behaviour in one ordered test:
        1. Streaming request → api1 (quota not yet used); daily counter increments.
        2. Non-streaming request → api2 (api1 quota exhausted).
        """
        with self.client() as c:
            u1_before = _api_usage(c, "api1")
            u2_before = _api_usage(c, "api2")

            # First request (streaming) → api1 (daily_reqs = 0 < 1)
            stream_req = dict(CHAT_REQ, stream=True)
            with c.stream("POST", "/v1/chat/completions", json=stream_req) as r1:
                self.assertEqual(r1.status_code, 200)
                chunks = list(r1.iter_lines())
            self.assertGreater(len(chunks), 0, "Streaming response must have content")

            # Second request (non-streaming) → api2 (api1 daily quota exhausted)
            r2 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r2.status_code, 200)

            u1_after = _api_usage(c, "api1")
            u2_after = _api_usage(c, "api2")

        d1 = u1_after["total_requests"] - u1_before["total_requests"]
        d2 = u2_after["total_requests"] - u2_before["total_requests"]

        self.assertEqual(d1, 1, "api1 should handle exactly the first (streaming) request")
        self.assertEqual(d2, 1, "api2 should handle the second request after quota exhausted")

        # Confirm streaming request incremented the daily counter
        daily_delta = u1_after["daily_requests"] - u1_before["daily_requests"]
        self.assertEqual(daily_delta, 1, "Streaming request must count toward daily_requests")


# =======================================================================
# Round 2 – Scenario 12: Retry attempts count toward request quota
# =======================================================================

class TestRetryCountsTowardQuota(RouterTestCase):
    """
    Each failed retry attempt increments the daily_requests counter.
    api1 always returns 500; with error_limits={500:2} (up to 2 retries per call)
    and daily_requests=2, a single user request exhausts the quota via two
    failed attempts, forcing the NEXT user request to fall back to api2.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": {
                    "base_url": MOCK_BASE,
                    "api_key": "test-key",
                    "type": "openai",
                    # Always returns 500
                    "endpoint_path": "/v1/chat/completions/error/500",
                    "retry": {
                        "max_retries": 2,
                        "cooldown_after": 100,   # high threshold: don't trigger cooldown
                        "cooldown_duration": 5,
                        "error_limits": {500: 2},  # retry 500 up to 2 times per call
                    },
                    "usage": {"rpm": 100, "daily_requests": 2},
                },
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def test_retry_failures_exhaust_daily_quota(self):
        """
        User request 1: api1 makes 2 HTTP attempts (both 500) → daily_reqs = 2 → UpstreamError
                         → falls back to api2 → user sees 200.
        User request 2: api1 daily_reqs = 2 >= 2 → skipped immediately → api2 handles.
        """
        with self.client() as c:
            # First user request: api1 fails with retries → api2 succeeds
            r1 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r1.status_code, 200, r1.text)

            u1_mid = _api_usage(c, "api1")
            # api1 should have made exactly 2 HTTP attempts (both failed)
            self.assertEqual(u1_mid["total_requests"], 2,
                             "api1 should record 2 failed attempts (1 original + 1 retry)")
            self.assertEqual(u1_mid["daily_requests"], 2,
                             "daily_requests should equal 2 after two failed attempts")
            self.assertEqual(u1_mid["total_success"], 0, "api1 had no successes")

            # Second user request: api1 quota exhausted → api2 handles immediately
            r2 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r2.status_code, 200, r2.text)

            u1_final = _api_usage(c, "api1")
            u2_final = _api_usage(c, "api2")

        # api1 still has 2 total requests (was skipped for request 2)
        self.assertEqual(u1_final["total_requests"], 2)
        # api2 handled both user requests (as fallback for both)
        self.assertEqual(u2_final["total_requests"], 2)


# =======================================================================
# Round 2 – Scenario 13: High-concurrency RPM ceiling
# =======================================================================

class TestHighConcurrencyRPM(RouterTestCase):
    """
    api1 has rpm=10, api2 is unlimited.
    20 concurrent requests: api1 must serve exactly 10 (atomic RPM claim),
    the remaining 10 overflow to api2.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 10}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def _send_one(self) -> int:
        with httpx.Client(base_url=f"http://127.0.0.1:{ROUTER_PORT}", timeout=30.0) as c:
            return c.post("/v1/chat/completions", json=CHAT_REQ).status_code

    def test_high_concurrency_all_succeed(self):
        """All 20 concurrent requests must succeed."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(self._send_one) for _ in range(20)]
            results = [f.result() for f in futures]

        self.assertEqual(results.count(200), 20, "All 20 requests must succeed")

    def test_high_concurrency_respects_rpm(self):
        """api1 must not exceed rpm=10; exactly 10 overflow to api2."""
        with self.client() as c:
            u1_before = _api_usage(c, "api1")
            u2_before = _api_usage(c, "api2")

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(self._send_one) for _ in range(20)]
            [f.result() for f in futures]

        with self.client() as c:
            u1_after = _api_usage(c, "api1")
            u2_after = _api_usage(c, "api2")

        d1 = u1_after["total_requests"] - u1_before["total_requests"]
        d2 = u2_after["total_requests"] - u2_before["total_requests"]

        self.assertEqual(d1 + d2, 20, "All 20 requests must be accounted for")
        self.assertLessEqual(d1, 10, "api1 must not exceed rpm=10")
        self.assertGreaterEqual(d2, 10, "api2 should receive at least 10 overflow requests")


# =======================================================================
# Round 2 – Scenario 14: Monthly budget exhaustion → fallback
# =======================================================================

class TestBudgetMonthlyFallback(RouterTestCase):
    """
    api1 has a tiny monthly budget ($0.000015).
    Mock returns 10 input + 8 output tokens; cost = $0.000018 per request.
    After the first request the monthly budget is exceeded → fallback to api2.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {
                    "usage": {
                        "rpm": 100,
                        "budget": {
                            "input_price_per_1m": 1.0,
                            "output_price_per_1m": 1.0,
                            "monthly": 0.000015,
                        },
                    },
                }),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def test_monthly_budget_triggers_fallback(self):
        with self.client() as c:
            r1 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r1.status_code, 200)

            r2 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r2.status_code, 200)

            u1 = _api_usage(c, "api1")
            u2 = _api_usage(c, "api2")

        self.assertEqual(u1["total_requests"], 1)
        self.assertEqual(u2["total_requests"], 1)
        budget = u1.get("budget", {})
        self.assertGreater(budget.get("monthly_usd", 0), 0)


# =======================================================================
# Round 2 – Scenario 15: Load-balance group with RPM-exhausted member
# =======================================================================

class TestLoadBalanceRPMExhaustion(RouterTestCase):
    """
    load_balance group: api1 (rpm=3), api2 (unlimited).
    Once api1's rpm=3 slots fill, all subsequent requests must go to api2.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 3}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "load_balance",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def _send_one(self) -> int:
        with httpx.Client(base_url=f"http://127.0.0.1:{ROUTER_PORT}", timeout=30.0) as c:
            return c.post("/v1/chat/completions", json=CHAT_REQ).status_code

    def test_load_balance_respects_rpm_exhaustion(self):
        """After api1 rpm=3 slots are used, all further requests go to api2."""
        # Send 10 requests (mix of concurrent to get some distribution)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(self._send_one) for _ in range(10)]
            results = [f.result() for f in futures]

        self.assertEqual(results.count(200), 10, "All 10 requests must succeed")

        with self.client() as c:
            u1 = _api_usage(c, "api1")
            u2 = _api_usage(c, "api2")

        total = u1["total_requests"] + u2["total_requests"]
        self.assertEqual(total, 10)
        self.assertLessEqual(u1["total_requests"], 3,
                             "api1 (rpm=3) should handle at most 3 requests")


# =======================================================================
# Round 3 – Scenario 16: Nested sequential groups
# =======================================================================

class TestNestedGroupFallback(RouterTestCase):
    """
    main (sequential):
      inner_group (sequential): api1 (daily=1), api2 (daily=1)
      api3 (unlimited)

    After api1 and api2 each handle one request, inner_group is fully
    exhausted.  main must then skip inner_group and use api3.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {"usage": {"daily_requests": 1, "rpm": 100}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"daily_requests": 1, "rpm": 100}}),
                "api3": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "inner_group": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
                "main": {
                    "strategy": "sequential",
                    "members": [{"group": "inner_group"}, {"api": "api3"}],
                },
            },
        ), cls.config_path)

    def test_nested_group_fallback(self):
        """After inner_group is exhausted, requests reach api3 at the top level."""
        with self.client() as c:
            # Request 1 → inner_group → api1
            r1 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r1.status_code, 200)

            # Request 2 → inner_group → api2 (api1 exhausted)
            r2 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r2.status_code, 200)

            # Request 3 → api3 (inner_group exhausted)
            r3 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r3.status_code, 200)

            u1 = _api_usage(c, "api1")
            u2 = _api_usage(c, "api2")
            u3 = _api_usage(c, "api3")

        self.assertEqual(u1["total_requests"], 1, "api1 handled 1 request via inner_group")
        self.assertEqual(u2["total_requests"], 1, "api2 handled 1 request via inner_group")
        self.assertEqual(u3["total_requests"], 1, "api3 handled request after inner_group exhausted")

    def test_nested_stats_navigation(self):
        """Stats helper navigates nested group tree to find API stats."""
        with self.client() as c:
            c.post("/v1/chat/completions", json=CHAT_REQ)
            stats_raw = c.get("/stats").json()

        tree = stats_raw["tree"]
        # Outer group is "main"
        self.assertEqual(tree["group_id"], "main")
        # First member should be inner_group
        inner = tree["members"][0]
        self.assertIn("group_id", inner, "First member of main should be a group")
        self.assertEqual(inner["group_id"], "inner_group")
        # Inner group's members should be api1 and api2
        inner_member_ids = [m["api_id"] for m in inner["members"]]
        self.assertIn("api1", inner_member_ids)
        self.assertIn("api2", inner_member_ids)

        # _find_api_usage helper should traverse the nested structure
        u3 = _find_api_usage(tree, "api3")
        self.assertIsNotNone(u3, "_find_api_usage must locate api3 in nested tree")


# =======================================================================
# Round 3 – Scenario 17: Budget exceeded w/o 429 → no automatic cooldown
# =======================================================================

class TestBudgetNoAutoCooldown(RouterTestCase):
    """
    When the dollar budget is exceeded by normal successful requests
    (without a concurrent upstream 429), no time-based cooldown timer
    should be set.  The endpoint is merely marked 'budget exceeded' on
    each availability check; budget_blocked_until must stay at 0.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {
                    "usage": {
                        "rpm": 100,
                        "budget": {
                            "input_price_per_1m": 1.0,
                            "output_price_per_1m": 1.0,
                            "daily": 0.000015,
                        },
                    },
                }),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def test_no_cooldown_timer_after_budget_exceeded(self):
        """budget_blocked_until must be 0 when budget is exceeded via normal usage."""
        with self.client() as c:
            # First request exhausts the budget
            c.post("/v1/chat/completions", json=CHAT_REQ)

            # Second request falls back (budget exceeded, but no cooldown timer)
            c.post("/v1/chat/completions", json=CHAT_REQ)

            u1 = _api_usage(c, "api1")

        # No cooldown timer set — only budget_exceeded from dynamic check
        self.assertEqual(u1["budget_blocked_until"], 0.0,
                         "budget_blocked_until must be 0 when exceeded via normal (non-429) usage")
        # Confirm budget is actually exceeded (not just 0)
        budget = u1.get("budget", {})
        self.assertGreater(budget.get("daily_usd", 0), 0.000015,
                           "daily budget must be exceeded after first request")


# =======================================================================
# Round 3 – Scenario 18: RPM counts one slot per call(), not per retry
# =======================================================================

class TestRPMOneSlotPerCall(RouterTestCase):
    """
    With the atomic RPM fix, try_claim_rpm_slot() is called exactly ONCE per
    call() invocation.  Even if the call retries (say, 3 total HTTP attempts
    each returning 500), only ONE RPM slot is consumed.

    Config: api1 always returns 500 (endpoint_path=error/500) with
    max_retries=2 (3 total HTTP attempts per call()); rpm=10; api2 fallback.

    After 2 user requests (each triggering 3 HTTP attempts on api1),
    rpm_current must be 2, not 6.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": {
                    "base_url": MOCK_BASE,
                    "api_key": "test-key",
                    "type": "openai",
                    "endpoint_path": "/v1/chat/completions/error/500",
                    "retry": {
                        "max_retries": 2,
                        "cooldown_after": 100,
                        "cooldown_duration": 5,
                        "error_limits": {500: 2},
                    },
                    "usage": {"rpm": 10},
                },
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {
                    "strategy": "sequential",
                    "members": [{"api": "api1"}, {"api": "api2"}],
                },
            },
        ), cls.config_path)

    def test_rpm_consumed_once_per_logical_request(self):
        """
        Two user requests, each hitting the 500-error retry limit (2 HTTP attempts
        per call due to error_limits={500:2}), should consume only 2 RPM slots.

        Total HTTP attempts: 2 attempts × 2 user requests = 4.
        But RPM slots claimed: 1 slot × 2 user requests = 2.
        """
        with self.client() as c:
            # Both succeed via api2 fallback
            r1 = c.post("/v1/chat/completions", json=CHAT_REQ)
            r2 = c.post("/v1/chat/completions", json=CHAT_REQ)
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r2.status_code, 200)

            u1 = _api_usage(c, "api1")

        # error_limits={500:2} means 2 HTTP attempts per call (1 retry)
        # 2 attempts × 2 user requests = 4 total_requests recorded
        self.assertEqual(u1["total_requests"], 4,
                         "2 HTTP attempts × 2 user requests = 4 total_requests")

        # But only 2 RPM slots were claimed (one per call() invocation)
        self.assertEqual(u1["rpm_current"], 2,
                         "rpm_current must be 2 (one slot per call(), not per retry attempt)")


if __name__ == "__main__":
    unittest.main()
