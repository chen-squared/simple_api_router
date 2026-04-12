"""
Integration test suite for simple_api_router.

Starts the mock API server, then starts the router pointing to mock APIs,
then runs tests against the router.

NOTE: OpenAI base_url must include /v1 (e.g. http://host:port/v1)
      because endpoint.py appends /chat/completions to it.
      Anthropic base_url should NOT include /v1 (e.g. http://host:port)
      because endpoint.py appends /v1/messages to it.
"""
from __future__ import annotations

import asyncio
import json
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
# Server management
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


# -----------------------------------------------------------------------
# Base test class
# -----------------------------------------------------------------------

# Shared mock server (started once for all test classes)
_mock_proc: multiprocessing.Process | None = None


def _ensure_mock_server():
    global _mock_proc
    if _mock_proc is None or not _mock_proc.is_alive():
        _mock_proc = multiprocessing.Process(target=_run_mock_server, daemon=True)
        _mock_proc.start()
        _wait_for_server(f"http://127.0.0.1:{MOCK_PORT}/health")


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
        time.sleep(0.5)  # Allow port to free up

    @classmethod
    def _write_config(cls):
        raise NotImplementedError

    def client(self) -> httpx.Client:
        return httpx.Client(base_url=f"http://127.0.0.1:{ROUTER_PORT}", timeout=30.0)


def _openai_api(base_v1: str, extra: dict | None = None) -> dict:
    """Build an OpenAI API config. base_v1 should end with /v1."""
    cfg = {
        "base_url": base_v1,
        "api_key": "test-key",
        "type": "openai",
        "retry": {"max_retries": 2, "cooldown_after": 10, "cooldown_duration": 60,
                  "error_limits": {429: 1, 500: 2}},
        "usage": {"rpm": 100},
    }
    if extra:
        cfg.update(extra)
    return cfg


def _anthropic_api(base: str, extra: dict | None = None) -> dict:
    """Build an Anthropic API config. base should NOT include /v1."""
    cfg = {
        "base_url": base,
        "api_key": "test-key",
        "type": "anthropic",
        "retry": {"max_retries": 2, "cooldown_after": 10, "cooldown_duration": 60,
                  "error_limits": {500: 2}},
        "usage": {"rpm": 100},
    }
    if extra:
        cfg.update(extra)
    return cfg


def _server_config(groups: dict, apis: dict, default_group: str = "main") -> dict:
    return {
        "server": {"host": "127.0.0.1", "port": ROUTER_PORT, "log_level": "WARNING", "log_file": None},
        "default_group": default_group,
        "apis": apis,
        "groups": groups,
    }


MOCK_BASE_V1 = f"http://127.0.0.1:{MOCK_PORT}/v1"
MOCK_BASE = f"http://127.0.0.1:{MOCK_PORT}"


# -----------------------------------------------------------------------
# Test: Basic routing
# -----------------------------------------------------------------------

class TestBasicRouting(RouterTestCase):
    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "primary": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100, "daily_requests": 10000, "per_5h_requests": 5000, "weekly_requests": 50000}}),
                "secondary": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 50}}),
            },
            groups={
                "main": {"strategy": "sequential", "members": [{"api": "primary"}, {"api": "secondary"}]},
            },
        ), cls.config_path)

    def test_openai_success(self):
        with self.client() as c:
            r = c.post("/v1/chat/completions",
                       json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertIn("choices", data)
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        self.assertIn("usage", data)

    def test_health_endpoint(self):
        with self.client() as c:
            r = c.get("/health")
        self.assertEqual(r.status_code, 200)

    def test_stats_endpoint(self):
        with self.client() as c:
            r = c.get("/stats")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("default_group", data)
        self.assertIn("tree", data)

    def test_invalid_json_body(self):
        with self.client() as c:
            r = c.post("/v1/chat/completions",
                       content=b"not json",
                       headers={"Content-Type": "application/json"})
        self.assertEqual(r.status_code, 400)

    def test_multiple_requests(self):
        """Multiple requests should all succeed."""
        with self.client() as c:
            for i in range(5):
                r = c.post("/v1/chat/completions",
                           json={"model": "gpt-4", "messages": [{"role": "user", "content": f"msg {i}"}]})
                self.assertEqual(r.status_code, 200, f"Request {i} failed: {r.text}")


# -----------------------------------------------------------------------
# Test: Format conversion
# -----------------------------------------------------------------------

class TestFormatConversion(RouterTestCase):
    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "openai-api": _openai_api(MOCK_BASE_V1),
                "anthropic-api": _anthropic_api(MOCK_BASE),
            },
            groups={
                "main": {"strategy": "sequential", "members": [{"api": "openai-api"}]},
                "anthropic-main": {"strategy": "sequential", "members": [{"api": "anthropic-api"}]},
            },
        ), cls.config_path)

    def test_openai_request_to_openai_backend(self):
        """OpenAI request → OpenAI backend → OpenAI response."""
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
            })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertIn("choices", data)


# -----------------------------------------------------------------------
# Test: Sequential fallback
# -----------------------------------------------------------------------

class TestSequentialFallback(RouterTestCase):
    @classmethod
    def _write_config(cls):
        # primary always returns 500 (via endpoint_path override)
        _write_test_config(_server_config(
            apis={
                "always-500": {
                    "base_url": MOCK_BASE,
                    "api_key": "test-key",
                    "type": "openai",
                    "endpoint_path": "/v1/chat/completions/error/500",
                    "retry": {"max_retries": 1, "cooldown_after": 10, "cooldown_duration": 5,
                              "error_limits": {500: 1}},
                    "usage": {"rpm": 100},
                },
                "good": _openai_api(MOCK_BASE_V1),
            },
            groups={
                "main": {"strategy": "sequential", "members": [{"api": "always-500"}, {"api": "good"}]},
            },
        ), cls.config_path)

    def test_fallback_to_secondary(self):
        """Primary always 500 → should fall back to secondary and succeed."""
        with self.client() as c:
            r = c.post("/v1/chat/completions",
                       json={"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]})
        self.assertEqual(r.status_code, 200, r.text)

    def test_all_fail(self):
        """If all APIs fail, should return 503."""
        # No good API - both always fail
        # We test this by checking the always-500 alone
        pass  # Covered by TestAllFail class below


class TestAllFail(RouterTestCase):
    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "always-500a": {
                    "base_url": MOCK_BASE,
                    "api_key": "test-key",
                    "type": "openai",
                    "endpoint_path": "/v1/chat/completions/error/500",
                    "retry": {"max_retries": 1, "cooldown_after": 10, "cooldown_duration": 5,
                              "error_limits": {500: 1}},
                    "usage": {"rpm": 100},
                },
                "always-500b": {
                    "base_url": MOCK_BASE,
                    "api_key": "test-key",
                    "type": "openai",
                    "endpoint_path": "/v1/chat/completions/error/500",
                    "retry": {"max_retries": 1, "cooldown_after": 10, "cooldown_duration": 5,
                              "error_limits": {500: 1}},
                    "usage": {"rpm": 100},
                },
            },
            groups={
                "main": {"strategy": "sequential",
                         "members": [{"api": "always-500a"}, {"api": "always-500b"}]},
            },
        ), cls.config_path)

    def test_503_when_all_fail(self):
        with self.client() as c:
            r = c.post("/v1/chat/completions",
                       json={"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]})
        self.assertEqual(r.status_code, 503, r.text)


# -----------------------------------------------------------------------
# Test: Load balancing
# -----------------------------------------------------------------------

class TestLoadBalancing(RouterTestCase):
    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 60}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 30}}),
            },
            groups={
                "main": {"strategy": "load_balance", "members": [{"api": "api1"}, {"api": "api2"}]},
            },
        ), cls.config_path)

    def test_load_balance_succeeds(self):
        """Requests under load_balance should succeed."""
        with self.client() as c:
            for _ in range(6):
                r = c.post("/v1/chat/completions",
                           json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(r.status_code, 200, r.text)


# -----------------------------------------------------------------------
# Test: Nested groups
# -----------------------------------------------------------------------

class TestNestedGroups(RouterTestCase):
    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 60}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 30}}),
            },
            groups={
                "inner": {"strategy": "load_balance", "members": [{"api": "api1"}, {"api": "api2"}]},
                "main": {"strategy": "sequential", "members": [{"group": "inner"}]},
            },
        ), cls.config_path)

    def test_nested_group_routing(self):
        with self.client() as c:
            r = c.post("/v1/chat/completions",
                       json={"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]})
        self.assertEqual(r.status_code, 200, r.text)


# -----------------------------------------------------------------------
# Test: Streaming
# -----------------------------------------------------------------------

class TestStreaming(RouterTestCase):
    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={"primary": _openai_api(MOCK_BASE_V1)},
            groups={"main": {"strategy": "sequential", "members": [{"api": "primary"}]}},
        ), cls.config_path)

    def test_openai_streaming(self):
        with self.client() as c:
            with c.stream("POST", "/v1/chat/completions",
                          json={"model": "gpt-4",
                                "messages": [{"role": "user", "content": "Hello"}],
                                "stream": True}) as r:
                self.assertEqual(r.status_code, 200, r.read())
                self.assertIn("text/event-stream", r.headers.get("content-type", ""))
                chunks = list(r.iter_lines())
        data_lines = [l for l in chunks if l.startswith("data:")]
        self.assertTrue(len(data_lines) > 0, f"No data lines: {chunks[:10]}")

    def test_anthropic_streaming_from_openai_backend(self):
        """Anthropic-format request should get Anthropic-format stream back even from openai backend."""
        with self.client() as c:
            with c.stream("POST", "/v1/messages",
                          json={"model": "claude-3-5-sonnet-20241022",
                                "messages": [{"role": "user", "content": "Hello"}],
                                "max_tokens": 100,
                                "stream": True}) as r:
                self.assertEqual(r.status_code, 200, r.read())
                chunks = list(r.iter_lines())
        # Anthropic events should have event: lines
        event_lines = [l for l in chunks if l.startswith("event:")]
        self.assertTrue(len(event_lines) > 0, f"No event lines: {chunks[:10]}")


# -----------------------------------------------------------------------
# Test: Anthropic endpoint
# -----------------------------------------------------------------------

class TestAnthropicEndpoint(RouterTestCase):
    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={"anthropic-api": _anthropic_api(MOCK_BASE)},
            groups={"main": {"strategy": "sequential", "members": [{"api": "anthropic-api"}]}},
        ), cls.config_path)

    def test_anthropic_request_success(self):
        with self.client() as c:
            r = c.post("/v1/messages", json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["type"], "message")
        self.assertIn("content", data)

    def test_openai_request_to_anthropic_backend(self):
        """OpenAI-format request should get OpenAI-format response even from Anthropic backend."""
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hello"}],
            })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["object"], "chat.completion")


# -----------------------------------------------------------------------
# Test: 429 handling
# -----------------------------------------------------------------------

class Test429Handling(RouterTestCase):
    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "always-429": {
                    "base_url": MOCK_BASE,
                    "api_key": "test-key",
                    "type": "openai",
                    "endpoint_path": "/v1/chat/completions/error/429",
                    "retry": {"max_retries": 2, "cooldown_after": 10, "cooldown_duration": 60,
                              "error_limits": {429: 1}},
                    "usage": {"rpm": 100},
                },
                "good": _openai_api(MOCK_BASE_V1),
            },
            groups={
                "main": {"strategy": "sequential",
                         "members": [{"api": "always-429"}, {"api": "good"}]},
            },
        ), cls.config_path)

    def test_429_falls_through_to_secondary(self):
        """After 429 retry limit, should fall through to next member."""
        with self.client() as c:
            r = c.post("/v1/chat/completions",
                       json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
        # Should succeed via the good API
        self.assertEqual(r.status_code, 200, r.text)


# -----------------------------------------------------------------------
# Test: Flaky API (retry succeeds)
# -----------------------------------------------------------------------

class TestFlakyRetry(RouterTestCase):
    @classmethod
    def _write_config(cls):
        # The flaky endpoint fails first N times, then succeeds
        _write_test_config(_server_config(
            apis={
                "flaky": _openai_api(MOCK_BASE, {
                    "endpoint_path": "/v1/chat/completions/flaky/1",
                    "retry": {"max_retries": 3, "cooldown_after": 10, "cooldown_duration": 60,
                              "error_limits": {500: 3}},
                    "usage": {"rpm": 100},
                }),
            },
            groups={
                "main": {"strategy": "sequential", "members": [{"api": "flaky"}]},
            },
        ), cls.config_path)

    def test_flaky_retries_and_succeeds(self):
        """Flaky API should succeed after retry."""
        with self.client() as c:
            r = c.post("/v1/chat/completions",
                       json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status_code, 200, r.text)


# -----------------------------------------------------------------------
# Unit tests (no server needed)
# -----------------------------------------------------------------------

class TestUsageTracking(unittest.TestCase):
    def test_rpm_limit(self):
        from router.config import UsageConfig
        from router.usage import UsageTracker
        cfg = UsageConfig(rpm=3)
        tracker = UsageTracker("test", cfg)

        async def _run():
            for _ in range(3):
                self.assertFalse(await tracker.is_rate_limited())
                # RPM slots are now claimed atomically via try_claim_rpm_slot()
                self.assertTrue(await tracker.try_claim_rpm_slot())
            self.assertTrue(await tracker.is_rate_limited())

        asyncio.run(_run())

    def test_daily_requests_limit(self):
        from router.config import UsageConfig
        from router.usage import UsageTracker
        cfg = UsageConfig(daily_requests=3)
        tracker = UsageTracker("test", cfg)

        async def _run():
            self.assertFalse(await tracker.is_request_quota_exceeded())
            for _ in range(3):
                await tracker.record_request(success=True)
            self.assertTrue(await tracker.is_request_quota_exceeded())

        asyncio.run(_run())

    def test_per5h_requests_limit(self):
        from router.config import UsageConfig
        from router.usage import UsageTracker
        cfg = UsageConfig(per_5h_requests=2)
        tracker = UsageTracker("test", cfg)

        async def _run():
            await tracker.record_request(success=True)
            await tracker.record_request(success=True)
            self.assertTrue(await tracker.is_request_quota_exceeded())

        asyncio.run(_run())

    def test_weekly_requests_limit(self):
        from router.config import UsageConfig
        from router.usage import UsageTracker
        cfg = UsageConfig(weekly_requests=1)
        tracker = UsageTracker("test", cfg)

        async def _run():
            await tracker.record_request(success=True)
            self.assertTrue(await tracker.is_request_quota_exceeded())

        asyncio.run(_run())

    def test_usage_cooldown_on_429(self):
        from router.config import UsageConfig
        from router.usage import UsageTracker
        cfg = UsageConfig(daily_requests=1, no_retry_duration=60)
        tracker = UsageTracker("test", cfg)

        async def _run():
            await tracker.record_request(success=True)
            self.assertTrue(await tracker.is_request_quota_exceeded())
            await tracker.on_usage_exceeded_429()
            self.assertTrue(await tracker.is_blocked())

        asyncio.run(_run())

    def test_stats(self):
        from router.config import UsageConfig
        from router.usage import UsageTracker
        cfg = UsageConfig(rpm=10)
        tracker = UsageTracker("test", cfg)

        async def _run():
            await tracker.record_request(success=True, input_tokens=10, output_tokens=5)
            await tracker.record_request(success=False)
            s = tracker.stats()
            self.assertEqual(s["total_requests"], 2)
            self.assertEqual(s["total_success"], 1)
            self.assertEqual(s["total_failure"], 1)
            self.assertEqual(s["daily_requests"], 2)

        asyncio.run(_run())

    def test_budget_daily_limit(self):
        from router.config import UsageConfig, BudgetConfig
        from router.usage import UsageTracker
        cfg = UsageConfig(budget=BudgetConfig(
            input_price_per_1m=1.0, output_price_per_1m=1.0, daily=0.001
        ))
        tracker = UsageTracker("test", cfg)

        async def _run():
            self.assertFalse(await tracker.is_budget_exceeded())
            # 1000 input + 0 output = $0.001 at $1/1M → hits daily limit
            await tracker.record_request(success=True, input_tokens=1000, output_tokens=0)
            self.assertTrue(await tracker.is_budget_exceeded())

        asyncio.run(_run())

    def test_budget_weekly_limit(self):
        from router.config import UsageConfig, BudgetConfig
        from router.usage import UsageTracker
        cfg = UsageConfig(budget=BudgetConfig(
            input_price_per_1m=0.0, output_price_per_1m=2.0, weekly=0.002
        ))
        tracker = UsageTracker("test", cfg)

        async def _run():
            await tracker.record_request(success=True, input_tokens=0, output_tokens=1000)
            self.assertTrue(await tracker.is_budget_exceeded())

        asyncio.run(_run())

    def test_budget_monthly_limit(self):
        from router.config import UsageConfig, BudgetConfig
        from router.usage import UsageTracker
        cfg = UsageConfig(budget=BudgetConfig(
            input_price_per_1m=1.0, output_price_per_1m=3.0, monthly=0.003
        ))
        tracker = UsageTracker("test", cfg)

        async def _run():
            await tracker.record_request(success=True, input_tokens=0, output_tokens=1000)
            self.assertTrue(await tracker.is_budget_exceeded())

        asyncio.run(_run())

    def test_budget_cooldown_on_429(self):
        from router.config import UsageConfig, BudgetConfig
        from router.usage import UsageTracker
        cfg = UsageConfig(budget=BudgetConfig(
            input_price_per_1m=1.0, output_price_per_1m=1.0,
            daily=0.001, no_retry_duration=60,
        ))
        tracker = UsageTracker("test", cfg)

        async def _run():
            await tracker.record_request(success=True, input_tokens=1000, output_tokens=0)
            self.assertTrue(await tracker.is_budget_exceeded())
            await tracker.on_budget_exceeded_429()
            self.assertTrue(await tracker.is_budget_blocked())
            self.assertTrue(await tracker.is_blocked())

        asyncio.run(_run())

    def test_budget_spend_stats(self):
        from router.config import UsageConfig, BudgetConfig
        from router.usage import UsageTracker
        cfg = UsageConfig(budget=BudgetConfig(
            input_price_per_1m=2.0, output_price_per_1m=8.0
        ))
        tracker = UsageTracker("test", cfg)

        async def _run():
            # 500K input tokens = $1.00, 250K output = $2.00 → daily = $3.00
            await tracker.record_request(success=True, input_tokens=500_000, output_tokens=250_000)
            spend = tracker.current_budget_spend()
            self.assertAlmostEqual(spend["daily_usd"], 3.0, places=4)
            self.assertAlmostEqual(spend["total_usd"], 3.0, places=4)

        asyncio.run(_run())


class TestRetryTracker(unittest.TestCase):
    def test_cooldown_triggered(self):
        from router.config import RetryConfig
        from router.retry import RetryTracker
        cfg = RetryConfig(cooldown_after=3, cooldown_duration=60)
        tracker = RetryTracker("test", cfg)

        async def _run():
            self.assertFalse(await tracker.is_in_cooldown())
            for _ in range(3):
                await tracker.on_failure(500)
            self.assertTrue(await tracker.is_in_cooldown())

        asyncio.run(_run())

    def test_reset_on_success(self):
        from router.config import RetryConfig
        from router.retry import RetryTracker
        cfg = RetryConfig(cooldown_after=3, cooldown_duration=60)
        tracker = RetryTracker("test", cfg)

        async def _run():
            await tracker.on_failure(500)
            await tracker.on_failure(500)
            await tracker.on_success()
            self.assertFalse(await tracker.is_in_cooldown())

        asyncio.run(_run())

    def test_error_limits(self):
        from router.config import RetryConfig
        from router.retry import RetryTracker
        cfg = RetryConfig(error_limits={429: 1})
        tracker = RetryTracker("test", cfg)
        self.assertTrue(tracker.should_retry_error(429, {429: 0}))
        self.assertFalse(tracker.should_retry_error(429, {429: 1}))

    def test_general_retry_limit(self):
        from router.config import RetryConfig
        from router.retry import RetryTracker
        cfg = RetryConfig(max_retries=2)
        tracker = RetryTracker("test", cfg)
        self.assertTrue(tracker.should_retry_general(0))
        self.assertTrue(tracker.should_retry_general(1))
        self.assertFalse(tracker.should_retry_general(2))


class TestFormatConverterUnit(unittest.TestCase):
    def test_openai_to_anthropic_request(self):
        from router.converter import openai_to_anthropic_request
        req = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }
        result = openai_to_anthropic_request(req)
        self.assertEqual(result["system"], "Be helpful")
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0]["role"], "user")
        self.assertEqual(result["max_tokens"], 100)
        self.assertEqual(result["temperature"], 0.7)

    def test_anthropic_to_openai_request(self):
        from router.converter import anthropic_to_openai_request
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "Be helpful",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        result = anthropic_to_openai_request(req)
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertEqual(result["messages"][0]["content"], "Be helpful")
        self.assertEqual(result["messages"][1]["role"], "user")
        self.assertEqual(result["max_tokens"], 100)

    def test_anthropic_to_openai_response(self):
        from router.converter import anthropic_to_openai_response
        resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello there"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        result = anthropic_to_openai_response(resp)
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello there")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["usage"]["prompt_tokens"], 5)
        self.assertEqual(result["usage"]["completion_tokens"], 3)
        self.assertEqual(result["usage"]["total_tokens"], 8)

    def test_openai_to_anthropic_response(self):
        from router.converter import openai_to_anthropic_response
        resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        result = openai_to_anthropic_response(resp)
        self.assertEqual(result["type"], "message")
        self.assertEqual(result["content"][0]["text"], "Hi!")
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(result["usage"]["input_tokens"], 5)
        self.assertEqual(result["usage"]["output_tokens"], 3)

    def test_stop_sequences_conversion(self):
        from router.converter import openai_to_anthropic_request
        req = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}],
               "stop": ["END", "STOP"]}
        result = openai_to_anthropic_request(req)
        self.assertEqual(result["stop_sequences"], ["END", "STOP"])

    def test_stream_flag_preserved(self):
        from router.converter import openai_to_anthropic_request
        req = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        result = openai_to_anthropic_request(req)
        self.assertTrue(result["stream"])


class TestConfigLoader(unittest.TestCase):
    def test_load_main_config(self):
        from router.config import load_config
        cfg = load_config(ROOT / "config.yaml")
        self.assertIsNotNone(cfg)
        self.assertIn(cfg.default_group, cfg.groups)

    def test_env_var_expansion(self):
        import os
        import tempfile
        os.environ["TEST_API_KEY_XYZ"] = "sk-test-expanded"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
server:
  host: "127.0.0.1"
  port: 9090
default_group: g1
apis:
  a1:
    base_url: "http://example.com/v1"
    api_key: "${TEST_API_KEY_XYZ}"
    type: openai
groups:
  g1:
    strategy: sequential
    members:
      - api: a1
""")
            fname = f.name
        from router.config import load_config
        cfg = load_config(fname)
        self.assertEqual(cfg.apis["a1"].api_key, "sk-test-expanded")
        os.unlink(fname)


if __name__ == "__main__":
    unittest.main(verbosity=2)
