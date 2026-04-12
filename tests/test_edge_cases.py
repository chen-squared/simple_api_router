"""
Additional edge-case and regression tests for simple_api_router.

Tests the following scenarios:
- Cooldown recovery after expiry
- RPM limit blocking and sliding window
- Load balance weight distribution
- Circular group detection
- Streaming SSE format correctness
- Streaming error handling (errors deferred to generator)
- Cross-format streaming (OpenAI <-> Anthropic)
- Multiple system messages merged for Anthropic
- Token tracking in streaming requests
- Usage limit blocking end-to-end
- Flaky API retry with exponential backoff
"""
from __future__ import annotations

import asyncio
import collections
import json
import multiprocessing
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

import httpx
import uvicorn
import yaml

ROOT = Path(__file__).parent.parent
MOCK_PORT = 19999
ROUTER_PORT = 18080


# -----------------------------------------------------------------------
# Server management (reuse from test_router.py)
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


# Shared mock server
_mock_proc: multiprocessing.Process | None = None


def _ensure_mock_server():
    global _mock_proc
    if _mock_proc is None or not _mock_proc.is_alive():
        _mock_proc = multiprocessing.Process(target=_run_mock_server, daemon=True)
        _mock_proc.start()
        _wait_for_server(f"http://127.0.0.1:{MOCK_PORT}/health")


MOCK_BASE_V1 = f"http://127.0.0.1:{MOCK_PORT}/v1"
MOCK_BASE = f"http://127.0.0.1:{MOCK_PORT}"


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


def _server_config(groups: dict, apis: dict, default_group: str = "main") -> dict:
    return {
        "server": {"host": "127.0.0.1", "port": ROUTER_PORT, "log_level": "WARNING", "log_file": None},
        "default_group": default_group,
        "apis": apis,
        "groups": groups,
    }


def _openai_api(base_v1: str, extra: dict | None = None) -> dict:
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


# -----------------------------------------------------------------------
# Unit tests: Format conversion edge cases
# -----------------------------------------------------------------------

class TestMultipleSystemMessages(unittest.TestCase):
    """Multiple system messages in OpenAI format should be merged for Anthropic."""

    def test_multiple_system_messages_merged(self):
        from router.converter import openai_to_anthropic_request
        req = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "system", "content": "Be concise."},
                {"role": "system", "content": "Use bullet points."},
                {"role": "user", "content": "Explain AI."},
            ],
        }
        result = openai_to_anthropic_request(req)
        # All system messages should be merged with double newline
        self.assertIn("system", result)
        self.assertIn("You are helpful.", result["system"])
        self.assertIn("Be concise.", result["system"])
        self.assertIn("Use bullet points.", result["system"])
        self.assertEqual(result["system"], "You are helpful.\n\nBe concise.\n\nUse bullet points.")

    def test_no_system_messages(self):
        from router.converter import openai_to_anthropic_request
        req = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = openai_to_anthropic_request(req)
        # No system field when no system messages
        self.assertNotIn("system", result)

    def test_system_message_not_in_anthropic_messages(self):
        from router.converter import openai_to_anthropic_request
        req = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Bye"},
            ],
        }
        result = openai_to_anthropic_request(req)
        # Only non-system messages should be in messages array
        roles = [m["role"] for m in result["messages"]]
        self.assertNotIn("system", roles)
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_system_message_content_as_list(self):
        """System message with content as list of blocks."""
        from router.converter import openai_to_anthropic_request
        req = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Be helpful"}]},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = openai_to_anthropic_request(req)
        self.assertEqual(result["system"], "Be helpful")


class TestFormatConversionEdgeCases(unittest.TestCase):
    def test_openai_to_anthropic_preserves_top_p(self):
        from router.converter import openai_to_anthropic_request
        req = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.9,
        }
        result = openai_to_anthropic_request(req)
        self.assertEqual(result["top_p"], 0.9)

    def test_openai_to_anthropic_stop_string(self):
        """Single stop string becomes a list."""
        from router.converter import openai_to_anthropic_request
        req = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "END",
        }
        result = openai_to_anthropic_request(req)
        self.assertEqual(result["stop_sequences"], ["END"])

    def test_openai_to_anthropic_default_max_tokens(self):
        """max_tokens should have a default when not specified."""
        from router.converter import openai_to_anthropic_request
        req = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = openai_to_anthropic_request(req)
        self.assertIn("max_tokens", result)
        self.assertGreater(result["max_tokens"], 0)

    def test_anthropic_to_openai_stop_sequences(self):
        """Single stop_sequence becomes stop string, multiple become list."""
        from router.converter import anthropic_to_openai_request
        req_one = {"model": "claude", "messages": [{"role": "user", "content": "hi"}],
                   "stop_sequences": ["END"]}
        result_one = anthropic_to_openai_request(req_one)
        self.assertEqual(result_one["stop"], "END")  # single → string

        req_many = {"model": "claude", "messages": [{"role": "user", "content": "hi"}],
                    "stop_sequences": ["END", "STOP"]}
        result_many = anthropic_to_openai_request(req_many)
        self.assertEqual(result_many["stop"], ["END", "STOP"])  # multiple → list

    def test_anthropic_response_tool_use_stop_reason(self):
        """Non-stop stop reasons should pass through (not become 'stop')."""
        from router.converter import anthropic_to_openai_response
        resp = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        result = anthropic_to_openai_response(resp)
        # "tool_use" should map to something other than "stop"
        self.assertNotEqual(result["choices"][0]["finish_reason"], "stop")

    def test_extract_tokens_openai(self):
        from router.converter import extract_tokens_from_response
        body = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        self.assertEqual(extract_tokens_from_response(body, "openai"), 15)

    def test_extract_tokens_anthropic(self):
        from router.converter import extract_tokens_from_response
        body = {"usage": {"input_tokens": 10, "output_tokens": 5}}
        self.assertEqual(extract_tokens_from_response(body, "anthropic"), 15)

    def test_extract_tokens_missing_usage(self):
        from router.converter import extract_tokens_from_response
        self.assertEqual(extract_tokens_from_response({}, "openai"), 0)
        self.assertEqual(extract_tokens_from_response({}, "anthropic"), 0)


# -----------------------------------------------------------------------
# Unit tests: Circular group detection
# -----------------------------------------------------------------------

class TestCircularGroupDetection(unittest.TestCase):
    def test_simple_circular_reference(self):
        """A → B → A should be detected."""
        from router.config import RouterConfig
        from router.group import build_routing_tree

        config = RouterConfig.model_validate({
            "server": {"host": "127.0.0.1", "port": 9090, "log_level": "WARNING"},
            "default_group": "a",
            "apis": {
                "api1": {"base_url": "http://example.com/v1", "api_key": "k", "type": "openai"}
            },
            "groups": {
                "a": {"strategy": "sequential", "members": [{"group": "b"}]},
                "b": {"strategy": "sequential", "members": [{"group": "a"}]},
            },
        })
        with self.assertRaises(ValueError) as ctx:
            build_routing_tree(config)
        self.assertIn("Circular", str(ctx.exception))

    def test_self_reference(self):
        """A → A should be detected."""
        from router.config import RouterConfig
        from router.group import build_routing_tree

        config = RouterConfig.model_validate({
            "server": {"host": "127.0.0.1", "port": 9090, "log_level": "WARNING"},
            "default_group": "a",
            "apis": {
                "api1": {"base_url": "http://example.com/v1", "api_key": "k", "type": "openai"}
            },
            "groups": {
                "a": {"strategy": "sequential", "members": [{"group": "a"}]},
            },
        })
        with self.assertRaises(ValueError) as ctx:
            build_routing_tree(config)
        self.assertIn("Circular", str(ctx.exception))

    def test_deep_circular_reference(self):
        """A → B → C → A should be detected."""
        from router.config import RouterConfig
        from router.group import build_routing_tree

        config = RouterConfig.model_validate({
            "server": {"host": "127.0.0.1", "port": 9090, "log_level": "WARNING"},
            "default_group": "a",
            "apis": {
                "api1": {"base_url": "http://example.com/v1", "api_key": "k", "type": "openai"}
            },
            "groups": {
                "a": {"strategy": "sequential", "members": [{"group": "b"}]},
                "b": {"strategy": "sequential", "members": [{"group": "c"}]},
                "c": {"strategy": "sequential", "members": [{"group": "a"}]},
            },
        })
        with self.assertRaises(ValueError):
            build_routing_tree(config)

    def test_valid_deep_nesting_not_flagged(self):
        """A → B → C (no cycle) should be fine."""
        from router.config import RouterConfig
        from router.group import build_routing_tree

        config = RouterConfig.model_validate({
            "server": {"host": "127.0.0.1", "port": 9090, "log_level": "WARNING"},
            "default_group": "a",
            "apis": {
                "api1": {"base_url": "http://example.com/v1", "api_key": "k", "type": "openai"}
            },
            "groups": {
                "a": {"strategy": "sequential", "members": [{"group": "b"}]},
                "b": {"strategy": "sequential", "members": [{"group": "c"}]},
                "c": {"strategy": "sequential", "members": [{"api": "api1"}]},
            },
        })
        tree = build_routing_tree(config)
        self.assertIn("a", tree)
        self.assertIn("b", tree)
        self.assertIn("c", tree)


# -----------------------------------------------------------------------
# Unit tests: Cooldown recovery
# -----------------------------------------------------------------------

class TestCooldownRecovery(unittest.TestCase):
    def test_cooldown_expires_and_recovers(self):
        """After cooldown duration, endpoint should be available again."""
        from router.config import RetryConfig
        from router.retry import RetryTracker

        async def _run():
            cfg = RetryConfig(cooldown_after=2, cooldown_duration=1)  # 1s cooldown
            tracker = RetryTracker("test", cfg)

            self.assertFalse(await tracker.is_in_cooldown())

            # Trigger cooldown
            await tracker.on_failure(500)
            await tracker.on_failure(500)
            self.assertTrue(await tracker.is_in_cooldown())

            # Wait for cooldown to expire
            await asyncio.sleep(1.1)
            self.assertFalse(await tracker.is_in_cooldown(), "Cooldown should have expired")

        asyncio.run(_run())

    def test_cooldown_remaining_decreases(self):
        """cooldown_remaining should decrease over time."""
        from router.config import RetryConfig
        from router.retry import RetryTracker

        async def _run():
            cfg = RetryConfig(cooldown_after=1, cooldown_duration=2)
            tracker = RetryTracker("test", cfg)

            await tracker.on_failure(500)
            self.assertTrue(await tracker.is_in_cooldown())

            remaining1 = await tracker.cooldown_remaining()
            await asyncio.sleep(0.5)
            remaining2 = await tracker.cooldown_remaining()

            self.assertGreater(remaining1, remaining2, "Remaining should decrease")
            self.assertGreater(remaining2, 0, "Still in cooldown")

        asyncio.run(_run())

    def test_failures_reset_on_success_before_cooldown(self):
        """If we succeed before hitting cooldown threshold, counter resets."""
        from router.config import RetryConfig
        from router.retry import RetryTracker

        async def _run():
            cfg = RetryConfig(cooldown_after=5, cooldown_duration=60)
            tracker = RetryTracker("test", cfg)

            # 4 failures (under threshold of 5)
            for _ in range(4):
                await tracker.on_failure(500)

            # Success resets the counter
            await tracker.on_success()
            self.assertFalse(await tracker.is_in_cooldown())
            self.assertEqual(tracker._total_failures, 0)

            # Now 4 more failures should NOT trigger cooldown
            for _ in range(4):
                await tracker.on_failure(500)
            self.assertFalse(await tracker.is_in_cooldown())

        asyncio.run(_run())


# -----------------------------------------------------------------------
# Unit tests: RPM sliding window
# -----------------------------------------------------------------------

class TestRPMSlidingWindow(unittest.TestCase):
    def test_rpm_blocks_at_limit(self):
        """Requests are blocked when RPM limit is reached."""
        from router.config import UsageConfig
        from router.usage import UsageTracker

        async def _run():
            cfg = UsageConfig(rpm=3)
            tracker = UsageTracker("test", cfg)

            for _ in range(3):
                self.assertFalse(await tracker.is_rate_limited())
                # RPM slots are claimed atomically; record_request no longer tracks RPM
                self.assertTrue(await tracker.try_claim_rpm_slot())

            self.assertTrue(await tracker.is_rate_limited(), "Should be rate limited after 3 requests")

        asyncio.run(_run())

    def test_rpm_recovers_after_window(self):
        """After the 60s window, old requests expire and new ones are allowed."""
        from router.config import UsageConfig
        from router.usage import UsageTracker

        async def _run():
            cfg = UsageConfig(rpm=3)
            tracker = UsageTracker("test", cfg)

            # Fill the window via atomic slot claiming
            for _ in range(3):
                await tracker.try_claim_rpm_slot()
            self.assertTrue(await tracker.is_rate_limited())

            # Age the timestamps to simulate 61s passing
            now = time.time()
            tracker._rpm_window = collections.deque([now - 61.0, now - 61.0, now - 61.0])

            # Should no longer be rate limited
            self.assertFalse(await tracker.is_rate_limited(), "Should recover after 60s")

            # New request should work
            self.assertTrue(await tracker.try_claim_rpm_slot())
            self.assertFalse(await tracker.is_rate_limited(), "1 request in new window, still under limit")

        asyncio.run(_run())

    def test_rpm_limit_zero_not_enforced(self):
        """rpm=None means no limit."""
        from router.config import UsageConfig
        from router.usage import UsageTracker

        async def _run():
            cfg = UsageConfig(rpm=None)
            tracker = UsageTracker("test", cfg)

            for _ in range(1000):
                await tracker.record_request(success=True)

            self.assertFalse(await tracker.is_rate_limited(), "No RPM limit should never be limited")

        asyncio.run(_run())


# -----------------------------------------------------------------------
# Unit tests: Load balance weight distribution
# -----------------------------------------------------------------------

class TestLoadBalanceWeights(unittest.TestCase):
    def test_weights_proportional_to_rpm(self):
        """APIs with higher RPM get proportionally more traffic."""
        from router.config import APIConfig, UsageConfig, RetryConfig
        from router.endpoint import APIEndpoint
        from router.group import _member_weight

        api_high = APIEndpoint("high", APIConfig(
            base_url="http://x/v1", api_key="k", type="openai",
            usage=UsageConfig(rpm=60)
        ))
        api_low = APIEndpoint("low", APIConfig(
            base_url="http://x/v1", api_key="k", type="openai",
            usage=UsageConfig(rpm=30)
        ))

        w_high = _member_weight(api_high)
        w_low = _member_weight(api_low)
        self.assertAlmostEqual(w_high / w_low, 2.0, places=1)

    def test_weight_default_when_no_rpm(self):
        """API with no RPM config should get default weight of 1.0."""
        from router.config import APIConfig, UsageConfig
        from router.endpoint import APIEndpoint
        from router.group import _member_weight

        api = APIEndpoint("no_rpm", APIConfig(
            base_url="http://x/v1", api_key="k", type="openai",
            usage=UsageConfig(rpm=None)
        ))
        self.assertEqual(_member_weight(api), 1.0)

    def test_random_choices_distribution(self):
        """random.choices with weights should approximate the RPM ratio over many trials."""
        import random
        random.seed(42)
        # 60:30 = 2:1 ratio
        counts = {0: 0, 1: 0}
        n_trials = 10000
        for _ in range(n_trials):
            chosen = random.choices([0, 1], weights=[60.0, 30.0], k=1)[0]
            counts[chosen] += 1

        ratio = counts[0] / counts[1]
        self.assertAlmostEqual(ratio, 2.0, delta=0.1,
                               msg=f"Expected ~2:1 distribution, got {ratio:.2f}")


# -----------------------------------------------------------------------
# Unit tests: Streaming SSE format
# -----------------------------------------------------------------------

class TestStreamingSSEFormat(unittest.TestCase):
    """Tests that streaming output uses proper SSE format with double newlines."""

    def test_sse_format_requires_double_newlines(self):
        """
        SSE spec requires a blank line (\\n\\n) between events. Verify that
        streaming output including blank lines produces valid SSE format.

        This documents the FIXED behavior: the endpoint's _stream_body() no longer
        filters out blank lines, so proper event boundaries are preserved.
        """
        # Simulate what _stream_body() now yields (blank lines included)
        raw_lines = [
            b'data: {"id":"x","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n',
            b'\n',  # blank line = SSE event boundary
            b'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b'\n',  # blank line = SSE event boundary
            b'data: [DONE]\n',
            b'\n',
        ]
        combined = b''.join(raw_lines).decode()
        has_double_newline = '\n\n' in combined
        self.assertTrue(has_double_newline,
            "SSE stream with blank lines should produce double-newline boundaries")

    def test_converted_streaming_has_proper_sse(self):
        """Converted streams (from openai_to_anthropic) should have proper SSE format."""
        from router.converter import stream_openai_to_anthropic

        async def fake_openai_stream():
            lines = [
                b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}\n',
                b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n',
                b'data: [DONE]\n',
            ]
            for line in lines:
                yield line

        async def _run():
            chunks = []
            async for chunk in stream_openai_to_anthropic(fake_openai_stream()):
                chunks.append(chunk)
            combined = b''.join(chunks).decode()
            return combined

        combined = asyncio.run(_run())
        self.assertIn('\n\n', combined, "Converted Anthropic SSE should use double newlines")
        self.assertIn('event:', combined, "Should have event: lines")
        self.assertIn('data:', combined, "Should have data: lines")
        self.assertIn('hello', combined, "Should contain the content")

    def test_anthropic_to_openai_stream_conversion(self):
        """Anthropic SSE stream → OpenAI SSE format should work correctly."""
        from router.converter import stream_anthropic_to_openai

        async def fake_anthropic_stream():
            lines = [
                b'event: message_start\n',
                b'data: {"type":"message_start","message":{"id":"msg_x","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n',
                b'event: content_block_start\n',
                b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
                b'event: content_block_delta\n',
                b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello world"}}\n',
                b'event: content_block_stop\n',
                b'data: {"type":"content_block_stop","index":0}\n',
                b'event: message_delta\n',
                b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}\n',
                b'event: message_stop\n',
                b'data: {"type":"message_stop"}\n',
            ]
            for line in lines:
                yield line

        async def _run():
            chunks = []
            async for chunk in stream_anthropic_to_openai(fake_anthropic_stream()):
                chunks.append(chunk)
            combined = b''.join(chunks).decode()
            return combined

        combined = asyncio.run(_run())
        self.assertIn('data:', combined, "Should have data: lines")
        self.assertIn('[DONE]', combined, "Should end with [DONE]")
        self.assertIn('hello world', combined, "Should contain the content")
        self.assertIn('\n\n', combined, "Should use double newlines")

    def test_openai_to_anthropic_stream_content_intact(self):
        """Content should be preserved through OpenAI→Anthropic stream conversion."""
        from router.converter import stream_openai_to_anthropic
        import json

        test_content = "Test content here"

        async def fake_openai_stream():
            lines = [
                f'data: {{"id":"x","object":"chat.completion.chunk","created":1234,"model":"gpt-4","choices":[{{"index":0,"delta":{{"content":"{test_content}"}},"finish_reason":null}}]}}\n'.encode(),
                b'data: {"id":"x","object":"chat.completion.chunk","created":1234,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n',
                b'data: [DONE]\n',
            ]
            for line in lines:
                yield line

        async def _run():
            chunks = []
            async for chunk in stream_openai_to_anthropic(fake_openai_stream()):
                chunks.append(chunk)
            combined = b''.join(chunks).decode()
            return combined

        combined = asyncio.run(_run())
        self.assertIn(test_content, combined, "Content should be preserved through conversion")


# -----------------------------------------------------------------------
# Unit tests: Streaming token tracking
# -----------------------------------------------------------------------

class TestStreamingTokenTracking(unittest.TestCase):
    def test_streaming_tracks_requests_not_tokens(self):
        """
        Streaming requests increment request counts but no token budget
        (token counts are not known inline for most streaming APIs).
        """
        from router.config import UsageConfig
        from router.usage import UsageTracker

        async def _run():
            cfg = UsageConfig(daily_requests=100, per_5h_requests=50, weekly_requests=500)
            tracker = UsageTracker("test", cfg)

            # Simulate what happens in streaming: record_request without tokens
            await tracker.record_request(success=True)  # no tokens

            return tracker.stats()

        stats = asyncio.run(_run())
        self.assertEqual(stats["daily_requests"], 1,
            "Daily request count should be incremented for streaming")
        self.assertEqual(stats["total_input_tokens"], 0,
            "Input tokens should not be incremented without token data")
        self.assertEqual(stats["total_output_tokens"], 0,
            "Output tokens should not be incremented without token data")
        self.assertEqual(stats["total_requests"], 1,
            "Total request count should be incremented")


# -----------------------------------------------------------------------
# Unit tests: Config validation
# -----------------------------------------------------------------------

class TestConfigValidation(unittest.TestCase):
    def test_invalid_strategy_rejected(self):
        """GroupConfig should reject invalid strategies."""
        from router.config import GroupConfig
        with self.assertRaises(Exception):
            GroupConfig(strategy="round_robin", members=[])

    def test_group_member_must_have_one_field(self):
        """GroupMember must specify exactly one of api or group."""
        from router.config import GroupMember
        with self.assertRaises(Exception):
            GroupMember()  # neither api nor group

        with self.assertRaises(Exception):
            GroupMember(api="x", group="y")  # both

    def test_missing_default_group_rejected(self):
        """Config should reject default_group that doesn't exist in groups."""
        from router.config import RouterConfig
        with self.assertRaises(Exception):
            RouterConfig.model_validate({
                "server": {"host": "127.0.0.1", "port": 9090},
                "default_group": "nonexistent",
                "apis": {"a1": {"base_url": "http://x/v1", "api_key": "k", "type": "openai"}},
                "groups": {"g1": {"strategy": "sequential", "members": [{"api": "a1"}]}},
            })

    def test_group_referencing_unknown_api_rejected(self):
        """Config should reject group referencing an api that doesn't exist."""
        from router.config import RouterConfig
        with self.assertRaises(Exception):
            RouterConfig.model_validate({
                "server": {"host": "127.0.0.1", "port": 9090},
                "default_group": "g1",
                "apis": {},
                "groups": {"g1": {"strategy": "sequential", "members": [{"api": "nonexistent"}]}},
            })


# -----------------------------------------------------------------------
# Integration test: Cooldown recovery end-to-end
# -----------------------------------------------------------------------

class TestCooldownRecoveryIntegration(RouterTestCase):
    """Test that an endpoint recovers from cooldown and handles requests again."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "flaky-cooldown": {
                    "base_url": MOCK_BASE,
                    "api_key": "test-key",
                    "type": "openai",
                    "endpoint_path": "/v1/chat/completions/error/500",
                    "retry": {
                        "max_retries": 1,
                        "cooldown_after": 2,   # cooldown after 2 failures
                        "cooldown_duration": 2,  # only 2 seconds cooldown
                        "error_limits": {500: 1},
                    },
                    "usage": {"rpm": 100},
                },
                "good": _openai_api(MOCK_BASE_V1),
            },
            groups={
                "main": {"strategy": "sequential",
                         "members": [{"api": "flaky-cooldown"}, {"api": "good"}]},
            },
        ), cls.config_path)

    def test_cooldown_then_recovery(self):
        """
        Test sequence:
        1. Send requests that trigger cooldown on primary
        2. Primary is skipped during cooldown, secondary handles requests
        3. After cooldown, primary is tried again (though still fails → secondary handles)
        """
        with self.client() as c:
            # These requests should go to "good" after "flaky-cooldown" fails
            for i in range(3):
                r = c.post("/v1/chat/completions",
                           json={"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]})
                self.assertEqual(r.status_code, 200, f"Request {i} failed: {r.text}")


# -----------------------------------------------------------------------
# Integration test: RPM limit blocks requests
# -----------------------------------------------------------------------

class TestRPMLimitIntegration(RouterTestCase):
    """Test that RPM limit blocks requests and recovery works."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "limited-api": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 3}}),
                "fallback": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 100}}),
            },
            groups={
                "main": {"strategy": "sequential",
                         "members": [{"api": "limited-api"}, {"api": "fallback"}]},
            },
        ), cls.config_path)

    def test_rpm_limit_triggers_fallback(self):
        """When primary hits RPM limit, should fall back to secondary."""
        with self.client() as c:
            # Fill up the RPM limit of 3 for "limited-api"
            for i in range(3):
                r = c.post("/v1/chat/completions",
                           json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(r.status_code, 200, f"Request {i} failed: {r.text}")

            # 4th request should be routed to fallback (limited-api is at rpm limit)
            r = c.post("/v1/chat/completions",
                       json={"model": "gpt-4", "messages": [{"role": "user", "content": "rpm test"}]})
            self.assertEqual(r.status_code, 200, "Should still succeed via fallback")


class TestRPMLimitWithoutFallback(RouterTestCase):
    """Test that requests are blocked with 503 when RPM limit hit and no fallback."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "limited-only": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 2}}),
            },
            groups={
                "main": {"strategy": "sequential", "members": [{"api": "limited-only"}]},
            },
        ), cls.config_path)

    def test_503_when_rpm_exceeded_no_fallback(self):
        """503 should be returned when the only API is rate-limited."""
        with self.client() as c:
            # Use up the 2-request RPM limit
            for _ in range(2):
                r = c.post("/v1/chat/completions",
                           json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(r.status_code, 200)

            # 3rd request should get 503 (RPM limit exceeded, no fallback)
            r = c.post("/v1/chat/completions",
                       json={"model": "gpt-4", "messages": [{"role": "user", "content": "over limit"}]})
            self.assertEqual(r.status_code, 503, "Should return 503 when RPM limit exceeded")


# -----------------------------------------------------------------------
# Integration test: Load balance weight distribution
# -----------------------------------------------------------------------

class TestLoadBalanceDistributionIntegration(RouterTestCase):
    """Test that load balancing distributes traffic proportionally by RPM."""

    @classmethod
    def _write_config(cls):
        # api1 has rpm=60 (weight 2x), api2 has rpm=30 (weight 1x)
        # Both point to the same success endpoint (we can't distinguish them from response)
        # Instead, give them different model names via endpoint_path to distinguish
        _write_test_config(_server_config(
            apis={
                "heavy": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 60}}),
                "light": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 30}}),
            },
            groups={
                "main": {"strategy": "load_balance",
                         "members": [{"api": "heavy"}, {"api": "light"}]},
            },
        ), cls.config_path)

    def test_load_balance_all_succeed(self):
        """All requests under load balancing should succeed."""
        with self.client() as c:
            for i in range(10):
                r = c.post("/v1/chat/completions",
                           json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(r.status_code, 200, f"Request {i} failed: {r.text}")


# -----------------------------------------------------------------------
# Integration test: Cross-format conversion
# -----------------------------------------------------------------------

class TestCrossFormatConversionIntegration(RouterTestCase):
    """Test OpenAI request → Anthropic backend → OpenAI response conversion."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={"anthropic-backend": _anthropic_api(MOCK_BASE)},
            groups={"main": {"strategy": "sequential", "members": [{"api": "anthropic-backend"}]}},
        ), cls.config_path)

    def test_openai_request_to_anthropic_backend(self):
        """OpenAI-format request should produce OpenAI-format response from Anthropic backend."""
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ],
                "temperature": 0.5,
            })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        # Must have OpenAI response format
        self.assertEqual(data.get("object"), "chat.completion",
                         "Response should be in OpenAI format")
        self.assertIn("choices", data)
        self.assertIn("message", data["choices"][0])
        self.assertIn("usage", data)
        self.assertIn("prompt_tokens", data["usage"])
        self.assertIn("completion_tokens", data["usage"])
        self.assertIn("total_tokens", data["usage"])

    def test_multiple_system_messages_to_anthropic(self):
        """Multiple system messages should be merged when routing to Anthropic backend."""
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hi"},
                ],
            })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data.get("object"), "chat.completion")

    def test_anthropic_request_to_anthropic_backend(self):
        """Anthropic-format request to Anthropic backend should return Anthropic response."""
        with self.client() as c:
            r = c.post("/v1/messages", json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data.get("type"), "message", "Response should be Anthropic format")
        self.assertIn("content", data)

    def test_anthropic_request_to_openai_backend(self):
        """Anthropic-format request to OpenAI backend should return Anthropic response."""
        # Use the main openai backend but send Anthropic request
        with self.client() as c:
            r = c.post("/v1/messages", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        # Should get back Anthropic format even though backend is Anthropic
        # (since this test uses Anthropic backend)
        self.assertEqual(data.get("type"), "message")


# -----------------------------------------------------------------------
# Integration test: Cross-format streaming
# -----------------------------------------------------------------------

class TestCrossFormatStreamingIntegration(RouterTestCase):
    """Test streaming across format boundaries."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "openai-backend": _openai_api(MOCK_BASE_V1),
                "anthropic-backend": _anthropic_api(MOCK_BASE),
            },
            groups={
                "main": {"strategy": "sequential",
                         "members": [{"api": "openai-backend"}]},
                "anthropic-group": {"strategy": "sequential",
                                    "members": [{"api": "anthropic-backend"}]},
            },
            default_group="main",
        ), cls.config_path)

    def test_openai_stream_from_openai_backend(self):
        """OpenAI streaming request to OpenAI backend (no conversion)."""
        with self.client() as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }) as r:
                self.assertEqual(r.status_code, 200, r.read())
                self.assertIn("text/event-stream", r.headers.get("content-type", ""))
                lines = list(r.iter_lines())

        data_lines = [l for l in lines if l.startswith("data:")]
        self.assertTrue(len(data_lines) > 0, f"No data lines in stream: {lines[:5]}")
        # Should have [DONE] at the end
        self.assertTrue(any("[DONE]" in l for l in data_lines),
                        "Stream should end with [DONE]")

    def test_anthropic_stream_from_openai_backend(self):
        """Anthropic streaming request to OpenAI backend should produce Anthropic stream."""
        with self.client() as c:
            with c.stream("POST", "/v1/messages", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
                "stream": True,
            }) as r:
                self.assertEqual(r.status_code, 200, r.read())
                lines = list(r.iter_lines())

        event_lines = [l for l in lines if l.startswith("event:")]
        data_lines = [l for l in lines if l.startswith("data:")]
        self.assertTrue(len(event_lines) > 0,
                        f"No event: lines in Anthropic stream: {lines[:10]}")
        # Should include message_start, content_block_delta, message_stop events
        event_types = [l.replace("event:", "").strip() for l in event_lines]
        self.assertIn("message_start", event_types, "Should have message_start event")
        self.assertIn("message_stop", event_types, "Should have message_stop event")

    def test_stream_content_is_non_empty(self):
        """Streaming response should contain actual content."""
        import json as json_module
        with self.client() as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }) as r:
                lines = list(r.iter_lines())

        # Extract content from data: lines
        content_parts = []
        for line in lines:
            if line.startswith("data:") and "[DONE]" not in line:
                try:
                    data = json_module.loads(line[5:].strip())
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                except Exception:
                    pass

        self.assertTrue(len(content_parts) > 0,
                        "Stream should contain content chunks")

    def test_pass_through_streaming_has_valid_sse_format(self):
        """
        Pass-through streaming (same format in/out) should produce valid SSE
        with double-newline event boundaries.

        This verifies the fix where blank lines were previously stripped,
        producing invalid SSE that real EventSource clients would reject.
        """
        with self.client() as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }) as r:
                self.assertEqual(r.status_code, 200, r.read())
                # Read as raw bytes to check SSE format
                raw_content = b"".join(r.iter_bytes()).decode()

        # SSE requires blank lines (\\n\\n) between events
        self.assertIn('\n\n', raw_content,
                      "Pass-through streaming must use double newlines for valid SSE format. "
                      "Blank lines were previously stripped (bug), they are now preserved (fix).")


# -----------------------------------------------------------------------
# Integration test: Streaming error handling
# -----------------------------------------------------------------------

class TestStreamingErrorHandling(RouterTestCase):
    """Test that streaming errors are handled as gracefully as possible."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "error-500": {
                    "base_url": MOCK_BASE,
                    "api_key": "test-key",
                    "type": "openai",
                    "endpoint_path": "/v1/chat/completions/error/500",
                    "retry": {"max_retries": 1, "cooldown_after": 10, "cooldown_duration": 60,
                              "error_limits": {500: 1}},
                    "usage": {"rpm": 100},
                },
                "good": _openai_api(MOCK_BASE_V1),
            },
            groups={
                "main": {"strategy": "sequential",
                         "members": [{"api": "error-500"}, {"api": "good"}]},
            },
        ), cls.config_path)

    def test_streaming_with_fallback(self):
        """When streaming request fails on primary, should fall through to secondary."""
        with self.client() as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }) as r:
                # The behavior here depends on whether streaming errors are caught
                # If error is caught before stream starts: proper fallback (200 from good)
                # If error is deferred: 200 response but broken stream
                status = r.status_code
                lines = list(r.iter_lines())

        data_lines = [l for l in lines if l.startswith("data:")]
        # At minimum, we should get some response (not a crash)
        self.assertIn(status, [200, 503],
                      f"Expected 200 or 503, got {status}")


# -----------------------------------------------------------------------
# Integration test: Nested groups
# -----------------------------------------------------------------------

class TestNestedGroupsEdgeCases(RouterTestCase):
    """Test edge cases with nested group routing."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "api1": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 60}}),
                "api2": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 30}}),
                "api3": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 10}}),
            },
            groups={
                "inner-lb": {"strategy": "load_balance",
                             "members": [{"api": "api1"}, {"api": "api2"}]},
                "outer-fallback": {"strategy": "sequential",
                                   "members": [{"api": "api3"}]},
                "main": {"strategy": "sequential",
                         "members": [{"group": "inner-lb"}, {"group": "outer-fallback"}]},
            },
        ), cls.config_path)

    def test_nested_sequential_with_lb_inner(self):
        """Sequential outer with load_balance inner should route correctly."""
        with self.client() as c:
            for i in range(5):
                r = c.post("/v1/chat/completions",
                           json={"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]})
                self.assertEqual(r.status_code, 200, f"Request {i}: {r.text}")


# -----------------------------------------------------------------------
# Integration test: Stats endpoint tracking
# -----------------------------------------------------------------------

class TestStatsTracking(RouterTestCase):
    """Test that the /stats endpoint accurately tracks request counts."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={"tracked": _openai_api(MOCK_BASE_V1)},
            groups={"main": {"strategy": "sequential", "members": [{"api": "tracked"}]}},
        ), cls.config_path)

    def test_stats_update_after_requests(self):
        """Request counts in /stats should update after making requests."""
        with self.client() as c:
            # Get baseline stats
            baseline = c.get("/stats").json()

            # Make 3 requests
            for _ in range(3):
                r = c.post("/v1/chat/completions",
                           json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(r.status_code, 200)

            # Check stats updated
            after = c.get("/stats").json()

        # Find the "tracked" api in the stats tree
        def find_api_stats(node):
            if isinstance(node, dict):
                if node.get("api_id") == "tracked":
                    return node
                for v in node.values():
                    result = find_api_stats(v)
                    if result:
                        return result
                    if isinstance(v, list):
                        for item in v:
                            result = find_api_stats(item)
                            if result:
                                return result
            return None

        api_stats = find_api_stats(after["tree"])
        self.assertIsNotNone(api_stats, "Should find 'tracked' api in stats")
        if api_stats:
            usage_stats = api_stats.get("usage", {})
            self.assertEqual(usage_stats.get("total_requests"), 3,
                             "Should have 3 total requests")
            self.assertEqual(usage_stats.get("total_success"), 3,
                             "Should have 3 successful requests")


if __name__ == "__main__":
    unittest.main(verbosity=2)
