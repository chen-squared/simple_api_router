"""
QA Round 3 tests for simple_api_router.

Covers:
1. Config validation edge cases (self-loops, circular deps, missing required fields)
2. Non-JSON/HTML upstream responses handled gracefully (no crash, retried/503)
3. Content-block list format in messages (Anthropic ↔ OpenAI converter unit tests)
4. Concurrent requests (thread safety with asyncio.Lock)
5. Slow-backend simulation (2s delay, verifies request still succeeds)
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
from pydantic import ValidationError

ROOT = Path(__file__).parent.parent
MOCK_PORT = 19999
ROUTER_PORT = 18080


# ---------------------------------------------------------------------------
# Server management (same pattern as other test files)
# ---------------------------------------------------------------------------

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
    cfg: dict = {
        "base_url": base_v1,
        "api_key": "test-key",
        "type": "openai",
        "retry": {"max_retries": 1, "cooldown_after": 20, "cooldown_duration": 5,
                  "error_limits": {502: 1, 429: 1}},
        "usage": {"rpm": 200},
    }
    if extra:
        cfg.update(extra)
    return cfg


def _anthropic_api(base: str, extra: dict | None = None) -> dict:
    cfg: dict = {
        "base_url": base,
        "api_key": "test-key",
        "type": "anthropic",
        "retry": {"max_retries": 1, "cooldown_after": 20, "cooldown_duration": 5,
                  "error_limits": {502: 1}},
        "usage": {"rpm": 200},
    }
    if extra:
        cfg.update(extra)
    return cfg


# ===========================================================================
# 1. Config Validation Edge Cases (pure unit tests, no server)
# ===========================================================================

class TestConfigValidation(unittest.TestCase):
    """Validate that bad configs are rejected at load/build time."""

    def _load(self, cfg: Dict[str, Any]):
        """Helper: load config dict through RouterConfig."""
        from router.config import RouterConfig
        return RouterConfig.model_validate(cfg)

    def _build_tree(self, cfg: Dict[str, Any]):
        """Helper: load config and build routing tree."""
        from router.group import build_routing_tree
        router_cfg = self._load(cfg)
        return build_routing_tree(router_cfg)

    def _minimal_config(self, groups, apis, default_group="main"):
        return {
            "server": {"host": "127.0.0.1", "port": 9999, "log_level": "WARNING", "log_file": None},
            "default_group": default_group,
            "apis": apis,
            "groups": groups,
        }

    def _api(self):
        return {"base_url": "http://localhost/v1", "api_key": "k", "type": "openai"}

    def test_default_group_not_found_raises(self):
        """default_group pointing at non-existent group → ValidationError."""
        cfg = self._minimal_config(
            groups={"main": {"strategy": "sequential", "members": [{"api": "a"}]}},
            apis={"a": self._api()},
            default_group="nonexistent",
        )
        with self.assertRaises(ValidationError):
            self._load(cfg)

    def test_missing_base_url_raises(self):
        """API entry without base_url → ValidationError."""
        cfg = self._minimal_config(
            groups={"main": {"strategy": "sequential", "members": [{"api": "a"}]}},
            apis={"a": {"api_key": "k", "type": "openai"}},  # no base_url
        )
        with self.assertRaises(ValidationError):
            self._load(cfg)

    def test_unknown_api_in_group_raises(self):
        """Group member referencing non-existent api → ValidationError."""
        cfg = self._minimal_config(
            groups={"main": {"strategy": "sequential", "members": [{"api": "ghost"}]}},
            apis={"a": self._api()},
        )
        with self.assertRaises(ValidationError):
            self._load(cfg)

    def test_invalid_strategy_raises(self):
        """Group with invalid strategy → ValidationError."""
        cfg = self._minimal_config(
            groups={"main": {"strategy": "random_pick", "members": [{"api": "a"}]}},
            apis={"a": self._api()},
        )
        with self.assertRaises(ValidationError):
            self._load(cfg)

    def test_self_loop_group_raises_at_build(self):
        """Group A references itself → build_routing_tree raises ValueError."""
        cfg = self._minimal_config(
            groups={
                "main": {"strategy": "sequential", "members": [{"group": "main"}]},
            },
            apis={"a": self._api()},
        )
        with self.assertRaises(ValueError, msg="Circular group reference not detected"):
            self._build_tree(cfg)

    def test_circular_group_ab_raises_at_build(self):
        """Group A → B → A circular reference → build_routing_tree raises ValueError."""
        cfg = self._minimal_config(
            groups={
                "main": {"strategy": "sequential", "members": [{"group": "sub"}]},
                "sub": {"strategy": "sequential", "members": [{"group": "main"}]},
            },
            apis={"a": self._api()},
        )
        with self.assertRaises(ValueError, msg="Circular group A→B not detected"):
            self._build_tree(cfg)

    def test_group_member_both_api_and_group_raises(self):
        """GroupMember with both api and group set → ValidationError."""
        cfg = self._minimal_config(
            groups={"main": {"strategy": "sequential", "members": [{"api": "a", "group": "main"}]}},
            apis={"a": self._api()},
        )
        with self.assertRaises(ValidationError):
            self._load(cfg)

    def test_group_member_neither_api_nor_group_raises(self):
        """GroupMember with neither api nor group → ValidationError."""
        cfg = self._minimal_config(
            groups={"main": {"strategy": "sequential", "members": [{}]}},
            apis={"a": self._api()},
        )
        with self.assertRaises(ValidationError):
            self._load(cfg)


# ===========================================================================
# 2. Content Block List Format (converter unit tests)
# ===========================================================================

class TestContentBlocksUnit(unittest.TestCase):
    """Unit tests for format converter with list-type content blocks."""

    def test_anthropic_list_content_to_openai_flattens(self):
        """Anthropic user message with list content → OpenAI string content."""
        from router.converter import anthropic_to_openai_request
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello world"}]},
            ],
            "max_tokens": 100,
        }
        result = anthropic_to_openai_request(body)
        msg = result["messages"][0]
        self.assertEqual(msg["role"], "user")
        self.assertIsInstance(msg["content"], str)
        self.assertEqual(msg["content"], "Hello world")

    def test_anthropic_multi_block_content_flattens(self):
        """Multiple text blocks in Anthropic content → concatenated string."""
        from router.converter import anthropic_to_openai_request
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": " world"},
                    ],
                }
            ],
            "max_tokens": 100,
        }
        result = anthropic_to_openai_request(body)
        self.assertEqual(result["messages"][0]["content"], "Hello world")

    def test_openai_user_list_content_to_anthropic_passthrough(self):
        """OpenAI user message with list content → Anthropic passes list through."""
        from router.converter import openai_to_anthropic_request
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ],
        }
        result = openai_to_anthropic_request(body)
        msg = result["messages"][0]
        self.assertEqual(msg["role"], "user")
        # Anthropic natively accepts content as a list of blocks
        self.assertIsInstance(msg["content"], list)
        self.assertEqual(msg["content"][0]["text"], "Hello")

    def test_openai_system_list_content_merged(self):
        """OpenAI system message with list content → merged into Anthropic system string."""
        from router.converter import openai_to_anthropic_request
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Be helpful"}]},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = openai_to_anthropic_request(body)
        self.assertIn("system", result)
        self.assertEqual(result["system"], "Be helpful")

    def test_openai_assistant_list_content_flattened(self):
        """OpenAI assistant message with list content → flattened before sending to Anthropic."""
        from router.converter import openai_to_anthropic_request
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I can "},
                        {"type": "text", "text": "help you"},
                    ],
                },
            ],
        }
        result = openai_to_anthropic_request(body)
        assistant_msg = result["messages"][1]
        # Assistant content should be a list of blocks (Anthropic format)
        # or a string — either way it must be non-empty
        content = assistant_msg["content"]
        if isinstance(content, list):
            flat = "".join(b.get("text", "") for b in content)
        else:
            flat = content
        self.assertIn("help", flat)

    def test_flatten_content_with_none_text(self):
        """_flatten_content handles blocks with missing 'text' key gracefully."""
        from router.converter import _flatten_content
        blocks = [
            {"type": "text", "text": "Hello"},
            {"type": "text"},  # no 'text' key
            {"type": "image"},  # non-text block
        ]
        result = _flatten_content(blocks)
        self.assertEqual(result, "Hello")


# ===========================================================================
# 3. Non-JSON / HTML Upstream Error Responses (integration)
# ===========================================================================

class TestNonJSONUpstreamHTMLError(RouterTestCase):
    """Router must handle HTML upstream error bodies without crashing."""

    @classmethod
    def _write_config(cls):
        # Single API endpoint that always returns HTML 502
        cfg = _server_config(
            groups={"main": {"strategy": "sequential", "members": [{"api": "html_err"}]}},
            apis={
                "html_err": _openai_api(
                    MOCK_BASE_V1,
                    extra={
                        "endpoint_path": "/chat/completions/html/502",
                        "retry": {
                            "max_retries": 1,
                            "cooldown_after": 20,
                            "cooldown_duration": 5,
                            "error_limits": {502: 1},
                        },
                    },
                ),
            },
        )
        with open(cls.config_path, "w") as f:
            yaml.dump(cfg, f)

    def test_html_upstream_returns_503_not_500(self):
        """HTML 502 from upstream → router responds with 5xx (no crash)."""
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn(r.status_code, (502, 503), f"Expected 502 or 503, got {r.status_code}")

    def test_html_upstream_response_is_json(self):
        """Even with HTML upstream error, router response body is valid JSON."""
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        # Should not raise — the router always returns JSON
        body = r.json()
        self.assertIsInstance(body, dict)


class TestNonJSONUpstreamPlainText(RouterTestCase):
    """Router must handle plain-text upstream error bodies without crashing."""

    @classmethod
    def _write_config(cls):
        cfg = _server_config(
            groups={"main": {"strategy": "sequential", "members": [{"api": "text_err"}]}},
            apis={
                "text_err": _openai_api(
                    MOCK_BASE_V1,
                    extra={
                        "endpoint_path": "/chat/completions/text/503",
                        "retry": {
                            "max_retries": 1,
                            "cooldown_after": 20,
                            "cooldown_duration": 5,
                            "error_limits": {503: 1},
                        },
                    },
                ),
            },
        )
        with open(cls.config_path, "w") as f:
            yaml.dump(cfg, f)

    def test_text_upstream_returns_5xx(self):
        """Plain-text 503 from upstream → router responds 5xx without crash."""
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn(r.status_code, (502, 503))
        body = r.json()
        self.assertIsInstance(body, dict)


class TestNonJSONUpstreamFallback(RouterTestCase):
    """HTML error from primary, healthy secondary → secondary is used."""

    @classmethod
    def _write_config(cls):
        cfg = _server_config(
            groups={"main": {"strategy": "sequential", "members": [
                {"api": "html_err"},
                {"api": "ok"},
            ]}},
            apis={
                "html_err": _openai_api(
                    MOCK_BASE_V1,
                    extra={
                        "endpoint_path": "/chat/completions/html/502",
                        "retry": {"max_retries": 0, "cooldown_after": 20,
                                  "cooldown_duration": 5, "error_limits": {502: 1}},
                    },
                ),
                "ok": _openai_api(MOCK_BASE_V1),
            },
        )
        with open(cls.config_path, "w") as f:
            yaml.dump(cfg, f)

    def test_fallback_to_healthy_after_html_error(self):
        """Primary returns HTML 502, secondary is healthy → 200 from secondary."""
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("choices", body)


# ===========================================================================
# 4. Content Block Integration Test
# ===========================================================================

class TestContentBlocksIntegration(RouterTestCase):
    """Verify that list-type content blocks pass through the router end-to-end."""

    @classmethod
    def _write_config(cls):
        # OpenAI backend, request goes through openai→anthropic→openai path if needed
        cfg = _server_config(
            groups={"main": {"strategy": "sequential", "members": [{"api": "oai"}]}},
            apis={"oai": _openai_api(MOCK_BASE_V1)},
        )
        with open(cls.config_path, "w") as f:
            yaml.dump(cfg, f)

    def test_openai_request_with_list_content_succeeds(self):
        """OpenAI request where user message content is a list of text blocks → 200."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, what is 2+2?"}],
                }
            ],
        }
        with self.client() as c:
            r = c.post("/v1/chat/completions", json=payload)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("choices", body)

    def test_anthropic_request_with_list_content_succeeds(self):
        """Anthropic request where content is a list of text blocks, sent to OpenAI backend → 200."""
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello from Anthropic client"}],
                }
            ],
            "max_tokens": 100,
        }
        with self.client() as c:
            # Send to /v1/messages (Anthropic endpoint), router converts to OpenAI
            r = c.post("/v1/messages", json=payload)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        # Response should be converted back to Anthropic format
        self.assertIn("content", body)

    def test_multi_turn_with_list_content(self):
        """Multi-turn conversation with list content in all roles → 200."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Be concise"}]},
                {"role": "user", "content": [{"type": "text", "text": "Ping"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Pong"}]},
                {"role": "user", "content": "One more"},
            ],
        }
        with self.client() as c:
            r = c.post("/v1/chat/completions", json=payload)
        self.assertEqual(r.status_code, 200)


# ===========================================================================
# 5. Concurrent Requests (thread safety)
# ===========================================================================

class TestConcurrentRequests(RouterTestCase):
    """10 concurrent requests must all succeed without race conditions."""

    NUM_REQUESTS = 10

    @classmethod
    def _write_config(cls):
        cfg = _server_config(
            groups={"main": {"strategy": "load_balance", "members": [
                {"api": "a"},
                {"api": "b"},
            ]}},
            apis={
                "a": _openai_api(MOCK_BASE_V1, extra={"usage": {"rpm": 200}}),
                "b": _openai_api(MOCK_BASE_V1, extra={"usage": {"rpm": 200}}),
            },
        )
        with open(cls.config_path, "w") as f:
            yaml.dump(cfg, f)

    def _send_one(self, _):
        with httpx.Client(base_url=f"http://127.0.0.1:{ROUTER_PORT}", timeout=30.0) as c:
            return c.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "ping"}]},
            )

    def test_concurrent_requests_all_succeed(self):
        """10 concurrent OpenAI requests → all return 200."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.NUM_REQUESTS) as pool:
            futures = [pool.submit(self._send_one, i) for i in range(self.NUM_REQUESTS)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        statuses = [r.status_code for r in results]
        failures = [s for s in statuses if s != 200]
        self.assertEqual(failures, [], f"Some requests failed: {statuses}")

    def test_concurrent_anthropic_requests_all_succeed(self):
        """10 concurrent Anthropic requests → all return 200."""
        def _send(_):
            with httpx.Client(base_url=f"http://127.0.0.1:{ROUTER_PORT}", timeout=30.0) as c:
                return c.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3-5-sonnet-20241022",
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 50,
                    },
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.NUM_REQUESTS) as pool:
            futures = [pool.submit(_send, i) for i in range(self.NUM_REQUESTS)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        statuses = [r.status_code for r in results]
        failures = [s for s in statuses if s != 200]
        self.assertEqual(failures, [], f"Some Anthropic requests failed: {statuses}")

    def test_stats_after_concurrent_requests_tracks_totals(self):
        """After N requests, stats endpoint shows total_requests >= N."""
        n = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(self._send_one, i) for i in range(n)]
            [f.result() for f in concurrent.futures.as_completed(futures)]

        with httpx.Client(base_url=f"http://127.0.0.1:{ROUTER_PORT}", timeout=5.0) as c:
            stats = c.get("/stats").json()

        # Sum total_requests across all members in the tree
        def _sum_requests(node: dict) -> int:
            if "members" in node:
                return sum(_sum_requests(m) for m in node["members"])
            return node.get("usage", {}).get("total_requests", 0)

        total = _sum_requests(stats["tree"])
        # Previous tests in this class also fire requests, so just check >= n
        self.assertGreaterEqual(total, n, f"Expected at least {n} tracked, got {total}")


# ===========================================================================
# 6. Slow Backend (2-second delay)
# ===========================================================================

class TestSlowBackend(RouterTestCase):
    """Requests to a 2-second-delay endpoint still succeed."""

    @classmethod
    def _write_config(cls):
        cfg = _server_config(
            groups={"main": {"strategy": "sequential", "members": [{"api": "slow"}]}},
            apis={
                "slow": _openai_api(
                    MOCK_BASE_V1,
                    extra={"endpoint_path": "/chat/completions/slow"},
                ),
            },
        )
        with open(cls.config_path, "w") as f:
            yaml.dump(cfg, f)

    def test_slow_endpoint_returns_success(self):
        """Request to slow endpoint (2s delay) → 200 with valid response."""
        with self.client() as c:
            r = c.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
            )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("choices", body)
        self.assertIn("content", body["choices"][0]["message"])

    def test_slow_endpoint_timing(self):
        """Slow endpoint adds ~2s latency (3s tolerance)."""
        start = time.time()
        with self.client() as c:
            r = c.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
            )
        elapsed = time.time() - start
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(elapsed, 1.8, f"Expected ~2s delay, got {elapsed:.2f}s")
        self.assertLess(elapsed, 8.0, f"Took too long: {elapsed:.2f}s")

    def test_slow_anthropic_endpoint_succeeds(self):
        """Slow endpoint also works for Anthropic-format requests (converted on-the-fly)."""
        with self.client() as c:
            r = c.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "hello slow"}],
                    "max_tokens": 50,
                },
            )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("content", body)


if __name__ == "__main__":
    unittest.main()
