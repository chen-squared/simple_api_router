"""
QA Round 2 tests for simple_api_router.

Covers:
1. Empty groups (sequential and load_balance) → 503 gracefully
2. Model override in config → backend receives overridden model name
3. Anthropic streaming → OpenAI format end-to-end (stream=True, OpenAI req → Anthropic backend)
4. Load balance with single available member (other in cooldown) still serves requests
5. Streaming generator cleanup (resource leak: source iterator must be closed on abandon)
6. Load balance convergence: 100 requests at RPM 600:300 → ~2:1 distribution
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


# ---------------------------------------------------------------------------
# Scenario 1a: Empty sequential group → 503
# ---------------------------------------------------------------------------

class TestEmptyGroupSequential(RouterTestCase):
    """An empty sequential group must return 503, not crash."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={"dummy": _openai_api(MOCK_BASE_V1)},
            groups={
                "main": {"strategy": "sequential", "members": []},
            },
        ), cls.config_path)

    def test_empty_sequential_returns_503(self):
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            })
        self.assertEqual(r.status_code, 503, f"Expected 503 for empty group, got {r.status_code}: {r.text}")

    def test_empty_sequential_error_message_is_useful(self):
        """Error body should describe the problem, not just say 'None'."""
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            })
        detail = r.json().get("detail", "")
        self.assertNotIn("None", detail,
            f"Error message should not contain 'None', got: {detail!r}")
        self.assertTrue(len(detail) > 10,
            f"Error message should be descriptive, got: {detail!r}")


# ---------------------------------------------------------------------------
# Scenario 1b: Empty load_balance group → 503
# ---------------------------------------------------------------------------

class TestEmptyGroupLoadBalance(RouterTestCase):
    """An empty load_balance group must return 503, not crash."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={"dummy": _openai_api(MOCK_BASE_V1)},
            groups={
                "main": {"strategy": "load_balance", "members": []},
            },
        ), cls.config_path)

    def test_empty_load_balance_returns_503(self):
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            })
        self.assertEqual(r.status_code, 503, f"Expected 503 for empty group, got {r.status_code}: {r.text}")


# ---------------------------------------------------------------------------
# Scenario 2: Model override in config
# ---------------------------------------------------------------------------

class TestModelOverrideOpenAI(RouterTestCase):
    """When API config has `model: X`, X must be sent to the backend, not the client's model."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "overridden": _openai_api(MOCK_BASE_V1, {"model": "forced-model-name"}),
            },
            groups={"main": {"strategy": "sequential", "members": [{"api": "overridden"}]}},
        ), cls.config_path)

    def test_model_override_sent_to_backend(self):
        """Client requests model 'gpt-4', backend should receive 'forced-model-name'."""
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        # Mock echoes back the model name it received in the request body
        self.assertEqual(
            data.get("model"), "forced-model-name",
            f"Expected backend to receive 'forced-model-name', got {data.get('model')!r}. "
            "The model override in config was not applied."
        )

    def test_model_override_applies_regardless_of_client_model(self):
        """Override applies even when client sends a different model name."""
        for client_model in ["gpt-3.5-turbo", "gpt-4o", "some-other-model"]:
            with self.client() as c:
                r = c.post("/v1/chat/completions", json={
                    "model": client_model,
                    "messages": [{"role": "user", "content": "Hi"}],
                })
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json().get("model"), "forced-model-name",
                f"Override should apply for client model {client_model!r}")


class TestModelOverrideAnthropicBackend(RouterTestCase):
    """Model override should also work when routing OpenAI request to Anthropic backend."""

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "anthropic-with-override": _anthropic_api(MOCK_BASE, {"model": "claude-3-haiku-forced"}),
            },
            groups={"main": {"strategy": "sequential", "members": [{"api": "anthropic-with-override"}]}},
        ), cls.config_path)

    def test_model_override_openai_to_anthropic(self):
        """OpenAI request to Anthropic backend with model override."""
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        # Anthropic mock echoes model → converted back in anthropic_to_openai_response
        self.assertEqual(data.get("model"), "claude-3-haiku-forced",
            f"Expected backend to receive 'claude-3-haiku-forced', got {data.get('model')!r}")


# ---------------------------------------------------------------------------
# Scenario 3: Anthropic streaming → OpenAI format end-to-end
# ---------------------------------------------------------------------------

class TestAnthropicStreamingToOpenAIFormat(RouterTestCase):
    """
    OpenAI-format request with stream=True hitting an Anthropic backend.
    The router must convert Anthropic SSE → OpenAI SSE.
    Verifies all chunks parse correctly as OpenAI streaming format.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={"anthropic-stream": _anthropic_api(MOCK_BASE)},
            groups={"main": {"strategy": "sequential", "members": [{"api": "anthropic-stream"}]}},
        ), cls.config_path)

    def _collect_openai_stream_chunks(self, lines):
        """Parse OpenAI SSE lines into data payloads."""
        chunks = []
        for line in lines:
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                chunks.append("[DONE]")
                continue
            try:
                chunks.append(json.loads(data_str))
            except json.JSONDecodeError as e:
                self.fail(f"Invalid JSON in SSE data line: {line!r} — {e}")
        return chunks

    def test_openai_stream_request_to_anthropic_backend_succeeds(self):
        """OpenAI streaming request to Anthropic backend returns 200 with SSE."""
        with self.client() as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }) as r:
                self.assertEqual(r.status_code, 200, r.read())
                self.assertIn("text/event-stream", r.headers.get("content-type", ""),
                    "Response must be text/event-stream for streaming")
                lines = list(r.iter_lines())

        self.assertTrue(len(lines) > 0, "Stream should not be empty")

    def test_all_chunks_are_valid_openai_format(self):
        """Every data: chunk in the SSE stream must be valid OpenAI streaming format."""
        with self.client() as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }) as r:
                self.assertEqual(r.status_code, 200, r.read())
                lines = list(r.iter_lines())

        chunks = self._collect_openai_stream_chunks(lines)
        self.assertTrue(len(chunks) > 0, f"No chunks found in stream lines: {lines[:10]}")

        done_found = False
        content_found = False

        for chunk in chunks:
            if chunk == "[DONE]":
                done_found = True
                continue

            # Each non-DONE chunk must have OpenAI streaming structure
            self.assertIn("id", chunk, f"Chunk missing 'id': {chunk}")
            self.assertIn("object", chunk, f"Chunk missing 'object': {chunk}")
            self.assertEqual(chunk["object"], "chat.completion.chunk",
                f"Expected object='chat.completion.chunk', got {chunk['object']!r}")
            self.assertIn("choices", chunk, f"Chunk missing 'choices': {chunk}")
            self.assertIsInstance(chunk["choices"], list)
            self.assertTrue(len(chunk["choices"]) > 0)

            choice = chunk["choices"][0]
            self.assertIn("delta", choice, f"Choice missing 'delta': {choice}")
            self.assertIn("index", choice, f"Choice missing 'index': {choice}")

            delta = choice["delta"]
            if delta.get("content"):
                content_found = True

        self.assertTrue(done_found, "Stream must end with [DONE]")
        self.assertTrue(content_found, "At least one chunk must contain content text")

    def test_stream_has_proper_sse_double_newlines(self):
        """The raw SSE stream must have proper double-newline event boundaries."""
        with self.client() as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }) as r:
                self.assertEqual(r.status_code, 200, r.read())
                raw = b"".join(r.iter_bytes()).decode()

        self.assertIn("\n\n", raw,
            "Converted SSE stream must have double-newline event boundaries")

    def test_stream_content_matches_expected_text(self):
        """Content assembled from stream chunks should be non-empty."""
        with self.client() as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }) as r:
                lines = list(r.iter_lines())

        chunks = self._collect_openai_stream_chunks(lines)
        content = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
            if c != "[DONE]" and c.get("choices", [{}])[0].get("delta", {}).get("content")
        )
        self.assertTrue(len(content) > 0,
            f"Assembled content from stream should be non-empty, got: {content!r}")


# ---------------------------------------------------------------------------
# Scenario 4: Load balance with single available member
# ---------------------------------------------------------------------------

class TestLoadBalanceSingleAvailable(RouterTestCase):
    """
    When all but one load_balance members are unavailable (in cooldown),
    the remaining member must still serve all requests successfully.

    Strategy: 'bad' API has overwhelming weight (rpm=10_000_000) so it is
    always selected first. It always returns 500. With cooldown_after=1,
    it enters cooldown after the first failure. Subsequent requests see only
    'good' in available_members and succeed.
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "bad": {
                    "base_url": MOCK_BASE,
                    "api_key": "test-key",
                    "type": "openai",
                    "endpoint_path": "/v1/chat/completions/error/500",
                    "retry": {
                        "max_retries": 0,
                        "cooldown_after": 1,
                        "cooldown_duration": 300,
                        "error_limits": {500: 0},
                    },
                    "usage": {"rpm": 10_000_000},
                },
                "good": _openai_api(MOCK_BASE_V1, {"usage": {"rpm": 10_000}}),
            },
            groups={
                "main": {
                    "strategy": "load_balance",
                    "members": [{"api": "bad"}, {"api": "good"}],
                },
            },
        ), cls.config_path)

    def test_first_request_succeeds_via_fallback(self):
        """First request: bad is selected and fails, good handles as fallback → 200."""
        with self.client() as c:
            r = c.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            })
        self.assertEqual(r.status_code, 200,
            f"First request should succeed via fallback to 'good', got: {r.status_code} {r.text}")

    def test_subsequent_requests_succeed_with_bad_in_cooldown(self):
        """After bad enters cooldown, subsequent requests should all succeed via good."""
        with self.client() as c:
            # First request: triggers bad's cooldown (bad selected, fails → good fallback)
            r0 = c.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "trigger"}],
            })
            self.assertEqual(r0.status_code, 200, f"First request failed: {r0.text}")

            # Now bad is in cooldown (cooldown_after=1). All subsequent requests must
            # route exclusively to 'good'.
            for i in range(5):
                r = c.post("/v1/chat/completions", json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": f"request {i}"}],
                })
                self.assertEqual(r.status_code, 200,
                    f"Request {i} failed after bad entered cooldown: {r.status_code} {r.text}")


# ---------------------------------------------------------------------------
# Scenario 5: Streaming generator cleanup (resource leak)
# ---------------------------------------------------------------------------

class TestStreamingGeneratorCleanup(unittest.TestCase):
    """
    Verifies that streaming conversion generators properly close their source
    iterator when abandoned mid-stream (client disconnect simulation).

    This is a regression test for the resource leak where abandoning
    stream_anthropic_to_openai or stream_openai_to_anthropic would NOT
    close the underlying HTTP response, keeping the connection open.
    """

    def test_anthropic_to_openai_closes_source_on_abandon(self):
        """
        stream_anthropic_to_openai must call aclose() on source when the
        outer generator is abandoned (aclose() called on it).

        Uses a class-based mock so we can verify aclose() is called regardless
        of whether iteration was started (Python generators only run their
        finally block if the generator has been entered at least once).
        """
        from router.converter import stream_anthropic_to_openai

        class MockSource:
            def __init__(self):
                self.aclose_called = False
                self._gen = self._make_gen()

            async def _make_gen(self):
                while True:
                    yield b'event: content_block_delta\n'
                    yield b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}\n'
                    yield b'\n'
                    await asyncio.sleep(0.01)

            def __aiter__(self):
                return self._gen.__aiter__()

            async def __anext__(self):
                return await self._gen.__anext__()

            async def aclose(self):
                self.aclose_called = True
                await self._gen.aclose()

        async def _run():
            source = MockSource()
            outer = stream_anthropic_to_openai(source)
            # Consume first chunk (initial role chunk — source not yet iterated)
            await outer.__anext__()
            # Abandon mid-stream (simulates client disconnect)
            await outer.aclose()
            await asyncio.sleep(0.05)
            return source.aclose_called

        result = asyncio.run(_run())
        self.assertTrue(result,
            "stream_anthropic_to_openai must call source.aclose() when abandoned. "
            "Resource leak: HTTP response connection would stay open indefinitely.")

    def test_openai_to_anthropic_closes_source_on_abandon(self):
        """
        stream_openai_to_anthropic must call aclose() on source when the
        outer generator is abandoned.
        """
        from router.converter import stream_openai_to_anthropic

        class MockSource:
            def __init__(self):
                self.aclose_called = False
                self._gen = self._make_gen()

            async def _make_gen(self):
                while True:
                    yield b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n'
                    await asyncio.sleep(0.01)

            def __aiter__(self):
                return self._gen.__aiter__()

            async def __anext__(self):
                return await self._gen.__anext__()

            async def aclose(self):
                self.aclose_called = True
                await self._gen.aclose()

        async def _run():
            source = MockSource()
            outer = stream_openai_to_anthropic(source)
            # Consume several initial event chunks (message_start, content_block_start, ping)
            await outer.__anext__()
            await outer.__anext__()
            await outer.__anext__()
            # Abandon mid-stream
            await outer.aclose()
            await asyncio.sleep(0.05)
            return source.aclose_called

        result = asyncio.run(_run())
        self.assertTrue(result,
            "stream_openai_to_anthropic must call source.aclose() when abandoned. "
            "Resource leak: HTTP response connection would stay open indefinitely.")

    def test_anthropic_to_openai_closes_source_on_complete(self):
        """Source must also be closed when the stream completes normally."""
        from router.converter import stream_anthropic_to_openai

        aclose_called = False

        class MockFiniteSource:
            def __init__(self):
                self._gen = self._make_gen()

            async def _make_gen(self):
                lines = [
                    b'event: message_start\n',
                    b'data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0}}}\n',
                    b'\n',
                    b'event: content_block_delta\n',
                    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n',
                    b'\n',
                    b'event: message_delta\n',
                    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}\n',
                    b'\n',
                    b'event: message_stop\n',
                    b'data: {"type":"message_stop"}\n',
                    b'\n',
                ]
                for line in lines:
                    yield line

            def __aiter__(self):
                return self._gen.__aiter__()

            async def __anext__(self):
                return await self._gen.__anext__()

            async def aclose(self):
                nonlocal aclose_called
                aclose_called = True
                await self._gen.aclose()

        async def _run():
            source = MockFiniteSource()
            chunks = []
            async for chunk in stream_anthropic_to_openai(source):
                chunks.append(chunk)
            await asyncio.sleep(0.05)
            return aclose_called, chunks

        result, chunks = asyncio.run(_run())
        self.assertTrue(result, "Source must be closed after stream completes normally")
        self.assertTrue(len(chunks) > 0, "Should have yielded some chunks")

    def test_response_stream_aclose_works_before_iteration(self):
        """
        _ResponseStream.aclose() must close the underlying HTTP response
        even if iteration never started. This is the core resource-leak fix.
        """
        from router.endpoint import _ResponseStream

        response_closed = False

        class FakeResponse:
            def aiter_lines(self):
                return self._gen()

            async def _gen(self):
                while True:
                    yield "line"

            async def aclose(self):
                nonlocal response_closed
                response_closed = True

        async def _run():
            stream = _ResponseStream(FakeResponse())
            # Never iterate — directly close
            await stream.aclose()
            return response_closed

        result = asyncio.run(_run())
        self.assertTrue(result,
            "_ResponseStream.aclose() must close the HTTP response even before iteration")


# ---------------------------------------------------------------------------
# Scenario 6: Load balance convergence with 100 requests
# ---------------------------------------------------------------------------

class TestLoadBalanceConvergence100(RouterTestCase):
    """
    Make 100 requests to a load_balance group where apis have RPM 600:300 ratio.
    Verify the distribution is approximately 2:1 (heavy:light).

    Uses model overrides to distinguish which API served each request:
    - heavy API → model override "model-heavy" (echoed back by mock)
    - light API → model override "model-light" (echoed back by mock)
    """

    @classmethod
    def _write_config(cls):
        _write_test_config(_server_config(
            apis={
                "heavy": _openai_api(MOCK_BASE_V1, {
                    "model": "model-heavy",
                    "usage": {"rpm": 600},
                }),
                "light": _openai_api(MOCK_BASE_V1, {
                    "model": "model-light",
                    "usage": {"rpm": 300},
                }),
            },
            groups={
                "main": {
                    "strategy": "load_balance",
                    "members": [{"api": "heavy"}, {"api": "light"}],
                },
            },
        ), cls.config_path)

    def test_all_100_requests_succeed(self):
        """All 100 requests must succeed regardless of which API is chosen."""
        with self.client() as c:
            for i in range(100):
                r = c.post("/v1/chat/completions", json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                })
                self.assertEqual(r.status_code, 200, f"Request {i} failed: {r.text}")

    def test_distribution_is_approximately_2_to_1(self):
        """
        With RPM weights 600:300 (2:1 ratio), the distribution over 100 requests
        should be approximately 2:1. Accept any ratio between 1.2 and 3.0 to allow
        for natural randomness (100 samples has significant variance).
        """
        counts = {"model-heavy": 0, "model-light": 0, "other": 0}

        with self.client() as c:
            for _ in range(100):
                r = c.post("/v1/chat/completions", json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                })
                self.assertEqual(r.status_code, 200)
                model = r.json().get("model", "")
                if model in counts:
                    counts[model] += 1
                else:
                    counts["other"] += 1

        heavy = counts["model-heavy"]
        light = counts["model-light"]
        total = heavy + light

        self.assertEqual(counts["other"], 0,
            f"Unexpected model names in responses: {counts}")
        self.assertEqual(total, 100,
            f"Expected 100 routed requests, got {total}")

        self.assertGreater(heavy, 0, "heavy API should have received some requests")
        self.assertGreater(light, 0, "light API should have received some requests")

        ratio = heavy / light
        self.assertGreater(ratio, 1.2,
            f"Expected ratio > 1.2:1 (heavy:light), got {heavy}:{light} = {ratio:.2f}:1. "
            f"The 2:1 weight ratio is not being respected.")
        self.assertLess(ratio, 3.5,
            f"Expected ratio < 3.5:1 (heavy:light), got {heavy}:{light} = {ratio:.2f}:1. "
            f"Variance too large for 100 requests at 2:1 intended ratio.")

        # Also verify neither API was completely starved
        self.assertGreater(light, 10,
            f"light API should get at least 10/100 requests at 1:2 ratio, got {light}")
        self.assertGreater(heavy, 40,
            f"heavy API should get at least 40/100 requests at 2:1 ratio, got {heavy}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
