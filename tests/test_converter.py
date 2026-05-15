"""Unit tests for converter.py — Anthropic ↔ OpenAI format conversion."""
from __future__ import annotations

import asyncio
import json
import unittest
from typing import AsyncIterator

from simple_api_router.converter import (
    anthropic_to_openai_request,
    clean_schema,
    is_o_series,
    openai_to_anthropic_response,
    sanitize_system_text,
    stream_openai_to_anthropic,
    strip_private_params,
    supports_reasoning_effort,
)


class TestAnthropicToOpenAI(unittest.TestCase):
    # ------------------------------------------------------------------
    # Basic message conversion
    # ------------------------------------------------------------------
    def test_simple_text(self):
        body = {
            "model": "anthropic/claude-sonnet-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        self.assertEqual(result["model"], "gpt-4o")
        self.assertEqual(result["messages"][-1]["role"], "user")
        self.assertEqual(result["messages"][-1]["content"], "Hello")

    def test_system_string(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "system": "Be helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertEqual(result["messages"][0]["content"], "Be helpful.")

    def test_system_block_list(self):
        # Multiple system blocks each become their own system message so that
        # cache_control can be preserved per-block.
        body = {
            "model": "x",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "Part 1."},
                {"type": "text", "text": "Part 2."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        sys_msgs = [m for m in result["messages"] if m["role"] == "system"]
        self.assertEqual(len(sys_msgs), 2)
        self.assertEqual(sys_msgs[0]["content"], "Part 1.")
        self.assertEqual(sys_msgs[1]["content"], "Part 2.")

    def test_content_block_list_text(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        self.assertEqual(result["messages"][0]["content"], "Hello")

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------
    def test_tools_converted(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "tools": [
                {
                    "name": "calculator",
                    "description": "Compute math",
                    "input_schema": {
                        "type": "object",
                        "properties": {"expr": {"type": "string"}},
                    },
                }
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        self.assertIn("tools", result)
        self.assertEqual(result["tools"][0]["type"], "function")
        self.assertEqual(result["tools"][0]["function"]["name"], "calculator")

    def test_tool_choice_any_becomes_required(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"name": "t", "description": "", "input_schema": {}}],
            "tool_choice": {"type": "any"},
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        self.assertEqual(result["tool_choice"], "required")

    def test_tool_choice_specific(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"name": "my_tool", "description": "", "input_schema": {}}],
            "tool_choice": {"type": "tool", "name": "my_tool"},
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        self.assertEqual(result["tool_choice"]["type"], "function")
        self.assertEqual(result["tool_choice"]["function"]["name"], "my_tool")

    def test_tool_result_becomes_tool_message(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc",
                            "content": "42",
                        }
                    ],
                }
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        tool_msg = next(m for m in result["messages"] if m["role"] == "tool")
        self.assertEqual(tool_msg["tool_call_id"], "toolu_abc")
        self.assertEqual(tool_msg["content"], "42")

    def test_assistant_tool_use_block(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "search",
                            "input": {"q": "python"},
                        }
                    ],
                }
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertIn("tool_calls", asst)
        self.assertEqual(asst["tool_calls"][0]["id"], "toolu_01")
        self.assertEqual(asst["tool_calls"][0]["function"]["name"], "search")

    def test_thinking_blocks_stripped(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me think..."},
                        {"type": "text", "text": "Answer"},
                    ],
                }
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertEqual(asst["content"], "Answer")

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    def test_image_base64(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc123",
                            },
                        }
                    ],
                }
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        img = result["messages"][0]["content"][0]
        self.assertEqual(img["type"], "image_url")
        self.assertTrue(img["image_url"]["url"].startswith("data:image/png;base64,"))

    # ------------------------------------------------------------------
    # Streaming flag
    # ------------------------------------------------------------------
    def test_stream_options_added(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        self.assertTrue(result["stream"])
        self.assertEqual(result.get("stream_options"), {"include_usage": True})


# ---------------------------------------------------------------------------
# OpenAI → Anthropic response
# ---------------------------------------------------------------------------

class TestOpenAIToAnthropic(unittest.TestCase):
    def test_simple_text(self):
        oai = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = openai_to_anthropic_response(oai, "openai/gpt-4o")
        self.assertEqual(result["type"], "message")
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(result["content"][0]["text"], "Hi there!")
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertEqual(result["usage"]["output_tokens"], 5)

    def test_tool_call(self):
        oai = {
            "id": "chatcmpl-456",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": '{"expr": "2+2"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        }
        result = openai_to_anthropic_response(oai, "openai/gpt-4o")
        self.assertEqual(result["stop_reason"], "tool_use")
        tool = result["content"][0]
        self.assertEqual(tool["type"], "tool_use")
        self.assertEqual(tool["id"], "call_abc")
        self.assertEqual(tool["name"], "calculator")
        self.assertEqual(tool["input"], {"expr": "2+2"})

    def test_max_tokens_stop_reason(self):
        oai = {
            "id": "x",
            "choices": [
                {"message": {"role": "assistant", "content": "..."}, "finish_reason": "length"}
            ],
            "usage": {},
        }
        result = openai_to_anthropic_response(oai, "openai/gpt-4o")
        self.assertEqual(result["stop_reason"], "max_tokens")


# ---------------------------------------------------------------------------
# OpenAI SSE → Anthropic SSE streaming
# ---------------------------------------------------------------------------

class TestStreamConversion(unittest.TestCase):
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    async def _collect(self, chunks):
        """Collect all SSE events from an async generator over raw bytes."""
        async def _source():
            for c in chunks:
                yield c

        events = []
        async for raw in stream_openai_to_anthropic(_source(), "openai/gpt-4o", msg_id="msg_test"):
            for line in raw.decode().split("\n"):
                if line.startswith("event: "):
                    events.append(("event", line[7:]))
                elif line.startswith("data: "):
                    events.append(("data", json.loads(line[6:])))
        return events

    def _events_by_type(self, events):
        data_events = [d for k, d in events if k == "data"]
        return {e["type"]: e for e in data_events}

    def test_basic_text_stream(self):
        chunks = [
            b'data: {"id":"c1","choices":[{"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',
            b'data: {"id":"c1","choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n\n',
            b'data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        types = [d["type"] for k, d in events if k == "data"]
        self.assertIn("message_start", types)
        self.assertIn("ping", types)
        self.assertIn("content_block_start", types)
        self.assertIn("content_block_delta", types)
        self.assertIn("content_block_stop", types)
        self.assertIn("message_delta", types)
        self.assertIn("message_stop", types)
        text_deltas = [d for k, d in events if k == "data" and d.get("type") == "content_block_delta"]
        combined = "".join(d["delta"]["text"] for d in text_deltas)
        self.assertEqual(combined, "Hello world")

    def test_tool_call_stream(self):
        chunks = [
            b'data: {"id":"c2","choices":[{"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"calculator","arguments":""}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"c2","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"expr\\""}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"c2","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": ": \\"2+2\\""}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"c2","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        types = [d["type"] for k, d in events if k == "data"]
        self.assertIn("content_block_start", types)
        starts = [d for k, d in events if k == "data" and d.get("type") == "content_block_start"]
        tool_start = next((s for s in starts if s["content_block"].get("type") == "tool_use"), None)
        self.assertIsNotNone(tool_start)
        self.assertEqual(tool_start["content_block"]["name"], "calculator")
        msg_delta = next(d for k, d in events if k == "data" and d.get("type") == "message_delta")
        self.assertEqual(msg_delta["delta"]["stop_reason"], "tool_use")

    def test_stream_ends_with_message_stop(self):
        chunks = [
            b'data: {"id":"c3","choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        last_event = [d for k, d in events if k == "data"][-1]
        self.assertEqual(last_event["type"], "message_stop")

    def test_reasoning_delta_becomes_thinking_block(self):
        """OpenAI reasoning_content → Anthropic thinking block in streaming."""
        chunks = [
            b'data: {"id":"c4","choices":[{"delta":{"reasoning":"Let me think..."},"finish_reason":null}]}\n\n',
            b'data: {"id":"c4","choices":[{"delta":{"content":"Answer"},"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]
        block_starts = [d for d in data_events if d["type"] == "content_block_start"]
        thinking_start = next((s for s in block_starts if s["content_block"]["type"] == "thinking"), None)
        self.assertIsNotNone(thinking_start)
        text_start = next((s for s in block_starts if s["content_block"]["type"] == "text"), None)
        self.assertIsNotNone(text_start)

    def test_deferred_tool_block_start(self):
        """Tool content_block_start should be deferred until both id AND name are known."""
        # First chunk: id but no name. Second chunk: name arrives.
        chunks = [
            b'data: {"id":"c5","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"","arguments":""}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"c5","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"search","arguments":"{\\"q\\":"}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"c5","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"hello\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]
        # The content_block_start for tool_use should have a non-empty name
        tool_starts = [d for d in data_events
                       if d["type"] == "content_block_start"
                       and d.get("content_block", {}).get("type") == "tool_use"]
        self.assertTrue(len(tool_starts) > 0)
        self.assertEqual(tool_starts[0]["content_block"]["name"], "search")


# ---------------------------------------------------------------------------
# Helpers: sanitize, clean_schema, strip_private, o-series detection
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    def test_sanitize_billing_header(self):
        text = "You are helpful.\nx-anthropic-billing-header: cc_version=2.1\nDo good work."
        result = sanitize_system_text(text)
        self.assertNotIn("x-anthropic-billing-header", result)
        self.assertIn("You are helpful.", result)
        self.assertIn("Do good work.", result)

    def test_sanitize_no_header(self):
        text = "Simple system prompt."
        self.assertEqual(sanitize_system_text(text), text)

    def test_clean_schema_removes_uri_format(self):
        schema = {
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri"},
                "name": {"type": "string", "format": "date-time"},
            },
        }
        result = clean_schema(schema)
        self.assertNotIn("format", result["properties"]["url"])
        self.assertIn("format", result["properties"]["name"])  # date-time kept

    def test_clean_schema_nested(self):
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"link": {"type": "string", "format": "uri-reference"}},
                }
            },
        }
        result = clean_schema(schema)
        self.assertNotIn("format", result["properties"]["nested"]["properties"]["link"])

    def test_strip_private_params(self):
        body = {"model": "x", "_billing": "abc", "max_tokens": 100, "_internal": True}
        result = strip_private_params(body)
        self.assertNotIn("_billing", result)
        self.assertNotIn("_internal", result)
        self.assertIn("model", result)
        self.assertIn("max_tokens", result)

    def test_is_o_series(self):
        self.assertTrue(is_o_series("o1"))
        self.assertTrue(is_o_series("o1-mini"))
        self.assertTrue(is_o_series("o3"))
        self.assertTrue(is_o_series("o4-mini"))
        self.assertFalse(is_o_series("gpt-4o"))
        self.assertFalse(is_o_series("claude-sonnet-4-5"))

    def test_supports_reasoning_effort(self):
        self.assertTrue(supports_reasoning_effort("o1"))
        self.assertTrue(supports_reasoning_effort("o3-mini"))
        self.assertTrue(supports_reasoning_effort("gpt-5"))
        self.assertFalse(supports_reasoning_effort("gpt-4o"))


class TestAnthropicToOpenAIExtended(unittest.TestCase):
    def test_billing_header_stripped_from_system(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "system": "Be helpful.\nx-anthropic-billing-header: cc_version=2.1.0; plan=pro\nDo your best.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        sys_msg = result["messages"][0]
        self.assertEqual(sys_msg["role"], "system")
        self.assertNotIn("x-anthropic-billing-header", sys_msg["content"])
        self.assertIn("Be helpful.", sys_msg["content"])

    def test_batch_tool_filtered(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {"name": "real_tool", "description": "ok", "input_schema": {}},
                {"name": "BatchTool", "type": "BatchTool", "description": "batch", "input_schema": {}},
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        names = [t["function"]["name"] for t in result["tools"]]
        self.assertIn("real_tool", names)
        self.assertNotIn("BatchTool", names)

    def test_uri_format_stripped_from_tool_schema(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {
                    "name": "fetch",
                    "description": "fetch url",
                    "input_schema": {
                        "type": "object",
                        "properties": {"url": {"type": "string", "format": "uri"}},
                    },
                }
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        url_prop = result["tools"][0]["function"]["parameters"]["properties"]["url"]
        self.assertNotIn("format", url_prop)

    def test_o_series_uses_max_completion_tokens(self):
        body = {
            "model": "x",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_openai_request(body, "o3-mini")
        self.assertIn("max_completion_tokens", result)
        self.assertNotIn("max_tokens", result)
        self.assertEqual(result["max_completion_tokens"], 2048)

    def test_non_o_series_uses_max_tokens(self):
        body = {
            "model": "x",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        self.assertIn("max_tokens", result)
        self.assertNotIn("max_completion_tokens", result)

    def test_thinking_maps_to_reasoning_effort(self):
        body = {
            "model": "x",
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
            "messages": [{"role": "user", "content": "solve it"}],
        }
        result = anthropic_to_openai_request(body, "o3")
        self.assertIn("reasoning_effort", result)
        self.assertEqual(result["reasoning_effort"], "medium")

    def test_adaptive_thinking_maps_to_xhigh_for_gpt5(self):
        """thinking.type == 'adaptive' should produce reasoning_effort 'xhigh' for gpt-5 models."""
        body = {
            "model": "x",
            "max_tokens": 1024,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_openai_request(body, "gpt-5.4")
        self.assertEqual(result.get("reasoning_effort"), "xhigh")

    def test_thinking_block_not_emitted_as_reasoning_content_by_default(self):
        """thinking blocks in assistant history must NOT produce reasoning_content in OpenAI format."""
        body = {
            "model": "x",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I should call the tool."},
                        {"type": "tool_use", "id": "call_1", "name": "get_weather",
                         "input": {"city": "Tokyo"}},
                    ],
                }
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        msg = result["messages"][0]
        self.assertIsNone(msg.get("reasoning_content"))
        # The tool_call should still be present
        self.assertEqual(len(msg.get("tool_calls", [])), 1)
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_weather")

    def test_private_params_stripped(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "_blocking": True,
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        self.assertNotIn("_blocking", result)

    def test_tool_result_list_content_serialized(self):
        """tool_result with a list of content blocks should be serialized properly."""
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "Result: 42"},
                                {"type": "text", "text": "Extra info"},
                            ],
                        }
                    ],
                }
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        tool_msg = next(m for m in result["messages"] if m["role"] == "tool")
        self.assertIn("Result: 42", tool_msg["content"])
        self.assertIn("Extra info", tool_msg["content"])


class TestOpenAIToAnthropicExtended(unittest.TestCase):
    def test_reasoning_content_becomes_thinking_block(self):
        oai = {
            "id": "x",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Here's my answer.",
                        "reasoning_content": "Let me think step by step...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = openai_to_anthropic_response(oai, "openai/o3")
        types = [b["type"] for b in result["content"]]
        self.assertIn("thinking", types)
        self.assertIn("text", types)
        thinking = next(b for b in result["content"] if b["type"] == "thinking")
        self.assertEqual(thinking["thinking"], "Let me think step by step...")
        # thinking must come before text
        self.assertLess(types.index("thinking"), types.index("text"))

    def test_refusal_becomes_text_block(self):
        oai = {
            "id": "x",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "refusal": "I cannot help with that.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        result = openai_to_anthropic_response(oai, "openai/gpt-4o")
        text_block = next((b for b in result["content"] if b["type"] == "text"), None)
        self.assertIsNotNone(text_block)
        self.assertIn("cannot help", text_block["text"])

    def test_legacy_function_call(self):
        oai = {
            "id": "x",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "my_func",
                            "arguments": '{"key": "value"}',
                        },
                    },
                    "finish_reason": "function_call",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 8},
        }
        result = openai_to_anthropic_response(oai, "openai/gpt-3.5")
        self.assertEqual(result["stop_reason"], "tool_use")
        tool = next(b for b in result["content"] if b["type"] == "tool_use")
        self.assertEqual(tool["name"], "my_func")
        self.assertEqual(tool["input"], {"key": "value"})

    def test_cache_tokens_from_prompt_tokens_details(self):
        oai = {
            "id": "x",
            "choices": [
                {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
        }
        result = openai_to_anthropic_response(oai, "openai/gpt-4o")
        self.assertEqual(result["usage"].get("cache_read_input_tokens"), 80)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    def test_load_config(self):
        import tempfile, os
        yaml_content = """
server:
  port: 9000
providers:
  anthropic:
    type: anthropic
    api_key: test-key
    models:
      - claude-sonnet-4-5
  openai:
    type: openai
    api_key: sk-test
    models:
      - gpt-4o
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            from simple_api_router.config import load_config
            cfg = load_config(path)
            self.assertEqual(cfg.server.port, 9000)
            self.assertIn("anthropic", cfg.providers)
            self.assertIn("openai", cfg.providers)
            self.assertEqual(cfg.providers["anthropic"].type, "anthropic")
        finally:
            os.unlink(path)

    def test_env_expansion(self):
        import os, tempfile
        os.environ["TEST_KEY_CONV"] = "sk-expanded"
        yaml_content = """
providers:
  test:
    type: anthropic
    api_key: "${TEST_KEY_CONV}"
    models: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            from simple_api_router.config import load_config
            cfg = load_config(path)
            self.assertEqual(cfg.providers["test"].api_key, "sk-expanded")
        finally:
            os.unlink(path)
            del os.environ["TEST_KEY_CONV"]


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------

class TestProxy(unittest.TestCase):
    def test_parse_model_with_slash(self):
        from simple_api_router.proxy import parse_model
        provider, model = parse_model("anthropic/claude-opus-4-5")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(model, "claude-opus-4-5")

    def test_parse_model_without_slash(self):
        from simple_api_router.proxy import parse_model
        provider, model = parse_model("claude-sonnet-4-5")
        self.assertIsNone(provider)
        self.assertEqual(model, "claude-sonnet-4-5")

    def test_resolve_provider_by_name(self):
        from simple_api_router.config import ProviderConfig, RouterConfig
        from simple_api_router.proxy import resolve_provider
        cfg = RouterConfig(providers={
            "anthropic": ProviderConfig(type="anthropic", api_key="k", models=["claude-opus-4-5"]),
        })
        prov, bmodel = resolve_provider("anthropic", "claude-opus-4-5", cfg)
        self.assertEqual(prov.type, "anthropic")
        self.assertEqual(bmodel, "claude-opus-4-5")

    def test_resolve_provider_not_found(self):
        from fastapi import HTTPException
        from simple_api_router.config import RouterConfig
        from simple_api_router.proxy import resolve_provider
        cfg = RouterConfig(providers={})
        with self.assertRaises(HTTPException):
            resolve_provider("nonexistent", "model", cfg)

    def test_model_map_remapping(self):
        from simple_api_router.config import ProviderConfig, RouterConfig
        from simple_api_router.proxy import resolve_provider
        cfg = RouterConfig(providers={
            "myapi": ProviderConfig(
                type="openai",
                api_key="k",
                models=["my-fast"],
                model_map={"my-fast": "gpt-4o-mini"},
            )
        })
        prov, bmodel = resolve_provider("myapi", "my-fast", cfg)
        self.assertEqual(bmodel, "gpt-4o-mini")

    def test_parse_model_strips_1m_suffix(self):
        from simple_api_router.proxy import parse_model
        provider, model = parse_model("anthropic/claude-opus-4-5[1m]")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(model, "claude-opus-4-5")

    def test_parse_model_strips_other_bracket_suffix(self):
        from simple_api_router.proxy import parse_model
        provider, model = parse_model("openai/gpt-4o[128k]")
        self.assertEqual(provider, "openai")
        self.assertEqual(model, "gpt-4o")

    def test_parse_model_no_suffix_unchanged(self):
        from simple_api_router.proxy import parse_model
        provider, model = parse_model("anthropic/claude-sonnet-4-5")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(model, "claude-sonnet-4-5")

    def test_strip_model_suffixes_standalone(self):
        from simple_api_router.proxy import strip_model_suffixes
        self.assertEqual(strip_model_suffixes("model[1m]"), "model")
        self.assertEqual(strip_model_suffixes("model[4k]"), "model")
        self.assertEqual(strip_model_suffixes("model"), "model")

    def test_resolve_provider_strips_bracket_suffix(self):
        from simple_api_router.config import ProviderConfig, RouterConfig
        from simple_api_router.proxy import resolve_provider
        cfg = RouterConfig(providers={
            "anthropic": ProviderConfig(
                type="anthropic", api_key="k", models=["claude-opus-4-5"]
            ),
        })
        # Suffix is already stripped by parse_model before reaching resolve_provider,
        # so this tests that the stripped name resolves correctly.
        prov, bmodel = resolve_provider("anthropic", "claude-opus-4-5", cfg)
        self.assertEqual(bmodel, "claude-opus-4-5")


# ---------------------------------------------------------------------------
# cc-switch-ported tests: transform / sanitize / cache_control
# ---------------------------------------------------------------------------

class TestCCSwitchTransform(unittest.TestCase):
    """Tests ported from cc-switch-cli transform.rs test suite."""

    def test_sanitize_system_text_preserves_exact_whitespace(self):
        """Billing header with leading whitespace is stripped; surrounding newlines preserved."""
        text = "First line\n  x-anthropic-billing-header: cc_version=2.1\n\nLast line\n"
        result = sanitize_system_text(text)
        self.assertNotIn("x-anthropic-billing-header", result)
        self.assertIn("First line", result)
        self.assertIn("Last line", result)
        # The blank line between the header and "Last line" should be preserved
        self.assertIn("\n\n", result)

    def test_billing_header_only_block_is_dropped(self):
        """A system block containing only a billing header (empty after sanitize) is dropped."""
        body = {
            "model": "x",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: v=1"},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        sys_msgs = [m for m in result["messages"] if m["role"] == "system"]
        self.assertEqual(len(sys_msgs), 0)

    def test_cache_control_preserved_in_system_block(self):
        """cache_control on a system text block is passed through to the OpenAI message."""
        body = {
            "model": "x",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        sys_msg = next(m for m in result["messages"] if m["role"] == "system")
        self.assertEqual(sys_msg.get("cache_control"), {"type": "ephemeral"})

    def test_cache_control_forces_array_format_in_user_block(self):
        """User text block with cache_control should appear as an array item (not a plain string)."""
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}},
                    ],
                }
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        user_content = result["messages"][0]["content"]
        # Must be an array (not a plain string) so cache_control can be attached
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[0].get("cache_control"), {"type": "ephemeral"})
        self.assertEqual(user_content[0]["text"], "hello")

    def test_cache_control_preserved_in_tool(self):
        """cache_control on a tool definition is passed through to the OpenAI tool object."""
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {
                    "name": "search",
                    "description": "search the web",
                    "input_schema": {"type": "object", "properties": {}},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        tool = result["tools"][0]
        self.assertEqual(tool.get("cache_control"), {"type": "ephemeral"})


# ---------------------------------------------------------------------------
# cc-switch-ported tests: streaming edge cases
# ---------------------------------------------------------------------------

class TestCCSwitchStreaming(unittest.TestCase):
    """Tests ported from cc-switch-cli streaming.rs test suite."""

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    async def _collect(self, chunks):
        async def gen():
            for c in chunks:
                yield c
        events = []
        async for raw in stream_openai_to_anthropic(gen(), "openai/gpt-4o", msg_id="msg_test"):
            for line in raw.decode().split("\n"):
                if line.startswith("event: "):
                    events.append(("event", line[7:]))
                elif line.startswith("data: "):
                    events.append(("data", json.loads(line[6:])))
        return events

    def test_empty_content_delta_does_not_open_text_block(self):
        """An empty string delta must not open a text content_block_start."""
        chunks = [
            b'data: {"id":"c1","choices":[{"delta":{"content":""},"finish_reason":null}]}\n\n',
            b'data: {"id":"c1","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]
        block_starts = [d for d in data_events if d["type"] == "content_block_start"]
        # Only one text block should be opened (for "hello"), not one for ""
        self.assertEqual(len(block_starts), 1)

    def test_streaming_usage_zero_cached_tokens_preserved(self):
        """cache_read_input_tokens=0 must be emitted (not silently dropped)."""
        chunks = [
            b'data: {"id":"c2","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n',
            b'data: {"id":"c2","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3,"cache_read_input_tokens":0}}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]
        delta = next((d for d in data_events if d["type"] == "message_delta"), None)
        self.assertIsNotNone(delta)
        usage = delta.get("usage", {})
        self.assertIn("cache_read_input_tokens", usage)
        self.assertEqual(usage["cache_read_input_tokens"], 0)

    def test_streaming_tool_calls_route_arguments_by_index(self):
        """Interleaved tool_calls by index must route arguments to the correct tool block."""
        chunks = [
            b'data: {"id":"c3","choices":[{"delta":{"tool_calls":[{"index":0,"id":"id0","function":{"name":"tool_a","arguments":""}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"c3","choices":[{"delta":{"tool_calls":[{"index":1,"id":"id1","function":{"name":"tool_b","arguments":""}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"c3","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"x\\":1}"}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"c3","choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"y\\":2}"}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"c3","choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]
        # Collect all input_json_delta events; the arguments must match index
        input_deltas = [d for d in data_events if d["type"] == "content_block_delta"
                        and d.get("delta", {}).get("type") == "input_json_delta"]
        args_by_block = {}
        for d in input_deltas:
            idx = d["index"]
            args_by_block.setdefault(idx, "")
            args_by_block[idx] += d["delta"]["partial_json"]
        # Two tool blocks should have been opened and their args are separate
        self.assertEqual(len(args_by_block), 2)
        import json
        for idx, args_str in args_by_block.items():
            parsed = json.loads(args_str)
            if "x" in parsed:
                self.assertEqual(parsed["x"], 1)
            elif "y" in parsed:
                self.assertEqual(parsed["y"], 2)

    def test_finish_chunk_without_usage_still_has_message_delta(self):
        """A finish_reason chunk with no usage field should still produce a message_delta."""
        chunks = [
            b'data: {"id":"c4","choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n',
            b'data: {"id":"c4","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]
        delta = next((d for d in data_events if d["type"] == "message_delta"), None)
        self.assertIsNotNone(delta)
        self.assertEqual(delta.get("delta", {}).get("stop_reason"), "end_turn")

    def test_finish_reason_flushes_unfinished_tool_block(self):
        """finish_reason must flush a pending tool block even if stop reason is tool_calls."""
        chunks = [
            b'data: {"id":"c5","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"do_thing","arguments":"{\\"a\\":1}"}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"c5","choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]
        # A content_block_stop should be emitted for the tool block
        stops = [d for d in data_events if d["type"] == "content_block_stop"]
        self.assertTrue(len(stops) >= 1)
        # stop_reason in message_delta should be "tool_use"
        delta = next((d for d in data_events if d["type"] == "message_delta"), None)
        self.assertIsNotNone(delta)
        self.assertEqual(delta.get("delta", {}).get("stop_reason"), "tool_use")

    def test_finish_reason_flushes_anonymous_tool_with_fallback_id_and_name(self):
        """Tool chunks with no id/name must still be flushed with synthetic fallback values."""
        # Mirrors cc-switch streaming_tool_calls_finish_reason_flushes_pending_and_closes_in_order
        chunks = [
            b'data: {"id":"c6","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\":1}"}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"c6","choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]

        starts = [d for d in data_events if d["type"] == "content_block_start"
                  and d.get("content_block", {}).get("type") == "tool_use"]
        deltas = [d for d in data_events if d["type"] == "content_block_delta"
                  and d.get("delta", {}).get("type") == "input_json_delta"]
        stops = [d for d in data_events if d["type"] == "content_block_stop"]
        msg_delta = next((d for d in data_events if d["type"] == "message_delta"), None)

        # Fallback id and name must be synthetic
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["content_block"]["id"], "tool_call_0")
        self.assertEqual(starts[0]["content_block"]["name"], "unknown_tool")
        self.assertEqual(starts[0]["content_block"]["input"], {})

        # Pending args must be flushed
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["delta"]["partial_json"], '{"a":1}')

        # Ordering: start < delta < stop < message_delta
        start_pos = data_events.index(starts[0])
        delta_pos = data_events.index(deltas[0])
        stop_pos = data_events.index(stops[0])
        msg_delta_pos = data_events.index(msg_delta)
        self.assertLess(start_pos, delta_pos)
        self.assertLess(delta_pos, stop_pos)
        self.assertLess(stop_pos, msg_delta_pos)
        self.assertEqual(msg_delta.get("delta", {}).get("stop_reason"), "tool_use")

    def test_streaming_accepts_deepseek_reasoning_content_alias(self):
        """reasoning_content in streaming delta → thinking_delta (DeepSeek alias)."""
        chunks = [
            b'data: {"id":"chatcmpl_1","model":"deepseek-v4-pro","choices":[{"delta":{"reasoning_content":"think"}}]}\n\n',
            b'data: {"id":"chatcmpl_1","model":"deepseek-v4-pro","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]
        thinking_delta = next(
            (d for d in data_events if d["type"] == "content_block_delta"),
            None,
        )
        self.assertIsNotNone(thinking_delta, "thinking delta should be emitted")
        self.assertEqual(thinking_delta["delta"]["type"], "thinking_delta")
        self.assertEqual(thinking_delta["delta"]["thinking"], "think")

    def test_empty_content_delta_does_not_open_text_block(self):
        """An empty string content delta must not open a text block."""
        chunks = [
            b'data: {"id":"chatcmpl_1","model":"gpt-4o","choices":[{"delta":{"content":""}}]}\n\n',
            b'data: {"id":"chatcmpl_1","model":"gpt-4o","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":0}}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]
        text_starts = [
            d for d in data_events
            if d["type"] == "content_block_start"
            and d.get("content_block", {}).get("type") == "text"
        ]
        self.assertEqual(len(text_starts), 0, "empty content delta should not open a text block")


# ---------------------------------------------------------------------------
# DeepSeek reasoning_content passthrough
# ---------------------------------------------------------------------------

from simple_api_router.converter import is_deepseek_model  # noqa: E402


class TestDeepSeekReasoning(unittest.TestCase):
    def test_is_deepseek_model_positive(self):
        self.assertTrue(is_deepseek_model("deepseek-chat"))
        self.assertTrue(is_deepseek_model("DeepSeek-R1"))
        self.assertTrue(is_deepseek_model("deepseek-reasoner"))

    def test_is_deepseek_model_negative(self):
        self.assertFalse(is_deepseek_model("gpt-4o"))
        self.assertFalse(is_deepseek_model("claude-opus-4-5"))

    def test_thinking_block_becomes_reasoning_content(self):
        """When use_reasoning_content=True, thinking → reasoning_content on assistant message."""
        body = {
            "model": "deepseek/deepseek-chat",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me think..."},
                        {"type": "text", "text": "Hi there!"},
                    ],
                },
                {"role": "user", "content": "Again"},
            ],
        }
        result = anthropic_to_openai_request(body, "deepseek-chat", use_reasoning_content=True)
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertEqual(asst["reasoning_content"], "Let me think...")
        self.assertEqual(asst["content"], "Hi there!")

    def test_no_thinking_block_no_tools_no_reasoning_content(self):
        """Without thinking and without tools, no reasoning_content even in deepseek mode."""
        body = {
            "model": "deepseek/deepseek-chat",
            "max_tokens": 100,
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
            ],
        }
        result = anthropic_to_openai_request(body, "deepseek-chat", use_reasoning_content=True)
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertNotIn("reasoning_content", asst)

    def test_tool_use_without_thinking_gets_placeholder(self):
        """Tool-only assistant message without thinking gets placeholder reasoning_content."""
        body = {
            "model": "deepseek/deepseek-chat",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "search",
                            "input": {"q": "weather"},
                        }
                    ],
                },
            ],
        }
        result = anthropic_to_openai_request(body, "deepseek-chat", use_reasoning_content=True)
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertEqual(asst["reasoning_content"], "tool call")
        self.assertEqual(len(asst["tool_calls"]), 1)

    def test_thinking_not_emitted_when_disabled(self):
        """With use_reasoning_content=False (default), thinking blocks are silently dropped."""
        body = {
            "model": "gpt-4o",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "secret"},
                        {"type": "text", "text": "Result"},
                    ],
                },
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertNotIn("reasoning_content", asst)
        self.assertEqual(asst["content"], "Result")

    def test_thinking_and_tool_use_together(self):
        """Thinking + tool_use: reasoning_content gets thinking text, tool_calls present."""
        body = {
            "model": "deepseek/deepseek-chat",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I need to search"},
                        {
                            "type": "tool_use",
                            "id": "toolu_02",
                            "name": "lookup",
                            "input": {"key": "val"},
                        },
                    ],
                },
            ],
        }
        result = anthropic_to_openai_request(body, "deepseek-chat", use_reasoning_content=True)
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertEqual(asst["reasoning_content"], "I need to search")
        self.assertEqual(len(asst["tool_calls"]), 1)
        self.assertEqual(asst["tool_calls"][0]["function"]["name"], "lookup")

    def test_openai_to_anthropic_maps_reasoning_content_to_thinking_block(self):
        """OpenAI response reasoning_content → thinking block first, then text, then tool_use."""
        oai_response = {
            "id": "chatcmpl-deepseek",
            "model": "deepseek-v4-flash",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Need the current date before calling weather.",
                    "content": "Let me check.",
                    "tool_calls": [{
                        "id": "call_date",
                        "type": "function",
                        "function": {"name": "get_date", "arguments": "{}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = openai_to_anthropic_response(oai_response, "deepseek-v4-flash")
        self.assertEqual(result["content"][0]["type"], "thinking")
        self.assertEqual(result["content"][0]["thinking"], "Need the current date before calling weather.")
        self.assertEqual(result["content"][1]["type"], "text")
        self.assertEqual(result["content"][2]["type"], "tool_use")

    def test_deepseek_reasoning_content_round_trips_for_tool_calls(self):
        """reasoning_content survives the full OpenAI → Anthropic → OpenAI round-trip."""
        upstream_response = {
            "id": "chatcmpl-deepseek",
            "model": "deepseek-v4-flash",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Need the current date before calling weather.",
                    "content": "Let me check.",
                    "tool_calls": [{
                        "id": "call_date",
                        "type": "function",
                        "function": {"name": "get_date", "arguments": "{}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        # Step 1: OpenAI response → Anthropic
        ant_response = openai_to_anthropic_response(upstream_response, "deepseek-v4-flash")
        # Step 2: Anthropic content used as follow-up request → OpenAI with reasoning passthrough
        follow_up = {
            "model": "deepseek-v4-flash",
            "max_tokens": 100,
            "messages": [{"role": "assistant", "content": ant_response["content"]}],
        }
        replayed = anthropic_to_openai_request(follow_up, "deepseek-v4-flash", use_reasoning_content=True)
        msg = replayed["messages"][0]
        self.assertEqual(msg["reasoning_content"], "Need the current date before calling weather.")
        self.assertEqual(msg["tool_calls"][0]["id"], "call_date")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_date")


# ---------------------------------------------------------------------------
# Responses API — request conversion
# ---------------------------------------------------------------------------

from simple_api_router.converter import anthropic_to_responses_request  # noqa: E402


class TestResponsesAPIRequest(unittest.TestCase):
    def test_basic_text(self):
        body = {
            "model": "openai/gpt-5",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        self.assertEqual(result["model"], "gpt-5")
        self.assertEqual(result["max_output_tokens"], 512)
        self.assertIn("input", result)
        inp = result["input"]
        self.assertEqual(len(inp), 1)
        self.assertEqual(inp[0]["role"], "user")
        self.assertEqual(inp[0]["content"][0]["type"], "input_text")
        self.assertEqual(inp[0]["content"][0]["text"], "Hello")

    def test_system_becomes_instructions(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        self.assertEqual(result["instructions"], "You are helpful.")

    def test_assistant_text_becomes_output_text(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
            ],
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        asst_item = result["input"][-1]
        self.assertEqual(asst_item["role"], "assistant")
        self.assertEqual(asst_item["content"][0]["type"], "output_text")
        self.assertEqual(asst_item["content"][0]["text"], "Hello!")

    def test_tool_use_becomes_function_call(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "search",
                            "input": {"q": "test"},
                        }
                    ],
                }
            ],
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        fn_call = result["input"][0]
        self.assertEqual(fn_call["type"], "function_call")
        self.assertEqual(fn_call["call_id"], "toolu_1")
        self.assertEqual(fn_call["name"], "search")
        self.assertEqual(json.loads(fn_call["arguments"]), {"q": "test"})

    def test_tool_result_becomes_function_call_output(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "result text",
                        }
                    ],
                }
            ],
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        fn_out = result["input"][0]
        self.assertEqual(fn_out["type"], "function_call_output")
        self.assertEqual(fn_out["call_id"], "toolu_1")
        self.assertEqual(fn_out["output"], "result text")

    def test_thinking_becomes_reasoning(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Think"}],
            "thinking": {"type": "enabled", "budget_tokens": 5000},
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        self.assertIn("reasoning", result)
        self.assertEqual(result["reasoning"]["effort"], "medium")
        self.assertEqual(result["reasoning"]["summary"], "auto")

    def test_adaptive_thinking_becomes_high_effort(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Think"}],
            "thinking": {"type": "adaptive"},
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        self.assertEqual(result["reasoning"]["effort"], "high")

    def test_tool_choice_any_becomes_required(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"name": "fn", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
            "tool_choice": {"type": "any"},
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        self.assertEqual(result["tool_choice"], "required")

    def test_thinking_block_in_messages_is_skipped(self):
        """Thinking blocks inside conversation messages must be skipped (not added to input)."""
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private"},
                        {"type": "text", "text": "Hello"},
                    ],
                }
            ],
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        item = result["input"][0]
        self.assertEqual(item["role"], "assistant")
        self.assertEqual(len(item["content"]), 1)
        self.assertEqual(item["content"][0]["type"], "output_text")


# ---------------------------------------------------------------------------
# Responses API — non-streaming response conversion
# ---------------------------------------------------------------------------

from simple_api_router.converter import responses_to_anthropic_response  # noqa: E402


class TestResponsesAPIResponse(unittest.TestCase):
    def test_simple_text_response(self):
        body = {
            "id": "resp_abc",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = responses_to_anthropic_response(body, "gpt-5", "msg_1")
        self.assertEqual(result["id"], "msg_1")
        self.assertEqual(result["model"], "gpt-5")
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(result["content"][0]["text"], "Hello!")
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertEqual(result["usage"]["output_tokens"], 5)

    def test_function_call_response(self):
        body = {
            "id": "resp_2",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "search",
                    "arguments": '{"q": "test"}',
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        result = responses_to_anthropic_response(body, "gpt-5")
        self.assertEqual(result["stop_reason"], "tool_use")
        tu = result["content"][0]
        self.assertEqual(tu["type"], "tool_use")
        self.assertEqual(tu["id"], "call_1")
        self.assertEqual(tu["name"], "search")
        self.assertEqual(tu["input"], {"q": "test"})

    def test_reasoning_item_becomes_thinking(self):
        body = {
            "id": "resp_3",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "I thought about this."}],
                }
            ],
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
        result = responses_to_anthropic_response(body, "gpt-5")
        self.assertEqual(result["content"][0]["type"], "thinking")
        self.assertEqual(result["content"][0]["thinking"], "I thought about this.")

    def test_incomplete_max_tokens(self):
        body = {
            "id": "resp_4",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        result = responses_to_anthropic_response(body, "gpt-5")
        self.assertEqual(result["stop_reason"], "max_tokens")

    def test_cache_read_tokens(self):
        body = {
            "id": "resp_5",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "input_tokens_details": {"cached_tokens": 80},
            },
        }
        result = responses_to_anthropic_response(body, "gpt-5")
        self.assertEqual(result["usage"]["cache_read_input_tokens"], 80)


# ---------------------------------------------------------------------------
# Responses API — streaming conversion
# ---------------------------------------------------------------------------

from simple_api_router.converter import stream_responses_to_anthropic  # noqa: E402


def _run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _collect_responses_sse(chunks):
    async def _gen():
        for c in chunks:
            yield c

    events = []
    async for raw in stream_responses_to_anthropic(_gen(), "gpt-5", "msg_test"):
        decoded = raw.decode()
        lines = decoded.strip().split("\n")
        event_name = None
        data_str = None
        for line in lines:
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                data_str = line[6:]
        if event_name and data_str:
            events.append((event_name, json.loads(data_str)))
    return events


def _make_sse(event, data) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class TestResponsesAPIStreaming(unittest.TestCase):
    def test_basic_text_stream(self):
        chunks = [
            _make_sse("response.created", {
                "type": "response.created",
                "response": {
                    "id": "resp_x",
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            }),
            _make_sse("response.content_part.added", {
                "type": "response.content_part.added",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text"},
            }),
            _make_sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello",
            }),
            _make_sse("response.content_part.done", {
                "type": "response.content_part.done",
                "output_index": 0,
                "content_index": 0,
            }),
            _make_sse("response.completed", {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
            }),
        ]
        events = _run_async(_collect_responses_sse(chunks))
        types = [e for e, _ in events]
        self.assertIn("message_start", types)
        self.assertIn("content_block_start", types)
        self.assertIn("content_block_delta", types)
        self.assertIn("content_block_stop", types)
        self.assertIn("message_delta", types)
        self.assertIn("message_stop", types)

        msg_start = next(d for e, d in events if e == "message_start")
        self.assertEqual(msg_start["message"]["usage"]["input_tokens"], 5)

        delta = next(d for e, d in events if e == "content_block_delta")
        self.assertEqual(delta["delta"]["type"], "text_delta")
        self.assertEqual(delta["delta"]["text"], "Hello")

        msg_delta = next(d for e, d in events if e == "message_delta")
        self.assertEqual(msg_delta["delta"]["stop_reason"], "end_turn")

    def test_function_call_stream(self):
        chunks = [
            _make_sse("response.created", {
                "type": "response.created",
                "response": {"id": "resp_y", "usage": {"input_tokens": 10, "output_tokens": 0}},
            }),
            _make_sse("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_abc", "name": "lookup"},
            }),
            _make_sse("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": '{"q":',
            }),
            _make_sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
            }),
            _make_sse("response.completed", {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            }),
        ]
        events = _run_async(_collect_responses_sse(chunks))

        start = next(d for e, d in events if e == "content_block_start")
        self.assertEqual(start["content_block"]["type"], "tool_use")
        self.assertEqual(start["content_block"]["id"], "call_abc")
        self.assertEqual(start["content_block"]["name"], "lookup")

        delta = next(d for e, d in events if e == "content_block_delta")
        self.assertEqual(delta["delta"]["type"], "input_json_delta")
        self.assertEqual(delta["delta"]["partial_json"], '{"q":')

        msg_delta = next(d for e, d in events if e == "message_delta")
        self.assertEqual(msg_delta["delta"]["stop_reason"], "tool_use")

    def test_reasoning_stream(self):
        chunks = [
            _make_sse("response.created", {
                "type": "response.created",
                "response": {"id": "resp_z", "usage": {"input_tokens": 3, "output_tokens": 0}},
            }),
            _make_sse("response.reasoning.delta", {
                "type": "response.reasoning.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": {"type": "text", "text": "thinking..."},
            }),
            _make_sse("response.reasoning.done", {
                "type": "response.reasoning.done",
                "output_index": 0,
                "content_index": 0,
            }),
            _make_sse("response.completed", {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            }),
        ]
        events = _run_async(_collect_responses_sse(chunks))

        start = next(d for e, d in events if e == "content_block_start")
        self.assertEqual(start["content_block"]["type"], "thinking")

        delta = next(d for e, d in events if e == "content_block_delta")
        self.assertEqual(delta["delta"]["type"], "thinking_delta")
        self.assertEqual(delta["delta"]["thinking"], "thinking...")

    def test_message_stop_emitted_after_completed(self):
        chunks = [
            _make_sse("response.created", {
                "type": "response.created",
                "response": {"id": "r", "usage": {"input_tokens": 1, "output_tokens": 0}},
            }),
            _make_sse("response.completed", {
                "type": "response.completed",
                "response": {"status": "completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
            }),
        ]
        events = _run_async(_collect_responses_sse(chunks))
        types = [e for e, _ in events]
        self.assertIn("message_stop", types)
        md_pos = types.index("message_delta")
        ms_pos = types.index("message_stop")
        self.assertLess(md_pos, ms_pos)


# ---------------------------------------------------------------------------
# Config — api_format and deepseek_reasoning validation
# ---------------------------------------------------------------------------

from simple_api_router.config import ProviderConfig  # noqa: E402


class TestIsRealKey(unittest.TestCase):
    """Tests for proxy._is_real_key."""

    def setUp(self):
        from simple_api_router.proxy import _is_real_key
        self._is_real_key = _is_real_key

    def test_real_key_returns_true(self):
        self.assertTrue(self._is_real_key("sk-abc123"))

    def test_empty_string_returns_false(self):
        self.assertFalse(self._is_real_key(""))

    def test_none_literal_returns_false(self):
        self.assertFalse(self._is_real_key("none"))

    def test_none_upper_returns_false(self):
        self.assertFalse(self._is_real_key("NONE"))

    def test_null_literal_returns_false(self):
        self.assertFalse(self._is_real_key("null"))

    def test_false_literal_returns_false(self):
        self.assertFalse(self._is_real_key("false"))

    def test_no_literal_returns_false(self):
        self.assertFalse(self._is_real_key("no"))

    def test_zero_literal_returns_false(self):
        self.assertFalse(self._is_real_key("0"))


class TestBuildAnthropicHeaders(unittest.TestCase):
    """Tests for proxy._build_anthropic_headers dual-auth behaviour."""

    def _make_request(self, headers: dict):
        from starlette.testclient import TestClient
        from starlette.requests import Request
        from starlette.applications import Starlette
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "query_string": b"",
        }
        return Request(scope)

    def test_real_key_sends_both_auth_headers(self):
        from simple_api_router.proxy import _build_anthropic_headers
        from simple_api_router.config import ProviderConfig
        prov = ProviderConfig(type="anthropic", api_key="sk-real")
        req = self._make_request({"anthropic-version": "2023-06-01"})
        hdrs = _build_anthropic_headers(req, prov)
        self.assertEqual(hdrs["x-api-key"], "sk-real")
        self.assertEqual(hdrs["Authorization"], "Bearer sk-real")

    def test_fake_key_none_sends_no_auth(self):
        from simple_api_router.proxy import _build_anthropic_headers
        from simple_api_router.config import ProviderConfig
        prov = ProviderConfig(type="anthropic", api_key="none")
        req = self._make_request({})
        hdrs = _build_anthropic_headers(req, prov)
        self.assertNotIn("x-api-key", hdrs)
        self.assertNotIn("Authorization", hdrs)

    def test_empty_key_sends_no_auth(self):
        from simple_api_router.proxy import _build_anthropic_headers
        from simple_api_router.config import ProviderConfig
        prov = ProviderConfig(type="anthropic", api_key="")
        req = self._make_request({})
        hdrs = _build_anthropic_headers(req, prov)
        self.assertNotIn("x-api-key", hdrs)
        self.assertNotIn("Authorization", hdrs)

    def test_anthropic_version_forwarded(self):
        from simple_api_router.proxy import _build_anthropic_headers
        from simple_api_router.config import ProviderConfig
        prov = ProviderConfig(type="anthropic", api_key="sk-real")
        req = self._make_request({"anthropic-version": "2024-01-01"})
        hdrs = _build_anthropic_headers(req, prov)
        self.assertEqual(hdrs["anthropic-version"], "2024-01-01")

    def test_default_anthropic_version_injected_when_missing(self):
        from simple_api_router.proxy import _build_anthropic_headers
        from simple_api_router.config import ProviderConfig
        prov = ProviderConfig(type="anthropic", api_key="sk-real")
        req = self._make_request({})
        hdrs = _build_anthropic_headers(req, prov)
        self.assertEqual(hdrs["anthropic-version"], "2023-06-01")


class TestFindUnexpanded(unittest.TestCase):
    """Tests for config._find_unexpanded and load_config env-var validation."""

    def test_detects_unset_var_in_string(self):
        import os
        os.environ.pop("DEFINITELY_NOT_SET_XYZ", None)
        from simple_api_router.config import _find_unexpanded
        problems = _find_unexpanded({"api_key": "${DEFINITELY_NOT_SET_XYZ}"})
        self.assertEqual(len(problems), 1)
        self.assertIn("DEFINITELY_NOT_SET_XYZ", problems[0])

    def test_no_problems_when_var_is_set(self):
        import os
        os.environ["TEST_EXPANDED_VAR"] = "value"
        try:
            from simple_api_router.config import _find_unexpanded
            problems = _find_unexpanded({"api_key": "${TEST_EXPANDED_VAR}"})
            self.assertEqual(problems, [])
        finally:
            del os.environ["TEST_EXPANDED_VAR"]

    def test_detects_nested_unset_var(self):
        import os
        os.environ.pop("NESTED_UNSET_VAR", None)
        from simple_api_router.config import _find_unexpanded
        problems = _find_unexpanded({"providers": {"p": {"api_key": "${NESTED_UNSET_VAR}"}}})
        self.assertEqual(len(problems), 1)

    def test_load_config_raises_on_unset_var(self):
        import os, tempfile
        os.environ.pop("LOAD_CONFIG_MISSING_VAR", None)
        yaml_content = """
providers:
  test:
    type: anthropic
    api_key: "${LOAD_CONFIG_MISSING_VAR}"
    models: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            from simple_api_router.config import load_config
            with self.assertRaises(ValueError) as ctx:
                load_config(path)
            self.assertIn("LOAD_CONFIG_MISSING_VAR", str(ctx.exception))
        finally:
            os.unlink(path)


class TestRetryExhaustion(unittest.TestCase):
    """Tests for _post_with_retry and _streaming_request_with_retry exhaustion behaviour."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_post_with_retry_exhaustion_returns_last_response(self):
        """On exhaustion with HTTP errors, returns the last response (not None)."""
        import httpx
        from unittest.mock import AsyncMock
        from simple_api_router.proxy import _post_with_retry

        mock_resp = httpx.Response(429)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        resp, err = self._run(_post_with_retry(mock_client, "http://x", {}, {}, max_retries=1))
        self.assertIsNotNone(err)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.status_code, 429)

    def test_post_with_retry_network_error_returns_none_resp(self):
        """On exhaustion with network errors, returns None for resp."""
        import httpx
        from unittest.mock import AsyncMock
        from simple_api_router.proxy import _post_with_retry

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        resp, err = self._run(_post_with_retry(mock_client, "http://x", {}, {}, max_retries=0))
        self.assertIsNone(resp)
        self.assertIsNotNone(err)

    def test_streaming_retry_exhaustion_raises_last_status(self):
        """On exhaustion, _streaming_request_with_retry raises HTTPException with last HTTP status."""
        import httpx
        from unittest.mock import AsyncMock, MagicMock
        from fastapi import HTTPException
        from simple_api_router.proxy import _streaming_request_with_retry

        mock_resp = httpx.Response(503)
        mock_resp.aclose = AsyncMock()

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_resp)

        with self.assertRaises(HTTPException) as ctx:
            self._run(_streaming_request_with_retry(mock_client, "http://x", {}, {}, max_retries=1))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_streaming_network_error_exhaustion_raises_502(self):
        """On exhaustion with network errors, raises HTTPException(502)."""
        import httpx
        from unittest.mock import AsyncMock, MagicMock
        from fastapi import HTTPException
        from simple_api_router.proxy import _streaming_request_with_retry

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with self.assertRaises(HTTPException) as ctx:
            self._run(_streaming_request_with_retry(mock_client, "http://x", {}, {}, max_retries=0))
        self.assertEqual(ctx.exception.status_code, 502)


class TestProviderConfigExtended(unittest.TestCase):
    def test_default_api_format_is_openai_chat(self):
        cfg = ProviderConfig(type="openai", api_key="k")
        self.assertEqual(cfg.api_format, "openai_chat")

    def test_openai_responses_api_format_accepted(self):
        cfg = ProviderConfig(type="openai", api_key="k", api_format="openai_responses")
        self.assertEqual(cfg.api_format, "openai_responses")

    def test_invalid_api_format_raises(self):
        with self.assertRaises(Exception):
            ProviderConfig(type="openai", api_key="k", api_format="invalid_format")

    def test_deepseek_reasoning_default_none(self):
        cfg = ProviderConfig(type="openai", api_key="k")
        self.assertIsNone(cfg.deepseek_reasoning)

    def test_deepseek_reasoning_can_be_set(self):
        cfg = ProviderConfig(type="openai", api_key="k", deepseek_reasoning=True)
        self.assertTrue(cfg.deepseek_reasoning)


if __name__ == "__main__":
    unittest.main()
