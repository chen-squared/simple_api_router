"""Unit tests for converter.py — Anthropic ↔ OpenAI format conversion."""
from __future__ import annotations

import asyncio
import json
import unittest
from typing import AsyncIterator

from simple_api_router.converter import (
    clean_schema,
    is_o_series,
    sanitize_system_text,
    strip_private_params,
)
from simple_api_router.converter_openai import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
    stream_openai_to_anthropic,
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
        # Multiple system blocks are merged into one system message (OpenAI only allows one).
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
        self.assertEqual(len(sys_msgs), 1)
        self.assertIn("Part 1.", sys_msgs[0]["content"])
        self.assertIn("Part 2.", sys_msgs[0]["content"])

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

    def test_tool_choice_none_stays_none(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"name": "t", "description": "", "input_schema": {}}],
            "tool_choice": {"type": "none"},
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        self.assertEqual(result["tool_choice"], "none")

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

    def test_sequential_multi_tool_blocks_non_overlapping(self):
        """When a provider streams multiple tool calls sequentially (all of tool[0]'s
        args before tool[1]'s id/name), each content_block must be closed before the
        next is opened — Anthropic protocol forbids overlapping blocks.
        """
        # Simulates DeepSeek-style sequential tool streaming: idx=0 complete, then idx=1
        chunks = [
            # tool 0: id + name
            b'data: {"id":"r1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_A","type":"function","function":{"name":"Bash","arguments":""}}]},"finish_reason":null}]}\n\n',
            # tool 0: arguments
            b'data: {"id":"r1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"command\\":\\"ls\\"}"}}]},"finish_reason":null}]}\n\n',
            # tool 1: id + name  (tool 0 is now complete)
            b'data: {"id":"r1","choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_B","type":"function","function":{"name":"Read","arguments":""}}]},"finish_reason":null}]}\n\n',
            # tool 1: arguments
            b'data: {"id":"r1","choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"file_path\\":\\"/f\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]

        # Build ordered list of (event_type, index) for block lifecycle events
        lifecycle = []
        for d in data_events:
            t = d.get("type")
            if t in ("content_block_start", "content_block_stop"):
                lifecycle.append((t, d.get("index")))

        # Verify no two blocks are open at the same time.
        open_blocks = set()
        for evt, idx in lifecycle:
            if evt == "content_block_start":
                self.assertFalse(open_blocks,
                    f"Block {idx} opened while blocks {open_blocks} were still open (overlapping)")
                open_blocks.add(idx)
            else:
                open_blocks.discard(idx)

        # Exactly 2 tool_use blocks
        starts = [d for d in data_events if d.get("type") == "content_block_start"
                  and d.get("content_block", {}).get("type") == "tool_use"]
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[0]["content_block"]["name"], "Bash")
        self.assertEqual(starts[1]["content_block"]["name"], "Read")


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

    def test_streaming_error_in_chunk(self):
        """A streaming error object in a data chunk should emit an Anthropic error event."""
        chunks = [
            b'data: {"id":"c1","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            b'data: {"error":{"type":"api_error","message":"Content policy violation"}}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        event_types = [k for k, _ in events if k == "event"]
        data_events = [d for k, d in events if k == "data"]
        event_names = [v for k, v in events if k == "event"]

        # Should emit an 'error' SSE event (not message_delta/message_stop)
        self.assertIn("error", event_names)
        # The error data should contain our message
        error_data = [d for d in data_events if d.get("type") == "error"]
        self.assertTrue(len(error_data) > 0)
        self.assertIn("Content policy violation", error_data[0]["error"]["message"])
        # Should NOT emit message_delta or message_stop after an error
        data_types = [d["type"] for d in data_events]
        self.assertNotIn("message_delta", data_types)
        self.assertNotIn("message_stop", data_types)

    def test_streaming_error_closes_open_blocks(self):
        """An error mid-stream should close open content blocks before emitting error."""
        chunks = [
            b'data: {"id":"c1","choices":[{"delta":{"content":"Partial"},"finish_reason":null}]}\n\n',
            b'data: {"error":{"type":"overloaded_error","message":"Overloaded"}}\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]
        data_types = [d["type"] for d in data_events]
        # content_block_stop should be emitted to close the open text block
        self.assertIn("content_block_stop", data_types)
        # error event should follow
        self.assertIn("error", [v for k, v in events if k == "event"])


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

    def test_computer_use_tool_filtered(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {"name": "real_tool", "description": "ok", "input_schema": {}},
                {
                    "name": "computer",
                    "type": "computer_use_20250124",
                    "description": "control computer",
                    "input_schema": {},
                },
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        names = [t["function"]["name"] for t in result["tools"]]
        self.assertIn("real_tool", names)
        self.assertNotIn("computer", names)

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
        """thinking is always forwarded as reasoning_effort regardless of model name."""
        body = {
            "model": "x",
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
            "messages": [{"role": "user", "content": "solve it"}],
        }
        # Works for any model, not just known o-series names
        for model in ("o3", "gpt-5.4", "deepseek-r1", "some-future-model"):
            result = anthropic_to_openai_request(body, model)
            self.assertIn("reasoning_effort", result)
            self.assertEqual(result["reasoning_effort"], "medium")

    def test_adaptive_thinking_maps_to_high(self):
        """thinking.type == 'adaptive' without output_config → 'high' (Anthropic docs default)."""
        body = {
            "model": "x",
            "max_tokens": 1024,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        self.assertEqual(anthropic_to_openai_request(body, "gpt-5.4").get("reasoning_effort"), "high")
        self.assertEqual(anthropic_to_openai_request(body, "deepseek-r1").get("reasoning_effort"), "high")

    def test_disabled_thinking_omits_reasoning_effort(self):
        """thinking.type == 'disabled' must not enable reasoning on converted backends."""
        body = {
            "model": "x",
            "max_tokens": 1024,
            "thinking": {"type": "disabled", "display": "omitted"},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        self.assertNotIn("reasoning_effort", result)

    def test_adaptive_thinking_with_output_config_effort(self):
        """output_config.effort overrides the adaptive default; 'max' maps to 'xhigh'."""
        # For DeepSeek: max→xhigh, xhigh→xhigh, others pass through
        for effort, expected in (("max", "xhigh"), ("xhigh", "xhigh"), ("medium", "medium"), ("low", "low")):
            body = {
                "model": "x",
                "max_tokens": 1024,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
                "messages": [{"role": "user", "content": "Hello"}],
            }
            self.assertEqual(
                anthropic_to_openai_request(body, "deepseek-v3").get("reasoning_effort"), expected,
                f"deepseek effort={effort}",
            )
        # For OpenAI: max→xhigh (OpenAI also supports xhigh now), others pass through
        for effort, expected in (("max", "xhigh"), ("xhigh", "xhigh"), ("high", "high"), ("medium", "medium")):
            body = {
                "model": "x",
                "max_tokens": 1024,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
                "messages": [{"role": "user", "content": "Hello"}],
            }
            self.assertEqual(
                anthropic_to_openai_request(body, "gpt-5.4").get("reasoning_effort"), expected,
                f"openai effort={effort}",
            )

    def test_budget_thinking_with_output_config_effort_override(self):
        """output_config.effort overrides budget_tokens-derived effort."""
        body = {
            "model": "x",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 1000},  # would map to "low"
            "output_config": {"effort": "high"},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        self.assertEqual(anthropic_to_openai_request(body, "gpt-5.4").get("reasoning_effort"), "high")

    def test_max_budget_maps_to_xhigh(self):
        """budget_tokens > 32000 → 'xhigh' for both OpenAI and DeepSeek."""
        body = {
            "model": "x",
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 100000},
            "messages": [{"role": "user", "content": "solve it"}],
        }
        self.assertEqual(
            anthropic_to_openai_request(body, "deepseek-reasoner").get("reasoning_effort"), "xhigh"
        )
        self.assertEqual(
            anthropic_to_openai_request(body, "o3").get("reasoning_effort"), "xhigh"
        )

    def test_max_reasoning_effort_cap(self):
        """max_reasoning_effort caps the effort sent to the provider."""
        body = {
            "model": "x",
            "max_tokens": 1024,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        # Provider only supports up to "high" (e.g. Kimi on opencode)
        self.assertEqual(
            anthropic_to_openai_request(body, "kimi-k2", max_reasoning_effort="high").get("reasoning_effort"),
            "high",
        )
        # Provider supports "xhigh" (e.g. DeepSeek) — max maps to xhigh
        self.assertEqual(
            anthropic_to_openai_request(body, "deepseek-v3", max_reasoning_effort="xhigh").get("reasoning_effort"),
            "xhigh",
        )
        # "medium" effort is below any cap — passes through unchanged
        body2 = dict(body, output_config={"effort": "medium"})
        self.assertEqual(
            anthropic_to_openai_request(body2, "kimi-k2", max_reasoning_effort="high").get("reasoning_effort"),
            "medium",
        )

    def test_thinking_block_emitted_as_reasoning_content(self):
        """thinking blocks in assistant history are always preserved as reasoning_content
        so that providers requiring it (Moonshot, DeepSeek, etc.) receive them back."""
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
        self.assertEqual(msg.get("reasoning_content"), "I should call the tool.")
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

    def test_user_message_with_tool_results_and_text_ordering(self):
        """OpenAI requires tool messages to immediately follow the assistant that
        issued tool_calls.  When a user message contains both tool_result blocks
        and a text block (user typed while tools ran), the tool messages must
        come BEFORE the user text message in the OpenAI output.
        """
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_A", "name": "Bash", "input": {"command": "ls"}},
                        {"type": "tool_use", "id": "call_B", "name": "Read", "input": {"file_path": "/f"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call_A", "content": "file1"},
                        {"type": "tool_result", "tool_use_id": "call_B", "content": "contents"},
                        {"type": "text", "text": "ok, continue"},
                    ],
                },
            ],
        }
        result = anthropic_to_openai_request(body, "gpt-4o")
        msgs = result["messages"]
        # Find the indices of tool messages and user text message
        tool_indices = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
        user_text_indices = [i for i, m in enumerate(msgs) if m.get("role") == "user" and m.get("content") == "ok, continue"]
        self.assertEqual(len(tool_indices), 2, "Expected 2 tool messages")
        self.assertEqual(len(user_text_indices), 1, "Expected 1 user text message")
        # Tool messages must come before the user text
        self.assertLess(max(tool_indices), user_text_indices[0],
                        "tool messages must precede the user text message")
        # The two tool messages must be consecutive right after the assistant
        asst_indices = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
        asst_idx = asst_indices[-1]
        self.assertEqual(tool_indices[0], asst_idx + 1)
        self.assertEqual(tool_indices[1], asst_idx + 2)
        self.assertEqual(user_text_indices[0], asst_idx + 3)



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
        # thinking block must include signature field (protocol conformance)
        self.assertIn("signature", thinking, "non-streaming thinking block must include 'signature' field")
        self.assertEqual(thinking["signature"], "")

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

    def test_null_message_does_not_crash(self):
        """Upstream returning 'message': null (e.g. content-filter block) must not raise."""
        oai = {
            "id": "x",
            "choices": [{"message": None, "finish_reason": "content_filter"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0},
        }
        result = openai_to_anthropic_response(oai, "openai/gpt-4o")
        # Should produce a valid (empty-content) Anthropic response, not raise
        self.assertEqual(result["type"], "message")
        self.assertEqual(result["content"], [])

    def test_null_usage_does_not_crash(self):
        """Upstream returning 'usage': null must not raise."""
        oai = {
            "id": "x",
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": None,
        }
        result = openai_to_anthropic_response(oai, "openai/gpt-4o")
        self.assertEqual(result["type"], "message")
        self.assertEqual(result["usage"]["input_tokens"], 0)


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
    api_key: test-key
    endpoints:
      anthropic:
        models:
          - claude-sonnet-4-5
  openai:
    api_key: sk-test
    endpoints:
      openai_chat:
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
            self.assertIn("anthropic", cfg.providers["anthropic"].endpoints)
        finally:
            os.unlink(path)

    def test_env_expansion(self):
        import os, tempfile
        os.environ["TEST_KEY_CONV"] = "sk-expanded"
        yaml_content = """
providers:
  test:
    api_key: "${TEST_KEY_CONV}"
    endpoints:
      anthropic:
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
        from simple_api_router.config import EndpointConfig, ProviderConfig, RouterConfig
        from simple_api_router.proxy import resolve_provider
        cfg = RouterConfig(providers={
            "anthropic": ProviderConfig(
                api_key="k",
                endpoints={"anthropic": EndpointConfig(models=["claude-opus-4-5"])},
            ),
        })
        prov, ep, api_format, bmodel = resolve_provider("anthropic", "claude-opus-4-5", cfg)
        self.assertEqual(api_format, "anthropic")
        self.assertEqual(bmodel, "claude-opus-4-5")

    def test_resolve_provider_not_found(self):
        from fastapi import HTTPException
        from simple_api_router.config import RouterConfig
        from simple_api_router.proxy import resolve_provider
        cfg = RouterConfig(providers={})
        with self.assertRaises(HTTPException):
            resolve_provider("nonexistent", "model", cfg)

    def test_model_map_remapping(self):
        from simple_api_router.config import EndpointConfig, ProviderConfig, RouterConfig
        from simple_api_router.proxy import resolve_provider
        cfg = RouterConfig(providers={
            "myapi": ProviderConfig(
                api_key="k",
                endpoints={
                    "openai_chat": EndpointConfig(
                        models=["my-fast"],
                        model_map={"my-fast": "gpt-4o-mini"},
                    )
                },
            )
        })
        prov, ep, api_format, bmodel = resolve_provider("myapi", "my-fast", cfg)
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
        from simple_api_router.config import EndpointConfig, ProviderConfig, RouterConfig
        from simple_api_router.proxy import resolve_provider
        cfg = RouterConfig(providers={
            "anthropic": ProviderConfig(
                api_key="k",
                endpoints={"anthropic": EndpointConfig(models=["claude-opus-4-5"])},
            ),
        })
        # Suffix is already stripped by parse_model before reaching resolve_provider,
        # so this tests that the stripped name resolves correctly.
        prov, ep, api_format, bmodel = resolve_provider("anthropic", "claude-opus-4-5", cfg)
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

    def test_cache_control_not_forwarded_in_system_block(self):
        """cache_control is Anthropic-specific and is not forwarded to OpenAI format messages."""
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
        self.assertEqual(sys_msg.get("content"), "sys")
        self.assertNotIn("cache_control", sys_msg)

    def test_cache_control_stripped_from_user_block(self):
        """cache_control is Anthropic-specific and must be stripped from OpenAI user content."""
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
        # Single text block → plain string (not an array)
        self.assertIsInstance(user_content, str)
        self.assertEqual(user_content, "hello")

    def test_cache_control_stripped_from_tool(self):
        """cache_control on a tool definition must be stripped for OpenAI format."""
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
        self.assertNotIn("cache_control", tool)


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

    def test_whitespace_content_between_reasoning_does_not_split_thinking_block(self):
        """A whitespace-only content delta between two reasoning_content chunks must not
        close the thinking block, so all reasoning ends up in a single thinking block."""
        chunks = [
            b'data: {"id":"r1","choices":[{"delta":{"reasoning_content":"part1"}}]}\n\n',
            # provider sends a bare "\n" between reasoning segments
            b'data: {"id":"r1","choices":[{"delta":{"content":"\\n"}}]}\n\n',
            b'data: {"id":"r1","choices":[{"delta":{"reasoning_content":"part2"}}]}\n\n',
            b'data: {"id":"r1","choices":[{"delta":{"content":"answer"}}]}\n\n',
            b'data: {"id":"r1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":5}}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]
        thinking_starts = [
            d for d in data_events
            if d["type"] == "content_block_start"
            and d.get("content_block", {}).get("type") == "thinking"
        ]
        # All reasoning must be in ONE thinking block (no split from the "\n")
        self.assertEqual(len(thinking_starts), 1, "consecutive reasoning must produce exactly one thinking block")
        # Both reasoning parts must be present as thinking_deltas on the same index
        thinking_deltas = [
            d for d in data_events
            if d["type"] == "content_block_delta"
            and d.get("delta", {}).get("type") == "thinking_delta"
        ]
        combined = "".join(d["delta"]["thinking"] for d in thinking_deltas)
        self.assertEqual(combined, "part1part2")

    def test_thinking_block_stream_has_signature_and_signature_delta(self):
        """Per Anthropic spec: content_block_start for thinking must include signature:'',
        and a signature_delta must be emitted just before content_block_stop."""
        chunks = [
            b'data: {"id":"r1","choices":[{"delta":{"reasoning_content":"I think"}}]}\n\n',
            b'data: {"id":"r1","choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
            b'data: [DONE]\n\n',
        ]
        events = self._run(self._collect(chunks))
        data_events = [d for k, d in events if k == "data"]

        # 1. content_block_start for thinking must have signature: ""
        thinking_start = next(
            (d for d in data_events
             if d["type"] == "content_block_start"
             and d.get("content_block", {}).get("type") == "thinking"),
            None,
        )
        self.assertIsNotNone(thinking_start)
        self.assertIn("signature", thinking_start["content_block"],
                      "thinking content_block_start must include 'signature' field")
        self.assertEqual(thinking_start["content_block"]["signature"], "")

        # 2. A signature_delta must appear before content_block_stop for the thinking block
        thinking_idx = thinking_start["index"]
        thinking_stop = next(
            (d for d in data_events
             if d["type"] == "content_block_stop" and d.get("index") == thinking_idx),
            None,
        )
        self.assertIsNotNone(thinking_stop)
        stop_pos = data_events.index(thinking_stop)
        sig_delta = next(
            (d for d in data_events[:stop_pos]
             if d["type"] == "content_block_delta"
             and d.get("index") == thinking_idx
             and d.get("delta", {}).get("type") == "signature_delta"),
            None,
        )
        self.assertIsNotNone(sig_delta,
            "signature_delta must be emitted before content_block_stop for thinking blocks")

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

    def test_thinking_emitted_as_reasoning_content_by_default(self):
        """With use_reasoning_content=False (default), thinking blocks are still preserved
        as reasoning_content so providers requiring it receive the thinking content."""
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
        self.assertEqual(asst.get("reasoning_content"), "secret")
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

    # ------------------------------------------------------------------
    # Interrupted-response history (openai_chat path)
    # ------------------------------------------------------------------

    def _make_interrupted_follow_up(self, partial_assistant_content):
        """Return a minimal follow-up request with a partial assistant turn."""
        return {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Do something"},
                {"role": "assistant", "content": partial_assistant_content},
                {"role": "user", "content": "Never mind, do this instead"},
            ],
        }

    def test_interrupted_empty_content_not_null(self):
        """Interrupted before any block → content must be '' not null."""
        body = self._make_interrupted_follow_up([])
        result = anthropic_to_openai_request(body, "gpt-4o")
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertEqual(asst["content"], "")
        self.assertNotIn("tool_calls", asst)

    def test_interrupted_thinking_only_not_null(self):
        """Interrupted during thinking (no text yet) → content must be '' not null."""
        body = self._make_interrupted_follow_up([
            {"type": "thinking", "thinking": "partial reasoning..."},
        ])
        result = anthropic_to_openai_request(body, "gpt-4o")
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertEqual(asst["content"], "")
        self.assertNotIn("tool_calls", asst)

    def test_interrupted_thinking_only_deepseek_not_null(self):
        """Same as above but DeepSeek mode: thinking → reasoning_content, content → ''."""
        body = self._make_interrupted_follow_up([
            {"type": "thinking", "thinking": "partial reasoning..."},
        ])
        result = anthropic_to_openai_request(body, "deepseek-r1", use_reasoning_content=True)
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertEqual(asst["content"], "")
        self.assertEqual(asst["reasoning_content"], "partial reasoning...")

    def test_interrupted_redacted_thinking_not_null(self):
        """Only redacted_thinking block → content must be '' not null."""
        body = self._make_interrupted_follow_up([
            {"type": "redacted_thinking", "data": "ENCRYPTED"},
        ])
        result = anthropic_to_openai_request(body, "gpt-4o")
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertEqual(asst["content"], "")

    def test_interrupted_partial_text_preserved(self):
        """Interrupted during text streaming → partial text is preserved."""
        body = self._make_interrupted_follow_up([
            {"type": "text", "text": "Partial answer..."},
        ])
        result = anthropic_to_openai_request(body, "gpt-4o")
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertEqual(asst["content"], "Partial answer...")

    def test_interrupted_tool_only_null_is_correct(self):
        """Tool-only (interrupted during tool streaming) → content: null is the OpenAI standard."""
        body = self._make_interrupted_follow_up([
            {"type": "tool_use", "id": "call_abc", "name": "do_thing", "input": {}},
        ])
        result = anthropic_to_openai_request(body, "gpt-4o")
        asst = next(m for m in result["messages"] if m["role"] == "assistant")
        self.assertIsNone(asst["content"])
        self.assertEqual(asst["tool_calls"][0]["function"]["name"], "do_thing")

    def test_new_user_message_survives_interrupted_history(self):
        """The follow-up user message is present after an interrupted empty assistant turn."""
        body = self._make_interrupted_follow_up([])
        result = anthropic_to_openai_request(body, "gpt-4o")
        roles = [m["role"] for m in result["messages"]]
        # Conversation structure: system?, user, assistant, user
        self.assertEqual(roles[-1], "user")
        last_user = result["messages"][-1]
        self.assertEqual(last_user["content"], "Never mind, do this instead")


# ---------------------------------------------------------------------------
# Responses API — request conversion
# ---------------------------------------------------------------------------

from simple_api_router.converter_responses import anthropic_to_responses_request  # noqa: E402


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
        """adaptive without output_config → effort: 'high' (Anthropic docs default)."""
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Think"}],
            "thinking": {"type": "adaptive"},
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        self.assertEqual(result["reasoning"]["effort"], "high")

    def test_adaptive_thinking_with_output_config_effort_responses(self):
        """output_config.effort overrides adaptive default for Responses API."""
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Think"}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        self.assertEqual(result["reasoning"]["effort"], "medium")

    def test_disabled_thinking_omits_reasoning_responses(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Think"}],
            "thinking": {"type": "disabled", "display": "omitted"},
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        self.assertNotIn("reasoning", result)

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

    def test_tool_choice_none_stays_none_responses(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"name": "fn", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
            "tool_choice": {"type": "none"},
        }
        result = anthropic_to_responses_request(body, "gpt-5")
        self.assertEqual(result["tool_choice"], "none")

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

    # ------------------------------------------------------------------
    # Interrupted-response history (openai_responses path)
    # ------------------------------------------------------------------

    def _interrupted_responses_body(self, partial_assistant_content):
        return {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Do something"},
                {"role": "assistant", "content": partial_assistant_content},
                {"role": "user", "content": "Never mind"},
            ],
        }

    def test_responses_interrupted_empty_content_placeholder(self):
        """Interrupted before first block (content=[]) → empty output_text placeholder."""
        body = self._interrupted_responses_body([])
        result = anthropic_to_responses_request(body, "gpt-5")
        items = result["input"]
        asst = next(it for it in items if isinstance(it, dict) and it.get("role") == "assistant")
        self.assertEqual(asst["content"][0]["type"], "output_text")
        self.assertEqual(asst["content"][0]["text"], "")

    def test_responses_interrupted_thinking_only_placeholder(self):
        """Interrupted during thinking (no text) → empty output_text placeholder."""
        body = self._interrupted_responses_body([
            {"type": "thinking", "thinking": "partial reasoning..."},
        ])
        result = anthropic_to_responses_request(body, "gpt-5")
        items = result["input"]
        asst = next(it for it in items if isinstance(it, dict) and it.get("role") == "assistant")
        self.assertEqual(asst["content"][0]["text"], "")

    def test_responses_interrupted_redacted_thinking_placeholder(self):
        """Only redacted_thinking → empty output_text placeholder."""
        body = self._interrupted_responses_body([
            {"type": "redacted_thinking", "data": "ENCRYPTED"},
        ])
        result = anthropic_to_responses_request(body, "gpt-5")
        items = result["input"]
        asst = next(it for it in items if isinstance(it, dict) and it.get("role") == "assistant")
        self.assertEqual(asst["content"][0]["text"], "")

    def test_responses_interrupted_tool_use_no_extra_message(self):
        """Tool-only interrupted turn → function_call added, no spurious empty text message."""
        body = self._interrupted_responses_body([
            {"type": "tool_use", "id": "call_abc", "name": "do_thing", "input": {}},
        ])
        result = anthropic_to_responses_request(body, "gpt-5")
        items = result["input"]
        func_calls = [it for it in items if isinstance(it, dict) and it.get("type") == "function_call"]
        self.assertEqual(len(func_calls), 1)
        # Must NOT have a spurious empty assistant message
        spurious = [
            it for it in items
            if isinstance(it, dict)
            and it.get("role") == "assistant"
            and it.get("content", [{}])[0].get("text") == ""
        ]
        self.assertEqual(len(spurious), 0)

    def test_responses_interrupted_text_preserved(self):
        """Partial text survives."""
        body = self._interrupted_responses_body([
            {"type": "text", "text": "Partial answer..."},
        ])
        result = anthropic_to_responses_request(body, "gpt-5")
        asst = next(it for it in result["input"] if isinstance(it, dict) and it.get("role") == "assistant")
        self.assertEqual(asst["content"][0]["text"], "Partial answer...")

    def test_responses_follow_up_user_message_present(self):
        """The new user message is always the last item in input, even after interrupted turn."""
        body = self._interrupted_responses_body([])
        result = anthropic_to_responses_request(body, "gpt-5")
        last = result["input"][-1]
        self.assertEqual(last.get("role"), "user")
        self.assertEqual(last["content"][0]["text"], "Never mind")


# ---------------------------------------------------------------------------
# Responses API — non-streaming response conversion
# ---------------------------------------------------------------------------

from simple_api_router.converter_responses import responses_to_anthropic_response  # noqa: E402


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

from simple_api_router.converter_responses import stream_responses_to_anthropic  # noqa: E402


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

    def test_stream_terminates_without_response_completed(self):
        """If response.completed never arrives, the generator must still emit
        message_delta + message_stop so the client doesn't hang forever."""
        def _run(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        chunks = [
            _make_sse("response.created", {
                "type": "response.created",
                "response": {"id": "resp_trunc", "usage": {}},
            }),
            _make_sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "output_index": 0, "content_index": 0,
                "delta": "partial answer",
            }),
            # Stream ends here — no response.completed
            b"data: [DONE]\n\n",
        ]

        async def _collect():
            async def _gen():
                for c in chunks:
                    yield c
            events = []
            async for raw in stream_responses_to_anthropic(_gen(), "gpt-5", "msg_trunc"):
                for line in raw.decode().split("\n"):
                    if line.startswith("event: "):
                        events.append(line[7:])
            return events

        event_types = _run(_collect())
        self.assertIn("message_delta", event_types, "message_delta must be emitted even if response.completed is missing")
        self.assertIn("message_stop", event_types, "message_stop must be emitted even if response.completed is missing")


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
        req = self._make_request({"anthropic-version": "2023-06-01"})
        hdrs = _build_anthropic_headers(req, "sk-real")
        self.assertEqual(hdrs["x-api-key"], "sk-real")
        self.assertEqual(hdrs["Authorization"], "Bearer sk-real")

    def test_fake_key_none_sends_no_auth(self):
        from simple_api_router.proxy import _build_anthropic_headers
        req = self._make_request({})
        hdrs = _build_anthropic_headers(req, "none")
        self.assertNotIn("x-api-key", hdrs)
        self.assertNotIn("Authorization", hdrs)

    def test_empty_key_sends_no_auth(self):
        from simple_api_router.proxy import _build_anthropic_headers
        req = self._make_request({})
        hdrs = _build_anthropic_headers(req, "")
        self.assertNotIn("x-api-key", hdrs)
        self.assertNotIn("Authorization", hdrs)

    def test_anthropic_version_forwarded(self):
        from simple_api_router.proxy import _build_anthropic_headers
        req = self._make_request({"anthropic-version": "2024-01-01"})
        hdrs = _build_anthropic_headers(req, "sk-real")
        self.assertEqual(hdrs["anthropic-version"], "2024-01-01")

    def test_default_anthropic_version_injected_when_missing(self):
        from simple_api_router.proxy import _build_anthropic_headers
        req = self._make_request({})
        hdrs = _build_anthropic_headers(req, "sk-real")
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
    api_key: "${LOAD_CONFIG_MISSING_VAR}"
    endpoints:
      anthropic:
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
    def test_valid_endpoint_formats_accepted(self):
        from simple_api_router.config import EndpointConfig, ProviderConfig
        cfg = ProviderConfig(api_key="k", endpoints={
            "anthropic": EndpointConfig(),
            "openai_chat": EndpointConfig(),
        })
        self.assertIn("anthropic", cfg.endpoints)
        self.assertIn("openai_chat", cfg.endpoints)

    def test_all_valid_formats_accepted(self):
        from simple_api_router.config import EndpointConfig, ProviderConfig, VALID_FORMATS
        for fmt in VALID_FORMATS:
            cfg = ProviderConfig(api_key="k", endpoints={fmt: EndpointConfig()})
            self.assertIn(fmt, cfg.endpoints)

    def test_invalid_endpoint_format_raises(self):
        from simple_api_router.config import EndpointConfig, ProviderConfig
        with self.assertRaises(Exception):
            ProviderConfig(api_key="k", endpoints={"invalid_format": EndpointConfig()})

    def test_duplicate_model_across_endpoints_raises(self):
        from simple_api_router.config import EndpointConfig, ProviderConfig
        with self.assertRaises(Exception):
            ProviderConfig(api_key="k", endpoints={
                "anthropic": EndpointConfig(models=["shared-model"]),
                "openai_chat": EndpointConfig(models=["shared-model"]),
            })

    def test_deepseek_reasoning_default_none(self):
        from simple_api_router.config import EndpointConfig
        ep = EndpointConfig()
        self.assertIsNone(ep.deepseek_reasoning)

    def test_deepseek_reasoning_can_be_set(self):
        from simple_api_router.config import EndpointConfig
        ep = EndpointConfig(deepseek_reasoning=True)
        self.assertTrue(ep.deepseek_reasoning)

    def test_find_model_exact_match(self):
        from simple_api_router.config import EndpointConfig, ProviderConfig
        prov = ProviderConfig(api_key="k", endpoints={
            "anthropic": EndpointConfig(models=["claude-opus-4-5"]),
            "openai_chat": EndpointConfig(models=["gpt-4o"]),
        })
        result = prov.find_model("gpt-4o")
        self.assertIsNotNone(result)
        fmt, ep = result
        self.assertEqual(fmt, "openai_chat")

    def test_find_model_wildcard_fallback(self):
        from simple_api_router.config import EndpointConfig, ProviderConfig
        prov = ProviderConfig(api_key="k", endpoints={
            "openai_chat": EndpointConfig(models=[]),  # wildcard
        })
        result = prov.find_model("any-model")
        self.assertIsNotNone(result)
        fmt, ep = result
        self.assertEqual(fmt, "openai_chat")

    def test_find_model_not_found_returns_none(self):
        from simple_api_router.config import EndpointConfig, ProviderConfig
        prov = ProviderConfig(api_key="k", endpoints={
            "anthropic": EndpointConfig(models=["claude-opus-4-5"]),
        })
        self.assertIsNone(prov.find_model("unknown-model"))

    def test_endpoint_resolve_base_url_default(self):
        from simple_api_router.config import EndpointConfig
        ep = EndpointConfig()
        self.assertEqual(ep.resolve_base_url("anthropic"), "https://api.anthropic.com")
        self.assertEqual(ep.resolve_base_url("openai_chat"), "https://api.openai.com")
        self.assertEqual(ep.resolve_base_url("openai_responses"), "https://api.openai.com")
        self.assertEqual(ep.resolve_base_url("google"), "https://generativelanguage.googleapis.com")

    def test_endpoint_resolve_base_url_custom(self):
        from simple_api_router.config import EndpointConfig
        ep = EndpointConfig(base_url="https://myserver.com/")
        self.assertEqual(ep.resolve_base_url("anthropic"), "https://myserver.com")

    def test_endpoint_resolve_base_url_strips_trailing_v1(self):
        from simple_api_router.config import EndpointConfig
        ep = EndpointConfig(base_url="https://api.anthropic.com/v1")
        self.assertEqual(ep.resolve_base_url("anthropic"), "https://api.anthropic.com")

    def test_endpoint_resolve_base_url_strips_trailing_v1_with_slash(self):
        from simple_api_router.config import EndpointConfig
        ep = EndpointConfig(base_url="https://api.openai.com/v1/")
        self.assertEqual(ep.resolve_base_url("openai_chat"), "https://api.openai.com")

    def test_endpoint_inherits_provider_base_url(self):
        from simple_api_router.config import EndpointConfig
        ep = EndpointConfig()  # no base_url
        self.assertEqual(ep.resolve_base_url("anthropic", "https://myserver.com"), "https://myserver.com")

    def test_endpoint_base_url_overrides_provider(self):
        from simple_api_router.config import EndpointConfig
        ep = EndpointConfig(base_url="https://specific.com")
        self.assertEqual(ep.resolve_base_url("anthropic", "https://ignored.com"), "https://specific.com")

    def test_provider_base_url_field(self):
        from simple_api_router.config import ProviderConfig, EndpointConfig
        prov = ProviderConfig(
            api_key="key",
            base_url="https://shared.com",
            endpoints={
                "anthropic": EndpointConfig(models=["claude-3"]),
                "openai_chat": EndpointConfig(base_url="https://other.com", models=["gpt-4o"]),
            },
        )
        fmt, ep = prov.find_model("claude-3")
        self.assertEqual(ep.resolve_base_url("anthropic", prov.base_url), "https://shared.com")
        fmt2, ep2 = prov.find_model("gpt-4o")
        self.assertEqual(ep2.resolve_base_url("openai_chat", prov.base_url), "https://other.com")


# ---------------------------------------------------------------------------
# Google Gemini converter
# ---------------------------------------------------------------------------

class TestGoogleConverter(unittest.TestCase):
    """Tests for converter_google.py — Anthropic ↔ Gemini format conversion."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    async def _collect_stream(self, sse_chunks):
        """Collect all SSE events from stream_google_to_anthropic over raw bytes."""
        from simple_api_router.converter_google import stream_google_to_anthropic

        async def _source():
            for c in sse_chunks:
                yield c

        events = []
        async for raw in stream_google_to_anthropic(_source(), "google/gemini-2.0-flash"):
            for line in raw.decode().split("\n"):
                if line.startswith("event: "):
                    events.append(("event", line[7:]))
                elif line.startswith("data: "):
                    events.append(("data", json.loads(line[6:])))
        return events

    # ------------------------------------------------------------------
    # anthropic_to_google_request
    # ------------------------------------------------------------------

    def test_simple_text_message(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "google/gemini-2.0-flash",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        self.assertEqual(result["contents"][0]["role"], "user")
        self.assertEqual(result["contents"][0]["parts"][0]["text"], "Hello")
        self.assertEqual(result["generationConfig"]["maxOutputTokens"], 512)

    def test_assistant_role_becomes_model(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        self.assertEqual(result["contents"][1]["role"], "model")

    def test_system_string_becomes_system_instruction(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "system": "Be concise.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        self.assertEqual(result["systemInstruction"]["parts"][0]["text"], "Be concise.")

    def test_system_list_merged(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "Part A."},
                {"type": "text", "text": "Part B."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        text = result["systemInstruction"]["parts"][0]["text"]
        self.assertIn("Part A.", text)
        self.assertIn("Part B.", text)

    def test_generation_config_fields(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        gc = result["generationConfig"]
        self.assertEqual(gc["maxOutputTokens"], 256)
        self.assertEqual(gc["temperature"], 0.7)
        self.assertEqual(gc["topP"], 0.9)
        self.assertEqual(gc["stopSequences"], ["END"])

    def test_tools_converted_to_function_declarations(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Calculate"}],
            "tools": [
                {
                    "name": "calculator",
                    "description": "Does math",
                    "input_schema": {
                        "type": "object",
                        "properties": {"expr": {"type": "string"}},
                        "required": ["expr"],
                    },
                }
            ],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        decls = result["tools"][0]["functionDeclarations"]
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0]["name"], "calculator")
        self.assertEqual(decls[0]["description"], "Does math")
        self.assertIn("properties", decls[0]["parameters"])

    def test_tool_choice_auto(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "t", "input_schema": {}}],
            "tool_choice": {"type": "auto"},
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        self.assertEqual(result["toolConfig"]["functionCallingConfig"]["mode"], "AUTO")

    def test_tool_choice_none(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "t", "input_schema": {}}],
            "tool_choice": {"type": "none"},
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        self.assertEqual(result["toolConfig"]["functionCallingConfig"]["mode"], "NONE")

    def test_tool_choice_specific_tool(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "calculator", "input_schema": {}}],
            "tool_choice": {"type": "tool", "name": "calculator"},
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        fc = result["toolConfig"]["functionCallingConfig"]
        self.assertEqual(fc["mode"], "ANY")
        self.assertEqual(fc["allowedFunctionNames"], ["calculator"])

    def test_tool_use_in_assistant_message(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Calculate 2+2"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_01", "name": "calculator", "input": {"expr": "2+2"}}
                    ],
                },
            ],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        assistant_parts = result["contents"][1]["parts"]
        self.assertEqual(assistant_parts[0]["functionCall"]["name"], "calculator")
        self.assertEqual(assistant_parts[0]["functionCall"]["args"], {"expr": "2+2"})

    def test_tool_result_resolves_name_from_history(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Calculate 2+2"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_01", "name": "calculator", "input": {"expr": "2+2"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_01", "content": "4"}
                    ],
                },
            ],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        user_parts = result["contents"][2]["parts"]
        fr = user_parts[0]["functionResponse"]
        self.assertEqual(fr["name"], "calculator")
        self.assertEqual(fr["response"]["output"], "4")

    def test_tool_result_fallback_to_id_when_name_not_found(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "unknown_id", "content": "result"}
                    ],
                }
            ],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        fr = result["contents"][0]["parts"][0]["functionResponse"]
        self.assertEqual(fr["name"], "unknown_id")

    def test_image_base64_converted(self):
        from simple_api_router.converter_google import anthropic_to_google_request
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
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        part = result["contents"][0]["parts"][0]
        self.assertEqual(part["inlineData"]["mimeType"], "image/png")
        self.assertEqual(part["inlineData"]["data"], "abc123")

    def test_thinking_blocks_skipped(self):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "internal thought"},
                        {"type": "text", "text": "Answer"},
                    ],
                }
            ],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        parts = result["contents"][0]["parts"]
        # thinking block is forwarded as thought: true so Gemini sees prior reasoning
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["text"], "internal thought")
        self.assertTrue(parts[0].get("thought"))
        self.assertEqual(parts[1]["text"], "Answer")

    def test_thinking_config_gemini25(self):
        """budget_tokens → thinkingBudget for Gemini 2.5."""
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x", "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 5000},
        }
        result = anthropic_to_google_request(body, "gemini-2.5-flash")
        tc = result["generationConfig"]["thinkingConfig"]
        self.assertEqual(tc["thinkingBudget"], 5000)

    def test_thinking_config_gemini3(self):
        """budget_tokens → thinkingLevel for Gemini 3."""
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x", "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 5000},
        }
        result = anthropic_to_google_request(body, "gemini-3-flash")
        tc = result["generationConfig"]["thinkingConfig"]
        self.assertEqual(tc["thinkingLevel"], "medium")

    def test_thinking_config_adaptive_gemini3(self):
        """adaptive thinking without output_config → thinkingLevel: high for Gemini 3."""
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x", "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "adaptive"},
        }
        result = anthropic_to_google_request(body, "gemini-3-flash")
        tc = result["generationConfig"]["thinkingConfig"]
        self.assertEqual(tc["thinkingLevel"], "high")

    def test_thinking_config_adaptive_output_effort_gemini3(self):
        """output_config.effort overrides adaptive default for Gemini 3."""
        from simple_api_router.converter_google import anthropic_to_google_request
        for effort, expected in [("low", "low"), ("medium", "medium"), ("max", "high")]:
            body = {
                "model": "x", "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
            }
            result = anthropic_to_google_request(body, "gemini-3-flash")
            tc = result["generationConfig"]["thinkingConfig"]
            self.assertEqual(tc["thinkingLevel"], expected, f"effort={effort}")

    def test_thought_part_becomes_thinking_block(self):
        """Gemini response thought: true part → Anthropic thinking block."""
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {
            "candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [
                    {"text": "I need to think carefully", "thought": True},
                    {"text": "The answer is 42"},
                ]},
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }
        result = google_to_anthropic_response(data, "gemini-2.5-flash")
        types = [b["type"] for b in result["content"]]
        self.assertEqual(types, ["thinking", "text"])
        self.assertEqual(result["content"][0]["thinking"], "I need to think carefully")
        self.assertEqual(result["content"][1]["text"], "The answer is 42")

    # ------------------------------------------------------------------
    # google_to_anthropic_response
    # ------------------------------------------------------------------

    def test_response_simple_text(self):
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello there"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
        }
        result = google_to_anthropic_response(data, "google/gemini-2.0-flash")
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(result["content"][0]["text"], "Hello there")
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(result["usage"]["input_tokens"], 5)
        self.assertEqual(result["usage"]["output_tokens"], 3)

    def test_response_max_tokens_finish_reason(self):
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Truncated"}]},
                    "finishReason": "MAX_TOKENS",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        }
        result = google_to_anthropic_response(data, "google/gemini-2.0-flash")
        self.assertEqual(result["stop_reason"], "max_tokens")

    def test_response_stop_sequence(self):
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Answer"}]},
                    "finishReason": "STOP_SEQUENCE",
                }
            ],
        }
        result = google_to_anthropic_response(data, "google/gemini-2.0-flash")
        self.assertEqual(result["stop_reason"], "stop_sequence")

    def test_response_function_call(self):
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "calculator", "args": {"expr": "1+1"}}}
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
        }
        result = google_to_anthropic_response(data, "google/gemini-2.0-flash")
        self.assertEqual(result["stop_reason"], "tool_use")
        tool = result["content"][0]
        self.assertEqual(tool["type"], "tool_use")
        self.assertEqual(tool["name"], "calculator")
        self.assertEqual(tool["input"], {"expr": "1+1"})
        self.assertTrue(tool["id"].startswith("toolu_"))

    def test_response_safety_block(self):
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {
            "promptFeedback": {"blockReason": "SAFETY"},
            "candidates": [],
        }
        result = google_to_anthropic_response(data, "google/gemini-2.0-flash")
        self.assertEqual(result["stop_reason"], "refusal")
        self.assertIn("SAFETY", result["content"][0]["text"])

    def test_response_empty_candidates(self):
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {"candidates": []}
        result = google_to_anthropic_response(data, "google/gemini-2.0-flash")
        self.assertEqual(result["content"], [])
        self.assertEqual(result["stop_reason"], "end_turn")

    # ------------------------------------------------------------------
    # stream_google_to_anthropic
    # ------------------------------------------------------------------

    def _make_sse_chunk(self, parts, finish_reason=None, usage=None):
        candidate = {"content": {"parts": parts}}
        if finish_reason:
            candidate["finishReason"] = finish_reason
        data = {"candidates": [candidate]}
        if usage:
            data["usageMetadata"] = usage
        return f"data: {json.dumps(data)}\n\n".encode()

    def test_stream_text(self):
        chunks = [
            self._make_sse_chunk([{"text": "Hello "}]),
            self._make_sse_chunk([{"text": "world"}], finish_reason="STOP",
                                  usage={"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}),
        ]
        events = self._run(self._collect_stream(chunks))
        types = [d["type"] for k, d in events if k == "data"]
        self.assertIn("message_start", types)
        self.assertIn("content_block_start", types)
        self.assertIn("content_block_delta", types)
        self.assertIn("content_block_stop", types)
        self.assertIn("message_delta", types)
        self.assertIn("message_stop", types)

        # Collect text from all text_delta events
        text = "".join(
            d["delta"]["text"]
            for _, d in events
            if isinstance(d, dict) and d.get("type") == "content_block_delta"
            and d.get("delta", {}).get("type") == "text_delta"
        )
        self.assertEqual(text, "Hello world")

    def test_stream_stop_reason_end_turn(self):
        chunks = [self._make_sse_chunk([{"text": "Hi"}], finish_reason="STOP")]
        events = self._run(self._collect_stream(chunks))
        delta = next(d for _, d in events if isinstance(d, dict) and d.get("type") == "message_delta")
        self.assertEqual(delta["delta"]["stop_reason"], "end_turn")

    def test_stream_stop_reason_max_tokens(self):
        chunks = [self._make_sse_chunk([{"text": "..."}], finish_reason="MAX_TOKENS")]
        events = self._run(self._collect_stream(chunks))
        delta = next(d for _, d in events if isinstance(d, dict) and d.get("type") == "message_delta")
        self.assertEqual(delta["delta"]["stop_reason"], "max_tokens")

    def test_stream_tool_use(self):
        chunks = [
            self._make_sse_chunk(
                [{"functionCall": {"name": "calculator", "args": {"expr": "2+2"}}}],
                finish_reason="STOP",
            )
        ]
        events = self._run(self._collect_stream(chunks))
        types = [d["type"] for k, d in events if k == "data"]
        self.assertIn("content_block_start", types)
        self.assertIn("content_block_delta", types)

        # Find the tool_use block_start
        tool_start = next(
            d for _, d in events
            if isinstance(d, dict) and d.get("type") == "content_block_start"
            and d.get("content_block", {}).get("type") == "tool_use"
        )
        self.assertEqual(tool_start["content_block"]["name"], "calculator")

        # stop_reason should be tool_use
        delta = next(d for _, d in events if isinstance(d, dict) and d.get("type") == "message_delta")
        self.assertEqual(delta["delta"]["stop_reason"], "tool_use")

    def test_stream_usage_in_message_delta(self):
        chunks = [
            self._make_sse_chunk(
                [{"text": "Hi"}],
                finish_reason="STOP",
                usage={"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            )
        ]
        events = self._run(self._collect_stream(chunks))
        delta = next(d for _, d in events if isinstance(d, dict) and d.get("type") == "message_delta")
        self.assertEqual(delta["usage"]["output_tokens"], 5)

    def test_stream_message_start_present(self):
        chunks = [self._make_sse_chunk([{"text": "Hi"}], finish_reason="STOP")]
        events = self._run(self._collect_stream(chunks))
        start = next(d for _, d in events if isinstance(d, dict) and d.get("type") == "message_start")
        self.assertEqual(start["message"]["role"], "assistant")
        self.assertEqual(start["message"]["model"], "google/gemini-2.0-flash")

    # ------------------------------------------------------------------
    # cc-switch ported: request (TEST 5, 16, 17)
    # ------------------------------------------------------------------

    def test_parameters_json_schema_for_additional_properties(self):
        """TEST 5 (cc-switch): additionalProperties triggers parametersJsonSchema; $schema stripped."""
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "name": "my_tool",
                    "description": "Does stuff",
                    "input_schema": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "additionalProperties": False,
                    },
                }
            ],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        decl = result["tools"][0]["functionDeclarations"][0]
        # Must use parametersJsonSchema, not parameters
        self.assertIn("parametersJsonSchema", decl)
        self.assertNotIn("parameters", decl)
        # $schema must be stripped
        self.assertNotIn("$schema", decl["parametersJsonSchema"])
        # additionalProperties is preserved
        self.assertFalse(decl["parametersJsonSchema"]["additionalProperties"])

    def test_parameters_json_schema_not_used_for_simple_schema(self):
        """TEST 5 variant: plain schema stays as parameters, not parametersJsonSchema."""
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "name": "plain",
                    "input_schema": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        decl = result["tools"][0]["functionDeclarations"][0]
        self.assertIn("parameters", decl)
        self.assertNotIn("parametersJsonSchema", decl)

    def test_synthesized_id_stripped_from_function_call_in_request(self):
        """TEST 16 (cc-switch): tool_use with synthesised toolu_* id → no id in functionCall."""
        from simple_api_router.converter_google import anthropic_to_google_request
        synth_id = "toolu_" + "a" * 24
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": synth_id,
                            "name": "Bash",
                            "input": {"command": "ls"},
                        }
                    ],
                }
            ],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        fc = result["contents"][0]["parts"][0]["functionCall"]
        self.assertNotIn("id", fc)

    def test_synthesized_id_stripped_from_function_response_in_request(self):
        """TEST 16 (cc-switch): tool_result with synthesised toolu_* id → no id in functionResponse."""
        from simple_api_router.converter_google import anthropic_to_google_request
        synth_id = "toolu_" + "b" * 24
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                # Provide the tool_use so the name is in the map
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": synth_id, "name": "Bash", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": synth_id, "content": "ok"}
                    ],
                },
            ],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        fr = result["contents"][1]["parts"][0]["functionResponse"]
        self.assertNotIn("id", fr)

    def test_genuine_id_preserved_in_function_call_request(self):
        """TEST 17 (cc-switch): tool_use with real Gemini id → id preserved in functionCall."""
        from simple_api_router.converter_google import anthropic_to_google_request
        real_id = "call_real_1"
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": real_id,
                            "name": "Bash",
                            "input": {"command": "ls"},
                        }
                    ],
                }
            ],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        fc = result["contents"][0]["parts"][0]["functionCall"]
        self.assertEqual(fc["id"], real_id)

    def test_genuine_id_preserved_in_function_response_request(self):
        """TEST 17 (cc-switch): tool_result with real Gemini id → id preserved in functionResponse."""
        from simple_api_router.converter_google import anthropic_to_google_request
        real_id = "call_real_1"
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": real_id, "name": "Bash", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": real_id, "content": "done"}
                    ],
                },
            ],
        }
        result = anthropic_to_google_request(body, "gemini-2.0-flash")
        fr = result["contents"][1]["parts"][0]["functionResponse"]
        self.assertEqual(fr["id"], real_id)

    # ------------------------------------------------------------------
    # cc-switch ported: response (TEST 6, 7, 10, responseId)
    # ------------------------------------------------------------------

    def test_response_uses_response_id_from_gemini(self):
        """responseId in Gemini response → used as Anthropic message id."""
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {
            "responseId": "resp_1",
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "Hi"}]},
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "totalTokenCount": 12},
        }
        result = google_to_anthropic_response(data, "gemini-2.0-flash")
        self.assertEqual(result["id"], "resp_1")

    def test_response_cache_read_tokens(self):
        """TEST 6 (cc-switch): cachedContentTokenCount → cache_read_input_tokens in usage."""
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {
            "responseId": "resp_1",
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "Hello"}]},
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12,
                "totalTokenCount": 20,
                "cachedContentTokenCount": 3,
            },
        }
        result = google_to_anthropic_response(data, "gemini-2.0-flash")
        # prompt=12 includes 3 cached; input_tokens = non-cached only = 9
        self.assertEqual(result["usage"]["input_tokens"], 9)
        self.assertEqual(result["usage"]["output_tokens"], 8)   # 20 - 12
        self.assertEqual(result["usage"]["cache_read_input_tokens"], 3)

    def test_response_no_cache_tokens_omits_field(self):
        """cache_read_input_tokens should not appear when cachedContentTokenCount is absent."""
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {
            "candidates": [
                {"finishReason": "STOP", "content": {"parts": [{"text": "Hi"}]}}
            ],
            "usageMetadata": {"promptTokenCount": 5, "totalTokenCount": 8},
        }
        result = google_to_anthropic_response(data, "gemini-2.0-flash")
        self.assertNotIn("cache_read_input_tokens", result["usage"])

    def test_response_function_call_id_preserved(self):
        """TEST 7 (cc-switch): Gemini functionCall.id is used verbatim as tool_use.id."""
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {
            "responseId": "resp_2",
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "call_1",
                                    "name": "get_weather",
                                    "args": {"city": "Tokyo"},
                                }
                            }
                        ]
                    },
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "totalTokenCount": 8},
        }
        result = google_to_anthropic_response(data, "gemini-2.0-flash")
        self.assertEqual(len(result["content"]), 1)
        tool_block = result["content"][0]
        self.assertEqual(tool_block["type"], "tool_use")
        self.assertEqual(tool_block["id"], "call_1")
        self.assertEqual(tool_block["name"], "get_weather")

    def test_response_function_call_no_id_synthesizes(self):
        """TEST 15 (cc-switch): functionCall without id gets a synthesised toolu_* id."""
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "search", "args": {"q": "hi"}}}
                        ]
                    },
                }
            ],
            "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 5},
        }
        result = google_to_anthropic_response(data, "gemini-2.0-flash")
        tool_id = result["content"][0]["id"]
        self.assertTrue(tool_id.startswith("toolu_"), tool_id)

    def test_response_safety_block_is_refusal(self):
        """TEST 10 (cc-switch): promptFeedback.blockReason → stop_reason: 'refusal'."""
        from simple_api_router.converter_google import google_to_anthropic_response
        data = {
            "promptFeedback": {"blockReason": "SAFETY"},
            "usageMetadata": {},
        }
        result = google_to_anthropic_response(data, "gemini-2.0-flash")
        self.assertEqual(result["stop_reason"], "refusal")
        self.assertEqual(result["content"][0]["text"], "Content blocked: SAFETY")

    # ------------------------------------------------------------------
    # cc-switch ported: streaming (TEST S1, S3)
    # ------------------------------------------------------------------

    def test_stream_cumulative_text_delta(self):
        """TEST S1 (cc-switch): Gemini sends cumulative text; converter must diff to get delta."""
        # Chunk 1 has "Hel", chunk 2 has "Hello" (cumulative snapshot)
        # Expected SSE deltas: "Hel" then "lo"
        chunks = [
            (
                b'data: {"responseId":"resp_1","candidates":[{"content":{"parts":[{"text":"Hel"}]}}],'
                b'"usageMetadata":{"promptTokenCount":10,"totalTokenCount":13}}\n\n'
            ),
            (
                b'data: {"responseId":"resp_1","candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"Hello"}]}}],'
                b'"usageMetadata":{"promptTokenCount":10,"totalTokenCount":15}}\n\n'
            ),
        ]
        events = self._run(self._collect_stream(chunks))
        text_deltas = [
            d["delta"]["text"]
            for _, d in events
            if isinstance(d, dict) and d.get("type") == "content_block_delta"
            and d.get("delta", {}).get("type") == "text_delta"
        ]
        self.assertEqual(text_deltas, ["Hel", "lo"])

    def test_stream_incremental_text_passthrough(self):
        """Incremental text (not cumulative) passes straight through unchanged."""
        chunks = [
            b'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}],"usageMetadata":{"promptTokenCount":4,"totalTokenCount":6}}\n\n',
            b'data: {"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"lo"}]}}],"usageMetadata":{"promptTokenCount":4,"totalTokenCount":8}}\n\n',
        ]
        events = self._run(self._collect_stream(chunks))
        text_deltas = [
            d["delta"]["text"]
            for _, d in events
            if isinstance(d, dict) and d.get("type") == "content_block_delta"
            and d.get("delta", {}).get("type") == "text_delta"
        ]
        self.assertEqual(text_deltas, ["Hel", "lo"])

    def test_stream_crlf_delimiters(self):
        """TEST S3 (cc-switch): CRLF line endings in SSE stream are handled correctly."""
        chunks = [
            b'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}],"usageMetadata":{"promptTokenCount":4,"totalTokenCount":6}}\r\n\r\n',
            b'data: {"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"Hi there"}]}}],"usageMetadata":{"promptTokenCount":4,"totalTokenCount":9}}\r\n\r\n',
        ]
        events = self._run(self._collect_stream(chunks))
        text_deltas = [
            d["delta"]["text"]
            for _, d in events
            if isinstance(d, dict) and d.get("type") == "content_block_delta"
            and d.get("delta", {}).get("type") == "text_delta"
        ]
        # First chunk "Hi", second cumulative "Hi there" → delta " there"
        self.assertEqual(text_deltas, ["Hi", " there"])

    def test_stream_tool_call_real_id_preserved(self):
        """Streaming: real functionCall.id from Gemini is used as tool_use.id."""
        chunks = [
            (
                b'data: {"responseId":"r1","candidates":[{"finishReason":"STOP","content":{"parts":['
                b'{"functionCall":{"id":"call_1","name":"Bash","args":{"command":"ls"}}}]}}],'
                b'"usageMetadata":{"promptTokenCount":5,"totalTokenCount":8}}\n\n'
            ),
        ]
        events = self._run(self._collect_stream(chunks))
        tool_start = next(
            d for _, d in events
            if isinstance(d, dict) and d.get("type") == "content_block_start"
            and d.get("content_block", {}).get("type") == "tool_use"
        )
        self.assertEqual(tool_start["content_block"]["id"], "call_1")

    def test_stream_tool_call_no_id_synthesized(self):
        """Streaming: functionCall without id gets synthesised toolu_* id."""
        chunks = [
            (
                b'data: {"responseId":"r2","candidates":[{"finishReason":"STOP","content":{"parts":['
                b'{"functionCall":{"name":"search","args":{"q":"hi"}}}]}}],'
                b'"usageMetadata":{"promptTokenCount":3,"totalTokenCount":5}}\n\n'
            ),
        ]
        events = self._run(self._collect_stream(chunks))
        tool_start = next(
            d for _, d in events
            if isinstance(d, dict) and d.get("type") == "content_block_start"
            and d.get("content_block", {}).get("type") == "tool_use"
        )
        self.assertTrue(tool_start["content_block"]["id"].startswith("toolu_"))

    # ------------------------------------------------------------------
    # Interrupted-response history (Google path)
    # ------------------------------------------------------------------

    def _google_interrupted_body(self, partial_assistant_content):
        from simple_api_router.converter_google import anthropic_to_google_request
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Do something"},
                {"role": "assistant", "content": partial_assistant_content},
                {"role": "user", "content": "Never mind"},
            ],
        }
        return anthropic_to_google_request(body, "gemini-2.5-flash")

    def test_google_interrupted_empty_content_placeholder(self):
        """Interrupted before any block (content=[]) → model placeholder, not dropped."""
        result = self._google_interrupted_body([])
        roles = [c["role"] for c in result["contents"]]
        # user, model, user  — alternating
        self.assertEqual(roles, ["user", "model", "user"])
        model_parts = result["contents"][1]["parts"]
        self.assertEqual(len(model_parts), 1)
        self.assertEqual(model_parts[0]["text"], "")

    def test_google_interrupted_redacted_thinking_placeholder(self):
        """Only redacted_thinking (cannot round-trip) → placeholder, not dropped."""
        result = self._google_interrupted_body([
            {"type": "redacted_thinking", "data": "ENCRYPTED"},
        ])
        roles = [c["role"] for c in result["contents"]]
        self.assertEqual(roles, ["user", "model", "user"])

    def test_google_interrupted_thinking_only_preserved(self):
        """Partial thinking-only turn → thought part present (not a placeholder)."""
        result = self._google_interrupted_body([
            {"type": "thinking", "thinking": "some reasoning"},
        ])
        roles = [c["role"] for c in result["contents"]]
        self.assertEqual(roles, ["user", "model", "user"])
        model_parts = result["contents"][1]["parts"]
        self.assertTrue(model_parts[0].get("thought"))
        self.assertEqual(model_parts[0]["text"], "some reasoning")

    def test_google_interrupted_partial_text_preserved(self):
        """Partial text survives."""
        result = self._google_interrupted_body([
            {"type": "text", "text": "partial answer"},
        ])
        roles = [c["role"] for c in result["contents"]]
        self.assertEqual(roles, ["user", "model", "user"])
        self.assertEqual(result["contents"][1]["parts"][0]["text"], "partial answer")

    def test_google_interrupted_tool_only_no_placeholder(self):
        """Tool-only turn already has parts → no extra placeholder."""
        result = self._google_interrupted_body([
            {"type": "tool_use", "id": "call_xyz", "name": "search", "input": {"q": "hi"}},
        ])
        roles = [c["role"] for c in result["contents"]]
        self.assertEqual(roles, ["user", "model", "user"])
        model_parts = result["contents"][1]["parts"]
        self.assertIn("functionCall", model_parts[0])

    def test_google_follow_up_user_message_present(self):
        """The new user message is the last content item after an interrupted turn."""
        result = self._google_interrupted_body([])
        last = result["contents"][-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(last["parts"][0]["text"], "Never mind")

    def test_stream_incremental_text_shorter_than_accumulated_is_skipped(self):
        """If Gemini sends text shorter than what's accumulated and it doesn't
        start with the accumulated prefix, it's a retransmission — skip it."""
        # Simulated: chunk 1 = "ABC", chunk 2 = "AB" (shorter retransmission), chunk 3 = "ABCD"
        chunks = [
            b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"ABC"}]}}]}\n\n',
            b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"AB"}]}}]}\n\n',
            b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"ABCD"}]}}]}\n\n',
        ]
        events = self._run(self._collect_stream(chunks))
        data_events = [d for k, d in events if k == "data"]
        text_deltas = [d for d in data_events
                       if d["type"] == "content_block_delta"
                       and d.get("delta", {}).get("type") == "text_delta"]
        combined = "".join(d["delta"]["text"] for d in text_deltas)
        # "AB" retransmission should be silently skipped; only "ABC" + "D" emitted
        self.assertEqual(combined, "ABCD")

    def test_stream_cumulative_text_len_guard_blocks_partial_retransmission(self):
        """When accumulated="ABC" and new="AB" (shorter), the length guard must
        detect it as a retransmission and produce an empty delta."""
        chunks = [
            b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"ABC"}]}}]}\n\n',
            # Retransmission: "AB" is shorter, can't be cumulative
            b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"AB"}]}}]}\n\n',
            b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"ABCDEF"}]}}]}\n\n',
        ]
        events = self._run(self._collect_stream(chunks))
        data_events = [d for k, d in events if k == "data"]
        text_deltas = [d for d in data_events
                       if d["type"] == "content_block_delta"
                       and d.get("delta", {}).get("type") == "text_delta"]
        combined = "".join(d["delta"]["text"] for d in text_deltas)
        self.assertEqual(combined, "ABCDEF")


class TestStreamIdlePing(unittest.TestCase):
    def test_emits_ping_when_upstream_is_silent(self):
        from simple_api_router.converter_utils import stream_with_idle_ping

        async def _slow_source() -> AsyncIterator[bytes]:
            yield b"event: message_start\ndata: {}\n\n"
            await asyncio.sleep(0.05)
            yield b"event: content_block_delta\ndata: {}\n\n"

        async def _collect() -> list[bytes]:
            out: list[bytes] = []
            async for chunk in stream_with_idle_ping(_slow_source(), idle_seconds=0.02):
                out.append(chunk)
            return out

        chunks = asyncio.run(_collect())
        joined = b"".join(chunks)
        self.assertIn(b"event: ping", joined)
        self.assertIn(b"event: message_start", joined)
        self.assertIn(b"event: content_block_delta", joined)


if __name__ == "__main__":
    unittest.main()
