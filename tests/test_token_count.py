"""Tests for token_count helpers."""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

import httpx

from simple_api_router.token_count import (
    _responses_count_payload,
    count_openai_chat_tokens,
    count_openai_responses_api_tokens,
    count_responses_tokens,
    estimate_anthropic_body_tokens,
)


class TestOpenAIChatTokenCount(unittest.TestCase):
    def test_counts_text_messages(self):
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello world"},
            ],
        }
        tokens = count_openai_chat_tokens(body, "gpt-4o")
        self.assertGreater(tokens, 10)
        self.assertLess(tokens, 80)

    def test_disabled_thinking_not_in_openai_body_still_counts_messages(self):
        body = {
            "messages": [{"role": "user", "content": "ping"}],
        }
        self.assertGreater(count_openai_chat_tokens(body, "gpt-4o-mini"), 0)

    def test_image_url_adds_vision_overhead(self):
        body = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ],
            }],
        }
        text_only = count_openai_chat_tokens(
            {"messages": [{"role": "user", "content": "look"}]},
            "gpt-4o",
        )
        with_image = count_openai_chat_tokens(body, "gpt-4o")
        self.assertGreater(with_image, text_only + 500)

    def test_tools_included(self):
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }],
        }
        without_tools = count_openai_chat_tokens(
            {"messages": [{"role": "user", "content": "hi"}]},
            "gpt-4o",
        )
        self.assertGreater(count_openai_chat_tokens(body, "gpt-4o"), without_tools)


class TestOpenAIResponsesApiTokenCount(unittest.IsolatedAsyncioTestCase):
    async def test_calls_input_tokens_endpoint(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"input_tokens": 42}
        client.post.return_value = response

        total = await count_openai_responses_api_tokens(
            {
                "model": "gpt-4o",
                "input": [{"role": "user", "content": "hi"}],
                "stream": True,
                "max_output_tokens": 100,
            },
            "https://api.openai.com",
            {"Authorization": "Bearer sk-test"},
            client,
        )
        self.assertEqual(total, 42)
        _args, kwargs = client.post.call_args
        self.assertEqual(_args[0], "https://api.openai.com/v1/responses/input_tokens")
        payload = kwargs["json"]
        self.assertNotIn("stream", payload)
        self.assertNotIn("max_output_tokens", payload)
        self.assertEqual(payload["model"], "gpt-4o")


class TestResponsesCountPayload(unittest.TestCase):
    def test_strips_generation_fields(self):
        payload = _responses_count_payload({
            "model": "o3",
            "input": [],
            "stream": True,
            "temperature": 0.2,
            "tools": [{"type": "function", "name": "fn"}],
        })
        self.assertEqual(set(payload.keys()), {"model", "input", "tools"})


class TestResponsesTokenCount(unittest.TestCase):
    def test_counts_instructions_and_input(self):
        body = {
            "instructions": "Be concise.",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            ],
        }
        tokens = count_responses_tokens(body, "gpt-4o")
        self.assertGreater(tokens, 8)
        self.assertLess(tokens, 60)


class TestAnthropicBodyEstimate(unittest.TestCase):
    def test_system_and_messages(self):
        body = {
            "system": "Rules here.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            ],
        }
        tokens = estimate_anthropic_body_tokens(body, "claude-sonnet-4-5")
        self.assertGreater(tokens, 10)
        self.assertLess(tokens, 120)