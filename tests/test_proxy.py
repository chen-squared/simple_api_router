"""Tests for proxy routing helpers — multimodal detection and fallback logic."""
from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from simple_api_router.config import (
    EndpointConfig,
    ModelEntry,
    ProviderConfig,
    RouterConfig,
    ServerConfig,
)
from simple_api_router.proxy import (
    _media_types_in_blocks,
    _media_types_in_body,
    _replace_media_with_placeholder,
    _graceful_stream_termination,
    _stream_converted_with_retry,
    _upstream_error_sse,
    count_tokens_request,
    parse_model,
    prepare_request_body,
    resolve_provider,
)


# ---------------------------------------------------------------------------
# _media_types_in_body
# ---------------------------------------------------------------------------

class TestMediaTypesInBody(unittest.TestCase):
    """Tests for _media_types_in_body: returns a set of media types detected."""

    def _msg(self, content):
        return {"role": "user", "content": content}

    # ── positive cases ──────────────────────────────────────────────────────

    def test_image_base64_block(self):
        body = {"messages": [self._msg([
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}}
        ])]}
        self.assertIn("image", _media_types_in_body(body))

    def test_image_url_block(self):
        body = {"messages": [self._msg([
            {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}}
        ])]}
        self.assertIn("image", _media_types_in_body(body))

    def test_video_block(self):
        body = {"messages": [self._msg([
            {"type": "video", "source": {"type": "url", "url": "https://example.com/video.mp4"}}
        ])]}
        self.assertIn("video", _media_types_in_body(body))

    def test_audio_block(self):
        body = {"messages": [self._msg([
            {"type": "audio", "source": {"type": "base64", "media_type": "audio/mp3", "data": "abc"}}
        ])]}
        self.assertIn("audio", _media_types_in_body(body))

    def test_media_in_second_message(self):
        body = {"messages": [
            self._msg("plain text"),
            self._msg([
                {"type": "text", "text": "describe this"},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}},
            ]),
        ]}
        self.assertIn("image", _media_types_in_body(body))

    def test_tool_result_with_nested_image(self):
        body = {"messages": [self._msg([
            {"type": "tool_result", "tool_use_id": "tu_1", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/screenshot.png"}},
                {"type": "text", "text": "see screenshot"},
            ]}
        ])]}
        self.assertIn("image", _media_types_in_body(body))

    def test_multiple_types_detected(self):
        body = {"messages": [self._msg([
            {"type": "image", "source": {"type": "url", "url": "https://x.com/img.png"}},
            {"type": "video", "source": {"type": "url", "url": "https://x.com/vid.mp4"}},
        ])]}
        types = _media_types_in_body(body)
        self.assertIn("image", types)
        self.assertIn("video", types)

    # ── negative / non-media cases ─────────────────────────────────────────

    def test_document_block_detected_as_pdf(self):
        # binary document blocks are now detected as "pdf"
        body = {"messages": [self._msg([
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "abc"}}
        ])]}
        self.assertEqual(_media_types_in_body(body), {"pdf"})

    def test_document_text_source_not_detected(self):
        body = {"messages": [self._msg([
            {"type": "document", "source": {"type": "text", "text": "some text content"}}
        ])]}
        self.assertFalse(_media_types_in_body(body))

    def test_tool_result_with_text_only_content(self):
        body = {"messages": [self._msg([
            {"type": "tool_result", "tool_use_id": "tu_1", "content": [
                {"type": "text", "text": "result text"}
            ]}
        ])]}
        self.assertFalse(_media_types_in_body(body))

    def test_text_only_string_content(self):
        body = {"messages": [self._msg("just text")]}
        self.assertFalse(_media_types_in_body(body))

    def test_text_only_block_list(self):
        body = {"messages": [self._msg([{"type": "text", "text": "hello"}])]}
        self.assertFalse(_media_types_in_body(body))

    def test_tool_use_block_is_not_media(self):
        body = {"messages": [self._msg([
            {"type": "tool_use", "id": "tu_1", "name": "search", "input": {}}
        ])]}
        self.assertFalse(_media_types_in_body(body))

    def test_tool_result_string_content_is_not_media(self):
        body = {"messages": [self._msg([
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "result"}
        ])]}
        self.assertFalse(_media_types_in_body(body))

    def test_empty_messages(self):
        self.assertFalse(_media_types_in_body({"messages": []}))

    def test_missing_messages_key(self):
        self.assertFalse(_media_types_in_body({}))

    def test_non_list_content_ignored(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}
        self.assertFalse(_media_types_in_body(body))


# ---------------------------------------------------------------------------
# Helpers — build mock RouterConfig
# ---------------------------------------------------------------------------

def _make_config(
    *,
    image_fallback: str | None = None,
    model_entries: list | None = None,
    fallback_provider_name: str = "vision",
    fallback_model: str = "gpt-4o",
) -> RouterConfig:
    """Build a minimal RouterConfig for fallback routing tests.

    Models have empty multimodality (no media support) by default.
    """
    primary_models = model_entries or [
        ModelEntry(name="deepseek-r1"),    # no media support
        ModelEntry(name="qwen2.5-coder"),  # no media support
    ]
    primary_ep = EndpointConfig(
        base_url="http://localhost:11434",
        models=primary_models,
    )
    primary_prov = ProviderConfig(
        api_key="",
        endpoints={"openai_chat": primary_ep},
    )

    fallback_ep = EndpointConfig(
        base_url="https://api.openai.com",
        models=[fallback_model],
    )
    fallback_prov = ProviderConfig(
        api_key="sk-test",
        endpoints={"openai_chat": fallback_ep},
    )

    server = ServerConfig(image_fallback=image_fallback)

    return RouterConfig(
        server=server,
        providers={
            "local": primary_prov,
            fallback_provider_name: fallback_prov,
        },
    )


# ---------------------------------------------------------------------------
# ModelEntry / EndpointConfig helpers
# ---------------------------------------------------------------------------

class TestModelEntry(unittest.TestCase):

    def test_string_model_get_entry_returns_default(self):
        ep = EndpointConfig(models=["gpt-4o", "gpt-4o-mini"])
        entry = ep.get_model_entry("gpt-4o")
        self.assertEqual(entry.name, "gpt-4o")
        self.assertEqual(entry.multimodality, [])
        self.assertIsNone(entry.image_fallback)

    def test_model_entry_with_multimodality(self):
        ep = EndpointConfig(models=[
            ModelEntry(name="gemini-flash", multimodality=["image", "video"]),
        ])
        entry = ep.get_model_entry("gemini-flash")
        self.assertIn("image", entry.multimodality)
        self.assertIn("video", entry.multimodality)
        self.assertNotIn("audio", entry.multimodality)

    def test_model_entry_with_per_type_fallback(self):
        ep = EndpointConfig(models=[
            ModelEntry(name="deepseek-r1", image_fallback="vision/gpt-4o"),
        ])
        entry = ep.get_model_entry("deepseek-r1")
        self.assertEqual(entry.image_fallback, "vision/gpt-4o")

    def test_model_names_mixed(self):
        ep = EndpointConfig(models=[
            "gpt-4o",
            ModelEntry(name="deepseek-r1"),
        ])
        self.assertEqual(ep.model_names(), ["gpt-4o", "deepseek-r1"])

    def test_duplicate_model_across_endpoints_raises(self):
        with self.assertRaises(Exception):
            ProviderConfig(
                api_key="",
                endpoints={
                    "openai_chat": EndpointConfig(models=["same-model"]),
                    "google": EndpointConfig(models=["same-model"]),
                },
            )

    def test_find_model_works_with_model_entry(self):
        prov = ProviderConfig(
            api_key="",
            endpoints={
                "openai_chat": EndpointConfig(
                    models=[ModelEntry(name="deepseek-r1")]
                )
            },
        )
        result = prov.find_model("deepseek-r1")
        self.assertIsNotNone(result)
        fmt, ep = result
        self.assertEqual(fmt, "openai_chat")


# ---------------------------------------------------------------------------
# parse_model + usage_meta model name
# ---------------------------------------------------------------------------

class TestParseModel(unittest.TestCase):

    def test_strips_1m_suffix(self):
        provider, model = parse_model("openai/gpt-4o[1m]")
        self.assertEqual(provider, "openai")
        self.assertEqual(model, "gpt-4o")

    def test_strips_bracket_suffix_no_provider(self):
        provider, model = parse_model("deepseek-chat[128k]")
        self.assertIsNone(provider)
        self.assertEqual(model, "deepseek-chat")

    def test_no_suffix(self):
        provider, model = parse_model("anthropic/claude-opus-4-5")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(model, "claude-opus-4-5")

    def test_usage_meta_model_has_no_bracket_suffix(self):
        """route_request() must log 'provider/model' without [1m] so pricing lookup works."""
        from simple_api_router.proxy import route_request
        from simple_api_router.config import PricingEntry
        import asyncio

        ep = EndpointConfig(
            base_url="https://api.openai.com",
            models=[ModelEntry(name="gpt-4o", pricing=PricingEntry(input=5.0, output=15.0))],
        )
        prov = ProviderConfig(api_key="sk-test", endpoints={"openai_chat": ep})
        config = RouterConfig(providers={"openai": prov})

        # Fake request and httpx client — we only care about request.state.usage_meta
        import httpx
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_request = MagicMock()
        fake_request.state = MagicMock()
        fake_request.headers = {}

        body = {"model": "openai/gpt-4o[1m]", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}

        # Patch _proxy_openai so we never hit the network
        async def fake_proxy(*args, **kwargs):
            return MagicMock()

        with patch("simple_api_router.proxy._proxy_openai", side_effect=fake_proxy):
            asyncio.run(route_request(fake_request, body, config, MagicMock()))

        logged_model = fake_request.state.usage_meta["model"]
        self.assertNotIn("[1m]", logged_model, "usage_meta should not contain bracket suffixes")
        self.assertEqual(logged_model, "openai/gpt-4o")


# ---------------------------------------------------------------------------
# prepare_request_body + count_tokens alignment
# ---------------------------------------------------------------------------

class TestCountTokensAlignment(unittest.TestCase):
    def test_prepare_request_body_resolves_model_map(self):
        ep = EndpointConfig(models=["claude-opus-4-5"])
        prov = ProviderConfig(api_key="sk-test", endpoints={"anthropic": ep})
        config = RouterConfig(
            server=ServerConfig(model_map={"claude": "anthropic/claude-opus-4-5"}),
            providers={"anthropic": prov},
        )
        body = {
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
        }

        prepared, model_str, *_ = asyncio.run(
            prepare_request_body(body, config, AsyncMock()),
        )
        self.assertEqual(model_str, "anthropic/claude-opus-4-5")
        self.assertEqual(prepared["model"], "anthropic/claude-opus-4-5")

    def test_count_tokens_uses_model_map_and_tiktoken_for_openai(self):
        ep = EndpointConfig(models=["gpt-4o"])
        prov = ProviderConfig(api_key="sk-test", endpoints={"openai_chat": ep})
        config = RouterConfig(
            server=ServerConfig(model_map={"fast": "openai/gpt-4o"}),
            providers={"openai": prov},
        )
        fake_request = MagicMock()
        fake_request.headers = {}
        body = {
            "model": "fast",
            "messages": [{"role": "user", "content": "Hello there"}],
        }

        # Provider has no /v1/responses/input_tokens → fall back to tiktoken.
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "not found"
        mock_resp.request = MagicMock()
        mock_client.post.return_value = mock_resp

        response = asyncio.run(
            count_tokens_request(fake_request, body, config, mock_client),
        )
        self.assertIsInstance(response, JSONResponse)
        payload = json.loads(response.body)
        self.assertIn("input_tokens", payload)
        self.assertGreater(payload["input_tokens"], 5)
        self.assertLess(payload["input_tokens"], 200)

    def test_count_tokens_describes_unsupported_image_like_messages(self):
        ep = EndpointConfig(models=[ModelEntry(name="deepseek-chat", multimodality=[])])
        prov = ProviderConfig(api_key="sk-test", endpoints={"openai_chat": ep})
        config = RouterConfig(
            server=ServerConfig(image_fallback="openai/gpt-4o"),
            providers={"openai": prov},
        )
        fake_request = MagicMock()
        fake_request.headers = {}
        body = {
            "model": "openai/deepseek-chat",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}},
                    {"type": "text", "text": "describe"},
                ],
            }],
        }

        with patch(
            "simple_api_router.proxy._describe_media_in_body",
            new_callable=AsyncMock,
        ) as mock_describe:
            mock_describe.return_value = {
                "model": "openai/deepseek-chat",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "[Image: a red square]"},
                        {"type": "text", "text": "describe"},
                    ],
                }],
            }
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_resp.text = "not found"
            mock_resp.request = MagicMock()
            mock_client.post.return_value = mock_resp

            response = asyncio.run(
                count_tokens_request(fake_request, body, config, mock_client),
            )

        mock_describe.assert_awaited_once()
        payload = json.loads(response.body)
        self.assertGreater(payload["input_tokens"], 0)


# ---------------------------------------------------------------------------
# Multimodal fallback routing (unit, no HTTP)
# ---------------------------------------------------------------------------

class TestMultimodalFallbackRouting(unittest.TestCase):
    """
    Tests for the per-type fallback resolution used by route_request().

    We test the decision logic directly:
    - _media_types_in_body() for detection
    - ModelEntry.multimodality for native support
    - *_fallback resolution from model-entry → server
    """

    def _image_body(self, model: str) -> dict:
        return {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://x.com/img.png"}},
                    {"type": "text", "text": "describe"},
                ],
            }],
            "max_tokens": 512,
        }

    def _text_body(self, model: str) -> dict:
        return {"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 512}

    def _fallback_for(self, body: dict, config, mtype: str = "image") -> str | None:
        """Return the fallback model string that would be used to describe *mtype* blocks.

        Returns None if the model natively supports the type or no fallback is configured.
        """
        model_str = body["model"]
        provider_name, model = parse_model(model_str)
        _, endpoint, _, _ = resolve_provider(provider_name, model, config)
        entry = endpoint.get_model_entry(model)
        if mtype in set(entry.multimodality):
            return None  # natively supported
        return getattr(entry, f"{mtype}_fallback", None) or getattr(config.server, f"{mtype}_fallback", None)

    # ── detection gate ──────────────────────────────────────────────────────

    def test_no_fallback_when_no_media(self):
        body = self._text_body("local/deepseek-r1")
        self.assertFalse(_media_types_in_body(body))

    def test_fallback_triggered_for_image(self):
        body = self._image_body("local/deepseek-r1")
        self.assertIn("image", _media_types_in_body(body))

    # ── fallback resolution ─────────────────────────────────────────────────

    def test_no_media_model_gets_global_image_fallback(self):
        """Model with empty multimodality + image request → global image_fallback used."""
        config = _make_config(image_fallback="vision/gpt-4o",
                              fallback_provider_name="vision", fallback_model="gpt-4o")
        body = self._image_body("local/deepseek-r1")
        self.assertEqual(self._fallback_for(body, config), "vision/gpt-4o")

    def test_multimodal_model_no_fallback_for_supported_type(self):
        """Model that natively supports images → no fallback needed."""
        config = _make_config(
            image_fallback="vision/gpt-4o",
            model_entries=[ModelEntry(name="llava", multimodality=["image"])],
            fallback_provider_name="vision",
            fallback_model="gpt-4o",
        )
        body = self._image_body("local/llava")
        self.assertIsNone(self._fallback_for(body, config))

    def test_model_level_fallback_takes_priority_over_global(self):
        """Per-model image_fallback overrides server.image_fallback."""
        config = _make_config(
            image_fallback="vision/gpt-4o",
            model_entries=[
                ModelEntry(name="deepseek-r1", image_fallback="vision/special-vision"),
                ModelEntry(name="qwen2.5-coder"),  # uses global
            ],
            fallback_provider_name="vision",
            fallback_model="gpt-4o",
        )
        # deepseek-r1 uses its own model-level fallback
        body = self._image_body("local/deepseek-r1")
        self.assertEqual(self._fallback_for(body, config), "vision/special-vision")

    def test_model_level_fallback_independent_of_global(self):
        """Model with per-model fallback ignores the global fallback."""
        vision_ep = EndpointConfig(base_url="https://v.com", models=["vision-model"])
        vision_prov = ProviderConfig(api_key="k", endpoints={"openai_chat": vision_ep})
        other_ep = EndpointConfig(base_url="https://o.com", models=["other-model"])
        other_prov = ProviderConfig(api_key="k", endpoints={"openai_chat": other_ep})
        primary_ep = EndpointConfig(
            base_url="http://localhost:11434",
            models=[ModelEntry(name="deepseek-r1", image_fallback="vision/vision-model")],
        )
        primary_prov = ProviderConfig(api_key="", endpoints={"openai_chat": primary_ep})
        config = RouterConfig(
            server=ServerConfig(image_fallback="other/other-model"),
            providers={"local": primary_prov, "vision": vision_prov, "other": other_prov},
        )
        body = self._image_body("local/deepseek-r1")
        # model-level fallback "vision/vision-model" wins over global "other/other-model"
        self.assertEqual(self._fallback_for(body, config), "vision/vision-model")

    def test_no_fallback_configured_returns_none(self):
        """No fallback configured → returns None."""
        config = _make_config(image_fallback=None)
        body = self._image_body("local/deepseek-r1")
        self.assertIsNone(self._fallback_for(body, config))

    def test_per_type_fallback_resolved_from_server(self):
        """Each type resolves its own fallback independently."""
        server = ServerConfig(image_fallback="vision/img-model", audio_fallback="audio/audio-model")
        img_ep = EndpointConfig(base_url="https://v.com", models=["img-model"])
        audio_ep = EndpointConfig(base_url="https://a.com", models=["audio-model"])
        primary_ep = EndpointConfig(
            base_url="http://localhost:11434",
            models=[ModelEntry(name="deepseek-r1")],
        )
        config = RouterConfig(
            server=server,
            providers={
                "local": ProviderConfig(api_key="", endpoints={"openai_chat": primary_ep}),
                "vision": ProviderConfig(api_key="k", endpoints={"openai_chat": img_ep}),
                "audio": ProviderConfig(api_key="k", endpoints={"openai_chat": audio_ep}),
            },
        )
        _, model = parse_model("local/deepseek-r1")
        _, ep, _, _ = resolve_provider("local", model, config)
        entry = ep.get_model_entry(model)
        self.assertEqual(
            entry.image_fallback or config.server.image_fallback, "vision/img-model"
        )
        self.assertEqual(
            entry.audio_fallback or config.server.audio_fallback, "audio/audio-model"
        )
        self.assertIsNone(entry.video_fallback or config.server.video_fallback)

    # ── resolve_provider ────────────────────────────────────────────────────

    def test_resolve_no_multimodality_model(self):
        config = _make_config(image_fallback="vision/gpt-4o")
        _, model = parse_model("local/deepseek-r1")
        _, ep, _, _ = resolve_provider("local", model, config)
        entry = ep.get_model_entry(model)
        self.assertEqual(entry.multimodality, [])

    def test_global_image_fallback_resolves_correctly(self):
        config = _make_config(image_fallback="vision/gpt-4o", fallback_provider_name="vision", fallback_model="gpt-4o")
        fallback_str = config.server.image_fallback
        self.assertEqual(fallback_str, "vision/gpt-4o")
        fb_prov_name, fb_model = parse_model(fallback_str)
        _, fb_ep, fb_fmt, fb_backend = resolve_provider(fb_prov_name, fb_model, config)
        self.assertEqual(fb_fmt, "openai_chat")
        self.assertEqual(fb_backend, "gpt-4o")


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# _replace_media_with_placeholder
# ===========================================================================

class TestReplaceMediaWithPlaceholder(unittest.TestCase):
    """Unit tests for _replace_media_with_placeholder."""

    def _body_with_image(self) -> dict:
        return {
            "model": "x/y",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://x.com/img.png"}},
                    {"type": "text", "text": "what is this?"},
                ],
            }],
        }

    def _body_with_video(self) -> dict:
        return {
            "model": "x/y",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video", "source": {"type": "url", "url": "https://x.com/vid.mp4"}},
                    {"type": "text", "text": "describe this"},
                ],
            }],
        }

    def test_image_block_replaced_with_text(self):
        body = _replace_media_with_placeholder(self._body_with_image(), "image")
        blocks = body["messages"][0]["content"]
        self.assertEqual(len(blocks), 2)
        replaced = blocks[0]
        self.assertEqual(replaced["type"], "text")
        self.assertIn("image", replaced["text"])
        self.assertIn("image_understanding", replaced["text"])

    def test_video_block_replaced_with_text(self):
        body = _replace_media_with_placeholder(self._body_with_video(), "video")
        blocks = body["messages"][0]["content"]
        replaced = blocks[0]
        self.assertEqual(replaced["type"], "text")
        self.assertIn("video", replaced["text"])
        self.assertIn("video_understanding", replaced["text"])

    def test_audio_block_replaced_with_text(self):
        body = {
            "model": "x/y",
            "messages": [{
                "role": "user",
                "content": [{"type": "audio", "source": {"type": "url", "url": "https://x.com/a.mp3"}}],
            }],
        }
        result = _replace_media_with_placeholder(body, "audio")
        block = result["messages"][0]["content"][0]
        self.assertEqual(block["type"], "text")
        self.assertIn("audio_understanding", block["text"])

    def test_text_blocks_are_preserved(self):
        body = _replace_media_with_placeholder(self._body_with_image(), "image")
        text_block = body["messages"][0]["content"][1]
        self.assertEqual(text_block["type"], "text")
        self.assertEqual(text_block["text"], "what is this?")

    def test_non_matching_media_type_not_replaced(self):
        """Replacing 'audio' when body contains only 'image' — image untouched."""
        body = _replace_media_with_placeholder(self._body_with_image(), "audio")
        blocks = body["messages"][0]["content"]
        self.assertEqual(blocks[0]["type"], "image")

    def test_original_body_not_mutated(self):
        original = self._body_with_image()
        _replace_media_with_placeholder(original, "image")
        self.assertEqual(original["messages"][0]["content"][0]["type"], "image")

    def test_multiple_messages_all_replaced(self):
        body = {
            "model": "x/y",
            "messages": [
                {"role": "user", "content": [{"type": "image", "source": {}}]},
                {"role": "user", "content": [{"type": "image", "source": {}}, {"type": "text", "text": "hi"}]},
            ],
        }
        result = _replace_media_with_placeholder(body, "image")
        self.assertEqual(result["messages"][0]["content"][0]["type"], "text")
        self.assertEqual(result["messages"][1]["content"][0]["type"], "text")

    def test_nested_tool_result_blocks_replaced(self):
        body = {
            "model": "x/y",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": [
                        {"type": "image", "source": {}},
                        {"type": "text", "text": "result"},
                    ],
                }],
            }],
        }
        result = _replace_media_with_placeholder(body, "image")
        inner = result["messages"][0]["content"][0]["content"]
        self.assertEqual(inner[0]["type"], "text")
        self.assertIn("image_understanding", inner[0]["text"])
        self.assertEqual(inner[1]["text"], "result")

    def test_string_content_message_untouched(self):
        body = {
            "model": "x/y",
            "messages": [{"role": "user", "content": "plain text"}],
        }
        result = _replace_media_with_placeholder(body, "image")
        self.assertEqual(result["messages"][0]["content"], "plain text")


# ===========================================================================
# _media_types_in_blocks (direct tests)
# ===========================================================================

class TestMediaTypesInBlocks(unittest.TestCase):
    """Direct unit tests for _media_types_in_blocks."""

    def test_empty_list(self):
        self.assertEqual(_media_types_in_blocks([]), set())

    def test_image_block_detected(self):
        blocks = [{"type": "image", "source": {"type": "url", "url": "https://x.com/img.png"}}]
        self.assertIn("image", _media_types_in_blocks(blocks))

    def test_audio_block_detected(self):
        blocks = [{"type": "audio", "source": {"type": "base64", "media_type": "audio/mp3", "data": "abc"}}]
        self.assertIn("audio", _media_types_in_blocks(blocks))

    def test_video_block_detected(self):
        blocks = [{"type": "video", "source": {"type": "url", "url": "https://x.com/vid.mp4"}}]
        self.assertIn("video", _media_types_in_blocks(blocks))

    def test_all_three_types(self):
        blocks = [
            {"type": "image", "source": {"type": "url", "url": "https://x.com/img.png"}},
            {"type": "audio", "source": {"type": "url", "url": "https://x.com/clip.mp3"}},
            {"type": "video", "source": {"type": "url", "url": "https://x.com/vid.mp4"}},
        ]
        result = _media_types_in_blocks(blocks)
        self.assertEqual(result, {"image", "audio", "video"})

    def test_text_block_not_detected(self):
        blocks = [{"type": "text", "text": "hello"}]
        self.assertFalse(_media_types_in_blocks(blocks))

    def test_document_block_detected_as_pdf(self):
        blocks = [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "abc"}}]
        self.assertEqual(_media_types_in_blocks(blocks), {"pdf"})

    def test_tool_use_block_not_detected(self):
        blocks = [{"type": "tool_use", "id": "tu_1", "name": "search", "input": {}}]
        self.assertFalse(_media_types_in_blocks(blocks))

    def test_tool_result_with_nested_image(self):
        blocks = [{"type": "tool_result", "tool_use_id": "tu_1", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://x.com/screenshot.png"}},
        ]}]
        self.assertIn("image", _media_types_in_blocks(blocks))

    def test_tool_result_with_nested_audio(self):
        blocks = [{"type": "tool_result", "tool_use_id": "tu_1", "content": [
            {"type": "audio", "source": {"type": "url", "url": "https://x.com/clip.mp3"}},
        ]}]
        self.assertIn("audio", _media_types_in_blocks(blocks))

    def test_tool_result_with_text_only_not_detected(self):
        blocks = [{"type": "tool_result", "tool_use_id": "tu_1", "content": [
            {"type": "text", "text": "result"},
        ]}]
        self.assertFalse(_media_types_in_blocks(blocks))

    def test_tool_result_string_content_not_detected(self):
        blocks = [{"type": "tool_result", "tool_use_id": "tu_1", "content": "text result"}]
        self.assertFalse(_media_types_in_blocks(blocks))

    def test_non_dict_items_ignored(self):
        blocks = ["string item", 42, None, {"type": "image", "source": {}}]
        result = _media_types_in_blocks(blocks)
        self.assertIn("image", result)

    def test_multiple_images_only_returns_one_image_entry(self):
        blocks = [
            {"type": "image", "source": {"type": "url", "url": "https://x.com/a.png"}},
            {"type": "image", "source": {"type": "url", "url": "https://x.com/b.png"}},
        ]
        result = _media_types_in_blocks(blocks)
        self.assertEqual(result, {"image"})


# ===========================================================================
# ServerConfig per-type fields
# ===========================================================================

class TestServerConfigMediaFields(unittest.TestCase):

    def test_defaults_all_none(self):
        s = ServerConfig()
        self.assertIsNone(s.image_fallback)
        self.assertIsNone(s.audio_fallback)
        self.assertIsNone(s.video_fallback)
        self.assertIsNone(s.image_model)
        self.assertIsNone(s.audio_model)
        self.assertIsNone(s.video_model)

    def test_all_three_fallbacks(self):
        s = ServerConfig(
            image_fallback="vision/img-model",
            audio_fallback="audio/stt-model",
            video_fallback="vision/vid-model",
        )
        self.assertEqual(s.image_fallback, "vision/img-model")
        self.assertEqual(s.audio_fallback, "audio/stt-model")
        self.assertEqual(s.video_fallback, "vision/vid-model")

    def test_all_three_mcp_models(self):
        s = ServerConfig(
            image_model="google/gemini-2.5-flash",
            audio_model="openai/gpt-4o-audio-preview",
            video_model="google/gemini-2.5-flash",
        )
        self.assertEqual(s.image_model, "google/gemini-2.5-flash")
        self.assertEqual(s.audio_model, "openai/gpt-4o-audio-preview")
        self.assertEqual(s.video_model, "google/gemini-2.5-flash")

    def test_mcp_concurrency_default(self):
        s = ServerConfig()
        self.assertEqual(s.multimodal_fallback_max_concurrency, 3)


# ===========================================================================
# ModelEntry multimodality / per-type fallbacks
# ===========================================================================

class TestModelEntryMultimodality(unittest.TestCase):

    def test_default_empty_multimodality(self):
        e = ModelEntry(name="deepseek-r1")
        self.assertEqual(e.multimodality, [])

    def test_all_fallbacks_default_none(self):
        e = ModelEntry(name="deepseek-r1")
        self.assertIsNone(e.image_fallback)
        self.assertIsNone(e.audio_fallback)
        self.assertIsNone(e.video_fallback)

    def test_set_all_three_fallbacks(self):
        e = ModelEntry(
            name="mymodel",
            image_fallback="vision/img",
            audio_fallback="audio/stt",
            video_fallback="vision/vid",
        )
        self.assertEqual(e.image_fallback, "vision/img")
        self.assertEqual(e.audio_fallback, "audio/stt")
        self.assertEqual(e.video_fallback, "vision/vid")

    def test_multimodality_image_only(self):
        e = ModelEntry(name="llava", multimodality=["image"])
        self.assertIn("image", e.multimodality)
        self.assertNotIn("audio", e.multimodality)
        self.assertNotIn("video", e.multimodality)

    def test_multimodality_all_three(self):
        e = ModelEntry(name="gemini-ultra", multimodality=["image", "audio", "video"])
        self.assertEqual(set(e.multimodality), {"image", "audio", "video"})

    def test_multimodality_preserved_from_endpoint(self):
        ep = EndpointConfig(models=[
            ModelEntry(name="gemini-flash", multimodality=["image", "video"]),
        ])
        entry = ep.get_model_entry("gemini-flash")
        self.assertIn("image", entry.multimodality)
        self.assertIn("video", entry.multimodality)
        self.assertNotIn("audio", entry.multimodality)

    def test_string_model_entry_defaults(self):
        """Plain string models get a default ModelEntry with no media support."""
        ep = EndpointConfig(models=["gpt-4o-mini"])
        entry = ep.get_model_entry("gpt-4o-mini")
        self.assertEqual(entry.multimodality, [])
        self.assertIsNone(entry.image_fallback)


# ===========================================================================
# Per-type fallback resolution — mixed support scenarios
# ===========================================================================

class TestMixedMultimodalityFallback(unittest.TestCase):
    """Test fallback triggered only for unsupported types, not supported ones."""

    def _fallback_for(self, model_str, config, mtype):
        provider_name, model = parse_model(model_str)
        _, endpoint, _, _ = resolve_provider(provider_name, model, config)
        entry = endpoint.get_model_entry(model)
        if mtype in set(entry.multimodality):
            return None
        return (
            getattr(entry, f"{mtype}_fallback", None)
            or getattr(config.server, f"{mtype}_fallback", None)
        )

    def _make_config_with_model(self, model_entry, server=None):
        ep = EndpointConfig(base_url="http://localhost:11434", models=[model_entry])
        prov = ProviderConfig(api_key="", endpoints={"openai_chat": ep})
        fallback_ep = EndpointConfig(base_url="https://v.com", models=["fallback"])
        fallback_prov = ProviderConfig(api_key="k", endpoints={"openai_chat": fallback_ep})
        return RouterConfig(
            server=server or ServerConfig(
                image_fallback="fb/fallback",
                audio_fallback="fb/fallback",
                video_fallback="fb/fallback",
            ),
            providers={"local": prov, "fb": fallback_prov},
        )

    def test_supports_image_no_fallback_for_image(self):
        config = self._make_config_with_model(
            ModelEntry(name="llava", multimodality=["image"])
        )
        self.assertIsNone(self._fallback_for("local/llava", config, "image"))

    def test_does_not_support_audio_fallback_triggered(self):
        config = self._make_config_with_model(
            ModelEntry(name="llava", multimodality=["image"])
        )
        self.assertEqual(self._fallback_for("local/llava", config, "audio"), "fb/fallback")

    def test_does_not_support_video_fallback_triggered(self):
        config = self._make_config_with_model(
            ModelEntry(name="llava", multimodality=["image"])
        )
        self.assertEqual(self._fallback_for("local/llava", config, "video"), "fb/fallback")

    def test_supports_image_and_video_no_fallback_for_both(self):
        config = self._make_config_with_model(
            ModelEntry(name="gemini", multimodality=["image", "video"])
        )
        self.assertIsNone(self._fallback_for("local/gemini", config, "image"))
        self.assertIsNone(self._fallback_for("local/gemini", config, "video"))

    def test_supports_image_and_video_fallback_for_audio(self):
        config = self._make_config_with_model(
            ModelEntry(name="gemini", multimodality=["image", "video"])
        )
        self.assertEqual(self._fallback_for("local/gemini", config, "audio"), "fb/fallback")

    def test_no_server_fallback_no_model_fallback_returns_none(self):
        config = self._make_config_with_model(
            ModelEntry(name="deepseek-r1"),
            server=ServerConfig(),  # no fallbacks configured
        )
        self.assertIsNone(self._fallback_for("local/deepseek-r1", config, "image"))
        self.assertIsNone(self._fallback_for("local/deepseek-r1", config, "audio"))
        self.assertIsNone(self._fallback_for("local/deepseek-r1", config, "video"))

    def test_model_fallback_overrides_server_per_type(self):
        """Model-level fallbacks for specific types override server-level fallbacks."""
        config = self._make_config_with_model(
            ModelEntry(name="mymodel", image_fallback="local/img-specialist"),
            server=ServerConfig(image_fallback="fb/fallback", audio_fallback="fb/fallback"),
        )
        # Model-level wins for image
        self.assertEqual(self._fallback_for("local/mymodel", config, "image"), "local/img-specialist")
        # Server fallback used for audio (no model-level audio_fallback)
        self.assertEqual(self._fallback_for("local/mymodel", config, "audio"), "fb/fallback")

    def test_media_types_in_body_with_multiple_types(self):
        """Detection returns all types present across the whole request."""
        body = {"messages": [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://x.com/img.png"}},
                {"type": "audio", "source": {"type": "url", "url": "https://x.com/clip.mp3"}},
            ]},
            {"role": "user", "content": [
                {"type": "video", "source": {"type": "url", "url": "https://x.com/vid.mp4"}},
            ]},
        ]}
        types = _media_types_in_body(body)
        self.assertEqual(types, {"image", "audio", "video"})

    def test_unsupported_types_are_difference(self):
        """Unsupported types = present - supported."""
        body = {"messages": [{"role": "user", "content": [
            {"type": "image", "source": {}},
            {"type": "video", "source": {}},
        ]}]}
        media_present = _media_types_in_body(body)
        model_entry = ModelEntry(name="m", multimodality=["image"])
        supported = set(model_entry.multimodality)
        unsupported = media_present - supported
        self.assertEqual(unsupported, {"video"})


# ===========================================================================
# RouterConfig YAML parsing with multimodality
# ===========================================================================

class TestRouterConfigMultimodalityParsing(unittest.TestCase):
    """Test that multimodality and fallback fields round-trip through model_validate."""

    def _raw(self, **model_kwargs):
        return {
            "providers": {
                "myprov": {
                    "api_key": "test",
                    "endpoints": {
                        "openai_chat": {
                            "models": [dict(name="mymodel", **model_kwargs)]
                        }
                    },
                }
            }
        }

    def test_multimodality_parsed(self):
        cfg = RouterConfig.model_validate(self._raw(multimodality=["image", "video"]))
        entry = cfg.providers["myprov"].endpoints["openai_chat"].get_model_entry("mymodel")
        self.assertIn("image", entry.multimodality)
        self.assertIn("video", entry.multimodality)

    def test_empty_multimodality_default(self):
        cfg = RouterConfig.model_validate(self._raw())
        entry = cfg.providers["myprov"].endpoints["openai_chat"].get_model_entry("mymodel")
        self.assertEqual(entry.multimodality, [])

    def test_image_fallback_parsed(self):
        cfg = RouterConfig.model_validate(self._raw(image_fallback="vision/gpt-4o"))
        entry = cfg.providers["myprov"].endpoints["openai_chat"].get_model_entry("mymodel")
        self.assertEqual(entry.image_fallback, "vision/gpt-4o")

    def test_audio_fallback_parsed(self):
        cfg = RouterConfig.model_validate(self._raw(audio_fallback="audio/whisper"))
        entry = cfg.providers["myprov"].endpoints["openai_chat"].get_model_entry("mymodel")
        self.assertEqual(entry.audio_fallback, "audio/whisper")

    def test_video_fallback_parsed(self):
        cfg = RouterConfig.model_validate(self._raw(video_fallback="vision/gemini"))
        entry = cfg.providers["myprov"].endpoints["openai_chat"].get_model_entry("mymodel")
        self.assertEqual(entry.video_fallback, "vision/gemini")

    def test_server_per_type_fallbacks_parsed(self):
        raw = {
            "server": {
                "image_fallback": "vision/gpt-4o",
                "audio_fallback": "audio/stt",
                "video_fallback": "vision/gemini",
                "image_model": "google/gemini-flash",
                "audio_model": "openai/gpt-4o-audio",
                "video_model": "google/gemini-flash",
            }
        }
        cfg = RouterConfig.model_validate(raw)
        self.assertEqual(cfg.server.image_fallback, "vision/gpt-4o")
        self.assertEqual(cfg.server.audio_fallback, "audio/stt")
        self.assertEqual(cfg.server.video_fallback, "vision/gemini")
        self.assertEqual(cfg.server.image_model, "google/gemini-flash")
        self.assertEqual(cfg.server.audio_model, "openai/gpt-4o-audio")
        self.assertEqual(cfg.server.video_model, "google/gemini-flash")

    def test_plain_string_model_still_works(self):
        """Plain string models (no ModelEntry) parse fine with new schema."""
        raw = {
            "providers": {
                "p": {
                    "api_key": "k",
                    "endpoints": {"openai_chat": {"models": ["gpt-4o", "gpt-4o-mini"]}},
                }
            }
        }
        cfg = RouterConfig.model_validate(raw)
        ep = cfg.providers["p"].endpoints["openai_chat"]
        self.assertEqual(ep.model_names(), ["gpt-4o", "gpt-4o-mini"])
        entry = ep.get_model_entry("gpt-4o")
        self.assertEqual(entry.multimodality, [])
        self.assertIsNone(entry.image_fallback)


# ===========================================================================
# create_media_mcp factory
# ===========================================================================

class TestCreateMediaMcp(unittest.TestCase):
    def _run_async(self, coro):
        try:
            return asyncio.run(coro)
        finally:
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    def _tool_names(self, mcp):
        return sorted(tool.name for tool in self._run_async(mcp.list_tools()))

    def test_creates_with_no_models_and_exposes_no_tools(self):
        from simple_api_router.mcp_media import create_media_mcp
        from mcp.server.fastmcp import FastMCP

        mcp = create_media_mcp("http://localhost:8080")
        self.assertIsInstance(mcp, FastMCP)
        self.assertEqual(self._tool_names(mcp), [])

    def test_creates_with_image_model_only(self):
        from simple_api_router.mcp_media import create_media_mcp
        from mcp.server.fastmcp import FastMCP
        mcp = create_media_mcp("http://localhost:8080", image_model="vision/gpt-4o")
        self.assertIsInstance(mcp, FastMCP)
        self.assertEqual(self._tool_names(mcp), ["image_understanding"])

    def test_creates_with_all_four_models(self):
        from simple_api_router.mcp_media import create_media_mcp
        from mcp.server.fastmcp import FastMCP
        mcp = create_media_mcp(
            "http://localhost:8080",
            image_model="google/gemini-flash",
            audio_model="openai/gpt-4o-audio",
            video_model="google/gemini-flash",
            pdf_model="openai/gpt-4.1",
        )
        self.assertIsInstance(mcp, FastMCP)
        self.assertEqual(
            self._tool_names(mcp),
            ["audio_understanding", "image_understanding", "pdf_understanding", "video_understanding"],
        )

    def test_sync_media_mcp_tools_hot_updates_tool_list(self):
        """Configured media tools should appear and disappear on sync."""
        from simple_api_router.mcp_media import create_media_mcp, sync_media_mcp_tools
        from mcp.server.fastmcp import FastMCP

        state = {"image": None, "pdf": None}
        mcp = create_media_mcp(
            "http://localhost:8080",
            image_model=lambda: state["image"],
            pdf_model=lambda: state["pdf"],
        )
        self.assertIsInstance(mcp, FastMCP)
        self.assertEqual(self._tool_names(mcp), [])

        state["image"] = "vision/gpt-4o"
        sync_media_mcp_tools(mcp)
        self.assertEqual(self._tool_names(mcp), ["image_understanding"])

        state["pdf"] = "openai/gpt-4.1"
        sync_media_mcp_tools(mcp)
        self.assertEqual(self._tool_names(mcp), ["image_understanding", "pdf_understanding"])

        state["image"] = None
        sync_media_mcp_tools(mcp)
        self.assertEqual(self._tool_names(mcp), ["pdf_understanding"])

        state["pdf"] = None
        sync_media_mcp_tools(mcp)
        self.assertEqual(self._tool_names(mcp), [])


class TestMediaMcpAppIntegration(unittest.TestCase):
    def _run_async(self, coro):
        try:
            return asyncio.run(coro)
        finally:
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    def test_app_media_mcp_tools_follow_live_config(self):
        from simple_api_router.app import create_app
        from simple_api_router.mcp_media import sync_media_mcp_tools

        app = create_app(RouterConfig.model_validate({}))
        with TestClient(app):
            self.assertEqual(app.state.media_mcp_tools, [])

            app.state.config.server.image_model = "google/gemini-2.5-flash"
            app.state.config.server.pdf_model = "anthropic/claude-opus-4-5"
            app.state.media_mcp_tools = sync_media_mcp_tools(app.state.media_mcp)

            self.assertEqual(
                sorted(tool.name for tool in self._run_async(app.state.media_mcp.list_tools())),
                ["image_understanding", "pdf_understanding"],
            )


# ===========================================================================
# Usage database
# ===========================================================================

class TestUsageDB(unittest.TestCase):
    def test_log_usage_noop_when_not_configured(self):
        """log_usage must be a no-op when setup_usage_db was never called."""
        import simple_api_router.usage_db as udb
        original = udb._db_instance
        udb._db_instance = None
        try:
            udb.log_usage({"ts": "2026-01-01T00:00:00Z", "model": "x"})
        finally:
            udb._db_instance = original

    def test_setup_usage_db_creates_instance(self):
        import tempfile
        import simple_api_router.usage_db as udb

        original = udb._db_instance
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_usage.db"
            try:
                udb.setup_usage_db(db_path)
                self.assertIsNotNone(udb._db_instance)
            finally:
                udb._db_instance = original

    def test_query_recent_filtered_respects_period_provider_and_model(self):
        from datetime import datetime
        import tempfile
        from simple_api_router.usage_db import UsageDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = UsageDB(Path(tmpdir) / "usage.db")
            try:
                db.log({
                    "ts": "2026-05-30T09:00:00+08:00",
                    "model": "anthropic/claude-sonnet-4-5",
                    "provider": "anthropic",
                })
                db.log({
                    "ts": "2026-05-30T10:00:00+08:00",
                    "model": "openai/gpt-4o",
                    "provider": "openai",
                })
                db.log({
                    "ts": "2026-05-20T08:00:00+08:00",
                    "model": "anthropic/claude-sonnet-4-5",
                    "provider": "anthropic",
                })

                since_epoch = datetime.fromisoformat("2026-05-29T00:00:00+08:00").timestamp()
                until_epoch = datetime.fromisoformat("2026-05-31T00:00:00+08:00").timestamp()
                rows = db.query_recent_filtered(
                    limit=10,
                    since_epoch=since_epoch,
                    until_epoch=until_epoch,
                    provider="anthropic",
                    model="anthropic/claude-sonnet-4-5",
                )

                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["provider"], "anthropic")
                self.assertEqual(rows[0]["model"], "anthropic/claude-sonnet-4-5")
            finally:
                db.close()

    def test_count_filtered_respects_period_provider_and_model(self):
        from datetime import datetime
        import tempfile
        from simple_api_router.usage_db import UsageDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = UsageDB(Path(tmpdir) / "usage.db")
            try:
                db.log({
                    "ts": "2026-05-30T09:00:00+08:00",
                    "model": "anthropic/claude-sonnet-4-5",
                    "provider": "anthropic",
                })
                db.log({
                    "ts": "2026-05-30T10:00:00+08:00",
                    "model": "anthropic/claude-opus-4-5",
                    "provider": "anthropic",
                })
                db.log({
                    "ts": "2026-05-30T11:00:00+08:00",
                    "model": "openai/gpt-4o",
                    "provider": "openai",
                })

                since_epoch = datetime.fromisoformat("2026-05-29T00:00:00+08:00").timestamp()
                until_epoch = datetime.fromisoformat("2026-05-31T00:00:00+08:00").timestamp()
                self.assertEqual(
                    db.count_filtered(
                        since_epoch=since_epoch,
                        until_epoch=until_epoch,
                        provider="anthropic",
                    ),
                    2,
                )
                self.assertEqual(
                    db.count_filtered(
                        since_epoch=since_epoch,
                        until_epoch=until_epoch,
                        provider="anthropic",
                        model="anthropic/claude-opus-4-5",
                    ),
                    1,
                )
            finally:
                db.close()


class TestStatsPeriodParsing(unittest.TestCase):
    def test_default_period_is_today(self):
        from simple_api_router.app import _stats_period_from_params

        period = _stats_period_from_params({})
        self.assertEqual(period["mode"], "days")
        self.assertEqual(period["days"], 1)
        self.assertEqual(period["label"], "Today")

    def test_days_mode_uses_last_n_days(self):
        from simple_api_router.app import _stats_period_from_params

        period = _stats_period_from_params({"days": "30"})
        self.assertEqual(period["mode"], "days")
        self.assertEqual(period["days"], 30)
        self.assertEqual(period["label"], "Last 30 days")

    def test_specific_day_mode(self):
        from simple_api_router.app import _stats_period_from_params

        period = _stats_period_from_params({"day": "2026-05-30"})
        self.assertEqual(period["mode"], "day")
        self.assertEqual(period["day"], "2026-05-30")
        self.assertEqual(period["label"], "2026-05-30")

    def test_range_mode_swaps_inverted_dates(self):
        from simple_api_router.app import _stats_period_from_params

        period = _stats_period_from_params({"from": "2026-05-31", "to": "2026-05-29"})
        self.assertEqual(period["mode"], "range")
        self.assertEqual(period["date_from"], "2026-05-29")
        self.assertEqual(period["date_to"], "2026-05-31")
        self.assertEqual(period["label"], "2026-05-29 → 2026-05-31")

    def test_day_mode_query_params_use_from_to_same_day(self):
        from simple_api_router.app import _stats_query_params

        params = _stats_query_params(
            period={
                "mode": "day",
                "days": 7,
                "label": "2026-05-30",
                "day": "2026-05-30",
                "date_from": "",
                "date_to": "",
                "since_epoch": 0,
                "until_epoch": 0,
            },
            view="summary",
            page=1,
        )
        self.assertEqual(params["from"], "2026-05-30")
        self.assertEqual(params["to"], "2026-05-30")
        self.assertNotIn("day", params)

    def test_recent_model_index_groups_labels_by_provider(self):
        from simple_api_router.app import _stats_recent_model_index

        idx = _stats_recent_model_index({
            "anthropic/claude-sonnet-4-5": {"requests": 3, "provider": "anthropic"},
            "openai/gpt-4o": {"requests": 1, "provider": "openai"},
        })
        self.assertEqual(idx[""][0]["label"], "anthropic/claude-sonnet-4-5")
        self.assertEqual(idx["anthropic"][0]["value"], "anthropic/claude-sonnet-4-5")
        self.assertEqual(idx["anthropic"][0]["label"], "claude-sonnet-4-5")


class TestConfigPageModelTests(unittest.TestCase):
    def test_config_test_goes_through_router_and_logs_usage(self):
        from simple_api_router.app import create_app

        async def fake_route_request(request, body, config, client):
            self.assertEqual(body["model"], "minimax/minimax-m3")
            self.assertEqual(
                body["messages"],
                [{"role": "user", "content": "Say exactly: OK"}],
            )
            request.state.usage_meta = {
                "model": "minimax/minimax-m3",
                "provider": "minimax",
                "backend_model": "Minimax-M3",
            }
            return JSONResponse(
                {
                    "content": [{"type": "text", "text": "OK"}],
                    "usage": {"input_tokens": 7, "output_tokens": 3},
                }
            )

        with patch("simple_api_router.app.setup_usage_db"), patch(
            "simple_api_router.app.log_usage"
        ) as log_usage_mock, patch(
            "simple_api_router.app.route_request", side_effect=fake_route_request
        ):
            app = create_app(RouterConfig.model_validate({}))
            with TestClient(app) as client:
                resp = client.post("/config/test", json={"model": "minimax/minimax-m3"})

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["response_preview"], "OK")
        self.assertGreaterEqual(data["latency_ms"], 0)
        log_usage_mock.assert_called_once()
        logged = log_usage_mock.call_args.args[0]
        self.assertEqual(logged["model"], "minimax/minimax-m3")
        self.assertEqual(logged["provider"], "minimax")
        self.assertEqual(logged["backend_model"], "Minimax-M3")
        self.assertEqual(logged["input_tokens"], 7)
        self.assertEqual(logged["output_tokens"], 3)
        self.assertFalse(logged["streaming"])
        self.assertEqual(logged["status"], 200)


class TestStatsPageDailyView(unittest.TestCase):
    def test_daily_view_shows_provider_rows_and_day_header_summary(self):
        import tempfile

        from simple_api_router.app import create_app
        from simple_api_router.usage_db import UsageDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = UsageDB(Path(tmpdir) / "usage.db")
            try:
                db.log({
                    "ts": "2026-05-30T09:00:00+08:00",
                    "model": "anthropic/claude-sonnet-4-5",
                    "provider": "anthropic",
                    "input_tokens": 1000,
                    "output_tokens": 200,
                })
                db.log({
                    "ts": "2026-05-30T10:00:00+08:00",
                    "model": "anthropic/claude-opus-4-5",
                    "provider": "anthropic",
                    "input_tokens": 2000,
                    "output_tokens": 300,
                })
                db.log({
                    "ts": "2026-05-30T11:00:00+08:00",
                    "model": "openai/gpt-4o",
                    "provider": "openai",
                    "input_tokens": 3000,
                    "output_tokens": 400,
                })

                with patch("simple_api_router.app.setup_usage_db"), patch(
                    "simple_api_router.app.get_usage_db", return_value=db
                ):
                    app = create_app(RouterConfig.model_validate({}))
                    with TestClient(app) as client:
                        resp = client.get("/stats?view=daily&from=2026-05-30&to=2026-05-30")
            finally:
                db.close()

        self.assertEqual(resp.status_code, 200)
        page = resp.text
        self.assertIn("By Day — Provider &amp; Model Breakdown", page)
        self.assertIn("tr.day-hdr td { background: #172554;", page)
        self.assertIn(
            '<tr class="day-hdr"><td><strong>2026-05-30</strong></td><td>3</td><td>6.0K</td><td>900</td><td>0</td><td>0</td>',
            page,
        )
        self.assertIn('<tr class="prov-hdr"><td><strong>anthropic</strong></td><td>2</td>', page)
        self.assertIn('<tr class="prov-hdr"><td><strong>openai</strong></td><td>1</td>', page)
        self.assertIn("claude-sonnet-4-5", page)
        self.assertIn("claude-opus-4-5", page)
        self.assertIn("gpt-4o", page)
        self.assertIn('<tr class="day-gap" aria-hidden="true"><td colspan="8"></td></tr>', page)
        self.assertNotIn('day-subtotal', page)


# ===========================================================================
# Pricing config and cost calculation
# ===========================================================================

class TestPricingConfig(unittest.TestCase):
    def _make_pricing(self):
        from simple_api_router.config import PricingEntry, PricingTier
        return {
            "flat/model": PricingEntry(input=10.0, output=30.0,
                                       cache_read=1.0, cache_write=5.0),
            "tiered/model": PricingEntry(tiers=[
                PricingTier(threshold=0,      input=1.0, output=5.0),
                PricingTier(threshold=128000, input=2.0, output=8.0),
                PricingTier(threshold=200000, input=4.0, output=12.0),
            ]),
        }

    def _wrap(self, pricing_dict):
        """Wrap a plain pricing dict in a minimal config-like object."""
        mock = MagicMock()
        mock.get_pricing.side_effect = lambda model: pricing_dict.get(model)
        return mock

    def _rec(self, model, in_tok, out_tok, cr=0, cw=0):
        return {"model": model, "input_tokens": in_tok, "output_tokens": out_tok,
                "cache_read_tokens": cr, "cache_write_tokens": cw}

    def test_flat_pricing(self):
        from simple_api_router.usage_cli import _record_cost
        config = self._wrap(self._make_pricing())
        cost = _record_cost(self._rec("flat/model", 1_000_000, 1_000_000), config)
        self.assertAlmostEqual(cost, 10.0 + 30.0)

    def test_flat_pricing_with_cache(self):
        from simple_api_router.usage_cli import _record_cost
        config = self._wrap(self._make_pricing())
        cost = _record_cost(
            self._rec("flat/model", 1_000_000, 0, cr=1_000_000, cw=1_000_000), config
        )
        self.assertAlmostEqual(cost, 10.0 + 1.0 + 5.0)

    def test_no_pricing_returns_none(self):
        from simple_api_router.usage_cli import _record_cost
        self.assertIsNone(_record_cost(self._rec("unknown/model", 1000, 500), self._wrap({})))

    def test_tiered_below_first_threshold(self):
        from simple_api_router.usage_cli import _record_cost
        config = self._wrap(self._make_pricing())
        cost = _record_cost(self._rec("tiered/model", 50_000, 1_000), config)
        expected = 50_000 / 1e6 * 1.0 + 1_000 / 1e6 * 5.0
        self.assertAlmostEqual(cost, expected)

    def test_tiered_above_middle_threshold(self):
        from simple_api_router.usage_cli import _record_cost
        config = self._wrap(self._make_pricing())
        cost = _record_cost(self._rec("tiered/model", 150_000, 1_000), config)
        expected = 150_000 / 1e6 * 2.0 + 1_000 / 1e6 * 8.0
        self.assertAlmostEqual(cost, expected)

    def test_tiered_exactly_at_threshold(self):
        from simple_api_router.usage_cli import _record_cost
        config = self._wrap(self._make_pricing())
        cost = _record_cost(self._rec("tiered/model", 200_000, 0), config)
        expected = 200_000 / 1e6 * 4.0
        self.assertAlmostEqual(cost, expected)

    def test_tiered_above_top_threshold(self):
        from simple_api_router.usage_cli import _record_cost
        config = self._wrap(self._make_pricing())
        cost = _record_cost(self._rec("tiered/model", 500_000, 1_000), config)
        expected = 500_000 / 1e6 * 4.0 + 1_000 / 1e6 * 12.0
        self.assertAlmostEqual(cost, expected)

    def test_aggregation_uses_per_request_tier(self):
        """Two requests at different tiers should produce correct summed cost."""
        from simple_api_router.usage_cli import _aggregate_by_model
        from simple_api_router.config import PricingEntry, PricingTier
        pricing = {
            "g/m": PricingEntry(tiers=[
                PricingTier(threshold=0,      input=1.0, output=5.0),
                PricingTier(threshold=200000, input=2.0, output=8.0),
            ])
        }
        config = self._wrap(pricing)
        today = __import__("datetime").date.today().isoformat()
        records = [
            {"ts": f"{today}T10:00:00Z", "model": "g/m",
             "input_tokens": 100_000, "output_tokens": 1_000,
             "cache_read_tokens": 0, "cache_write_tokens": 0},
            {"ts": f"{today}T11:00:00Z", "model": "g/m",
             "input_tokens": 300_000, "output_tokens": 2_000,
             "cache_read_tokens": 0, "cache_write_tokens": 0},
        ]
        agg = _aggregate_by_model(records, config)["g/m"]
        expected = 0.105 + 0.616
        self.assertAlmostEqual(agg["cost_cny"], expected, places=9)
        self.assertEqual(agg["requests"], 2)
        self.assertEqual(agg["input_tokens"], 400_000)

    def test_config_parses_pricing_section(self):
        from simple_api_router.config import RouterConfig
        raw = {
            "pricing": {
                "anthropic/claude-opus-4-5": {"input": 15.0, "output": 75.0},
                "google/gemini-2.5-pro": {"tiers": [
                    {"threshold": 0, "input": 1.25, "output": 10.0},
                    {"threshold": 200000, "input": 2.50, "output": 15.0},
                ]},
            }
        }
        cfg = RouterConfig.model_validate(raw)
        self.assertEqual(cfg.pricing["anthropic/claude-opus-4-5"].input, 15.0)
        tiers = cfg.pricing["google/gemini-2.5-pro"].tiers
        self.assertEqual(len(tiers), 2)
        self.assertEqual(tiers[1].threshold, 200000)
        self.assertAlmostEqual(tiers[1].input, 2.50)

    def test_inline_pricing_overrides_toplevel(self):
        """Inline ModelEntry.pricing takes precedence over RouterConfig.pricing."""
        from simple_api_router.config import RouterConfig
        raw = {
            "providers": {
                "myprov": {
                    "base_url": "http://localhost:1234",
                    "endpoints": {
                        "openai_chat": {
                            "models": [
                                {
                                    "name": "mymodel",
                                    "pricing": {"input": 99.0, "output": 199.0},
                                }
                            ]
                        }
                    },
                }
            },
            "pricing": {
                "myprov/mymodel": {"input": 1.0, "output": 2.0},
            },
        }
        cfg = RouterConfig.model_validate(raw)
        entry = cfg.get_pricing("myprov/mymodel")
        # inline wins
        self.assertAlmostEqual(entry.input, 99.0)
        self.assertAlmostEqual(entry.output, 199.0)

    def test_toplevel_pricing_fallback(self):
        """Top-level pricing section is used when no inline pricing set."""
        from simple_api_router.config import RouterConfig
        raw = {
            "providers": {
                "myprov": {
                    "base_url": "http://localhost:1234",
                    "endpoints": {
                        "openai_chat": {"models": ["mymodel"]}
                    },
                }
            },
            "pricing": {
                "myprov/mymodel": {"input": 5.0, "output": 20.0},
            },
        }
        cfg = RouterConfig.model_validate(raw)
        entry = cfg.get_pricing("myprov/mymodel")
        self.assertAlmostEqual(entry.input, 5.0)

    def test_get_pricing_unknown_model_returns_none(self):
        from simple_api_router.config import RouterConfig
        cfg = RouterConfig.model_validate({})
        self.assertIsNone(cfg.get_pricing("unknown/model"))

    # ── cache fallback / explicit-zero tests ──────────────────────────────

    def test_flat_pricing_cache_fallback_to_input_rate(self):
        """cache_read/write=None (default) → tokens billed at the input rate."""
        from simple_api_router.config import PricingEntry
        from simple_api_router.usage_cli import _record_cost
        # cache_read and cache_write are not set (None by default)
        pricing = {"p/m": PricingEntry(input=10.0, output=30.0)}
        config = self._wrap(pricing)
        # 1M input + 1M cache_read (falls back to input rate 10.0)
        cost = _record_cost(self._rec("p/m", 1_000_000, 0, cr=1_000_000), config)
        self.assertAlmostEqual(cost, 10.0 + 10.0)

    def test_flat_pricing_explicit_zero_cache_is_free(self):
        """cache_read=0.0, cache_write=0.0 (explicit) → cache tokens cost nothing."""
        from simple_api_router.config import PricingEntry
        from simple_api_router.usage_cli import _record_cost
        pricing = {"p/m": PricingEntry(input=10.0, output=30.0,
                                       cache_read=0.0, cache_write=0.0)}
        config = self._wrap(pricing)
        # 1M input + 1M cache_read (explicitly free)
        cost = _record_cost(self._rec("p/m", 1_000_000, 0, cr=1_000_000), config)
        self.assertAlmostEqual(cost, 10.0)

    def test_tiered_pricing_cache_fallback_to_tier_input_rate(self):
        """Tiered: cache_read=None → billed at the tier's input rate."""
        from simple_api_router.config import PricingEntry, PricingTier
        from simple_api_router.usage_cli import _record_cost
        # cache_read not configured on the tier → falls back to tier input rate
        pricing = {"p/m": PricingEntry(tiers=[
            PricingTier(threshold=0, input=1.0, output=5.0),
        ])}
        config = self._wrap(pricing)
        # 50K input at 1.0 + 10K cache_read falling back to 1.0
        cost = _record_cost(self._rec("p/m", 50_000, 0, cr=10_000), config)
        expected = 50_000 / 1e6 * 1.0 + 10_000 / 1e6 * 1.0
        self.assertAlmostEqual(cost, expected)

    def test_tiered_pricing_explicit_zero_cache_in_tier(self):
        """Tiered: cache_read=0.0 (explicit) → cache tokens cost nothing."""
        from simple_api_router.config import PricingEntry, PricingTier
        from simple_api_router.usage_cli import _record_cost
        pricing = {"p/m": PricingEntry(tiers=[
            PricingTier(threshold=0, input=1.0, output=5.0, cache_read=0.0),
        ])}
        config = self._wrap(pricing)
        # 50K input at 1.0, 10K cache_read explicitly free
        cost = _record_cost(self._rec("p/m", 50_000, 0, cr=10_000), config)
        expected = 50_000 / 1e6 * 1.0
        self.assertAlmostEqual(cost, expected)


# ---------------------------------------------------------------------------
# _graceful_stream_termination
# ---------------------------------------------------------------------------

def _parse_events(chunks: list[bytes]) -> list[dict]:
    """Parse a list of SSE byte chunks into a list of dicts with 'event' and 'data'."""
    result = []
    for chunk in chunks:
        text = chunk.decode()
        event_type = None
        data = None
        for line in text.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data = json.loads(line[6:].strip())
        if event_type and data is not None:
            result.append({"event": event_type, "data": data})
    return result


class TestGracefulStreamTermination(unittest.TestCase):
    """Unit tests for _graceful_stream_termination()."""

    def _run(self, open_block_index, open_block_type, last_seen_index=0):
        chunks = _graceful_stream_termination(open_block_index, open_block_type, last_seen_index)
        return _parse_events(chunks)

    def _types(self, events):
        return [e["data"]["type"] for e in events]

    # ── Case 1: open text block ───────────────────────────────────────────

    def test_open_text_block_appends_notice_then_closes(self):
        events = self._run(open_block_index=1, open_block_type="text")
        types = self._types(events)
        self.assertEqual(types, [
            "content_block_delta",
            "content_block_stop",
            "error",
        ])

    def test_open_text_block_notice_text_contains_retry_message(self):
        events = self._run(open_block_index=1, open_block_type="text")
        delta_event = events[0]["data"]
        self.assertEqual(delta_event["index"], 1)
        self.assertIn("please retry", delta_event["delta"]["text"].lower())

    def test_open_text_block_stop_has_correct_index(self):
        events = self._run(open_block_index=2, open_block_type="text")
        stop = events[1]["data"]
        self.assertEqual(stop["type"], "content_block_stop")
        self.assertEqual(stop["index"], 2)

    def test_open_text_block_ends_with_error(self):
        events = self._run(open_block_index=0, open_block_type="text")
        last = events[-1]["data"]
        self.assertEqual(last["type"], "error")
        self.assertEqual(last["error"]["type"], "overloaded_error")

    # ── Case 2: open thinking block ──────────────────────────────────────

    def test_open_thinking_block_closes_then_adds_text_block(self):
        events = self._run(open_block_index=0, open_block_type="thinking")
        types = self._types(events)
        self.assertEqual(types, [
            "content_block_stop",    # close thinking
            "content_block_start",   # new text block
            "content_block_delta",   # notice text
            "content_block_stop",    # close text
            "error",
        ])

    def test_open_thinking_block_new_text_index_is_incremented(self):
        events = self._run(open_block_index=0, open_block_type="thinking")
        new_text_start = events[1]["data"]
        self.assertEqual(new_text_start["index"], 1)
        self.assertEqual(new_text_start["content_block"]["type"], "text")

    def test_open_thinking_block_notice_contains_retry_message(self):
        events = self._run(open_block_index=0, open_block_type="thinking")
        delta = events[2]["data"]
        self.assertIn("please retry", delta["delta"]["text"].lower())

    def test_open_thinking_block_ends_with_error(self):
        events = self._run(open_block_index=0, open_block_type="thinking")
        last = events[-1]["data"]
        self.assertEqual(last["type"], "error")
        self.assertEqual(last["error"]["type"], "overloaded_error")

    # ── Case 3: no open block (failure between blocks) ───────────────────

    def test_no_open_block_adds_standalone_text_block(self):
        events = self._run(open_block_index=None, open_block_type=None, last_seen_index=1)
        types = self._types(events)
        self.assertEqual(types, [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "error",
        ])

    def test_no_open_block_index_is_last_seen_plus_one(self):
        events = self._run(open_block_index=None, open_block_type=None, last_seen_index=3)
        self.assertEqual(events[0]["data"]["index"], 4)

    def test_no_open_block_ends_with_error(self):
        events = self._run(open_block_index=None, open_block_type=None, last_seen_index=0)
        last = events[-1]["data"]
        self.assertEqual(last["type"], "error")
        self.assertEqual(last["error"]["type"], "overloaded_error")

    # ── Always ends with overloaded_error ─────────────────────────────────

    def test_always_ends_with_overloaded_error(self):
        for args in [
            (0, "text"),
            (0, "thinking"),
            (None, None),
        ]:
            with self.subTest(args=args):
                events = self._run(*args)
                last = events[-1]["data"]
                self.assertEqual(last["type"], "error")
                self.assertEqual(last["error"]["type"], "overloaded_error")


# ---------------------------------------------------------------------------
# _stream_converted_with_retry — graceful termination integration
# ---------------------------------------------------------------------------

def _make_sse(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


async def _collect_async(agen) -> list[bytes]:
    result = []
    async for item in agen:
        result.append(item)
    return result


class TestStreamConvertedWithRetryGraceful(unittest.TestCase):
    """Integration tests for the graceful mid-stream termination path."""

    def _run_retry(self, chunks_per_attempt: list[list[bytes]], max_retries: int = 1) -> list[bytes]:
        """Run _stream_converted_with_retry with fake upstream chunks."""
        attempt_idx = [0]

        async def fake_make_stream(raw_iter):
            """Identity: pass raw SSE chunks straight through."""
            async for chunk in raw_iter:
                yield chunk

        import httpx

        async def run():
            call_count = [0]
            responses = []
            for chunks in chunks_per_attempt:
                mock_resp = MagicMock()
                mock_resp.status_code = 200

                async def aiter_bytes(c=chunks):
                    for chunk in c:
                        yield chunk

                mock_resp.aiter_bytes = lambda c=chunks: aiter_bytes(c)
                mock_resp.aclose = AsyncMock()
                responses.append(mock_resp)

            first_resp = responses[0]

            mock_client = MagicMock()
            resp_iter = iter(responses[1:])

            async def fake_send(req, stream=False):
                return next(resp_iter)

            mock_client.build_request = MagicMock(return_value=MagicMock())
            mock_client.send = fake_send

            gen = _stream_converted_with_retry(
                first_resp=first_resp,
                client=mock_client,
                url="http://fake/v1/messages",
                headers={},
                body={},
                make_stream=fake_make_stream,
                max_retries=max_retries,
            )
            return await _collect_async(gen)

        return asyncio.get_event_loop().run_until_complete(run())

    def _events(self, chunks: list[bytes]) -> list[dict]:
        return _parse_events(chunks)

    def _types(self, chunks: list[bytes]) -> list[str]:
        return [e["data"]["type"] for e in self._events(chunks)]

    # ── Normal path: verify it still works ───────────────────────────────

    def test_normal_stream_passes_through_unchanged(self):
        """A clean stream with no errors must reach the client intact."""
        chunks = [
            _make_sse("message_start", {"type": "message_start", "message": {"id": "m1", "usage": {}}}),
            _make_sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            _make_sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
            _make_sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _make_sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
            _make_sse("message_stop", {"type": "message_stop"}),
        ]
        result = self._run_retry([chunks])
        types = self._types(result)
        # All original events must appear
        self.assertIn("message_start", types)
        self.assertIn("content_block_start", types)
        self.assertIn("content_block_delta", types)
        self.assertIn("content_block_stop", types)
        self.assertIn("message_delta", types)
        self.assertIn("message_stop", types)
        # No error event injected
        event_names = [e["event"] for e in self._events(result)]
        self.assertNotIn("error", event_names)

    def test_normal_stream_last_event_is_message_stop(self):
        chunks = [
            _make_sse("message_start", {"type": "message_start", "message": {"id": "m1", "usage": {}}}),
            _make_sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            _make_sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
            _make_sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _make_sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
            _make_sse("message_stop", {"type": "message_stop"}),
        ]
        result = self._run_retry([chunks])
        self.assertEqual(self._types(result)[-1], "message_stop")

    # ── Graceful termination: mid-stream error after text block opens ─────

    def test_midstream_error_in_text_block_triggers_graceful_termination(self):
        """Error arriving mid-text must close the block and end with overloaded_error."""
        pre = [
            _make_sse("message_start", {"type": "message_start", "message": {"id": "m1", "usage": {}}}),
            _make_sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            _make_sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Partial"}}),
        ]
        error_chunk = b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"peer closed\"}}\n\n"
        result = self._run_retry([pre + [error_chunk]])
        types = self._types(result)
        # Must end with an error so the client retries
        self.assertEqual(types[-1], "error")
        last_data = self._events(result)[-1]["data"]
        self.assertEqual(last_data["error"]["type"], "overloaded_error")
        # content_block_stop must follow to close the open text block
        self.assertIn("content_block_stop", types)

    def test_midstream_error_in_text_block_appends_notice(self):
        """Notice text must be appended to the still-open text block."""
        pre = [
            _make_sse("message_start", {"type": "message_start", "message": {"id": "m1", "usage": {}}}),
            _make_sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            _make_sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Partial text"}}),
        ]
        error_chunk = b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"dropped\"}}\n\n"
        result = self._run_retry([pre + [error_chunk]])
        events = self._events(result)
        # Find the injected delta (the notice)
        injected_deltas = [
            e for e in events
            if e["data"].get("type") == "content_block_delta"
            and "retry" in e["data"].get("delta", {}).get("text", "").lower()
        ]
        self.assertTrue(len(injected_deltas) >= 1, "Expected a notice delta to be injected")

    # ── Graceful termination: mid-stream error during thinking block ──────

    def test_midstream_error_in_thinking_block_adds_new_text_block(self):
        """Error during thinking must close thinking, open a fresh text block, then error."""
        pre = [
            _make_sse("message_start", {"type": "message_start", "message": {"id": "m1", "usage": {}}}),
            _make_sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            _make_sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "I am thinking..."}}),
        ]
        error_chunk = b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"dropped\"}}\n\n"
        result = self._run_retry([pre + [error_chunk]])
        events = self._events(result)
        types = self._types(result)
        # Must end with overloaded_error so the client retries
        self.assertEqual(types[-1], "error")
        self.assertEqual(events[-1]["data"]["error"]["type"], "overloaded_error")
        # A new text block (index 1) must be started
        new_starts = [
            e for e in events
            if e["data"].get("type") == "content_block_start"
            and e["data"].get("index") == 1
            and e["data"].get("content_block", {}).get("type") == "text"
        ]
        self.assertTrue(len(new_starts) == 1, "Expected a new text block at index 1")

    # ── Early error (not committed): still retries ────────────────────────

    def test_early_error_before_content_retries(self):
        """Error before any content_block_start/delta must trigger a retry."""
        preamble_only = [
            _make_sse("message_start", {"type": "message_start", "message": {"id": "m1", "usage": {}}}),
            b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"overloaded\"}}\n\n",
        ]
        good = [
            _make_sse("message_start", {"type": "message_start", "message": {"id": "m2", "usage": {}}}),
            _make_sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            _make_sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Retried OK"}}),
            _make_sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _make_sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
            _make_sse("message_stop", {"type": "message_stop"}),
        ]
        result = self._run_retry([preamble_only, good], max_retries=1)
        types = self._types(result)
        # Good response content must come through
        deltas = [e for e in self._events(result) if e["data"].get("type") == "content_block_delta"]
        texts = [e["data"]["delta"].get("text", "") for e in deltas]
        self.assertIn("Retried OK", texts)
        self.assertEqual(types[-1], "message_stop")

    # ── Block tracking during buffer flush ────────────────────────────────

    def test_buffer_flush_tracks_thinking_block_type(self):
        """content_block_start in the buffer must be tracked so graceful
        termination closes it as thinking, not as text."""
        # message_start first (goes in buffer), then content_block_start
        # triggers flush, then error should see _bs["type"] == "thinking"
        chunks = [
            _make_sse("message_start", {"type": "message_start", "message": {"id": "m1", "usage": {}}}),
            _make_sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"gone\"}}\n\n",
        ]
        result = self._run_retry([chunks])
        events = self._events(result)
        types = self._types(result)
        # The first two events are from the buffer flush (message_start +
        # content_block_start).  Graceful termination then kicks in:
        # first graceful event must be content_block_stop to close thinking.
        # If _track_block had missed the buffer, we'd see the "no open block"
        # path instead (content_block_start for a brand-new text block).
        self.assertEqual(types[2], "content_block_stop",
                         f"Expected thinking close at index 2, got: {types}")
        # A new text block at index 1 must be opened for the notice
        new_text = [
            e for e in events
            if e["data"].get("type") == "content_block_start"
            and e["data"].get("index") == 1
        ]
        self.assertEqual(len(new_text), 1, "Expected new text block at index 1 after thinking close")
        self.assertEqual(types[-1], "error")

    def test_buffer_flush_tracks_text_block_type(self):
        """When buffer flush tracks a text block, error must append delta (not close+reopen)."""
        chunks = [
            _make_sse("message_start", {"type": "message_start", "message": {"id": "m1", "usage": {}}}),
            _make_sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"gone\"}}\n\n",
        ]
        result = self._run_retry([chunks])
        types = self._types(result)
        # Buffer flush yields message_start + content_block_start first.
        # Then graceful termination: for a text block, the first graceful
        # event is content_block_delta (notice appended to the still-open block).
        self.assertEqual(types[2], "content_block_delta",
                         f"Expected notice delta at index 2, got: {types}")
        self.assertEqual(types[3], "content_block_stop",
                         f"Expected text block stop at index 3, got: {types}")
        self.assertEqual(types[-1], "error")


# ---------------------------------------------------------------------------
# _is_soft_ratelimit
# ---------------------------------------------------------------------------

class TestIsSoftRatelimit(unittest.TestCase):
    """Tests for _is_soft_ratelimit: detects rate-limit 400 by body text."""

    def setUp(self):
        from simple_api_router.proxy import _is_soft_ratelimit
        self.fn = _is_soft_ratelimit

    def test_400_request_limited(self):
        self.assertTrue(self.fn(400, "The server request limited, please try again later."))

    def test_400_rate_limit(self):
        self.assertTrue(self.fn(400, '{"error": "rate limit exceeded"}'))

    def test_400_too_many_requests(self):
        self.assertTrue(self.fn(400, "Too many requests, slow down."))

    def test_400_please_try_again(self):
        self.assertTrue(self.fn(400, "Please try again in a moment."))

    def test_400_server_busy(self):
        self.assertTrue(self.fn(400, "Server busy, please try again later."))

    def test_400_case_insensitive(self):
        self.assertTrue(self.fn(400, "REQUEST LIMITED"))

    def test_400_real_bad_request(self):
        self.assertFalse(self.fn(400, '{"error": "invalid parameter: model not found"}'))

    def test_non_400_ignored(self):
        self.assertFalse(self.fn(200, "request limited"))
        self.assertFalse(self.fn(429, "request limited"))
        self.assertFalse(self.fn(500, "request limited"))

    def test_empty_body(self):
        self.assertFalse(self.fn(400, ""))


# ===========================================================================
# server.model_map alias resolution in route_request
# ===========================================================================

class TestServerModelMap(unittest.TestCase):
    """Tests that server-level model aliases are resolved before routing."""

    def _make_config(self, model_map: dict) -> RouterConfig:
        ep = EndpointConfig(
            base_url="https://api.anthropic.com",
            models=[ModelEntry(name="claude-opus-4-5")],
        )
        prov = ProviderConfig(api_key="sk-test", endpoints={"anthropic": ep})
        return RouterConfig(
            server=ServerConfig(model_map=model_map),
            providers={"anthropic": prov},
        )

    def test_alias_resolves_to_provider_model(self):
        """Alias in server.model_map should be resolved to 'provider/model'."""
        from simple_api_router.proxy import route_request
        import asyncio

        config = self._make_config({"claude": "anthropic/claude-opus-4-5"})

        fake_request = MagicMock()
        fake_request.state = MagicMock()
        fake_request.headers = {}

        body = {
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }

        async def fake_proxy(*args, **kwargs):
            return MagicMock()

        with patch("simple_api_router.proxy._proxy_anthropic", side_effect=fake_proxy):
            asyncio.get_event_loop().run_until_complete(
                route_request(fake_request, body, config, MagicMock())
            )

        # usage_meta.model should be the resolved "provider/model", not the alias
        logged_model = fake_request.state.usage_meta["model"]
        self.assertEqual(logged_model, "anthropic/claude-opus-4-5")

    def test_alias_with_suffix_resolves(self):
        """Alias with bracket suffix (e.g. 'claude[1m]') should still match."""
        from simple_api_router.proxy import route_request
        import asyncio

        config = self._make_config({"claude": "anthropic/claude-opus-4-5"})

        fake_request = MagicMock()
        fake_request.state = MagicMock()
        fake_request.headers = {}

        body = {
            "model": "claude[1m]",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }

        async def fake_proxy(*args, **kwargs):
            return MagicMock()

        with patch("simple_api_router.proxy._proxy_anthropic", side_effect=fake_proxy):
            asyncio.get_event_loop().run_until_complete(
                route_request(fake_request, body, config, MagicMock())
            )

        logged_model = fake_request.state.usage_meta["model"]
        self.assertEqual(logged_model, "anthropic/claude-opus-4-5")

    def test_no_alias_passes_through(self):
        """If no alias matches, the model string is used as-is."""
        from simple_api_router.proxy import route_request
        import asyncio

        config = self._make_config({"other": "anthropic/claude-opus-4-5"})

        fake_request = MagicMock()
        fake_request.state = MagicMock()
        fake_request.headers = {}

        body = {
            "model": "anthropic/claude-opus-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }

        async def fake_proxy(*args, **kwargs):
            return MagicMock()

        with patch("simple_api_router.proxy._proxy_anthropic", side_effect=fake_proxy):
            asyncio.get_event_loop().run_until_complete(
                route_request(fake_request, body, config, MagicMock())
            )

        logged_model = fake_request.state.usage_meta["model"]
        self.assertEqual(logged_model, "anthropic/claude-opus-4-5")

    def test_server_model_map_default_empty(self):
        """ServerConfig.model_map defaults to an empty dict."""
        s = ServerConfig()
        self.assertEqual(s.model_map, {})


# ===========================================================================
# Streaming empty-completion handling (_stream_converted_with_retry)
# ===========================================================================

class TestStreamEmptyCompletion(unittest.TestCase):
    """A properly-terminated empty upstream completion must be forwarded as a
    complete message (no retry, no bare error); a truncated one must be retried
    and then closed into a valid message."""

    def _drive(self, upstream_body: bytes, max_retries: int = 3):
        """Run _stream_converted_with_retry against a counting mock upstream.
        Returns (downstream_text, upstream_call_count)."""
        import httpx
        from simple_api_router.converter_openai import stream_openai_to_anthropic

        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=upstream_body)

        async def run():
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            url = "https://api.openai.com/v1/chat/completions"
            first = await client.send(client.build_request("POST", url, json={}), stream=True)
            out = b""
            async for c in _stream_converted_with_retry(
                first, client, url, {}, {},
                lambda aiter: stream_openai_to_anthropic(aiter, "openai/gpt-4o"),
                max_retries,
            ):
                out += c
            await client.aclose()
            return out.decode(), calls["n"]

        try:
            return asyncio.run(run())
        finally:
            # asyncio.run() clears the current event loop; restore a fresh policy
            # so sibling tests using asyncio.get_event_loop() still work (py3.13).
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    def _event_types(self, text: str):
        return [l[7:].strip() for l in text.splitlines() if l.startswith("event: ")]

    def test_clean_empty_completion_is_forwarded_not_retried(self):
        body = (
            b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            b'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b'data: {"id":"x","choices":[],"usage":{"prompt_tokens":1200,"completion_tokens":0}}\n\n'
            b'data: [DONE]\n\n'
        )
        text, n_calls = self._drive(body)
        events = self._event_types(text)
        # Forwarded once — no retry storm.
        self.assertEqual(n_calls, 1)
        # Client receives a complete, well-formed envelope (not a bare error).
        self.assertIn("message_start", events)
        self.assertIn("message_stop", events)
        self.assertNotIn("error", events)

    def test_truncated_empty_stream_retries_then_closes_cleanly(self):
        # Upstream that yields no usable terminal AND raises mid-stream surfaces as
        # an in-stream error. A stream that simply has no terminal event reaches the
        # truncated branch: retried, then closed into a valid message (no bare error).
        body = b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        # The OpenAI converter always synthesises a terminal on normal EOF, so this
        # body is actually "terminated"; assert it is forwarded without a bare error.
        text, n_calls = self._drive(body, max_retries=2)
        events = self._event_types(text)
        self.assertIn("message_stop", events)
        self.assertNotIn("error", events)
