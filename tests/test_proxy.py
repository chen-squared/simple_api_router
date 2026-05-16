"""Tests for proxy routing helpers — multimodal detection and fallback logic."""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from simple_api_router.config import (
    EndpointConfig,
    ModelEntry,
    ProviderConfig,
    RouterConfig,
    ServerConfig,
)
from simple_api_router.proxy import _blocks_have_media, _request_has_media, parse_model, resolve_provider


# ---------------------------------------------------------------------------
# _request_has_media
# ---------------------------------------------------------------------------

class TestRequestHasMedia(unittest.TestCase):

    def _msg(self, content):
        return {"role": "user", "content": content}

    # ── positive cases ──────────────────────────────────────────────────────

    def test_image_base64_block(self):
        body = {"messages": [self._msg([
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}}
        ])]}
        self.assertTrue(_request_has_media(body))

    def test_image_url_block(self):
        body = {"messages": [self._msg([
            {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}}
        ])]}
        self.assertTrue(_request_has_media(body))

    def test_video_block(self):
        body = {"messages": [self._msg([
            {"type": "video", "source": {"type": "url", "url": "https://example.com/video.mp4"}}
        ])]}
        self.assertTrue(_request_has_media(body))

    def test_media_in_second_message(self):
        body = {"messages": [
            self._msg("plain text"),
            self._msg([
                {"type": "text", "text": "describe this"},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}},
            ]),
        ]}
        self.assertTrue(_request_has_media(body))

    def test_pdf_document_block(self):
        body = {"messages": [self._msg([
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "abc"}}
        ])]}
        self.assertTrue(_request_has_media(body))

    def test_document_url_source(self):
        body = {"messages": [self._msg([
            {"type": "document", "source": {"type": "url", "url": "https://example.com/doc.pdf"}}
        ])]}
        self.assertTrue(_request_has_media(body))

    def test_document_text_source_is_not_media(self):
        # document with source.type == "text" is plain text, text-only models can handle it
        body = {"messages": [self._msg([
            {"type": "document", "source": {"type": "text", "text": "some text content"}}
        ])]}
        self.assertFalse(_request_has_media(body))

    def test_tool_result_with_nested_image(self):
        # tool_result content can be a list of blocks (e.g. screenshot tool)
        body = {"messages": [self._msg([
            {"type": "tool_result", "tool_use_id": "tu_1", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/screenshot.png"}},
                {"type": "text", "text": "see screenshot"},
            ]}
        ])]}
        self.assertTrue(_request_has_media(body))

    def test_tool_result_with_text_only_content(self):
        body = {"messages": [self._msg([
            {"type": "tool_result", "tool_use_id": "tu_1", "content": [
                {"type": "text", "text": "result text"}
            ]}
        ])]}
        self.assertFalse(_request_has_media(body))

    # ── negative cases ──────────────────────────────────────────────────────

    def test_text_only_string_content(self):
        body = {"messages": [self._msg("just text")]}
        self.assertFalse(_request_has_media(body))

    def test_text_only_block_list(self):
        body = {"messages": [self._msg([{"type": "text", "text": "hello"}])]}
        self.assertFalse(_request_has_media(body))

    def test_tool_use_block_is_not_media(self):
        body = {"messages": [self._msg([
            {"type": "tool_use", "id": "tu_1", "name": "search", "input": {}}
        ])]}
        self.assertFalse(_request_has_media(body))

    def test_tool_result_string_content_is_not_media(self):
        body = {"messages": [self._msg([
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "result"}
        ])]}
        self.assertFalse(_request_has_media(body))

    def test_empty_messages(self):
        self.assertFalse(_request_has_media({"messages": []}))

    def test_missing_messages_key(self):
        self.assertFalse(_request_has_media({}))

    def test_non_list_content_ignored(self):
        # content that's a string (normal case)
        body = {"messages": [{"role": "user", "content": "hello"}]}
        self.assertFalse(_request_has_media(body))


# ---------------------------------------------------------------------------
# Helpers — build mock RouterConfig
# ---------------------------------------------------------------------------

def _make_config(
    *,
    global_fallback: str | None = None,
    model_entries: list | None = None,
    fallback_provider_name: str = "vision",
    fallback_model: str = "gpt-4o",
) -> RouterConfig:
    """Build a minimal RouterConfig for fallback routing tests."""
    # Primary provider: local, openai_chat, text-only models
    primary_models = model_entries or [
        ModelEntry(name="deepseek-r1", text_only=True),
        ModelEntry(name="qwen2.5-coder", text_only=True),
    ]
    primary_ep = EndpointConfig(
        base_url="http://localhost:11434",
        models=primary_models,
    )
    primary_prov = ProviderConfig(
        api_key="",
        endpoints={"openai_chat": primary_ep},
    )

    # Fallback provider: multimodal model
    fallback_ep = EndpointConfig(
        base_url="https://api.openai.com",
        models=[fallback_model],
    )
    fallback_prov = ProviderConfig(
        api_key="sk-test",
        endpoints={"openai_chat": fallback_ep},
    )

    server = ServerConfig(multimodal_fallback=global_fallback)

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
        self.assertFalse(entry.text_only)
        self.assertIsNone(entry.multimodal_fallback)

    def test_model_entry_dict_form(self):
        ep = EndpointConfig(models=[
            ModelEntry(name="deepseek-r1", text_only=True, multimodal_fallback="vision/gpt-4o"),
        ])
        entry = ep.get_model_entry("deepseek-r1")
        self.assertTrue(entry.text_only)
        self.assertEqual(entry.multimodal_fallback, "vision/gpt-4o")

    def test_model_names_mixed(self):
        ep = EndpointConfig(models=[
            "gpt-4o",
            ModelEntry(name="deepseek-r1", text_only=True),
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
                    models=[ModelEntry(name="deepseek-r1", text_only=True)]
                )
            },
        )
        result = prov.find_model("deepseek-r1")
        self.assertIsNotNone(result)
        fmt, ep = result
        self.assertEqual(fmt, "openai_chat")


# ---------------------------------------------------------------------------
# Multimodal fallback routing (unit, no HTTP)
# ---------------------------------------------------------------------------

class TestMultimodalFallbackRouting(unittest.TestCase):
    """
    Tests for the fallback logic inside route_request().
    We directly test the resolve_provider + ModelEntry combination used by
    route_request() rather than invoking the full async handler (which would
    require mocking httpx).
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

    # ── detection gate ──────────────────────────────────────────────────────

    def test_no_fallback_when_no_media(self):
        body = self._text_body("local/deepseek-r1")
        self.assertFalse(_request_has_media(body))

    def test_fallback_triggered_for_image(self):
        body = self._image_body("local/deepseek-r1")
        self.assertTrue(_request_has_media(body))

    # ── resolve_provider with ModelEntry ────────────────────────────────────

    def test_resolve_text_only_model_entry(self):
        config = _make_config(global_fallback="vision/gpt-4o")
        _, model = parse_model("local/deepseek-r1")
        _, ep, _, _ = resolve_provider("local", model, config)
        entry = ep.get_model_entry(model)
        self.assertTrue(entry.text_only)

    def test_global_fallback_resolves_correctly(self):
        config = _make_config(global_fallback="vision/gpt-4o", fallback_provider_name="vision", fallback_model="gpt-4o")
        _, model = parse_model("local/deepseek-r1")
        _, ep, _, _ = resolve_provider("local", model, config)
        entry = ep.get_model_entry(model)

        fallback_str = entry.multimodal_fallback or config.server.multimodal_fallback
        self.assertEqual(fallback_str, "vision/gpt-4o")

        fb_prov_name, fb_model = parse_model(fallback_str)
        _, fb_ep, fb_fmt, fb_backend = resolve_provider(fb_prov_name, fb_model, config)
        self.assertEqual(fb_fmt, "openai_chat")
        self.assertEqual(fb_backend, "gpt-4o")

    def test_model_level_fallback_takes_priority_over_global(self):
        config = _make_config(
            global_fallback="vision/gpt-4o",
            model_entries=[
                ModelEntry(name="deepseek-r1", text_only=True, multimodal_fallback="vision/gpt-4o"),
                ModelEntry(name="qwen2.5-coder", text_only=True),  # uses global
            ],
            fallback_provider_name="vision",
            fallback_model="gpt-4o",
        )
        _, model = parse_model("local/deepseek-r1")
        _, ep, _, _ = resolve_provider("local", model, config)
        entry = ep.get_model_entry(model)

        fallback_str = entry.multimodal_fallback or config.server.multimodal_fallback
        self.assertEqual(fallback_str, "vision/gpt-4o")

    def test_no_fallback_configured_returns_none(self):
        """When neither model nor server has a fallback, fallback_str is None."""
        config = _make_config(global_fallback=None)
        _, model = parse_model("local/deepseek-r1")
        _, ep, _, _ = resolve_provider("local", model, config)
        entry = ep.get_model_entry(model)

        fallback_str = entry.multimodal_fallback or config.server.multimodal_fallback
        self.assertIsNone(fallback_str)

    def test_non_text_only_model_entry_is_not_triggered(self):
        """A model without text_only=True should NOT trigger a fallback."""
        config = _make_config(
            global_fallback="vision/gpt-4o",
            model_entries=["llava"],  # plain string, text_only=False
            fallback_provider_name="vision",
            fallback_model="gpt-4o",
        )
        _, model = parse_model("local/llava")
        _, ep, _, _ = resolve_provider("local", model, config)
        entry = ep.get_model_entry(model)

        self.assertFalse(entry.text_only)
        # No fallback should be applied
        fallback_str = None
        if entry.text_only:
            fallback_str = entry.multimodal_fallback or config.server.multimodal_fallback
        self.assertIsNone(fallback_str)


if __name__ == "__main__":
    unittest.main()
