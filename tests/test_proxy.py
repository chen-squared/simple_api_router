"""Tests for proxy routing helpers — multimodal detection and fallback logic."""
from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from simple_api_router.config import (
    EndpointConfig,
    ModelEntry,
    ProviderConfig,
    RouterConfig,
    ServerConfig,
)
from simple_api_router.proxy import (
    _blocks_have_media,
    _graceful_stream_termination,
    _request_has_media,
    _stream_converted_with_retry,
    _upstream_error_sse,
    parse_model,
    resolve_provider,
)


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
            asyncio.get_event_loop().run_until_complete(route_request(fake_request, body, config, MagicMock()))

        logged_model = fake_request.state.usage_meta["model"]
        self.assertNotIn("[1m]", logged_model, "usage_meta should not contain bracket suffixes")
        self.assertEqual(logged_model, "openai/gpt-4o")


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

    # ── helper: simulate route_request() decision without HTTP ──────────────

    def _decide(self, body: dict, config) -> tuple:
        """
        Run the same fallback-decision logic as route_request() and return
        (api_format, backend_model) that would actually be used.
        """
        from simple_api_router.proxy import parse_model, resolve_provider, _request_has_media
        model_str = body["model"]
        provider_name, model = parse_model(model_str)
        provider, endpoint, api_format, backend_model = resolve_provider(provider_name, model, config)

        if _request_has_media(body):
            entry = endpoint.get_model_entry(model)
            if entry.text_only:
                fallback = entry.multimodal_fallback or config.server.multimodal_fallback
                if fallback:
                    fb_prov_name, fb_model = parse_model(fallback)
                    _, _, api_format, backend_model = resolve_provider(fb_prov_name, fb_model, config)

        return api_format, backend_model

    # ── routing decision: text_only model ────────────────────────────────────

    def test_text_only_model_text_request_stays_on_primary(self):
        """text_only model + no media → primary model used, no fallback."""
        config = _make_config(global_fallback="vision/gpt-4o",
                              fallback_provider_name="vision", fallback_model="gpt-4o")
        body = self._text_body("local/deepseek-r1")
        _, backend = self._decide(body, config)
        # Must stay on deepseek-r1, not be re-routed to gpt-4o
        self.assertEqual(backend, "deepseek-r1")

    def test_text_only_model_image_request_uses_fallback(self):
        """text_only model + image → fallback model used."""
        config = _make_config(global_fallback="vision/gpt-4o",
                              fallback_provider_name="vision", fallback_model="gpt-4o")
        body = self._image_body("local/deepseek-r1")
        _, backend = self._decide(body, config)
        self.assertEqual(backend, "gpt-4o")

    def test_text_only_model_pdf_request_uses_fallback(self):
        """text_only model + PDF document → fallback model used."""
        config = _make_config(global_fallback="vision/gpt-4o",
                              fallback_provider_name="vision", fallback_model="gpt-4o")
        body = {
            "model": "local/deepseek-r1",
            "messages": [{"role": "user", "content": [
                {"type": "document", "source": {"type": "base64",
                                                "media_type": "application/pdf", "data": "abc"}},
                {"type": "text", "text": "summarise this"},
            ]}],
            "max_tokens": 512,
        }
        _, backend = self._decide(body, config)
        self.assertEqual(backend, "gpt-4o")

    def test_text_only_model_text_document_stays_on_primary(self):
        """text_only model + document with text source → NOT media, stays on primary."""
        config = _make_config(global_fallback="vision/gpt-4o",
                              fallback_provider_name="vision", fallback_model="gpt-4o")
        body = {
            "model": "local/deepseek-r1",
            "messages": [{"role": "user", "content": [
                {"type": "document", "source": {"type": "text", "text": "plain text doc"}},
            ]}],
            "max_tokens": 512,
        }
        _, backend = self._decide(body, config)
        self.assertEqual(backend, "deepseek-r1")

    def test_text_only_model_tool_result_image_uses_fallback(self):
        """text_only model + tool_result containing image → fallback model used."""
        config = _make_config(global_fallback="vision/gpt-4o",
                              fallback_provider_name="vision", fallback_model="gpt-4o")
        body = {
            "model": "local/deepseek-r1",
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": [
                    {"type": "image", "source": {"type": "url",
                                                 "url": "https://x.com/screenshot.png"}},
                ]},
            ]}],
            "max_tokens": 512,
        }
        _, backend = self._decide(body, config)
        self.assertEqual(backend, "gpt-4o")

    def test_non_text_only_model_image_stays_on_primary(self):
        """Model without text_only=True receives image → NOT re-routed (let upstream handle)."""
        config = _make_config(
            global_fallback="vision/gpt-4o",
            model_entries=["llava"],
            fallback_provider_name="vision",
            fallback_model="gpt-4o",
        )
        body = self._image_body("local/llava")
        _, backend = self._decide(body, config)
        # llava is multimodal-capable; must NOT be re-routed to gpt-4o
        self.assertEqual(backend, "llava")

    def test_text_only_model_no_fallback_configured_stays_on_primary(self):
        """text_only model + image but no fallback configured → stays on primary (warning path)."""
        config = _make_config(global_fallback=None)
        body = self._image_body("local/deepseek-r1")
        _, backend = self._decide(body, config)
        self.assertEqual(backend, "deepseek-r1")

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
        # Verify model-level fallback wins
        body = self._image_body("local/deepseek-r1")
        _, backend = self._decide(body, config)
        self.assertEqual(backend, "gpt-4o")

    def test_model_level_fallback_independent_of_global(self):
        """Model with its own fallback uses it even when global points elsewhere."""
        vision_ep = EndpointConfig(base_url="https://v.com", models=["vision-model"])
        vision_prov = ProviderConfig(api_key="k", endpoints={"openai_chat": vision_ep})
        other_ep = EndpointConfig(base_url="https://o.com", models=["other-model"])
        other_prov = ProviderConfig(api_key="k", endpoints={"openai_chat": other_ep})
        primary_ep = EndpointConfig(
            base_url="http://localhost:11434",
            models=[ModelEntry(name="deepseek-r1", text_only=True,
                               multimodal_fallback="vision/vision-model")],
        )
        primary_prov = ProviderConfig(api_key="", endpoints={"openai_chat": primary_ep})
        config = RouterConfig(
            server=ServerConfig(multimodal_fallback="other/other-model"),
            providers={"local": primary_prov, "vision": vision_prov, "other": other_prov},
        )
        body = self._image_body("local/deepseek-r1")
        _, backend = self._decide(body, config)
        # model-level fallback "vision/vision-model" wins over global "other/other-model"
        self.assertEqual(backend, "vision-model")

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


if __name__ == "__main__":
    unittest.main()


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
            "message_delta",
            "message_stop",
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

    def test_open_text_block_ends_with_end_turn(self):
        events = self._run(open_block_index=0, open_block_type="text")
        msg_delta = events[-2]["data"]
        self.assertEqual(msg_delta["delta"]["stop_reason"], "end_turn")

    # ── Case 2: open thinking block ──────────────────────────────────────

    def test_open_thinking_block_closes_then_adds_text_block(self):
        events = self._run(open_block_index=0, open_block_type="thinking")
        types = self._types(events)
        self.assertEqual(types, [
            "content_block_stop",    # close thinking
            "content_block_start",   # new text block
            "content_block_delta",   # notice text
            "content_block_stop",    # close text
            "message_delta",
            "message_stop",
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

    def test_open_thinking_block_ends_with_end_turn(self):
        events = self._run(open_block_index=0, open_block_type="thinking")
        self.assertEqual(events[-2]["data"]["delta"]["stop_reason"], "end_turn")

    # ── Case 3: no open block (failure between blocks) ───────────────────

    def test_no_open_block_adds_standalone_text_block(self):
        events = self._run(open_block_index=None, open_block_type=None, last_seen_index=1)
        types = self._types(events)
        self.assertEqual(types, [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ])

    def test_no_open_block_index_is_last_seen_plus_one(self):
        events = self._run(open_block_index=None, open_block_type=None, last_seen_index=3)
        self.assertEqual(events[0]["data"]["index"], 4)

    def test_no_open_block_ends_with_end_turn(self):
        events = self._run(open_block_index=None, open_block_type=None, last_seen_index=0)
        self.assertEqual(events[-2]["data"]["delta"]["stop_reason"], "end_turn")

    # ── Always ends with message_stop ────────────────────────────────────

    def test_always_ends_with_message_stop(self):
        for args in [
            (0, "text"),
            (0, "thinking"),
            (None, None),
        ]:
            with self.subTest(args=args):
                events = self._run(*args)
                self.assertEqual(events[-1]["data"]["type"], "message_stop")


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
        """Error arriving mid-text must close the block and end with message_stop."""
        pre = [
            _make_sse("message_start", {"type": "message_start", "message": {"id": "m1", "usage": {}}}),
            _make_sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            _make_sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Partial"}}),
        ]
        error_chunk = b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"peer closed\"}}\n\n"
        result = self._run_retry([pre + [error_chunk]])
        types = self._types(result)
        # Must end with message_stop
        self.assertEqual(types[-1], "message_stop")
        # message_delta with end_turn must be present
        self.assertIn("message_delta", types)
        # content_block_stop must follow to close the open text block
        self.assertIn("content_block_stop", types)
        # No raw error event forwarded to client
        event_names = [e["event"] for e in self._events(result)]
        self.assertNotIn("error", event_names)

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
        """Error during thinking must close thinking and open a fresh text block."""
        pre = [
            _make_sse("message_start", {"type": "message_start", "message": {"id": "m1", "usage": {}}}),
            _make_sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            _make_sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "I am thinking..."}}),
        ]
        error_chunk = b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"dropped\"}}\n\n"
        result = self._run_retry([pre + [error_chunk]])
        events = self._events(result)
        types = self._types(result)
        # Must end with message_stop, no error forwarded
        self.assertEqual(types[-1], "message_stop")
        event_names = [e["event"] for e in events]
        self.assertNotIn("error", event_names)
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
