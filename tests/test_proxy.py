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
# Usage logger
# ===========================================================================

class TestUsageLogger(unittest.TestCase):
    def test_log_usage_noop_when_not_configured(self):
        """log_usage must be a no-op when setup_usage_logging was never called."""
        import simple_api_router.usage_logger as ul
        original = ul._usage_logger
        ul._usage_logger = None
        try:
            ul.log_usage({"ts": "2026-01-01T00:00:00Z", "model": "x"})
        finally:
            ul._usage_logger = original

    def test_setup_creates_logger(self):
        import tempfile, os
        import simple_api_router.usage_logger as ul
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            ul.setup_usage_logging(log_path)
            self.assertIsNotNone(ul._usage_logger)
            self.assertIsNotNone(ul.get_usage_log_path())
            self.assertIn("router.usage.jsonl", ul.get_usage_log_path())
        finally:
            ul._usage_logger = None
            ul._usage_log_path = None
            os.unlink(log_path)
            jsonl = log_path.replace(".log", "") + "/../router.usage.jsonl"
            try:
                os.unlink(ul.get_usage_log_path() or "")
            except Exception:
                pass


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
