"""Quick connectivity test for a configured model.

Makes a minimal single-turn request directly to the upstream provider
(no router server needed) and returns timing + response preview.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx

from .config import RouterConfig
from .proxy import (
    _build_openai_headers,
    _is_real_key,
    parse_model,
    resolve_provider,
)

_TEST_MESSAGE = [{"role": "user", "content": "Say exactly: OK"}]


async def test_model_direct(
    model_str: str,
    config: RouterConfig,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Test a model by making a minimal request directly to the upstream provider.

    Creates a temporary httpx client if *client* is not provided.
    Returns a dict with keys: success, latency_ms, response_preview, error,
    model, backend_model, api_format.
    """
    if client is not None:
        return await _run_test(model_str, config, client)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        follow_redirects=True,
    ) as _client:
        return await _run_test(model_str, config, _client)


async def _run_test(
    model_str: str,
    config: RouterConfig,
    client: httpx.AsyncClient,
) -> Dict[str, Any]:
    # ── resolve provider ────────────────────────────────────────────────────
    try:
        provider_name, model = parse_model(model_str)
        provider, endpoint, api_format, backend_model = resolve_provider(
            provider_name, model, config
        )
    except Exception as exc:
        return {
            "success": False,
            "error": f"Config error: {exc}",
            "latency_ms": None,
            "model": model_str,
        }

    resolved_model_str = (
        f"{provider_name}/{model}" if provider_name else model
    )

    # ── build request ────────────────────────────────────────────────────────
    base_body: Dict[str, Any] = {
        "model": backend_model,
        "messages": _TEST_MESSAGE,
        "max_tokens": 10,
        "stream": False,
    }

    try:
        start = time.monotonic()

        if api_format == "anthropic":
            headers: Dict[str, str] = {
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            if _is_real_key(provider.api_key):
                headers["x-api-key"] = provider.api_key
                headers["Authorization"] = f"Bearer {provider.api_key}"
            base_url = endpoint.resolve_base_url("anthropic", provider.base_url)
            url = f"{base_url}/v1/messages"
            resp = await client.post(url, headers=headers, json=base_body)

        elif api_format in ("openai_chat", "openai_responses"):
            from .converter_openai import anthropic_to_openai_request
            headers = _build_openai_headers(provider.api_key)
            base_url = endpoint.resolve_base_url(api_format, provider.base_url)
            if api_format == "openai_responses":
                from .converter_responses import anthropic_to_responses_request
                req_body = anthropic_to_responses_request(base_body, backend_model)
                url = f"{base_url}/v1/responses"
            else:
                req_body = anthropic_to_openai_request(base_body, backend_model)
                url = f"{base_url}/v1/chat/completions"
            resp = await client.post(url, headers=headers, json=req_body)

        else:
            return {
                "success": False,
                "error": f"Unsupported api_format for direct test: {api_format}",
                "latency_ms": None,
                "model": resolved_model_str,
                "backend_model": backend_model,
                "api_format": api_format,
            }

        latency_ms = round((time.monotonic() - start) * 1000)

        if resp.status_code == 200:
            preview = _extract_preview(resp, api_format)
            return {
                "success": True,
                "latency_ms": latency_ms,
                "response_preview": preview,
                "model": resolved_model_str,
                "backend_model": backend_model,
                "api_format": api_format,
            }
        else:
            try:
                err_detail = resp.json()
            except Exception:
                err_detail = resp.text[:300]
            return {
                "success": False,
                "latency_ms": latency_ms,
                "error": f"HTTP {resp.status_code}: {err_detail}",
                "model": resolved_model_str,
                "backend_model": backend_model,
                "api_format": api_format,
            }

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": "Request timed out (30s)",
            "latency_ms": None,
            "model": resolved_model_str,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "latency_ms": None,
            "model": resolved_model_str,
        }


def _extract_preview(resp: httpx.Response, api_format: str) -> str:
    try:
        data = resp.json()
    except Exception:
        return resp.text[:100]

    if api_format == "anthropic":
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block["text"][:120]
        return ""

    if api_format == "openai_responses":
        for item in data.get("output", []):
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part.get("text", "")[:120]
        return ""

    # openai_chat
    choices = data.get("choices", [])
    if choices:
        return (choices[0].get("message", {}).get("content") or "")[:120]
    return ""
