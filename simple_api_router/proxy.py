"""Central routing and dispatch module.

Handles:
- Parsing 'provider/model' from request body
- Resolving provider config
- Proxying Anthropic backends (pure pass-through)
- Converting and proxying OpenAI backends (full format conversion)
"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import ProviderConfig, RouterConfig
from .converter import (
    anthropic_to_openai_request,
    anthropic_to_responses_request,
    is_deepseek_model,
    openai_to_anthropic_response,
    responses_to_anthropic_response,
    stream_openai_to_anthropic,
    stream_responses_to_anthropic,
)
from .logger import get_logger

logger = get_logger("proxy")

# Network-level errors that mean we couldn't reach the upstream at all.
_UPSTREAM_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    httpx.NetworkError,
)


def _upstream_error_json(exc: Exception) -> dict:
    return {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": f"Upstream connection error: {exc}",
        },
    }


def _upstream_error_sse(exc: Exception) -> bytes:
    data = json.dumps(_upstream_error_json(exc))
    return f"event: error\ndata: {data}\n\n".encode()


# ---------------------------------------------------------------------------
# Model string helpers
# ---------------------------------------------------------------------------

_BRACKET_SUFFIX_RE = re.compile(r"\[.*?\]$")


def strip_model_suffixes(model: str) -> str:
    """Strip bracket suffixes like [1m], [4k], [128k] from model names.

    These are Claude Code routing hints that must not be forwarded to providers.
    """
    return _BRACKET_SUFFIX_RE.sub("", model).strip()


def parse_model(model_str: str) -> Tuple[Optional[str], str]:
    """Split 'provider/model' into (provider, model), stripping any bracket suffixes."""
    clean = strip_model_suffixes(model_str)
    if "/" in clean:
        provider, model = clean.split("/", 1)
        return provider.strip(), model.strip()
    return None, clean


def resolve_provider(
    provider_name: Optional[str],
    model: str,
    config: RouterConfig,
) -> Tuple[ProviderConfig, str]:
    """Return (ProviderConfig, backend_model) for the given provider/model pair.

    Raises HTTPException(404) when no matching provider is found.
    """
    if provider_name is not None:
        if provider_name not in config.providers:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{provider_name}' not found. "
                       f"Available: {list(config.providers.keys())}",
            )
        prov = config.providers[provider_name]
        return prov, prov.resolve_model(model)

    # No explicit provider — search by model name
    for prov in config.providers.values():
        if not prov.models or model in prov.models:
            return prov, prov.resolve_model(model)

    raise HTTPException(
        status_code=404,
        detail=f"No provider found for model '{model}'. "
               f"Available providers: {list(config.providers.keys())}",
    )


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

_FORWARD_HEADERS = {
    "anthropic-version",
    "anthropic-beta",
    "content-type",
}


def _build_anthropic_headers(request: Request, api_key: str) -> Dict[str, str]:
    headers: Dict[str, str] = {"x-api-key": api_key}
    for h in _FORWARD_HEADERS:
        if v := request.headers.get(h):
            headers[h] = v
    if "anthropic-version" not in headers:
        headers["anthropic-version"] = "2023-06-01"
    return headers


def _build_openai_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

async def route_request(
    request: Request,
    body: Dict[str, Any],
    config: RouterConfig,
    client: httpx.AsyncClient,
) -> Any:
    """Route an Anthropic /v1/messages request to the appropriate backend."""
    model_str: str = body.get("model", "")
    if not model_str:
        raise HTTPException(status_code=400, detail="'model' field is required")

    provider_name, model = parse_model(model_str)
    provider, backend_model = resolve_provider(provider_name, model, config)

    if provider.type == "anthropic":
        return await _proxy_anthropic(request, body, backend_model, provider, client)
    else:
        return await _proxy_openai(request, body, model_str, backend_model, provider, client)


# ---------------------------------------------------------------------------
# Anthropic backend — pure pass-through
# ---------------------------------------------------------------------------

async def _proxy_anthropic(
    request: Request,
    body: Dict[str, Any],
    backend_model: str,
    provider: ProviderConfig,
    client: httpx.AsyncClient,
) -> Any:
    # Replace model with backend name; everything else passes through unchanged
    patched = {**body, "model": backend_model}
    headers = _build_anthropic_headers(request, provider.api_key)
    base_url = provider.resolve_base_url()
    url = f"{base_url}/v1/messages"

    is_stream = body.get("stream", False)

    if is_stream:
        return StreamingResponse(
            _stream_anthropic(client, url, headers, patched),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        resp = await client.post(url, json=patched, headers=headers)
    except _UPSTREAM_ERRORS as exc:
        logger.warning("Upstream connect error (%s): %s", url, exc)
        return JSONResponse(status_code=502, content=_upstream_error_json(exc))
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def _stream_anthropic(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
) -> AsyncIterator[bytes]:
    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk
    except _UPSTREAM_ERRORS as exc:
        logger.warning("Upstream connect error during stream (%s): %s", url, exc)
        yield _upstream_error_sse(exc)


# ---------------------------------------------------------------------------
# OpenAI backend — full conversion
# ---------------------------------------------------------------------------

async def _proxy_openai(
    request: Request,
    body: Dict[str, Any],
    original_model: str,
    backend_model: str,
    provider: ProviderConfig,
    client: httpx.AsyncClient,
) -> Any:
    use_reasoning = (
        provider.deepseek_reasoning
        if provider.deepseek_reasoning is not None
        else is_deepseek_model(backend_model)
    )

    base_url = provider.resolve_base_url()
    headers = _build_openai_headers(provider.api_key)
    is_stream = body.get("stream", False)

    if provider.api_format == "openai_responses":
        req_body = anthropic_to_responses_request(body, backend_model)
        url = f"{base_url}/responses"

        if is_stream:
            return StreamingResponse(
                _stream_responses_converted(client, url, headers, req_body, original_model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            resp = await client.post(url, json=req_body, headers=headers)
        except _UPSTREAM_ERRORS as exc:
            logger.warning("Upstream connect error (%s): %s", url, exc)
            return JSONResponse(status_code=502, content=_upstream_error_json(exc))
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=detail)
        return JSONResponse(content=responses_to_anthropic_response(resp.json(), original_model))

    # openai_chat (default)
    oai_body = anthropic_to_openai_request(body, backend_model, use_reasoning_content=use_reasoning)
    url = f"{base_url}/chat/completions"

    if is_stream:
        return StreamingResponse(
            _stream_openai_converted(client, url, headers, oai_body, original_model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        resp = await client.post(url, json=oai_body, headers=headers)
    except _UPSTREAM_ERRORS as exc:
        logger.warning("Upstream connect error (%s): %s", url, exc)
        return JSONResponse(status_code=502, content=_upstream_error_json(exc))
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)

    return JSONResponse(content=openai_to_anthropic_response(resp.json(), original_model))


async def _stream_openai_converted(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    original_model: str,
) -> AsyncIterator[bytes]:
    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            async for chunk in stream_openai_to_anthropic(resp.aiter_bytes(), original_model):
                yield chunk
    except _UPSTREAM_ERRORS as exc:
        logger.warning("Upstream connect error during stream (%s): %s", url, exc)
        yield _upstream_error_sse(exc)


async def _stream_responses_converted(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    original_model: str,
) -> AsyncIterator[bytes]:
    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            async for chunk in stream_responses_to_anthropic(resp.aiter_bytes(), original_model):
                yield chunk
    except _UPSTREAM_ERRORS as exc:
        logger.warning("Upstream connect error during stream (%s): %s", url, exc)
        yield _upstream_error_sse(exc)
