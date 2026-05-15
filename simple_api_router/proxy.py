"""Central routing and dispatch module.

Handles:
- Parsing 'provider/model' from request body
- Resolving provider config
- Proxying Anthropic backends (pure pass-through)
- Converting and proxying OpenAI backends (full format conversion)
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Callable, Dict, Optional, Tuple

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

# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

# HTTP status codes that warrant a retry
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})

# Network/transport errors that warrant a retry.
# httpx.TimeoutException  covers: ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout
# httpx.NetworkError      covers: ConnectError, ReadError, WriteError, CloseError
# httpx.RemoteProtocolError: upstream sent malformed HTTP (truncated response, etc.)
_UPSTREAM_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


def _backoff(attempt: int, retry_after: Optional[float] = None) -> float:
    """Return seconds to wait before attempt N (0-based). Respects Retry-After when present."""
    if retry_after is not None:
        return min(retry_after, 60.0)
    return min(0.5 * (2 ** attempt), 8.0)


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    max_retries: int,
) -> Tuple[Optional[httpx.Response], Optional[str]]:
    """POST with retry. Returns (response, None) on success, (None, error_str) on exhaustion."""
    last_err: Any = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            ra = float(last_err.headers.get("retry-after", 0)) if isinstance(last_err, httpx.Response) else None
            delay = _backoff(attempt - 1, ra or None)
            logger.warning("Retry %d/%d → %s (reason: %s), waiting %.1fs",
                           attempt, max_retries, url, last_err, delay)
            await asyncio.sleep(delay)
        try:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code not in _RETRY_STATUS:
                return resp, None
            last_err = resp
        except _UPSTREAM_ERRORS as exc:
            last_err = exc

    reason = f"HTTP {last_err.status_code}" if isinstance(last_err, httpx.Response) else str(last_err)
    logger.warning("All %d retries exhausted for %s: %s", max_retries, url, reason)
    return None, reason


async def _stream_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    max_retries: int,
    make_stream: Callable,
) -> AsyncIterator[bytes]:
    """Stream POST with retry. Yields from make_stream(aiter_bytes) on success.

    make_stream is a callable that takes an aiter_bytes and returns an async iterable.
    Retries happen before the first byte is yielded so the client sees no error.
    """
    last_err: Any = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            ra = float(last_err.headers.get("retry-after", 0)) if isinstance(last_err, httpx.Response) else None
            delay = _backoff(attempt - 1, ra or None)
            logger.warning("Stream retry %d/%d → %s (reason: %s), waiting %.1fs",
                           attempt, max_retries, url, last_err, delay)
            await asyncio.sleep(delay)
        try:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code in _RETRY_STATUS:
                    last_err = resp
                    continue
                # Once we start yielding chunks the client has received data —
                # we cannot restart the stream. Track whether we've begun so that
                # a mid-stream network error surfaces as an SSE error event instead
                # of silently looping and sending duplicate/corrupt chunks.
                started = False
                try:
                    async for chunk in make_stream(resp.aiter_bytes()):
                        started = True
                        yield chunk
                except _UPSTREAM_ERRORS as mid_exc:
                    if started:
                        logger.warning("Upstream error mid-stream for %s: %s", url, mid_exc)
                        yield _upstream_error_sse(mid_exc)
                        return
                    # Error before first chunk — treat as connection failure, retry
                    raise
                return  # success
        except _UPSTREAM_ERRORS as exc:
            last_err = exc

    reason = f"HTTP {last_err.status_code}" if isinstance(last_err, httpx.Response) else str(last_err)
    logger.warning("All %d stream retries exhausted for %s: %s", max_retries, url, reason)
    yield _upstream_error_sse(last_err)


def _upstream_error_json(exc: Any) -> dict:
    msg = f"HTTP {exc.status_code}" if isinstance(exc, httpx.Response) else str(exc)
    return {"type": "error", "error": {"type": "api_error", "message": f"Upstream error: {msg}"}}


def _upstream_error_sse(exc: Any) -> bytes:
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
    headers: Dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key
    for h in _FORWARD_HEADERS:
        if v := request.headers.get(h):
            headers[h] = v
    if "anthropic-version" not in headers:
        headers["anthropic-version"] = "2023-06-01"
    return headers


def _build_openai_headers(api_key: str) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


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
    max_retries = config.server.max_retries

    if provider.type == "anthropic":
        return await _proxy_anthropic(request, body, backend_model, provider, client, max_retries)
    else:
        return await _proxy_openai(request, body, model_str, backend_model, provider, client, max_retries)


# ---------------------------------------------------------------------------
# Anthropic backend — pure pass-through
# ---------------------------------------------------------------------------

async def _proxy_anthropic(
    request: Request,
    body: Dict[str, Any],
    backend_model: str,
    provider: ProviderConfig,
    client: httpx.AsyncClient,
    max_retries: int,
) -> Any:
    # Replace model with backend name; everything else passes through unchanged
    patched = {**body, "model": backend_model}
    headers = _build_anthropic_headers(request, provider.api_key)
    base_url = provider.resolve_base_url()
    url = f"{base_url}/v1/messages"

    if body.get("stream", False):
        return StreamingResponse(
            _stream_with_retry(client, url, headers, patched, max_retries,
                               lambda aiter: aiter),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    resp, err = await _post_with_retry(client, url, headers, patched, max_retries)
    if err:
        return JSONResponse(status_code=502, content=_upstream_error_json(err))
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def _proxy_openai(
    request: Request,
    body: Dict[str, Any],
    original_model: str,
    backend_model: str,
    provider: ProviderConfig,
    client: httpx.AsyncClient,
    max_retries: int,
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
                _stream_with_retry(client, url, headers, req_body, max_retries,
                                   lambda aiter: stream_responses_to_anthropic(aiter, original_model)),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        resp, err = await _post_with_retry(client, url, headers, req_body, max_retries)
        if err:
            return JSONResponse(status_code=502, content=_upstream_error_json(err))
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
            _stream_with_retry(client, url, headers, oai_body, max_retries,
                               lambda aiter: stream_openai_to_anthropic(aiter, original_model)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    resp, err = await _post_with_retry(client, url, headers, oai_body, max_retries)
    if err:
        return JSONResponse(status_code=502, content=_upstream_error_json(err))
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return JSONResponse(content=openai_to_anthropic_response(resp.json(), original_model))
