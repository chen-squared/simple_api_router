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

from .config import EndpointConfig, ProviderConfig, RouterConfig
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
    return last_err if isinstance(last_err, httpx.Response) else None, reason


async def _streaming_request_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    max_retries: int,
) -> httpx.Response:
    """Stream POST with retry. Returns an open httpx.Response on 2xx.

    Status code is checked BEFORE returning, so callers can raise HTTPException
    with the correct status code (401, 403, 404, …) instead of always 200.
    Caller MUST NOT close the response — _stream_raw/_stream_converted do that.

    Raises:
        HTTPException: non-retryable non-2xx upstream error (correct status preserved)
        HTTPException(502): all retries exhausted
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
            req = client.build_request("POST", url, json=body, headers=headers)
            resp = await client.send(req, stream=True)
            if resp.status_code in _RETRY_STATUS:
                await resp.aclose()
                last_err = resp
                continue
            if not (200 <= resp.status_code < 300):
                await resp.aread()
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                await resp.aclose()
                logger.warning("Upstream %d for %s: %s", resp.status_code, url, detail)
                raise HTTPException(status_code=resp.status_code, detail=detail)
            return resp  # open; _stream_raw/_stream_converted will close it
        except _UPSTREAM_ERRORS as exc:
            last_err = exc

    reason = f"HTTP {last_err.status_code}" if isinstance(last_err, httpx.Response) else str(last_err)
    status = last_err.status_code if isinstance(last_err, httpx.Response) else 502
    logger.warning("All %d stream retries exhausted for %s: %s", max_retries, url, reason)
    raise HTTPException(status_code=status, detail=f"Upstream error: {reason}")


async def _stream_raw(resp: httpx.Response, url: str) -> AsyncIterator[bytes]:
    """Yield raw bytes; emit SSE error event on mid-stream network failure."""
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
    except _UPSTREAM_ERRORS as exc:
        logger.warning("Upstream error mid-stream for %s: %s", url, exc)
        yield _upstream_error_sse(exc)
    finally:
        await resp.aclose()


async def _stream_converted(
    resp: httpx.Response,
    make_stream: Callable,
    url: str,
) -> AsyncIterator[bytes]:
    """Yield converted chunks; emit SSE error event on mid-stream network failure."""
    try:
        async for chunk in make_stream(resp.aiter_bytes()):
            yield chunk
    except _UPSTREAM_ERRORS as exc:
        logger.warning("Upstream error mid-stream for %s: %s", url, exc)
        yield _upstream_error_sse(exc)
    finally:
        await resp.aclose()


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
) -> Tuple[ProviderConfig, EndpointConfig, str, str]:
    """Return (provider, endpoint, api_format, backend_model)."""
    if provider_name is not None:
        if provider_name not in config.providers:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{provider_name}' not found. "
                       f"Available: {list(config.providers.keys())}",
            )
        prov = config.providers[provider_name]
        result = prov.find_model(model)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model}' not found in provider '{provider_name}'.",
            )
        api_format, ep = result
        return prov, ep, api_format, ep.resolve_model(model)

    # No explicit provider — search by model name
    for prov in config.providers.values():
        result = prov.find_model(model)
        if result is not None:
            api_format, ep = result
            return prov, ep, api_format, ep.resolve_model(model)

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


def _is_real_key(api_key: str) -> bool:
    """Return True only if api_key is a non-empty, non-placeholder string."""
    return bool(api_key) and api_key.lower() not in ("none", "null", "false", "no", "0")


def _build_anthropic_headers(request: Request, api_key: str) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if _is_real_key(api_key):
        # Send both auth header styles so the provider can use whichever it recognises.
        # Standard Anthropic uses x-api-key; some compatible servers (e.g. ollama.com)
        # use Authorization: Bearer. Unrecognised headers are silently ignored.
        headers["x-api-key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    for h in _FORWARD_HEADERS:
        if v := request.headers.get(h):
            headers[h] = v
    if "anthropic-version" not in headers:
        headers["anthropic-version"] = "2023-06-01"
    return headers


def _build_openai_headers(api_key: str) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if _is_real_key(api_key):
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
    provider, endpoint, api_format, backend_model = resolve_provider(provider_name, model, config)
    max_retries = config.server.max_retries

    if api_format == "anthropic":
        return await _proxy_anthropic(request, body, backend_model, provider, endpoint, client, max_retries)
    elif api_format == "google":
        return await _proxy_google(request, body, model_str, backend_model, provider, endpoint, client, max_retries)
    else:
        return await _proxy_openai(request, body, model_str, backend_model, api_format, provider, endpoint, client, max_retries)


# ---------------------------------------------------------------------------
# Anthropic backend — pure pass-through
# ---------------------------------------------------------------------------

async def _proxy_anthropic(
    request: Request,
    body: Dict[str, Any],
    backend_model: str,
    provider: ProviderConfig,
    endpoint: EndpointConfig,
    client: httpx.AsyncClient,
    max_retries: int,
) -> Any:
    # Replace model with backend name; everything else passes through unchanged
    patched = {**body, "model": backend_model}
    headers = _build_anthropic_headers(request, provider.api_key)
    base_url = endpoint.resolve_base_url("anthropic", provider.base_url)
    url = f"{base_url}/v1/messages"

    if body.get("stream", False):
        resp = await _streaming_request_with_retry(client, url, headers, patched, max_retries)
        return StreamingResponse(
            _stream_raw(resp, url),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    resp, err = await _post_with_retry(client, url, headers, patched, max_retries)
    if err:
        status = resp.status_code if resp is not None else 502
        return JSONResponse(status_code=status, content=_upstream_error_json(err))
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def _proxy_openai(
    request: Request,
    body: Dict[str, Any],
    original_model: str,
    backend_model: str,
    api_format: str,
    provider: ProviderConfig,
    endpoint: EndpointConfig,
    client: httpx.AsyncClient,
    max_retries: int,
) -> Any:
    use_reasoning = (
        endpoint.deepseek_reasoning
        if endpoint.deepseek_reasoning is not None
        else is_deepseek_model(backend_model)
    )

    base_url = endpoint.resolve_base_url(api_format, provider.base_url)
    headers = _build_openai_headers(provider.api_key)
    is_stream = body.get("stream", False)

    if api_format == "openai_responses":
        req_body = anthropic_to_responses_request(body, backend_model)
        url = f"{base_url}/v1/responses"

        if is_stream:
            resp = await _streaming_request_with_retry(client, url, headers, req_body, max_retries)
            return StreamingResponse(
                _stream_converted(resp, lambda aiter: stream_responses_to_anthropic(aiter, original_model), url),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        resp, err = await _post_with_retry(client, url, headers, req_body, max_retries)
        if err:
            status = resp.status_code if resp is not None else 502
            return JSONResponse(status_code=status, content=_upstream_error_json(err))
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=detail)
        return JSONResponse(content=responses_to_anthropic_response(resp.json(), original_model))

    # openai_chat (default)
    oai_body = anthropic_to_openai_request(body, backend_model, use_reasoning_content=use_reasoning)
    url = f"{base_url}/v1/chat/completions"

    if is_stream:
        resp = await _streaming_request_with_retry(client, url, headers, oai_body, max_retries)
        return StreamingResponse(
            _stream_converted(resp, lambda aiter: stream_openai_to_anthropic(aiter, original_model), url),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    resp, err = await _post_with_retry(client, url, headers, oai_body, max_retries)
    if err:
        status = resp.status_code if resp is not None else 502
        return JSONResponse(status_code=status, content=_upstream_error_json(err))
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return JSONResponse(content=openai_to_anthropic_response(resp.json(), original_model))


# ---------------------------------------------------------------------------
# Google backend
# ---------------------------------------------------------------------------

async def _proxy_google(
    request: Request,
    body: Dict[str, Any],
    original_model: str,
    backend_model: str,
    provider: ProviderConfig,
    endpoint: EndpointConfig,
    client: httpx.AsyncClient,
    max_retries: int,
) -> Any:
    from .converter_google import anthropic_to_google_request, google_to_anthropic_response, stream_google_to_anthropic

    base_url = endpoint.resolve_base_url("google", provider.base_url)
    headers = _build_openai_headers(provider.api_key)  # Bearer auth
    is_stream = body.get("stream", False)

    google_body = anthropic_to_google_request(body, backend_model)

    if is_stream:
        url = f"{base_url}/v1/models/{backend_model}:streamGenerateContent?alt=sse"
        resp = await _streaming_request_with_retry(client, url, headers, google_body, max_retries)
        return StreamingResponse(
            _stream_converted(resp, lambda aiter: stream_google_to_anthropic(aiter, original_model), url),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    url = f"{base_url}/v1/models/{backend_model}:generateContent"
    resp, err = await _post_with_retry(client, url, headers, google_body, max_retries)
    if err:
        status = resp.status_code if resp is not None else 502
        return JSONResponse(status_code=status, content=_upstream_error_json(err))
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return JSONResponse(content=google_to_anthropic_response(resp.json(), original_model))
