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
import secrets
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import EndpointConfig, ProviderConfig, RouterConfig
from . import debug_log as _dlog
from .converter import (
    anthropic_to_openai_request,
    anthropic_to_responses_request,
    is_deepseek_model,
    openai_to_anthropic_response,
    responses_to_anthropic_response,
    stream_openai_to_anthropic,
    stream_responses_to_anthropic,
    _sse_bytes,
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
    if retry_after is not None and retry_after > 0:
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
    debug_id: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """Yield converted chunks; emit SSE error event on mid-stream network failure."""
    raw_chunks: List[bytes] = []

    async def _raw() -> AsyncIterator[bytes]:
        async for chunk in resp.aiter_bytes():
            if debug_id:
                raw_chunks.append(chunk)
            yield chunk

    try:
        async for chunk in make_stream(_raw()):
            yield chunk
    except _UPSTREAM_ERRORS as exc:
        logger.warning("Upstream error mid-stream for %s: %s", url, exc)
        yield _upstream_error_sse(exc)
    finally:
        if debug_id:
            _dlog.log(debug_id, "3_upstream_raw", b"".join(raw_chunks))
        await resp.aclose()


async def _stream_converted_with_retry(
    first_resp: httpx.Response,
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    make_stream: Callable,
    max_retries: int,
    debug_id: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """Streaming pipeline with two-phase retry:

    Phase 1 (HTTP level): handled by the caller via _streaming_request_with_retry
    before this function is called.  ``first_resp`` is an already-open 200 response.

    Phase 2 (in-stream error): this function buffers converter output until real
    content (content_block_start / content_block_delta) arrives.  If an error
    event is detected before any real content, the upstream request is retried
    (up to max_retries times).  Once real content starts flowing the stream is
    committed — no further retries are possible.

    On retry, HTTP-level errors are handled inline: retryable codes are retried,
    non-retryable codes / network errors are forwarded as SSE error events
    (because HTTP 200 headers are already sent by then).
    """
    resp = first_resp

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = _backoff(attempt - 1)
            logger.warning(
                "Stream retry %d/%d → %s (early in-stream error), waiting %.1fs",
                attempt, max_retries, url, delay,
            )
            await asyncio.sleep(delay)
            # Make a fresh HTTP request for this retry.
            # HTTP 200 headers are already sent, so non-2xx errors become SSE events.
            try:
                req = client.build_request("POST", url, json=body, headers=headers)
                resp = await client.send(req, stream=True)
            except _UPSTREAM_ERRORS as exc:
                if attempt < max_retries:
                    continue
                yield _upstream_error_sse(exc)
                return
            if resp.status_code in _RETRY_STATUS:
                await resp.aclose()
                continue
            if not (200 <= resp.status_code < 300):
                await resp.aread()
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                await resp.aclose()
                logger.warning("Stream retry upstream %d for %s: %s", resp.status_code, url, detail)
                yield _sse_bytes("error", {
                    "type": "error",
                    "error": {"type": "api_error", "message": f"Upstream HTTP {resp.status_code}"},
                })
                return

        # --- stream + early-error buffer ---
        buffer: List[bytes] = []
        committed = False
        early_error = False

        async for chunk in _stream_converted(resp, make_stream, url, debug_id=debug_id):
            if committed:
                yield chunk
                continue
            buffer.append(chunk)
            # A chunk from the converter is always a single SSE event.
            if chunk.startswith(b"event: error\n"):
                early_error = True
                break
            # Real content has started — commit and flush the buffer.
            if b"content_block_start" in chunk or b"content_block_delta" in chunk:
                committed = True
                for c in buffer:
                    yield c
                buffer.clear()

        if not early_error:
            # If no real content arrived (committed=False), treat as retryable —
            # upstream returned 200 with an empty body (0/0 tokens, no content blocks).
            if not committed and attempt < max_retries:
                logger.warning(
                    "Stream completed without content from %s — events: [%s] — retrying %d/%d in %.1fs",
                    url, _buffer_event_summary(buffer), attempt + 1, max_retries, _backoff(attempt),
                )
                await asyncio.sleep(_backoff(attempt))
                # buffer (preamble) is discarded; resp already closed by _stream_converted
                continue
            # Retries exhausted with empty response — return error instead of empty preamble.
            if not committed:
                logger.error(
                    "All %d attempts returned empty response from %s — events: [%s]",
                    max_retries + 1, url, _buffer_event_summary(buffer),
                )
                yield _sse_bytes("error", {
                    "type": "error",
                    "error": {"type": "api_error", "message": "Upstream returned empty response after retries"},
                })
                return
            # Normal completion — flush buffer.
            for c in buffer:
                yield c
            return

        # Early error, no content delivered — retry if budget allows.
        if attempt < max_retries:
            continue

        # Budget exhausted — forward the error to the client.
        logger.warning("All %d stream retries exhausted (early in-stream error) for %s", max_retries, url)
        for c in buffer:
            yield c


def _buffer_event_summary(buffer: List[bytes]) -> str:
    """Return a concise summary of SSE event types in the buffer for diagnostics.

    For 'message_start' or 'message_delta' events, appends the usage/stop_reason
    so we can tell if the upstream sent 0/0 tokens vs an unrecognised payload.
    """
    parts: List[str] = []
    for chunk in buffer:
        try:
            text = chunk.decode(errors="replace")
            lines = text.split("\n")
            event_type = ""
            data_str = ""
            for line in lines:
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data_str = line[6:].strip()
            if not event_type:
                parts.append("(unknown)")
                continue
            # For key events, include relevant fields to aid diagnosis.
            if event_type in ("message_start", "message_delta") and data_str:
                try:
                    d = json.loads(data_str)
                    usage = d.get("usage") or (d.get("message") or {}).get("usage") or {}
                    stop = (d.get("delta") or {}).get("stop_reason", "")
                    extra = f"in={usage.get('input_tokens','')} out={usage.get('output_tokens','')}"
                    if stop:
                        extra += f" stop={stop}"
                    parts.append(f"{event_type}({extra})")
                except Exception:
                    parts.append(event_type)
            else:
                parts.append(event_type)
        except Exception:
            parts.append("?")
    return ", ".join(parts) if parts else "(empty)"


def _upstream_error_json(exc: Any) -> Dict[str, Any]:
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


_MEDIA_TYPES = frozenset({"image", "video", "document"})


def _blocks_have_media(blocks: list) -> bool:
    """Return True if any content block in *blocks* is a non-text media type."""
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in ("image", "video"):
            return True
        if btype == "document":
            # document blocks with source.type == "text" are plain-text and fine for
            # text-only models; base64/url sources are PDFs or binary docs that are not.
            src_type = (block.get("source") or {}).get("type")
            if src_type != "text":
                return True
        # tool_result content can itself contain image/video blocks (e.g. a screenshot tool)
        if btype == "tool_result":
            nested = block.get("content", "")
            if isinstance(nested, list) and _blocks_have_media(nested):
                return True
    return False


def _request_has_media(body: Dict[str, Any]) -> bool:
    """Return True if any message in the request contains image, video, or PDF content."""
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list) and _blocks_have_media(content):
            return True
    return False


def _last_user_has_media(body: Dict[str, Any]) -> bool:
    """Return True if the last user message contains image, video, or PDF content."""
    for msg in reversed(body.get("messages", [])):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return isinstance(content, list) and _blocks_have_media(content)
    return False


def _strip_media_from_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace media blocks in historical messages with text placeholders.

    The last user message is left intact (caller already confirmed it has no media).
    All other messages have image/video/document blocks replaced with [<type> omitted]
    so that text-only upstream models don't choke on stale image blocks in history.
    """
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    result: List[Dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i == last_user_idx:
            result.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, list):
            stripped = []
            changed = False
            for block in content:
                btype = block.get("type")
                if btype in ("image", "video"):
                    stripped.append({"type": "text", "text": f"[{btype} omitted]"})
                    changed = True
                elif btype == "document":
                    src_type = (block.get("source") or {}).get("type")
                    if src_type != "text":
                        stripped.append({"type": "text", "text": "[document omitted]"})
                        changed = True
                        continue
                    stripped.append(block)
                elif btype == "tool_result":
                    nested = block.get("content", "")
                    if isinstance(nested, list) and _blocks_have_media(nested):
                        cleaned = [
                            {"type": "text", "text": f"[{b.get('type')} omitted]"}
                            if b.get("type") in ("image", "video")
                            else b
                            for b in nested
                        ]
                        stripped.append({**block, "content": cleaned})
                        changed = True
                    else:
                        stripped.append(block)
                else:
                    stripped.append(block)
            if changed:
                result.append({**msg, "content": stripped})
                continue
        result.append(msg)
    return result


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

    # Multimodal fallback: if the resolved model is text-only, handle media appropriately.
    # Only check the LAST user message to decide whether to fallback — images in earlier
    # history should not permanently strand the conversation on the fallback model.
    # If the last user message has no media but history does, strip the stale media blocks
    # so the text-only upstream doesn't choke on them.
    entry = endpoint.get_model_entry(model)
    if entry.text_only:
        if _last_user_has_media(body):
            fallback = entry.multimodal_fallback or config.server.multimodal_fallback
            if fallback:
                logger.info(
                    "text_only model '%s' received media content; routing to multimodal fallback: %s",
                    model, fallback,
                )
                fb_provider_name, fb_model = parse_model(fallback)
                provider, endpoint, api_format, backend_model = resolve_provider(
                    fb_provider_name, fb_model, config
                )
                # Update provider_name/model so usage_meta and effort resolution use the fallback model.
                provider_name, model = fb_provider_name, fb_model
                model_str = f"{provider_name}/{model}" if provider_name else model
            else:
                logger.warning(
                    "text_only model '%s' received media content but no multimodal_fallback "
                    "is configured — forwarding anyway",
                    model,
                )
        elif _request_has_media(body):
            # Last user message is text-only but earlier history has image/video blocks.
            # Strip them so the text-only model doesn't error on stale media in history.
            logger.debug(
                "text_only model '%s': stripping stale media blocks from conversation history",
                model,
            )
            body = {**body, "messages": _strip_media_from_history(body.get("messages", []))}

    # Stash routing metadata so app.py can log usage after the response is done.
    # Use the clean "provider/model" form (bracket suffixes stripped) so pricing lookup works.
    resolved_provider_name = next(
        (name for name, p in config.providers.items() if p is provider), "unknown"
    )
    clean_model = f"{provider_name}/{model}" if provider_name else model
    request.state.usage_meta = {
        "model": clean_model,
        "provider": resolved_provider_name,
        "backend_model": backend_model,
    }

    is_stream = body.get("stream", False)
    logger.info(
        "POST /v1/messages model=%s provider=%s backend=%s stream=%s →",
        clean_model, resolved_provider_name, backend_model, is_stream,
    )

    # Debug logging: assign a short request ID and record the incoming body.
    if _dlog.enabled():
        debug_id = secrets.token_hex(4)
        request.state.debug_id = debug_id
        _dlog.log(debug_id, "1_incoming_request", body)
    else:
        debug_id = None

    for attempt in range(max_retries + 1):
        upstream_start = time.time()
        if api_format == "anthropic":
            result = await _proxy_anthropic(request, body, backend_model, provider, endpoint, client, max_retries, debug_id=debug_id)
        elif api_format == "google":
            result = await _proxy_google(request, body, model_str, backend_model, provider, endpoint, client, max_retries, debug_id=debug_id)
        else:
            result = await _proxy_openai(request, body, model_str, backend_model, api_format, provider, endpoint, client, max_retries, debug_id=debug_id)

        if is_stream:
            ttfb_ms = round((time.time() - upstream_start) * 1000)
            logger.info(
                "POST /v1/messages model=%s provider=%s TTFB=%dms",
                clean_model, resolved_provider_name, ttfb_ms,
            )
            break  # streaming handles retries internally

        # Non-streaming: retry if upstream returned a 200 with 0/0 tokens
        if isinstance(result, JSONResponse) and result.status_code == 200 and attempt < max_retries:
            try:
                data = json.loads(result.body)
                u = data.get("usage", {})
                if u.get("input_tokens", 0) == 0 and u.get("output_tokens", 0) == 0:
                    delay = _backoff(attempt)
                    logger.warning(
                        "POST /v1/messages model=%s provider=%s 0/0 tokens — retrying %d/%d in %.1fs",
                        clean_model, resolved_provider_name, attempt + 1, max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
            except Exception:
                pass
        break

    return result


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
    debug_id: Optional[str] = None,
) -> Any:
    # Replace model with backend name; everything else passes through unchanged
    patched = {**body, "model": backend_model}
    headers = _build_anthropic_headers(request, provider.api_key)
    base_url = endpoint.resolve_base_url("anthropic", provider.base_url)
    url = f"{base_url}/v1/messages"

    if debug_id:
        _dlog.log(debug_id, "2_upstream_request", patched)

    if body.get("stream", False):
        resp = await _streaming_request_with_retry(client, url, headers, patched, max_retries)
        stream: AsyncIterator[bytes] = _stream_raw(resp, url)
        if debug_id:
            stream = _dlog.tee_bytes_iter(stream, debug_id, "3_upstream_raw")
            stream = _dlog.tee_bytes_iter(stream, debug_id, "4_downstream_sse")
        return StreamingResponse(
            stream,
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
    data = resp.json()
    if debug_id:
        _dlog.log(debug_id, "3_upstream_raw", data)
        _dlog.log(debug_id, "4_downstream_response", data)
    return JSONResponse(content=data, status_code=resp.status_code)


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
    debug_id: Optional[str] = None,
) -> Any:
    # Precedence: model-level flag → endpoint-level flag → auto-detect from model name
    _, _req_model = parse_model(original_model)
    _model_entry = endpoint.get_model_entry(_req_model)
    use_reasoning = (
        _model_entry.deepseek_reasoning
        if _model_entry.deepseek_reasoning is not None
        else (
            endpoint.deepseek_reasoning
            if endpoint.deepseek_reasoning is not None
            else is_deepseek_model(backend_model)
        )
    )
    max_effort = endpoint.resolve_max_reasoning_effort(_req_model)

    base_url = endpoint.resolve_base_url(api_format, provider.base_url)
    headers = _build_openai_headers(provider.api_key)
    is_stream = body.get("stream", False)

    if api_format == "openai_responses":
        req_body = anthropic_to_responses_request(body, backend_model, max_reasoning_effort=max_effort)
        url = f"{base_url}/v1/responses"

        if debug_id:
            _dlog.log(debug_id, "2_upstream_request", req_body)

        if is_stream:
            resp = await _streaming_request_with_retry(client, url, headers, req_body, max_retries)
            stream = _stream_converted_with_retry(
                resp, client, url, headers, req_body,
                lambda aiter: stream_responses_to_anthropic(aiter, original_model),
                max_retries,
                debug_id=debug_id,
            )
            if debug_id:
                stream = _dlog.tee_bytes_iter(stream, debug_id, "4_downstream_sse")
            return StreamingResponse(
                stream,
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
        raw = resp.json()
        converted = responses_to_anthropic_response(raw, original_model)
        if debug_id:
            _dlog.log(debug_id, "3_upstream_raw", raw)
            _dlog.log(debug_id, "4_downstream_response", converted)
        return JSONResponse(content=converted)

    # openai_chat (default)
    oai_body = anthropic_to_openai_request(
        body, backend_model,
        use_reasoning_content=use_reasoning,
        max_reasoning_effort=max_effort,
    )
    url = f"{base_url}/v1/chat/completions"

    if debug_id:
        _dlog.log(debug_id, "2_upstream_request", oai_body)

    if is_stream:
        resp = await _streaming_request_with_retry(client, url, headers, oai_body, max_retries)
        stream = _stream_converted_with_retry(
            resp, client, url, headers, oai_body,
            lambda aiter: stream_openai_to_anthropic(aiter, original_model),
            max_retries,
            debug_id=debug_id,
        )
        if debug_id:
            stream = _dlog.tee_bytes_iter(stream, debug_id, "4_downstream_sse")
        return StreamingResponse(
            stream,
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
    raw = resp.json()
    converted = openai_to_anthropic_response(raw, original_model)
    if debug_id:
        _dlog.log(debug_id, "3_upstream_raw", raw)
        _dlog.log(debug_id, "4_downstream_response", converted)
    return JSONResponse(content=converted)


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
    debug_id: Optional[str] = None,
) -> Any:
    from .converter_google import anthropic_to_google_request, google_to_anthropic_response, stream_google_to_anthropic

    base_url = endpoint.resolve_base_url("google", provider.base_url)
    headers = _build_openai_headers(provider.api_key)  # Bearer auth
    is_stream = body.get("stream", False)

    google_body = anthropic_to_google_request(body, backend_model)

    if debug_id:
        _dlog.log(debug_id, "2_upstream_request", google_body)

    if is_stream:
        url = f"{base_url}/v1/models/{backend_model}:streamGenerateContent?alt=sse"
        resp = await _streaming_request_with_retry(client, url, headers, google_body, max_retries)
        stream = _stream_converted_with_retry(
            resp, client, url, headers, google_body,
            lambda aiter: stream_google_to_anthropic(aiter, original_model),
            max_retries,
            debug_id=debug_id,
        )
        if debug_id:
            stream = _dlog.tee_bytes_iter(stream, debug_id, "4_downstream_sse")
        return StreamingResponse(
            stream,
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
    raw = resp.json()
    converted = google_to_anthropic_response(raw, original_model)
    if debug_id:
        _dlog.log(debug_id, "3_upstream_raw", raw)
        _dlog.log(debug_id, "4_downstream_response", converted)
    return JSONResponse(content=converted)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

async def count_tokens_request(
    request: Request,
    body: Dict[str, Any],
    config: RouterConfig,
    client: httpx.AsyncClient,
) -> Any:
    """Handle POST /v1/messages/count_tokens.

    For Anthropic backends the request is forwarded to the backend's own
    count_tokens endpoint.  For OpenAI / Google backends (which have no
    equivalent) we return a rough character-based estimate (~4 chars/token).
    """
    model_str: str = body.get("model", "")
    if not model_str:
        raise HTTPException(status_code=400, detail="'model' field is required")

    provider_name, model = parse_model(model_str)
    provider, endpoint, api_format, backend_model = resolve_provider(provider_name, model, config)

    if api_format == "anthropic":
        patched = {**body, "model": backend_model}
        headers = _build_anthropic_headers(request, provider.api_key)
        base_url = endpoint.resolve_base_url("anthropic", provider.base_url)
        url = f"{base_url}/v1/messages/count_tokens"
        resp, err = await _post_with_retry(client, url, headers, patched, config.server.max_retries)
        if err:
            raise HTTPException(status_code=502, detail=f"Upstream error: {err}")
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=detail)
        return JSONResponse(content=resp.json())

    # Non-Anthropic backend: estimate from input size (~4 chars per token)
    total_chars = len(json.dumps(body.get("messages", [])))
    sys = body.get("system")
    if sys:
        total_chars += len(sys) if isinstance(sys, str) else len(json.dumps(sys))
    if "tools" in body:
        total_chars += len(json.dumps(body["tools"]))
    estimated = max(1, total_chars // 4)
    logger.debug("count_tokens model=%s: estimated %d tokens (non-Anthropic backend)", model_str, estimated)
    return JSONResponse({"input_tokens": estimated})
