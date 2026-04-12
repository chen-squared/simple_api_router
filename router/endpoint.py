"""Single API backend with retry, cooldown, and usage tracking."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx

from router.config import APIConfig
from router.converter import (
    anthropic_to_openai_request,
    anthropic_to_openai_response,
    extract_token_counts,
    extract_tokens_from_response,
    openai_to_anthropic_request,
    openai_to_anthropic_response,
    stream_anthropic_to_openai,
    stream_openai_to_anthropic,
)
from router.logger import get_logger
from router.retry import RetryTracker
from router.usage import UsageTracker

log = get_logger("endpoint")

# HTTP status codes that are retryable
RETRYABLE_CODES = {429, 500, 502, 503, 504, 529}


class APIEndpoint:
    """Wraps a single upstream API with retry and usage tracking."""

    def __init__(self, api_id: str, cfg: APIConfig) -> None:
        self.api_id = api_id
        self.cfg = cfg
        self.usage = UsageTracker(api_id, cfg.usage)
        self.retry = RetryTracker(api_id, cfg.retry)

    async def is_available(self) -> Tuple[bool, str]:
        """Check if this endpoint can accept new requests."""
        if await self.retry.is_in_cooldown():
            remaining = await self.retry.cooldown_remaining()
            return False, f"in retry cooldown ({remaining:.0f}s remaining)"
        available, reason = await self.usage.check_available()
        return available, reason

    async def call(
        self,
        request_body: Dict[str, Any],
        request_format: str,   # "openai" or "anthropic"
        client: httpx.AsyncClient,
    ) -> Tuple[bool, Any]:
        """
        Send the request to this API.

        Returns (is_streaming, response_body_or_stream).
        Raises exceptions on unrecoverable failures.
        """
        available, reason = await self.is_available()
        if not available:
            raise EndpointUnavailableError(self.api_id, reason)

        # Atomically claim an RPM slot to prevent concurrent over-dispatch.
        # is_available() above uses is_rate_limited() as a fast non-atomic hint;
        # try_claim_rpm_slot() is the authoritative gate that records the slot
        # inside the same lock acquisition as the check.
        if not await self.usage.try_claim_rpm_slot():
            raise EndpointUnavailableError(self.api_id, "RPM limit exceeded")

        # Convert request format if needed
        target_body = self._convert_request(request_body, request_format)

        # Build URL and headers
        url, headers = self._build_request_parts(target_body)

        is_streaming = target_body.get("stream", False)

        error_counts: Dict[int, int] = defaultdict(int)
        total_retries = 0

        while True:
            log.info(
                "[%s] → %s stream=%s retry=%d",
                self.api_id, url, is_streaming, total_retries,
            )
            try:
                if is_streaming:
                    # _call_streaming now connects eagerly; UpstreamError is raised here
                    # on non-200 so error counts and fallback work the same as non-streaming.
                    try:
                        stream = await self._call_streaming(
                            url, headers, target_body, request_format, client
                        )
                        return True, stream
                    except UpstreamError as e:
                        # Record failure so retry/cooldown/usage tracking is consistent
                        status = e.status
                        error_counts[status] += 1
                        await self.retry.on_failure(status)
                        await self.usage.record_request(success=False)
                        log.warning("[%s] streaming error status=%d", self.api_id, status)
                        # Apply the same retry/fallback logic as non-streaming below
                        if status == 429:
                            if await self.usage.is_request_quota_exceeded():
                                await self.usage.on_request_quota_exceeded_429()
                                raise EndpointUnavailableError(self.api_id, "request quota exceeded + 429")
                            if await self.usage.is_budget_exceeded():
                                await self.usage.on_budget_exceeded_429()
                                raise EndpointUnavailableError(self.api_id, "budget exceeded + 429")
                        if await self.retry.is_in_cooldown():
                            raise EndpointUnavailableError(self.api_id, "cooldown triggered")
                        if status in RETRYABLE_CODES:
                            if not self.retry.should_retry_error(status, dict(error_counts)):
                                raise UpstreamError(self.api_id, status, f"retry limit for {status}")
                            if not self.retry.should_retry_general(total_retries):
                                raise UpstreamError(self.api_id, status, "max retries reached")
                            total_retries += 1
                            backoff = min(2 ** total_retries, 30)
                            log.info("[%s] retrying stream in %ds", self.api_id, backoff)
                            await asyncio.sleep(backoff)
                            continue
                        else:
                            raise UpstreamError(self.api_id, status, "non-retryable streaming error")
                else:
                    resp_body, status = await self._call_once(url, headers, target_body, client)
            except httpx.RequestError as e:
                log.error("[%s] network error: %s", self.api_id, e)
                await self.retry.on_failure(0)
                await self.usage.record_request(success=False)
                raise

            if status == 200:
                input_tokens, output_tokens = extract_token_counts(resp_body, self.cfg.type)
                await self.usage.record_request(
                    success=True, input_tokens=input_tokens, output_tokens=output_tokens
                )
                await self.retry.on_success()
                # Convert response if needed
                result = self._convert_response(resp_body, request_format)
                log.info("[%s] ✓ success in=%d out=%d", self.api_id, input_tokens, output_tokens)
                return False, result

            # Non-200: handle errors
            error_counts[status] += 1
            await self.retry.on_failure(status)
            await self.usage.record_request(success=False)

            log.warning(
                "[%s] error status=%d error_count=%d",
                self.api_id, status, error_counts[status],
            )

            # Check for quota/budget exceeded + 429
            if status == 429:
                if await self.usage.is_request_quota_exceeded():
                    await self.usage.on_request_quota_exceeded_429()
                    raise EndpointUnavailableError(self.api_id, "request quota exceeded + 429")
                if await self.usage.is_budget_exceeded():
                    await self.usage.on_budget_exceeded_429()
                    raise EndpointUnavailableError(self.api_id, "budget exceeded + 429")

            # Check cooldown
            if await self.retry.is_in_cooldown():
                raise EndpointUnavailableError(self.api_id, "cooldown triggered")

            # Decide whether to retry
            if status in RETRYABLE_CODES:
                if not self.retry.should_retry_error(status, dict(error_counts)):
                    log.warning("[%s] error %d retry limit reached", self.api_id, status)
                    raise UpstreamError(self.api_id, status, f"retry limit for {status}")
                if not self.retry.should_retry_general(total_retries):
                    raise UpstreamError(self.api_id, status, "max retries reached")
                total_retries += 1
                backoff = min(2 ** total_retries, 30)
                log.info("[%s] retrying in %ds", self.api_id, backoff)
                await asyncio.sleep(backoff)
            else:
                raise UpstreamError(self.api_id, status, "non-retryable error")

    async def _call_once(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict,
        client: httpx.AsyncClient,
    ) -> Tuple[Dict, int]:
        resp = await client.post(url, json=body, headers=headers, timeout=120.0)
        status = resp.status_code
        try:
            resp_body = resp.json()
        except Exception:
            resp_body = {"error": resp.text}
        return resp_body, status

    async def _call_streaming(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict,
        request_format: str,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[bytes]:
        """
        Returns an async iterator of SSE bytes in the original request format.

        Eagerly connects to the backend and checks the HTTP status before returning
        so that non-200 errors raise UpstreamError immediately (enabling retry/fallback
        in the caller instead of surfacing the error mid-stream after 200 is sent).

        The returned iterator's aclose() always closes the underlying HTTP response,
        even if iteration never started (client abandons before first read).
        """
        backend_format = self.cfg.type
        needs_conversion = request_format != backend_format

        # Eagerly send the request so the status code is known before we commit
        # to streaming the response to the client.
        request = client.build_request("POST", url, json=body, headers=headers)
        response = await client.send(request, stream=True)

        if response.status_code != 200:
            content = await response.aread()
            await response.aclose()
            raise UpstreamError(self.api_id, response.status_code, content.decode())

        await self.usage.record_request(success=True)
        await self.retry.on_success()

        # Use a class-based wrapper so aclose() always closes the HTTP response
        # regardless of whether iteration has started. An async-generator-based
        # wrapper would not close the response if abandoned before iteration begins
        # (Python only runs generator finally blocks on started generators).
        raw = _ResponseStream(response)

        if not needs_conversion:
            return raw
        if backend_format == "openai":
            # backend streams OpenAI, convert to Anthropic
            return stream_openai_to_anthropic(raw)
        else:
            # backend streams Anthropic, convert to OpenAI
            return stream_anthropic_to_openai(raw)

    def _convert_request(self, body: Dict, request_format: str) -> Dict:
        backend = self.cfg.type
        if request_format == backend:
            result = dict(body)
        elif request_format == "openai" and backend == "anthropic":
            result = openai_to_anthropic_request(body)
        else:  # anthropic → openai
            result = anthropic_to_openai_request(body)

        # Apply model override
        if self.cfg.model:
            result["model"] = self.cfg.model
        return result

    def _convert_response(self, body: Dict, request_format: str) -> Dict:
        backend = self.cfg.type
        if request_format == backend:
            return body
        if backend == "anthropic" and request_format == "openai":
            return anthropic_to_openai_response(body)
        if backend == "openai" and request_format == "anthropic":
            return openai_to_anthropic_response(body)
        return body

    def _build_request_parts(self, body: Dict) -> Tuple[str, Dict[str, str]]:
        backend = self.cfg.type
        base = self.cfg.base_url.rstrip("/")
        if self.cfg.endpoint_path:
            url = f"{base}{self.cfg.endpoint_path}"
        elif backend == "openai":
            url = f"{base}/chat/completions"
        else:  # anthropic
            url = f"{base}/v1/messages"
        if backend == "openai":
            headers = {
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
            }
        else:
            headers = {
                "x-api-key": self.cfg.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        return url, headers

    def stats(self) -> dict:
        return {
            "api_id": self.api_id,
            "type": self.cfg.type,
            "usage": self.usage.stats(),
            "retry": self.retry.stats(),
        }


class EndpointUnavailableError(Exception):
    def __init__(self, api_id: str, reason: str) -> None:
        self.api_id = api_id
        self.reason = reason
        super().__init__(f"[{api_id}] unavailable: {reason}")


class UpstreamError(Exception):
    def __init__(self, api_id: str, status: int, detail: str) -> None:
        self.api_id = api_id
        self.status = status
        self.detail = detail
        super().__init__(f"[{api_id}] upstream error {status}: {detail}")


class _ResponseStream:
    """
    Wraps an httpx streaming response as an async iterator of SSE lines (bytes).

    Unlike an async-generator wrapper, this class guarantees that
    response.aclose() is called when aclose() is invoked, even if iteration
    has not yet started. This prevents HTTP connection leaks when a streaming
    response is abandoned before the client reads the first byte.
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._lines_iter = response.aiter_lines()

    def __aiter__(self) -> "_ResponseStream":
        return self

    async def __anext__(self) -> bytes:
        try:
            line = await self._lines_iter.__anext__()
        except StopAsyncIteration:
            await self._response.aclose()
            raise
        # Preserve blank lines so SSE double-newline event boundaries are intact.
        return (line + "\n").encode()

    async def aclose(self) -> None:
        await self._response.aclose()
