"""Token counting helpers for non-Anthropic backends and local estimation."""
from __future__ import annotations

import json
import re
from typing import Any, Dict

import httpx
import tiktoken

from .logger import get_logger

logger = get_logger("token_count")

# OpenAI vision: typical code-screenshot / mixed-detail estimate per image block.
_IMAGE_URL_TOKEN_ESTIMATE = 765

_O_SERIES_RE = re.compile(r"\bo[1-9](-|\b)|o4-mini|codex", re.IGNORECASE)
_GPT4O_RE = re.compile(r"gpt-4o|gpt-5|chatgpt-4o", re.IGNORECASE)
_DEEPSEEK_RE = re.compile(r"deepseek", re.IGNORECASE)


def _get_encoding(model: str) -> tiktoken.Encoding:
    """Pick a tiktoken encoding for *model*, with sensible fallbacks."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        pass
    if _GPT4O_RE.search(model) or _O_SERIES_RE.search(model):
        return tiktoken.get_encoding("o200k_base")
    if _DEEPSEEK_RE.search(model):
        return tiktoken.get_encoding("cl100k_base")
    return tiktoken.get_encoding("cl100k_base")


def _encode_len(encoding: tiktoken.Encoding, text: str) -> int:
    if not text:
        return 0
    return len(encoding.encode(text))


def _count_openai_message_content(encoding: tiktoken.Encoding, content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return _encode_len(encoding, content)
    if not isinstance(content, list):
        return _encode_len(encoding, json.dumps(content, ensure_ascii=False))

    tokens = 0
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            tokens += _encode_len(encoding, part.get("text", ""))
        elif ptype == "image_url":
            tokens += _IMAGE_URL_TOKEN_ESTIMATE
        elif ptype == "input_audio":
            # No stable public tokenizer for audio; rough duration-agnostic estimate.
            tokens += 256
    return tokens


def count_openai_chat_tokens(body: Dict[str, Any], model: str) -> int:
    """Estimate input tokens for an OpenAI Chat Completions request body."""
    encoding = _get_encoding(model)
    tokens = 0

    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        tokens += 4  # per-message framing overhead (OpenAI cookbook approximation)
        tokens += _encode_len(encoding, msg.get("role", ""))
        tokens += _count_openai_message_content(encoding, msg.get("content"))
        if tool_calls := msg.get("tool_calls"):
            tokens += _encode_len(encoding, json.dumps(tool_calls, ensure_ascii=False))
        if reasoning := msg.get("reasoning_content"):
            tokens += _encode_len(encoding, reasoning)

    if tools := body.get("tools"):
        tokens += _encode_len(encoding, json.dumps(tools, ensure_ascii=False))

    return max(1, tokens + 2)


# Fields omitted from the OpenAI input-token count request (generation-only).
_RESPONSES_COUNT_SKIP = frozenset({
    "stream",
    "max_output_tokens",
    "temperature",
    "top_p",
    "store",
    "metadata",
    "parallel_tool_calls",
    "service_tier",
    "user",
})


def _responses_count_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    """Build the JSON body for ``POST /v1/responses/input_tokens``."""
    return {k: v for k, v in body.items() if k not in _RESPONSES_COUNT_SKIP}


async def count_openai_responses_api_tokens(
    body: Dict[str, Any],
    base_url: str,
    headers: Dict[str, str],
    client: httpx.AsyncClient,
) -> int:
    """Call OpenAI ``POST /v1/responses/input_tokens`` (official Responses API counter).

    Raises ``httpx.HTTPStatusError`` on non-2xx so callers can fall back to tiktoken
    when the provider does not implement this endpoint (e.g. DeepSeek, Ollama).
    """
    url = f"{base_url.rstrip('/')}/v1/responses/input_tokens"
    resp = await client.post(
        url,
        headers=headers,
        json=_responses_count_payload(body),
        timeout=httpx.Timeout(30.0),
    )
    if resp.status_code != 200:
        raise httpx.HTTPStatusError(
            f"OpenAI input_tokens count failed: {resp.text}",
            request=resp.request,
            response=resp,
        )
    data = resp.json()
    total = data.get("input_tokens")
    if total is None:
        raise ValueError(f"Unexpected input_tokens response: {data}")
    return max(1, int(total))


def count_responses_tokens(body: Dict[str, Any], model: str) -> int:
    """Estimate input tokens for an OpenAI Responses API request body."""
    encoding = _get_encoding(model)
    tokens = 0

    if instructions := body.get("instructions"):
        tokens += _encode_len(encoding, instructions if isinstance(instructions, str) else json.dumps(instructions))

    for item in body.get("input", []):
        if not isinstance(item, dict):
            continue
        tokens += 4
        tokens += _encode_len(encoding, item.get("role", item.get("type", "")))
        content = item.get("content")
        if isinstance(content, str):
            tokens += _encode_len(encoding, content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in ("input_text", "output_text", "text"):
                        tokens += _encode_len(encoding, part.get("text", ""))
                    elif part.get("type") in ("input_image", "image_url"):
                        tokens += _IMAGE_URL_TOKEN_ESTIMATE

    if tools := body.get("tools"):
        tokens += _encode_len(encoding, json.dumps(tools, ensure_ascii=False))

    return max(1, tokens + 2)


async def count_google_tokens(
    body: Dict[str, Any],
    model: str,
    base_url: str,
    headers: Dict[str, str],
    client: httpx.AsyncClient,
) -> int:
    """Call Gemini ``countTokens`` for an accurate total on the converted body."""
    url = f"{base_url}/v1/models/{model}:countTokens"
    resp = await client.post(url, headers=headers, json=body, timeout=httpx.Timeout(30.0))
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise httpx.HTTPStatusError(
            f"Gemini countTokens failed: {detail}",
            request=resp.request,
            response=resp,
        )
    data = resp.json()
    total = data.get("totalTokens")
    if total is None:
        total = (data.get("total_tokens") or data.get("tokenCount"))
    if total is None:
        raise ValueError(f"Unexpected countTokens response: {data}")
    return max(1, int(total))


def estimate_anthropic_body_tokens(body: Dict[str, Any], model: str) -> int:
    """Local estimate for an Anthropic Messages body (fallback / sanity check)."""
    encoding = _get_encoding(model)
    tokens = 0

    system = body.get("system")
    if isinstance(system, str):
        tokens += _encode_len(encoding, system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                tokens += _encode_len(encoding, block.get("text", ""))

    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        tokens += 4
        tokens += _encode_len(encoding, msg.get("role", ""))
        content = msg.get("content")
        if isinstance(content, str):
            tokens += _encode_len(encoding, content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    tokens += _encode_len(encoding, block.get("text", ""))
                elif btype == "thinking":
                    tokens += _encode_len(encoding, block.get("thinking", ""))
                elif btype == "tool_use":
                    tokens += _encode_len(encoding, json.dumps(block.get("input", {}), ensure_ascii=False))
                    tokens += _encode_len(encoding, block.get("name", ""))
                elif btype == "tool_result":
                    nested = block.get("content", "")
                    if isinstance(nested, str):
                        tokens += _encode_len(encoding, nested)
                    elif isinstance(nested, list):
                        tokens += _count_openai_message_content(encoding, nested)
                elif btype in ("image", "document", "audio", "video"):
                    source = block.get("source") or {}
                    if source.get("type") == "base64":
                        raw_len = len(source.get("data", ""))
                        tokens += max(_IMAGE_URL_TOKEN_ESTIMATE, raw_len // 16)
                    else:
                        tokens += _IMAGE_URL_TOKEN_ESTIMATE

    if tools := body.get("tools"):
        tokens += _encode_len(encoding, json.dumps(tools, ensure_ascii=False))

    return max(1, tokens + 2)