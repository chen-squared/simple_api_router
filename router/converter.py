"""
Format conversion between OpenAI and Anthropic API formats.

Handles both request and response conversion, including streaming.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional


# ---------------------------------------------------------------------------
# Request conversion
# ---------------------------------------------------------------------------


def openai_to_anthropic_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI chat/completions request to Anthropic messages request."""
    messages = body.get("messages", [])
    system_parts: List[str] = []
    filtered_messages: List[Dict] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content if isinstance(content, str) else _flatten_content(content))
        else:
            filtered_messages.append(_convert_message_openai_to_anthropic(msg))

    result: Dict[str, Any] = {
        "model": body.get("model", "claude-3-5-sonnet-20241022"),
        "messages": filtered_messages,
        "max_tokens": body.get("max_tokens", 4096),
    }
    if system_parts:
        result["system"] = "\n\n".join(system_parts)
    if "temperature" in body:
        result["temperature"] = body["temperature"]
    if "top_p" in body:
        result["top_p"] = body["top_p"]
    if "stop" in body:
        stop = body["stop"]
        result["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    if body.get("stream"):
        result["stream"] = True
    return result


def anthropic_to_openai_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Anthropic messages request to OpenAI chat/completions request."""
    messages: List[Dict] = []

    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": system})

    for msg in body.get("messages", []):
        messages.append(_convert_message_anthropic_to_openai(msg))

    result: Dict[str, Any] = {
        "model": body.get("model", "gpt-4o"),
        "messages": messages,
    }
    if "max_tokens" in body:
        result["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        result["temperature"] = body["temperature"]
    if "top_p" in body:
        result["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        seq = body["stop_sequences"]
        result["stop"] = seq[0] if len(seq) == 1 else seq
    if body.get("stream"):
        result["stream"] = True
    return result


def _convert_message_openai_to_anthropic(msg: Dict) -> Dict:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    if role == "assistant" and isinstance(content, list):
        # Handle tool calls etc — flatten for now
        content = _flatten_content(content)
    if isinstance(content, str):
        return {"role": role, "content": [{"type": "text", "text": content}]}
    return {"role": role, "content": content}


def _convert_message_anthropic_to_openai(msg: Dict) -> Dict:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    if isinstance(content, list):
        text = _flatten_content(content)
        return {"role": role, "content": text}
    return {"role": role, "content": content}


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "content":
                    parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


def anthropic_to_openai_response(body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Anthropic messages response to OpenAI chat/completions response."""
    content_blocks = body.get("content", [])
    text = _flatten_content(content_blocks)
    stop_reason = body.get("stop_reason", "end_turn")
    finish_reason = "stop" if stop_reason in ("end_turn", "stop_sequence") else stop_reason

    usage = body.get("usage", {})
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)

    return {
        "id": body.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def openai_to_anthropic_response(body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI chat/completions response to Anthropic messages response."""
    choices = body.get("choices", [])
    text = ""
    finish_reason = "end_turn"
    if choices:
        msg = choices[0].get("message", {})
        text = msg.get("content", "")
        fr = choices[0].get("finish_reason", "stop")
        finish_reason = "end_turn" if fr == "stop" else fr

    usage = body.get("usage", {})
    return {
        "id": body.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": body.get("model", ""),
        "stop_reason": finish_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Streaming conversion
# ---------------------------------------------------------------------------


async def stream_openai_to_anthropic(
    source: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """
    Convert an OpenAI SSE stream to Anthropic SSE format.
    Yields raw bytes for each SSE event.

    All yields are inside the try/finally block so the source iterator is
    closed whether the stream completes normally or is abandoned mid-stream
    (e.g. client disconnect).
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    finish_reason = None
    try:
        # Send message_start
        yield _anthropic_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )
        yield _anthropic_event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        yield _anthropic_event("ping", {"type": "ping"})

        async for chunk in source:
            line = chunk.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            delta = data.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            finish_reason = data.get("choices", [{}])[0].get("finish_reason") or finish_reason

            if content:
                yield _anthropic_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": content}},
                )

        yield _anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0})
        stop_reason = "end_turn" if finish_reason in (None, "stop") else finish_reason
        yield _anthropic_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 0}},
        )
        yield _anthropic_event("message_stop", {"type": "message_stop"})
    finally:
        await source.aclose()


async def stream_anthropic_to_openai(
    source: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """
    Convert an Anthropic SSE stream to OpenAI SSE format.
    Yields raw bytes for each SSE line.

    All yields are inside the try/finally block so the source iterator is
    closed whether the stream completes normally or is abandoned mid-stream
    (e.g. client disconnect).
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model = ""
    created = int(time.time())

    event_type = None
    try:
        # Send initial role chunk before consuming source
        yield _openai_chunk(chat_id, model, created, {"role": "assistant", "content": None})

        async for chunk in source:
            line = chunk.decode("utf-8", errors="replace").strip()
            if line.startswith("event:"):
                event_type = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            dtype = data.get("type", "")

            if dtype == "message_start":
                model = data.get("message", {}).get("model", "")

            elif dtype == "content_block_delta":
                text = data.get("delta", {}).get("text", "")
                if text:
                    yield _openai_chunk(chat_id, model, created, {"content": text})

            elif dtype == "message_delta":
                stop_reason = data.get("delta", {}).get("stop_reason", "end_turn")
                finish = "stop" if stop_reason in ("end_turn", "stop_sequence") else stop_reason
                yield _openai_chunk(chat_id, model, created, {}, finish_reason=finish)

        yield b"data: [DONE]\n\n"
    finally:
        await source.aclose()


def _anthropic_event(event: str, data: Dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _openai_chunk(
    chat_id: str,
    model: str,
    created: int,
    delta: Dict,
    finish_reason: Optional[str] = None,
) -> bytes:
    payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def extract_token_counts(body: Dict, api_type: str) -> tuple[int, int]:
    """
    Extract (input_tokens, output_tokens) from a response body.
    Returns (0, 0) if usage info is absent.
    """
    usage = body.get("usage", {})
    if api_type == "openai":
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    # anthropic
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def extract_tokens_from_response(body: Dict, api_type: str) -> int:
    """Legacy helper — returns total token count (input + output)."""
    inp, out = extract_token_counts(body, api_type)
    return inp + out
