"""Anthropic Messages ↔ OpenAI Chat Completions conversion.

Handles:
- Request: Anthropic Messages → OpenAI Chat Completions
- Response: OpenAI Chat Completions → Anthropic Messages
- Streaming: OpenAI SSE → Anthropic SSE (full event mapping including tool use)
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from .converter import (
    _normalize_effort,
    _reasoning_effort_from_budget,
    clean_schema,
    is_o_series,
    sanitize_system_text,
    strip_private_params,
)
from .converter_utils import (
    is_anthropic_server_tool as _is_anthropic_server_tool,
    fold_midstream_system_into_user as _fold_midstream_system_into_user,
    sse as _sse_bytes,
    thinking_close_events as _thinking_close_events,
)
from .logger import get_logger

_logger = get_logger("converter")


# ---------------------------------------------------------------------------
# Anthropic → OpenAI  (request)
# ---------------------------------------------------------------------------

def anthropic_to_openai_request(
    body: Dict[str, Any],
    backend_model: str,
    use_reasoning_content: bool = False,
    max_reasoning_effort: str = "xhigh",
) -> Dict[str, Any]:
    """Convert an Anthropic /v1/messages request body to OpenAI chat completions format."""
    body = strip_private_params(body)
    messages: List[Dict[str, Any]] = []

    # --- system prompt ---
    # Merge all system text blocks into a single system message.
    # OpenAI format only allows one system message, and cache_control is
    # Anthropic-specific so it is dropped here.
    system = body.get("system")
    if system:
        if isinstance(system, str):
            sanitized = sanitize_system_text(system)
            if sanitized.strip():
                messages.append({"role": "system", "content": sanitized})
        elif isinstance(system, list):
            parts = [
                sanitize_system_text(b.get("text", ""))
                for b in system
                if b.get("type") == "text"
            ]
            merged = "\n".join(p for p in parts if p.strip())
            if merged:
                messages.append({"role": "system", "content": merged})

    # --- conversation messages ---
    for msg in _fold_midstream_system_into_user(body.get("messages", [])):
        role = msg["role"]
        content = msg.get("content", [])

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if role == "user":
            messages.extend(_user_blocks_to_openai(content))
        else:  # assistant
            messages.extend(_assistant_blocks_to_openai(content, use_reasoning_content))

    oai: Dict[str, Any] = {
        "model": backend_model,
        "messages": messages,
    }

    # max_tokens: o-series uses max_completion_tokens
    if "max_tokens" in body:
        key = "max_completion_tokens" if is_o_series(backend_model) else "max_tokens"
        oai[key] = body["max_tokens"]

    if "temperature" in body:
        oai["temperature"] = body["temperature"]
    if "top_p" in body:
        oai["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        oai["stop"] = body["stop_sequences"]

    stream = body.get("stream", False)
    oai["stream"] = stream
    if stream:
        oai["stream_options"] = {"include_usage": True}

    # --- thinking / reasoning_effort ---
    # Priority: top-level effort > output_config.effort > thinking.budget_tokens > default
    thinking = body.get("thinking")
    output_config = body.get("output_config") or {}
    explicit_effort = body.get("effort") or output_config.get("effort")  # "max"/"xhigh"/"high"/"medium"/"low"
    if thinking:
        if thinking.get("type") == "adaptive":
            # Respect explicit effort; default is "high" per Anthropic docs
            oai["reasoning_effort"] = _normalize_effort(explicit_effort or "high", max_reasoning_effort)
        else:
            budget = thinking.get("budget_tokens", 8192)
            oai["reasoning_effort"] = _normalize_effort(
                explicit_effort or _reasoning_effort_from_budget(budget), max_reasoning_effort
            )

    # --- tools (skip Anthropic server tools like web_search, computer_use, etc.) ---
    tools = body.get("tools")
    if tools:
        filtered = [t for t in tools if not _is_anthropic_server_tool(t)]
        if filtered:
            oai["tools"] = [_anthropic_tool_to_openai(t) for t in filtered]
            tool_choice = body.get("tool_choice")
            if tool_choice:
                oai["tool_choice"] = _convert_tool_choice(tool_choice)

    return oai


def _user_blocks_to_openai(blocks: List[Dict]) -> List[Dict[str, Any]]:
    """Convert user content blocks. tool_result blocks become separate tool messages.

    cache_control is Anthropic-specific and stripped from all content items.
    """
    openai_content: List[Any] = []
    tool_messages: List[Dict[str, Any]] = []

    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            openai_content.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            openai_content.append(_image_block_to_openai(block))
        elif btype == "audio":
            converted = _audio_block_to_openai(block)
            if converted is not None:
                openai_content.append(converted)
        elif btype == "video":
            converted = _video_block_to_openai(block)
            if converted is not None:
                openai_content.append(converted)
        elif btype == "tool_result":
            tool_messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": _tool_result_content(block),
            })
        # document blocks: skip (OpenAI doesn't support PDF natively)

    # OpenAI requires tool messages to immediately follow the assistant that
    # issued the tool_calls.  Any user text in the same Anthropic message must
    # come AFTER all the tool messages, not before.
    result: List[Dict[str, Any]] = list(tool_messages)
    if openai_content:
        # Use a plain string for a single text block (most compatible); array for mixed content.
        if len(openai_content) == 1 and openai_content[0].get("type") == "text":
            result.append({"role": "user", "content": openai_content[0]["text"]})
        else:
            result.append({"role": "user", "content": openai_content})
    return result


def _assistant_blocks_to_openai(
    blocks: List[Dict], use_reasoning_content: bool = False
) -> List[Dict[str, Any]]:
    """Convert assistant content blocks (text + tool_use).

    When *use_reasoning_content* is True (DeepSeek mode):
    - thinking blocks are passed as ``reasoning_content`` on the message.
    - If no thinking block precedes a tool_use, inject placeholder ``"tool call"``.
    """
    thinking_text: Optional[str] = None
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            thinking_text = block.get("thinking", "")
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
        # redacted_thinking: silently skip
        # server_tool_use / web_search_tool_result: Anthropic server-executed; no OpenAI equivalent

    msg: Dict[str, Any] = {"role": "assistant"}
    if text_parts:
        msg["content"] = "\n".join(text_parts)
    elif not tool_calls:
        # Interrupted message (e.g. only thinking was received, no text yet).
        # Use empty string; null/None is only valid for tool-only assistant messages
        # and many providers reject it otherwise.
        msg["content"] = ""
    else:
        msg["content"] = None  # tool-only message: null is the OpenAI standard
    if tool_calls:
        msg["tool_calls"] = tool_calls
    # Always preserve actual thinking content as reasoning_content so that any
    # provider requiring it (DeepSeek, Moonshot, etc.) receives it back on
    # subsequent turns.  The "inject placeholder" branch is DeepSeek-specific.
    if thinking_text is not None:
        msg["reasoning_content"] = thinking_text
    elif use_reasoning_content and tool_calls:
        msg["reasoning_content"] = "tool call"
    return [msg]


def _image_block_to_openai(block: Dict) -> Dict[str, Any]:
    source = block.get("source", {})
    if source.get("type") == "base64":
        url = f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"
    else:
        url = source.get("url", "")
    return {"type": "image_url", "image_url": {"url": url}}


def _normalize_audio_format(fmt: str) -> str:
    """Normalise an audio format string to a value OpenAI's ``input_audio`` accepts.

    ``mimetypes.guess_type`` returns non-standard subtypes like ``"x-wav"``,
    ``"x-flac"``, ``"x-aiff"`` — OpenAI only accepts canonical forms.
    """
    # Strip the "x-" prefix commonly used for unofficial MIME subtypes.
    if fmt.startswith("x-"):
        fmt = fmt[2:]
    # Well-known remappings.
    return {
        "mp4a-latm": "mp4",
        "mpeg": "mp3",
        "basic": "au",  # audio/basic = AU format
    }.get(fmt, fmt)


def _audio_block_to_openai(block: Dict) -> Optional[Dict[str, Any]]:
    """Convert an Anthropic audio content block to OpenAI input_audio format.

    Returns None if the audio source can't be represented in OpenAI format
    (e.g. URL sources, which OpenAI doesn't support for audio).
    """
    source = block.get("source", {})
    if source.get("type") == "base64":
        mt = source.get("media_type", "audio/mp3")
        fmt = mt.split("/")[-1] if "/" in mt else mt
        fmt = _normalize_audio_format(fmt)
        return {"type": "input_audio", "input_audio": {
            "data": source.get("data", ""),
            "format": fmt,
        }}
    # OpenAI input_audio doesn't support URL sources
    return None


def _video_block_to_openai(block: Dict) -> Optional[Dict[str, Any]]:
    """Convert an Anthropic video content block to an OpenAI-compatible format.

    Uses the ``video_url`` content type (OpenRouter / OpenAI-compatible).
    Supports both base64 data URIs and direct URLs.
    """
    source = block.get("source", {})
    if source.get("type") == "base64":
        mt = source.get("media_type", "video/mp4")
        url = f"data:{mt};base64,{source.get('data', '')}"
        return {"type": "video_url", "video_url": {"url": url}}
    url = source.get("url", "")
    if url:
        return {"type": "video_url", "video_url": {"url": url}}
    return None


def _tool_result_content(block: Dict) -> str:
    """Serialise tool_result content to a string for OpenAI tool messages."""
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(json.dumps(item))
        return "\n".join(parts)
    return json.dumps(content)


def _anthropic_tool_to_openai(tool: Dict) -> Dict[str, Any]:
    # cache_control is Anthropic-specific; strip it from the OpenAI tool definition.
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": clean_schema(tool.get("input_schema", {})),
        },
    }


def _convert_tool_choice(tc: Dict) -> Any:
    tc_type = tc.get("type")
    if tc_type == "auto":
        return "auto"
    if tc_type == "none":
        return "none"
    if tc_type == "any":
        return "required"
    if tc_type == "tool":
        return {"type": "function", "function": {"name": tc.get("name", "")}}
    return "auto"


# ---------------------------------------------------------------------------
# OpenAI → Anthropic  (non-streaming response)
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def openai_to_anthropic_response(body: Dict[str, Any], original_model: str) -> Dict[str, Any]:
    """Convert an OpenAI chat completions response to Anthropic Messages format."""
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = _FINISH_REASON_MAP.get(finish_reason, "end_turn")

    content: List[Dict[str, Any]] = []

    # reasoning_content (o-series) → thinking block (before text)
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning, "signature": ""})

    # text content
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    # refusal → text block
    refusal = message.get("refusal")
    if refusal:
        content.append({"type": "text", "text": refusal})

    # tool_calls
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, ValueError):
            _logger.warning("Failed to parse tool call arguments for %s: %s", fn.get("name", "?"), fn.get("arguments", "")[:200])
            args = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
            "name": fn.get("name", ""),
            "input": args,
        })

    # legacy function_call (old OpenAI API)
    if not content and (fc := message.get("function_call")):
        try:
            args = json.loads(fc.get("arguments", "{}"))
        except (json.JSONDecodeError, ValueError):
            _logger.warning("Failed to parse legacy function_call arguments for %s: %s", fc.get("name", "?"), fc.get("arguments", "")[:200])
            args = {}
        content.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:12]}",
            "name": fc.get("name", ""),
            "input": args,
        })

    usage = body.get("usage") or {}
    # OpenAI's prompt_tokens includes cached tokens in the total.
    # Anthropic separates them: input_tokens = non-cached only.
    cache_read: Optional[int] = usage.get("cache_read_input_tokens")
    if cache_read is None:
        cache_read = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    usage_out: Dict[str, Any] = {
        "input_tokens": max(0, usage.get("prompt_tokens", 0) - (cache_read or 0)),
        "output_tokens": usage.get("completion_tokens", 0),
    }
    if cache_read is not None:
        usage_out["cache_read_input_tokens"] = cache_read
    cache_create = usage.get("cache_creation_input_tokens")
    if cache_create is not None:
        usage_out["cache_creation_input_tokens"] = cache_create

    return {
        "id": body.get("id", f"msg_{uuid.uuid4().hex[:12]}"),
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_out,
    }


# ---------------------------------------------------------------------------
# OpenAI SSE → Anthropic SSE  (streaming)
# ---------------------------------------------------------------------------

async def stream_openai_to_anthropic(
    source: AsyncIterator[bytes],
    original_model: str,
    msg_id: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """Transform OpenAI Server-Sent Events into Anthropic Server-Sent Events."""
    if msg_id is None:
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"

    yield _sse_bytes("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": original_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield _sse_bytes("ping", {"type": "ping"})

    # --- state ---
    text_block_open = False
    text_block_index = 0
    thinking_block_open = False
    thinking_block_index = 0
    # oai tool index → ToolState
    tool_states: Dict[int, Dict[str, Any]] = {}
    open_tool_indices: Set[int] = set()   # anthropic indices of started+open tool blocks
    # legacy function_call tracking
    legacy_fn_block_index: Optional[int] = None
    next_block_index = 0
    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens: Optional[int] = None
    cache_create_tokens: Optional[int] = None
    partial: bytes = b""

    async for chunk in source:
        partial += chunk
        while b"\n" in partial:
            line, partial = partial.split(b"\n", 1)
            line = line.rstrip(b"\r")
            # SSE comment (e.g. ": keep-alive") — forward as Anthropic ping so the
            # downstream client doesn't time out during long provider queue waits.
            if line.startswith(b":"):
                yield _sse_bytes("ping", {"type": "ping"})
                continue
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload.strip() == b"[DONE]":
                continue
            try:
                data: Dict[str, Any] = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue

            # --- upstream error in stream (e.g. content-policy block, quota, etc.) ---
            if error_obj := data.get("error"):
                err_type = error_obj.get("type", "api_error") if isinstance(error_obj, dict) else "api_error"
                err_msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
                # Close any open blocks before emitting the error event so the
                # client doesn't get stuck with half-open content blocks.
                if thinking_block_open:
                    for ev in _thinking_close_events(thinking_block_index):
                        yield ev
                if text_block_open:
                    yield _sse_bytes("content_block_stop", {
                        "type": "content_block_stop", "index": text_block_index,
                    })
                for ant_idx in sorted(open_tool_indices):
                    yield _sse_bytes("content_block_stop", {
                        "type": "content_block_stop", "index": ant_idx,
                    })
                yield _sse_bytes("error", {
                    "type": "error",
                    "error": {"type": err_type, "message": err_msg},
                })
                return

            choices = data.get("choices") or []
            choice = choices[0] if choices else {}
            delta = choice.get("delta", {})
            finish = choice.get("finish_reason")

            if finish:
                stop_reason = _FINISH_REASON_MAP.get(finish, "end_turn")

            # --- usage (comes in last chunk with stream_options) ---
            if usage_data := data.get("usage"):
                raw_prompt = usage_data.get("prompt_tokens", input_tokens)
                output_tokens = usage_data.get("completion_tokens", output_tokens)
                raw_cache_read = (
                    usage_data.get("cache_read_input_tokens")
                    if usage_data.get("cache_read_input_tokens") is not None
                    else (usage_data.get("prompt_tokens_details") or {}).get("cached_tokens")
                )
                if raw_cache_read is not None:
                    cache_read_tokens = raw_cache_read
                # OpenAI's prompt_tokens includes cached; subtract to get non-cached only
                input_tokens = max(0, raw_prompt - (raw_cache_read or 0))
                raw_cache_create = usage_data.get("cache_creation_input_tokens")
                if raw_cache_create is not None:
                    cache_create_tokens = raw_cache_create

            # --- reasoning / thinking delta (o-series) ---
            reasoning_text = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning_text:
                # Skip reasoning if tool blocks are already open.  Some providers
                # stream tool_calls first and reasoning_content afterward; opening a
                # thinking block at that point would create overlapping open blocks
                # (protocol violation) and show thinking *after* tools in Claude Code.
                # Post-tool reasoning is discarded; pre-tool reasoning works normally.
                if not open_tool_indices:
                    if not thinking_block_open:
                        if text_block_open:
                            yield _sse_bytes("content_block_stop", {
                                "type": "content_block_stop", "index": text_block_index,
                            })
                            text_block_open = False
                        yield _sse_bytes("content_block_start", {
                            "type": "content_block_start",
                            "index": next_block_index,
                            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                        })
                        thinking_block_index = next_block_index
                        next_block_index += 1
                        thinking_block_open = True
                    yield _sse_bytes("content_block_delta", {
                        "type": "content_block_delta",
                        "index": thinking_block_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning_text},
                    })

            # --- text delta ---
            if text := delta.get("content"):
                if thinking_block_open:
                    # Only close thinking for real content; skip whitespace-only
                    # chunks that some providers insert between reasoning segments
                    # (e.g. a bare "\n"), which would otherwise split one logical
                    # thinking block into multiple consecutive blocks.
                    if not text.strip():
                        text = ""
                    else:
                        for ev in _thinking_close_events(thinking_block_index):
                            yield ev
                        thinking_block_open = False
                if text:
                    if not text_block_open:
                        yield _sse_bytes("content_block_start", {
                            "type": "content_block_start",
                            "index": next_block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                        text_block_index = next_block_index
                        next_block_index += 1
                        text_block_open = True
                    yield _sse_bytes("content_block_delta", {
                        "type": "content_block_delta",
                        "index": text_block_index,
                        "delta": {"type": "text_delta", "text": text},
                    })

            # --- refusal delta (content_filter responses) ---
            if refusal_text := delta.get("refusal"):
                if thinking_block_open:
                    for ev in _thinking_close_events(thinking_block_index):
                        yield ev
                    thinking_block_open = False
                if not text_block_open:
                    yield _sse_bytes("content_block_start", {
                        "type": "content_block_start",
                        "index": next_block_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                    text_block_index = next_block_index
                    next_block_index += 1
                    text_block_open = True
                yield _sse_bytes("content_block_delta", {
                    "type": "content_block_delta",
                    "index": text_block_index,
                    "delta": {"type": "text_delta", "text": refusal_text},
                })

            # --- tool_calls deltas ---
            for tc in delta.get("tool_calls") or []:
                oai_idx: int = tc.get("index", 0)

                if oai_idx not in tool_states:
                    # Create entry WITHOUT reserving an anthropic index yet.
                    # We defer index assignment to the moment we actually emit
                    # content_block_start (when both id and name are known).
                    # This means any interleaved reasoning_content that arrives
                    # before the tool is fully identified will claim a lower index
                    # and display *before* the tool call in Claude Code.
                    tool_states[oai_idx] = {
                        "oai_index": oai_idx,
                        "anthropic_index": None,
                        "id": "",
                        "name": "",
                        "started": False,
                        "pending_args": "",
                    }

                state = tool_states[oai_idx]
                fn = tc.get("function") or {}
                if tc.get("id"):
                    state["id"] = tc["id"]
                if fn.get("name"):
                    state["name"] = fn["name"]

                # Defer block start until both id AND name are known
                if not state["started"] and state["id"] and state["name"]:
                    # Close open text/thinking blocks right before opening the tool
                    if thinking_block_open:
                        for ev in _thinking_close_events(thinking_block_index):
                            yield ev
                        thinking_block_open = False
                    if text_block_open:
                        yield _sse_bytes("content_block_stop", {
                            "type": "content_block_stop", "index": text_block_index,
                        })
                        text_block_open = False
                    # Close any tool blocks that were opened for earlier oai_indices.
                    # Most providers (including DeepSeek) stream tool calls sequentially:
                    # all arguments for tool[N] arrive before tool[N+1]'s id/name,
                    # so when we see a new tool starting, the previous one is complete.
                    # Closing here keeps Anthropic content blocks non-overlapping.
                    for prev_ant_idx in sorted(open_tool_indices):
                        yield _sse_bytes("content_block_stop", {
                            "type": "content_block_stop", "index": prev_ant_idx,
                        })
                    open_tool_indices.clear()
                    state["anthropic_index"] = next_block_index
                    next_block_index += 1
                    state["started"] = True
                    yield _sse_bytes("content_block_start", {
                        "type": "content_block_start",
                        "index": state["anthropic_index"],
                        "content_block": {
                            "type": "tool_use",
                            "id": state["id"],
                            "name": state["name"],
                            "input": {},
                        },
                    })
                    open_tool_indices.add(state["anthropic_index"])
                    if state["pending_args"]:
                        yield _sse_bytes("content_block_delta", {
                            "type": "content_block_delta",
                            "index": state["anthropic_index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": state["pending_args"],
                            },
                        })
                        state["pending_args"] = ""

                args_delta: str = fn.get("arguments", "")
                if args_delta:
                    if state["started"]:
                        yield _sse_bytes("content_block_delta", {
                            "type": "content_block_delta",
                            "index": state["anthropic_index"],
                            "delta": {"type": "input_json_delta", "partial_json": args_delta},
                        })
                    else:
                        state["pending_args"] += args_delta

            # --- legacy function_call ---
            if fc_delta := delta.get("function_call"):
                if thinking_block_open:
                    for ev in _thinking_close_events(thinking_block_index):
                        yield ev
                    thinking_block_open = False
                if text_block_open:
                    yield _sse_bytes("content_block_stop", {
                        "type": "content_block_stop", "index": text_block_index,
                    })
                    text_block_open = False
                if legacy_fn_block_index is None:
                    legacy_fn_block_index = next_block_index
                    next_block_index += 1
                    yield _sse_bytes("content_block_start", {
                        "type": "content_block_start",
                        "index": legacy_fn_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": "",
                            "name": fc_delta.get("name", ""),
                            "input": {},
                        },
                    })
                    open_tool_indices.add(legacy_fn_block_index)
                if args := fc_delta.get("arguments"):
                    yield _sse_bytes("content_block_delta", {
                        "type": "content_block_delta",
                        "index": legacy_fn_block_index,
                        "delta": {"type": "input_json_delta", "partial_json": args},
                    })

            # --- finish_reason: flush any unstarted tool blocks ---
            if finish:
                # Close any open thinking/text before flushing unstarted tools
                if thinking_block_open:
                    for ev in _thinking_close_events(thinking_block_index):
                        yield ev
                    thinking_block_open = False
                if text_block_open:
                    yield _sse_bytes("content_block_stop", {
                        "type": "content_block_stop", "index": text_block_index,
                    })
                    text_block_open = False
                for state in sorted(tool_states.values(), key=lambda s: s["oai_index"]):
                    if state["started"]:
                        continue
                    if not state["id"] and not state["name"] and not state["pending_args"]:
                        continue
                    # Close any previously-open tool blocks before starting this one.
                    for prev_ant_idx in sorted(open_tool_indices):
                        yield _sse_bytes("content_block_stop", {
                            "type": "content_block_stop", "index": prev_ant_idx,
                        })
                    open_tool_indices.clear()
                    state["anthropic_index"] = next_block_index
                    next_block_index += 1
                    fallback_id = state["id"] or f"tool_call_{state['anthropic_index']}"
                    fallback_name = state["name"] or "unknown_tool"
                    state["started"] = True
                    yield _sse_bytes("content_block_start", {
                        "type": "content_block_start",
                        "index": state["anthropic_index"],
                        "content_block": {
                            "type": "tool_use",
                            "id": fallback_id,
                            "name": fallback_name,
                            "input": {},
                        },
                    })
                    open_tool_indices.add(state["anthropic_index"])
                    if state["pending_args"]:
                        yield _sse_bytes("content_block_delta", {
                            "type": "content_block_delta",
                            "index": state["anthropic_index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": state["pending_args"],
                            },
                        })

    # --- close any still-open blocks ---
    if thinking_block_open:
        for ev in _thinking_close_events(thinking_block_index):
            yield ev
    if text_block_open:
        yield _sse_bytes("content_block_stop", {
            "type": "content_block_stop", "index": text_block_index,
        })
    for ant_idx in sorted(open_tool_indices):
        yield _sse_bytes("content_block_stop", {
            "type": "content_block_stop", "index": ant_idx,
        })

    # --- message_delta + message_stop ---
    usage_delta: Dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cache_read_tokens is not None:
        usage_delta["cache_read_input_tokens"] = cache_read_tokens
    if cache_create_tokens is not None:
        usage_delta["cache_creation_input_tokens"] = cache_create_tokens

    yield _sse_bytes("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": usage_delta,
    })
    yield _sse_bytes("message_stop", {"type": "message_stop"})
