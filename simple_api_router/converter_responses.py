"""Anthropic Messages ↔ OpenAI Responses API conversion.

Handles:
- Request: Anthropic Messages → OpenAI Responses API
- Response: OpenAI Responses API → Anthropic Messages
- Streaming: OpenAI Responses API SSE → Anthropic SSE
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from .converter import (
    _normalize_effort,
    _reasoning_effort_from_budget,
    clean_schema,
    sanitize_system_text,
    strip_private_params,
)
from .converter_openai import _tool_result_content
from .converter_utils import (
    is_anthropic_server_tool as _is_anthropic_server_tool,
    fold_midstream_system_into_user as _fold_midstream_system_into_user,
    sse as _sse_bytes,
    thinking_close_events as _thinking_close_events,
)
from .logger import get_logger

_logger = get_logger("converter")


# ---------------------------------------------------------------------------
# Anthropic → OpenAI Responses API  (request)
# ---------------------------------------------------------------------------

def _responses_tool_choice(tc: Dict) -> Any:
    tc_type = tc.get("type")
    if tc_type == "auto":
        return "auto"
    if tc_type == "none":
        return "none"
    if tc_type == "any":
        return "required"
    if tc_type == "tool":
        return {"type": "function", "name": tc.get("name", "")}
    return "auto"


def _messages_to_responses_input(messages: List[Dict]) -> List[Dict[str, Any]]:
    """Convert Anthropic messages to the Responses API ``input`` array."""
    result: List[Dict[str, Any]] = []

    for msg in _fold_midstream_system_into_user(messages):
        role = msg.get("role", "user")
        content = msg.get("content", [])

        if isinstance(content, str):
            item_type = "output_text" if role == "assistant" else "input_text"
            result.append({"role": role, "content": [{"type": item_type, "text": content}]})
            continue

        current_parts: List[Dict[str, Any]] = []
        result_start_idx = len(result)  # track how many items belong to this message

        for block in content:
            btype = block.get("type")
            if btype == "text":
                item_type = "output_text" if role == "assistant" else "input_text"
                current_parts.append({"type": item_type, "text": block.get("text", "")})
            elif btype == "image":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    url = f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"
                else:
                    url = source.get("url", "")
                current_parts.append({"type": "input_image", "image_url": url})
            elif btype == "audio":
                source = block.get("source", {})
                mt = source.get("media_type", "audio/mp3")
                fmt = mt.split("/")[-1] if "/" in mt else mt
                current_parts.append({"type": "input_audio", "audio": {
                    "data": source.get("data", ""),
                    "format": fmt,
                }})
            elif btype == "video":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    mt = source.get("media_type", "video/mp4")
                    url = f"data:{mt};base64,{source.get('data', '')}"
                    current_parts.append({"type": "input_video", "video_url": url})
                else:
                    url = source.get("url", "")
                    if url:
                        current_parts.append({"type": "input_video", "video_url": url})
            elif btype == "tool_use":
                if current_parts:
                    result.append({"role": role, "content": current_parts})
                    current_parts = []
                result.append({
                    "type": "function_call",
                    "call_id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                })
            elif btype == "tool_result":
                if current_parts:
                    result.append({"role": role, "content": current_parts})
                    current_parts = []
                result.append({
                    "type": "function_call_output",
                    "call_id": block.get("tool_use_id", ""),
                    "output": _tool_result_content(block),
                })
            # thinking / redacted_thinking: skip

        if current_parts:
            result.append({"role": role, "content": current_parts})
        elif role == "assistant" and len(result) == result_start_idx:
            # No Responses-API items were emitted for this assistant turn at all:
            # content was empty (interrupted before first block) or contained only
            # thinking/redacted_thinking.  Emit an empty placeholder so the
            # conversation keeps proper structure — the model must not see two
            # consecutive user messages with no assistant turn between them.
            result.append({"role": "assistant", "content": [{"type": "output_text", "text": ""}]})

    return result


def anthropic_to_responses_request(
    body: Dict[str, Any],
    backend_model: str,
    max_reasoning_effort: str = "xhigh",
) -> Dict[str, Any]:
    """Convert an Anthropic /v1/messages body to OpenAI Responses API format."""
    body = strip_private_params(body)
    result: Dict[str, Any] = {"model": backend_model}

    # system → instructions
    system = body.get("system")
    if system:
        if isinstance(system, str):
            inst = sanitize_system_text(system)
        elif isinstance(system, list):
            parts = [
                sanitize_system_text(b.get("text", ""))
                for b in system
                if b.get("type") == "text"
            ]
            inst = "\n\n".join(p for p in parts if p.strip())
        else:
            inst = ""
        if inst.strip():
            result["instructions"] = inst

    # messages → input
    if messages := body.get("messages"):
        result["input"] = _messages_to_responses_input(messages)

    # max_tokens → max_output_tokens
    if "max_tokens" in body:
        result["max_output_tokens"] = body["max_tokens"]

    for key in ("temperature", "top_p"):
        if key in body:
            result[key] = body[key]

    result["stream"] = body.get("stream", False)

    # thinking → reasoning
    # Priority: top-level effort > output_config.effort > thinking.budget_tokens > default
    thinking = body.get("thinking")
    output_config = body.get("output_config") or {}
    explicit_effort = body.get("effort") or output_config.get("effort")
    if thinking:
        thinking_type = thinking.get("type")
        if thinking_type == "disabled":
            pass  # omit reasoning — thinking is off
        elif thinking_type == "adaptive":
            effort = _normalize_effort(explicit_effort or "high", max_reasoning_effort)
            result["reasoning"] = {"effort": effort, "summary": "auto"}
        else:
            effort = _normalize_effort(
                explicit_effort or _reasoning_effort_from_budget(thinking.get("budget_tokens", 8192)),
                max_reasoning_effort,
            )
            result["reasoning"] = {"effort": effort, "summary": "auto"}

    # tools (skip Anthropic server tools like web_search, computer_use, etc.)
    tools = body.get("tools")
    if tools:
        filtered = [t for t in tools if not _is_anthropic_server_tool(t)]
        if filtered:
            result["tools"] = [
                {
                    "type": "function",
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": clean_schema(t.get("input_schema", {})),
                }
                for t in filtered
            ]
            if tc := body.get("tool_choice"):
                result["tool_choice"] = _responses_tool_choice(tc)

    return result


# ---------------------------------------------------------------------------
# OpenAI Responses API → Anthropic  (non-streaming response)
# ---------------------------------------------------------------------------

def _responses_stop_reason(
    status: str, has_tool_use: bool, incomplete_reason: Optional[str]
) -> str:
    if status == "completed":
        return "tool_use" if has_tool_use else "end_turn"
    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return "max_tokens"
        return "end_turn"
    return "end_turn"


def _build_anthropic_usage_from_responses(usage: Dict[str, Any]) -> Dict[str, Any]:
    # OpenAI Responses API: input_tokens is the total (includes cached).
    # Anthropic: input_tokens is non-cached only.
    cache_read: Optional[int] = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    if cache_read is None:
        cache_read = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    out: Dict[str, Any] = {
        "input_tokens": max(0, usage.get("input_tokens", 0) - (cache_read or 0)),
        "output_tokens": usage.get("output_tokens", 0),
    }
    if cache_read is not None:
        out["cache_read_input_tokens"] = cache_read
    return out


def responses_to_anthropic_response(
    body: Dict[str, Any],
    original_model: str,
    msg_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert an OpenAI Responses API response to Anthropic Messages format."""
    if msg_id is None:
        msg_id = body.get("id", f"msg_{uuid.uuid4().hex[:12]}")

    output = body.get("output") or []
    content: List[Dict[str, Any]] = []
    has_tool_use = False

    for item in output:
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content") or []:
                if part.get("type") in ("output_text", "refusal"):
                    text = part.get("text", "")
                    if text:
                        content.append({"type": "text", "text": text})
        elif item_type == "function_call":
            has_tool_use = True
            try:
                args = json.loads(item.get("arguments", "{}"))
            except (json.JSONDecodeError, ValueError):
                _logger.warning("Failed to parse Responses API function_call arguments for %s: %s", item.get("name", "?"), item.get("arguments", "")[:200])
                args = {}
            content.append({
                "type": "tool_use",
                "id": item.get("call_id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": item.get("name", ""),
                "input": args,
            })
        elif item_type == "reasoning":
            summary = item.get("summary") or []
            texts = [s.get("text", "") for s in summary if s.get("type") == "summary_text"]
            joined = "\n\n".join(t for t in texts if t)
            if joined:
                content.append({"type": "thinking", "thinking": joined, "signature": ""})

    status = body.get("status", "completed")
    incomplete_reason = (body.get("incomplete_details") or {}).get("reason")
    stop_reason = _responses_stop_reason(status, has_tool_use, incomplete_reason)
    usage_out = _build_anthropic_usage_from_responses(body.get("usage") or {})

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_out,
    }


# ---------------------------------------------------------------------------
# OpenAI Responses API SSE → Anthropic SSE  (streaming)
# ---------------------------------------------------------------------------

async def stream_responses_to_anthropic(
    source: AsyncIterator[bytes],
    original_model: str,
    msg_id: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """Transform OpenAI Responses API SSE events into Anthropic SSE events."""
    if msg_id is None:
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"

    next_index = 0
    # (output_index, content_index) → anthropic block index (for text/reasoning parts)
    block_map: Dict[Tuple[int, int], int] = {}
    # output_index → anthropic block index (for function_call items)
    tool_map: Dict[int, int] = {}
    open_blocks: Set[int] = set()
    thinking_blocks: Set[int] = set()  # ant_indices that are thinking blocks
    has_tool_use = False

    pending = ""

    async for chunk in source:
        pending += chunk.decode(errors="replace")

        while "\n\n" in pending:
            event_block, pending = pending.split("\n\n", 1)
            data_str: Optional[str] = None
            is_comment_only = True
            for line in event_block.splitlines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    is_comment_only = False
                elif line and not line.startswith(":"):
                    is_comment_only = False

            # SSE comment-only block (e.g. ": keep-alive") — forward as Anthropic ping.
            if is_comment_only and event_block.strip():
                yield _sse_bytes("ping", {"type": "ping"})
                continue

            if not data_str or data_str == "[DONE]":
                continue

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            etype: str = data.get("type", "")
            # Some events wrap the response object under data["response"]
            resp_obj: Dict[str, Any] = data.get("response") or data

            if etype == "response.created":
                usage = resp_obj.get("usage") or {}
                yield _sse_bytes("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": resp_obj.get("id", msg_id),
                        "type": "message",
                        "role": "assistant",
                        "model": original_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": _build_anthropic_usage_from_responses(usage),
                    },
                })
                yield _sse_bytes("ping", {"type": "ping"})

            elif etype == "response.output_item.added":
                item = data.get("item") or {}
                if item.get("type") == "function_call":
                    has_tool_use = True
                    output_index: int = data.get("output_index", 0)
                    ant_index = next_index
                    next_index += 1
                    tool_map[output_index] = ant_index
                    open_blocks.add(ant_index)
                    yield _sse_bytes("content_block_start", {
                        "type": "content_block_start",
                        "index": ant_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": item.get("call_id", f"toolu_{uuid.uuid4().hex[:12]}"),
                            "name": item.get("name", ""),
                            "input": {},
                        },
                    })

            elif etype == "response.content_part.added":
                part = data.get("part") or {}
                if part.get("type") in ("output_text", "refusal"):
                    oi: int = data.get("output_index", 0)
                    ci: int = data.get("content_index", 0)
                    key = (oi, ci)
                    ant_index = next_index
                    next_index += 1
                    block_map[key] = ant_index
                    open_blocks.add(ant_index)
                    yield _sse_bytes("content_block_start", {
                        "type": "content_block_start",
                        "index": ant_index,
                        "content_block": {"type": "text", "text": ""},
                    })

            elif etype in ("response.output_text.delta", "response.refusal.delta"):
                oi = data.get("output_index", 0)
                ci = data.get("content_index", 0)
                ant_index = block_map.get((oi, ci))
                if ant_index is not None:
                    delta = data.get("delta", "")
                    yield _sse_bytes("content_block_delta", {
                        "type": "content_block_delta",
                        "index": ant_index,
                        "delta": {"type": "text_delta", "text": delta if isinstance(delta, str) else ""},
                    })

            elif etype == "response.content_part.done":
                oi = data.get("output_index", 0)
                ci = data.get("content_index", 0)
                ant_index = block_map.get((oi, ci))
                if ant_index is not None and ant_index in open_blocks:
                    open_blocks.discard(ant_index)
                    yield _sse_bytes("content_block_stop", {
                        "type": "content_block_stop", "index": ant_index,
                    })

            elif etype == "response.function_call_arguments.delta":
                oi = data.get("output_index", 0)
                ant_index = tool_map.get(oi)
                if ant_index is not None:
                    delta = data.get("delta", "")
                    yield _sse_bytes("content_block_delta", {
                        "type": "content_block_delta",
                        "index": ant_index,
                        "delta": {"type": "input_json_delta", "partial_json": delta},
                    })

            elif etype == "response.function_call_arguments.done":
                oi = data.get("output_index", 0)
                ant_index = tool_map.get(oi)
                if ant_index is not None and ant_index in open_blocks:
                    open_blocks.discard(ant_index)
                    yield _sse_bytes("content_block_stop", {
                        "type": "content_block_stop", "index": ant_index,
                    })

            elif etype == "response.reasoning.delta":
                oi = data.get("output_index", 0)
                ci = data.get("content_index", 0)
                key = (oi, ci)
                if key not in block_map:
                    ant_index = next_index
                    next_index += 1
                    block_map[key] = ant_index
                    open_blocks.add(ant_index)
                    thinking_blocks.add(ant_index)
                    yield _sse_bytes("content_block_start", {
                        "type": "content_block_start",
                        "index": ant_index,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                    })
                else:
                    ant_index = block_map[key]
                delta = data.get("delta") or {}
                text = delta.get("text", "") if isinstance(delta, dict) else str(delta)
                if text:
                    yield _sse_bytes("content_block_delta", {
                        "type": "content_block_delta",
                        "index": ant_index,
                        "delta": {"type": "thinking_delta", "thinking": text},
                    })

            elif etype == "response.reasoning.done":
                oi = data.get("output_index", 0)
                ci = data.get("content_index", 0)
                ant_index = block_map.get((oi, ci))
                if ant_index is not None and ant_index in open_blocks:
                    open_blocks.discard(ant_index)
                    for ev in _thinking_close_events(ant_index):
                        yield ev

            elif etype == "response.completed":
                # Close any remaining open blocks
                for ant_idx in sorted(open_blocks):
                    if ant_idx in thinking_blocks:
                        for ev in _thinking_close_events(ant_idx):
                            yield ev
                    else:
                        yield _sse_bytes("content_block_stop", {
                            "type": "content_block_stop", "index": ant_idx,
                        })
                open_blocks.clear()

                status = resp_obj.get("status", "completed")
                incomplete_reason = (resp_obj.get("incomplete_details") or {}).get("reason")
                stop_reason = _responses_stop_reason(status, has_tool_use, incomplete_reason)
                usage_out = _build_anthropic_usage_from_responses(resp_obj.get("usage") or {})

                yield _sse_bytes("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": usage_out,
                })
                yield _sse_bytes("message_stop", {"type": "message_stop"})
                return

    # Post-loop safety net: if response.completed never arrived (provider dropped
    # the connection or sent [DONE] prematurely), close any open blocks and emit
    # the terminal events so the client doesn't hang.
    for ant_idx in sorted(open_blocks):
        if ant_idx in thinking_blocks:
            for ev in _thinking_close_events(ant_idx):
                yield ev
        else:
            yield _sse_bytes("content_block_stop", {
                "type": "content_block_stop", "index": ant_idx,
            })
    open_blocks.clear()

    fallback_stop_reason = "tool_use" if has_tool_use else "end_turn"
    yield _sse_bytes("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": fallback_stop_reason, "stop_sequence": None},
        "usage": _build_anthropic_usage_from_responses({}),
    })
    yield _sse_bytes("message_stop", {"type": "message_stop"})
