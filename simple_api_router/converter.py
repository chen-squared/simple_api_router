"""Full bidirectional conversion between Anthropic and OpenAI API formats.

Handles:
- Request: Anthropic Messages → OpenAI Chat Completions
- Response: OpenAI Chat Completions → Anthropic Messages
- Streaming: OpenAI SSE → Anthropic SSE (full event mapping including tool use)
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_O_SERIES_RE = re.compile(r"\bo[1-9](-|\b)|o4-mini|codex", re.IGNORECASE)
_REASONING_EFFORT_MODELS_RE = re.compile(
    r"\bo[1-9](-|\b)|o4-mini|gpt-5|codex|deepseek", re.IGNORECASE
)


def sanitize_system_text(text: str) -> str:
    """Strip lines containing x-anthropic-billing-header injected by Claude Code.

    Preserves all other content and surrounding whitespace faithfully.
    """
    lines = text.split("\n")
    filtered = [l for l in lines if "x-anthropic-billing-header:" not in l.lower()]
    return "\n".join(filtered)


def clean_schema(schema: Any) -> Any:
    """Recursively remove JSON Schema keys that some OpenAI providers reject (e.g. format: uri)."""
    if not isinstance(schema, dict):
        return schema
    result: Dict[str, Any] = {}
    for k, v in schema.items():
        # Drop "format" when it's a URI-type validator (not "date-time" etc which are fine)
        if k == "format" and isinstance(v, str) and v in ("uri", "uri-reference", "iri", "iri-reference"):
            continue
        if isinstance(v, dict):
            result[k] = clean_schema(v)
        elif isinstance(v, list):
            result[k] = [clean_schema(i) for i in v]
        else:
            result[k] = v
    return result


def is_o_series(model: str) -> bool:
    """Return True for OpenAI o1/o3/o4-mini family that uses max_completion_tokens."""
    return bool(_O_SERIES_RE.search(model))


def supports_reasoning_effort(model: str) -> bool:
    """Return True for models that accept reasoning_effort instead of thinking.budget_tokens."""
    return bool(_REASONING_EFFORT_MODELS_RE.search(model))


def _reasoning_effort_from_budget(budget_tokens: int) -> str:
    """Map Anthropic budget_tokens to OpenAI reasoning_effort.

    OpenAI (and DeepSeek, which maps xhigh→max internally) all accept:
    none / minimal / low / medium / high / xhigh.
    """
    if budget_tokens <= 1024:
        return "low"
    if budget_tokens <= 8192:
        return "medium"
    if budget_tokens <= 32000:
        return "high"
    return "xhigh"


def strip_private_params(body: Dict[str, Any]) -> Dict[str, Any]:
    """Remove top-level keys starting with '_' (private Claude Code params)."""
    return {k: v for k, v in body.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Anthropic → OpenAI  (request)
# ---------------------------------------------------------------------------

def anthropic_to_openai_request(
    body: Dict[str, Any],
    backend_model: str,
    use_reasoning_content: bool = False,
) -> Dict[str, Any]:
    """Convert an Anthropic /v1/messages request body to OpenAI chat completions format."""
    body = strip_private_params(body)
    messages: List[Dict[str, Any]] = []

    # --- system prompt ---
    system = body.get("system")
    if system:
        if isinstance(system, str):
            sanitized = sanitize_system_text(system)
            if sanitized.strip():
                messages.append({"role": "system", "content": sanitized})
        elif isinstance(system, list):
            for b in system:
                if b.get("type") != "text":
                    continue
                sanitized = sanitize_system_text(b.get("text", ""))
                if not sanitized.strip():
                    continue
                sys_msg: Dict[str, Any] = {"role": "system", "content": sanitized}
                if cc := b.get("cache_control"):
                    sys_msg["cache_control"] = cc
                messages.append(sys_msg)

    # --- conversation messages ---
    for msg in body.get("messages", []):
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
    thinking = body.get("thinking")
    if thinking and supports_reasoning_effort(backend_model):
        if thinking.get("type") == "adaptive":
            oai["reasoning_effort"] = "xhigh"
        else:
            budget = thinking.get("budget_tokens", 8192)
            oai["reasoning_effort"] = _reasoning_effort_from_budget(budget)

    # --- tools (filter BatchTool, clean schemas) ---
    tools = body.get("tools")
    if tools:
        filtered = [t for t in tools if t.get("type") != "BatchTool"]
        if filtered:
            oai["tools"] = [_anthropic_tool_to_openai(t) for t in filtered]
            tool_choice = body.get("tool_choice")
            if tool_choice:
                oai["tool_choice"] = _convert_tool_choice(tool_choice)

    return oai


def _user_blocks_to_openai(blocks: List[Dict]) -> List[Dict[str, Any]]:
    """Convert user content blocks. tool_result blocks become separate tool messages."""
    openai_content: List[Any] = []
    tool_messages: List[Dict[str, Any]] = []
    has_cache_control = False

    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            item: Dict[str, Any] = {"type": "text", "text": block.get("text", "")}
            if cc := block.get("cache_control"):
                item["cache_control"] = cc
                has_cache_control = True
            openai_content.append(item)
        elif btype == "image":
            openai_content.append(_image_block_to_openai(block))
        elif btype == "tool_result":
            tool_messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": _tool_result_content(block),
            })
        # document blocks: skip (OpenAI doesn't support PDF natively)

    result: List[Dict[str, Any]] = []
    if openai_content:
        # Keep array form when cache_control is present or when there are mixed content types
        if not has_cache_control and len(openai_content) == 1 and openai_content[0].get("type") == "text":
            result.append({"role": "user", "content": openai_content[0]["text"]})
        else:
            result.append({"role": "user", "content": openai_content})
    result.extend(tool_messages)
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

    msg: Dict[str, Any] = {"role": "assistant"}
    msg["content"] = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if use_reasoning_content:
        if thinking_text is not None:
            msg["reasoning_content"] = thinking_text
        elif tool_calls:
            msg["reasoning_content"] = "tool call"
    return [msg]


def _image_block_to_openai(block: Dict) -> Dict[str, Any]:
    source = block.get("source", {})
    if source.get("type") == "base64":
        url = f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"
    else:
        url = source.get("url", "")
    return {"type": "image_url", "image_url": {"url": url}}


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
    item: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": clean_schema(tool.get("input_schema", {})),
        },
    }
    if cc := tool.get("cache_control"):
        item["cache_control"] = cc
    return item


def _convert_tool_choice(tc: Dict) -> Any:
    tc_type = tc.get("type")
    if tc_type == "auto":
        return "auto"
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
    "content_filter": "end_turn",
}


def openai_to_anthropic_response(body: Dict[str, Any], original_model: str) -> Dict[str, Any]:
    """Convert an OpenAI chat completions response to Anthropic Messages format."""
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = _FINISH_REASON_MAP.get(finish_reason, "end_turn")

    content: List[Dict[str, Any]] = []

    # reasoning_content (o-series) → thinking block (before text)
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})

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
            args = {}
        content.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:12]}",
            "name": fc.get("name", ""),
            "input": args,
        })

    usage = body.get("usage", {})
    # OpenAI's prompt_tokens includes cached tokens in the total.
    # Anthropic separates them: input_tokens = non-cached only.
    cache_read = (
        usage.get("cache_read_input_tokens")
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    )
    usage_out: Dict[str, Any] = {
        "input_tokens": max(0, usage.get("prompt_tokens", 0) - (cache_read or 0)),
        "output_tokens": usage.get("completion_tokens", 0),
    }
    if cache_read:
        usage_out["cache_read_input_tokens"] = cache_read
    cache_create = usage.get("cache_creation_input_tokens")
    if cache_create:
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
    """
    Transform OpenAI Server-Sent Events into Anthropic Server-Sent Events.

    Yields raw bytes suitable for streaming back to the client.
    """
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
                    yield _sse_bytes("content_block_stop", {
                        "type": "content_block_stop", "index": thinking_block_index,
                    })
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
                if not thinking_block_open:
                    if text_block_open:
                        yield _sse_bytes("content_block_stop", {
                            "type": "content_block_stop", "index": text_block_index,
                        })
                        text_block_open = False
                    yield _sse_bytes("content_block_start", {
                        "type": "content_block_start",
                        "index": next_block_index,
                        "content_block": {"type": "thinking", "thinking": ""},
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
                    yield _sse_bytes("content_block_stop", {
                        "type": "content_block_stop", "index": thinking_block_index,
                    })
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
                    "delta": {"type": "text_delta", "text": text},
                })

            # --- tool_calls deltas ---
            for tc in delta.get("tool_calls") or []:
                oai_idx: int = tc.get("index", 0)

                if oai_idx not in tool_states:
                    # Close open text/thinking blocks
                    if thinking_block_open:
                        yield _sse_bytes("content_block_stop", {
                            "type": "content_block_stop", "index": thinking_block_index,
                        })
                        thinking_block_open = False
                    if text_block_open:
                        yield _sse_bytes("content_block_stop", {
                            "type": "content_block_stop", "index": text_block_index,
                        })
                        text_block_open = False
                    tool_states[oai_idx] = {
                        "anthropic_index": next_block_index,
                        "id": "",
                        "name": "",
                        "started": False,
                        "pending_args": "",
                    }
                    next_block_index += 1

                state = tool_states[oai_idx]
                fn = tc.get("function") or {}
                if tc.get("id"):
                    state["id"] = tc["id"]
                if fn.get("name"):
                    state["name"] = fn["name"]

                # Defer block start until both id AND name are known
                if not state["started"] and state["id"] and state["name"]:
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
                    yield _sse_bytes("content_block_stop", {
                        "type": "content_block_stop", "index": thinking_block_index,
                    })
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
                for state in sorted(tool_states.values(), key=lambda s: s["anthropic_index"]):
                    if state["started"]:
                        continue
                    if not state["id"] and not state["name"] and not state["pending_args"]:
                        continue
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
        yield _sse_bytes("content_block_stop", {
            "type": "content_block_stop", "index": thinking_block_index,
        })
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


def _sse_bytes(event: str, data: Dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


# ---------------------------------------------------------------------------
# Helpers: DeepSeek detection
# ---------------------------------------------------------------------------

_DEEPSEEK_RE = re.compile(r"deepseek", re.IGNORECASE)


def is_deepseek_model(model: str) -> bool:
    """Return True if the model name looks like a DeepSeek model."""
    return bool(_DEEPSEEK_RE.search(model))


# ---------------------------------------------------------------------------
# Anthropic → OpenAI Responses API  (request)
# ---------------------------------------------------------------------------

def _responses_tool_choice(tc: Dict) -> Any:
    tc_type = tc.get("type")
    if tc_type == "auto":
        return "auto"
    if tc_type == "any":
        return "required"
    if tc_type == "tool":
        return {"type": "function", "name": tc.get("name", "")}
    return "auto"


def _messages_to_responses_input(messages: List[Dict]) -> List[Dict[str, Any]]:
    """Convert Anthropic messages to the Responses API ``input`` array."""
    result: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", [])

        if isinstance(content, str):
            item_type = "output_text" if role == "assistant" else "input_text"
            result.append({"role": role, "content": [{"type": item_type, "text": content}]})
            continue

        current_parts: List[Dict[str, Any]] = []

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

    return result


def anthropic_to_responses_request(body: Dict[str, Any], backend_model: str) -> Dict[str, Any]:
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
    thinking = body.get("thinking")
    if thinking:
        if thinking.get("type") == "adaptive":
            effort = "xhigh"
        else:
            effort = _reasoning_effort_from_budget(thinking.get("budget_tokens", 8192))
        result["reasoning"] = {"effort": effort, "summary": "auto"}

    # tools (skip BatchTool, convert to Responses format)
    tools = body.get("tools")
    if tools:
        filtered = [t for t in tools if t.get("type") != "BatchTool"]
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
    cache_read = (
        (usage.get("input_tokens_details") or {}).get("cached_tokens")
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    )
    out: Dict[str, Any] = {
        "input_tokens": max(0, usage.get("input_tokens", 0) - (cache_read or 0)),
        "output_tokens": usage.get("output_tokens", 0),
    }
    if cache_read:
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
                content.append({"type": "thinking", "thinking": joined})

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
    has_tool_use = False

    pending = ""

    async for chunk in source:
        pending += chunk.decode(errors="replace")

        while "\n\n" in pending:
            event_block, pending = pending.split("\n\n", 1)
            data_str: Optional[str] = None
            for line in event_block.splitlines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()

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
                    yield _sse_bytes("content_block_start", {
                        "type": "content_block_start",
                        "index": ant_index,
                        "content_block": {"type": "thinking", "thinking": ""},
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
                    yield _sse_bytes("content_block_stop", {
                        "type": "content_block_stop", "index": ant_index,
                    })

            elif etype == "response.completed":
                # Close any remaining open blocks
                for ant_idx in sorted(open_blocks):
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
