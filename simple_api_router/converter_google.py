"""Google Gemini API ↔ Anthropic Messages API converter.

Converts Anthropic Messages API requests to Google Gemini generateContent format
and converts Gemini responses back to Anthropic format.

Reference: Google Gemini API — https://ai.google.dev/api/generate-content
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from .converter_utils import (
    ANTHROPIC_SERVER_TOOL_RE as _ANTHROPIC_SERVER_TOOL_RE,
    is_anthropic_server_tool as _is_anthropic_server_tool,
    sse as _sse,
    thinking_close_events as _google_thinking_close_events,
)

_GEMINI3_RE = re.compile(r"gemini-3", re.IGNORECASE)


def _gemini_thinking_config(thinking: Dict[str, Any], model: str, output_effort: Optional[str] = None) -> Dict[str, Any]:
    """Convert Anthropic thinking config to Gemini thinkingConfig dict.

    Gemini 3+ uses thinkingLevel (string); Gemini 2.5 uses thinkingBudget (int).
    output_effort is the value from output_config.effort in the Anthropic request.
    includeThoughts must be true for thought summaries to be included in response parts.
    """
    thinking_type = thinking.get("type", "enabled")

    # Map Anthropic output_config.effort to Gemini thinkingLevel
    _EFFORT_TO_LEVEL = {"max": "high", "xhigh": "high", "high": "high", "medium": "medium", "low": "low"}

    if _GEMINI3_RE.search(model):
        # Gemini 3+ — thinkingLevel
        if thinking_type == "disabled":
            return {"thinkingLevel": "minimal"}
        if output_effort:
            return {"thinkingLevel": _EFFORT_TO_LEVEL.get(output_effort, "high"), "includeThoughts": True}
        budget = thinking.get("budget_tokens", 8192)
        if thinking_type == "adaptive" or budget > 32000:
            return {"thinkingLevel": "high", "includeThoughts": True}
        if budget <= 1024:
            return {"thinkingLevel": "low", "includeThoughts": True}
        if budget <= 8192:
            return {"thinkingLevel": "medium", "includeThoughts": True}
        return {"thinkingLevel": "high", "includeThoughts": True}
    else:
        # Gemini 2.5 and earlier — thinkingBudget
        if thinking_type == "disabled":
            return {"thinkingBudget": 0}
        if thinking_type == "adaptive":
            # No direct budget equivalent; omit to let Gemini use dynamic thinking.
            return {"includeThoughts": True}
        budget = thinking.get("budget_tokens", 8192)
        return {"thinkingBudget": budget, "includeThoughts": True}


def _generate_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _is_synthesized_id(tool_id: str) -> bool:
    """Return True if the id was synthesized by this converter (starts with toolu_).

    Synthesized ids must not be round-tripped back to Gemini as functionCall/
    functionResponse ids because Gemini does not recognise them.  Real ids
    (e.g. "call_1") that came from a previous Gemini response should be
    preserved so Gemini can match them across turns.
    """
    return tool_id.startswith("toolu_")


_JSON_SCHEMA_RICH_KEYS = frozenset({
    "$schema", "additionalProperties", "oneOf", "anyOf", "allOf",
    "not", "const", "if", "then", "else",
    "unevaluatedProperties", "unevaluatedItems",
})


def _needs_json_schema_encoding(schema: Dict[str, Any]) -> bool:
    """Return True when input_schema uses JSON-Schema features that Gemini cannot
    represent as an OpenAPI ``parameters`` object.  In that case the declaration
    should use ``parametersJsonSchema`` instead.
    """
    return bool(_JSON_SCHEMA_RICH_KEYS & schema.keys())


def _build_tool_id_map(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """Scan all messages and build a map of tool_use_id → tool name."""
    mapping: Dict[str, str] = {}
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    mapping[block["id"]] = block["name"]
    return mapping


def _content_block_to_gemini_part(
    block: Any,
    tool_id_map: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Convert one Anthropic content block to a Gemini part. Returns None to skip."""
    if isinstance(block, str):
        return {"text": block}
    if not isinstance(block, dict):
        return None

    btype = block.get("type", "")

    if btype == "text":
        text = block.get("text", "")
        if text:
            return {"text": text}
        return None

    if btype == "image":
        source = block.get("source", {})
        src_type = source.get("type", "")
        if src_type == "base64":
            return {
                "inlineData": {
                    "mimeType": source.get("media_type", "image/jpeg"),
                    "data": source.get("data", ""),
                }
            }
        if src_type == "url":
            return {
                "fileData": {
                    "mimeType": source.get("media_type", "image/jpeg"),
                    "fileUri": source.get("url", ""),
                }
            }
        return None

    if btype == "document":
        source = block.get("source", {})
        if source.get("type") == "base64":
            return {
                "inlineData": {
                    "mimeType": "application/pdf",
                    "data": source.get("data", ""),
                }
            }
        return None

    if btype == "tool_use":
        tool_id = block.get("id", "")
        fc: Dict[str, Any] = {
            "name": block.get("name", ""),
            "args": block.get("input", {}),
        }
        # Preserve the real Gemini id so it can be matched on subsequent turns.
        # Synthesised ids (toolu_*) are not meaningful to Gemini and are stripped.
        if tool_id and not _is_synthesized_id(tool_id):
            fc["id"] = tool_id
        return {"functionCall": fc}

    if btype == "tool_result":
        tool_use_id = block.get("tool_use_id", "")
        name = tool_id_map.get(tool_use_id, tool_use_id)
        raw_content = block.get("content", "")
        if isinstance(raw_content, str):
            response_val: Any = {"output": raw_content}
        elif isinstance(raw_content, list):
            texts = [
                cb.get("text", "")
                for cb in raw_content
                if isinstance(cb, dict) and cb.get("type") == "text"
            ]
            response_val = {"output": "\n".join(texts)}
        else:
            response_val = {"output": str(raw_content)}
        fr: Dict[str, Any] = {"name": name, "response": response_val}
        if tool_use_id and not _is_synthesized_id(tool_use_id):
            fr["id"] = tool_use_id
        return {"functionResponse": fr}

    # thinking → thought part (so Gemini sees prior reasoning in multi-turn)
    if btype == "thinking":
        text = block.get("thinking", "")
        if text:
            return {"text": text, "thought": True}
        return None

    if btype == "redacted_thinking":
        return None  # encrypted; cannot round-trip to Gemini

    if btype in ("server_tool_use", "web_search_tool_result"):
        return None  # Anthropic server-executed tools; no Gemini equivalent

    # Skip anything else unknown
    return None


def anthropic_to_google_request(body: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Convert an Anthropic Messages request body to Google Gemini generateContent format."""
    messages: List[Dict[str, Any]] = body.get("messages", [])
    tool_id_map = _build_tool_id_map(messages)

    # Build contents list
    contents: List[Dict[str, Any]] = []
    for msg in messages:
        role = "user" if msg.get("role") == "user" else "model"
        raw_content = msg.get("content", [])

        if isinstance(raw_content, str):
            parts: List[Dict[str, Any]] = [{"text": raw_content}]
        else:
            parts = []
            for block in raw_content:
                part = _content_block_to_gemini_part(block, tool_id_map)
                if part is not None:
                    parts.append(part)

        if parts:
            contents.append({"role": role, "parts": parts})
        elif role == "model":
            # Empty model turn: content list was empty (interrupted before first
            # block) or contained only redacted_thinking (cannot round-trip).
            # Gemini requires strictly alternating user/model turns, so synthesise
            # a placeholder rather than dropping the turn and causing a turn-order
            # error.
            contents.append({"role": "model", "parts": [{"text": ""}]})

    result: Dict[str, Any] = {"contents": contents}

    # System instruction
    system = body.get("system")
    if system:
        if isinstance(system, str):
            result["systemInstruction"] = {"parts": [{"text": system}]}
        elif isinstance(system, list):
            texts = [
                b.get("text", "")
                for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            combined = "\n".join(t for t in texts if t)
            if combined:
                result["systemInstruction"] = {"parts": [{"text": combined}]}

    # Generation config
    gc: Dict[str, Any] = {}
    if "max_tokens" in body:
        gc["maxOutputTokens"] = body["max_tokens"]
    if "temperature" in body:
        gc["temperature"] = body["temperature"]
    if "top_p" in body:
        gc["topP"] = body["top_p"]
    if "top_k" in body:
        gc["topK"] = body["top_k"]
    if "stop_sequences" in body:
        gc["stopSequences"] = body["stop_sequences"]
    if gc:
        result["generationConfig"] = gc

    # thinking → thinkingConfig
    thinking = body.get("thinking")
    if thinking:
        output_effort = (body.get("output_config") or {}).get("effort")
        tc = _gemini_thinking_config(thinking, model, output_effort)
        if tc:
            result.setdefault("generationConfig", {})["thinkingConfig"] = tc

    # Tools → functionDeclarations (skip Anthropic server tools like web_search, code_execution)
    tools = body.get("tools")
    if tools:
        user_tools = [t for t in tools if not _ANTHROPIC_SERVER_TOOL_RE.match(t.get("type") or "")]
        decls: List[Dict[str, Any]] = []
        for tool in user_tools:
            decl: Dict[str, Any] = {"name": tool["name"]}
            if "description" in tool:
                decl["description"] = tool["description"]
            if "input_schema" in tool:
                schema = tool["input_schema"]
                if _needs_json_schema_encoding(schema):
                    # Strip $schema — Gemini does not accept it in parametersJsonSchema
                    decl["parametersJsonSchema"] = {
                        k: v for k, v in schema.items() if k != "$schema"
                    }
                else:
                    decl["parameters"] = schema
            decls.append(decl)
        if decls:
            result["tools"] = [{"functionDeclarations": decls}]

    # Tool choice → functionCallingConfig
    tool_choice = body.get("tool_choice")
    if tool_choice:
        tc_type = tool_choice.get("type", "auto")
        mode_map = {"auto": "AUTO", "none": "NONE", "any": "ANY", "tool": "ANY"}
        mode = mode_map.get(tc_type, "AUTO")
        fc_config: Dict[str, Any] = {"mode": mode}
        if tc_type == "tool" and "name" in tool_choice:
            fc_config["allowedFunctionNames"] = [tool_choice["name"]]
        result["toolConfig"] = {"functionCallingConfig": fc_config}

    return result


# ---------------------------------------------------------------------------
# Gemini → Anthropic response
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP: Dict[str, str] = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "STOP_SEQUENCE": "stop_sequence",
    "SAFETY": "end_turn",
    "RECITATION": "end_turn",
    "SPII": "end_turn",
    "OTHER": "end_turn",
    "FINISH_REASON_UNSPECIFIED": "end_turn",
    # Additional values from Gemini API reference
    "LANGUAGE": "end_turn",
    "BLOCKLIST": "end_turn",
    "PROHIBITED_CONTENT": "end_turn",
    "MALFORMED_FUNCTION_CALL": "end_turn",
    "MISSING_THOUGHT_SIGNATURE": "end_turn",
    "MALFORMED_RESPONSE": "end_turn",
}


def google_to_anthropic_response(data: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Convert a Gemini generateContent response to Anthropic Messages format."""
    # Use responseId from Gemini when available for stable cross-turn tracking
    msg_id = data.get("responseId") or f"msg_{uuid.uuid4().hex[:24]}"

    # Safety block — content was blocked before any candidates were generated
    if "promptFeedback" in data and "blockReason" in data["promptFeedback"]:
        block_reason = data["promptFeedback"]["blockReason"]
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": f"Content blocked: {block_reason}"}],
            "stop_reason": "refusal",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    candidates = data.get("candidates", [])
    if not candidates:
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])
    finish_reason = candidate.get("finishReason", "STOP")

    content_blocks: List[Dict[str, Any]] = []
    has_tool_use = False

    for part in parts:
        if "text" in part:
            text = part["text"]
            if not text:
                continue
            if part.get("thought"):
                content_blocks.append({"type": "thinking", "thinking": text, "signature": ""})
            else:
                content_blocks.append({"type": "text", "text": text})
        elif "functionCall" in part:
            fc = part["functionCall"]
            # Preserve real Gemini ids; synthesise one only when absent
            fc_id = fc.get("id", "")
            tool_id = fc_id if fc_id else _generate_tool_id()
            content_blocks.append({
                "type": "tool_use",
                "id": tool_id,
                "name": fc.get("name", ""),
                "input": fc.get("args", {}),
            })
            has_tool_use = True
        # Skip "thought" / "thoughtSignature" parts silently

    stop_reason = "tool_use" if has_tool_use else _FINISH_REASON_MAP.get(finish_reason, "end_turn")

    usage_meta = data.get("usageMetadata", {})
    input_tokens = usage_meta.get("promptTokenCount", 0)
    candidates_tokens = usage_meta.get("candidatesTokenCount", 0)
    total_tokens = usage_meta.get("totalTokenCount", 0)
    # thoughtsTokenCount is separate from candidatesTokenCount (thinking tokens are billed)
    thoughts_tokens = usage_meta.get("thoughtsTokenCount", 0)
    output_tokens = (candidates_tokens or max(0, total_tokens - input_tokens)) + thoughts_tokens
    # Google's promptTokenCount includes cached tokens; subtract for Anthropic semantics.
    cache_tokens = usage_meta.get("cachedContentTokenCount", 0)

    usage: Dict[str, Any] = {
        "input_tokens": max(0, input_tokens - cache_tokens),
        "output_tokens": output_tokens,
    }
    if cache_tokens:
        usage["cache_read_input_tokens"] = cache_tokens

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Gemini SSE stream → Anthropic SSE stream
# ---------------------------------------------------------------------------

async def stream_google_to_anthropic(
    aiter: AsyncIterator[bytes], model: str
) -> AsyncIterator[bytes]:
    """Convert a Gemini SSE byte stream to Anthropic SSE byte stream."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    # Track open blocks
    next_block_index = 0
    thinking_block_open = False
    thinking_block_index = -1
    accumulated_thinking = ""
    text_block_open = False
    text_block_index = -1
    accumulated_text = ""          # running total of text emitted so far
    has_tool_use = False
    accumulated_usage = {"input_tokens": 0, "output_tokens": 0}
    finish_reason = "STOP"

    buffer = b""
    async for raw_chunk in aiter:
        buffer += raw_chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line.startswith(b"data: "):
                continue
            data_str = line[6:].decode("utf-8", errors="replace").strip()
            if data_str == "[DONE]":
                break
            try:
                chunk_data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Capture usage from any chunk that provides it
            usage_meta = chunk_data.get("usageMetadata", {})
            if usage_meta:
                prompt_tokens = usage_meta.get("promptTokenCount", 0)
                c_tokens = usage_meta.get("candidatesTokenCount", 0)
                total = usage_meta.get("totalTokenCount", 0)
                cache_tokens = usage_meta.get("cachedContentTokenCount", 0)
                # thoughtsTokenCount is separate from candidatesTokenCount (thinking tokens are billed)
                thoughts_tokens = usage_meta.get("thoughtsTokenCount", 0)
                # Google's promptTokenCount includes cached; subtract for Anthropic semantics.
                accumulated_usage["input_tokens"] = max(0, prompt_tokens - cache_tokens)
                accumulated_usage["output_tokens"] = (
                    c_tokens or max(0, total - prompt_tokens)
                ) + thoughts_tokens
                if cache_tokens:
                    accumulated_usage["cache_read_input_tokens"] = cache_tokens
                else:
                    accumulated_usage.pop("cache_read_input_tokens", None)

            candidates = chunk_data.get("candidates", [])
            if not candidates:
                continue

            candidate = candidates[0]
            if "finishReason" in candidate:
                finish_reason = candidate["finishReason"]

            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                if part.get("thought") and "text" in part:
                    # --- thinking delta ---
                    new_thinking = part["text"]
                    if not new_thinking:
                        continue
                    # Close text block if somehow open (shouldn't happen)
                    if text_block_open:
                        yield _sse("content_block_stop", {
                            "type": "content_block_stop", "index": text_block_index,
                        })
                        text_block_open = False
                    if not thinking_block_open:
                        thinking_block_index = next_block_index
                        next_block_index += 1
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": thinking_block_index,
                            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                        })
                        thinking_block_open = True
                    if new_thinking.startswith(accumulated_thinking):
                        delta = new_thinking[len(accumulated_thinking):]
                        accumulated_thinking = new_thinking
                    else:
                        delta = new_thinking
                        accumulated_thinking += new_thinking
                    if delta:
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": thinking_block_index,
                            "delta": {"type": "thinking_delta", "thinking": delta},
                        })
                elif "text" in part:
                    new_text = part["text"]
                    if not new_text:
                        continue
                    # Close thinking block if transitioning to answer
                    if thinking_block_open:
                        for ev in _google_thinking_close_events(thinking_block_index):
                            yield ev
                        thinking_block_open = False
                        accumulated_thinking = ""  # reset so next thought segment deltas correctly
                    # Gemini may send cumulative text (each chunk = full text so far)
                    # or incremental text (each chunk = new text only).  Handle both:
                    # if new_text starts with what we have so far it is cumulative.
                    if new_text.startswith(accumulated_text):
                        delta = new_text[len(accumulated_text):]
                        accumulated_text = new_text
                    else:
                        delta = new_text
                        accumulated_text += new_text
                    if not delta:
                        continue
                    if not text_block_open:
                        text_block_index = next_block_index
                        next_block_index += 1
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": text_block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                        text_block_open = True
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": text_block_index,
                        "delta": {"type": "text_delta", "text": delta},
                    })
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    # Close any open thinking/text blocks before emitting the tool
                    if thinking_block_open:
                        for ev in _google_thinking_close_events(thinking_block_index):
                            yield ev
                        thinking_block_open = False
                        accumulated_thinking = ""
                    if text_block_open:
                        yield _sse("content_block_stop", {
                            "type": "content_block_stop", "index": text_block_index,
                        })
                        text_block_open = False
                    # Emit the tool block inline (not accumulated) so it appears in
                    # the correct position relative to thinking blocks.
                    fc_id = fc.get("id", "") or _generate_tool_id()
                    tool_idx = next_block_index
                    next_block_index += 1
                    has_tool_use = True
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": tool_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": fc_id,
                            "name": fc.get("name", ""),
                            "input": {},
                        },
                    })
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": tool_idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(fc.get("args", {})),
                        },
                    })
                    yield _sse("content_block_stop", {
                        "type": "content_block_stop", "index": tool_idx,
                    })

    # Close any open blocks
    if thinking_block_open:
        for ev in _google_thinking_close_events(thinking_block_index):
            yield ev
    if text_block_open:
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": text_block_index,
        })

    stop_reason = "tool_use" if has_tool_use else _FINISH_REASON_MAP.get(finish_reason, "end_turn")

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {k: v for k, v in accumulated_usage.items()},
    })

    yield _sse("message_stop", {"type": "message_stop"})
