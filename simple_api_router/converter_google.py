"""Google Gemini API ↔ Anthropic Messages API converter.

Converts Anthropic Messages API requests to Google Gemini generateContent format
and converts Gemini responses back to Anthropic format.

Reference: Google Gemini API — https://ai.google.dev/api/generate-content
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional


def _generate_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


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
        return {
            "functionCall": {
                "name": block.get("name", ""),
                "args": block.get("input", {}),
            }
        }

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
        return {
            "functionResponse": {
                "name": name,
                "response": response_val,
            }
        }

    # Skip thinking / redacted_thinking and anything unknown
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

    # Tools → functionDeclarations
    tools = body.get("tools")
    if tools:
        decls: List[Dict[str, Any]] = []
        for tool in tools:
            decl: Dict[str, Any] = {"name": tool["name"]}
            if "description" in tool:
                decl["description"] = tool["description"]
            if "input_schema" in tool:
                decl["parameters"] = tool["input_schema"]
            decls.append(decl)
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
}


def google_to_anthropic_response(data: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Convert a Gemini generateContent response to Anthropic Messages format."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Safety block
    if "promptFeedback" in data and "blockReason" in data["promptFeedback"]:
        block_reason = data["promptFeedback"]["blockReason"]
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": f"Content blocked: {block_reason}"}],
            "stop_reason": "end_turn",
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
            if text:
                content_blocks.append({"type": "text", "text": text})
        elif "functionCall" in part:
            fc = part["functionCall"]
            content_blocks.append({
                "type": "tool_use",
                "id": _generate_tool_id(),
                "name": fc.get("name", ""),
                "input": fc.get("args", {}),
            })
            has_tool_use = True
        # Skip "thought" parts (thinking blocks) silently

    stop_reason = "tool_use" if has_tool_use else _FINISH_REASON_MAP.get(finish_reason, "end_turn")

    usage_meta = data.get("usageMetadata", {})
    input_tokens = usage_meta.get("promptTokenCount", 0)
    candidates_tokens = usage_meta.get("candidatesTokenCount", 0)
    total_tokens = usage_meta.get("totalTokenCount", 0)
    output_tokens = candidates_tokens or max(0, total_tokens - input_tokens)

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Gemini SSE stream → Anthropic SSE stream
# ---------------------------------------------------------------------------

def _sse(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


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

    # Track open text block and accumulated tool calls
    text_block_open = False
    text_block_index = 0
    tool_blocks: List[Dict[str, Any]] = []
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
                accumulated_usage["input_tokens"] = usage_meta.get("promptTokenCount", 0)
                c_tokens = usage_meta.get("candidatesTokenCount", 0)
                total = usage_meta.get("totalTokenCount", 0)
                accumulated_usage["output_tokens"] = (
                    c_tokens or max(0, total - accumulated_usage["input_tokens"])
                )

            candidates = chunk_data.get("candidates", [])
            if not candidates:
                continue

            candidate = candidates[0]
            if "finishReason" in candidate:
                finish_reason = candidate["finishReason"]

            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    text = part["text"]
                    if not text:
                        continue
                    if not text_block_open:
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": text_block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                        text_block_open = True
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": text_block_index,
                        "delta": {"type": "text_delta", "text": text},
                    })
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_blocks.append({
                        "id": _generate_tool_id(),
                        "name": fc.get("name", ""),
                        "input": fc.get("args", {}),
                    })

    # Close open text block
    if text_block_open:
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": text_block_index,
        })

    # Emit tool use blocks
    next_index = text_block_index + (1 if text_block_open else 0)
    for i, tool in enumerate(tool_blocks):
        idx = next_index + i
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": idx,
            "content_block": {
                "type": "tool_use",
                "id": tool["id"],
                "name": tool["name"],
                "input": {},
            },
        })
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": idx,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(tool["input"]),
            },
        })
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": idx,
        })

    has_tool_use = len(tool_blocks) > 0
    stop_reason = "tool_use" if has_tool_use else _FINISH_REASON_MAP.get(finish_reason, "end_turn")

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": accumulated_usage["output_tokens"]},
    })

    yield _sse("message_stop", {"type": "message_stop"})
