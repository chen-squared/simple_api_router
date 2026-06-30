"""Shared SSE helpers used by both OpenAI and Google converters."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Dict, List

# Anthropic server-executed tool types — handled by Anthropic's infrastructure
# and must NOT be forwarded to third-party backends.  Pattern covers all versioned
# variants, e.g. web_search_20260209, code_execution_20250522, mcp_toolset, etc.
ANTHROPIC_SERVER_TOOL_RE = re.compile(
    r"^(web_search|web_fetch|code_execution|computer_use|mcp_toolset|advisor|tool_search_tool_|BatchTool)",
    re.IGNORECASE,
)

# Emit a downstream ping when a converted stream is silent this long (seconds).
# Claude Code's stream idle watchdog aborts after 5 minutes with no SSE events;
# periodic pings keep long upstream thinking gaps alive.
STREAM_IDLE_PING_SECONDS = 60.0


def is_anthropic_server_tool(tool: Dict) -> bool:
    """Return True for tools that Anthropic executes server-side (not forwarded to backends)."""
    return bool(ANTHROPIC_SERVER_TOOL_RE.match(tool.get("type") or ""))


def _content_as_block_list(content: Any) -> List[Dict[str, Any]]:
    """Normalise an Anthropic message ``content`` (string or block list) to a list
    of content blocks. Empty/whitespace-only strings become an empty list."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        return list(content)
    return []


def fold_midstream_system_into_user(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold mid-conversation ``role: "system"`` messages into an adjacent user turn.

    Anthropic's Messages API supports *mid-conversation system messages*
    (``{"role": "system"}`` entries inside ``messages``) — Claude Code uses them
    to relay input the user typed while the model was working. Non-Anthropic
    backends (OpenAI/DeepSeek/Gemini/…) have no equivalent: leaving such messages
    in place gets them ignored by weak models and *rejected outright* by strict
    providers (e.g. MiniMax → ``invalid message role: system (2013)``).

    This collapses every mid-array system message into a neighbouring user turn as
    plain text block(s), preserving order and content. It merges into the
    *preceding* user turn when there is one (where Anthropic requires these
    messages to sit — right after a user/tool_result turn), otherwise prepends to
    the next user turn. Merging (rather than emitting a standalone user message)
    guarantees we never create two consecutive same-role turns, which backends
    like deepseek-reasoner and Gemini reject.

    The top-level ``system`` field is unaffected; only entries inside ``messages``
    are touched. Returns the original list unchanged (no copy) when there are no
    mid-array system messages — the overwhelmingly common case.
    """
    if not isinstance(messages, list) or not any(
        isinstance(m, dict) and m.get("role") == "system" for m in messages
    ):
        return messages

    out: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []  # system blocks awaiting the next user turn

    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        role = msg.get("role")

        if role == "system":
            blocks = _content_as_block_list(msg.get("content"))
            if out and isinstance(out[-1], dict) and out[-1].get("role") == "user":
                # Merge into the preceding user turn (the canonical position).
                out[-1]["content"] = _content_as_block_list(out[-1].get("content")) + blocks
            else:
                # No preceding user turn — carry forward to the next user turn.
                pending.extend(blocks)
            continue

        new_msg = dict(msg)
        if role == "user":
            if pending:
                new_msg["content"] = pending + _content_as_block_list(msg.get("content"))
                pending = []
            out.append(new_msg)
        else:
            # assistant/other: flush any carried system blocks as a user turn first
            if pending:
                out.append({"role": "user", "content": pending})
                pending = []
            out.append(new_msg)

    if pending:
        out.append({"role": "user", "content": pending})
    return out


def sse(event: str, data: Any) -> bytes:
    """Encode a single SSE event as bytes."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def stream_with_idle_ping(
    source: AsyncIterator[bytes],
    idle_seconds: float = STREAM_IDLE_PING_SECONDS,
) -> AsyncIterator[bytes]:
    """Wrap a converted SSE byte stream, emitting Anthropic pings during upstream silence."""
    ait = source.__aiter__()
    pending: asyncio.Task[Any] = asyncio.create_task(ait.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=idle_seconds)
            if not done:
                yield sse("ping", {"type": "ping"})
                continue
            try:
                chunk = pending.result()
            except StopAsyncIteration:
                break
            yield chunk
            pending = asyncio.create_task(ait.__anext__())
    finally:
        if not pending.done():
            pending.cancel()


def thinking_close_events(index: int) -> List[bytes]:
    """Return [signature_delta, content_block_stop] SSE bytes for a thinking block.

    Per Anthropic streaming spec a signature_delta must precede content_block_stop
    for every thinking block.  Non-Anthropic backends have no real cryptographic
    signature so we emit an empty string.
    """
    return [
        sse("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "signature_delta", "signature": ""},
        }),
        sse("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        }),
    ]
