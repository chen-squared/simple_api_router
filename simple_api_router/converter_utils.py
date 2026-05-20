"""Shared SSE helpers used by both OpenAI and Google converters."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

# Anthropic server-executed tool types — handled by Anthropic's infrastructure
# and must NOT be forwarded to third-party backends.  Pattern covers all versioned
# variants, e.g. web_search_20260209, code_execution_20250522, mcp_toolset, etc.
ANTHROPIC_SERVER_TOOL_RE = re.compile(
    r"^(web_search|web_fetch|code_execution|mcp_toolset|advisor|tool_search_tool_|BatchTool)",
    re.IGNORECASE,
)


def is_anthropic_server_tool(tool: Dict) -> bool:
    """Return True for tools that Anthropic executes server-side (not forwarded to backends)."""
    return bool(ANTHROPIC_SERVER_TOOL_RE.match(tool.get("type") or ""))


def sse(event: str, data: Any) -> bytes:
    """Encode a single SSE event as bytes."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


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
