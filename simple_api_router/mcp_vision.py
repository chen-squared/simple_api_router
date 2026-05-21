"""MCP vision server — ``understand_image`` tool backed by the router itself.

When ``server.vision_model`` is set in config.yaml the server mounts this at
``/mcp`` (same port).  Claude Code config example::

    {
      "mcpServers": {
        "vision": {
          "type": "sse",
          "url": "http://localhost:8080/mcp/sse"
        }
      }
    }

Standalone usage (runs its own SSE server on a separate port)::

    python -m simple_api_router.mcp_vision \\
        --model provider/model \\
        --router-url http://localhost:8080 \\
        --port 8081
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path
from typing import Callable, Optional, Union

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_vision_mcp(
    router_url: str,
    vision_model: Union[str, Callable[[], Optional[str]], None] = None,
) -> FastMCP:
    """Return a FastMCP instance wired to call *router_url*/v1/messages.

    Args:
        router_url:   Base URL of the router, e.g. ``http://127.0.0.1:8080``.
        vision_model: Either a fixed "provider/model" string, or a zero-arg
                      callable that returns the current model name (supports
                      hot-reload when the router's config is reloaded).
    """
    if callable(vision_model):
        get_model = vision_model
    else:
        _fixed = vision_model
        get_model = lambda: _fixed  # noqa: E731
    mcp = FastMCP(
        "vision",
        instructions=(
            "Use understand_image to inspect screenshots, diagrams, or any "
            "visual content. Provide a local file path, an HTTPS URL, or raw "
            "base64 image data."
        ),
        stateless_http=True,  # each POST is independent; fully concurrent
    )

    messages_url = router_url.rstrip("/") + "/v1/messages"

    @mcp.tool()
    async def understand_image(
        path: Optional[str] = None,
        url: Optional[str] = None,
        base64_data: Optional[str] = None,
        media_type: str = "image/jpeg",
        question: str = "Please describe this image in detail.",
        max_tokens: int = 16384,
    ) -> str:
        """Describe or answer questions about an image using a vision model.

        Provide **exactly one** of:
        - ``path``        — absolute or relative path to a local image file
        - ``url``         — HTTPS URL of an image
        - ``base64_data`` — raw base64-encoded image (without the ``data:`` prefix)

        ``media_type`` is only used with ``base64_data`` (default: ``image/jpeg``).
        ``question`` defaults to a general description request.
        ``max_tokens`` controls response length (default: 16384).
        """
        # ── Resolve image source ─────────────────────────────────────────
        if path:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"Error: file not found: {p}"
            detected, _ = mimetypes.guess_type(str(p))
            effective_media_type = detected or media_type
            with open(p, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            image_block: dict = {
                "type": "image",
                "source": {"type": "base64", "media_type": effective_media_type, "data": b64},
            }
        elif url:
            image_block = {"type": "image", "source": {"type": "url", "url": url}}
        elif base64_data:
            image_block = {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": base64_data},
            }
        else:
            return "Error: provide one of path, url, or base64_data."

        # ── Call router (streaming to avoid read-timeout on slow models) ──
        current_model = get_model()
        if not current_model:
            return "Error: vision_model is not configured."
        body = {
            "model": current_model,
            "max_tokens": max_tokens,
            "stream": True,  # streaming resets read-timeout per chunk
            "messages": [
                {
                    "role": "user",
                    "content": [image_block, {"type": "text", "text": question}],
                }
            ],
        }
        try:
            # connect_timeout: time to establish the TCP connection
            # read_timeout:    time between receiving successive chunks (NOT total)
            #                  → slow models are fine as long as each chunk arrives
            #                    within this window; effectively no overall timeout
            timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    messages_url,
                    json=body,
                    headers={"content-type": "application/json"},
                ) as resp:
                    resp.raise_for_status()
                    text_parts: list[str] = []
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()
                        if payload in ("", "[DONE]"):
                            continue
                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text_parts.append(delta.get("text", ""))
                    return "".join(text_parts) or "(no text response)"
        except httpx.HTTPStatusError as exc:
            return f"Router error {exc.response.status_code}: {exc.response.text[:500]}"
        except Exception as exc:
            return f"Error: {exc}"

    return mcp


# ---------------------------------------------------------------------------
# Entry point (standalone HTTP server)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Vision MCP server (standalone SSE mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True,
        help="Vision model in provider/model format, e.g. opencode/qwen-vl-plus",
    )
    parser.add_argument(
        "--router-url", default="http://127.0.0.1:8080",
        help="Base URL of the running router (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind the MCP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8081,
        help="Port for the MCP SSE server (default: 8081)",
    )
    args = parser.parse_args(argv)

    vision_mcp = create_vision_mcp(
        vision_model=args.model,
        router_url=args.router_url,
    )
    mcp_app = vision_mcp.streamable_http_app()

    print(f"Vision MCP server → {args.host}:{args.port}/mcp  (model: {args.model})")
    uvicorn.run(mcp_app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
