"""MCP media server — image / audio / video understanding tools.

When any of ``server.image_model``, ``server.audio_model``, or
``server.video_model`` is set in config.yaml, the server mounts the
corresponding tools at ``/mcp`` (same port).  Claude Code config example::

    {
      "mcpServers": {
        "media": {
          "type": "sse",
          "url": "http://localhost:8080/mcp/sse"
        }
      }
    }

The tools are registered only for the models that are configured:
- ``image_understanding``  → requires ``server.image_model``
- ``audio_understanding``  → requires ``server.audio_model``
- ``video_understanding``  → requires ``server.video_model``

Standalone usage (runs its own SSE server on a separate port)::

    python -m simple_api_router.mcp_media \\
        --image-model provider/model \\
        --audio-model provider/model \\
        --video-model provider/model \\
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
# Helpers
# ---------------------------------------------------------------------------

def _make_getter(value: Union[str, Callable[[], Optional[str]], None]):
    """Return a zero-arg callable that returns the current model name."""
    if callable(value):
        return value
    _fixed = value
    return lambda: _fixed  # noqa: E731


async def _call_model_streaming(
    messages_url: str,
    model: str,
    content_blocks: list,
    question: str,
    max_tokens: int,
) -> str:
    """Send a streaming request to the router and collect the text response."""
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [{
            "role": "user",
            "content": content_blocks + [{"type": "text", "text": question}],
        }],
    }
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_media_mcp(
    router_url: str,
    image_model: Union[str, Callable[[], Optional[str]], None] = None,
    audio_model: Union[str, Callable[[], Optional[str]], None] = None,
    video_model: Union[str, Callable[[], Optional[str]], None] = None,
) -> FastMCP:
    """Return a FastMCP instance with tools for the models that are configured.

    Args:
        router_url:   Base URL of the router, e.g. ``http://127.0.0.1:8080``.
        image_model:  "provider/model" or callable; enables image_understanding.
        audio_model:  "provider/model" or callable; enables audio_understanding.
        video_model:  "provider/model" or callable; enables video_understanding.

    At least one model must be provided or an error is raised.
    """
    if not any([image_model, audio_model, video_model]):
        raise ValueError("At least one of image_model, audio_model, video_model must be set")

    get_image = _make_getter(image_model)
    get_audio = _make_getter(audio_model)
    get_video = _make_getter(video_model)

    tool_names = []
    if image_model:
        tool_names.append("image_understanding")
    if audio_model:
        tool_names.append("audio_understanding")
    if video_model:
        tool_names.append("video_understanding")

    mcp = FastMCP(
        "media",
        instructions=(
            f"Available tools: {', '.join(tool_names)}. "
            "Use image_understanding for screenshots, diagrams, or any visual content. "
            "Use audio_understanding for voice recordings or audio files. "
            "Use video_understanding for video content. "
            "Provide a local file path, an HTTPS URL, or raw base64 data."
        ),
        stateless_http=True,
    )

    messages_url = router_url.rstrip("/") + "/v1/messages"

    # ── image_understanding ────────────────────────────────────────────────
    if image_model:
        @mcp.tool()
        async def image_understanding(
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
            current_model = get_image()
            if not current_model:
                return "Error: image_model is not configured."

            if path:
                p = Path(path).expanduser().resolve()
                if not p.exists():
                    return f"Error: file not found: {p}"
                detected, _ = mimetypes.guess_type(str(p))
                mt = detected or media_type
                with open(p, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                block = {"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}}
            elif url:
                block = {"type": "image", "source": {"type": "url", "url": url}}
            elif base64_data:
                block = {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64_data}}
            else:
                return "Error: provide one of path, url, or base64_data."

            try:
                return await _call_model_streaming(messages_url, current_model, [block], question, max_tokens)
            except httpx.HTTPStatusError as exc:
                return f"Router error {exc.response.status_code}: {exc.response.text[:500]}"
            except Exception as exc:
                return f"Error: {exc}"

    # ── audio_understanding ────────────────────────────────────────────────
    if audio_model:
        @mcp.tool()
        async def audio_understanding(
            path: Optional[str] = None,
            url: Optional[str] = None,
            base64_data: Optional[str] = None,
            media_type: str = "audio/mp3",
            question: str = "Please transcribe and describe this audio in detail.",
            max_tokens: int = 16384,
        ) -> str:
            """Transcribe or answer questions about audio using an audio-capable model.

            Provide **exactly one** of:
            - ``path``        — absolute or relative path to a local audio file
            - ``url``         — HTTPS URL of an audio file
            - ``base64_data`` — raw base64-encoded audio (without the ``data:`` prefix)

            ``media_type`` is only used with ``base64_data`` (default: ``audio/mp3``).
            ``question`` defaults to a transcription + description request.
            ``max_tokens`` controls response length (default: 16384).
            """
            current_model = get_audio()
            if not current_model:
                return "Error: audio_model is not configured."

            if path:
                p = Path(path).expanduser().resolve()
                if not p.exists():
                    return f"Error: file not found: {p}"
                detected, _ = mimetypes.guess_type(str(p))
                mt = detected or media_type
                with open(p, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                block = {"type": "audio", "source": {"type": "base64", "media_type": mt, "data": b64}}
            elif url:
                block = {"type": "audio", "source": {"type": "url", "url": url}}
            elif base64_data:
                block = {"type": "audio", "source": {"type": "base64", "media_type": media_type, "data": base64_data}}
            else:
                return "Error: provide one of path, url, or base64_data."

            try:
                return await _call_model_streaming(messages_url, current_model, [block], question, max_tokens)
            except httpx.HTTPStatusError as exc:
                return f"Router error {exc.response.status_code}: {exc.response.text[:500]}"
            except Exception as exc:
                return f"Error: {exc}"

    # ── video_understanding ────────────────────────────────────────────────
    if video_model:
        @mcp.tool()
        async def video_understanding(
            path: Optional[str] = None,
            url: Optional[str] = None,
            base64_data: Optional[str] = None,
            media_type: str = "video/mp4",
            question: str = "Please describe this video in detail.",
            max_tokens: int = 16384,
        ) -> str:
            """Describe or answer questions about a video using a video-capable model.

            Provide **exactly one** of:
            - ``path``        — absolute or relative path to a local video file
            - ``url``         — HTTPS URL of a video
            - ``base64_data`` — raw base64-encoded video (without the ``data:`` prefix)

            ``media_type`` is only used with ``base64_data`` (default: ``video/mp4``).
            ``question`` defaults to a general description request.
            ``max_tokens`` controls response length (default: 16384).
            """
            current_model = get_video()
            if not current_model:
                return "Error: video_model is not configured."

            if path:
                p = Path(path).expanduser().resolve()
                if not p.exists():
                    return f"Error: file not found: {p}"
                detected, _ = mimetypes.guess_type(str(p))
                mt = detected or media_type
                with open(p, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                block = {"type": "video", "source": {"type": "base64", "media_type": mt, "data": b64}}
            elif url:
                block = {"type": "video", "source": {"type": "url", "url": url}}
            elif base64_data:
                block = {"type": "video", "source": {"type": "base64", "media_type": media_type, "data": base64_data}}
            else:
                return "Error: provide one of path, url, or base64_data."

            try:
                return await _call_model_streaming(messages_url, current_model, [block], question, max_tokens)
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
        description="Media MCP server (standalone SSE mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image-model", default=None, help="Image model in provider/model format")
    parser.add_argument("--audio-model", default=None, help="Audio model in provider/model format")
    parser.add_argument("--video-model", default=None, help="Video model in provider/model format")
    parser.add_argument(
        "--router-url", default="http://127.0.0.1:8080",
        help="Base URL of the running router (default: http://127.0.0.1:8080)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8081, help="Port for the MCP SSE server (default: 8081)")
    args = parser.parse_args(argv)

    if not any([args.image_model, args.audio_model, args.video_model]):
        parser.error("At least one of --image-model, --audio-model, --video-model must be provided")

    media_mcp = create_media_mcp(
        router_url=args.router_url,
        image_model=args.image_model,
        audio_model=args.audio_model,
        video_model=args.video_model,
    )
    mcp_app = media_mcp.streamable_http_app()

    tools = [t for t, m in [("image", args.image_model), ("audio", args.audio_model), ("video", args.video_model)] if m]
    print(f"Media MCP server → {args.host}:{args.port}/mcp  (tools: {', '.join(tools)})")
    uvicorn.run(mcp_app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
