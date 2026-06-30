"""MCP media server — image / audio / video / PDF understanding tools.

The router always exposes ``/mcp`` (same port, Streamable HTTP transport).
The available tools are derived from the current config and hot-reload when
``server.image_model``, ``server.audio_model``, ``server.video_model``, or
``server.pdf_model`` changes.

Claude Code config example::

    {
      "mcpServers": {
        "media": {
          "type": "http",
          "url": "http://localhost:8080/mcp"
        }
      }
    }

Standalone usage (runs its own HTTP server on a separate port)::

    python -m simple_api_router.mcp_media \\
        --image-model provider/model \\
        --audio-model provider/model \\
        --video-model provider/model \\
        --pdf-model provider/model \\
        --router-url http://localhost:8080 \\
        --port 8081
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Callable, Optional, Union

import httpx
from mcp.server.fastmcp import FastMCP


MaybeGetter = Union[str, Callable[[], Optional[str]], None]

# Align with Claude Code CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT default (5 minutes).
_MCP_STREAM_READ_TIMEOUT = 300.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_media_type(mt: str | None) -> str | None:
    """Return a well-known media type, mapping non-standard forms to canonical ones.

    ``mimetypes.guess_type`` returns non-standard MIME subtypes on some platforms
    (e.g. ``"audio/x-wav"`` instead of ``"audio/wav"``). Strips the ``"x-"``
    prefix and handles other well-known remappings.
    """
    if mt is None:
        return None
    main, sub = mt.split("/", 1) if "/" in mt else (mt, "")
    if sub.startswith("x-"):
        sub = sub[2:]
    sub = {
        "mp4a-latm": "mp4",
        "mpeg": "mp3",
        "quicktime": "mp4",
        "msvideo": "avi",
        "basic": "au",
    }.get(sub, sub)
    return f"{main}/{sub}"


def _make_getter(value: MaybeGetter) -> Callable[[], Optional[str]]:
    """Return a zero-arg callable that returns the current model name."""
    if callable(value):
        return value
    fixed = value
    return lambda: fixed  # noqa: E731


def _build_media_block(
    kind: str,
    path: Optional[str],
    url: Optional[str],
    base64_data: Optional[str],
    media_type: str,
) -> dict:
    """Build one Anthropic-format media/document block for the router."""
    block_type = "document" if kind == "pdf" else kind

    if path:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))
        resolved_media_type = media_type if kind == "pdf" else (
            _normalize_media_type(mimetypes.guess_type(str(file_path))[0]) or media_type
        )
        with open(file_path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode()
        return {
            "type": block_type,
            "source": {"type": "base64", "media_type": resolved_media_type, "data": encoded},
        }

    if url:
        return {"type": block_type, "source": {"type": "url", "url": url}}

    if base64_data:
        return {
            "type": block_type,
            "source": {"type": "base64", "media_type": media_type, "data": base64_data},
        }

    raise ValueError("provide one of path, url, or base64_data.")


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
    timeout = httpx.Timeout(
        connect=30.0,
        read=_MCP_STREAM_READ_TIMEOUT,
        write=30.0,
        pool=5.0,
    )
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


async def _run_media_tool(
    messages_url: str,
    kind: str,
    model_getter: Callable[[], Optional[str]],
    path: Optional[str],
    url: Optional[str],
    base64_data: Optional[str],
    media_type: str,
    question: str,
    max_tokens: int,
) -> str:
    """Execute one media MCP tool call through the router."""
    current_model = model_getter()
    if not current_model:
        return f"Error: {kind}_model is not configured."

    try:
        block = _build_media_block(kind, path, url, base64_data, media_type)
    except FileNotFoundError as exc:
        return f"Error: file not found: {exc}"
    except ValueError as exc:
        return f"Error: {exc}"

    try:
        return await _call_model_streaming(messages_url, current_model, [block], question, max_tokens)
    except httpx.HTTPStatusError as exc:
        return f"Router error {exc.response.status_code}: {exc.response.text[:500]}"
    except Exception as exc:
        return f"Error: {exc}"


def sync_media_mcp_tools(mcp: FastMCP) -> list[str]:
    """Make the registered MCP tools match the currently configured models."""
    tool_defs = getattr(mcp, "_media_tool_defs", None)
    if tool_defs is None:
        return []

    desired = {
        name
        for name, spec in tool_defs.items()
        if spec["getter"]()
    }
    active = set(getattr(mcp, "_media_active_tools", set()))

    for name in sorted(active - desired):
        mcp.remove_tool(name)
    for name in sorted(desired - active):
        mcp.add_tool(tool_defs[name]["fn"])

    mcp._media_active_tools = desired
    return sorted(desired)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_media_mcp(
    router_url: str,
    image_model: MaybeGetter = None,
    audio_model: MaybeGetter = None,
    video_model: MaybeGetter = None,
    pdf_model: MaybeGetter = None,
) -> FastMCP:
    """Return a FastMCP instance whose tool list tracks the current config."""
    get_image = _make_getter(image_model)
    get_audio = _make_getter(audio_model)
    get_video = _make_getter(video_model)
    get_pdf = _make_getter(pdf_model)

    mcp = FastMCP(
        "media",
        instructions=(
            "Media understanding tools that call the router's Anthropic-compatible "
            "messages endpoint. The available tool list hot-reloads based on the "
            "configured image/audio/video/pdf models. Provide exactly one of path, "
            "url, or base64_data."
        ),
        stateless_http=True,
    )

    messages_url = router_url.rstrip("/") + "/v1/messages"

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
        return await _run_media_tool(
            messages_url, "image", get_image, path, url, base64_data, media_type, question, max_tokens
        )

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
        return await _run_media_tool(
            messages_url, "audio", get_audio, path, url, base64_data, media_type, question, max_tokens
        )

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
        return await _run_media_tool(
            messages_url, "video", get_video, path, url, base64_data, media_type, question, max_tokens
        )

    async def pdf_understanding(
        path: Optional[str] = None,
        url: Optional[str] = None,
        base64_data: Optional[str] = None,
        question: str = "Please read and summarize this PDF in detail.",
        max_tokens: int = 16384,
    ) -> str:
        """Read or answer questions about a PDF using a PDF-capable model.

        Provide **exactly one** of:
        - ``path``        — absolute or relative path to a local PDF file
        - ``url``         — HTTPS URL of a PDF
        - ``base64_data`` — raw base64-encoded PDF (without the ``data:`` prefix)

        ``question`` defaults to a summarization request.
        ``max_tokens`` controls response length (default: 16384).
        """
        return await _run_media_tool(
            messages_url,
            "pdf",
            get_pdf,
            path,
            url,
            base64_data,
            "application/pdf",
            question,
            max_tokens,
        )

    mcp._media_tool_defs = {
        "image_understanding": {"getter": get_image, "fn": image_understanding},
        "audio_understanding": {"getter": get_audio, "fn": audio_understanding},
        "video_understanding": {"getter": get_video, "fn": video_understanding},
        "pdf_understanding": {"getter": get_pdf, "fn": pdf_understanding},
    }
    mcp._media_active_tools = set()
    sync_media_mcp_tools(mcp)
    return mcp


# ---------------------------------------------------------------------------
# Entry point (standalone HTTP server)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Media MCP server (standalone Streamable HTTP mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image-model", default=None, help="Image model in provider/model format")
    parser.add_argument("--audio-model", default=None, help="Audio model in provider/model format")
    parser.add_argument("--video-model", default=None, help="Video model in provider/model format")
    parser.add_argument("--pdf-model", default=None, help="PDF model in provider/model format")
    parser.add_argument(
        "--router-url", default="http://127.0.0.1:8080",
        help="Base URL of the running router (default: http://127.0.0.1:8080)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8081, help="Port for the MCP Streamable HTTP server (default: 8081)")
    args = parser.parse_args(argv)

    media_mcp = create_media_mcp(
        router_url=args.router_url,
        image_model=args.image_model,
        audio_model=args.audio_model,
        video_model=args.video_model,
        pdf_model=args.pdf_model,
    )
    mcp_app = media_mcp.streamable_http_app()

    tools = sync_media_mcp_tools(media_mcp)
    print(f"Media MCP server → {args.host}:{args.port}/mcp  (tools: {', '.join(tools) or 'none'})")
    uvicorn.run(mcp_app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
