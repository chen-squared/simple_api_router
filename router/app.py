"""FastAPI application."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from router.config import RouterConfig
from router.endpoint import EndpointUnavailableError, UpstreamError
from router.group import Group, RoutingError, build_routing_tree
from router.logger import get_logger

log = get_logger("app")


def create_app(config: RouterConfig) -> FastAPI:
    app = FastAPI(title="simple_api_router", version="1.0.0")

    # Build routing tree
    nodes = build_routing_tree(config)
    default_group: Group = nodes[config.default_group]  # type: ignore[assignment]

    # Shared httpx client
    http_client = httpx.AsyncClient(timeout=120.0)

    @app.on_event("shutdown")
    async def shutdown():
        await http_client.aclose()

    # ------------------------------------------------------------------
    # OpenAI-compatible endpoint
    # ------------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def openai_completions(request: Request) -> Response:
        return await _handle_request(request, "openai", default_group, http_client)

    # ------------------------------------------------------------------
    # Anthropic-compatible endpoint
    # ------------------------------------------------------------------
    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        return await _handle_request(request, "anthropic", default_group, http_client)

    # ------------------------------------------------------------------
    # Stats endpoint
    # ------------------------------------------------------------------
    @app.get("/stats")
    async def stats() -> JSONResponse:
        return JSONResponse(
            {
                "default_group": config.default_group,
                "tree": default_group.stats(),
            }
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return app


async def _handle_request(
    request: Request,
    request_format: str,
    group: Group,
    client: httpx.AsyncClient,
) -> Response:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    log.info(
        "← request format=%s model=%s stream=%s",
        request_format,
        body.get("model", "?"),
        body.get("stream", False),
    )

    try:
        is_streaming, result = await group.call(body, request_format, client)
    except RoutingError as e:
        log.error("routing failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except EndpointUnavailableError as e:
        log.error("endpoint unavailable: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except UpstreamError as e:
        log.error("upstream error: %s", e)
        raise HTTPException(status_code=e.status or 502, detail=str(e))

    if is_streaming:
        media_type = (
            "text/event-stream"
            if request_format == "openai"
            else "text/event-stream"
        )
        return StreamingResponse(
            result,
            media_type=media_type,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    log.info("→ response format=%s", request_format)
    return JSONResponse(content=result)
