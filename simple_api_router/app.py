"""FastAPI application factory."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import RouterConfig
from .logger import setup_logging as setup_logger
from .proxy import route_request


def create_app(config: RouterConfig) -> FastAPI:
    logger = setup_logger(
        log_level=config.server.log_level,
        log_file=config.server.log_file,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            follow_redirects=True,
        )
        app.state.start_time = time.time()
        logger.info("Router started — providers: %s", list(config.providers.keys()))
        yield
        await app.state.http_client.aclose()

    app = FastAPI(
        title="Simple API Router",
        description="Multi-provider LLM router exposing a unified Anthropic Messages API",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # POST /v1/messages  — main entry point
    # ------------------------------------------------------------------
    @app.post("/v1/messages")
    async def messages(request: Request) -> Any:
        body: Dict[str, Any] = await request.json()
        return await route_request(
            request=request,
            body=body,
            config=config,
            client=request.app.state.http_client,
        )

    # ------------------------------------------------------------------
    # GET /v1/models  — list available models
    # ------------------------------------------------------------------
    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        model_data = []
        for prov_name, prov in config.providers.items():
            for m in prov.models:
                model_data.append({
                    "id": f"{prov_name}/{m}",
                    "object": "model",
                    "created": 0,
                    "owned_by": prov_name,
                })
        return JSONResponse({"object": "list", "data": model_data})

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        uptime = round(time.time() - request.app.state.start_time, 1)
        return JSONResponse({
            "status": "ok",
            "uptime_seconds": uptime,
            "providers": list(config.providers.keys()),
        })

    # ------------------------------------------------------------------
    # GET /stats
    # ------------------------------------------------------------------
    @app.get("/stats")
    async def stats() -> JSONResponse:
        provider_info = {}
        for name, prov in config.providers.items():
            provider_info[name] = {
                "type": prov.type,
                "models": prov.models,
                "base_url": prov.resolve_base_url(),
            }
        return JSONResponse({"providers": provider_info})

    return app
