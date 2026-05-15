"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from watchfiles import awatch

from .config import RouterConfig, load_config
from .logger import setup_logging as setup_logger
from .proxy import route_request


async def _try_load_config(config_path: Path, logger, retries: int = 3, delay: float = 0.5):
    """Load config with retries to handle mid-write auto-saves.

    Editors often truncate then rewrite — the first parse attempt may hit an
    incomplete file. We retry a few times (with a short sleep) to let the
    editor finish before giving up and keeping the current config.
    Returns the new RouterConfig on success, or None if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        if attempt > 0:
            await asyncio.sleep(delay)
        try:
            return load_config(config_path)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                logger.debug(
                    "Config reload attempt %d/%d failed, retrying in %.1fs: %s",
                    attempt + 1, retries, delay, exc,
                )
    logger.warning("Config reload failed (keeping current config): %s", last_exc)
    return None


async def _watch_config(app: FastAPI, config_path: Path, logger) -> None:
    """Background task: reload config whenever config_path is saved."""
    try:
        async for _ in awatch(str(config_path)):
            new_config = await _try_load_config(config_path, logger)
            if new_config is not None:
                app.state.config = new_config
                logger.info("Config reloaded from %s", config_path)
    except asyncio.CancelledError:
        pass


def create_app(config: RouterConfig, config_path: Optional[Path] = None) -> FastAPI:
    logger = setup_logger(
        log_level=config.server.log_level,
        log_file=config.server.log_file,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        app.state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            follow_redirects=True,
        )
        app.state.start_time = time.time()
        logger.info("Router started — providers: %s", list(config.providers.keys()))

        watch_task = None
        if config_path is not None:
            watch_task = asyncio.create_task(
                _watch_config(app, config_path, logger),
                name="config-watcher",
            )
            logger.info("Watching config file for changes: %s", config_path)

        yield

        if watch_task is not None:
            watch_task.cancel()
            try:
                await watch_task
            except asyncio.CancelledError:
                pass
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
            config=request.app.state.config,
            client=request.app.state.http_client,
        )

    # ------------------------------------------------------------------
    # GET /v1/models  — list available models
    # ------------------------------------------------------------------
    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        cfg: RouterConfig = request.app.state.config
        model_data = []
        for prov_name, prov in cfg.providers.items():
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
        cfg: RouterConfig = request.app.state.config
        uptime = round(time.time() - request.app.state.start_time, 1)
        return JSONResponse({
            "status": "ok",
            "uptime_seconds": uptime,
            "providers": list(cfg.providers.keys()),
        })

    # ------------------------------------------------------------------
    # GET /stats
    # ------------------------------------------------------------------
    @app.get("/stats")
    async def stats(request: Request) -> JSONResponse:
        cfg: RouterConfig = request.app.state.config
        provider_info = {}
        for name, prov in cfg.providers.items():
            provider_info[name] = {
                "type": prov.type,
                "models": prov.models,
                "base_url": prov.resolve_base_url(),
            }
        return JSONResponse({"providers": provider_info})

    return app
