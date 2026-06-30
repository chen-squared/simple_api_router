"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import httpx
import yaml as _yaml_module
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from watchfiles import awatch

from .config import RouterConfig, load_config
from .debug_log import configure as configure_debug_log
from .logger import setup_logging as setup_logger
from .proxy import count_tokens_request, route_request
from .service import LOG_DIR
from .usage_cli import (
    _aggregate_by_day_model,
    _aggregate_by_model,
    _group_by_provider,
    _total_agg,
)
from .usage_db import get_usage_db, log_usage, setup_usage_db
from .web_ui import (
    STATIC_DIR,
    build_config_context,
    build_stats_context,
    stats_period_from_params,
    stats_query_params,
    stats_recent_model_index,
    templates,
)

# Re-exported for tests and backward compatibility.
_stats_period_from_params = stats_period_from_params
_stats_query_params = stats_query_params
_stats_recent_model_index = stats_recent_model_index


def _now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# Config data helpers (for /config API) — HTML/CSS/JS live in web_ui + templates.


def _config_to_data(cfg: RouterConfig) -> Dict[str, Any]:
    """Serialize a RouterConfig to a plain JSON-serialisable dict for the GUI."""
    s = cfg.server
    server_data = {
        "host": s.host,
        "port": s.port,
        "log_level": s.log_level,
        "log_file": s.log_file,
        "max_retries": s.max_retries,
        "multimodal_fallback_max_concurrency": s.multimodal_fallback_max_concurrency,
        "image_model": s.image_model,
        "audio_model": s.audio_model,
        "video_model": s.video_model,
        "image_fallback": s.image_fallback,
        "audio_fallback": s.audio_fallback,
        "video_fallback": s.video_fallback,
        "debug_log": s.debug_log,
    }
    providers_data: Dict[str, Any] = {}
    for pname, prov in cfg.providers.items():
        eps: Dict[str, Any] = {}
        for fmt, ep in prov.endpoints.items():
            models_list = []
            for m in ep.models:
                from .config import ModelEntry as _ME
                entry = m if isinstance(m, _ME) else _ME(name=str(m))
                models_list.append({
                    "name": entry.name,
                    "multimodality": list(entry.multimodality),
                    "image_fallback": entry.image_fallback,
                    "audio_fallback": entry.audio_fallback,
                    "video_fallback": entry.video_fallback,
                    "deepseek_reasoning": entry.deepseek_reasoning,
                    "max_reasoning_effort": entry.max_reasoning_effort,
                })
            eps[fmt] = {
                "base_url": ep.base_url,
                "deepseek_reasoning": ep.deepseek_reasoning,
                "max_reasoning_effort": ep.max_reasoning_effort,
                "models": models_list,
            }
        providers_data[pname] = {
            # Mask the key so it is never exposed in the HTTP response.
            # The GUI shows "***" and skips updating the key if unchanged.
            "api_key": "***" if prov.api_key else "",
            "base_url": prov.base_url,
            "endpoints": eps,
        }
    return {"server": server_data, "providers": providers_data}


def _patch_yaml(existing_yaml: str, gui_data: Dict[str, Any]) -> str:
    """Merge GUI changes into the existing parsed YAML dict.

    Unknown fields (e.g. pricing, model_map) are preserved because we only
    overwrite the keys the GUI explicitly manages.
    api_key is never overwritten when the GUI sends "***" (the masked value).
    """
    existing: Dict[str, Any] = _yaml_module.safe_load(existing_yaml) or {}

    # ── Server settings ──────────────────────────────────────────────────────
    gui_server = gui_data.get("server") or {}
    if gui_server:
        srv = existing.setdefault("server", {})
        for k, v in gui_server.items():
            if v is None or v == "":
                srv.pop(k, None)
            else:
                srv[k] = v
        if not srv:
            existing.pop("server", None)

    # ── Providers ────────────────────────────────────────────────────────────
    gui_provs = gui_data.get("providers") or {}
    for pname, gui_prov in gui_provs.items():
        provs = existing.setdefault("providers", {})
        if pname not in provs:
            continue  # Settings tab doesn't add/remove providers; use YAML tab
        ep_dict = provs[pname]

        # api_key — skip update when the GUI sent the masked placeholder
        gui_key = gui_prov.get("api_key")
        if gui_key and gui_key != "***":
            ep_dict["api_key"] = gui_key
        elif gui_key == "":
            ep_dict.pop("api_key", None)
        # gui_key == "***" → leave existing value untouched

        # base_url
        if "base_url" in gui_prov:
            if gui_prov["base_url"]:
                ep_dict["base_url"] = gui_prov["base_url"]
            else:
                ep_dict.pop("base_url", None)

        # endpoints
        gui_eps = gui_prov.get("endpoints") or {}
        for fmt, gui_ep in gui_eps.items():
            eps_section = ep_dict.setdefault("endpoints", {})
            if fmt not in eps_section:
                continue  # don't add new endpoint formats via Settings tab
            existing_ep = eps_section[fmt]
            if existing_ep is None:
                existing_ep = {}
                eps_section[fmt] = existing_ep

            # endpoint base_url
            if gui_ep.get("base_url"):
                existing_ep["base_url"] = gui_ep["base_url"]
            else:
                existing_ep.pop("base_url", None)

            # models — merge by name so pricing / other per-model fields survive
            gui_models_list = gui_ep.get("models") or []
            raw_existing: List[Any] = existing_ep.get("models") or []
            existing_by_name: Dict[str, Any] = {}
            for m in raw_existing:
                if isinstance(m, dict) and m.get("name"):
                    existing_by_name[m["name"]] = dict(m)
                elif isinstance(m, str):
                    existing_by_name[m] = m

            new_models: List[Any] = []
            for gm in gui_models_list:
                name = (gm.get("name") or "").strip()
                if not name:
                    continue
                base = existing_by_name.get(name)
                if isinstance(base, dict):
                    new_m: Dict[str, Any] = dict(base)
                else:
                    new_m = {"name": name}

                mm = gm.get("multimodality") or []
                if mm:
                    new_m["multimodality"] = mm
                else:
                    new_m.pop("multimodality", None)
                for fb in ("image_fallback", "audio_fallback", "video_fallback"):
                    if gm.get(fb):
                        new_m[fb] = gm[fb]
                    else:
                        new_m.pop(fb, None)

                other_keys = {k for k in new_m if k != "name"}
                new_models.append(name if not other_keys else new_m)

            if new_models:
                existing_ep["models"] = new_models
            else:
                existing_ep.pop("models", None)

    return _yaml_module.dump(existing, default_flow_style=False, allow_unicode=True, sort_keys=False)


async def _sse_with_usage(
    original: AsyncIterator[bytes],
    meta: dict,
    start: float,
    app_logger,
) -> AsyncIterator[bytes]:
    """Wrap a streaming SSE body, extracting token counts from Anthropic events."""
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    cache_write_tokens = 0

    try:
        async for chunk in original:
            yield chunk
            try:
                text = chunk.decode("utf-8", errors="replace")
                for line in text.splitlines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    data = json.loads(payload)
                    t = data.get("type")
                    if t == "message_start":
                        u = data.get("message", {}).get("usage", {})
                        input_tokens = u.get("input_tokens", 0)
                        cache_read_tokens = u.get("cache_read_input_tokens", 0)
                        cache_write_tokens = u.get("cache_creation_input_tokens", 0)
                    elif t == "message_delta":
                        u = data.get("usage", {})
                        output_tokens = u.get("output_tokens", 0)
                        # Converted streams (OpenAI/Google) put the real input count here
                        # because it's only known at end-of-stream.  Native Anthropic
                        # message_delta never contains input_tokens, so this is safe.
                        if "input_tokens" in u:
                            input_tokens = u["input_tokens"]
                        if "cache_read_input_tokens" in u:
                            cache_read_tokens = u["cache_read_input_tokens"]
                        if "cache_creation_input_tokens" in u:
                            cache_write_tokens = u["cache_creation_input_tokens"]
            except Exception:
                pass
    finally:
        duration_ms = round((time.time() - start) * 1000)
        if input_tokens == 0 and output_tokens == 0:
            app_logger.warning(
                "POST /v1/messages model=%s provider=%s backend=%s "
                "in=0 out=0 (no usage events received — backend may have returned an error "
                "or does not include usage in stream) streaming=true status=200 duration=%dms",
                meta["model"], meta["provider"], meta["backend_model"], duration_ms,
            )
        else:
            app_logger.info(
                "POST /v1/messages model=%s provider=%s backend=%s "
                "in=%d out=%d cache_r=%d cache_w=%d streaming=true status=200 duration=%dms",
                meta["model"], meta["provider"], meta["backend_model"],
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, duration_ms,
            )
        log_usage({
            "ts": _now_local(),
            "model": meta["model"],
            "provider": meta["provider"],
            "backend_model": meta["backend_model"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "streaming": True,
            "status": 200,
            "duration_ms": duration_ms,
        })


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


async def _watch_config(
    app: FastAPI,
    config_path: Path,
    logger,
    on_reload: Optional[Callable[[FastAPI], None]] = None,
) -> None:
    """Background task: reload config whenever config_path is saved."""
    try:
        async for _ in awatch(str(config_path)):
            new_config = await _try_load_config(config_path, logger)
            if new_config is not None:
                old_config = app.state.config
                app.state.config = new_config
                if on_reload is not None:
                    on_reload(app)
                _reapply_debug_log(old_config, new_config, logger)
                logger.info("Config reloaded from %s", config_path)
    except asyncio.CancelledError:
        pass


def _reapply_debug_log(old_config: Optional[RouterConfig], new_config: RouterConfig, logger) -> bool:
    """(Re)apply the debug_log setting from config. Enables, disables, or
    re-paths the debug logger. No-op (and silent) when the value is unchanged.
    Returns True if the setting changed.
    """
    old_debug = old_config.server.debug_log if old_config is not None else None
    new_debug = new_config.server.debug_log
    if new_debug == old_debug:
        return False
    configure_debug_log(new_debug)  # None/"" disables
    logger.info("Debug logging %s", f"enabled → {new_debug}" if new_debug else "disabled")
    return True


def create_app(config: RouterConfig, config_path: Optional[Path] = None) -> FastAPI:
    logger = setup_logger(
        log_level=config.server.log_level,
        log_file=config.server.log_file,
    )
    setup_usage_db(LOG_DIR / "router_usage.db")
    _reapply_debug_log(None, config, logger)

    # Create media MCP instance early (before lifespan) so its session
    # manager can be started inside the FastAPI lifespan context.
    # Use a list as a mutable reference so the lambdas pick up hot-reloaded
    # config values from app.state.config (set in lifespan, updated by watcher).
    from .mcp_media import create_media_mcp, sync_media_mcp_tools

    _app_ref: list = []  # populated below after app is created
    _media_mcp = create_media_mcp(
        router_url=f"http://127.0.0.1:{config.server.port}",
        image_model=lambda: (
            _app_ref[0].state.config.server.image_model
            if _app_ref else config.server.image_model
        ),
        audio_model=lambda: (
            _app_ref[0].state.config.server.audio_model
            if _app_ref else config.server.audio_model
        ),
        video_model=lambda: (
            _app_ref[0].state.config.server.video_model
            if _app_ref else config.server.video_model
        ),
        pdf_model=lambda: (
            _app_ref[0].state.config.server.pdf_model
            if _app_ref else config.server.pdf_model
        ),
    )
    _media_mcp_asgi = _media_mcp.streamable_http_app()
    active_media_tools = sync_media_mcp_tools(_media_mcp)
    logger.info("Media MCP ready at /mcp  (tools: %s)", ", ".join(active_media_tools) or "none")

    def _refresh_media_mcp(current_app: FastAPI) -> None:
        current_tools = sync_media_mcp_tools(_media_mcp)
        previous_tools = getattr(current_app.state, "media_mcp_tools", None)
        current_app.state.media_mcp_tools = current_tools
        if previous_tools is not None and previous_tools != current_tools:
            logger.info("Media MCP tools updated at /mcp  (tools: %s)", ", ".join(current_tools) or "none")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        app.state.media_mcp = _media_mcp
        app.state.media_mcp_tools = list(active_media_tools)
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
                _watch_config(app, config_path, logger, _refresh_media_mcp),
                name="config-watcher",
            )
            logger.info("Watching config file for changes: %s", config_path)

        async with _media_mcp.session_manager.run():
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
    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Any:
        body: Dict[str, Any] = await request.json()
        return await count_tokens_request(
            request=request,
            body=body,
            config=request.app.state.config,
            client=request.app.state.http_client,
        )

    # ------------------------------------------------------------------
    # POST /v1/messages  — main entry point
    # ------------------------------------------------------------------
    @app.post("/v1/messages")
    async def messages(request: Request) -> Any:
        body: Dict[str, Any] = await request.json()
        start = time.time()
        response = await route_request(
            request=request,
            body=body,
            config=request.app.state.config,
            client=request.app.state.http_client,
        )

        meta = getattr(request.state, "usage_meta", None)
        if meta:
            if isinstance(response, StreamingResponse):
                response.body_iterator = _sse_with_usage(
                    response.body_iterator, meta, start, logger
                )
            elif isinstance(response, JSONResponse):
                try:
                    data = json.loads(response.body)
                    u = data.get("usage", {})
                    in_tok = u.get("input_tokens", 0)
                    out_tok = u.get("output_tokens", 0)
                    cr_tok = u.get("cache_read_input_tokens", 0)
                    cw_tok = u.get("cache_creation_input_tokens", 0)
                    duration_ms = round((time.time() - start) * 1000)
                    logger.info(
                        "POST /v1/messages model=%s provider=%s backend=%s "
                        "in=%d out=%d cache_r=%d cache_w=%d streaming=false "
                        "status=%d duration=%dms",
                        meta["model"], meta["provider"], meta["backend_model"],
                        in_tok, out_tok, cr_tok, cw_tok,
                        response.status_code, duration_ms,
                    )
                    log_usage({
                        "ts": _now_local(),
                        "model": meta["model"],
                        "provider": meta["provider"],
                        "backend_model": meta["backend_model"],
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "cache_read_tokens": cr_tok,
                        "cache_write_tokens": cw_tok,
                        "streaming": False,
                        "status": response.status_code,
                        "duration_ms": duration_ms,
                    })
                except Exception:
                    pass

        return response

    # ------------------------------------------------------------------
    # GET /v1/models  — list available models
    # ------------------------------------------------------------------
    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        cfg: RouterConfig = request.app.state.config
        model_data = []
        for prov_name, prov in cfg.providers.items():
            for fmt, ep in prov.endpoints.items():
                for m in ep.model_names():
                    model_data.append({
                        "id": f"{prov_name}/{m}",
                        "object": "model",
                        "created": 0,
                        "owned_by": prov_name,
                    })
        for alias in cfg.server.model_map:
            model_data.append({
                "id": alias,
                "object": "model",
                "created": 0,
                "owned_by": "alias",
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
    # GET /stats/data
    # ------------------------------------------------------------------
    @app.get("/stats/data")
    async def stats_data(request: Request) -> JSONResponse:
        period = _stats_period_from_params(request.query_params)
        recent_provider = (request.query_params.get("provider") or "").strip()
        recent_model = (request.query_params.get("model") or "").strip()
        since_epoch = period["since_epoch"]
        until_epoch = period["until_epoch"]
        db = get_usage_db()
        config = request.app.state.config
        records = db.query_raw(since_epoch, until_epoch) if db is not None else []
        by_model_agg = _aggregate_by_model(records, config)
        by_day_agg = _aggregate_by_day_model(records, config)
        recent_model_index = _stats_recent_model_index(by_model_agg)
        if recent_provider and recent_model:
            valid_models = {item["value"] for item in recent_model_index.get(recent_provider, [])}
            if recent_model not in valid_models:
                recent_model = ""
        recent = (
            db.query_recent_filtered(
                50,
                since_epoch=since_epoch,
                until_epoch=until_epoch,
                provider=recent_provider or None,
                model=recent_model or None,
            )
            if db is not None
            else []
        )

        def _agg_row(model: str, agg: dict) -> dict:
            return {
                "model": model,
                "provider": agg.get("provider", model.split("/")[0] if "/" in model else None),
                "requests": agg["requests"],
                "input_tokens": agg["input_tokens"],
                "output_tokens": agg["output_tokens"],
                "cache_read_tokens": agg["cache_read_tokens"],
                "cache_write_tokens": agg["cache_write_tokens"],
                "cost_cny": round(agg["cost_cny"], 6) if agg.get("_has_cost_cny") else None,
                "cost_usd": round(agg["cost_usd"], 6) if agg.get("_has_cost_usd") else None,
            }

        by_model = [_agg_row(m, a) for m, a in sorted(by_model_agg.items(), key=lambda x: -x[1]["requests"])]
        by_day = []
        for day in sorted(by_day_agg.keys(), reverse=True):
            dt = _total_agg(by_day_agg[day])
            by_day.append({
                "day": day,
                "requests": dt["requests"],
                "input_tokens": dt["input_tokens"],
                "output_tokens": dt["output_tokens"],
                "cache_read_tokens": dt["cache_read_tokens"],
                "cache_write_tokens": dt["cache_write_tokens"],
                "cost_cny": round(dt["cost_cny"], 6) if dt.get("_has_cost_cny") else None,
                "cost_usd": round(dt["cost_usd"], 6) if dt.get("_has_cost_usd") else None,
            })
        by_provider_agg = _group_by_provider(by_model_agg)
        by_provider = []
        for prov in sorted(by_provider_agg):
            subtotal = _total_agg(by_provider_agg[prov])
            by_provider.append({
                "provider": prov,
                "requests": subtotal["requests"],
                "input_tokens": subtotal["input_tokens"],
                "output_tokens": subtotal["output_tokens"],
                "cache_read_tokens": subtotal["cache_read_tokens"],
                "cache_write_tokens": subtotal["cache_write_tokens"],
                "cost_cny": round(subtotal["cost_cny"], 6) if subtotal.get("_has_cost_cny") else None,
                "cost_usd": round(subtotal["cost_usd"], 6) if subtotal.get("_has_cost_usd") else None,
                "models": [_agg_row(m, a) for m, a in sorted(by_provider_agg[prov].items(), key=lambda x: -x[1]["requests"])],
            })
        return JSONResponse({
            "period_days": period["days"],
            "period": {
                "mode": period["mode"],
                "label": period["label"],
                "day": period["day"],
                "from": period["date_from"],
                "to": period["date_to"],
            },
            "recent_filters": {
                "provider": recent_provider or None,
                "model": recent_model or None,
            },
            "by_model": by_model,
            "by_provider": by_provider,
            "by_day": by_day,
            "recent": recent,
        })

    # ------------------------------------------------------------------
    # GET /stats
    # ------------------------------------------------------------------
    @app.get("/stats")
    async def stats(request: Request):
        period = stats_period_from_params(request.query_params)
        days = period["days"]
        try:
            page = max(1, int(request.query_params.get("page", "1") or "1"))
        except ValueError:
            page = 1
        view = request.query_params.get("view", "summary")
        recent_provider = (request.query_params.get("provider") or "").strip()
        recent_model = (request.query_params.get("model") or "").strip()
        page_size = 25

        since_epoch = period["since_epoch"]
        until_epoch = period["until_epoch"]
        db = get_usage_db()
        config = request.app.state.config
        records = db.query_raw(since_epoch, until_epoch) if db is not None else []
        by_model_agg = _aggregate_by_model(records, config)
        by_day_agg = _aggregate_by_day_model(records, config)
        recent_model_index = stats_recent_model_index(by_model_agg)
        if recent_provider and recent_model:
            valid_models = {item["value"] for item in recent_model_index.get(recent_provider, [])}
            if recent_model not in valid_models:
                recent_model = ""

        total_recent = (
            db.count_filtered(
                since_epoch=since_epoch,
                until_epoch=until_epoch,
                provider=recent_provider or None,
                model=recent_model or None,
            )
            if db is not None
            else 0
        )
        total_pages = max(1, (total_recent + page_size - 1) // page_size)
        page = min(page, total_pages)
        recent_offset = (page - 1) * page_size
        recent = (
            db.query_recent_filtered(
                page_size,
                recent_offset,
                since_epoch=since_epoch,
                until_epoch=until_epoch,
                provider=recent_provider or None,
                model=recent_model or None,
            )
            if db is not None
            else []
        )

        context = build_stats_context(
            period=period,
            days=days,
            view=view,
            page=page,
            recent_provider=recent_provider,
            recent_model=recent_model,
            by_model_agg=by_model_agg,
            by_day_agg=by_day_agg,
            recent=recent,
            config=config,
            total_recent=total_recent,
            total_pages=total_pages,
        )
        return templates.TemplateResponse(request, "stats.html", context)

    # ------------------------------------------------------------------
    # Config API  (GET/POST /config/data, POST /config/test, GET /config)
    # GET/POST /config/yaml  — raw YAML for YAML tab
    # ------------------------------------------------------------------

    @app.get("/config/data")
    async def config_data_get(request: Request) -> JSONResponse:
        cfg: RouterConfig = request.app.state.config
        return JSONResponse(_config_to_data(cfg))

    @app.post("/config/data")
    async def config_data_post(request: Request) -> JSONResponse:
        if config_path is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Config path unknown (server started without a config file)"},
            )
        data = await request.json()
        try:
            existing_yaml = config_path.read_text(encoding="utf-8")
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": f"Read error: {exc}"})
        try:
            new_yaml = _patch_yaml(existing_yaml, data)
            parsed = _yaml_module.safe_load(new_yaml)
            RouterConfig.model_validate(parsed)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": f"Invalid config: {exc}"})
        try:
            config_path.write_text(new_yaml, encoding="utf-8")
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": f"Write error: {exc}"})
        return JSONResponse({"ok": True})

    @app.post("/config/test")
    async def config_test(request: Request) -> JSONResponse:
        body: Dict[str, Any] = await request.json()
        model_str: str = body.get("model", "")
        if not model_str:
            return JSONResponse(status_code=400, content={"error": "model is required"})
        from .test_model import run_model_via_router_test

        transport = httpx.ASGITransport(app=request.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url=str(request.base_url).rstrip("/"),
            timeout=httpx.Timeout(60.0),
            follow_redirects=True,
        ) as client:
            result = await run_model_via_router_test(model_str, client)
        return JSONResponse(result)

    @app.get("/config/yaml")
    async def config_yaml_get(request: Request):
        if config_path is None:
            return JSONResponse(status_code=503, content={"error": "Config path unknown"})
        try:
            raw = config_path.read_text()
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})
        return PlainTextResponse(raw, media_type="text/plain; charset=utf-8")

    @app.post("/config/yaml")
    async def config_yaml_post(request: Request):
        if config_path is None:
            return JSONResponse(status_code=503, content={"error": "Config path unknown"})
        body_bytes = await request.body()
        new_yaml = body_bytes.decode("utf-8")
        try:
            parsed = _yaml_module.safe_load(new_yaml)
            if not isinstance(parsed, dict):
                raise ValueError("YAML must be a mapping at the top level")
            RouterConfig.model_validate(parsed)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": f"Invalid config: {exc}"})
        try:
            config_path.write_text(new_yaml, encoding="utf-8")
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": f"Write error: {exc}"})
        return JSONResponse({"ok": True})

    @app.get("/config")
    async def config_page(request: Request):
        cfg: RouterConfig = request.app.state.config
        context = build_config_context(cfg, config_path)
        return templates.TemplateResponse(request, "config.html", context)

    # ------------------------------------------------------------------
    # Media MCP — mount LAST so FastAPI routes take priority.
    # session_manager is started in lifespan above.
    # ------------------------------------------------------------------
    _app_ref.append(app)          # let the lambdas above read live config
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/", _media_mcp_asgi)

    return app
