"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import html
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
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
    _record_cost_currency,
    _total_agg,
)
from .usage_db import get_usage_db, log_usage, setup_usage_db


def _now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_stats_days(raw: Optional[str]) -> int:
    try:
        days = int(raw or "7")
    except ValueError:
        days = 7
    return max(1, min(days, 90))


def _fmt_stat_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_stat_cost(value: Optional[float], symbol: str) -> str:
    if value in (None, 0):
        return "-"
    return f"{symbol}{value:.4f}"


def _sum_usage_rows(total_agg: dict) -> dict:
    return {
        "requests": total_agg["requests"],
        "input_tokens": total_agg["input_tokens"],
        "output_tokens": total_agg["output_tokens"],
        "cache_read_tokens": total_agg.get("cache_read_tokens", 0),
        "cache_write_tokens": total_agg.get("cache_write_tokens", 0),
        "cost_cny": total_agg["cost_cny"] if total_agg.get("_has_cost_cny") else None,
        "cost_usd": total_agg["cost_usd"] if total_agg.get("_has_cost_usd") else None,
    }


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
    setup_usage_db(LOG_DIR / "router_usage.db")
    if config.server.debug_log:
        configure_debug_log(config.server.debug_log)
        logger.info("Debug logging enabled → %s", config.server.debug_log)

    # Create media MCP instance early (before lifespan) so its session
    # manager can be started inside the FastAPI lifespan context.
    # Use a list as a mutable reference so the lambdas pick up hot-reloaded
    # config values from app.state.config (set in lifespan, updated by watcher).
    _media_mcp = None
    _app_ref: list = []  # populated below after app is created
    if any([config.server.image_model, config.server.audio_model, config.server.video_model]):
        from .mcp_media import create_media_mcp
        _media_mcp = create_media_mcp(
            router_url=f"http://127.0.0.1:{config.server.port}",
            image_model=lambda: (
                _app_ref[0].state.config.server.image_model
                if _app_ref else config.server.image_model
            ) if config.server.image_model else None,
            audio_model=lambda: (
                _app_ref[0].state.config.server.audio_model
                if _app_ref else config.server.audio_model
            ) if config.server.audio_model else None,
            video_model=lambda: (
                _app_ref[0].state.config.server.video_model
                if _app_ref else config.server.video_model
            ) if config.server.video_model else None,
        )
        # Call streamable_http_app() now to initialise the session_manager.
        _media_mcp_asgi = _media_mcp.streamable_http_app()
        active = [t for t, m in [("image", config.server.image_model), ("audio", config.server.audio_model), ("video", config.server.video_model)] if m]
        logger.info("Media MCP ready at /mcp  (tools: %s)", ", ".join(active))

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

        if _media_mcp is not None:
            async with _media_mcp.session_manager.run():
                yield
        else:
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
        days = _parse_stats_days(request.query_params.get("days"))
        from datetime import datetime as _dt, timedelta as _td
        _today_midnight = _dt.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        since_epoch = (_today_midnight - _td(days=days - 1)).timestamp()
        until_epoch = time.time()
        db = get_usage_db()
        config = request.app.state.config
        records = db.query_raw(since_epoch, until_epoch) if db is not None else []
        by_model_agg = _aggregate_by_model(records, config)
        by_day_agg = _aggregate_by_day_model(records, config)
        recent = db.query_recent(50) if db is not None else []

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
            "period_days": days,
            "by_model": by_model,
            "by_provider": by_provider,
            "by_day": by_day,
            "recent": recent,
        })

    # ------------------------------------------------------------------
    # GET /stats
    # ------------------------------------------------------------------
    @app.get("/stats")
    async def stats(request: Request) -> HTMLResponse:  # noqa: C901
        days = _parse_stats_days(request.query_params.get("days"))
        try:
            page = max(1, int(request.query_params.get("page", "1") or "1"))
        except ValueError:
            page = 1
        view = request.query_params.get("view", "summary")  # "summary" | "daily"
        page_size = 25

        from datetime import datetime as _dt, timedelta as _td
        _today_midnight = _dt.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        since_epoch = (_today_midnight - _td(days=days - 1)).timestamp()
        until_epoch = time.time()
        db = get_usage_db()
        config = request.app.state.config
        records = db.query_raw(since_epoch, until_epoch) if db is not None else []
        by_model_agg = _aggregate_by_model(records, config)
        by_day_agg = _aggregate_by_day_model(records, config)

        total_recent = db.count_all() if db is not None else 0
        recent_offset = (page - 1) * page_size
        recent = db.query_recent(page_size, recent_offset) if db is not None else []
        total_pages = max(1, (total_recent + page_size - 1) // page_size)

        total = _total_agg(by_model_agg)
        summary = _sum_usage_rows(total)

        def _url(*, d=None, v=None, p=None) -> str:
            nd = d if d is not None else days
            nv = v if v is not None else view
            np = p if p is not None else page
            return f"/stats?days={nd}&view={nv}&page={np}"

        def period_link(label: str, value: int) -> str:
            cls = "tab active" if value == days and view == "summary" else "tab"
            return f'<a class="{cls}" href="{_url(d=value, v="summary", p=1)}">{label}</a>'

        def daily_link(label: str, target_view: str) -> str:
            cls = "tab active" if view == target_view else "tab"
            return f'<a class="{cls}" href="{_url(v=target_view, p=1)}">{label}</a>'

        def by_model_rows() -> str:
            if not by_model_agg:
                return '<tr><td colspan="8" class="empty">No usage data</td></tr>'
            cells = []
            grouped = _group_by_provider(by_model_agg)
            for prov in sorted(grouped, key=lambda p: -_total_agg(grouped[p])["requests"]):
                sub = _total_agg(grouped[prov])
                sub_cny = round(sub["cost_cny"], 4) if sub.get("_has_cost_cny") else None
                sub_usd = round(sub["cost_usd"], 4) if sub.get("_has_cost_usd") else None
                cells.append(
                    '<tr class="prov-hdr">'
                    f"<td><strong>{html.escape(prov)}</strong></td>"
                    f"<td>{sub['requests']}</td>"
                    f"<td>{_fmt_stat_tokens(sub['input_tokens'])}</td>"
                    f"<td>{_fmt_stat_tokens(sub['output_tokens'])}</td>"
                    f"<td>{_fmt_stat_tokens(sub['cache_write_tokens'])}</td>"
                    f"<td>{_fmt_stat_tokens(sub['cache_read_tokens'])}</td>"
                    f"<td>{_fmt_stat_cost(sub_cny, '¥')}</td>"
                    f"<td>{_fmt_stat_cost(sub_usd, '$')}</td>"
                    "</tr>"
                )
                for model, agg in sorted(grouped[prov].items(), key=lambda x: -x[1]["requests"]):
                    model_label = model.split("/", 1)[1] if "/" in model else model
                    cost_cny = round(agg["cost_cny"], 4) if agg.get("_has_cost_cny") else None
                    cost_usd = round(agg["cost_usd"], 4) if agg.get("_has_cost_usd") else None
                    cells.append(
                        "<tr>"
                        f"<td>&ensp;{html.escape(model_label)}</td>"
                        f"<td>{agg['requests']}</td>"
                        f"<td>{_fmt_stat_tokens(agg['input_tokens'])}</td>"
                        f"<td>{_fmt_stat_tokens(agg['output_tokens'])}</td>"
                        f"<td>{_fmt_stat_tokens(agg['cache_write_tokens'])}</td>"
                        f"<td>{_fmt_stat_tokens(agg['cache_read_tokens'])}</td>"
                        f"<td>{_fmt_stat_cost(cost_cny, '¥')}</td>"
                        f"<td>{_fmt_stat_cost(cost_usd, '$')}</td>"
                        "</tr>"
                    )
            return "".join(cells)

        def by_day_rows() -> str:
            if not by_day_agg:
                return '<tr><td colspan="8" class="empty">No usage data</td></tr>'
            cells = []
            for day in sorted(by_day_agg.keys(), reverse=True):
                dt = _total_agg(by_day_agg[day])
                cost_cny = round(dt["cost_cny"], 4) if dt.get("_has_cost_cny") else None
                cost_usd = round(dt["cost_usd"], 4) if dt.get("_has_cost_usd") else None
                cells.append(
                    "<tr>"
                    f"<td>{html.escape(day)}</td>"
                    f"<td>{dt['requests']}</td>"
                    f"<td>{_fmt_stat_tokens(dt['input_tokens'])}</td>"
                    f"<td>{_fmt_stat_tokens(dt['output_tokens'])}</td>"
                    f"<td>{_fmt_stat_tokens(dt['cache_write_tokens'])}</td>"
                    f"<td>{_fmt_stat_tokens(dt['cache_read_tokens'])}</td>"
                    f"<td>{_fmt_stat_cost(cost_cny, '¥')}</td>"
                    f"<td>{_fmt_stat_cost(cost_usd, '$')}</td>"
                    "</tr>"
                )
            return "".join(cells)

        def daily_detail_rows() -> str:
            if not by_day_agg:
                return '<tr><td colspan="8" class="empty">No usage data</td></tr>'
            cells = []
            for day in sorted(by_day_agg.keys(), reverse=True):
                dt = _total_agg(by_day_agg[day])
                day_cny = round(dt["cost_cny"], 4) if dt.get("_has_cost_cny") else None
                day_usd = round(dt["cost_usd"], 4) if dt.get("_has_cost_usd") else None
                cells.append(
                    f'<tr class="day-hdr">'
                    f'<td colspan="8"><strong>{html.escape(day)}</strong>'
                    f'&emsp;<span class="muted">{dt["requests"]} req'
                    f'&ensp;in {_fmt_stat_tokens(dt["input_tokens"])}'
                    f'&ensp;out {_fmt_stat_tokens(dt["output_tokens"])}'
                    + (f'&ensp;{_fmt_stat_cost(day_cny, "¥")}' if day_cny is not None else '')
                    + (f'&ensp;{_fmt_stat_cost(day_usd, "$")}' if day_usd is not None else '')
                    + f'</span></td></tr>'
                )
                for model, agg in sorted(by_day_agg[day].items(), key=lambda x: -x[1]["requests"]):
                    cost_cny = round(agg["cost_cny"], 4) if agg.get("_has_cost_cny") else None
                    cost_usd = round(agg["cost_usd"], 4) if agg.get("_has_cost_usd") else None
                    cells.append(
                        "<tr>"
                        f"<td>&ensp;{html.escape(model)}</td>"
                        f"<td>{agg['requests']}</td>"
                        f"<td>{_fmt_stat_tokens(agg['input_tokens'])}</td>"
                        f"<td>{_fmt_stat_tokens(agg['output_tokens'])}</td>"
                        f"<td>{_fmt_stat_tokens(agg['cache_write_tokens'])}</td>"
                        f"<td>{_fmt_stat_tokens(agg['cache_read_tokens'])}</td>"
                        f"<td>{_fmt_stat_cost(cost_cny, '¥')}</td>"
                        f"<td>{_fmt_stat_cost(cost_usd, '$')}</td>"
                        "</tr>"
                    )
            return "".join(cells)

        def recent_rows() -> str:
            if not recent:
                return '<tr><td colspan="10" class="empty">No usage data</td></tr>'
            cells = []
            for row in recent:
                result = _record_cost_currency(row, config)
                if result:
                    cost, currency = result
                    cost_str = f"¥{cost:.4f}" if currency.upper() != "USD" else f"${cost:.4f}"
                else:
                    cost_str = "-"
                cells.append(
                    "<tr>"
                    f"<td>{html.escape(row.get('ts', ''))}</td>"
                    f"<td>{html.escape(row.get('model', ''))}</td>"
                    f"<td>{_fmt_stat_tokens(row.get('input_tokens', 0))}</td>"
                    f"<td>{_fmt_stat_tokens(row.get('output_tokens', 0))}</td>"
                    f"<td>{_fmt_stat_tokens(row.get('cache_write_tokens', 0))}</td>"
                    f"<td>{_fmt_stat_tokens(row.get('cache_read_tokens', 0))}</td>"
                    f"<td>{cost_str}</td>"
                    f"<td>{'✓' if row.get('streaming') else '—'}</td>"
                    f"<td>{row.get('status', 0)}</td>"
                    f"<td>{row.get('duration_ms', 0)}</td>"
                    "</tr>"
                )
            return "".join(cells)

        def pagination() -> str:
            if total_pages <= 1:
                return ""
            prev_href = _url(p=page - 1) if page > 1 else ""
            next_href = _url(p=page + 1) if page < total_pages else ""
            first_btn = f'<a class="page-btn" href="{_url(p=1)}">«</a>' if page > 1 else '<span class="page-btn disabled">«</span>'
            prev_btn = f'<a class="page-btn" href="{prev_href}">‹ Prev</a>' if prev_href else '<span class="page-btn disabled">‹ Prev</span>'
            next_btn = f'<a class="page-btn" href="{next_href}">Next ›</a>' if next_href else '<span class="page-btn disabled">Next ›</span>'
            last_btn = f'<a class="page-btn" href="{_url(p=total_pages)}">»</a>' if page < total_pages else '<span class="page-btn disabled">»</span>'
            return f'<div class="pagination">{first_btn}{prev_btn}<span class="page-info">Page {page} / {total_pages} ({total_recent} total)</span>{next_btn}{last_btn}</div>'

        main_section = ""
        if view == "daily":
            main_section = f"""
  <section class="panel">
    <h2>By Day — Model Breakdown</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Day / Model</th><th>Req</th><th>Input</th><th>Output</th><th>Cache↑</th><th>Cache↓</th><th>¥ Cost</th><th>$ Cost</th></tr></thead>
        <tbody>{daily_detail_rows()}</tbody>
      </table>
    </div>
  </section>"""
        else:
            main_section = f"""
  <section class="panel">
    <h2>By Model</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th>Req</th><th>Input</th><th>Output</th><th>Cache↑</th><th>Cache↓</th><th>¥ Cost</th><th>$ Cost</th></tr></thead>
        <tbody>{by_model_rows()}</tbody>
      </table>
    </div>
  </section>

  <section class="panel">
    <h2>By Day</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Day</th><th>Req</th><th>Input</th><th>Output</th><th>Cache↑</th><th>Cache↓</th><th>¥ Cost</th><th>$ Cost</th></tr></thead>
        <tbody>{by_day_rows()}</tbody>
      </table>
    </div>
  </section>"""

        page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Router Stats</title>
  <style>
    :root {{ color-scheme: dark light; }}
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #111827; color: #e5e7eb; }}
    a {{ color: inherit; text-decoration: none; }}
    .top {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
    .ctrl-row {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
    .tab-group {{ display: flex; gap: 6px; align-items: center; }}
    .tab {{ border: 1px solid #374151; border-radius: 999px; padding: 5px 10px; color: #cbd5e1; font-size: 13px; }}
    .tab.active {{ background: #2563eb; border-color: #2563eb; color: #fff; }}
    .tab-sep {{ color: #4b5563; padding: 0 4px; }}
    .days-form {{ display: inline-flex; gap: 6px; align-items: center; }}
    .days-form input {{ width: 54px; padding: 4px 6px; border: 1px solid #374151; border-radius: 6px; background: #1f2937; color: #e5e7eb; font-size: 13px; }}
    .days-form button {{ padding: 4px 10px; border: 1px solid #374151; border-radius: 6px; background: #1f2937; color: #cbd5e1; cursor: pointer; font-size: 13px; }}
    .days-form button:hover {{ border-color: #2563eb; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 18px 0 24px; }}
    .card, .panel {{ background: #111827; border: 1px solid #374151; border-radius: 12px; }}
    .card {{ padding: 14px; }}
    .label {{ font-size: 12px; color: #94a3b8; margin-bottom: 6px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 22px; font-weight: 600; }}
    .panel {{ margin-top: 18px; overflow: hidden; }}
    .panel h2 {{ margin: 0; padding: 14px 16px; font-size: 15px; border-bottom: 1px solid #374151; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 12px; border-bottom: 1px solid #1f2937; text-align: left; white-space: nowrap; }}
    th {{ color: #94a3b8; font-weight: 600; }}
    tr:last-child td {{ border-bottom: 0; }}
    tr.day-hdr td {{ background: #1a2236; color: #93c5fd; padding: 10px 12px 6px; }}
    tr.prov-hdr td {{ background: #1a2236; color: #7dd3fc; padding: 8px 12px; }}
    .empty {{ color: #94a3b8; text-align: center; }}
    .muted {{ color: #94a3b8; font-size: 13px; }}
    .pagination {{ display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-top: 1px solid #1f2937; }}
    .page-btn {{ border: 1px solid #374151; border-radius: 6px; padding: 5px 10px; font-size: 13px; color: #cbd5e1; }}
    .page-btn:hover {{ border-color: #2563eb; }}
    .page-btn.disabled {{ color: #4b5563; border-color: #1f2937; cursor: default; pointer-events: none; }}
    .page-info {{ font-size: 13px; color: #94a3b8; }}
  </style>
</head>
<body>
  <div class="top">
    <div>
      <h1 style="margin:0 0 6px 0;font-size:28px;">Usage Stats</h1>
      <div class="muted">Last {days} day{'s' if days != 1 else ''} · <a href="/stats/data?days={days}">JSON data</a></div>
    </div>
  </div>
  <div class="ctrl-row">
    <div class="tab-group">
      {period_link('Today', 1)}{period_link('7 days', 7)}{period_link('30 days', 30)}
      <form class="days-form" method="GET" action="/stats">
        <input type="hidden" name="view" value="{view}">
        <input type="number" name="days" value="{days}" min="1" max="365" placeholder="days">
        <button type="submit">Go</button>
      </form>
    </div>
    <div class="tab-group">
      <span class="tab-sep">View:</span>
      {daily_link('Summary', 'summary')}{daily_link('Daily', 'daily')}
    </div>
  </div>

  <section class="summary">
    <div class="card"><div class="label">Requests</div><div class="value">{summary['requests']}</div></div>
    <div class="card"><div class="label">Input Tokens</div><div class="value">{_fmt_stat_tokens(summary['input_tokens'])}</div></div>
    <div class="card"><div class="label">Output Tokens</div><div class="value">{_fmt_stat_tokens(summary['output_tokens'])}</div></div>
    <div class="card"><div class="label">Cache Write</div><div class="value">{_fmt_stat_tokens(summary['cache_write_tokens'])}</div></div>
    <div class="card"><div class="label">Cache Read</div><div class="value">{_fmt_stat_tokens(summary['cache_read_tokens'])}</div></div>
    <div class="card"><div class="label">¥ Cost</div><div class="value">{_fmt_stat_cost(summary['cost_cny'], '¥')}</div></div>
    <div class="card"><div class="label">$ Cost</div><div class="value">{_fmt_stat_cost(summary['cost_usd'], '$')}</div></div>
  </section>
{main_section}
  <section class="panel">
    <h2>Recent Requests</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Time</th><th>Model</th><th>In</th><th>Out</th><th>Cache↑</th><th>Cache↓</th><th>Cost</th><th>Stream</th><th>Status</th><th>ms</th></tr></thead>
        <tbody>{recent_rows()}</tbody>
      </table>
    </div>
    {pagination()}
  </section>
</body>
</html>
"""
        return HTMLResponse(page_html)

    # ------------------------------------------------------------------
    # GET /config/yaml  — return raw YAML config
    # POST /config/yaml — save new YAML config
    # POST /config/test — quick-test a model
    # GET  /config      — HTML config editor page
    # ------------------------------------------------------------------

    @app.get("/config/yaml")
    async def config_yaml_get(request: Request):
        if config_path is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Config path unknown (server started without --config file path)"},
            )
        try:
            raw = config_path.read_text()
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(raw, media_type="text/plain; charset=utf-8")

    @app.post("/config/yaml")
    async def config_yaml_post(request: Request):
        if config_path is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Config path unknown (server started without --config file path)"},
            )
        body_bytes = await request.body()
        new_yaml = body_bytes.decode("utf-8")
        # Validate before saving
        import yaml as _yaml
        try:
            parsed = _yaml.safe_load(new_yaml)
            if not isinstance(parsed, dict):
                raise ValueError("YAML must be a mapping at the top level")
            from .config import RouterConfig as _RC
            _RC.model_validate(parsed)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": f"Invalid config: {exc}"})
        try:
            config_path.write_text(new_yaml, encoding="utf-8")
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": f"Write error: {exc}"})
        return JSONResponse({"ok": True, "message": "Config saved. Reload will apply automatically."})

    @app.post("/config/test")
    async def config_test(request: Request):
        body: Dict[str, Any] = await request.json()
        model_str: str = body.get("model", "")
        if not model_str:
            return JSONResponse(status_code=400, content={"error": "model is required"})
        cfg: RouterConfig = request.app.state.config
        client: httpx.AsyncClient = request.app.state.http_client
        from .test_model import test_model_direct
        result = await test_model_direct(model_str, cfg, client)
        return JSONResponse(result)

    @app.get("/config")
    async def config_page(request: Request) -> HTMLResponse:  # noqa: C901
        cfg: RouterConfig = request.app.state.config

        def _fmt_multimodal(entry) -> str:
            mm = getattr(entry, "multimodality", []) or []
            if not mm:
                return '<span class="badge badge-grey">text</span>'
            icons = {"image": "🖼", "audio": "🎵", "video": "🎬"}
            return " ".join(
                f'<span class="badge badge-blue">{icons.get(m, m)}{m}</span>'
                for m in mm
            )

        model_rows = []
        for prov_name, prov in cfg.providers.items():
            for fmt, ep in prov.endpoints.items():
                for m in ep.models:
                    from .config import ModelEntry as _ME
                    entry = m if isinstance(m, _ME) else _ME(name=str(m))
                    full_id = f"{prov_name}/{entry.name}"
                    model_rows.append(
                        f'<tr data-model="{html.escape(full_id)}">'
                        f'<td class="prov-cell">{html.escape(prov_name)}</td>'
                        f'<td>{html.escape(entry.name)}</td>'
                        f'<td><code class="fmt">{html.escape(fmt)}</code></td>'
                        f'<td>{_fmt_multimodal(entry)}</td>'
                        f'<td class="test-cell">'
                        f'  <button class="test-btn" onclick="testModel(this, \'{html.escape(full_id)}\')">'
                        f"    Test"
                        f"  </button>"
                        f'  <span class="test-result"></span>'
                        f"</td>"
                        f"</tr>"
                    )

        model_table = "\n".join(model_rows) or '<tr><td colspan="5" class="empty">No models configured</td></tr>'
        has_config_path = config_path is not None
        config_path_str = str(config_path) if config_path else "(unknown)"
        editor_disabled = "" if has_config_path else "disabled"
        editor_notice = "" if has_config_path else '<p class="notice warn">Config path unknown — editor disabled.</p>'

        page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Router Config</title>
  <style>
    :root {{ color-scheme: dark light; }}
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #111827; color: #e5e7eb; }}
    a {{ color: #93c5fd; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    h1 {{ margin: 0 0 4px; font-size: 26px; }}
    .subhead {{ color: #94a3b8; font-size: 13px; margin-bottom: 18px; }}
    .nav {{ display: flex; gap: 12px; margin-bottom: 24px; align-items: center; }}
    .nav a {{ border: 1px solid #374151; border-radius: 999px; padding: 5px 14px; color: #cbd5e1; font-size: 13px; }}
    .nav a:hover {{ border-color: #2563eb; text-decoration: none; }}
    .nav a.active {{ background: #2563eb; border-color: #2563eb; color: #fff; }}
    .layout {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} }}
    .panel {{ background: #111827; border: 1px solid #374151; border-radius: 12px; overflow: hidden; }}
    .panel-header {{ padding: 14px 16px; border-bottom: 1px solid #374151; display: flex; align-items: center; justify-content: space-between; }}
    .panel-header h2 {{ margin: 0; font-size: 15px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 12px; border-bottom: 1px solid #1f2937; text-align: left; white-space: nowrap; }}
    th {{ color: #94a3b8; font-weight: 600; }}
    tr:last-child td {{ border-bottom: 0; }}
    .prov-cell {{ color: #93c5fd; font-weight: 500; }}
    .fmt {{ background: #1f2937; padding: 2px 6px; border-radius: 4px; font-size: 11px; color: #94a3b8; }}
    .badge {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 11px; }}
    .badge-grey {{ background: #1f2937; color: #94a3b8; }}
    .badge-blue {{ background: #1e3a5f; color: #93c5fd; }}
    .empty {{ color: #94a3b8; text-align: center; }}
    .test-btn {{
      padding: 3px 10px; font-size: 12px; border-radius: 6px; cursor: pointer;
      border: 1px solid #374151; background: #1f2937; color: #cbd5e1;
    }}
    .test-btn:hover {{ border-color: #2563eb; color: #93c5fd; }}
    .test-btn:disabled {{ opacity: 0.5; cursor: default; }}
    .test-result {{ font-size: 12px; margin-left: 8px; }}
    .test-result.ok {{ color: #4ade80; }}
    .test-result.err {{ color: #f87171; }}
    .test-result.loading {{ color: #94a3b8; }}
    .editor-wrap {{ padding: 16px; display: flex; flex-direction: column; gap: 12px; }}
    textarea#yaml-editor {{
      width: 100%; box-sizing: border-box; height: 560px; resize: vertical;
      font-family: 'SF Mono', 'Fira Code', Consolas, monospace; font-size: 12px;
      background: #0d1117; color: #e6edf3; border: 1px solid #374151;
      border-radius: 8px; padding: 12px; outline: none; tab-size: 2;
    }}
    textarea#yaml-editor:focus {{ border-color: #2563eb; }}
    .btn-row {{ display: flex; gap: 10px; align-items: center; }}
    .btn {{
      padding: 7px 18px; font-size: 13px; border-radius: 8px; cursor: pointer;
      border: 1px solid #374151; background: #1f2937; color: #cbd5e1;
    }}
    .btn-primary {{ background: #2563eb; border-color: #2563eb; color: #fff; font-weight: 500; }}
    .btn:hover {{ opacity: 0.85; }}
    .btn:disabled {{ opacity: 0.5; cursor: default; }}
    .save-status {{ font-size: 13px; }}
    .save-status.ok {{ color: #4ade80; }}
    .save-status.err {{ color: #f87171; }}
    .notice {{ font-size: 13px; padding: 8px 12px; border-radius: 8px; margin: 0; }}
    .notice.warn {{ background: #422006; color: #fbbf24; border: 1px solid #78350f; }}
    pre.preview {{ background: #1f2937; padding: 8px 12px; border-radius: 6px; font-size: 11px; color: #94a3b8; margin: 4px 0 0; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  </style>
</head>
<body>
  <h1>Simple API Router</h1>
  <p class="subhead">Config: <code>{html.escape(config_path_str)}</code></p>
  <nav class="nav">
    <a href="/stats">📊 Stats</a>
    <a href="/config" class="active">⚙ Config</a>
    <a href="/health">🩺 Health</a>
  </nav>

  <div class="layout">
    <!-- ── Left: model list ───────────────────────────────────────── -->
    <div class="panel">
      <div class="panel-header">
        <h2>Models</h2>
        <button class="btn" onclick="testAll()" id="test-all-btn">Test All</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Provider</th><th>Model</th><th>Format</th><th>Modality</th><th>Test</th>
            </tr>
          </thead>
          <tbody>
{model_table}
          </tbody>
        </table>
      </div>
    </div>

    <!-- ── Right: YAML editor ─────────────────────────────────────── -->
    <div class="panel">
      <div class="panel-header">
        <h2>Config Editor</h2>
        <span class="save-status" id="save-status"></span>
      </div>
      <div class="editor-wrap">
        {editor_notice}
        <textarea id="yaml-editor" spellcheck="false" {editor_disabled}
          placeholder="Loading config..."></textarea>
        <div class="btn-row">
          <button class="btn btn-primary" id="save-btn" onclick="saveConfig()" {editor_disabled}>
            Save Config
          </button>
          <button class="btn" id="reload-btn" onclick="loadConfig()" {editor_disabled}>
            ↺ Reload
          </button>
          <span style="color:#94a3b8;font-size:12px">Config reloads automatically after save.</span>
        </div>
      </div>
    </div>
  </div>

  <script>
    // ── load config YAML ─────────────────────────────────────────────────
    async function loadConfig() {{
      const ta = document.getElementById('yaml-editor');
      const btn = document.getElementById('reload-btn');
      if (!ta || ta.disabled) return;
      btn.disabled = true;
      btn.textContent = '↺ Loading…';
      try {{
        const r = await fetch('/config/yaml');
        if (r.ok) {{
          ta.value = await r.text();
          setStatus('', '');
        }} else {{
          const d = await r.json().catch(() => ({{}}));
          setStatus('err', 'Load failed: ' + (d.error || r.status));
        }}
      }} catch(e) {{
        setStatus('err', 'Load failed: ' + e.message);
      }} finally {{
        btn.disabled = false;
        btn.textContent = '↺ Reload';
      }}
    }}

    // ── save config YAML ─────────────────────────────────────────────────
    async function saveConfig() {{
      const ta = document.getElementById('yaml-editor');
      const btn = document.getElementById('save-btn');
      if (!ta || ta.disabled) return;
      btn.disabled = true;
      btn.textContent = 'Saving…';
      setStatus('', 'Saving…');
      try {{
        const r = await fetch('/config/yaml', {{
          method: 'POST',
          headers: {{'Content-Type': 'text/plain; charset=utf-8'}},
          body: ta.value,
        }});
        const d = await r.json().catch(() => ({{}}));
        if (r.ok) {{
          setStatus('ok', '✓ Saved');
        }} else {{
          setStatus('err', '✗ ' + (d.error || r.status));
        }}
      }} catch(e) {{
        setStatus('err', '✗ ' + e.message);
      }} finally {{
        btn.disabled = false;
        btn.textContent = 'Save Config';
      }}
    }}

    function setStatus(cls, msg) {{
      const el = document.getElementById('save-status');
      el.className = 'save-status' + (cls ? ' ' + cls : '');
      el.textContent = msg;
    }}

    // ── test a single model ───────────────────────────────────────────────
    async function testModel(btn, model) {{
      const resultEl = btn.parentElement.querySelector('.test-result');
      btn.disabled = true;
      resultEl.className = 'test-result loading';
      resultEl.textContent = '…testing';
      try {{
        const r = await fetch('/config/test', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{model}}),
        }});
        const d = await r.json();
        if (d.success) {{
          resultEl.className = 'test-result ok';
          const ms = d.latency_ms != null ? d.latency_ms + 'ms' : '';
          resultEl.textContent = '✓ ' + ms;
          if (d.response_preview) {{
            const pre = document.createElement('pre');
            pre.className = 'preview';
            pre.textContent = d.response_preview;
            resultEl.appendChild(pre);
          }}
        }} else {{
          resultEl.className = 'test-result err';
          const ms = d.latency_ms != null ? ' ' + d.latency_ms + 'ms' : '';
          resultEl.textContent = '✗' + ms + ' ' + (d.error || 'error');
        }}
      }} catch(e) {{
        resultEl.className = 'test-result err';
        resultEl.textContent = '✗ ' + e.message;
      }} finally {{
        btn.disabled = false;
      }}
    }}

    // ── test all models sequentially ──────────────────────────────────────
    async function testAll() {{
      const allBtn = document.getElementById('test-all-btn');
      allBtn.disabled = true;
      const rows = document.querySelectorAll('tr[data-model]');
      for (const row of rows) {{
        const btn = row.querySelector('.test-btn');
        const model = row.getAttribute('data-model');
        if (btn && model) await testModel(btn, model);
      }}
      allBtn.disabled = false;
    }}

    // ── handle Tab key in textarea ────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {{
      loadConfig();
      const ta = document.getElementById('yaml-editor');
      if (ta && !ta.disabled) {{
        ta.addEventListener('keydown', (e) => {{
          if (e.key === 'Tab') {{
            e.preventDefault();
            const s = ta.selectionStart, end = ta.selectionEnd;
            ta.value = ta.value.substring(0, s) + '  ' + ta.value.substring(end);
            ta.selectionStart = ta.selectionEnd = s + 2;
          }}
        }});
      }}
    }});
  </script>
</body>
</html>
"""
        return HTMLResponse(page_html)

    # ------------------------------------------------------------------
    # Media MCP — mount LAST so FastAPI routes take priority.
    # session_manager is started in lifespan above.
    # ------------------------------------------------------------------
    if _media_mcp is not None:
        _app_ref.append(app)          # let the lambdas above read live config
        app.mount("/", _media_mcp_asgi)

    return app
