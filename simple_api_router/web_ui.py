"""Web UI: Jinja2 templates, static assets, and page context builders."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from fastapi.templating import Jinja2Templates

from .config import ModelEntry, RouterConfig
from .usage_cli import (
    _aggregate_by_day_model,
    _aggregate_by_model,
    _group_by_provider,
    _record_cost_currency,
    _total_agg,
)

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

MONACO_VERSION = "0.52.0"

_MODALITY_EMOJI = {
    "text": "\u270f\ufe0f",
    "image": "\U0001f5bc\ufe0f",
    "audio": "\U0001f3b5",
    "video": "\U0001f3ac",
    "pdf": "\U0001f4c4",
}

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters["fmt_tokens"] = lambda n: _fmt_stat_tokens(int(n or 0))
templates.env.filters["fmt_cost"] = lambda value, symbol: _fmt_stat_cost(
    None if value in ("", None) else value,
    symbol,
)


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


def _parse_stats_days(raw: Optional[str]) -> int:
    try:
        days = int(raw or "1")
    except ValueError:
        days = 1
    return max(1, min(days, 90))


def _parse_stats_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def stats_period_from_params(params: Any) -> dict:
    now = datetime.now().astimezone()
    tz = now.tzinfo
    days = _parse_stats_days(params.get("days"))
    day = _parse_stats_date(params.get("day"))
    date_from = _parse_stats_date(params.get("from"))
    date_to = _parse_stats_date(params.get("to"))

    if day is not None:
        since_dt = datetime(day.year, day.month, day.day, tzinfo=tz)
        until_dt = since_dt + timedelta(days=1)
        return {
            "mode": "day",
            "days": days,
            "label": day.isoformat(),
            "day": day.isoformat(),
            "date_from": "",
            "date_to": "",
            "since_epoch": since_dt.timestamp(),
            "until_epoch": until_dt.timestamp(),
        }

    if date_from is not None or date_to is not None:
        if date_from is None:
            date_from = date_to
        if date_to is None:
            date_to = date_from
        if date_from > date_to:
            date_from, date_to = date_to, date_from
        since_dt = datetime(date_from.year, date_from.month, date_from.day, tzinfo=tz)
        until_dt = datetime(date_to.year, date_to.month, date_to.day, tzinfo=tz) + timedelta(days=1)
        label = (
            date_from.isoformat()
            if date_from == date_to
            else f"{date_from.isoformat()} → {date_to.isoformat()}"
        )
        return {
            "mode": "range",
            "days": days,
            "label": label,
            "day": "",
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
            "since_epoch": since_dt.timestamp(),
            "until_epoch": until_dt.timestamp(),
        }

    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    since_dt = today_midnight - timedelta(days=days - 1)
    return {
        "mode": "days",
        "days": days,
        "label": "Today" if days == 1 else f"Last {days} days",
        "day": "",
        "date_from": "",
        "date_to": "",
        "since_epoch": since_dt.timestamp(),
        "until_epoch": now.timestamp(),
    }


def stats_query_params(
    *,
    period: dict,
    view: str,
    page: int,
    provider: str = "",
    model: str = "",
) -> Dict[str, str]:
    params = {
        "view": view,
        "page": str(page),
        "days": str(period["days"]),
    }
    if period.get("day") and not period.get("date_from") and not period.get("date_to"):
        params["from"] = period["day"]
        params["to"] = period["day"]
    if period.get("date_from"):
        params["from"] = period["date_from"]
    if period.get("date_to"):
        params["to"] = period["date_to"]
    if provider:
        params["provider"] = provider
    if model:
        params["model"] = model
    return params


def stats_recent_model_index(by_model_agg: Dict[str, dict]) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = {
        "": [
            {"value": model, "label": model}
            for model, _agg in sorted(by_model_agg.items(), key=lambda item: (-item[1]["requests"], item[0]))
        ]
    }
    grouped = _group_by_provider(by_model_agg)
    for provider, models in grouped.items():
        index[provider] = [
            {
                "value": model,
                "label": model.split("/", 1)[1] if "/" in model else model,
            }
            for model, _agg in sorted(models.items(), key=lambda item: (-item[1]["requests"], item[0]))
        ]
    return index


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


def _usage_row_from_agg(agg: dict) -> dict:
    return {
        "requests": agg["requests"],
        "input_tokens": agg["input_tokens"],
        "output_tokens": agg["output_tokens"],
        "cache_write_tokens": agg["cache_write_tokens"],
        "cache_read_tokens": agg["cache_read_tokens"],
        "cost_cny": round(agg["cost_cny"], 4) if agg.get("_has_cost_cny") else None,
        "cost_usd": round(agg["cost_usd"], 4) if agg.get("_has_cost_usd") else None,
    }


class StatsUrls:
    def __init__(self, base_query: Dict[str, str]) -> None:
        self._base_query = dict(base_query)
        self.recent_anchor = "#recent-requests"

    def _url(self, path: str = "/stats", updates: Optional[Dict[str, Any]] = None) -> str:
        params = dict(self._base_query)
        for key, value in (updates or {}).items():
            if value in (None, ""):
                params.pop(key, None)
            else:
                params[key] = str(value)
        query = urlencode(params)
        return f"{path}?{query}" if query else path

    @property
    def data_json(self) -> str:
        return self._url("/stats/data", {"page": None})

    def period(self, days: int) -> str:
        return self._url(updates={"days": days, "day": None, "from": None, "to": None, "page": 1})

    def view(self, target_view: str) -> str:
        return self._url(updates={"view": target_view, "page": 1})

    @property
    def reset_filters(self) -> str:
        return self._url(updates={"provider": None, "model": None, "page": 1}) + self.recent_anchor

    def page_href(self, page: int) -> str:
        return self._url(updates={"page": page}) + self.recent_anchor


def build_stats_context(
    *,
    period: dict,
    days: int,
    view: str,
    page: int,
    recent_provider: str,
    recent_model: str,
    by_model_agg: Dict[str, dict],
    by_day_agg: Dict[str, Dict[str, dict]],
    recent: List[dict],
    config: RouterConfig,
    total_recent: int,
    total_pages: int,
) -> dict:
    grouped_by_provider = _group_by_provider(by_model_agg)
    total = _total_agg(by_model_agg)
    summary = _sum_usage_rows(total)
    base_query = stats_query_params(
        period=period,
        view=view,
        page=page,
        provider=recent_provider,
        model=recent_model,
    )
    urls = StatsUrls(base_query)

    provider_options = sorted(
        grouped_by_provider,
        key=lambda prov: (-_total_agg(grouped_by_provider[prov])["requests"], prov),
    )
    if recent_provider and recent_provider not in provider_options:
        provider_options.append(recent_provider)

    recent_model_index = stats_recent_model_index(by_model_agg)
    recent_model_options = recent_model_index.get(recent_provider, recent_model_index.get("", []))

    filter_parts = [period["label"]]
    if recent_provider:
        filter_parts.append(f"provider={recent_provider}")
    if recent_model:
        filter_parts.append(f"model={recent_model}")

    by_model_sections = []
    for prov in sorted(grouped_by_provider, key=lambda p: -_total_agg(grouped_by_provider[p])["requests"]):
        sub = _total_agg(grouped_by_provider[prov])
        models = []
        for model, agg in sorted(grouped_by_provider[prov].items(), key=lambda x: -x[1]["requests"]):
            model_label = model.split("/", 1)[1] if "/" in model else model
            models.append({"label": model_label, "row": _usage_row_from_agg(agg)})
        by_model_sections.append({"label": prov, "total": _usage_row_from_agg(sub), "models": models})

    by_day_rows = []
    for day_key in sorted(by_day_agg.keys(), reverse=True):
        dt = _total_agg(by_day_agg[day_key])
        row = _usage_row_from_agg(dt)
        row["label"] = day_key
        by_day_rows.append(row)

    daily_detail_rows = []
    for day_key in sorted(by_day_agg.keys(), reverse=True):
        day_models = by_day_agg[day_key]
        dt = _total_agg(day_models)
        providers = []
        day_grouped = _group_by_provider(day_models)
        for prov in sorted(day_grouped, key=lambda p: (-_total_agg(day_grouped[p])["requests"], p)):
            prov_total = _total_agg(day_grouped[prov])
            models = []
            for model, agg in sorted(day_grouped[prov].items(), key=lambda item: (-item[1]["requests"], item[0])):
                model_label = model.split("/", 1)[1] if "/" in model else model
                models.append({"label": model_label, "row": _usage_row_from_agg(agg)})
            providers.append({
                "label": prov,
                "total": _usage_row_from_agg(prov_total),
                "models": models,
            })
        daily_detail_rows.append({
            "day_label": day_key,
            "day_total": _usage_row_from_agg(dt),
            "providers": providers,
        })

    recent_rows = []
    for row in recent:
        result = _record_cost_currency(row, config)
        if result:
            cost, currency = result
            cost_str = f"¥{cost:.4f}" if currency.upper() != "USD" else f"${cost:.4f}"
        else:
            cost_str = "-"
        provider_label = row.get("provider", "") or (
            row.get("model", "").split("/", 1)[0] if "/" in row.get("model", "") else ""
        )
        model_label = row.get("model", "")
        if provider_label and model_label.startswith(f"{provider_label}/"):
            model_label = model_label.split("/", 1)[1]
        recent_rows.append({
            "ts": row.get("ts", ""),
            "provider": provider_label,
            "model": model_label,
            "input_tokens": row.get("input_tokens", 0),
            "output_tokens": row.get("output_tokens", 0),
            "cache_write_tokens": row.get("cache_write_tokens", 0),
            "cache_read_tokens": row.get("cache_read_tokens", 0),
            "cost": cost_str,
            "streaming": "✓" if row.get("streaming") else "—",
            "status": row.get("status", 0),
            "duration_ms": row.get("duration_ms", 0),
        })

    pagination = None
    if total_pages > 1:
        pagination = {
            "page": page,
            "total_pages": total_pages,
            "total_recent": total_recent,
            "first_href": urls.page_href(1) if page > 1 else None,
            "prev_href": urls.page_href(page - 1) if page > 1 else None,
            "next_href": urls.page_href(page + 1) if page < total_pages else None,
            "last_href": urls.page_href(total_pages) if page < total_pages else None,
        }

    return {
        "period": period,
        "days": days,
        "view": view,
        "summary": summary,
        "base_query": base_query,
        "urls": urls,
        "recent_provider": recent_provider,
        "recent_model": recent_model,
        "provider_options": provider_options,
        "recent_model_options": recent_model_options,
        "recent_filter_summary": " · ".join(filter_parts),
        "by_model_sections": by_model_sections,
        "by_day_rows": by_day_rows,
        "daily_detail_rows": daily_detail_rows,
        "recent_rows": recent_rows,
        "pagination": pagination,
        "recent_model_index_json": json.dumps(recent_model_index, separators=(",", ":")).replace("<", "\\u003c"),
    }


def build_config_context(cfg: RouterConfig, config_path: Optional[Path]) -> dict:
    models = []
    for pname, prov in cfg.providers.items():
        for fmt, ep in prov.endpoints.items():
            for m in ep.models:
                entry = m if isinstance(m, ModelEntry) else ModelEntry(name=str(m))
                mm = list(entry.multimodality)
                extra = [mt for mt in ("image", "audio", "video", "pdf") if mt in mm]
                models.append({
                    "full_id": f"{pname}/{entry.name}",
                    "provider": pname,
                    "name": entry.name,
                    "format": fmt,
                    "extra_modalities": extra,
                })

    return {
        "config_path": str(config_path) if config_path else "(unknown)",
        "has_config_path": config_path is not None,
        "models": models,
        "monaco_version": MONACO_VERSION,
        "modality_emoji": _MODALITY_EMOJI,
    }