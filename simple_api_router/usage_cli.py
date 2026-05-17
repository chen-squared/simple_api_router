"""Usage statistics command — reads router.usage.jsonl and prints reports.

Tiered pricing note
-------------------
Cost is computed **per individual request** (not on the aggregated total) so
that each request is billed at the correct tier.  The tier is selected by
comparing the request's *input_tokens* against each tier's *threshold*: the
tier with the highest threshold that is still ≤ input_tokens applies, and
ALL tokens in that request are billed at that tier's rates.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config / path helpers
# ---------------------------------------------------------------------------

def _find_config(explicit: Optional[str]) -> Optional[str]:
    """Return a resolved config path, or None."""
    from simple_api_router.service import resolve_config
    try:
        return str(resolve_config(explicit))
    except SystemExit:
        return None


def _resolve_usage_path(config_file: str) -> Path:
    """Derive the usage JSONL path from the config's log_file setting."""
    from simple_api_router.config import load_config

    cfg_path = Path(config_file).expanduser().resolve()
    config = load_config(cfg_path, skip_env_check=True)
    log_file = config.server.log_file or "router.log"
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = cfg_path.parent / log_file
    return log_path.parent / "router.usage.jsonl"


# ---------------------------------------------------------------------------
# Loading records
# ---------------------------------------------------------------------------

def _load_records(usage_path: Path, since: date, until: date) -> List[dict]:
    """Load records from the current JSONL file and its rotated siblings."""
    records: List[dict] = []
    candidates: List[Path] = []

    if usage_path.exists():
        candidates.append(usage_path)

    if usage_path.parent.exists():
        for p in sorted(usage_path.parent.glob(f"{usage_path.name}.*")):
            # Rotated files are named <base>.<YYYY-MM-DD>; skip clearly out-of-range ones.
            suffix = p.name[len(usage_path.name) + 1:]
            try:
                file_date = date.fromisoformat(suffix)
                if file_date < since or file_date >= until:
                    continue
            except ValueError:
                pass  # unknown suffix — include anyway
            candidates.append(p)

    for path in candidates:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ts_str = rec.get("ts", "")
                    rec_date = datetime.fromisoformat(ts_str.rstrip("Z")).date()
                    if since <= rec_date < until:
                        records.append(rec)
                except (json.JSONDecodeError, ValueError, KeyError):
                    pass
        except OSError:
            pass

    return records


# ---------------------------------------------------------------------------
# Cost calculation (per-record, supports tiered pricing)
# ---------------------------------------------------------------------------

def _record_cost(rec: dict, config) -> Optional[float]:
    """Return the estimated cost in USD for a single request record, or None.

    Uses ``config.get_pricing(model)`` which checks inline ModelEntry.pricing
    first, then falls back to the top-level RouterConfig.pricing section.

    If ``cache_read`` or ``cache_write`` rates are not configured (None), the
    corresponding tokens are billed at the ``input`` rate of the applicable
    tier.  Explicitly set them to ``0.0`` to treat them as free.
    """
    model = rec.get("model", "")
    entry = config.get_pricing(model)
    if entry is None:
        return None

    in_tok = rec.get("input_tokens", 0)
    out_tok = rec.get("output_tokens", 0)
    cr_tok = rec.get("cache_read_tokens", 0)
    cw_tok = rec.get("cache_write_tokens", 0)

    if entry.tiers:
        # Pick the tier with the highest threshold that is still ≤ input tokens.
        rate = sorted(entry.tiers, key=lambda t: t.threshold)[0]
        for tier in entry.tiers:
            if in_tok >= tier.threshold:
                rate = tier
        cr_rate = rate.cache_read if rate.cache_read is not None else rate.input
        cw_rate = rate.cache_write if rate.cache_write is not None else rate.input
        return (
            in_tok / 1_000_000 * rate.input
            + out_tok / 1_000_000 * rate.output
            + cr_tok / 1_000_000 * cr_rate
            + cw_tok / 1_000_000 * cw_rate
        )
    else:
        cr_rate = entry.cache_read if entry.cache_read is not None else entry.input
        cw_rate = entry.cache_write if entry.cache_write is not None else entry.input
        return (
            in_tok / 1_000_000 * entry.input
            + out_tok / 1_000_000 * entry.output
            + cr_tok / 1_000_000 * cr_rate
            + cw_tok / 1_000_000 * cw_rate
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _empty_agg() -> Dict[str, Any]:
    return {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": 0.0,
        "_has_cost": False,
    }


def _add(agg: dict, rec: dict, config) -> None:
    agg["requests"] += 1
    agg["input_tokens"] += rec.get("input_tokens", 0)
    agg["output_tokens"] += rec.get("output_tokens", 0)
    agg["cache_read_tokens"] += rec.get("cache_read_tokens", 0)
    agg["cache_write_tokens"] += rec.get("cache_write_tokens", 0)
    cost = _record_cost(rec, config)
    if cost is not None:
        agg["cost_usd"] += cost
        agg["_has_cost"] = True


def _aggregate_by_model(records: List[dict], config) -> Dict[str, dict]:
    agg: Dict[str, dict] = defaultdict(_empty_agg)
    for rec in records:
        _add(agg[rec.get("model", "unknown")], rec, config)
    return dict(agg)


def _aggregate_by_day_model(
    records: List[dict], config
) -> Dict[str, Dict[str, dict]]:
    agg: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(_empty_agg))
    for rec in records:
        try:
            day = datetime.fromisoformat(rec.get("ts", "").rstrip("Z")).date().isoformat()
        except ValueError:
            day = "unknown"
        _add(agg[day][rec.get("model", "unknown")], rec, config)
    return {d: dict(models) for d, models in agg.items()}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_cost(agg: dict) -> str:
    if not agg.get("_has_cost"):
        return "-"
    return f"¥{agg['cost_usd']:.4f}"


_COLS: Tuple[str, ...] = ("Model", "Req", "Input", "Output", "Cache↑", "Cache↓", "Cost")
_COL_W: Tuple[int, ...] = (38, 6, 9, 9, 9, 9, 10)
_TABLE_W = sum(_COL_W) + 2 * (len(_COL_W) - 1)

_BOLD  = "\033[1m"
_GREY  = "\033[90m"
_RESET = "\033[0m"


def _hdr() -> str:
    return "  ".join(h.ljust(w) for h, w in zip(_COLS, _COL_W))


def _divider() -> str:
    return "─" * _TABLE_W


def _format_row(label: str, agg: dict, indent: int = 0) -> str:
    """Format one data row. label is the display name (provider prefix stripped)."""
    cells = (
        " " * indent + label,
        str(agg["requests"]),
        _fmt_tok(agg["input_tokens"]),
        _fmt_tok(agg["output_tokens"]),
        _fmt_tok(agg["cache_write_tokens"]),
        _fmt_tok(agg["cache_read_tokens"]),
        _fmt_cost(agg),
    )
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, _COL_W))


def _total_agg(rows: Dict[str, dict]) -> dict:
    total = _empty_agg()
    for a in rows.values():
        total["requests"] += a["requests"]
        total["input_tokens"] += a["input_tokens"]
        total["output_tokens"] += a["output_tokens"]
        total["cache_read_tokens"] += a["cache_read_tokens"]
        total["cache_write_tokens"] += a["cache_write_tokens"]
        total["cost_usd"] += a["cost_usd"]
        total["_has_cost"] = total["_has_cost"] or a["_has_cost"]
    return total


def _group_by_provider(by_model: Dict[str, dict]) -> Dict[str, Dict[str, dict]]:
    """Split a model→agg map into provider→{model→agg} groups."""
    grouped: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for model, agg in by_model.items():
        provider = model.split("/", 1)[0] if "/" in model else "(unknown)"
        grouped[provider][model] = agg
    return dict(grouped)


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

def _print_provider_block(provider: str, models: Dict[str, dict],
                          show_subtotal: bool = True) -> None:
    """Print provider header, model rows, and optional subtotal."""
    print(f"\n{_BOLD}{provider}{_RESET}")
    for model, agg in sorted(models.items(), key=lambda kv: -kv[1]["requests"]):
        name = model.split("/", 1)[1] if "/" in model else model
        print("  " + _format_row(name, agg, indent=0))
    if show_subtotal and len(models) > 1:
        print(_GREY + "  " + _format_row("subtotal", _total_agg(models)) + _RESET)


def _print_summary(by_model: Dict[str, dict], period_str: str) -> None:
    print(f"\nUsage summary — {period_str}")
    print(_hdr())
    print(_divider())

    grouped = _group_by_provider(by_model)
    # Sort providers by total requests descending.
    for provider in sorted(grouped, key=lambda p: -sum(a["requests"] for a in grouped[p].values())):
        _print_provider_block(provider, grouped[provider], show_subtotal=True)

    print()
    print(_divider())
    print(_format_row("TOTAL", _total_agg(by_model)))
    print()


def _print_daily(by_day: Dict[str, Dict[str, dict]], period_str: str) -> None:
    print(f"\nDaily breakdown — {period_str}\n")
    for day in sorted(by_day.keys(), reverse=True):
        print(f"{_BOLD}{day}{_RESET}")
        print("  " + _hdr())
        print("  " + _divider())
        grouped = _group_by_provider(by_day[day])
        for provider in sorted(grouped, key=lambda p: -sum(a["requests"] for a in grouped[p].values())):
            models = grouped[provider]
            print(f"  {_BOLD}{provider}{_RESET}")
            for model, agg in sorted(models.items(), key=lambda kv: -kv[1]["requests"]):
                name = model.split("/", 1)[1] if "/" in model else model
                print("    " + _format_row(name, agg))
            if len(models) > 1:
                print(_GREY + "    " + _format_row("subtotal", _total_agg(models)) + _RESET)
        day_total = _total_agg(by_day[day])
        print("  " + _divider())
        print("  " + _format_row("day total", day_total))
        print()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _agg_to_dict(model: str, agg: dict) -> dict:
    return {
        "model": model,
        "provider": model.split("/")[0] if "/" in model else None,
        "requests": agg["requests"],
        "input_tokens": agg["input_tokens"],
        "output_tokens": agg["output_tokens"],
        "cache_read_tokens": agg["cache_read_tokens"],
        "cache_write_tokens": agg["cache_write_tokens"],
        "cost_usd": round(agg["cost_usd"], 6) if agg["_has_cost"] else None,
    }


def _json_summary(by_model: Dict[str, dict], period: dict) -> dict:
    rows = [_agg_to_dict(m, a) for m, a in sorted(by_model.items())]
    total = _total_agg(by_model)
    return {
        "period": period,
        "by_model": rows,
        "total": {
            "requests": total["requests"],
            "input_tokens": total["input_tokens"],
            "output_tokens": total["output_tokens"],
            "cache_read_tokens": total["cache_read_tokens"],
            "cache_write_tokens": total["cache_write_tokens"],
            "cost_usd": round(total["cost_usd"], 6) if total["_has_cost"] else None,
        },
    }


def _json_daily(by_day: Dict[str, Dict[str, dict]], period: dict) -> dict:
    days = []
    for day in sorted(by_day.keys()):
        days.append({
            "date": day,
            "models": [_agg_to_dict(m, a) for m, a in sorted(by_day[day].items())],
        })
    return {"period": period, "by_day": days}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def usage_command(args) -> None:
    """Execute the ``usage`` subcommand."""

    # ── load config ────────────────────────────────────────────────────────
    config_file = _find_config(getattr(args, "config", None))
    if config_file is None:
        print("Error: could not find config.yaml. Use --config PATH.", file=sys.stderr)
        sys.exit(1)
    try:
        from simple_api_router.config import load_config
        config = load_config(config_file, skip_env_check=True)
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── resolve usage log path ─────────────────────────────────────────────
    try:
        usage_path = _resolve_usage_path(config_file)
    except Exception as exc:
        print(f"Error resolving usage log: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── date range ─────────────────────────────────────────────────────────
    period = getattr(args, "period", None)
    last_n: int = getattr(args, "last", 7)
    if period == "day":
        last_n = 1
    elif period == "week":
        last_n = 7
    elif period == "month":
        last_n = 30

    today = date.today()
    since = today - timedelta(days=last_n - 1)
    until = today + timedelta(days=1)
    period_str = f"{since.isoformat()} → {today.isoformat()}"
    period_dict = {"from": since.isoformat(), "to": today.isoformat(), "days": last_n}

    # ── load & filter records ──────────────────────────────────────────────
    records = _load_records(usage_path, since, until)

    model_filter: Optional[str] = getattr(args, "model", None)
    provider_filter: Optional[str] = getattr(args, "provider", None)
    if model_filter:
        records = [r for r in records if model_filter.lower() in r.get("model", "").lower()]
    if provider_filter:
        records = [r for r in records if r.get("provider", "") == provider_filter]

    if not records:
        print(f"No usage records found for {period_str}.")
        if not usage_path.exists():
            print(f"  (Usage log not found: {usage_path})")
        return

    # ── output ─────────────────────────────────────────────────────────────
    fmt: str = getattr(args, "format", "table")
    daily: bool = getattr(args, "daily", False)

    if daily:
        by_day = _aggregate_by_day_model(records, config)
        if fmt == "json":
            print(json.dumps(_json_daily(by_day, period_dict), indent=2))
        else:
            _print_daily(by_day, period_str)
    else:
        by_model = _aggregate_by_model(records, config)
        if fmt == "json":
            print(json.dumps(_json_summary(by_model, period_dict), indent=2))
        else:
            _print_summary(by_model, period_str)
