"""Shared helpers used by all converter modules.

The format-specific conversion logic lives in:
- ``converter_openai.py``   — Anthropic ↔ OpenAI Chat Completions
- ``converter_responses.py`` — Anthropic ↔ OpenAI Responses API
- ``converter_google.py``    — Anthropic ↔ Google Gemini
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Shared helpers for all converter modules
# ---------------------------------------------------------------------------

_O_SERIES_RE = re.compile(r"\bo[1-9](-|\b)|o4-mini|codex", re.IGNORECASE)


def sanitize_system_text(text: str) -> str:
    """Strip lines containing x-anthropic-billing-header injected by Claude Code.

    Preserves all other content and surrounding whitespace faithfully.
    """
    lines = text.split("\n")
    filtered = [l for l in lines if "x-anthropic-billing-header:" not in l.lower()]
    return "\n".join(filtered)


def clean_schema(schema: Any) -> Any:
    """Recursively remove JSON Schema keys that some OpenAI providers reject (e.g. format: uri)."""
    if not isinstance(schema, dict):
        return schema
    result: Dict[str, Any] = {}
    for k, v in schema.items():
        # Drop "format" when it's a URI-type validator (not "date-time" etc which are fine)
        if k == "format" and isinstance(v, str) and v in ("uri", "uri-reference", "iri", "iri-reference"):
            continue
        if isinstance(v, dict):
            result[k] = clean_schema(v)
        elif isinstance(v, list):
            result[k] = [clean_schema(i) for i in v]
        else:
            result[k] = v
    return result


def is_o_series(model: str) -> bool:
    """Return True for OpenAI o1/o3/o4-mini family that uses max_completion_tokens."""
    return bool(_O_SERIES_RE.search(model))


def _reasoning_effort_from_budget(budget_tokens: int) -> str:
    """Map Anthropic budget_tokens to OpenAI reasoning_effort."""
    if budget_tokens <= 1024:
        return "low"
    if budget_tokens <= 8192:
        return "medium"
    if budget_tokens <= 32000:
        return "high"
    return "xhigh"


_EFFORT_ORDER = ("none", "low", "medium", "high", "xhigh")


def _normalize_effort(effort: str, max_effort: str = "xhigh") -> str:
    """Map Anthropic 'max' → 'xhigh', then cap to max_effort.

    Anthropic uses "max" as the highest level; both OpenAI and DeepSeek use "xhigh".
    Providers that only support low/medium/high can set max_effort="high".
    """
    if effort == "max":
        effort = "xhigh"
    try:
        if _EFFORT_ORDER.index(effort) > _EFFORT_ORDER.index(max_effort):
            return max_effort
    except ValueError:
        pass  # unknown value; pass through and let the provider reject it
    return effort


def strip_private_params(body: Dict[str, Any]) -> Dict[str, Any]:
    """Remove top-level keys starting with '_' (private Claude Code params)."""
    return {k: v for k, v in body.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# DeepSeek detection
# ---------------------------------------------------------------------------

_DEEPSEEK_RE = re.compile(r"deepseek", re.IGNORECASE)


def is_deepseek_model(model: str) -> bool:
    """Return True if the model name looks like a DeepSeek model."""
    return bool(_DEEPSEEK_RE.search(model))
