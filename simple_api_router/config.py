"""Configuration models and loader."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, model_validator

VALID_FORMATS = frozenset({"anthropic", "openai_chat", "openai_responses", "google"})

_FORMAT_DEFAULT_URLS: Dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai_chat": "https://api.openai.com",
    "openai_responses": "https://api.openai.com",
    "google": "https://generativelanguage.googleapis.com",
}


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    log_file: Optional[str] = "router.log"
    max_retries: int = 3  # max retry attempts per request on upstream errors
    # Global fallback model for text-only models that receive image/video content.
    # Value is "provider/model" (same format as the client `model` field).
    multimodal_fallback: Optional[str] = None


class ModelEntry(BaseModel):
    """Per-model attributes.  Used when a model needs extra flags beyond its name."""
    name: str
    # Set True to mark this model as text-only.  When a request with image/video
    # content is routed here, the router will redirect to multimodal_fallback (or the
    # global server.multimodal_fallback) instead of forwarding and letting the upstream
    # return an error.
    text_only: bool = False
    # Override the global server.multimodal_fallback for just this model.
    multimodal_fallback: Optional[str] = None
    # Inline pricing for this model.  Takes precedence over the top-level
    # RouterConfig.pricing section when both are present.
    pricing: Optional["PricingEntry"] = None


class EndpointConfig(BaseModel):
    base_url: Optional[str] = None
    # Accept either plain strings or ModelEntry dicts; both forms are normalised internally.
    models: List[Union[str, ModelEntry]] = Field(default_factory=list)
    model_map: Dict[str, str] = Field(default_factory=dict)
    # Enable DeepSeek reasoning_content passthrough. None = auto-detect from model name.
    deepseek_reasoning: Optional[bool] = None

    def model_names(self) -> List[str]:
        """Return the list of model names (works for both str and ModelEntry items)."""
        return [m if isinstance(m, str) else m.name for m in self.models]

    def get_model_entry(self, name: str) -> ModelEntry:
        """Return the ModelEntry for *name*, creating a default one if the model is a plain string."""
        for m in self.models:
            if isinstance(m, ModelEntry) and m.name == name:
                return m
        return ModelEntry(name=name)

    def resolve_base_url(self, api_format: str, provider_base_url: Optional[str] = None) -> str:
        raw = self.base_url or provider_base_url or _FORMAT_DEFAULT_URLS.get(api_format, "https://api.openai.com")
        url = raw.rstrip("/")
        # Strip accidental trailing /v1 — proxy.py always appends the full versioned path.
        if url.endswith("/v1"):
            url = url[:-3]
        return url

    def resolve_model(self, model: str) -> str:
        return self.model_map.get(model, model)


class ProviderConfig(BaseModel):
    """Configuration for a single upstream provider."""

    api_key: str = ""
    base_url: Optional[str] = None  # inherited by endpoints that omit their own base_url
    endpoints: Dict[str, EndpointConfig]

    @model_validator(mode="after")
    def _validate(self) -> "ProviderConfig":
        for fmt in self.endpoints:
            if fmt not in VALID_FORMATS:
                raise ValueError(f"Invalid endpoint format '{fmt}'. Valid: {sorted(VALID_FORMATS)}")
        # No duplicate model names across endpoints
        seen: Dict[str, str] = {}
        for fmt, ep in self.endpoints.items():
            for m in ep.model_names():
                if m in seen:
                    raise ValueError(
                        f"Model '{m}' is listed in both '{seen[m]}' and '{fmt}' endpoints"
                    )
                seen[m] = fmt
        return self

    def find_model(self, model: str) -> Optional[Tuple[str, "EndpointConfig"]]:
        """Return (api_format, endpoint) for model, or None.
        Exact match first; wildcard (empty models list) as fallback."""
        wildcard = None
        for fmt, ep in self.endpoints.items():
            if model in ep.model_names():
                return fmt, ep
            if not ep.models and wildcard is None:
                wildcard = (fmt, ep)
        return wildcard


class PricingTier(BaseModel):
    """One pricing bracket for a model.

    *threshold* is the minimum number of **input tokens** in a single request
    for this tier's rates to apply.  The tier with the highest threshold that
    is still ≤ the request's input_tokens is used for the whole request (i.e.
    all tokens are billed at that tier — it is NOT a progressive tax).

    Example (Gemini 2.5 Pro):
        tiers:
          - threshold: 0       # < 200 K tokens → low rate
            input: 1.25
            output: 10.0
          - threshold: 200000  # ≥ 200 K tokens → high rate
            input: 2.50
            output: 15.0
    """
    threshold: int = 0
    input: float = 0.0
    output: float = 0.0
    # None means "not configured — fall back to the input rate at cost-calc time".
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None


class PricingEntry(BaseModel):
    """Pricing for one model, in USD per million tokens.

    Use ``tiers`` for tiered (non-linear) pricing; leave it empty for flat pricing.
    When ``tiers`` is present the top-level ``input``/``output``/``cache_*``
    fields are ignored.

    If ``cache_read`` or ``cache_write`` are omitted (None), those tokens are
    billed at the ``input`` rate of the applicable tier.  Set them explicitly to
    ``0.0`` to make them free.
    """
    input: float = 0.0
    output: float = 0.0
    # None means "not configured — fall back to the input rate at cost-calc time".
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None
    tiers: List[PricingTier] = Field(default_factory=list)


class RouterConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    # Keys are "provider/model" strings (same format as the client `model` field).
    # Used as a fallback when a model's inline ModelEntry.pricing is not set.
    pricing: Dict[str, PricingEntry] = Field(default_factory=dict)

    def get_pricing(self, model_str: str) -> Optional[PricingEntry]:
        """Return PricingEntry for *model_str* ("provider/model"), or None.

        Lookup order:
          1. Inline ``ModelEntry.pricing`` on the matching model entry.
          2. Top-level ``RouterConfig.pricing[model_str]``.
        """
        if "/" in model_str:
            prov_name, model_name = model_str.split("/", 1)
        else:
            prov_name, model_name = None, model_str

        if prov_name and prov_name in self.providers:
            result = self.providers[prov_name].find_model(model_name)
            if result is not None:
                _, ep = result
                entry = ep.get_model_entry(model_name)
                if entry.pricing is not None:
                    return entry.pricing

        return self.pricing.get(model_str)


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} in string values."""
    if isinstance(obj, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


def _find_unexpanded(obj: Any, path: str = "") -> List[str]:
    """Return a list of config paths where ${VAR} was not expanded (var not set)."""
    problems: List[str] = []
    if isinstance(obj, str):
        for m in _ENV_PATTERN.finditer(obj):
            if m.group(1) not in os.environ:
                problems.append(f"{path}: ${{{m.group(1)}}} (env var not set)")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            problems.extend(_find_unexpanded(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            problems.extend(_find_unexpanded(v, f"{path}[{i}]"))
    return problems


def load_config(path: Union[str, Path] = "config.yaml") -> RouterConfig:
    """Load and validate configuration from YAML file.

    Raises ValueError if any ${VAR} placeholders remain unexpanded
    (i.e. the referenced environment variable is not set).
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    raw = _expand_env_vars(raw)
    unexpanded = _find_unexpanded(raw)
    if unexpanded:
        raise ValueError(
            "Config contains unexpanded environment variables — "
            "set them before starting:\n  " + "\n  ".join(unexpanded)
        )
    return RouterConfig.model_validate(raw)
