"""Configuration models and loader."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, model_validator

_log = logging.getLogger(__name__)

VALID_FORMATS = frozenset({"anthropic", "openai_chat", "openai_responses", "google"})

_FORMAT_DEFAULT_URLS: Dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai_chat": "https://api.openai.com",
    "openai_responses": "https://api.openai.com",
    "google": "https://generativelanguage.googleapis.com",
}


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "INFO"
    log_file: Optional[str] = "router.log"
    max_retries: int = 3  # max retry attempts per request on upstream errors
    # Global per-type fallback models.  When a model doesn't natively support a
    # media type, the router auto-describes the blocks using this model.
    # Per-model overrides (ModelEntry.*_fallback) take precedence.
    # Value is "provider/model" (same format as the client `model` field).
    image_fallback: Optional[str] = None
    audio_fallback: Optional[str] = None
    video_fallback: Optional[str] = None
    pdf_fallback: Optional[str] = None
    # Max concurrent media description calls during fallback.
    multimodal_fallback_max_concurrency: int = 3
    # MCP tools.  When any of these is set, the router mounts the corresponding
    # understand_* tool at /mcp.  Each uses its own model.
    image_model: Optional[str] = None   # enables image_understanding tool
    audio_model: Optional[str] = None   # enables audio_understanding tool
    video_model: Optional[str] = None   # enables video_understanding tool
    pdf_model: Optional[str] = None     # enables pdf_understanding tool
    # Path to debug log file.  When set, all 4 request/response stages are
    # appended to this file for every request.  None = disabled.
    debug_log: Optional[str] = None
    # Global model aliases.  Clients may use the alias as the model name instead
    # of the full "provider/model" string.  The alias is resolved before routing;
    # billing uses the resolved "provider/model", not the alias.
    # Example: model_map: {claude: "anthropic/claude-opus-4-5", fast: "openai/gpt-4o-mini"}
    model_map: Dict[str, str] = Field(default_factory=dict)


class ModelEntry(BaseModel):
    """Per-model attributes.  Used when a model needs extra flags beyond its name."""
    name: str
    # Media types this model natively supports.  Any type NOT listed here will be
    # auto-described via the corresponding *_fallback before forwarding.
    # Valid values: "image", "audio", "video", "pdf".
    # Example: multimodality: [image, pdf]  — supports images and PDFs but not audio/video.
    # Empty list (default) = model does not support any media; all will be described.
    multimodality: List[str] = Field(default_factory=list)
    # Per-model fallback overrides for each media type.
    # Overrides the global server.*_fallback when set.
    image_fallback: Optional[str] = None
    audio_fallback: Optional[str] = None
    video_fallback: Optional[str] = None
    pdf_fallback: Optional[str] = None
    # Inline pricing for this model.  Takes precedence over the top-level
    # RouterConfig.pricing section when both are present.
    pricing: Optional["PricingEntry"] = None
    # Enable DeepSeek reasoning_content passthrough for this specific model.
    # Overrides the endpoint-level deepseek_reasoning when set.
    # None = fall through to endpoint setting, then auto-detect from model name.
    deepseek_reasoning: Optional[bool] = None
    # Cap for reasoning_effort sent to this model.  None = fall through to endpoint setting.
    # Valid values: "none", "low", "medium", "high", "xhigh".
    # Example: kimi models only support up to "high"; deepseek supports "xhigh".
    max_reasoning_effort: Optional[str] = None


class EndpointConfig(BaseModel):
    base_url: Optional[str] = None
    # Accept either plain strings or ModelEntry dicts; both forms are normalised internally.
    models: List[Union[str, ModelEntry]] = Field(default_factory=list)
    model_map: Dict[str, str] = Field(default_factory=dict)
    # Enable DeepSeek reasoning_content passthrough. None = auto-detect from model name.
    deepseek_reasoning: Optional[bool] = None
    # Default cap for reasoning_effort on this endpoint.  None = no cap ("xhigh" default).
    # Per-model max_reasoning_effort takes precedence when set.
    max_reasoning_effort: Optional[str] = None

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

    def resolve_max_reasoning_effort(self, model_name: str) -> str:
        """Return the effective max_reasoning_effort for a model.

        Lookup order: ModelEntry.max_reasoning_effort → EndpointConfig.max_reasoning_effort → "xhigh".
        """
        entry = self.get_model_entry(model_name)
        return entry.max_reasoning_effort or self.max_reasoning_effort or "xhigh"

    def resolve_model(self, model: str) -> str:
        return self.model_map.get(model, model)


class ProviderConfig(BaseModel):
    """Configuration for a single upstream provider."""

    api_key: str = ""
    base_url: Optional[str] = None  # inherited by endpoints that omit their own base_url
    endpoints: Dict[str, EndpointConfig]

    @model_validator(mode="after")
    def _validate(self) -> "ProviderConfig":
        # Normalize endpoint format keys: allow hyphens as aliases for underscores
        # e.g. "openai-chat" → "openai_chat"  to tolerate common config typos.
        normalized: Dict[str, "EndpointConfig"] = {}
        for fmt, ep in self.endpoints.items():
            norm = fmt.replace("-", "_")
            if norm in normalized:
                _log.warning(
                    "Duplicate endpoint format key '%s': both '%s' and the "
                    "earlier key map to the same format; last value wins",
                    norm, fmt,
                )
            normalized[norm] = ep
        self.endpoints = normalized

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
    """Pricing for one model, per million tokens.

    ``currency`` declares whether prices are in CNY or USD (default: ``"CNY"``).
    Use ``tiers`` for tiered (non-linear) pricing; leave it empty for flat pricing.
    When ``tiers`` is present the top-level ``input``/``output``/``cache_*``
    fields are ignored.

    If ``cache_read`` or ``cache_write`` are omitted (None), those tokens are
    billed at the ``input`` rate of the applicable tier.  Set them explicitly to
    ``0.0`` to make them free.
    """
    currency: str = "CNY"   # "CNY" or "USD"
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


def load_config(path: Union[str, Path] = "config.yaml", *, skip_env_check: bool = False) -> RouterConfig:
    """Load and validate configuration from YAML file.

    Raises ValueError if any ${VAR} placeholders remain unexpanded
    (i.e. the referenced environment variable is not set), unless
    *skip_env_check* is True (useful for read-only commands like `usage`
    that don't need API keys).
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    raw = _expand_env_vars(raw)
    if not skip_env_check:
        unexpanded = _find_unexpanded(raw)
        if unexpanded:
            raise ValueError(
                "Config contains unexpanded environment variables — "
                "set them before starting:\n  " + "\n  ".join(unexpanded)
            )
    return RouterConfig.model_validate(raw)
