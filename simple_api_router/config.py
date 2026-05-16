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


class EndpointConfig(BaseModel):
    base_url: Optional[str] = None
    models: List[str] = Field(default_factory=list)
    model_map: Dict[str, str] = Field(default_factory=dict)
    # Enable DeepSeek reasoning_content passthrough. None = auto-detect from model name.
    deepseek_reasoning: Optional[bool] = None

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
            for m in ep.models:
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
            if model in ep.models:
                return fmt, ep
            if not ep.models and wildcard is None:
                wildcard = (fmt, ep)
        return wildcard


class RouterConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)


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
