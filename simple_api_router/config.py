"""Configuration models and loader."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    log_file: Optional[str] = "router.log"
    max_retries: int = 3  # max retry attempts per request on upstream errors


class ProviderConfig(BaseModel):
    """Configuration for a single upstream provider (Anthropic or OpenAI-compatible)."""

    type: str  # "anthropic" or "openai"
    api_key: str = ""   # optional for local endpoints that don't require auth
    base_url: Optional[str] = None
    # "openai_chat" (default) or "openai_responses" (OpenAI Responses API)
    api_format: str = "openai_chat"
    # List of model names this provider exposes.  Empty list = accept any model.
    models: List[str] = Field(default_factory=list)
    # Optional remap: external model name → backend model name.
    model_map: Dict[str, str] = Field(default_factory=dict)
    # Enable DeepSeek reasoning_content passthrough for assistant messages.
    # None = auto-detect from provider name / model name.
    deepseek_reasoning: Optional[bool] = None

    @model_validator(mode="after")
    def _validate_fields(self) -> "ProviderConfig":
        if self.type not in ("anthropic", "openai"):
            raise ValueError(
                f"provider type must be 'anthropic' or 'openai', got '{self.type}'"
            )
        if self.api_format not in ("openai_chat", "openai_responses"):
            raise ValueError(
                f"api_format must be 'openai_chat' or 'openai_responses', got '{self.api_format}'"
            )
        return self

    def resolve_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        return (
            "https://api.anthropic.com"
            if self.type == "anthropic"
            else "https://api.openai.com/v1"
        )

    def resolve_model(self, model: str) -> str:
        """Return the backend model name for a client-facing model name."""
        return self.model_map.get(model, model)


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


def load_config(path: Union[str, Path] = "config.yaml") -> RouterConfig:
    """Load and validate configuration from YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    raw = _expand_env_vars(raw)
    return RouterConfig.model_validate(raw)
