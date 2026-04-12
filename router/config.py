"""Configuration models and loader."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator


class RetryConfig(BaseModel):
    max_retries: int = 3
    cooldown_after: int = 5
    cooldown_duration: int = 300
    error_limits: Dict[int, int] = Field(default_factory=dict)


class UsageConfig(BaseModel):
    rpm: Optional[int] = None
    daily: Optional[int] = None
    per_5h: Optional[int] = None
    weekly: Optional[int] = None
    no_retry_duration: int = 3600  # seconds to block after usage exceeded + 429


class APIConfig(BaseModel):
    base_url: str
    api_key: str
    type: str  # "openai" or "anthropic"
    model: Optional[str] = None  # override model name in requests
    endpoint_path: Optional[str] = None  # override the default endpoint path
    retry: RetryConfig = Field(default_factory=RetryConfig)
    usage: UsageConfig = Field(default_factory=UsageConfig)


class GroupMember(BaseModel):
    api: Optional[str] = None
    group: Optional[str] = None

    @model_validator(mode="after")
    def validate_one_of(self) -> "GroupMember":
        if self.api is None and self.group is None:
            raise ValueError("Group member must specify 'api' or 'group'")
        if self.api is not None and self.group is not None:
            raise ValueError("Group member must specify only one of 'api' or 'group'")
        return self


class GroupConfig(BaseModel):
    strategy: str = "sequential"  # "sequential" or "load_balance"
    members: List[GroupMember] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_strategy(self) -> "GroupConfig":
        if self.strategy not in ("sequential", "load_balance"):
            raise ValueError(f"Invalid strategy: {self.strategy}")
        return self


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    log_file: Optional[str] = "router.log"


class RouterConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    default_group: str = "main"
    apis: Dict[str, APIConfig] = Field(default_factory=dict)
    groups: Dict[str, GroupConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> "RouterConfig":
        # Validate default_group exists
        if self.default_group not in self.groups:
            raise ValueError(f"default_group '{self.default_group}' not found in groups")

        # Validate all group member references
        for gname, group in self.groups.items():
            for member in group.members:
                if member.api and member.api not in self.apis:
                    raise ValueError(
                        f"Group '{gname}' references unknown api '{member.api}'"
                    )
                if member.group and member.group not in self.groups:
                    raise ValueError(
                        f"Group '{gname}' references unknown group '{member.group}'"
                    )
        return self


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
