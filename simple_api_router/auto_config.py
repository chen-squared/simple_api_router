"""auto-config: auto-generate router config entries from models.dev data."""
from __future__ import annotations

import difflib
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml

# ── Format inference ───────────────────────────────────────────────────────
_ANTHROPIC_FORMAT_PROVIDERS = frozenset({"anthropic", "google-vertex-anthropic"})
_GOOGLE_FORMAT_PROVIDERS = frozenset({"google", "google-vertex"})

# model-level provider.npm → router format
# @ai-sdk/openai         uses the Responses API (not OpenAI Chat)
# @ai-sdk/openai-compatible uses the Chat Completions API
# No provider field at all  → default openai_chat (handled by fallback)
_NPM_TO_FORMAT: Dict[str, str] = {
    "@ai-sdk/anthropic":               "anthropic",
    "@ai-sdk/google":                  "google",
    "@ai-sdk/google-vertex/anthropic": "anthropic",
    "@ai-sdk/openai":                  "openai_responses",
    "@ai-sdk/openai-compatible":       "openai_chat",
}

# Well-known base URLs for providers whose models.dev entry has no api URL.
_KNOWN_URLS: Dict[str, str] = {
    "cerebras":     "https://api.cerebras.ai",
    "cohere":       "https://api.cohere.com",
    "deepinfra":    "https://api.deepinfra.com/v1/openai",
    "groq":         "https://api.groq.com/openai",
    "mistral":      "https://api.mistral.ai",
    "perplexity":   "https://api.perplexity.ai",
    "togetherai":   "https://api.together.xyz",
    "xai":          "https://api.x.ai",
    "venice":       "https://api.venice.ai/api",
    # Native providers — no base_url needed in config.
    "anthropic":    "",
    "openai":       "",
    "google":       "",
    "google-vertex": "",
    "google-vertex-anthropic": "",
}

MODELS_DEV_URL = "https://models.dev/api.json"


# ── Data fetching ──────────────────────────────────────────────────────────

def fetch_models_dev() -> Dict[str, Any]:
    """Fetch and return the models.dev provider/model registry."""
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(MODELS_DEV_URL)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        print(f"Error fetching models.dev: HTTP {exc.response.status_code}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error fetching models.dev: {exc}", file=sys.stderr)
        sys.exit(1)


# ── Provider / model lookup ────────────────────────────────────────────────

def find_provider(data: Dict[str, Any], provider_id: str) -> Tuple[str, Dict[str, Any]]:
    """Return (canonical_id, provider_data), or exit with suggestions."""
    if provider_id in data:
        return provider_id, data[provider_id]
    # Case-insensitive fuzzy match.
    lower = provider_id.lower()
    matches = [k for k in data if k.lower() == lower]
    if not matches:
        matches = [k for k in data if lower in k.lower()]
    if len(matches) == 1:
        return matches[0], data[matches[0]]
    if matches:
        suggestions = ", ".join(matches[:10])
        print(
            f"Provider '{provider_id}' not found in models.dev. Did you mean: {suggestions}?",
            file=sys.stderr,
        )
    else:
        print(
            f"Provider '{provider_id}' not found in models.dev. "
            "Run with --list to see all providers.",
            file=sys.stderr,
        )
    sys.exit(1)


def find_model(provider_data: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """Return model data dict, or exit with suggestions."""
    models: Dict[str, Any] = provider_data.get("models") or {}
    if model_id in models:
        return models[model_id]
    lower = model_id.lower()
    matches = [k for k in models if k.lower() == lower or lower in k.lower()]
    if len(matches) == 1:
        return models[matches[0]]
    if matches:
        suggestions = "\n  ".join(matches[:10])
        print(
            f"Model '{model_id}' not found. Partial matches:\n  {suggestions}",
            file=sys.stderr,
        )
    else:
        print(
            f"Model '{model_id}' not found in provider. "
            "Run 'auto-config <provider> --list' to see all models.",
            file=sys.stderr,
        )
    sys.exit(1)


# ── Inference helpers ──────────────────────────────────────────────────────

def infer_format(provider_id: str, _provider_data: Dict[str, Any]) -> str:
    """Infer the API format for a whole provider (fallback when no per-model hint)."""
    if provider_id in _ANTHROPIC_FORMAT_PROVIDERS:
        return "anthropic"
    if provider_id in _GOOGLE_FORMAT_PROVIDERS:
        return "google"
    return "openai_chat"


def infer_model_format(
    model_data: Dict[str, Any],
    provider_id: str,
    provider_data: Dict[str, Any],
) -> str:
    """Infer format for a single model, checking model-level provider.npm first."""
    npm = (model_data.get("provider") or {}).get("npm", "")
    if npm in _NPM_TO_FORMAT:
        return _NPM_TO_FORMAT[npm]
    return infer_format(provider_id, provider_data)


def infer_base_url(provider_id: str, provider_data: Dict[str, Any]) -> Optional[str]:
    api = provider_data.get("api")
    if isinstance(api, str) and api:
        # Strip trailing /v1 — the router appends the versioned path itself.
        url = api.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        return url
    # api is None — check known table.
    url = _KNOWN_URLS.get(provider_id, None)
    if url is not None:
        return url if url else None  # empty string → native provider, no base_url
    return None  # Unknown; user must fill in.


def infer_multimodality(model_data: Dict[str, Any]) -> List[str]:
    input_types = model_data.get("modalities", {}).get("input") or []
    return [m for m in ("image", "audio", "video") if m in input_types]


def infer_pricing(model_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cost = model_data.get("cost")
    if not cost or not isinstance(cost, dict):
        return None
    p: Dict[str, Any] = {
        "currency": "USD",
        "input": cost["input"],
        "output": cost["output"],
    }
    if "cache_read" in cost:
        p["cache_read"] = cost["cache_read"]
    if "cache_write" in cost:
        p["cache_write"] = cost["cache_write"]
    return p


def build_model_entry(
    model_data: Dict[str, Any],
    local_name: str,
) -> Any:
    """Return a dict (for rich entries) or string (for text-only models)."""
    multi = infer_multimodality(model_data)
    pricing = infer_pricing(model_data)

    entry: Dict[str, Any] = {}
    if multi:
        entry["multimodality"] = multi
    if pricing:
        entry["pricing"] = pricing

    if not entry:
        return local_name  # plain string entry

    return {"name": local_name, **entry}


# ── YAML merge ─────────────────────────────────────────────────────────────

def _model_name(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    return entry.get("name", "")


def merge_model_into_endpoint(
    ep: Dict[str, Any],
    model_entry: Any,
    local_name: str,
    online_id: str,
) -> None:
    """Insert or update *model_entry* in endpoint dict; update model_map if needed."""
    models_list: List[Any] = ep.setdefault("models", [])
    existing_names = [_model_name(m) for m in models_list]
    if local_name in existing_names:
        idx = existing_names.index(local_name)
        models_list[idx] = model_entry
    else:
        models_list.append(model_entry)

    if local_name != online_id:
        ep.setdefault("model_map", {})[local_name] = online_id


def merge_into_config(
    config_dict: Dict[str, Any],
    local_provider: str,
    base_url: Optional[str],
    api_key_env: Optional[str],
    # (local_name, online_id, model_data, api_format) — format may vary per model
    models_to_add: List[Tuple[str, str, Dict[str, Any], str]],
) -> None:
    """Merge provider+models into *config_dict* in-place."""
    providers: Dict[str, Any] = config_dict.setdefault("providers", {})

    if local_provider not in providers:
        provider_dict: Dict[str, Any] = {}
        if api_key_env:
            provider_dict["api_key"] = f"${{{api_key_env}}}"
        if base_url:
            provider_dict["base_url"] = base_url
        provider_dict["endpoints"] = {}
        providers[local_provider] = provider_dict

    provider_dict = providers[local_provider]
    endpoints: Dict[str, Any] = provider_dict.setdefault("endpoints", {})

    for local_name, online_id, model_data, api_format in models_to_add:
        if api_format not in endpoints:
            endpoints[api_format] = {"models": []}
        ep = endpoints[api_format]
        entry = build_model_entry(model_data, local_name)
        merge_model_into_endpoint(ep, entry, local_name, online_id)


# ── Backup ─────────────────────────────────────────────────────────────────

def backup_config(config_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = config_path.with_suffix(f".yaml.bak.{ts}")
    shutil.copy2(config_path, backup)
    return backup


# ── Colored diff ────────────────────────────────────────────────────────────

def print_diff(old_text: str, new_text: str, label: str) -> None:
    """Print a git-style colored unified diff between old and new YAML."""
    RED    = "\033[31m"
    GREEN  = "\033[32m"
    CYAN   = "\033[36m"
    RESET  = "\033[0m"

    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{label}",
        tofile=f"b/{label}",
        lineterm="",
    ))

    if not diff:
        print("(no changes)")
        return

    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            print(f"\033[1m{line}\033[0m")
        elif line.startswith("@@"):
            print(f"{CYAN}{line}{RESET}")
        elif line.startswith("+"):
            print(f"{GREEN}{line}{RESET}")
        elif line.startswith("-"):
            print(f"{RED}{line}{RESET}")
        else:
            print(line, end="")


# ── Display helpers ────────────────────────────────────────────────────────

def list_providers(data: Dict[str, Any]) -> None:
    BOLD = "\033[1m"
    GREY = "\033[90m"
    RESET = "\033[0m"
    print(f"\n{BOLD}{'Provider ID':<30} {'Name':<35} {'Models':>6}  Base URL{RESET}")
    print("─" * 95)
    for pid, pdata in sorted(data.items()):
        name = pdata.get("name", pid)
        model_count = len(pdata.get("models") or {})
        api = pdata.get("api") or ""
        if not isinstance(api, str):
            api = ""
        url = api[:45] if api else _KNOWN_URLS.get(pid, GREY + "(native)" + RESET)
        print(f"  {pid:<30} {name:<35} {model_count:>6}  {url}")
    print(f"\n{len(data)} providers total")


def list_models(provider_id: str, provider_data: Dict[str, Any]) -> None:
    BOLD = "\033[1m"
    GREY = "\033[90m"
    RESET = "\033[0m"
    models: Dict[str, Any] = provider_data.get("models") or {}
    name = provider_data.get("name", provider_id)
    print(f"\n{BOLD}{name} ({provider_id}){RESET} — {len(models)} models\n")
    print(f"{BOLD}{'Model ID':<50} {'Format':<14} {'Modalities':<22} Cost ($/MTok){RESET}")
    print("─" * 110)
    for mid, mdata in models.items():
        fmt = infer_model_format(mdata, provider_id, provider_data)
        input_types = mdata.get("modalities", {}).get("input") or []
        modalities = "+".join(t for t in input_types if t != "text") or "text-only"
        cost = mdata.get("cost") or {}
        if cost:
            cost_str = f"in={cost.get('input',0):.3f} out={cost.get('output',0):.3f}"
        else:
            cost_str = GREY + "n/a" + RESET
        print(f"  {mid:<50} {fmt:<14} {modalities:<22} {cost_str}")


# ── Main command ───────────────────────────────────────────────────────────

def auto_config_command(args: Any, config_path: Path) -> None:
    data = fetch_models_dev()

    # ── --list mode ────────────────────────────────────────────────────────
    if args.list:
        if not args.online_provider:
            list_providers(data)
        else:
            _, pdata = find_provider(data, args.online_provider)
            list_models(args.online_provider, pdata)
        return

    if not args.online_provider:
        print(
            "Usage: auto-config <online-provider> [<online-model-id>] [options]\n"
            "       auto-config --list                    # list all providers\n"
            "       auto-config <provider> --list         # list provider's models",
            file=sys.stderr,
        )
        sys.exit(1)

    online_provider_id, provider_data = find_provider(data, args.online_provider)
    local_provider = args.provider or online_provider_id

    base_url = infer_base_url(online_provider_id, provider_data)
    env_vars: List[str] = provider_data.get("env") or []
    api_key_env = env_vars[0] if env_vars else None

    # ── Build list of (local_name, online_id, model_data, api_format) ────────
    forced_format = args.format  # None unless user explicitly overrides
    if args.online_model_id:
        model_data = find_model(provider_data, args.online_model_id)
        local_model = args.model or args.online_model_id
        fmt = forced_format or infer_model_format(model_data, online_provider_id, provider_data)
        models_to_add: List[Tuple[str, str, Dict[str, Any], str]] = [
            (local_model, args.online_model_id, model_data, fmt)
        ]
    else:
        all_models = provider_data.get("models") or {}
        if not all_models:
            print(
                f"No models found for provider '{online_provider_id}' in models.dev.",
                file=sys.stderr,
            )
            sys.exit(1)
        models_to_add = [
            (
                mid, mid, mdata,
                forced_format or infer_model_format(mdata, online_provider_id, provider_data),
            )
            for mid, mdata in all_models.items()
        ]

    # ── Load existing config ───────────────────────────────────────────────
    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    config_dict: Dict[str, Any] = yaml.safe_load(config_text) or {}

    # ── Merge ──────────────────────────────────────────────────────────────
    merge_into_config(
        config_dict,
        local_provider=local_provider,
        base_url=base_url,
        api_key_env=api_key_env,
        models_to_add=models_to_add,
    )

    new_yaml = yaml.dump(config_dict, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # ── Dry-run ────────────────────────────────────────────────────────────
    if args.dry_run:
        # Normalize the old text through the same serializer so that
        # cosmetic differences (quotes, indentation style) don't show as changes.
        old_normalized = yaml.dump(
            yaml.safe_load(config_text) or {},
            default_flow_style=False, allow_unicode=True, sort_keys=False,
        )
        label = config_path.name
        print(f"# Dry run — would write to {config_path}\n")
        print_diff(old_normalized, new_yaml, label)
        return

    # ── Backup + write ─────────────────────────────────────────────────────
    if config_path.exists():
        backup = backup_config(config_path)
        print(f"Backed up existing config to: {backup}")

    config_path.write_text(new_yaml, encoding="utf-8")

    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    model_summary = (
        f"model '{models_to_add[0][0]}'"
        if len(models_to_add) == 1
        else f"{len(models_to_add)} models"
    )
    formats_used = sorted(set(fmt for _, _, _, fmt in models_to_add))
    print(
        f"{GREEN}✓{RESET} Added {model_summary} "
        f"from {BOLD}{online_provider_id}{RESET} "
        f"→ local provider {BOLD}{local_provider}{RESET} "
        f"(format: {', '.join(formats_used)})"
    )
    print(f"  Config: {config_path}")
    if api_key_env and local_provider not in (yaml.safe_load(config_text) or {}).get("providers", {}):
        print(f"  Remember to set ${{{api_key_env}}} in your environment.")
