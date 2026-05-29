#!/usr/bin/env python3
"""CLI entry point for simple_api_router."""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from simple_api_router.config import load_config
from simple_api_router.app import create_app
from simple_api_router.logger import setup_logging

_SERVICE_COMMANDS = frozenset(
    {"uninstall", "start", "stop", "restart", "status", "log"}
)


def _models_command(config_path: str) -> None:
    """Display all configured providers and models."""
    from simple_api_router.config import ModelEntry

    cfg_path = Path(config_path).expanduser().resolve()
    try:
        cfg = load_config(cfg_path, skip_env_check=True)
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    GREY       = "\033[90m"
    RESET      = "\033[0m"
    BOLD       = "\033[1m"

    _MTYPE_LABELS = {"image": "image", "audio": "audio", "video": "video", "pdf": "pdf"}

    def _fmt_modalities(entry) -> str:
        types = entry.multimodality or []
        if not types:
            return f"{GREY}{'text':<20}{RESET}"
        label = " ".join(_MTYPE_LABELS.get(t, t) for t in types)
        return f"\033[36m{label:<20}\033[0m"

    def _fmt_pricing(p) -> str:
        if p is None:
            return ""
        sym = "¥" if getattr(p, "currency", "CNY").upper() == "CNY" else "$"
        tier = p.tiers[0] if p.tiers else p
        parts = [f"{sym}{tier.input:.2f}in", f"{sym}{tier.output:.2f}out"]
        if tier.cache_read is not None:
            parts.append(f"{sym}{tier.cache_read:.2f}cr")
        if tier.cache_write is not None:
            parts.append(f"{sym}{tier.cache_write:.2f}cw")
        suffix = "+  /MTok" if p.tiers else "  /MTok"
        return f"  {GREY}{' '.join(parts)}{suffix}{RESET}"

    total = 0
    for pname, provider in cfg.providers.items():
        for ename, ep in provider.endpoints.items():
            if not ep.models:
                continue
            print(f"\n{BOLD}{pname}{RESET}  [{ename}]")
            for m in ep.models:
                entry = m if isinstance(m, ModelEntry) else ModelEntry(name=m)
                pricing = cfg.get_pricing(f"{pname}/{entry.name}")
                print(f"  {entry.name:<48} {_fmt_modalities(entry)}{_fmt_pricing(pricing)}")
                total += 1

    print(f"\n{total} models across {len(cfg.providers)} providers")

    aliases = cfg.server.model_map
    if aliases:
        print(f"\n{BOLD}[server aliases]{RESET}")
        for alias, target in aliases.items():
            print(f"  {alias:<48} → {target}")

    print(f"\nConfig: {cfg_path}")


def _test_command(model_str: str, config_path: str) -> None:
    """Quick-test a provider/model by sending a request through the running router server."""
    import time
    import httpx

    cfg_path = Path(config_path).expanduser().resolve()
    try:
        cfg = load_config(cfg_path, skip_env_check=True)
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    host = getattr(cfg.server, "host", None) or "127.0.0.1"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    port = getattr(cfg.server, "port", None) or 8080
    base_url = f"http://{host}:{port}"
    url = f"{base_url}/v1/messages"

    body = {
        "model": model_str,
        "messages": [{"role": "user", "content": "Say exactly: OK"}],
        "max_tokens": 10,
    }
    print(f"Testing {model_str} via {base_url} ...", flush=True)
    try:
        with httpx.Client(timeout=60.0) as client:
            start = time.monotonic()
            resp = client.post(url, json=body)
            ms = round((time.monotonic() - start) * 1000)

        if resp.status_code == 200:
            data = resp.json()
            preview = next(
                (b["text"][:120] for b in data.get("content", []) if b.get("type") == "text"),
                "",
            )
            print(f"\033[32m✓\033[0m  {model_str}  \033[90m{ms}ms\033[0m")
            if preview:
                print(f"   response: {preview.strip()}")
        else:
            try:
                err_detail = resp.json()
            except Exception:
                err_detail = resp.text[:300]
            print(f"\033[31m✗\033[0m  {model_str}  \033[90m{ms}ms\033[0m")
            print(f"   error: HTTP {resp.status_code}: {err_detail}")
            sys.exit(1)

    except httpx.ConnectError:
        print(f"\033[31m✗\033[0m  Cannot connect to router at {base_url}")
        print(f"   Start the server first: simple-api-router run --config {cfg_path}")
        sys.exit(1)
    except Exception as exc:
        print(f"\033[31m✗\033[0m  {model_str}")
        print(f"   error: {exc}")
        sys.exit(1)


def _load_env_file(path: Path) -> None:
    """Source a KEY=VALUE env file into os.environ (skipping comments/blanks)."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            # os.environ.setdefault so existing env vars take precedence
            os.environ.setdefault(key.strip(), value.strip())


def _run_server(config_str: str, env_file: Optional[str] = None) -> None:
    if env_file:
        _load_env_file(Path(env_file))

    config_path = Path(config_str)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    setup_logging(config.server.log_level, config.server.log_file)

    app = create_app(config, config_path=config_path)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
        log_config=None,  # don't let uvicorn reconfigure logging; we own the format
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="simple-api-router",
        description="Multi-provider LLM API router — unified Anthropic Messages API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "service commands (macOS launchd / Linux systemd):\n"
            "  install    Install and enable as a background service\n"
            "  uninstall  Remove the service\n"
            "  start      Start the service\n"
            "  stop       Stop the service\n"
            "  restart    Restart the service\n"
            "  status     Show service status\n"
            "  log        Tail service logs\n"
            "  usage      Show API usage statistics\n"
            "  models     List configured providers and models\n"
            "\n"
            "examples:\n"
            "  simple-api-router                          # start server (config.yaml)\n"
            "  simple-api-router --config ~/my.yaml       # start server with custom config\n"
            "  simple-api-router install                  # install as system service\n"
            "  simple-api-router install --config ~/my.yaml\n"
            "  simple-api-router status\n"
            "  simple-api-router log\n"
        ),
    )

    # ── top-level flags (backward-compat: `simple-api-router --config X`) ──
    parser.add_argument(
        "--config", "-c", metavar="PATH", default=None,
        help="Config file path (default: config.yaml)",
    )
    parser.add_argument(
        "--env-file", dest="env_file", metavar="PATH", default=None,
        help="Load KEY=VALUE env vars from file before starting",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── run (explicit subcommand mirror of the default behaviour) ──────────
    run_p = subparsers.add_parser("run", help="Start the API server (default)")
    run_p.add_argument("--config", "-c", metavar="PATH", default=None,
                       help="Config file path (default: config.yaml)")
    run_p.add_argument("--env-file", dest="env_file", metavar="PATH", default=None,
                       help="Load KEY=VALUE env vars from file before starting")

    # ── install ─────────────────────────────────────────────────────────────
    inst_p = subparsers.add_parser(
        "install",
        help="Install as a system service (launchd on macOS, systemd on Linux)",
    )
    inst_p.add_argument(
        "--config", "-c", metavar="PATH", default=None,
        help="Config file to use (default: auto-detected)",
    )
    inst_p.add_argument(
        "--exe", "-e", metavar="PATH", default=None,
        help="Explicit path to the simple-api-router executable",
    )

    # ── simple service commands ─────────────────────────────────────────────
    subparsers.add_parser("uninstall", help="Remove the system service")
    subparsers.add_parser("start",     help="Start the service")
    subparsers.add_parser("stop",      help="Stop the service")
    subparsers.add_parser("restart",   help="Restart the service")
    subparsers.add_parser("status",    help="Show service status")
    subparsers.add_parser("log",       help="Tail service logs")
    subparsers.add_parser("models",    help="List configured providers and models")

    # ── auto-config ─────────────────────────────────────────────────────────
    ac_p = subparsers.add_parser(
        "auto-config",
        help="Auto-generate config from models.dev (e.g. auto-config openrouter)",
    )
    ac_p.add_argument(
        "online_provider", nargs="?", default=None,
        metavar="ONLINE_PROVIDER",
        help="Provider ID on models.dev (e.g. openrouter, groq, deepseek)",
    )
    ac_p.add_argument(
        "online_model_id", nargs="?", default=None,
        metavar="ONLINE_MODEL_ID",
        help="Specific model ID to add (omit to add all models for the provider)",
    )
    ac_p.add_argument(
        "--provider", default=None, metavar="NAME",
        help="Local provider name in your config (default: same as online provider ID)",
    )
    ac_p.add_argument(
        "--model", default=None, metavar="NAME",
        help="Local model name in your config (default: same as online model ID)",
    )
    ac_p.add_argument(
        "--format", default=None,
        choices=["anthropic", "openai_chat", "openai_responses", "google"],
        help="Override inferred API format",
    )
    ac_p.add_argument(
        "--list", action="store_true",
        help="List providers (or models for a given provider) without modifying config",
    )
    ac_p.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Print the resulting YAML without writing to disk",
    )
    ac_p.add_argument(
        "--config", "-c", metavar="PATH", default=None,
        help="Config file to update (default: auto-detect)",
    )

    # ── test ────────────────────────────────────────────────────────────────
    test_p = subparsers.add_parser(
        "test",
        help="Quick-test a model connection (e.g. test anthropic/claude-3-5-haiku-20241022)",
    )
    test_p.add_argument(
        "model",
        metavar="PROVIDER/MODEL",
        help="Model to test (e.g. openai/gpt-4o or myprovider/mymodel)",
    )
    test_p.add_argument(
        "--config", "-c", metavar="PATH", default=None,
        help="Config file path (default: auto-detect)",
    )

    # ── usage ───────────────────────────────────────────────────────────────
    usage_p = subparsers.add_parser("usage", help="Show API usage statistics")
    usage_p.add_argument(
        "--last", type=int, default=7, metavar="N",
        help="Number of days to include (default: 7)",
    )
    usage_p.add_argument(
        "--period", choices=["day", "week", "month"], default=None,
        help="Preset period (day/week/month); overrides --last",
    )
    usage_p.add_argument(
        "--daily", action="store_true",
        help="Show per-day breakdown instead of summary",
    )
    usage_p.add_argument(
        "--model", default=None, metavar="PATTERN",
        help="Filter by model name (substring match)",
    )
    usage_p.add_argument(
        "--provider", default=None, metavar="NAME",
        help="Filter by provider name",
    )
    usage_p.add_argument(
        "--logs", action="store_true",
        help="Print raw records as JSON lines (like the old JSONL file)",
    )
    usage_p.add_argument(
        "--import-jsonl", metavar="PATH",
        help="Import records from a legacy JSONL file into the usage database",
    )
    usage_p.add_argument(
        "--format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )
    usage_p.add_argument(
        "--config", "-c", metavar="PATH", default=None,
        help="Config file for pricing (default: auto-detect)",
    )

    args = parser.parse_args()
    cmd = args.command

    # ── service management dispatch ─────────────────────────────────────────
    if cmd == "install":
        from simple_api_router.service import install
        install(config=args.config, exe_override=args.exe)
        return

    if cmd == "auto-config":
        from simple_api_router.service import resolve_config
        from simple_api_router.auto_config import auto_config_command
        cfg = Path(resolve_config(getattr(args, "config", None)))
        auto_config_command(args, cfg)
        return

    if cmd == "models":
        from simple_api_router.service import resolve_config
        _models_command(str(resolve_config(args.config)))
        return

    if cmd == "test":
        from simple_api_router.service import resolve_config
        _test_command(args.model, str(resolve_config(getattr(args, "config", None))))
        return

    if cmd == "usage":
        from simple_api_router.usage_cli import usage_command
        usage_command(args)
        return

    if cmd in _SERVICE_COMMANDS:
        import importlib
        mod = importlib.import_module("simple_api_router.service")
        getattr(mod, cmd)()
        return

    # ── run server (default when no subcommand, or explicit `run`) ──────────
    _run_server(args.config or "config.yaml", args.env_file)


if __name__ == "__main__":
    main()

