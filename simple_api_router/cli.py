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

    MULTIMODAL = "\033[36mmultimodal\033[0m"
    TEXT       = "\033[90mtext      \033[0m"
    GREY       = "\033[90m"
    RESET      = "\033[0m"
    BOLD       = "\033[1m"

    def _fmt_pricing(p) -> str:
        if p is None:
            return ""
        sym = "¥" if getattr(p, "currency", "CNY").upper() == "CNY" else "$"
        tier = p.tiers[0] if p.tiers else p
        parts = [f"{sym}{tier.input:.3f}in", f"{sym}{tier.output:.3f}out"]
        if tier.cache_read is not None:
            parts.append(f"{sym}{tier.cache_read:.3f}cr")
        if tier.cache_write is not None:
            parts.append(f"{sym}{tier.cache_write:.3f}cw")
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
                cap = TEXT if entry.text_only else MULTIMODAL
                pricing = cfg.get_pricing(f"{pname}/{entry.name}")
                print(f"  {entry.name:<50} {cap}{_fmt_pricing(pricing)}")
                total += 1

    print(f"\n{total} models across {len(cfg.providers)} providers")
    print(f"Config: {cfg_path}")


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

    if cmd == "models":
        from simple_api_router.service import resolve_config
        _models_command(str(resolve_config(args.config)))
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

