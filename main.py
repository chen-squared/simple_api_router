#!/usr/bin/env python3
"""Entry point for simple_api_router."""
import argparse
import sys
from pathlib import Path

import uvicorn

from router.config import load_config
from router.app import create_app
from router.logger import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple API Router")
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    setup_logging(config.server.log_level, config.server.log_file)

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
    )


if __name__ == "__main__":
    main()
