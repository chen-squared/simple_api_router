"""Request/response debug logging — captures all 4 stages per request.

Stages:
  1_incoming_request   – raw Anthropic body received from the client
  2_upstream_request   – converted body sent to the upstream provider
  3_upstream_raw       – raw bytes/JSON received from the upstream
  4_downstream_sse     – Anthropic SSE events sent back to the client

Enable in config.yaml::

    server:
      debug_log: /tmp/router_debug.log

Each request is prefixed with a short request ID so correlated stages can be
grep-ed together::

    grep "req=ab12cd34" /tmp/router_debug.log
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

_lock = threading.Lock()
_path: Optional[Path] = None


def configure(path: str) -> None:
    """Set the debug log file path. Creates parent directories if needed."""
    global _path
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)


def enabled() -> bool:
    return _path is not None


def log(req_id: str, stage: str, content: Any) -> None:
    """Append one labelled section to the debug log file."""
    if _path is None:
        return
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    sep = "=" * 80
    if isinstance(content, bytes):
        body = content.decode("utf-8", errors="replace")
    elif isinstance(content, (dict, list)):
        body = json.dumps(content, ensure_ascii=False, indent=2)
    else:
        body = str(content)
    text = f"\n{sep}\n[{ts}] req={req_id}  stage={stage}\n{sep}\n{body}\n"
    with _lock:
        with _path.open("a", encoding="utf-8") as f:
            f.write(text)


async def tee_bytes_iter(
    source: AsyncIterator[bytes],
    req_id: str,
    stage: str,
) -> AsyncIterator[bytes]:
    """Yield every byte from *source* unchanged, then log the full accumulated body."""
    parts: List[bytes] = []
    async for chunk in source:
        parts.append(chunk)
        yield chunk
    log(req_id, stage, b"".join(parts))
