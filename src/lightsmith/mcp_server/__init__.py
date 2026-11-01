"""MCP server exposing lightsmith to AI agents (``uv sync --extra mcp``).

    lightsmith-mcp                                 # stdio (what MCP clients launch)
    lightsmith-mcp --http [--host 127.0.0.1] [--port 8765]   # streamable HTTP at /mcp

Tools take and return JSON; problems follow the schema from ``get_schema``.
Long solves run as jobs; full results are MCP resources. See docs/mcp.md.
Configuration comes from environment variables (see ``config.py``) and the
flags below.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys

from .config import GIB, ServerConfig


def build_server(config: ServerConfig | None = None):
    from .server import build_server as _build

    return _build(config)


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _bearer_guard(app, token: str):
    """ASGI middleware: every HTTP request needs ``Authorization: Bearer <token>``."""
    import hmac

    expected = f"Bearer {token}".encode()

    async def guarded(scope, receive, send):
        if scope["type"] == "http":
            got = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(got, expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b'{"error": "Unauthorized"}'})
                return
        await app(scope, receive, send)

    return guarded


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lightsmith-mcp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="HTTP port (default 8765)")
    p.add_argument("--path", default="/mcp", help="HTTP endpoint path (default /mcp)")
    p.add_argument("--max-memory-gb", type=float, help="memory budget per solve (LIGHTSMITH_MCP_MAX_MEMORY_GB)")
    p.add_argument("--max-runtime-s", type=float, help="run-time budget per solve (LIGHTSMITH_MCP_MAX_RUNTIME_S)")
    p.add_argument("--max-jobs", type=int, help="concurrent local solves (LIGHTSMITH_MCP_MAX_JOBS)")
    p.add_argument("--inline-wait-s", type=float, help="how long solve waits before returning a job id")
    a = p.parse_args(argv)

    cfg = ServerConfig.from_env(
        max_memory_bytes=a.max_memory_gb * GIB if a.max_memory_gb else None, max_runtime_s=a.max_runtime_s,
        max_jobs=a.max_jobs, inline_wait_s=a.inline_wait_s)
    mcp = build_server(cfg)
    if not a.http:
        mcp.run()
        return 0

    import uvicorn

    token = os.environ.get("LIGHTSMITH_MCP_HTTP_TOKEN")
    if not _loopback(a.host) and not token:
        print(f"refusing to serve on {a.host} without authentication: set LIGHTSMITH_MCP_HTTP_TOKEN "
              "(clients then send 'Authorization: Bearer <token>'), or bind to 127.0.0.1", file=sys.stderr)
        return 2
    app = mcp.streamable_http_app(streamable_http_path=a.path, host=a.host)
    if token:
        app = _bearer_guard(app, token)
    print(f"lightsmith-mcp: streamable HTTP on http://{a.host}:{a.port}{a.path} "
          f"({'bearer token required' if token else 'localhost only, no auth'})", file=sys.stderr)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


__all__ = ["ServerConfig", "build_server", "main"]
