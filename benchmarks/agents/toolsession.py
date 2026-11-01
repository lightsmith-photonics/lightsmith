"""A thin, logged MCP client session for the agent benchmark.

The harness talks to the Lightsmith MCP server only through the MCP protocol:
tools (and resources, when the server has them) are listed at run time, so the
same harness drives the original six-tool server and the rewrite with jobs,
resources and prompts. Every call is recorded as a :class:`ToolCall` with its
arguments, the parsed result and timing; graders read those records.

Transports: ``stdio`` (default; launches ``lightsmith-mcp`` or any command) and
``inproc`` (an in-process server object, for fast tests).
"""

from __future__ import annotations

import base64
import json
import shutil
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def default_server_command() -> list[str]:
    """The ``lightsmith-mcp`` script of the running environment (falls back to PATH)."""
    exe = shutil.which("lightsmith-mcp", path=str(Path(sys.executable).parent)) or shutil.which("lightsmith-mcp")
    return [exe or "lightsmith-mcp"]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """One tool call as the agent made it and as the server answered."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    result: Any = None  # parsed JSON (structured content, or JSON text) or plain text
    text: str = ""  # the text the model sees
    images: list[dict[str, str]] = field(default_factory=list)  # {"mime_type", "data" (base64)}
    error: str | None = None
    duration_s: float = 0.0
    started_at: float = 0.0
    source: str = "mcp"  # mcp | harness (resource helpers, submit_answer)

    def to_json(self, max_chars: int | None = None) -> dict[str, Any]:
        d = asdict(self)
        d["images"] = [{"mime_type": i["mime_type"], "bytes": len(i["data"]) * 3 // 4} for i in self.images]
        if max_chars and len(d["text"]) > max_chars:
            d["text"] = d["text"][:max_chars] + f"... [{len(self.text)} chars]"
        return d


def _parse_text(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def parse_call_result(res) -> tuple[bool, Any, str, list[dict[str, str]]]:
    """(ok, parsed, text, images) of an MCP CallToolResult, whatever the server put where."""
    texts, images = [], []
    for c in res.content or []:
        kind = getattr(c, "type", None)
        if kind == "text":
            texts.append(c.text)
        elif kind == "image":
            images.append({"mime_type": c.mime_type, "data": c.data})
        elif kind == "resource":
            r = c.resource
            texts.append(getattr(r, "text", None) or f"[embedded resource {r.uri}]")
        elif kind == "resource_link":
            texts.append(f"[resource {c.uri}]")
    text = "\n".join(texts)
    parsed: Any = res.structured_content
    if isinstance(parsed, dict) and set(parsed) == {"result"}:
        parsed = parsed["result"]
    if parsed is None and texts:
        parsed = _parse_text(texts[0]) if len(texts) == 1 else [_parse_text(t) for t in texts]
    if not text and parsed is not None:
        text = json.dumps(parsed)
    return not bool(res.is_error), parsed, text, images


class ToolSession:
    """Async context manager around an MCP client; records every call in ``self.calls``."""

    def __init__(self, transport: str = "stdio", command: list[str] | None = None, server: Any = None,
                 cwd: str | None = None, env: dict[str, str] | None = None, call_timeout_s: float = 1800.0):
        self.transport = transport
        self.command = command or default_server_command()
        self.server = server
        self.cwd, self.env = cwd, env
        self.call_timeout_s = call_timeout_s
        self.calls: list[ToolCall] = []
        self.tools: list[ToolSpec] = []
        self.instructions: str | None = None
        self.server_name: str | None = None
        self.has_resources = False
        self._stack: AsyncExitStack | None = None
        self.client = None

    async def __aenter__(self) -> "ToolSession":
        from mcp import Client
        from mcp.client.stdio import StdioServerParameters

        if self.transport == "stdio":
            target = StdioServerParameters(command=self.command[0], args=self.command[1:], cwd=self.cwd, env=self.env)
        elif self.transport == "inproc":
            target = self.server if self.server is not None else _inproc_server()
        else:
            raise ValueError(f"unknown transport {self.transport!r}")
        self._stack = AsyncExitStack()
        self.client = await self._stack.enter_async_context(Client(target, read_timeout_seconds=self.call_timeout_s))
        self.instructions = self.client.instructions
        info = self.client.server_info
        self.server_name = f"{info.name} {getattr(info, 'version', '') or ''}".strip() if info else None
        self.has_resources = self.client.server_capabilities.resources is not None
        self.tools = await self.list_tools()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    async def list_tools(self) -> list[ToolSpec]:
        out, cursor = [], None
        while True:
            res = await self.client.list_tools(cursor=cursor)
            out += [ToolSpec(t.name, t.description or "", dict(t.input_schema or {"type": "object"}))
                     for t in res.tools]
            cursor = getattr(res, "next_cursor", None)
            if not cursor:
                return out

    def tool(self, *words: str, exclude: tuple[str, ...] = ()) -> str | None:
        """Name of the first listed tool whose name contains every word (case-insensitive)."""
        for t in self.tools:
            n = t.name.lower()
            if all(w in n for w in words) and not any(x in n for x in exclude):
                return t.name
        return None

    def schema_of(self, name: str) -> dict[str, Any]:
        return next((t.input_schema for t in self.tools if t.name == name), {})

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolCall:
        arguments = dict(arguments or {})
        t0 = time.perf_counter()
        rec = ToolCall(name=name, arguments=arguments, ok=False, started_at=time.time())
        try:
            res = await self.client.call_tool(name, arguments)
            rec.ok, rec.result, rec.text, rec.images = parse_call_result(res)
            if not rec.ok:
                rec.error = rec.text
        except Exception as e:  # protocol errors (unknown tool, bad arguments) reach the agent as errors
            rec.error = f"{type(e).__name__}: {e}"
            rec.text = rec.error
        rec.duration_s = round(time.perf_counter() - t0, 3)
        self.calls.append(rec)
        return rec

    # --- resources (exposed to agents as two helper tools when the server has any) ---

    async def list_resources(self) -> ToolCall:
        t0 = time.perf_counter()
        rec = ToolCall(name="list_resources", arguments={}, ok=False, started_at=time.time(), source="harness")
        try:
            res = await self.client.list_resources()
            items = [{"uri": str(r.uri), "name": r.name, "description": r.description} for r in res.resources]
            try:
                tpl = await self.client.list_resource_templates()
                items += [{"uri_template": t.uri_template, "name": t.name, "description": t.description}
                          for t in tpl.resource_templates]
            except Exception:  # templates are optional
                pass
            rec.ok, rec.result, rec.text = True, items, json.dumps(items)
        except Exception as e:
            rec.error = rec.text = f"{type(e).__name__}: {e}"
        rec.duration_s = round(time.perf_counter() - t0, 3)
        self.calls.append(rec)
        return rec

    async def read_resource(self, uri: str) -> ToolCall:
        t0 = time.perf_counter()
        rec = ToolCall(name="read_resource", arguments={"uri": uri}, ok=False, started_at=time.time(),
                       source="harness")
        try:
            res = await self.client.read_resource(uri)
            texts = []
            for c in res.contents:
                if getattr(c, "text", None) is not None:
                    texts.append(c.text)
                elif getattr(c, "blob", None) is not None:
                    mime = getattr(c, "mime_type", None) or "application/octet-stream"
                    if mime.startswith("image/"):
                        rec.images.append({"mime_type": mime, "data": c.blob})
                    else:
                        texts.append(f"[{mime}, {len(base64.b64decode(c.blob))} bytes]")
            rec.text = "\n".join(texts)
            rec.result = _parse_text(texts[0]) if len(texts) == 1 else texts
            rec.ok = True
        except Exception as e:
            rec.error = rec.text = f"{type(e).__name__}: {e}"
        rec.duration_s = round(time.perf_counter() - t0, 3)
        self.calls.append(rec)
        return rec


def _inproc_server():
    """The server object of the installed lightsmith.mcp_server (old: build_server(); new: build_server/create)."""
    import lightsmith.mcp_server as m

    for name in ("build_server", "create_server", "make_server"):
        if hasattr(m, name):
            return getattr(m, name)()
    if hasattr(m, "mcp"):
        return m.mcp
    raise RuntimeError("lightsmith.mcp_server exposes no server factory; use the stdio transport")
