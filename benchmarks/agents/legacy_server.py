"""Run the original (pre-rewrite) ``lightsmith-mcp`` server on the MCP Python SDK 2.x.

The six-tool server in ``lightsmith/mcp_server.py`` imports ``FastMCP``, which the
MCP SDK 2.x renamed to ``MCPServer``. This launcher aliases the old name and
starts the server unchanged (tool exceptions are reported with their message, as
SDK 1.x did), so the benchmark can exercise that server too:

    uv run --no-sync python benchmarks/agents/run.py --agent oracle \
        --server-cmd "python benchmarks/agents/legacy_server.py"

With a server that already supports SDK 2.x it simply runs it.
"""

import functools
import sys
import types

try:
    import mcp.server.fastmcp  # noqa: F401  (SDK 1.x)
except ImportError:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    class FastMCP(MCPServer):
        """SDK 2.x hides unexpected tool exceptions; SDK 1.x showed their message, which agents need."""

        def tool(self, *args, **kwargs):
            register = super().tool(*args, **kwargs)

            def deco(fn):
                @functools.wraps(fn)
                def wrapped(*a, **kw):
                    try:
                        return fn(*a, **kw)
                    except Exception as e:
                        raise ToolError(f"{type(e).__name__}: {e}") from e

                return register(wrapped)

            return deco

    shim = types.ModuleType("mcp.server.fastmcp")
    shim.FastMCP = FastMCP
    sys.modules["mcp.server.fastmcp"] = shim

from lightsmith.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
