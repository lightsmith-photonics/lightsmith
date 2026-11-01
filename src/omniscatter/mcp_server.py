"""MCP server exposing omniscatter to AI agents (``uv sync --extra mcp``).

Run with ``omniscatter-mcp`` (stdio transport). Tools take and return plain
JSON; the simulation argument follows the schema from ``get_schema``.
"""

from __future__ import annotations

from typing import Any

from . import api
from .backends import all_backends
from .device import doctor
from .problems import problem_json_schema


def build_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("omniscatter")

    @mcp.tool()
    def get_schema() -> dict[str, Any]:
        """JSON Schema for every problem kind (scattering, layer_stack, periodic, mode, device, bands,
        circuit). Read this before building a problem."""
        return problem_json_schema()

    @mcp.tool()
    def list_backends() -> list[dict[str, Any]]:
        """Available solver backends, their methods, licences, GPU support and availability."""
        return [b.info() for b in all_backends()]

    @mcp.tool()
    def check_environment() -> dict[str, Any]:
        """CPU/GPU frameworks and devices visible to the solvers."""
        return doctor()

    @mcp.tool()
    def estimate(simulation: dict[str, Any], method: str = "auto", device: str = "auto") -> dict[str, Any]:
        """Pre-flight size/memory estimate. Call before solve for DDA/FDTD problems."""
        return api.estimate(simulation, method, device)

    @mcp.tool()
    def solve(simulation: dict[str, Any], method: str = "auto", device: str = "auto") -> dict[str, Any]:
        """Solve a simulation; returns cross sections (nm^2) per wavelength with provenance."""
        return api.solve(simulation, method, device).model_dump()

    @mcp.tool()
    def compare(simulation: dict[str, Any], methods: list[str] | None = None, device: str = "auto") -> dict[str, Any]:
        """Solve with several methods and report relative differences against a reference (Mie if applicable)."""
        c = api.compare(simulation, methods, device)
        c["results"] = {k: v.model_dump() for k, v in c["results"].items()}
        return c

    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
