"""Drive the Lightsmith MCP server from Python, the way an agent does: over stdio.

Discovers the problem kinds, estimates and solves a gold nanosphere, reads the
full result as a CSV resource, then starts a waveguide mode solve as a job and
follows its progress.

    uv run python examples/mcp_client.py
"""

import json
import sys

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters


async def main() -> None:
    server = StdioServerParameters(command=sys.executable, args=["-m", "lightsmith.mcp_server"])
    async with Client(server, read_timeout_seconds=600) as c:
        kinds = (await c.call_tool("list_problem_kinds", {})).structured_content
        print("kinds:", [k["kind"] for k in kinds["kinds"]])

        sphere = (await c.call_tool("get_schema", {"kind": "scattering"})).structured_content["example"]
        est = (await c.call_tool("estimate", {"problem": sphere})).structured_content
        print("estimate:", est["estimate"]["backend"], "ok" if est["budget_check"]["ok"] else "over budget")

        out = (await c.call_tool("solve", {"problem": sphere})).structured_content
        peak = out["summary"]["peak_extinction"]
        print(f"peak extinction at {peak['wavelength_nm']} nm, Q_ext = {peak['q_ext']}")
        csv = (await c.read_resource(out["job"]["resources"]["csv"])).contents[0].text
        print("CSV header:", csv.splitlines()[0])

        mode = (await c.call_tool("get_schema", {"kind": "mode"})).structured_content["example"]
        job = (await c.call_tool("start_solve", {"problem": mode})).structured_content
        print("job", job["job_id"], job["status"])

        async def progress(p, total, message):
            print(f"  {p:.0%} {message}")

        done = (await c.call_tool("get_job", {"job_id": job["job_id"], "wait_s": 300},
                                  progress_callback=progress)).structured_content
        print(json.dumps(done["summary"]["at_wavelengths"][0]["modes"][0], indent=1))


if __name__ == "__main__":
    anyio.run(main)
