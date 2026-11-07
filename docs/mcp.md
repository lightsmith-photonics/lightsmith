# MCP server

`lightsmith-mcp` exposes the library to AI agents through the [Model Context Protocol](https://modelcontextprotocol.io).
It is built on the official `mcp` Python SDK (2.x) and speaks protocol 2026-07-28, falling back to the older
handshake for clients that need it. Transports: stdio (default) and streamable HTTP.

Everything is JSON. Units are nm and vacuum wavelengths; the time convention is `exp(-iωt)` (see
[Conventions](conventions.md) and the repository's [AGENTS.md](https://github.com/lightsmith-photonics/lightsmith/blob/main/AGENTS.md)).

It offers the same operations as the Python API and the [CLI](cli.md), with results as the same JSON models,
including `provenance` and `warnings`.

## Recommended agent workflow

1. Read the schema (`get_schema(kind)`) before building a problem.
2. Use nm and vacuum wavelengths, with `Im(ε) > 0` for loss (see [Conventions](conventions.md)).
3. Estimate before solving grid-based problems.
4. Solve, then check `warnings` before reporting numbers.
5. Cross-check important results with a second method (`compare`).

The repository's [AGENTS.md](https://github.com/lightsmith-photonics/lightsmith/blob/main/AGENTS.md) and [llms.txt](https://github.com/lightsmith-photonics/lightsmith/blob/main/llms.txt) give the same
guidance in compact form. The [agent benchmark](benchmarks/agents.md) measures how well an agent follows
it through this server.

## Install and run

```bash
uvx --from "lightsmith[mcp]" lightsmith-mcp                     # stdio, from PyPI: what MCP clients launch
uvx --from "lightsmith[mcp]" lightsmith-mcp --http --port 8765  # streamable HTTP at http://127.0.0.1:8765/mcp
```

Add the `gpu` extra (`"lightsmith[mcp,gpu]"`) for GPU solvers. From a clone, run `uv sync --extra gpu --extra mcp`
and then `uv run lightsmith-mcp`. `python -m lightsmith.mcp_server` is the same command.

## Client configuration

**Claude Code**:

```bash
claude mcp add lightsmith -- uvx --from "lightsmith[mcp]" lightsmith-mcp
```

From a clone, use `-- uv run --directory /path/to/lightsmith lightsmith-mcp` instead.

Add `-e LIGHTSMITH_MCP_MAX_MEMORY_GB=16` (or any other variable below) before `--` to configure it. To share the
server with a project, put it in `.mcp.json` at the project root:

```json
{
  "mcpServers": {
    "lightsmith": {
      "command": "uvx",
      "args": ["--from", "lightsmith[mcp]", "lightsmith-mcp"],
      "env": {"LIGHTSMITH_MCP_MAX_RUNTIME_S": "900"}
    }
  }
}
```

**Claude Desktop**: the same `mcpServers` block goes into `claude_desktop_config.json` (Settings → Developer →
Edit Config). For a clone, use `"command": "uv"` with `["run", "--directory", "/path/to/lightsmith", "lightsmith-mcp"]`
and absolute paths; on Windows write them as `C:/Users/you/lightsmith` or with escaped backslashes.

**VS Code** (`.vscode/mcp.json`), **Cursor** (`.cursor/mcp.json`) and most other clients take the same command:

```json
{
  "servers": {
    "lightsmith": {"type": "stdio", "command": "uvx",
                   "args": ["--from", "lightsmith[mcp]", "lightsmith-mcp"]}
  }
}
```

(Cursor and Windsurf use the `mcpServers` key instead of `servers`.)

**Over HTTP**: start `lightsmith-mcp --http --port 8765`, then point the client at `http://127.0.0.1:8765/mcp`,
for example `claude mcp add --transport http lightsmith http://127.0.0.1:8765/mcp`. The server binds to
127.0.0.1 by default. It refuses any other address unless `LIGHTSMITH_MCP_HTTP_TOKEN` is set; clients must then
send `Authorization: Bearer <token>`.

**From Python**: see [`examples/mcp_client.py`](https://github.com/lightsmith-photonics/lightsmith/blob/main/examples/mcp_client.py).

## Tools

| Tool | What it does | Hints |
|---|---|---|
| `list_problem_kinds` | Every problem kind, its methods, auto order, schema and example URIs. Start here. | read-only |
| `get_schema(kind?)` | Exact JSON Schema of one kind, a working example, its backends and the conventions. | read-only |
| `list_backends(kind?)` | Backend capability matrix: method, kinds, licence, GPU, availability, citation, auto rank. | read-only |
| `check_environment` | CPU/GPU devices, server mode (local/remote), budget, job slots. Never shows the API key. | read-only |
| `validate_problem(problem)` | Schema errors with field paths, the normalised problem, and which backends can run it and why not. | read-only |
| `estimate(problem, method, device)` | Size, memory, estimated run time and the budget verdict. Never solves. | read-only |
| `solve(problem, method, device, wait_s, plot)` | Budget-checked solve; returns a summary, or a job id if it takes longer than `wait_s`. | |
| `start_solve(problem, ...)` | Same check; returns a job id at once. | |
| `compare(problem, methods, reference, ...)` | Several methods, relative differences against a reference. | |
| `get_job(job_id, wait_s, plot)` | Status and progress; the summary when finished. Waits with progress notifications. | read-only |
| `list_jobs(status?)` | Jobs, newest first. Finished jobs survive restarts. | read-only |
| `cancel_job(job_id)` | Cancels a queued or running job; a running solver process is killed. | destructive, idempotent |
| `get_result(job_id, format)` | Full result as JSON or CSV, for clients that cannot read resources. | read-only |
| `search_materials(query, sources)` | Bundled datasets, refractiveindex.info catalogue, Lightsmith Materials API. | read-only, network |
| `get_material(material, ...)` | n, k, ε at chosen wavelengths, valid range, citation, and the material JSON. | read-only, network |
| `fit_drude_lorentz(material, start_nm, stop_nm)` | Passive Drude–Lorentz model for FDTD, with fit errors. | read-only, network |

Tool inputs have JSON Schemas. The `problem` argument accepts every kind. Its `kind` enum, the method names and
`list_problem_kinds` are read from the problem union and the backend registry when the server starts, so new
kinds and backends, including plugins registered through the `lightsmith.backends` entry point, appear without
server changes. The exact per-kind schema is a separate call (`get_schema(kind)`) instead of being inlined into
every tool, because the full union schema is about 45 kB.

Errors are tool errors (`isError: true`) whose text says what to change. For example, a bad problem lists each
failing field path (`particles.0.material: Field required`) and points at `get_schema(kind=…)` and the example.

## Budget

Every `solve`, `start_solve` and `compare` is estimated first. A solve is refused when its estimated memory or run
time exceeds the budget, or when the backend says it does not fit (`fits: false`). The refusal includes the
estimate and specific ways to shrink the problem, such as `lower options.dda.dipoles_per_diameter` or
`raise options.fdfd.dl_nm`. Run-time estimates come from the backend when it reports one, from a calibrated model
for DDA, and otherwise are an order-of-magnitude guess. The `runtime_basis` field says which.

## Jobs

All solves are jobs. `solve` waits up to `wait_s` (default 20 s) and returns the summary if the job finished in
time; otherwise it returns the job id while the solve carries on. Progress arrives as MCP progress notifications
while a tool waits. Inverse design reports every iteration, `compare` reports each method, and other jobs report
coarse stages with a time-based estimate. If the client cancels a waiting `solve`, the job is cancelled too.

Runners:

- **Quick problems** (Mie, TMM, circuit models and anything else estimated under about 1.5 s) run in a thread of
  the server.
- **Everything else** runs in its own solver subprocess, so `cancel_job` kills it at once. A warm spare process,
  with its imports already done, is kept ready after the first `estimate` or job. Up to `LIGHTSMITH_MCP_MAX_JOBS`
  run at once; the rest queue.
- **Remote mode**: jobs go to Lightsmith Cloud (see below).

Results are kept under `LIGHTSMITH_MCP_DATA_DIR`, by default the user cache (`lightsmith/Cache/mcp-jobs`).

## Resources

| URI | Content |
|---|---|
| `lightsmith://conventions` | Units, conventions and AGENTS.md (markdown) |
| `lightsmith://schema`, `lightsmith://schema/{kind}` | JSON Schema of all kinds, or of one |
| `lightsmith://examples`, `lightsmith://examples/{name}` | Index of examples; one small working problem per kind, plus `examples/*.json` in a clone |
| `lightsmith://backends` | Backend capability matrix |
| `lightsmith://materials` | Bundled material datasets with citations |
| `lightsmith://jobs`, `lightsmith://jobs/{job_id}` | Job list; one job's state |
| `lightsmith://jobs/{job_id}/problem` | The problem as solved, with defaults filled in |
| `lightsmith://jobs/{job_id}/result` | The full result (JSON; NaN/Inf become `null`) |
| `lightsmith://jobs/{job_id}/result.csv` | Main spectra as a table: cross sections, R/T/A, n_eff, S-parameters, bands, optimisation history |
| `lightsmith://jobs/{job_id}/plot.png` | A plot of the result (needs matplotlib; `plot=true` on `solve`/`get_job` attaches it as image content) |

Completions are offered for `kind`, example names, job ids and prompt `material` arguments.

## Prompts

| Prompt | Arguments |
|---|---|
| `setup_simulation` | `kind`, `goal` |
| `scattering_study` | `particle`, `material`, `wavelength_range_nm`, `medium` |
| `cross_check_methods` | `problem_description` |
| `design_waveguide_component` | `component`, `platform`, `wavelength_nm` |
| `fit_material_for_fdtd` | `material`, `band_nm` |

## Materials

`search_materials` searches the bundled CC0 datasets offline. With `sources=["refractiveindex"]` it also
searches the refractiveindex.info catalogue, which is downloaded once and then cached. With
`LIGHTSMITH_MATERIALS_URL` set, it searches the Lightsmith Materials API as well.

`get_material` and `fit_drude_lorentz` accept any of these forms:

- a bundled name: `"Au"` or `"Au/Johnson-Christy"`;
- any material JSON;
- a refractiveindex.info entry: `{"refractiveindex": {"shelf", "book", "page"}}`;
- a Materials API dataset: `{"materials_api": "au/johnson-christy-1972"}`.

Each returns the `material_json` to put into a problem.

## Remote mode

When `LIGHTSMITH_CLOUD_URL` and `LIGHTSMITH_API_KEY` are both set, `estimate` and every job go to Lightsmith
Cloud (`POST /v1/estimate`, `POST /v1/jobs`, `GET /v1/jobs/{id}`, `DELETE /v1/jobs/{id}`). Estimates then carry
`credits_estimate`. A local `lightsmith-server` (`http://127.0.0.1:8200`) needs no key. Problems are still
validated locally first, so schema errors never cost a request.

The key is sent only in the `Authorization` header. It is never logged, echoed, put in a URL or returned:
`check_environment` reports only `api_key_set`.

In remote mode the memory budget is left to the service, which applies plan limits; the run-time budget still
applies.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LIGHTSMITH_MCP_MAX_MEMORY_GB` | 8 | Memory budget per solve (`--max-memory-gb`) |
| `LIGHTSMITH_MCP_MAX_RUNTIME_S` | 1800 | Run-time budget per solve (`--max-runtime-s`) |
| `LIGHTSMITH_MCP_JOB_TIMEOUT_S` | 2 × max runtime | Hard kill for local jobs |
| `LIGHTSMITH_MCP_INLINE_WAIT_S` | 20 | Default `wait_s` of `solve`/`compare` (`--inline-wait-s`) |
| `LIGHTSMITH_MCP_MAX_JOBS` | 1 | Concurrent local solver processes (`--max-jobs`) |
| `LIGHTSMITH_MCP_DATA_DIR` | user cache | Where job results are kept |
| `LIGHTSMITH_MCP_HTTP_TOKEN` | unset | Bearer token required on HTTP (required for a non-loopback `--host`) |
| `LIGHTSMITH_CLOUD_URL`, `LIGHTSMITH_API_KEY` | unset | Remote mode |
| `LIGHTSMITH_MATERIALS_URL` | unset | Lightsmith Materials API base URL |
| `LIGHTSMITH_CACHE_DIR` | user cache | Downloaded material data |

## MCP registry

[`server.json`](https://github.com/lightsmith-photonics/lightsmith/blob/main/server.json) describes the server for the MCP registry. Before publishing:

1. Release the package on PyPI.
2. Keep the `mcp-name` comment in the README; the registry uses it to verify ownership.
3. Check the file with `mcp-publisher validate`.

## Development

`uv run pytest tests/test_mcp_server.py` covers the tools, resources, prompts, completions, the budget, jobs,
progress, cancellation, persistence, remote mode against a fake Cloud API, and real client sessions over stdio
(both protocol versions) and streamable HTTP. The code is in `src/lightsmith/mcp_server/`:

| File | Contents |
|---|---|
| `server.py` | Tools, resources, prompts |
| `kinds.py` | Kind discovery, schemas, examples |
| `budget.py` | Guardrails |
| `jobs.py` and `_worker.py` | Runners |
| `summaries.py` | Summaries, CSV and plots |
| `matlib.py` | Materials |
| `remote.py` | Cloud and Materials API clients |
| `config.py` | Configuration |
