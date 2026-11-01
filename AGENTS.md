# AGENTS.md

Guidance for AI agents (and humans) using or developing lightsmith.

## Using the library

1. Build a problem of one of the kinds in `lightsmith/problems.py` (scattering, layer_stack, periodic, mode, device, bands, circuit, optimization); `lightsmith schema` prints the JSON Schema of all of them. Units are **nm**; wavelengths are **vacuum** wavelengths.
2. Call `estimate` before `solve` for grid-based solvers (DDA, FDTD, FDFD, RCWA, PWE): it reports problem size, precision and memory without solving.
3. `solve(problem, method="auto")` picks the preferred applicable method for the problem kind (see `AUTO_ORDER` in `api.py`). Errors explain why a backend was skipped and what to change.
4. `compare(problem, methods=[...])` cross-checks methods; Mie is the reference for single spheres, otherwise the first method listed.
5. Results carry `provenance` (backend versions, device, precision, iterations, residuals) and `warnings`. Check `warnings` before trusting numbers.

Interfaces, all JSON in / JSON out:

- Python: `import lightsmith as ls`
- CLI: `lightsmith {doctor,backends,schema,estimate,solve,compare}` — errors are JSON on stderr, exit code 1.
- MCP: `lightsmith-mcp` (stdio; `--http` for streamable HTTP). Tools:
  - discovery: `list_problem_kinds`, `get_schema(kind)`, `list_backends`, `check_environment`, `validate_problem`;
  - solving: `estimate`, `solve`, `start_solve`, `compare`;
  - jobs: `get_job`, `list_jobs`, `cancel_job`, `get_result`;
  - materials: `search_materials`, `get_material`, `fit_drude_lorentz`.

  Solves over the server's memory/time budget are refused with ways to shrink them. Long solves return a job id.
  Full results are resources at `lightsmith://jobs/{id}/result` (JSON) and `/result.csv`. See `docs/mcp.md`.

## Conventions (do not change without a schema version bump)

- Time dependence `exp(-iωt)`: lossy materials have `Im(eps) > 0`, `Im(n) > 0`. Backends using `exp(+iωt)` conjugate at their boundary.
- Cross sections in nm². Efficiencies are normalised by `π r_eq²`, with `r_eq` the volume-equivalent radius of all particles.
- `Simulation.options[<method>]` holds method-specific options, validated by that backend's own options model.

## Developing

- `uv sync --extra gpu --extra mcp`, then `uv run pytest`. GPU tests are marked `gpu` and skip without CUDA.
- New backends subclass `lightsmith.backends.Backend` and implement `supports`, `solve` (and ideally `estimate`, `available`). Register with `@register`, or from another package via the `lightsmith.backends` entry-point group. GPL-licensed solvers must live in separate plugin packages, never in this MIT core.
- Native solvers use the array API (`device.namespace()`), never `numpy`/`torch` directly in hot loops, so one code path serves CPU and GPU.
- Every new backend needs a test against Mie (or another reference) with a stated tolerance.
