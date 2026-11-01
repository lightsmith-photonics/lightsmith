# AGENTS.md

Guidance for AI agents (and humans) using or developing omniscatter.

## Using the library

1. Build a `Simulation` (see `omniscatter schema` or `osc.json_schema()`). Units are **nm**; wavelengths are **vacuum** wavelengths; `medium_index` is the real host index.
2. Call `estimate` before `solve` for DDA/FDTD: it reports dipole count, grid, precision and memory without solving.
3. `solve(sim, method="auto")` picks the most accurate applicable method (Mie > T-matrix > DDA > FDTD). Errors explain why a backend was skipped and what to change.
4. `compare(sim, methods=[...])` cross-checks methods; Mie is the reference when the problem is a single sphere.
5. Results carry `provenance` (backend versions, device, precision, iterations, residuals) and `warnings`. Check `warnings` before trusting numbers.

Interfaces, all JSON in / JSON out:

- Python: `import omniscatter as osc`
- CLI: `omniscatter {doctor,backends,schema,estimate,solve,compare}` — errors are JSON on stderr, exit code 1.
- MCP: `omniscatter-mcp` (stdio), tools `get_schema`, `list_backends`, `check_environment`, `estimate`, `solve`, `compare`.

## Conventions (do not change without a schema version bump)

- Time dependence `exp(-iωt)`: lossy materials have `Im(eps) > 0`, `Im(n) > 0`. Backends using `exp(+iωt)` conjugate at their boundary.
- Cross sections in nm². Efficiencies are normalised by `π r_eq²`, with `r_eq` the volume-equivalent radius of all particles.
- `Simulation.options[<method>]` holds method-specific options, validated by that backend's own options model.

## Developing

- `uv sync --extra gpu --extra mcp`, then `uv run pytest`. GPU tests are marked `gpu` and skip without CUDA.
- New backends subclass `omniscatter.backends.Backend` and implement `supports`, `solve` (and ideally `estimate`, `available`). Register with `@register`, or from another package via the `omniscatter.backends` entry-point group. GPL-licensed solvers must live in separate plugin packages, never in this MIT core.
- Native solvers use the array API (`device.namespace()`), never `numpy`/`torch` directly in hot loops, so one code path serves CPU and GPU.
- Every new backend needs a test against Mie (or another reference) with a stated tolerance.
