# omniscatter

**One problem spec, many electromagnetic scattering solvers — on CPU or GPU, for humans and AI agents.**

Describe a nanoparticle scattering problem once (shapes, materials, host medium, illumination, wavelengths) and solve it with Mie theory, the T-matrix method, the discrete dipole approximation (DDA) or FDTD. Results come back in the same units and conventions, with provenance, so methods can be compared directly.

> Status: pre-alpha (v0.1). Mie and DDA work; T-matrix and FDTD backends are next (see the roadmap).

## Install

```bash
uv add omniscatter                 # CPU
uv add "omniscatter[gpu]"          # + PyTorch for GPU solvers
uv add "omniscatter[mcp]"          # + MCP server for AI agents
```

From a clone (pulls CUDA 12.8 PyTorch wheels on Windows/Linux):

```bash
uv sync --extra gpu --extra mcp
uv run omniscatter doctor
```

## Quick start

```python
import omniscatter as osc

sim = osc.Simulation(
    particles=[osc.Particle(shape=osc.Sphere(radius_nm=40),
                            material=osc.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07))],
    medium_index=1.33,
    wavelengths=osc.WavelengthRange(start_nm=400, stop_nm=800, num=81),
)

res = osc.solve(sim)                         # auto -> Mie for a single sphere
print(res.summary())

osc.estimate(sim, method="dda", device="gpu")  # size and memory, no solve
cmp = osc.compare(sim, ["mie", "dda"])        # cross-check against Mie
print(cmp["relative_difference"])
```

The same from the shell, with JSON in and out:

```bash
uv run omniscatter solve examples/drude_sphere.json --method dda --device gpu
```

## Methods

| Method | Backend | Licence | Device | Geometry | Status |
|---|---|---|---|---|---|
| Mie | [miepython](https://github.com/scottprahl/miepython) | MIT | CPU | single sphere | ✅ |
| DDA | native (FFT + BiCGStab, array API) | MIT | CPU, CUDA GPU | any voxelised shapes, multiple particles | ✅ |
| T-matrix | planned | — | — | — | 🚧 |
| FDTD | planned | — | — | — | 🚧 |

`omniscatter backends` lists what is installed. GPL-licensed solvers can be added as separate plugin packages through the `omniscatter.backends` entry point, keeping this core MIT.

## Conventions

- Lengths in nm; wavelengths are vacuum wavelengths; host medium has a real index.
- Time dependence `exp(-iωt)`: absorbing materials have `Im(ε) > 0`.
- Cross sections in nm². Efficiencies use the volume-equivalent radius: `Q = C / (π r_eq²)`.

## For AI agents

- `omniscatter schema` prints the JSON Schema of a simulation.
- Every CLI command prints JSON; failures print `{"error", "message"}` to stderr with exit code 1.
- `omniscatter-mcp` runs an MCP server with `get_schema`, `list_backends`, `check_environment`, `estimate`, `solve` and `compare` tools.
- See [AGENTS.md](AGENTS.md) and [llms.txt](llms.txt).

## Roadmap

- T-matrix and FDTD backends (chosen from the survey in [docs/landscape.md](docs/landscape.md))
- Material database loader (refractiveindex.info)
- Near fields, orientation averaging, substrates
- Cross-method benchmark suite (the basis of the accompanying preprint)

## Licence

MIT. Each backend keeps its own licence; please cite the backends you use (`Result.provenance`, `omniscatter backends`).
