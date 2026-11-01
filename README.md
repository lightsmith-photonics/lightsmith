# Lightsmith

**The open photonics suite — GPU-native, differentiable, and built for AI agents.**

Describe a photonics problem once, as typed JSON, and solve it with the method that fits: from nanoparticles and thin films to metasurfaces, waveguides, photonic crystals, devices and circuits. Every result comes back in the same units and conventions, with provenance, so methods can be cross-checked directly. Many solvers run on GPU and can be differentiated, which is what powers inverse design.

> Status: pre-alpha (v0.1). Validation numbers for every solver are in `tests/` and in the technical report (`lightsmith-paper` repository).

## Install

```bash
uv add lightsmith                    # CPU
uv add "lightsmith[gpu]"             # + PyTorch for GPU solvers
uv add "lightsmith[mcp]"             # + MCP server for AI agents
uv add "lightsmith[tmatrix]"         # + treams T-matrix backend
uv add "lightsmith[layout]"          # + gdsfactory import (Python >= 3.11)
```

From a clone (pulls CUDA 12.8 PyTorch wheels on Windows/Linux):

```bash
uv sync --extra gpu --extra mcp
uv run lightsmith doctor
```

## Quick start

```python
import lightsmith as ls

sim = ls.Simulation(
    particles=[ls.Particle(shape=ls.Sphere(radius_nm=40),
                           material=ls.LibraryMaterial(name="Au"))],   # Johnson & Christy
    medium_index=1.33,
    wavelengths=ls.WavelengthRange(start_nm=400, stop_nm=800, num=81),
)

ls.estimate(sim, method="fdtd", device="gpu")   # grid, steps, memory; never solves
res = ls.solve(sim)                             # auto -> exact Mie for a sphere
cmp = ls.compare(sim, ["mie", "dda", "fdtd"])   # cross-check methods
```

The same from the shell, JSON in and out:

```bash
uv run lightsmith solve examples/drude_sphere.json --method dda --device gpu
```

## Problems and solvers

| Problem kind | Solvers (auto order first) | GPU | Gradients |
|---|---|---|---|
| `scattering` — particles, clusters | Mie (native, miepython), T-matrix (treams), DDA, FDTD, quasistatic | ✅ | Mie |
| `layer_stack` — thin films | TMM, RCWA, FDTD | ✅ | TMM, RCWA |
| `periodic` — gratings, metasurfaces | RCWA, FDTD (normal incidence) | ✅ | RCWA |
| `mode` — waveguide cross-sections | FDE (full vector, bends) | — | — |
| `device` — S-parameters between ports | EME, FDFD (2D) | FDFD | FDFD |
| `bands` — photonic crystals | PWE (2D TE/TM, 3D) | ✅ | dense path |
| `circuit` — netlists of compact models | S-matrix circuit solver | ✅ | ✅ |
| `optimization` — inverse design | topology optimisation on FDFD | ✅ | — |

`lightsmith backends` lists what is installed. Materials: bundled CC0 datasets (Au, Ag, Al, Cu, Si, SiO2, TiO2, water) from the refractiveindex.info database, online fetch of any entry, and passive Drude–Lorentz fitting for time-domain solvers.

## From components to circuits (`lightsmith.pdk`)

An open design kit: a technology as data (`generic_soi()`: 220 nm Si on 2 µm oxide, open material data only; define others as JSON), parametric cells (straight, circular/Euler bends, taper, MMI 1×2/2×2, directional coupler, ring/racetrack, Y-branch, crossing, grating coupler, MZI) with gdsfactory-style ports and GDS output, and a flow that simulates each component through `solve` (EME, FDFD or whatever `AUTO_ORDER` picks), fits a compact model, caches it by parameter hash and uses it in netlists as `pdk:<cell>`:

```python
from lightsmith.pdk import get_component
from lightsmith.pdk.flow import component_model, netlist_problem
mzi = get_component("mzi", delta_length_nm=4000)
mzi.write_gds("mzi.gds")
model = component_model("mmi1x2")                  # simulate -> fit -> cache
circuit = netlist_problem(mzi, wavelengths)        # {"component": "pdk:mmi1x2", "settings": {...}}
```

See [examples/pdk_mzi.py](examples/pdk_mzi.py) (layout, per-component simulation, circuit vs direct simulation).

GPL-licensed solvers live in separate plugin packages discovered through the `lightsmith.backends` entry point (`lightsmith-adda`, `lightsmith-scattnlay`), keeping this core MIT.

## Conventions

- Lengths in nm; wavelengths are vacuum wavelengths.
- Time dependence `exp(-iωt)`: absorbing materials have `Im(ε) > 0`.
- Scattering cross sections in nm²; efficiencies use the volume-equivalent radius, `Q = C / (π r_eq²)`.

## For AI agents

- `lightsmith schema` prints the JSON Schema of every problem kind.
- Every CLI command prints JSON; failures print `{"error", "message"}` to stderr with exit code 1.
- `lightsmith-mcp` serves `get_schema`, `list_backends`, `check_environment`, `estimate`, `solve` and `compare`.
- See [AGENTS.md](AGENTS.md) and [llms.txt](llms.txt).

## Licence

MIT. Backends keep their own licences; cite the ones you use (`Result.provenance`, `lightsmith backends`).

## Related repositories

Lightsmith is split across sibling repositories:

| Repository | Contents |
|---|---|
| `lightsmith` (this one) | Core library, solvers, benchmarks |
| `lightsmith-adda`, `lightsmith-scattnlay` | GPL solver plugins (separate licence) |
| `lightsmith-paper` | arXiv technical report |
| `lightsmith-studio`, `lightsmith-community`, `lightsmith-cloud`, `lightsmith-materials`, `lightsmith-deploy` | GUI, web platform, compute service, materials database, full-stack deployment |
