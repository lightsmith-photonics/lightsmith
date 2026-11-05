# Lightsmith

**The open photonics suite — GPU-native, differentiable, and built for AI agents.**

Lightsmith lets you describe a photonics problem once and solve it with whichever method fits. It covers
nanoparticles, thin films, metasurfaces, waveguides, photonic crystals, devices and circuits. Every result uses the same units and conventions and records its provenance, so you can
cross-check methods directly. Many solvers run on a GPU and can be differentiated, which is what powers
inverse design.

```python
import lightsmith as ls

sim = ls.Simulation(
    particles=[ls.Particle(shape=ls.Sphere(radius_nm=40), material=ls.LibraryMaterial(name="Au"))],
    medium_index=1.33,
    wavelengths=ls.WavelengthRange(start_nm=400, stop_nm=800, num=81),
)
res = ls.solve(sim)                             # auto -> exact Mie for a sphere
cmp = ls.compare(sim, ["mie", "dda", "fdtd"])   # cross-check methods
```

!!! warning "Pre-alpha"
    Lightsmith 0.2 is pre-alpha, so the API may still change between minor versions. Each solver page
    lists the validation behind it, with numbers taken from the test suite. Read the `warnings` on every
    result before you trust its numbers.

## What it solves

| Problem kind | Solvers (auto order first) | GPU | Gradients |
|---|---|---|---|
| [`scattering`](problems/scattering.md): particles, clusters | [Mie](solvers/mie.md), [T-matrix](solvers/tmatrix.md), [BEM](solvers/bem.md), [DDA](solvers/dda.md), [FDTD](solvers/fdtd.md), [quasistatic](solvers/quasistatic.md) | Mie, BEM, DDA, FDTD | Mie (functional API) |
| [`layer_stack`](problems/layer-stack.md): thin films | [TMM](solvers/tmm.md), [RCWA](solvers/rcwa.md), FDTD | ✅ | TMM, RCWA |
| [`periodic`](problems/periodic.md): gratings, metasurfaces | [RCWA](solvers/rcwa.md), FDTD (rectangular lattices) | ✅ | RCWA |
| [`mode`](problems/mode.md): waveguide cross-sections | [FDE](solvers/fde.md) | — | — |
| [`device`](problems/device.md): S-parameters between ports | [EME](solvers/eme.md), [BPM](solvers/bpm.md), [FDFD](solvers/fdfd.md) | BPM, FDFD | FDFD (adjoint) |
| [`bands`](problems/bands.md): photonic crystals | [PWE](solvers/pwe.md) | ✅ | dense path |
| [`circuit`](problems/circuit.md): netlists of compact models | [S-matrix circuit solver](solvers/circuit.md) | ✅ | ✅ |
| [`optimization`](problems/optimization.md): inverse design | [Topology optimisation on FDFD](solvers/inverse.md) | ✅ | — |

## Three ways in

All three take the same problem description, so a problem you build in Python can be saved to a file,
run from the shell, or handed to an agent unchanged.

- **Python**: `import lightsmith as ls`, then `ls.solve`, `ls.estimate` and `ls.compare`. See the
  [quick start](quickstart.md).
- **Command line**: `lightsmith solve spec.json`. See the [CLI](cli.md).
- **AI agents**: the `lightsmith-mcp` server. See [MCP](mcp.md).

## Where to go next

- [Installation](install.md): CPU, GPU and optional extras.
- [Conventions](conventions.md): units, time convention, and how cross sections are normalised.
- [Solvers](solvers/index.md): method, options, validation and limits for each solver.
- [PDK flow](pdk.md): parametric components, their simulation, and fitted models for circuits.
- [Benchmarks](benchmarks/index.md): cross-method accuracy and cost, and the agent benchmark.
- [Contributing](contributing.md): how to write a solver backend or a plugin package.
- [Related software](landscape.md): a survey of other open-source solvers.

## Licence and citation

Lightsmith is MIT-licensed. GPL-licensed solvers live in separate plugin packages (`lightsmith-adda`,
`lightsmith-scattnlay`), so the core stays MIT. If you use Lightsmith, please cite it. GitHub's "Cite this
repository" button reads `CITATION.cff`. Please also cite the solver backends you used: they are listed
in `Result.provenance` and by `lightsmith backends`.
