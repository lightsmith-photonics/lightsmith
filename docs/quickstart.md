# Quick start

The workflow is the same for every problem kind:

1. Build a problem.
2. Call `estimate` to see what it will cost.
3. Call `solve`, or `compare` several methods.
4. Read `warnings` and `provenance` before you use the numbers.

## A gold nanosphere

```python
import lightsmith as ls

sim = ls.Simulation(
    particles=[ls.Particle(shape=ls.Sphere(radius_nm=40),
                           material=ls.LibraryMaterial(name="Au"))],   # Johnson & Christy
    medium_index=1.33,                                                 # water
    wavelengths=ls.WavelengthRange(start_nm=400, stop_nm=800, num=81),
)

res = ls.solve(sim)          # method="auto" -> Mie for a single sphere
print(res.method, res.backend)
print(max(res.c_ext_nm2))    # extinction cross section in nm²
print(res.warnings)          # always check these
```

`solve` returns a `Result`, a pydantic model. It holds:

- `wavelengths_nm`, `c_ext_nm2`, `c_sca_nm2`, `c_abs_nm2`;
- `equivalent_radius_nm`, for efficiencies `Q = C / (π r_eq²)`;
- optional `near_field` and `far_field`;
- `provenance` and `warnings`.

`res.model_dump()` turns it into plain Python data (dicts, lists and numbers).

## Estimate before you solve

Grid-based solvers (DDA, FDTD, FDFD, RCWA, PWE) can be expensive. `estimate` reports which backend and
device would run, the grid or matrix size, the precision and the memory. It never solves:

```python
ls.estimate(sim, method="fdtd", device="gpu")
```

## Cross-check methods

```python
cmp = ls.compare(sim, ["mie", "dda"])
cmp["reference"]             # "mie": Mie is the reference for spheres
cmp["relative_difference"]   # per method: c_ext/c_sca/c_abs relative differences
```

Mie is the reference whenever it runs. Otherwise the reference is the first method you list or, with no
list, the method `auto` would pick. Cross sections are compared relative to the reference, but never
relative to less than 0.1 % of its extinction, so the near-zero absorption of a lossless particle does not
blow up. The other problem kinds report absolute differences: R, T and A for spectra, |S|² and S for
devices and circuits, n_eff and n_g for modes, and a/λ for bands.

Method-specific options go in `options`, keyed by method (or by backend name):

```python
sim = sim.model_copy(update={"options": {"dda": {"dipoles_per_diameter": 16}}})
```

Each backend validates its own options, so an unknown key is an error rather than being silently ignored.

## A thin film

Every problem kind beyond scattering lives in `lightsmith.problems`:

```python
from lightsmith.problems import Layer, LayerStackProblem

ar = LayerStackProblem(
    superstrate=ls.ConstantIndex(n_real=1.0),
    layers=[Layer(thickness_nm=100, material=ls.ConstantIndex(n_real=1.38))],   # MgF2-like
    substrate=ls.ConstantIndex(n_real=1.52),
    wavelengths=ls.WavelengthRange(start_nm=400, stop_nm=800, num=41),
    angles_deg=[0, 45],
)
spec = ls.solve(ar)               # auto -> TMM
spec.r["s"][0]                    # reflectance, s-polarised, first angle, per wavelength
```

## Saving a problem and using the command line

Every problem, and every result, can be saved to a file and loaded back exactly:

```python
from pathlib import Path
from lightsmith.problems import parse_problem

Path("gold_sphere.json").write_text(sim.model_dump_json(indent=2))
sim2 = parse_problem(Path("gold_sphere.json").read_text())   # any problem kind
```

A saved problem is what the command line takes. It is the same problem, so the same methods and options
apply:

```bash
lightsmith estimate gold_sphere.json --method dda
lightsmith solve gold_sphere.json --method dda --device gpu -o result.json
lightsmith compare gold_sphere.json --methods mie,dda
```

You rarely need to write these files by hand. If you do, `examples/drude_sphere.json` in the repository is
a complete one, and `lightsmith schema` lists every field of every problem kind. AI agents use that same
schema through the [MCP server](mcp.md).

## Next steps

- [Problem kinds](problems/index.md): what each kind describes and returns.
- [Solvers](solvers/index.md): the options and the validated accuracy of each backend.
- The repository's `examples/` folder has runnable scripts: a Mach–Zehnder and a ring circuit, a
  splitter tree, an RCWA metalens, and an inverse-designed waveguide bend.
