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

`res.model_dump()` turns it into plain JSON-ready data.

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

## JSON in, JSON out

Every problem is plain JSON, and `kind` selects the problem type. A missing `kind` means `scattering`.

```json
{
  "particles": [
    {"shape": {"kind": "sphere", "radius_nm": 40},
     "material": {"kind": "drude", "eps_inf": 9.5, "wp_ev": 9.0, "gamma_ev": 0.07}}
  ],
  "medium_index": 1.33,
  "wavelengths": {"start_nm": 400, "stop_nm": 800, "num": 41},
  "options": {"dda": {"dipoles_per_diameter": 16}}
}
```

```bash
lightsmith estimate drude_sphere.json --method dda
lightsmith solve drude_sphere.json --method dda --device gpu -o result.json
lightsmith compare drude_sphere.json --methods mie,dda
```

From Python, `ls.solve` also accepts a dict or a JSON string. `lightsmith schema` prints the JSON Schema of
every problem kind; that is what an AI agent reads before it builds a problem.

## Next steps

- [Problem kinds](problems/index.md): what each kind describes and returns.
- [Solvers](solvers/index.md): the options and the validated accuracy of each backend.
- The repository's `examples/` folder has runnable scripts: a Mach–Zehnder and a ring circuit, a
  splitter tree, an RCWA metalens, and an inverse-designed waveguide bend.
