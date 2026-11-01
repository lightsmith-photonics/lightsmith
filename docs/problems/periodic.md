# Periodic structures

`kind: "periodic"`, model `lightsmith.problems.PeriodicProblem`. A stack of patterned layers that is
periodic in 1D or 2D, such as a grating, a metasurface or a photonic-crystal slab, illuminated by a plane
wave.

## Fields

| Field | Meaning |
|---|---|
| `lattice_a1_nm` | First lattice vector (x, y) |
| `lattice_a2_nm` | Second lattice vector, or `None` for a 1D grating. A 1D grating is periodic along a1 and invariant perpendicular to it; its shapes are sampled along the line through the origin in the a1 direction. |
| `superstrate`, `substrate` | Semi-infinite materials |
| `layers` | `PatternedLayer(thickness_nm, background, shapes)`, listed from the superstrate side. `shapes` are 2D structures in one unit cell. |
| `wavelengths` | Vacuum wavelengths in nm |
| `incidence` | `Incidence(theta_deg, phi_deg, polarizations)` |
| `options` | Per-method solver options |

## Result

`SpectrumResult`, with:

- total `r`, `t` and `a`;
- `orders`: a list of `DiffractionOrder(order=(m, n), r, t)` for the propagating orders;
- zeroth-order complex amplitudes.

## Solvers

- [RCWA](../solvers/rcwa.md) is the default. It handles any lattice, conical incidence and metals, and is
  differentiable with respect to thicknesses and pixelated permittivity maps.
- [FDTD](../solvers/fdtd.md) handles normal incidence on rectangular lattices only.

## Example

```python
import lightsmith as ls
from lightsmith.problems import Circle, PatternedLayer, PeriodicProblem, Structure2D

holes = PeriodicProblem(
    lattice_a1_nm=(400, 0), lattice_a2_nm=(0, 400),
    superstrate=ls.ConstantIndex(n_real=1.0),
    layers=[PatternedLayer(thickness_nm=150, background=ls.ConstantIndex(n_real=2.0),
                           shapes=[Structure2D(profile=Circle(radius_nm=100),
                                               material=ls.ConstantIndex(n_real=1.0))])],
    substrate=ls.ConstantIndex(n_real=1.45),
    wavelengths=ls.WavelengthRange(start_nm=588, stop_nm=604, num=9),
    options={"rcwa": {"harmonics": 5}},
)
res = ls.solve(holes)
```
