# Layer stacks

Model `lightsmith.problems.LayerStackProblem`. A planar multilayer between a
semi-infinite superstrate and a semi-infinite substrate. Computes reflection, transmission and absorption
against wavelength, angle and polarisation.

## Fields

| Field | Meaning |
|---|---|
| `superstrate` | The incidence medium, a material |
| `layers` | `Layer(thickness_nm, material, coherent=True)`, listed from the superstrate side to the substrate side. Set `coherent=False` for thick slabs, such as a 1 mm substrate. |
| `substrate` | The exit medium, a material |
| `wavelengths` | Vacuum wavelengths in nm |
| `angles_deg` | Angles of incidence in the superstrate (default `[0]`) |
| `polarizations` | Any of `"s"` and `"p"` (default both) |
| `options` | Per-method solver options |

## Result

`SpectrumResult`:

- `r`, `t` and `a` per polarisation, indexed `[angle][wavelength]`;
- `r_amplitude` and `t_amplitude` as (real, imaginary) pairs, when every layer is coherent;
- `provenance` and `warnings`.

`SpectrumResult.difference(other)` gives the maximum and RMS differences between two spectra on the same
grids.

## Solvers

`auto` order: [TMM](../solvers/tmm.md) → [RCWA](../solvers/rcwa.md) → [FDTD](../solvers/fdtd.md).

- TMM is exact and fast, handles incoherent layers, and is differentiable.
- RCWA treats an unpatterned stack as a single-harmonic problem.
- FDTD handles angles up to 85° from the normal. Oblique angles cost one run per angle, wavelength and
  polarisation.

## Example

```python
import lightsmith as ls
from lightsmith.problems import Layer, LayerStackProblem

bragg = LayerStackProblem(
    superstrate=ls.ConstantIndex(n_real=1.0),
    layers=[Layer(thickness_nm=600 / 4 / n, material=ls.ConstantIndex(n_real=n))
            for _ in range(10) for n in (2.3, 1.45)],
    substrate=ls.ConstantIndex(n_real=1.52),
    wavelengths=ls.WavelengthRange(start_nm=450, stop_nm=800, num=351),
)
res = ls.solve(bragg)
```
