# Scattering

Model [`lightsmith.Simulation`](../api/spec.md). One or more particles in a
homogeneous, non-absorbing host medium, illuminated by a plane wave.

## Fields

| Field | Meaning |
|---|---|
| `particles` | One or more `Particle(shape, material, position_nm)` |
| `medium_index` | Real refractive index of the host (default 1) |
| `illumination` | `PlaneWave(direction, polarization)`, default +z with x polarisation. `polarization` must be orthogonal to `direction`. |
| `wavelengths` | Vacuum wavelengths in nm (a list or a `WavelengthRange`) |
| `orientation_average` | Average the cross sections over all particle orientations |
| `outputs` | `cross_sections` (default on); optional `near_field` and `far_field` requests |
| `substrate` | A half-space below the particles. Part of the schema, but no built-in backend supports it yet. |
| `options` | Per-method solver options |

### Shapes

| Shape | Parameters | Notes |
|---|---|---|
| `Sphere` | `radius_nm` | |
| `LayeredSphere` | `core_radius_nm`, `shells`: a list of `Shell(thickness_nm, material)` | The particle `material` is the core |
| `Spheroid` | `semi_axis_xy_nm`, `semi_axis_z_nm` | Symmetry axis along z |
| `Cylinder` | `radius_nm`, `length_nm` | Flat ends, axis along z |

### Field outputs

- `NearFieldRequest`: the total |E|² enhancement, and optionally the complex E, sampled on a square
  plane. Settings:
  - `plane`: `xy`, `xz` or `yz`;
  - `offset_nm`, `half_width_nm`, `points`;
  - `wavelengths_nm`.
- `FarFieldRequest`: dC_sca/dΩ in nm²/sr on a θ × φ grid. θ is measured from the incident direction.
  φ = 0 is the E-plane and φ = 90° the H-plane.

## Result

`Result` contains:

- `wavelengths_nm`, `c_ext_nm2`, `c_sca_nm2`, `c_abs_nm2`;
- `equivalent_radius_nm`;
- optional `near_field` and `far_field` lists;
- `provenance` and `warnings`.

`Result.to_xarray()` converts the cross sections to an xarray Dataset (`xarray` extra).

## Solvers

`method="auto"` tries [Mie](../solvers/mie.md), then [T-matrix](../solvers/tmatrix.md), then
[BEM](../solvers/bem.md), then [DDA](../solvers/dda.md), then [FDTD](../solvers/fdtd.md). The [quasistatic](../solvers/quasistatic.md)
solver is never picked automatically.

| | Mie | T-matrix | BEM | DDA | FDTD | Quasistatic |
|---|---|---|---|---|---|---|
| Spheres | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Layered spheres | ✅ (native) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Spheroids, cylinders | — | — | ✅ | ✅ | ✅ | spheroids |
| Several particles | — | ✅ (spheres) | ✅ (disjoint) | ✅ | ✅ | — |
| Near field | native | ✅ | ✅ | ✅ | ✅ | — |
| Far field | native | ✅ | ✅ | ✅ | — | ✅ |
| Orientation average | ✅ | ✅ (analytic) | ✅ (sampled) | ✅ (sampled) | — | ✅ |
| GPU | native | — | ✅ | ✅ | ✅ | — |

BEM does not support a substrate.

## Example

```python
import lightsmith as ls

sim = ls.Simulation(
    particles=[ls.Particle(shape=ls.Spheroid(semi_axis_xy_nm=10, semi_axis_z_nm=30),
                           material=ls.LibraryMaterial(name="Au"))],
    medium_index=1.33,
    illumination=ls.PlaneWave(direction=(1, 0, 0), polarization=(0, 0, 1)),  # E along the long axis
    wavelengths=ls.WavelengthRange(start_nm=450, stop_nm=900, num=91),
)
ls.estimate(sim, method="dda")
res = ls.solve(sim, method="dda")
```
