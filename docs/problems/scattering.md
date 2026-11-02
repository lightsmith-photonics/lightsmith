# Scattering

`kind: "scattering"`, model [`lightsmith.Simulation`](../api/spec.md). One or more particles in a
homogeneous, non-absorbing host medium, illuminated by a plane wave.

## Fields

| Field | Meaning |
|---|---|
| `particles` | One or more `Particle(shape, material, position_nm)` |
| `medium_index` | Real refractive index of the host (default 1) |
| `illumination` | `PlaneWave(direction, polarization)`, default +z with x polarisation. `polarization` must be orthogonal to `direction`. Or `GaussianBeam(direction, polarization, waist_nm, focus_nm, waist_wavelength_nm)` (FDTD only; see below). |
| `wavelengths` | Vacuum wavelengths in nm (a list or a `WavelengthRange`) |
| `orientation_average` | Average the cross sections over all particle orientations |
| `outputs` | `cross_sections` (default on); optional `near_field` and `far_field` requests |
| `substrate` | `Substrate(material, interface_z_nm)`: a half-space z < `interface_z_nm` below the particles (FDTD only; see below). |
| `options` | Per-method solver options |

### Shapes

| `kind` | Parameters | Notes |
|---|---|---|
| `sphere` | `radius_nm` | |
| `layered_sphere` | `core_radius_nm`, `shells: [{thickness_nm, material}]` | The particle `material` is the core |
| `spheroid` | `semi_axis_xy_nm`, `semi_axis_z_nm` | Symmetry axis along z |
| `cylinder` | `radius_nm`, `length_nm` | Flat ends, axis along z |

### Gaussian beams

`GaussianBeam` is a focused beam with the angular spectrum exp(−(k_t w₀/2)²), each plane wave polarised
by rotating `polarization` with its direction. The angular weights are fixed, so the beam behaves like
one focused by a lens of fixed NA: the waist is `waist_nm` at `waist_wavelength_nm` (default: the middle
of the band) and scales with the wavelength. Waists below about λ/n need angles beyond NA 0.95, where the
spectrum is cut; such a focus is non-paraxial, wider than the Gaussian and elongated along the
polarisation. Cross sections are normalised by the intensity at the focus, ½ n |E(focus)|².

### Substrates

With a `substrate`, the background is the host above the plane and the substrate below; particles may sit on
it or cut into it. FDTD supports plane waves at normal incidence, from above (direction (0, 0, −1)) or
through a non-absorbing substrate (direction (0, 0, 1)); oblique incidence is not supported yet.

- The incident field is the background's own solution (incident, reflected and transmitted waves), and
  cross sections are per unit intensity of the incident wave alone, in the medium it comes from.
- `c_sca` is the power the particle scatters into both half-spaces.
- `c_abs` is the power absorbed inside the source box beyond what the background absorbs there. Over a
  lossy substrate that includes extra substrate absorption near the particle.
- `c_ext = c_sca + c_abs` is the power the particle removes from the background field.

Near fields are supported; far fields are not yet (they need the layered-medium far field).

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
[DDA](../solvers/dda.md), then [FDTD](../solvers/fdtd.md). The [quasistatic](../solvers/quasistatic.md)
solver is never picked automatically.

| | Mie | T-matrix | DDA | FDTD | Quasistatic |
|---|---|---|---|---|---|
| Spheres | ✅ | ✅ | ✅ | ✅ | ✅ |
| Layered spheres | ✅ (native) | ✅ | ✅ | ✅ | ✅ |
| Spheroids, cylinders | — | — | ✅ | ✅ | spheroids |
| Several particles | — | ✅ (spheres) | ✅ | ✅ | — |
| Near field | native | ✅ | ✅ | ✅ | — |
| Far field | native | ✅ | ✅ | ✅ | ✅ |
| Resonances (Q) | — | — | — | ✅ | — |
| Orientation average | ✅ | ✅ (analytic) | ✅ (sampled) | — | ✅ |
| Substrate | — | — | — | ✅ (normal incidence) | — |
| GPU | native | — | ✅ | ✅ | — |

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
