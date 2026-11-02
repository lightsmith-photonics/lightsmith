# Dipole emission

`kind: "emission"`, model `lightsmith.problems.EmissionProblem`. A point electric dipole near particles in a
homogeneous, non-absorbing host medium. Every quantity is relative to the same dipole in the host alone
(power P0), so the numbers are the usual rate enhancements of a quantum emitter.

## Fields

| Field | Meaning |
|---|---|
| `particles` | Zero or more `Particle(shape, material, position_nm)`, the same shapes as [scattering](scattering.md) |
| `medium_index` | Real refractive index of the host (default 1) |
| `dipole` | `Dipole(position_nm, orientation)`; the orientation is normalised |
| `wavelengths` | Vacuum wavelengths in nm (a list or a `WavelengthRange`) |
| `radiation_pattern` | Optional `RadiationPatternRequest(theta_deg, phi_deg, wavelengths_nm)`. θ is measured from +z and φ from +x. |
| `options` | Per-method solver options |

## Result

`EmissionResult` contains, per wavelength:

- `purcell`: total decay-rate enhancement, the power the dipole emits over P0 (radiated plus absorbed);
- `radiative`: power reaching the far field over P0;
- `efficiency`: `radiative / purcell`, the antenna's quantum efficiency for an ideal emitter;
- optional `radiation_pattern`: dP/dΩ / P0 on the θ × φ grid, which integrates to `radiative`;
- `provenance` and `warnings`.

## Solvers

`method="auto"` tries [Mie](../solvers/mie.md), then [FDTD](../solvers/fdtd.md).

| | Mie (`lightsmith-mie`) | FDTD |
|---|---|---|
| One sphere (plain or layered), dipole outside | ✅ exact | ✅ |
| Other shapes, several particles, dipole inside a particle | — | ✅ |
| Radiation pattern | — | ✅ |
| GPU | — | ✅ |

## Example

```python
import lightsmith as ls
from lightsmith.problems import Dipole, EmissionProblem

prob = EmissionProblem(
    particles=[ls.Particle(shape=ls.Sphere(radius_nm=40), material=ls.LibraryMaterial(name="Au"))],
    medium_index=1.33,
    dipole=Dipole(position_nm=(0, 0, 50), orientation=(0, 0, 1)),  # radial, 10 nm from the surface
    wavelengths=ls.WavelengthRange(start_nm=500, stop_nm=700, num=41),
)
res = ls.solve(prob)  # Mie
print(res.purcell, res.efficiency)
```
