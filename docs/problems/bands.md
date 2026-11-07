# Photonic bands

Model `lightsmith.problems.BandProblem`. The band structure of a 2D or 3D photonic
crystal along a path in k-space.

## Fields

| Field | Meaning |
|---|---|
| `dimension` | `2` (default) or `3` |
| `lattice_vectors_nm` | Two or three lattice vectors (x, y, z) |
| `background` | Material that fills the unit cell |
| `structures_2d` | Unit-cell inclusions for 2D crystals (profiles) |
| `structures` | Unit-cell inclusions for 3D crystals (geometries) |
| `k_path` | `KPoint(label, reduced)`: corners of the path, in units of the reciprocal lattice vectors |
| `points_per_segment` | Number of k-points between corners (default 16) |
| `num_bands` | Number of bands (default 8) |
| `polarization` | 2D only. `te` (E in plane), `tm` (E along z) or `both`. |
| `options` | Per-method solver options |

## Result

`BandResult` contains:

- `k_reduced`;
- `k_distance`, for plotting;
- `k_labels`;
- `frequencies[polarization][k][band]`, in normalised units a/λ, with `a = |lattice_vectors_nm[0]|`.

## Solvers

[PWE](../solvers/pwe.md): plane-wave expansion, with dense or LOBPCG eigensolvers.
`lightsmith.backends.pwe` also provides these helpers:

- `band_gaps`;
- `density_of_states`;
- `supercell`, for line and point defects;
- `bands_from_permittivity`, which is differentiable.

## Example

```python
import lightsmith as ls
from lightsmith.problems import BandProblem, Circle, KPoint, Structure2D

a = 500.0
rods = BandProblem(
    lattice_vectors_nm=[(a, 0, 0), (0, a, 0)],
    background=ls.ConstantIndex(n_real=1.0),
    structures_2d=[Structure2D(profile=Circle(radius_nm=0.2 * a), material=ls.ConstantPermittivity(eps_real=8.9))],
    k_path=[KPoint(label="Γ", reduced=(0, 0, 0)), KPoint(label="X", reduced=(0.5, 0, 0)),
            KPoint(label="M", reduced=(0.5, 0.5, 0)), KPoint(label="Γ", reduced=(0, 0, 0))],
    polarization="tm",
)
res = ls.solve(rods)
```
