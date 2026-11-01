# Waveguide modes

`kind: "mode"`, model `lightsmith.problems.ModeProblem`. The eigenmodes of a waveguide cross-section in
the xy plane, propagating along +z. The waveguide can be straight or bent.

## Fields

| Field | Meaning |
|---|---|
| `structures` | 2D structures (profile and material) of the cross-section |
| `background` | Cladding material |
| `window_center_nm`, `window_size_nm` | The computational window. Set the y size to 0 for a slab (1D) problem. |
| `wavelengths` | Vacuum wavelengths in nm |
| `num_modes` | Number of modes (default 2) |
| `target_neff` | Search near this effective index. Default: the highest index in the window. |
| `bend_radius_nm` | Bend about the y axis; `None` means straight |
| `include_fields` | Return the mode fields |
| `options` | Per-method solver options |

## Result

`ModeResult.modes[wavelength][mode]`, sorted by decreasing `neff_real`. Each mode is a `ModeData` with:

- `neff_real` and `neff_imag`;
- `ng`, the group index;
- `te_fraction`, the share of transverse |E|² in Ex;
- `loss_db_per_cm`;
- `effective_area_um2`;
- `field`, when requested.

## Solvers

- [FDE](../solvers/fde.md): full-vector finite differences, with bends and PML.
- [`lightsmith.circuits.waveguide_from_modes`](../api/circuits.md) turns a mode result into a waveguide
  compact model for [circuits](circuit.md).

## Example

```python
import lightsmith as ls
from lightsmith.problems import ModeProblem, Rectangle, Structure2D

strip = ModeProblem(
    structures=[Structure2D(profile=Rectangle(size_nm=(500, 220)), material=ls.ConstantIndex(n_real=3.476))],
    background=ls.ConstantIndex(n_real=1.444),
    window_size_nm=(3000, 2000),
    wavelengths=[1500, 1550, 1600],
    num_modes=2,
    options={"fde": {"grid_spacing_nm": 20}},   # coarse grid for a quick look; refine to converge
)
res = ls.solve(strip)
print(res.modes[1][0].neff_real, res.modes[1][0].ng)
```
