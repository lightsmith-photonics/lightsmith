# EME

Backend `lightsmith-eme` (method `eme`, modules `lightsmith.backends.eme` and `lightsmith.backends._eme`).
Eigenmode expansion for [devices](../problems/device.md) whose ports lie on two opposite planes, in 2D
and 3D. It is exact for piecewise-uniform devices given enough modes, and well suited to long devices
where light travels mostly along one axis.

## Method

1. **Slicing.** The device between its port planes is cut into cells that are uniform along the
   port-normal axis. Tapers and curves are staircased into many cells, and adjacent identical cells are
   merged.
2. **Cell modes.** Each cell's modes come from the [FDE](fde.md) solver: full-vector cross-sections in
   3D, or slab modes in 2D.
3. **Interfaces.** S-matrices at each cell interface come from unconjugated mode overlaps, so lossy and
   evanescent modes stay biorthogonal. The cells and interfaces are cascaded with Redheffer star products
   (Bienstman & Baets 2001).
4. **Ports.** Port modes are solved in each port's window and coupled in through symmetrised overlaps.

GDS layouts are resolved through `lightsmith.layout`. Cell modes are cached across solves, in a
process-wide cache limited to 256 MB.

## Options

`options["eme"]` or `options["lightsmith-eme"]`.

| Option | Default | Meaning |
|---|---|---|
| `num_modes` | `None` | Modes per cell. Default 30 in 2D and 10 in 3D. |
| `grid_spacing_nm` | `None` | Cross-section grid step |
| `points_per_wavelength` | `None` | Default 40 in 2D and 15 in 3D |
| `cells_per_feature` | `6` | Used for the default spacing |
| `subpixel` | `"anisotropic"` | Permittivity averaging, as in [FDE](fde.md) |
| `subsamples` | `32` | Supersampling in cut cells |
| `boundary` | `"pec"` | Cross-section walls: `"pec"` or `"pml"` |
| `pml_cells` | `10` | PML thickness |
| `pml_reflection` | `1e-6` | Target PML reflection |
| `polarization` | `"auto"` | `"auto"` means `"te"` (in-plane E) in 2D and `"any"` in 3D. In 3D, filtering by `"te"` or `"tm"` uses `te_fraction` and makes the mode basis incomplete. |
| `taper_cells` | `None` | Cells per varying segment. Default: enough for the outline to move about 3 grid cells per step, clipped to 10–60. |
| `max_cell_length_nm` | `None` | Also split segments longer than this |
| `max_grid_cells` | `400000` | Refuse larger cross-section grids |
| `cache` | `True` | Reuse cell modes across solves |

The solver runs on the CPU. There are no gradients. Multimode ports are labelled `name@j`.

## Validation

All tests are in `tests/test_eme.py`.

| Test | Reference | Tolerance |
|---|---|---|
| `test_uniform_waveguide_3d_transmits_fully_with_the_fde_phase`: 500 × 220 nm, L = 2 µm | \|S21\| = 1; phase k₀·neff·L from FDE | 1e-6; reflection < 1e-8 |
| `test_uniform_waveguide_2d` | \|S21\| = 1 | 1e-5; \|S11\| < 1e-6 |
| `test_width_step_3d_is_reciprocal_and_passive`: 500 → 700 nm width step | reciprocity; passivity | \|S21 − S12\| < 1e-6; 0.97 < \|S21\|² < 1 (about 0.989 with 48+ modes) |
| `test_linear_taper_approaches_adiabatic_limit`: 0.5 → 3 µm taper, 1–30 µm long | adiabatic limit | T rises with length; T(30 µm) > 0.98 |
| `test_mmi_1x2_self_imaging_length`: W = 3 µm MMI | Soldano & Pennings self-imaging length | 50/50 split to 1e-6; best length within 0.9–1.0 × the paraxial estimate; transmission > 0.9 |

## Limits

- Every port must be normal to the same axis and lie on one of exactly two opposite planes. Each port must
  point into the device: `+` at the lower plane, `-` at the upper plane.
- In 2D, ports cannot be normal to z.
- Raises an error when a cross-section grid exceeds `max_grid_cells`, or when a port supports fewer modes
  than its `num_modes`.
- Warns when a cell has fewer modes than requested, and when Σ|S|² > 1.05 for an input. In that case,
  increase `num_modes` or refine the grid.
