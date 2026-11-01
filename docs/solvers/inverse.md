# Inverse design

Backend `lightsmith-inverse` (method `inverse`, package `lightsmith.inverse`). Density-based topology
optimisation of [optimization problems](../problems/optimization.md). It needs PyTorch (the `gpu`
extra), and runs on the CPU or a GPU.

!!! note "3D inverse design"
    Another branch is adding 3D inverse design. See
    [3D inverse design](in-development/inverse-3d.md).

## Method

- **Backends.** The optimiser works with any backend that is differentiable and implements
  `differentiable_solve`. In 0.1 that is [`lightsmith-fdfd`](fdfd.md) (2D), the default.
- **Density pipeline.** For each design region:
  1. raw density ρ ∈ [0, 1];
  2. mirror symmetry;
  3. normalised conic filter;
  4. tanh projection (β, η);
  5. ε = ε_low + ρ·(ε_high − ε_low).
- **Minimum feature size.** Imposed with the geometric constraints of Zhou et al. (2015), in the form of
  Hammond et al. (2021). They switch on once β reaches `lengthscale_from_beta`.
- **β continuation.** β is multiplied by `beta_growth` every `beta_every` iterations, up to `beta_max`.
- **Optimisers.**
  - `mma`: globally convergent MMA (Svanberg 2002); the objective never increases.
  - `adam`: Adam, projected onto the box.
  - `lbfgsb`: SciPy L-BFGS-B.
- **Final design.**
  1. Thresholded; `objective_binary` is its score.
  2. Contoured with marching squares.
  3. Exported as prism structures and re-solved for validation.

## Settings

The spec fields are listed on [Inverse design (problem kind)](../problems/optimization.md).
`OptimizerSettings` defaults:

| Setting | Default |
|---|---|
| `method` | `"mma"` |
| `max_iter` | `100` |
| `learning_rate` (Adam) | `0.02` |
| `move_limit` (MMA) | `0.2` |
| `beta_start`, `beta_max`, `beta_growth`, `beta_every` | `4`, `64`, `2`, `20` |
| `lengthscale_weight`, `lengthscale_from_beta` | `100`, `16` |
| `ftol` | `0`: stop when the relative improvement over 10 iterations falls below this |

**Export** (`lightsmith.inverse.export`, no extra dependencies):

- `marching_squares`, which returns contours with their nesting depth;
- `to_structures`;
- `write_gds`, a built-in GDSII writer. Holes go to datatype + 1, to be subtracted downstream;
- `to_gdsfactory`, which needs the `layout` extra and subtracts the holes itself.

**Checkpoints.** `inverse.optimize(..., checkpoint="run.npz")` saves progress periodically and resumes
from the file when it is run again.

## Validation

All tests are in `tests/test_inverse.py` and need PyTorch.

| Test | Check | Tolerance |
|---|---|---|
| `test_filter_and_projection` | the filter preserves constants; the projection is monotone | 1e-12 |
| `test_lengthscale_constraint_flags_thin_features` | a 20 nm line is penalised far more than a 200 nm line | > 100× |
| `test_mma_is_monotone_on_a_quadratic` | monotone decrease within the bounds | 1e-12 |
| `test_marching_squares_disc_and_ring` | disc area against πr²; re-rasterised ring | 2 %; > 98 % of pixels |
| `test_gds_roundtrip` | written polygons read back | 1 nm |
| `test_bend_topology_optimisation_improves_monotonically`: 2D Si 90° bend, 12 MMA iterations | objective rises from < 0.1 to > 0.8; binary design > 0.6; exported polygons agree with the binary score | within 0.15 |
| `test_checkpoint_resume_and_beta_continuation` | a resumed run reproduces the history | exact |
| `test_field_objective_and_lbfgsb` | a field objective improves | — |

## Limits

- 2D in 0.1: the filters and the region mapping are 2D, and FDFD is 2D.
- Design regions must stay at least 3 cells from every port plane, and span at least 2 grid cells.
- Warns when the final design is less than 90 % binary (raise `beta_max` or `max_iter`). Warnings from
  the validation solve are passed on with the prefix `validation:`.
