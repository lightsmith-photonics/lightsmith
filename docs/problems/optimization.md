# Inverse design

Model `lightsmith.inverse.OptimizationProblem`. Density-based topology
optimisation of one or more design regions inside a [device](device.md), driven by adjoint gradients.

## Fields

| Field | Meaning |
|---|---|
| `device` | The `DeviceProblem` being optimised. Its ports and fixed structures stay as given. |
| `design_regions` | `DesignRegion` entries. Each has a box (`center_nm`, `size_nm`), `material_low` and `material_high`, an initial density (`init`), `filter_radius_nm`, `eta`, an optional `min_feature_nm`, and optional mirror `symmetry` (`x`, `y`). |
| `objective` | A sum of terms. `SParameterTerm` is a port-to-port power or amplitude. `FieldIntensityTerm` is the mean \|field\|² in a box. Each term has a `goal` (`maximize`, `minimize` or `target`), a `weight`, and an `aggregate` over wavelengths (`mean`, or a smooth `min` or `max`). |
| `optimizer` | `OptimizerSettings`: `method` (`mma`, `adam` or `lbfgsb`), `max_iter`, β-continuation (`beta_start`, `beta_max`, `beta_growth`, `beta_every`), and the length-scale constraint weights. |
| `backend` | The differentiable device solver. Default `lightsmith-fdfd`. |
| `options` | Overrides for the device solver's options |

## Result

`OptimizationResult` contains:

- `history`;
- `objective_initial`, `objective_final`, and `objective_binary` (the score of the thresholded design);
- `designs`, the density maps;
- `structures`, the exported polygons as prisms;
- `final_device`;
- `sparameters`, from re-solving the exported design;
- `provenance` and `warnings`.

Export the design with `lightsmith.inverse.export.write_gds`, or with `to_gdsfactory` when the `layout`
extra is installed.

## Running

- `ls.solve(problem)` runs the optimisation and validates the final design.
- `lightsmith.inverse.optimize(problem, device=..., callback=..., checkpoint="run.npz")` adds progress
  callbacks and resumable checkpoints.
- Both need the `gpu` extra (PyTorch), even on a CPU.

See [Inverse design](../solvers/inverse.md) for the method, and `examples/inverse_design_bend.py` for a
complete 90° silicon bend.
