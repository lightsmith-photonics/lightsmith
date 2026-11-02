# 3D inverse design

`OptimizationProblem` runs density-based topology optimisation on any differentiable
backend. With `device.dimension = 3` the FDFD backend (`lightsmith-fdfd`) solves the
full-vector Maxwell equations, so the same JSON problem, optimisers, filters and
exports work for 3D devices such as SOI components with a real 220 nm silicon layer
and oxide cladding.

## Solver

- **Discretisation.** E-field curl-curl operator on a uniform cubic Yee grid with
  stretched-coordinate PML (Shin & Fan 2012), matrix-free and written against the
  Python array API, so the same code runs on NumPy (CPU) and PyTorch (CPU/CUDA).
  Interfaces are volume-averaged (4 x 4 x 4 subsamples per cell).
- **Ports.** Port modes are exact eigenmodes of the *discrete* 3D operator in a
  straight section (a generalised eigenproblem on the port plane), biorthogonal
  under the discrete `E x H` flux. Total-field/scattered-field mode sources are
  therefore unidirectional, and modal decomposition gives power-normalised
  S-parameters (multimode ports included) with no extra normalisation runs.
- **Linear solves.** Batched BiCGStab (one right-hand side per excited port mode),
  on the GPU when one is available. SuperLU is used only for tiny grids: its fill-in
  on 3D curl-curl matrices makes it impractical beyond ~10^4 unknowns. The
  grad-div regulariser of Shin & Fan (2013) and Jacobi preconditioning are
  available as options (`grad_div`, `jacobi`) but did not speed up BiCGStab in our
  tests, so they are off by default.
- **Adjoint.** One transposed solve per wavelength gives the gradient with respect
  to every voxel; the operator's dependence on permittivity is differentiated with
  autograd through one matvec. Forward and adjoint solves are warm-started from the
  previous optimisation iterate.

Why frequency domain rather than a time-domain adjoint on the native FDTD: the
inverse-design machinery is built on exact frequency-domain adjoints of
S-parameters. FDFD keeps that contract (two linear solves with A and A^T, exact
discrete port modes, one solve per wavelength) and benefits from warm starts across
iterations, which a time-domain run to steady state cannot exploit.

Options (`options["fdfd"]`): `dl_nm` (default: 12 cells per wavelength in the
densest material in 3D), `pml_cells` (int or `(x, y, z)`), `tol` (BiCGStab relative
residual, default 1e-6), `max_iterations`, `warm_start`, `precision`
(`"double"`/`"single"`), `solver` (`"auto"`, `"bicgstab"`, `"direct"`).

## Design regions in 3D

`DesignRegion.parametrization`:

- `"extruded"` (default): one 2D density in x-y, constant through the region's
  thickness `size_nm[2]`: a single vertical etch, as in planar fabrication. Partial
  cells at the top and bottom of the layer are mixed by their volume fraction. The
  conic filter, tanh projection with beta-continuation and the minimum-feature
  constraints act in 2D, exactly as in 2D designs, and the result exports to
  polygons and GDS (`inverse.marching_squares`, `inverse.write_gds`).
- `"volumetric"`: a free 3D density on the grid nodes, with a 3D conic filter and
  3D length-scale constraints. Exported as one slab of polygons per grid layer.

Keep at least four grid cells between port planes and design regions.

## Cost

`lightsmith.estimate(problem, method="inverse")` reports the grid, unknowns,
memory (BiCGStab work vectors plus the forward fields kept for the backward pass)
and a rough time per iteration (cold and warm-started). Each iteration costs one
forward solve (all excited port modes batched) and one adjoint solve per
wavelength.

## Gradient checks

`tests/test_fdfd3d.py` checks, among others:

- the matrix-free operator and its transpose against an independently assembled
  sparse matrix (to 1e-12);
- that launched port modes solve the discrete 3D equations (residual < 1e-9 of the
  operator scale) and are biorthogonal (to 1e-9);
- the TE0 index of a 500 x 220 nm strip against converged FDE (within 0.015 at
  10 nm cells, first-order staircasing);
- transmission (|S21|^2 = 1 within 2e-3, |S11|^2 < 1e-4) and reciprocity (`S21 = S12`
  to 1e-5) of a straight guide; reciprocity to 3e-3 with an asymmetric scatterer 4 cells
  from the monitor planes (radiation crossing the planes limits the two-plane modal
  decomposition, as in 2D);
- adjoint gradients against central finite differences through filter, projection,
  extrusion (or volumetric 3D filtering) and the 3D solve: relative error < 1e-4;
- (slow) an abrupt 450 -> 900 nm width step against EME (|S21|^2 within 0.01).

## Worked example

[`examples/inverse_design_3d_mode_converter.py`](../examples/inverse_design_3d_mode_converter.py)
optimises an SOI TE0 -> TE1 mode converter (2.0 x 1.6 um extruded design region,
120 nm minimum feature, 1530-1570 nm) from
[`inverse_design_3d_mode_converter.json`](../examples/inverse_design_3d_mode_converter.json),
exports GDS, and validates the exported polygons with EME, an independent method.

Results (one run, RTX 5070 and CPU shared with other jobs, so timings are not final):

| | TE0 -> TE1 power at 1550 nm |
|---|---|
| start (random density) | 0.006 |
| optimised, grey (beta = 64, 97 % binary), 3D FDFD 40 nm | 0.902 |
| hard-thresholded density, 3D FDFD 40 nm | 0.898 |
| exported polygons, 3D FDFD 40 nm (1530 / 1550 / 1570 nm) | 0.876 / 0.892 / 0.883 |
| exported polygons, 3D FDFD 25 nm (finer grid) | 0.884 |
| exported polygons, EME, 16 / 30 / 60 / 100 / 150 modes per cell | 0.20 / 0.50 / 0.74 / 0.80 / 0.82 |

30 MMA iterations (70 forward + adjoint evaluations, 1.3 million unknowns each)
took 73 minutes. The 3D FDFD result is grid-converged to about 0.01. EME converges
slowly in the number of modes on this strongly scattering, hole-riddled device (it
agrees with FDFD within 0.01 on a simple width step, `test_width_step_matches_eme`):
0.80 at 100 and 0.82 at 150 modes per cell, and a linear extrapolation in 1/(modes)
of the last three points gives 0.86-0.88, consistent with the 0.88 of 3D FDFD.
