# FDFD

Backend `lightsmith-fdfd` (method `fdfd`, modules `lightsmith.backends.fdfd` and `lightsmith.backends._fdfd`).
A 2D finite-difference frequency-domain solver for [devices](../problems/device.md). It gives adjoint
gradients with respect to every permittivity pixel, which makes it the engine for
[inverse design](inverse.md). It runs on the CPU or a CUDA GPU.

## Method

- **Equation.** The scalar 2D Helmholtz equation on a Yee grid: TM solves for E_z, TE for H_z with 1/ε at
  cell edges. A graded stretched-coordinate PML is added outside the domain (Shin & Fan 2012).
- **Geometry.** Permittivity comes from supersampled fill fractions. 3D geometries are cut at the domain
  centre in z.
- **Ports.** Port modes come from a 1D slab eigensolver on the same discretisation, so they are exact
  discrete modes. They are launched as unidirectional total-field/scattered-field sources.
- **S-parameters.** Each port's field is decomposed into forward and backward modal amplitudes. With
  every port excited, S = C·A⁻¹, which also removes waves that re-enter from imperfect PMLs.
- **Linear solvers.** SciPy SuperLU on the CPU. On the GPU, a block-tridiagonal Schur-complement LU in
  PyTorch, suited to devices up to about 500 cells across.
- **Gradients.** One transposed (adjoint) solve per wavelength, reusing the forward factorisation.
  `FDFD.differentiable_solve(...)` exposes them as a PyTorch autograd function.

## Options

`options["fdfd"]` or `options["lightsmith-fdfd"]`.

| Option | Default | Meaning |
|---|---|---|
| `polarization` | `"te"` | `"te"` means E in the xy plane (H_z); `"tm"` means E_z |
| `dl_nm` | `None` | Cell size. Default λ_min / (`cells_per_wavelength` · n_max). |
| `cells_per_wavelength` | `20` | Used for the default cell size |
| `pml_cells` | `16` | PML cells added outside the domain, as one value or (x, y). 0 puts a wall on that axis. |
| `pml_reflection` | `1e-8` | Target normal-incidence reflection |
| `solver` | `"auto"` | `"scipy"`, `"block"`, or `"auto"` (SuperLU on the CPU, block LU on the GPU) |
| `precision` | `"double"` | Factorisation precision of the block solver |
| `subsamples` | `4` | Supersampling per axis (1–16) |
| `max_cells` | `4000000` | Refuse larger grids |

## Validation

All tests are in `tests/test_fdfd.py`.

| Test | Reference | Tolerance |
|---|---|---|
| `test_straight_waveguide_transmits` (TE and TM) | \|S21\| = 1; discrete propagation phase | 1e-4; \|S11\| < 1e-3; phase < 1e-3 rad |
| `test_slab_mode_matches_analytic_neff` (TE and TM) | analytic slab neff | < 1e-3 at dl = 5 nm; second-order convergence |
| `test_multimode_ports_are_orthogonal` | no mode conversion in a straight guide | 1e-4 |
| `test_power_conservation_closed_guide` (TE and TM): lossless scatterers in a parallel-plate guide | power conservation; reciprocity | column sums 1e-6; S = Sᵀ to 1e-8 |
| `test_bend_transmission_grows_with_radius` | qualitative, plus passivity | T(R = 1.5 µm) > 0.95 |
| `test_y_splitter_is_symmetric` | symmetry | \|P2 − P3\| < 1e-6 |
| `test_adjoint_gradient_matches_finite_differences` (TE and TM) | central differences | 1e-3 relative |
| `test_block_solver_matches_scipy_on_cpu` | SuperLU | 1e-9 |
| `test_gds_layout_device` | explicit structures | 1e-12 |
| `test_gpu_matches_cpu` (gpu) | CPU | S to 1e-10; gradients 1e-6 |

## Limits

- 2D only (`dimension=2`); 3D FDFD is not implemented yet. Ports cannot be normal to z.
- Raises an error when:
  - the grid exceeds `max_cells`;
  - a port window is too narrow or too close to the PML;
  - a port has fewer propagating modes than requested;
  - the grid is too coarse for a mode (reduce `dl_nm`).
- Warns when a port mode is not guided, and when output power exceeds input by more than 1 % in a
  lossless device (refine `dl_nm` or thicken the PML).
