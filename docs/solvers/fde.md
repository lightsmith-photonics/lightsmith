# FDE

Backend `lightsmith-fde` (method `fde`, modules `lightsmith.backends.fde` and `lightsmith.backends._fde`).
A full-vector finite-difference eigenmode solver for [waveguide cross-sections](../problems/mode.md),
straight or bent. The EME and FDFD solvers reuse it for port modes.

## Method

- **Formulation.** The transverse E field on a 2D Yee grid (Zhu & Brown 2002). It solves (QP)E = n²E.
- **Interfaces.** Subpixel permittivity averaging: cells cut by an interface are supersampled. The
  default `anisotropic` scheme takes the harmonic mean along each E component and the arithmetic mean
  across it.
- **Boundaries.** Walls are PEC, or a PML implemented as complex coordinate stretching. Mirror planes can
  be PEC or PMC.
- **Bends** (about y). The exact conformal-map transformation: ε' = ε·diag(h, h, 1/h) and
  μ' = diag(h, h, 1/h), with h = 1 + (x − x_c)/R.
- **Eigensolver.** Shift-invert Arnoldi (SciPy `splu` plus ARPACK) around `target_neff`².
- **Group index.** A perturbation formula that includes material dispersion, or a central finite
  difference in wavelength.
- **Slabs.** `window_size_nm[1] == 0` gives a slab (1D) problem.
- **Normalisation.** Fields are normalised to unit power, ½ Re∫E×H*·z dA = 1.

## Options

`options["fde"]` or `options["lightsmith-fde"]`.

| Option | Default | Meaning |
|---|---|---|
| `grid_spacing_nm` | `None` | Grid step, as one value or (x, y). Default: the smaller of λ_min/(n_max · `points_per_wavelength`) and the smallest feature / `cells_per_feature`. |
| `points_per_wavelength` | `20` | Used for the default spacing |
| `cells_per_feature` | `8` | Used for the default spacing |
| `subpixel` | `"anisotropic"` | `"anisotropic"`, `"volume"` or `"none"` |
| `subsamples` | `32` | Samples per axis in cut cells |
| `boundary` | `"auto"` | `"auto"` means PML for bends and PEC otherwise; `"pec"` or `"pml"` force one |
| `pml_cells` | `12` | PML thickness |
| `pml_reflection` | `1e-6` | Target normal-incidence reflection of the PML |
| `symmetry` | `("none", "none")` | Mirror planes through the window centre, normal to x and to y: `"none"`, `"pec"` or `"pmc"` |
| `polarization` | `"any"` | `"te"` keeps modes with `te_fraction` > 0.5 (dominant Ex); `"tm"` keeps those below |
| `group_index` | `"auto"` | `"perturbation"`, `"finite_difference"` or `"none"`. `"auto"` uses the perturbation formula without a PML and the finite difference with one. |
| `track_modes` | `False` | Order modes at each wavelength by overlap with the previous wavelength |
| `bend_reference_x_nm` | `None` | The x position at which the bend radius is measured. Default: the window centre. |
| `max_cells` | `1000000` | Refuse larger grids |

The solver runs on the CPU. There are no gradients.

!!! note "Slab polarisation naming"
    For a slab normal to x, `te_fraction` is the share of Ex. The physical slab TE mode (Ey) is therefore
    selected with `polarization="tm"`.

## Validation

All tests are in `tests/test_fde.py`.

| Test | Reference | Tolerance |
|---|---|---|
| `test_slab_matches_dispersion_relation` (TE and TM): n = 2.0/1.45, d = 600 nm | analytic slab dispersion relation | < 1e-4 at dx = 2.5 nm; second-order convergence |
| `test_slab_group_index_matches_analytic_derivative` | analytic dispersion derivative | 2e-4 |
| `test_step_index_fibre_he11`: n = 1.50/1.45, a = 1.5 µm | exact vector HE11 eigenvalue equation (Snyder & Love) | 2e-6 |
| `test_si_strip_te0_and_grid_convergence`: 500 × 220 nm Si strip in SiO2, 1550 nm | published neff 2.44523 | < 1e-3 at dx = 10 nm; 3.9 < ng < 4.2 |
| `test_si_strip_fine_grid` (slow): dx = 5 nm | neff 2.44523 | 3e-4 |
| `test_symmetry_matches_full_window_and_group_index_methods_agree` | full window; the other group-index method | 1e-9; 2e-3 |
| `test_bend_loss_decreases_with_radius`: slab bend, R = 2–5 µm | exponential trend in R | strictly decreasing; log-linear fit residual < 0.3 |
| `test_port_modes_match_mode_problem` | `ModeProblem` | 1e-6 |

## Limits

- Bends only about the y axis, and no x-symmetry plane in a bend.
- Material dispersion enters only through the group index; materials are evaluated at each wavelength.
- Raises an error above `max_cells`: increase `grid_spacing_nm`, use `symmetry`, or raise the cap.
- Warnings:
  - fewer modes found than requested;
  - a mode is not guided (neff ≤ the background index);
  - more than 20 % of |E|² is in the PML.
