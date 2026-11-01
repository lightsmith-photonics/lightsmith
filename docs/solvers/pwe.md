# PWE

Backend `lightsmith-pwe` (method `pwe`, modules `lightsmith.backends.pwe` and `lightsmith.backends._pwe`).
Plane-wave expansion for [photonic band structures](../problems/bands.md). It covers 2D TE and TM, and
3D, for lossless, non-dispersive dielectric crystals.

## Method

PWE solves Maxwell's eigenproblem in a transverse plane-wave basis, so div H = 0 holds exactly and no
spurious modes appear. It has two discretisations:

- **`dense`**: the Ho–Chan–Soukoulis inverse rule followed by batched dense eigensolves. The Fourier
  coefficients of ε are analytic for inclusions that do not overlap: circles, ellipses, rectangles,
  polygons, balls, boxes and vertical prisms. Other inclusions use a supersampled raster.
- **`lobpcg`**: matrix-free FFT operators with an MPB-style subpixel-smoothed inverse-ε tensor. The block
  LOBPCG eigensolver uses a preconditioner and warm starts along the k-path.

`auto` picks between them:

- `num_plane_waves` alone selects `dense`; `resolution` alone selects `lobpcg`.
- Otherwise, TM and empty cells use `dense` when the matrix is small (size ≤ 2500), and TE and 3D use
  `lobpcg`.
- The reason: for TE at curved interfaces, `dense` converges slowly, with about 1 % error at 1200 plane
  waves, against about 0.3 % for `lobpcg` at 32 points per a.

Helpers in `lightsmith.backends.pwe`:

- `band_gaps(result)`;
- `density_of_states(problem, ...)`;
- `supercell(problem, repeats, remove, add)`, for line and point defects;
- `bands_from_permittivity(eps, ...)`, which is differentiable with PyTorch for non-degenerate bands.

## Options

`options["pwe"]` or `options["lightsmith-pwe"]`.

| Option | Default | Meaning |
|---|---|---|
| `solver` | `"auto"` | `"dense"`, `"lobpcg"` or `"auto"` |
| `num_plane_waves` | `None` | Dense basis size, completed to a full shell. Default 400 (2D) or 600 (3D) per unit cell. |
| `resolution` | `None` | LOBPCG grid points per a. Default 32 (2D) or 16 (3D). |
| `eps_fourier` | `"auto"` | `"analytic"` or `"raster"`. `"auto"` uses analytic when it is valid and inclusions do not overlap. |
| `subsamples` | `None` | Raster supersampling. Default 4 (2D) or 2 (3D). |
| `reference_wavelength_nm` | `None` | Wavelength at which materials are evaluated. Default 2a, with a warning if a material is dispersive. |
| `precision` | `"auto"` | Double, except single for `dense` on a GPU (about 1e-5 in a/λ). LOBPCG always uses double. |
| `tol` | `None` | LOBPCG relative residual. Default 1e-6 (double) or 1e-4 (single). |
| `max_iter` | `400` | LOBPCG iterations per k-point |
| `extra_bands` | `None` | LOBPCG block padding. Default max(2, num_bands/4). |

## Validation

All tests are in `tests/test_pwe.py`.

| Test | Reference | Tolerance |
|---|---|---|
| `test_empty_lattice_2d_free_photon_bands` (both solvers) | analytic free-photon bands | 1e-6 |
| `test_empty_lattice_3d_fcc_free_photon_bands` (both solvers) | analytic free-photon bands | 1e-6 |
| `test_square_rods_tm_gap_joannopoulos_and_convergence`: alumina rods, ε = 8.9, r = 0.2a | Joannopoulos et al., *Photonic Crystals*, ch. 5: gap 0.3224–0.4425, 31.4 % | gap edges 5e-3 relative; ratio 0.003 |
| `test_square_rods_eps12_matches_mpb_tutorial`: ε = 12 | MPB tutorial: 0.282623–0.419335 | 1 % (converged: 0.28066–0.41717) |
| `test_triangular_air_holes_complete_gap`: ε = 13, r = 0.48a | book: complete gap 0.4297–0.5199, 18.6 % | 1 %; ratio 0.01 |
| `test_lobpcg_matches_dense_for_tm` | dense solver | 3e-3 |
| `test_te_subpixel_smoothing_converges` | resolution 64 | < 3e-3 at resolution 32 |
| `test_supercell_line_defect_mode_in_gap`: W1 waveguide | one guided band in the gap | qualitative |
| `test_density_of_states` | integral = number of bands; zero in the gap | — |
| `test_bands_from_permittivity_is_differentiable` | central differences | 1e-5 |
| `test_gpu_matches_cpu` (gpu) | CPU | 2e-4 (dense, single precision); 1e-6 (LOBPCG) |
| `test_diamond_lattice_complete_gap_3d` (slow): ε = 11.56 spheres | gap 0.2815–0.3144 | 5e-3 |

## Limits

- Positive permittivity only. Metallic crystals need a dispersive solver.
- Loss is ignored and dispersive materials are evaluated at `reference_wavelength_nm`; both trigger
  warnings.
- In 2D, out-of-plane k is refused.
- Analytic Fourier coefficients do not support sloped sidewalls.
- `band_gaps` reports true gaps only if the k-path traces the edge of the irreducible Brillouin zone.
