# DDA

Backend `lightsmith-dda` (method `dda`, module `lightsmith.backends.dda`). The discrete dipole
approximation handles any shape, any number of particles, and orientation averaging. It runs on the CPU or
the GPU.

## Method

- **Lattice.** Point dipoles sit on a cubic lattice. Where particles overlap, the first one listed wins.
- **Solver.** The block-Toeplitz interaction matrix is applied by FFT on a zero-padded doubled grid
  (Goodman, Draine & Flatau 1991), and the system is solved with BiCGStab. The physics follows Draine &
  Flatau (1994).
- **Polarisability.** The lattice dispersion relation (LDR; Draine & Goodman 1993) by default, or
  Clausius–Mossotti with radiative reaction.
- **Cross sections.** C_ext and C_abs are computed from the dipole moments, and C_sca = C_ext − C_abs.
- **Discretisation.** The default `center` discretisation rescales the lattice so that the dipole volume
  equals the true volume.
- **Extrapolation (optional).** Extrapolates to zero dipole size (Yurkin et al. 2006).
- **Orientation average.** Uses Fibonacci-sphere incidence directions, each with two polarisations.
- **Fields.** Near and far fields come from the finest lattice. Near-field points inside particles are NaN.

The same code runs on NumPy, PyTorch (CPU or CUDA) and CuPy through the array API.

## Options

`options["dda"]` or `options["lightsmith-dda"]`.

| Option | Default | Meaning |
|---|---|---|
| `dipoles_per_diameter` | `None` | Lattice sites across the smallest particle's largest dimension. Default: max(16, enough for \|m\| k d ≤ 0.5). |
| `polarizability` | `"ldr"` | `"ldr"`, or `"cm_rr"` (Clausius–Mossotti + radiative reaction) |
| `discretization` | `"center"` | `"center"` (staircase with volume rescaling), or `"weighted"` (boundary sites mixed by volume fraction; as accurate on dielectrics, but BiCGStab often fails to converge on metals) |
| `precision` | `"auto"` | `"auto"` means double on the CPU and single on the GPU |
| `tol` | `None` | BiCGStab relative residual. Default 1e-8 (double) or 1e-5 (single). |
| `max_iter` | `10000` | BiCGStab iteration cap |
| `max_dipoles` | `5000000` | Refuse problems larger than this |
| `orientations` | `32` | Incidence directions (× 2 polarisations) for `orientation_average` |
| `extrapolate` | `None` | List of `dipoles_per_diameter` values to extrapolate to zero dipole size, for example `[12, 16, 24]`. It improves smooth dielectric ladders; staircase noise makes it unreliable for metals. |

There are no gradients.

## Validation

| Test | Reference | Tolerance |
|---|---|---|
| `tests/test_core.py::test_dda_matches_mie_for_dielectric_sphere`: r = 40 nm, n = 1.5 + 0.02i in water; 12 dipoles per diameter | Mie | C_ext, C_abs < 1 % |
| `test_core.py::test_dda_lossless_has_no_absorption`: n = 1.6 | C_abs = 0 | < 1e-6 C_ext |
| `test_core.py::test_dda_gpu_matches_cpu` (gpu) | CPU | 1e-3 relative |
| `tests/test_dda.py::test_far_field_integrates_to_c_sca` | ∫ dC/dΩ = C_sca | 1e-3 |
| `test_dda.py::test_shell_of_core_material_equals_sphere` | plain sphere | 1e-10 |
| `test_dda.py::test_orientation_average_of_sphere`: 4 orientations | fixed incidence | 1e-3 |
| `test_dda.py::test_near_field_tends_to_incident_and_masks_inside` | \|E\|² → 1 far away; NaN inside | 0.02 absolute |
| `test_dda.py::test_spheroid_polarization_anisotropy`: Drude rod | qualitative: the longitudinal peak lies red of the transverse peak | — |

The test suite has no DDA accuracy test against Mie for metal spheres. DDA converges slowly for metals,
so check convergence by increasing `dipoles_per_diameter`.

## Limits

- No substrate.
- Field outputs cannot be combined with `orientation_average`.
- Raises an error above `max_dipoles` (lower `dipoles_per_diameter` or raise the cap). Also raises an
  error when no dipole falls inside a particle (increase `dipoles_per_diameter`).
- Warns when BiCGStab breaks down or does not converge (try `precision="double"`), and when results are
  extrapolated.
