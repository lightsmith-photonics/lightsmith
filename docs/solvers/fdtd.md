# FDTD

Backend `lightsmith-fdtd` (method `fdtd`, modules `lightsmith.backends.fdtd` and
`lightsmith.backends._fdtd`). A 3D finite-difference time-domain solver for three problem kinds:

- [scattering](../problems/scattering.md);
- [periodic structures](../problems/periodic.md) and [layer stacks](../problems/layer-stack.md), at normal
  incidence.

It runs on the CPU or a CUDA GPU.

!!! note "Oblique incidence on periodic structures"
    Another branch is adding oblique incidence for periodic structures. See
    [FDTD oblique incidence](in-development/fdtd-oblique.md).

## Method

### Scattering

- **Grid and boundaries.** A Yee grid with a convolutional PML (Roden & Gedney 2000).
- **Source.** A total-field/scattered-field plane-wave source fed by a 1D auxiliary grid. It has no
  leakage for axis-aligned incidence and about 1e-3 for oblique incidence.
- **Dispersion.** Drude and Lorentz auxiliary differential equations.
- **Interfaces.** Anisotropic subpixel smoothing (Farjadpour et al. 2006), extended to dispersive media.
- **Spectra.** One broadband pulse covers every wavelength, and running DFTs give the spectra:
  - C_sca is the scattered flux leaving the TFSF box;
  - C_abs is the net total-field flux into the box;
  - C_ext = C_sca + C_abs.
- **Stopping.** The run stops when the monitor energy has decayed by `decay_by`, or at the time cap.
- **Materials.** A constant lossy permittivity becomes a Drude term matched exactly at the band centre,
  with a warning that it is not causal over a band. Tabulated and library materials are fitted with
  `fit_drude_lorentz`, with a warning that reports the RMS error.
- **GPU.** On PyTorch CUDA, the time loop is replayed as CUDA graphs.

### Periodic structures and layer stacks

- **Cell and boundaries.** One unit cell, with periodic x and y boundaries and a CPML along z. The cells
  are non-cubic, so each period holds a whole number of cells.
- **Source.** A one-sided plane-wave source in the superstrate.
- **Diffraction orders.** The reflected and transmitted DFT planes are split into orders by a 2D FFT. R
  and T are sums over the propagating orders, and A = 1 − R − T.
- **Layer stacks.** A layer stack at normal incidence runs once and serves both s and p.

## Options

`options["fdtd"]` or `options["lightsmith-fdtd"]`.

| Option | Default | Meaning |
|---|---|---|
| `spacing_nm` | `None` | Yee cell size. Overrides the two options below. |
| `cells_per_wavelength` | `20` | Cells per shortest wavelength in the densest material |
| `cells_per_radius` | `None` | Minimum cells across the smallest particle. Default 10, or 20 if a material is plasmonic (Re ε < 0) in the band. |
| `pml_cells` | `10` | CPML thickness |
| `pml_kappa` | `1.0` | CPML κ_max; values above 1 damp evanescent fields faster |
| `padding_nm` | `None` | Gap between the particles (or the stack) and the TFSF box. Default 4 cells. |
| `pml_gap_nm` | `None` | Gap between the flux monitors and the CPML. Default 4 cells for scattering. For periodic structures, 3 decay lengths of the slowest evanescent order (0.25–1.5 λ). |
| `courant` | `0.95` | Time step as a fraction of the 3D Courant limit |
| `run_time_fs` | `None` | Fixed simulated time; disables the decay shut-off |
| `decay_by` | `1e-5` | Stop when the monitor energy falls below this fraction of its peak |
| `max_time_fs` | `2000` | Hard cap on simulated time; high-Q resonances need picoseconds |
| `precision` | `"auto"` | `"auto"` means single on the GPU and double on the CPU |
| `subpixel` | `True` | Anisotropic subpixel smoothing |
| `subpixel_samples` | `8` | Supersamples per axis for interface cells (2–24) |
| `dft_stride` | `None` | Time steps between DFT samples. Default: aliasing-safe, from the source bandwidth. |
| `cuda_graphs` | `True` | Replay time steps as CUDA graphs |
| `max_cells` | `300000000` | Refuse grids larger than this |

There are no gradients.

## Validation

### Scattering (`tests/test_fdtd.py`)

| Test | Reference | Tolerance |
|---|---|---|
| `test_fdtd_dielectric_sphere_matches_mie_cpu`: n = 1.5, r = 40 nm, dx = 6 nm | Mie | C_ext, C_sca < 3 %; \|C_abs\| < 1e-3 C_ext |
| `test_fdtd_gpu_matches_cpu` (gpu): GPU float32 with CUDA graphs | CPU float64 | 1e-4 |
| `test_fdtd_drude_sphere_matches_mie` (gpu): Drude r = 40 nm in water, dx = 2 nm | Mie | peak within 4 nm; peak height < 15 %; C_sca < 30 % everywhere |
| `test_fdtd_layered_sphere_near_field_and_oblique_incidence` (gpu) | `lightsmith-mie` | C_ext < 3 %; near-field \|E\|² RMS < 6 % |
| `test_fdtd_dielectric_sphere_fine_grid` (slow, gpu): r = 60 nm, dx = 2 nm | Mie | C_ext < 0.6 % (measured 0.40 %) |
| `test_fdtd_drude_sphere_fine_grid` (slow, gpu): dx = 1.33 nm | Mie | peak within one 10 nm sample; height < 7 % |
| `test_fdtd_gold_library_material_via_drude_lorentz_fit` (slow, gpu): Johnson–Christy gold | Mie | peak within 10 nm; height < 20 % |

Measured in the test module docstring (RTX 5070, float32; CPU float64 agrees to about 1e-7):

- Dielectric r = 60 nm sphere: C_ext error 1.1 %, 0.8 %, 0.6 % and 0.4 % at dx = 6, 4, 3 and 2 nm.
- Drude r = 40 nm sphere: peak at 532–536 nm (Mie: 534 nm), with height errors of 5–13 %. The error does
  not fall steadily as the grid is refined: it comes from excess absorption in interface cells.

### Periodic structures (`tests/test_fdtd_periodic.py`)

Differences are maximum absolute differences in R, T and A.

| Test | Reference | Tolerance |
|---|---|---|
| `test_fdtd_layer_stack_matches_tmm_cpu`: 300 nm n = 2 slab; 30 nm Drude film | [TMM](tmm.md) | < 0.01 (slab); < 0.005 (film) |
| `test_fdtd_layer_stack_gpu_matches_cpu_and_metal_film` (gpu): 40 nm Ag film | TMM; CPU | < 0.01; GPU vs CPU 2e-5 |
| `test_fdtd_1d_grating_matches_rcwa` (gpu): lamellar grating | [RCWA](rcwa.md), 30 harmonics | total < 0.015; per order 0.01 |
| `test_fdtd_hole_array_guided_mode_resonance_matches_rcwa` (slow, gpu) | RCWA | resonance within 0.5 nm (measured 0.09 nm) |

## Limits

**Scattering**
- No far-field output, substrate or orientation averaging; near fields are supported.
- Warns about:
  - oblique incidence (about 0.1 % incident-field leakage);
  - fields that have not decayed by the time cap;
  - negative absorption;
  - non-causal constant lossy materials;
  - fitted materials.

**Periodic structures and layer stacks**
- Normal incidence only. For oblique incidence, use [RCWA](rcwa.md) or [TMM](tmm.md).
- Rectangular lattices only: a1 along x and a2 along y.
- The superstrate must be lossless and non-dispersive.
- No incoherent layers.

**Both**
- Raises an error above `max_cells` (increase `spacing_nm`).
- Raises `FloatingPointError` on divergence (lower `courant` or check the material model).
