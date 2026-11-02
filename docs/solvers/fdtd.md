# FDTD

Backend `lightsmith-fdtd` (method `fdtd`, modules `lightsmith.backends.fdtd` and
`lightsmith.backends._fdtd`). A 3D finite-difference time-domain solver for three problem kinds:

- [scattering](../problems/scattering.md);
- [periodic structures](../problems/periodic.md) and [layer stacks](../problems/layer-stack.md), at normal,
  oblique and conical incidence up to 85° from the normal.

It runs on the CPU or a CUDA GPU.

## Method

### Scattering

- **Grid and boundaries.** A Yee grid with a convolutional PML (Roden & Gedney 2000).
- **Source.** A total-field/scattered-field plane-wave source fed by a 1D auxiliary grid, for any
  incidence direction. It has no leakage for axis-aligned incidence. For other directions the 1D feed is
  interpolated, with a leakage of about 1e-3 of the incident field.
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
  and T are sums over the propagating orders, and A = 1 − R − T. Order (m, n) has in-plane wavevector
  k_∥ + m b₁ + n b₂, as in [RCWA](rcwa.md).
- **Normal incidence.** Real fields and plain periodic boundaries. One broadband pulse per polarisation
  gives every wavelength. A layer stack at normal incidence runs once and serves both s and p.
- **Oblique and conical incidence.** Complex fields with Bloch-periodic boundaries,
  F(x + L) = F(x) e^{i k_∥ L}, at a fixed in-plane wavevector k_∥ (the constant-transverse-wavenumber
  method). The TFSF plane is fed by an auxiliary one-cell Bloch column that solves the same discrete
  equations as the main grid, so the source does not leak at any angle or polarisation. This keeps the
  usual Courant limit and works unchanged with the dispersive media, subpixel smoothing and the CPML.
- **Fixed angle, not fixed k_∥.** Within one fixed-k_∥ run, each frequency sees a different angle. So every
  (angle, wavelength, polarisation) gets its own run, with k_∥ = n_sup (2π/λ) sin θ and a pulse centred on
  that wavelength, and every result is at the requested angle. The pulse is kept clear of the frequency
  at which the zeroth order turns grazing, where fields ring for a long time. An oblique spectrum
  therefore costs one run per wavelength; `estimate` reports the count (`runs`, `oblique_runs`).

The design choices are explained in the docstring of `lightsmith.backends._fdtd.periodic`.

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
| `test_fdtd_oblique_tfsf_sphere_is_direction_independent` (gpu): n = 1.5, r = 60 nm, dx = 6 nm, three directions | Mie; the other directions | C_ext, C_sca < 2 % (measured 1.0–1.2 %); spread between directions < 0.6 % (measured 0.35 %) |
| `test_fdtd_oblique_tfsf_spheroid_matches_dda` (gpu): 80 × 160 nm spheroid, axial and off-axis | [DDA](dda.md), 80 dipoles per diameter | C_ext < 2 % (measured 0.5–1.1 %) |
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
| `test_fdtd_oblique_layer_stack_matches_tmm_cpu`: 300 nm n = 2 slab, 0°, 30°, 60°, s and p | TMM | < 0.008 (measured 0.0046, 0.0039, 0.0018) |
| `test_fdtd_oblique_drude_film_gpu_matches_cpu_and_tmm` (gpu): 30 nm Drude film, 30° and 60° | TMM; CPU | < 0.003 (measured 0.0008, 0.0006); GPU vs CPU 5e-5 |
| `test_fdtd_oblique_grating_matches_rcwa` (gpu): lamellar grating at 30°, 60° (slow) and conical 30°/30° | RCWA | total < 0.008 (measured ≤ 0.0043); per order 0.012 (measured ≤ 0.0071) |
| `test_fdtd_oblique_2d_lattice_conical_matches_rcwa` (slow, gpu): disks, P = 400 nm, θ = 40°, φ = 20° | RCWA, 11 harmonics | total < 0.005 (measured 0.0015); per order 0.01 (measured ≤ 0.0039) |
| `test_fdtd_1d_grating_matches_rcwa` (gpu): lamellar grating | [RCWA](rcwa.md), 30 harmonics | total < 0.015; per order 0.01 |
| `test_fdtd_hole_array_guided_mode_resonance_matches_rcwa` (slow, gpu) | RCWA | resonance within 0.5 nm (measured 0.09 nm) |

At oblique incidence the error on the n = 2 slab is the Yee grid's own: FDTD reproduces the discrete-Yee
Fresnel coefficient of a bare interface to 1e-4, and the difference from TMM falls 4× per halving of the
cell (60°, 400 nm: 0.0095 → 0.0024), according to the test module docstring.

## Limits

**Scattering**
- No far-field output, substrate or orientation averaging; near fields are supported.
- Warns about:
  - off-axis incidence (about 0.1 % incident-field leakage);
  - fields that have not decayed by the time cap;
  - negative absorption;
  - non-causal constant lossy materials;
  - fitted materials.

**Periodic structures and layer stacks**
- Angles of incidence up to 85° from the normal. Beyond that, use [RCWA](rcwa.md) or [TMM](tmm.md).
- An oblique spectrum costs one run per (angle, wavelength, polarisation), and runs at large angles use
  longer, narrow-band pulses. For many angles or wavelengths, RCWA or TMM is far cheaper.
- Rectangular lattices only: a1 along x and a2 along y.
- The superstrate must be lossless and non-dispersive.
- No incoherent layers.

**Both**
- Raises an error above `max_cells` (increase `spacing_nm`).
- Raises `FloatingPointError` on divergence: non-finite fields, or monitor energy above 1e8 times its
  peak while the pulse passed (float64 CPU runs stay finite long after diverging). Refine the mesh,
  lower `courant`, or give lossy materials as Drude/DrudeLorentz instead of a constant complex index.
