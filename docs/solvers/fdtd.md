# FDTD

Backend `lightsmith-fdtd` (method `fdtd`, modules `lightsmith.backends.fdtd` and
`lightsmith.backends._fdtd`). A 3D finite-difference time-domain solver for four problem kinds:

- [scattering](../problems/scattering.md);
- [periodic structures](../problems/periodic.md) and [layer stacks](../problems/layer-stack.md), at normal
  incidence;
- [dipole emission](../problems/emission.md): Purcell factor, radiative enhancement, efficiency and radiation
  pattern.

It runs on the CPU or a CUDA GPU.

!!! note "Oblique incidence on periodic structures"
    Another branch is adding oblique incidence for periodic structures. See
    [FDTD oblique incidence](in-development/fdtd-oblique.md).

## Method

### Scattering

- **Grid and boundaries.** A Yee grid with a convolutional PML (Roden & Gedney 2000).
- **Source.** A total-field/scattered-field plane-wave source fed by a 1D auxiliary grid. It has no
  leakage for axis-aligned incidence and about 1e-3 for oblique incidence.
- **Gaussian beams.** A `GaussianBeam` is its angular spectrum at fixed directions: Gauss–Legendre rings in
  the transverse wavevector and uniform azimuths, as many as the TFSF box needs (about 60–120 plane waves).
  Each is one TFSF plane wave with its phase reference at the focus. Their 1D feeds are stacked and stepped
  together, their face corrections merged, and all put the focus equally far from their source cell so the
  components arrive in phase. A beam run costs about 1.6× a plane-wave run. Cross sections are per unit
  intensity at the focus.
- **Dispersion.** Drude and Lorentz auxiliary differential equations.
- **Interfaces.** Conformal meshing by tensor subpixel smoothing (Kottke, Farjadpour & Johnson 2008): each interface
  cell gets 1/ε = P⟨1/ε⟩ + (1 − P)/⟨ε⟩, P = n nᵀ, off-diagonal terms included. The off-diagonal coupling
  between neighbouring cells is symmetrised, which keeps long runs stable. Dispersive cells use the
  diagonal of the tensor, integrated as two parallel ADE paths. See `mesh_refinement` below.
- **Spectra.** One broadband pulse covers every wavelength, and running DFTs give the spectra:
  - C_sca is the scattered flux leaving the TFSF box;
  - C_abs is the net total-field flux into the box;
  - C_ext = C_sca + C_abs.
- **Far field.** The scattered E and H on the outer flux box become equivalent surface currents, which are
  projected to the requested directions (near-to-far-field transformation) to give dC_sca/dΩ.
- **Resonances.** E at probes around each particle is recorded after the pulse has passed, and a
  multi-channel matrix pencil (`lightsmith.resonance.harmonic_inversion`) gives each mode's wavelength,
  Q and amplitude, as Harminv does in Meep.
- **Stopping.** The run stops when the monitor energy has decayed by `decay_by`, or at the time cap.
- **Materials.** A constant lossy permittivity becomes a Drude term matched exactly at the band centre,
  with a warning that it is not causal over a band. Tabulated and library materials are fitted with
  `fit_drude_lorentz`, with a warning that reports the RMS error.
- **GPU.** On PyTorch CUDA, the time loop is replayed as CUDA graphs.

### Dipole emission

- **Source.** A point current along the dipole, spread over the eight E nodes around it by trilinear weights.
  Its time profile is the exact discrete derivative of a Gaussian pulse, so it leaves no net charge: a
  static dipole would keep a self-field ~1/(k dx)³ larger than the radiating field, and its DFT would
  depend on when the run stops.
- **Powers.** The emitted power is −½ Re Σ E·J* dV at the source; the radiated power is the flux through a
  closed box around the dipole and the particles. A second run of the dipole in the host alone gives P0, so
  the discretisation of the point source cancels in the ratios. That run's emitted and radiated powers agree
  to 1e-6 or better (`reference_balance` in the provenance).
- **Precision.** Double by default, even on the GPU: the emitted power is the small real part of the
  self-field, and single precision leaves 1–4 % errors. DFT sampling keeps pulse content above
  ω_c + 6 ω_half (e⁻³⁶) out of the band, for the same reason.
- **Radiation pattern.** The near-to-far-field projection of the fields on the box, normalised by P0.

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
| `padding_nm` | `None` | Gap between the particles (or the stack) and the TFSF box. Default 4 cells, and for scattering at least 3 % of the longest wavelength in the medium. |
| `pml_gap_nm` | `None` | Gap between the flux monitors and the CPML. Default for scattering: 4 cells, at least 3 % of the longest wavelength in the medium. For periodic structures, 3 decay lengths of the slowest evanescent order (0.25–1.5 λ). |
| `courant` | `0.95` | Time step as a fraction of the 3D Courant limit |
| `run_time_fs` | `None` | Fixed simulated time; disables the decay shut-off |
| `decay_by` | `1e-5` | Stop when the monitor energy falls below this fraction of its peak |
| `max_time_fs` | `2000` | Hard cap on simulated time; high-Q resonances need picoseconds |
| `precision` | `"auto"` | `"auto"` means single on the GPU and double on the CPU |
| `subpixel` | `True` | Smooth interfaces; `False` means staircase. Superseded by `mesh_refinement`. |
| `mesh_refinement` | `None` | `staircase`, `volume_average` (⟨ε⟩), `subpixel` (diagonal tensor, the 0.1 behaviour) or `conformal` (full tensor). Default: `conformal`, or `staircase` when `subpixel` is false. |
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
| `test_fdtd_conformal_tensor_smoothing_beats_diagonal` (gpu): r = 60 nm, dx = 6 nm | Mie | conformal < 0.2 %, subpixel > 0.8 % |
| `test_fdtd_far_field_matches_mie` (gpu): r = 60 nm, dx = 4 nm | `lightsmith-mie` | dC_sca/dΩ median < 0.2 %, max < 1 %; integral = C_sca within 0.5 % |
| `test_fdtd_drude_sphere_resonance` (gpu): Drude r = 40 nm in water | Mie extinction peak | dipole within 3 nm; Q between 6 and 10 |
| `test_fdtd_emission_near_dielectric_sphere_matches_mie` (gpu): dipole 10 nm from n = 2, r = 60 nm, tilted 45° | `lightsmith-mie` | Purcell and radiative < 0.5 % (measured 0.07 % radial, 0.17 % tangential) |
| `test_fdtd_lone_dipole_radiation_pattern` (gpu) | 3/(8π) sin²θ | < 0.2 % of the peak (measured 0.05 %); integral 1 within 0.2 % |
| `test_fdtd_gaussian_beam_focus_and_profile` (gpu): r = 40 nm sphere, 1 µm waist, at the focus and 333 nm off axis | Mie × the beam's |E|² | 0.5 % (measured 0.13 % at the focus, 0.1 % off axis) |
| `test_fdtd_conformal_is_stable_at_late_times` (slow, gpu): n = 3.5 sphere, 3 ps | — | field decays below 1e-12 |
| `test_fdtd_dielectric_sphere_fine_grid` (slow, gpu): r = 60 nm, dx = 2 nm | Mie | C_ext < 0.2 % |
| `test_fdtd_drude_sphere_fine_grid` (slow, gpu): dx = 1.33 nm | Mie | peak within one 10 nm sample; height < 7 % |
| `test_fdtd_gold_library_material_via_drude_lorentz_fit` (slow, gpu): Johnson–Christy gold | Mie | peak within 10 nm; height < 20 % |

Measured in the test module docstring (RTX 5070, float32; CPU float64 agrees to about 1e-7):

- Dielectric r = 60 nm sphere, maximum C_ext error over 400–800 nm at dx = 6, 4, 3 and 2 nm:
  - `conformal`: 0.04 %, 0.05 %, 0.07 % and 0.05 %;
  - `subpixel` (diagonal only): 1.1 %, 0.8 %, 0.6 % and 0.4 % (first order).

  With the conformal default the remaining error is the source, monitor and PML floor, not the interfaces.
- Far field of the same sphere at dx = 4 nm: median 0.01–0.03 % from Mie, worst 0.4 % (at 90° in the
  E-plane, where dC_sca/dΩ is near zero).
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
- No substrate or orientation averaging. Near fields, far fields and resonances are supported.
- Gaussian beams: no near-field output yet. Each oblique plane wave leaks ~0.1 % of the incident field,
  which matters only for particles with very small cross sections (1–3 % on a 20 nm glass sphere).
- Resonances of metal particles include weak extra modes (amplitude ≤ 0.15 of the plasmon for a 10 nm
  Drude sphere) from the interface cells, and the plasmon Q comes out about 10 % low (30.6 against 34.2
  from Mie's line width, at 0.5 nm cells).
- Emission near metals: the interface cells over-absorb (below), and a dipole a few cells from a metal
  surface sees it as extra quenching. Dipole 10 nm from a Drude sphere (r = 40 nm in water, dx = 2 nm):
  Purcell factor within 10 % of Mie for a radial dipole, but up to 2.4× too high for a tangential one off
  resonance (6.8 against 2.9 at 600 nm). At a 20 nm gap the excess is about 0.4 (radial within 10 %,
  tangential 0.94 against 0.56). Staircase meshing is 4–5× worse.
- Metal interfaces: tensor smoothing is applied to non-dispersive cells only. Off-diagonal terms in Drude
  cells diverged (the averaged ε crosses zero inside the band). Of staircase, volume average, harmonic and
  diagonal averaging, the diagonal (default) gives the best plasmon peak, still about 10 % low.
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
