# BPM

Backend `lightsmith-bpm` (method `bpm`, modules `lightsmith.backends.bpm` and `lightsmith.backends._bpm`). A
finite-difference beam propagation method for [devices](../problems/device.md) whose ports lie on two
opposite planes, in 2D and 3D. It suits long, slowly varying devices such as tapers, bends and branches,
where light travels mostly along one axis. It runs on the CPU or a CUDA GPU.

BPM is **unidirectional**: it neglects reflections. Use [EME](eme.md) (bidirectional) or [FDFD](fdfd.md)
when reflections or abrupt high-contrast junctions matter.

## Method

- **Propagator.** The transverse magnetic field is marched from one port plane to the other with the
  one-way equation dH_t/dw = i k₀ √N H_t. N = PQ is built from the [FDE](fde.md) engine's own Yee-grid
  operators, so FDE and EME port modes are exact eigenvectors of the propagator in uniform sections, and
  mode conventions match EME.
- **Wide angle.** The square root is replaced by a Padé (m, m) approximant (Hadley 1992); m = 0 is the
  paraxial Fresnel equation. Steps are Crank–Nicolson, factored into m shifted solves. The reference index
  adapts to the field in every slice (its Rayleigh quotient).
- **Slices.** The device is cut into piecewise-uniform slices with the EME slicer. Continuously varying
  segments are staircased so that outlines move about `taper_grid_cells_per_slice` grid cells per slice.
  Each slice's operator is factorised once.
- **Why H.** H is continuous across every interface, so it is carried across slice boundaries. On top of
  that, each input's power Re⟨E, H*⟩ is rescaled to be continuous at slice boundaries (energy-conserving
  BPM). Plain H continuity conserves an n-weighted norm instead, so a taper's transmission would be off by
  about √(n_out / n_in).
- **Dimensions.**
  - `dimension=2`: slab BPM in the xy plane, `te` (in-plane E, the default, as in EME) or `tm`.
  - `dimension=3`: full-vectorial (Hx and Hy coupled, the default) or semi-vectorial (quasi-TE: Hy;
    quasi-TM: Hx) on the 2D cross-section (Huang & Xu 1993).
- **Boundaries.** Stretched-coordinate PMLs on the transverse walls.
- **Bends.** `bend_radius_nm` propagates along a circular arc through a conformal transformation, as for
  FDE bend modes. The device is given unrolled.
- **S-parameters.** One run from the lower port plane fills S(upper ← lower), and a second run from the
  upper plane fills S(lower ← upper). Entries between ports on the same plane, including reflections, are
  set to 0.
- **Linear algebra.** Block-tridiagonal parallel cyclic reduction in the array API: lines run along the
  longer transverse axis, with dense blocks across the shorter one. The same code runs on NumPy and
  PyTorch.

## Options

`options["bpm"]` or `options["lightsmith-bpm"]`.

| Option | Default | Meaning |
|---|---|---|
| `vector` | `"auto"` | 3D only: `"full"` (Hx, Hy) or `"semi"` (one H component). `"auto"` means full. |
| `polarization` | `"auto"` | 2D: `"te"` (default) or `"tm"`. 3D semi-vectorial: `"te"` (dominant Ex, default) or `"tm"`. 3D full-vectorial: filters port modes by TE fraction; `"any"` by default. |
| `grid_spacing_nm` | `None` | Cross-section grid step, a number or a pair. Default from the two options below. |
| `points_per_wavelength` | `None` | Default 40 in 2D and 15 in 3D |
| `cells_per_feature` | `6` | Used for the default spacing |
| `subpixel` | `"anisotropic"` | Permittivity averaging, as in [FDE](fde.md): `"anisotropic"`, `"volume"` or `"none"` |
| `subsamples` | `32` | Supersampling in cut cells |
| `pml_cells` | `15` | PML thickness |
| `pml_reflection` | `1e-8` | Target PML reflection |
| `step_nm` | `None` | Largest propagation step. Default λ / 10. |
| `pade_order` | `2` | 0 = paraxial; m = Padé (m, m), costing m solves per step (0–6). (1, 1) suffices for straight guides and gentle tapers; tilted guides, branches and bends need (2, 2). |
| `reference_index` | `"adaptive"` | `"adaptive"` (per slice, from the field) or a fixed number |
| `taper_grid_cells_per_slice` | `1.0` | How far outlines may move between slices, in grid cells |
| `max_taper_slices` | `400` | Most slices per varying segment |
| `power_continuity` | `True` | Keep each input's power continuous at slice boundaries |
| `directions` | `"both"` | `"forward"` runs from the lower plane only and fills the reverse direction by reciprocity |
| `bend_radius_nm` | `None` | Propagate along an arc of this radius; positive means the centre of curvature is towards −u |
| `precision` | `"auto"` | `"auto"` means single on the GPU and double on the CPU |
| `max_grid_cells` | `400000` | Refuse larger cross-section grids |
| `gpu_min_unknowns` | `2000` | Cross-sections with fewer unknowns run on the CPU even when a GPU is selected; `provenance["device_note"]` says so. 0 = always use the given device. |

There are no gradients. Multimode ports are labelled `name@j`, as in EME. `estimate` reports the grid, the
block size, the number of steps and runs, and the memory of the factorisations.

## Validation

All tests are in `tests/test_bpm.py`. Unless stated, devices are 2D with core n = 2.85 and cladding
n = 1.44 at 1550 nm.

| Test | Reference | Tolerance |
|---|---|---|
| `test_operator_blocks_and_solves_match_the_fde_matrices`: slab, 3D, PMC mirror, bend, PEC walls | FDE matrices; dense LU | 1e-12; solves 1e-11 |
| `test_pade_steps_approach_the_exact_one_way_propagator` | exact one-way step | \|G\| = 1 to 1e-12 for lossless media; error falls with the Padé order; lossy media decay |
| `test_straight_slab_is_lossless_with_the_fde_phase`: te and tm | \|S21\| = 1; EME phase | 1e-9; phase 1e-9 rad |
| `test_straight_strip_3d_full_vector_keeps_te_and_tm`: 500 × 220 nm SOI strip | \|S\| = 1, no conversion; k₀·neff·L from FDE | 2e-3; phase 1e-2 rad (measured 4.6e-3) |
| `test_linear_taper_matches_eme`: 500 nm → 3 µm, 3 and 10 µm long | [EME](eme.md), same grid | \|S21\|² within 0.02 (measured ≤ 0.010); phase 0.01 rad |
| `test_long_taper_is_reciprocal_and_adiabatic` (slow): 30 µm, 10 nm grid | EME (0.994); reciprocity | 0.01 (measured 0.996 / 0.995) |
| `test_mmi_1x2_self_imaging_length`: W = 3 µm, 0.9–1.05 × the paraxial length | EME | same best length (0.95 ×); 50/50 to 1e-9; total within 0.06 of EME; > 0.9 |
| `test_slab_bend_loss_matches_the_fde_bend_mode`: R = 20 µm, n 1.60 / 1.45 | FDE bend mode | 2 % (measured 0.2 %) |
| `test_y_branch_splits_evenly`: arms diverging at 4.8° | EME (0.936) | 50/50 to 1e-9; S21 and S12 within 0.02; total within 0.1 of EME (measured 0.856–0.864) |
| `test_torch_cpu_matches_numpy` | NumPy | 1e-10 |
| `test_gpu_matches_cpu_3d_and_small_2d_falls_back_to_cpu` (gpu) | CPU double | 1e-4 (single precision) |
| `test_strip_bend_loss_3d_matches_the_fde_bend_mode` (slow): SiN 800 × 400 nm, R = 8 µm | FDE bend mode (1244 dB/cm) | 2 % (measured 0.4 %) |

The test docstrings record where one-way BPM is weakest:

- **Short converging tapers** (wide → narrow): a 3 µm taper gives 0.68 against EME's 0.77.
- **High-contrast branches**: the Y-branch total of 0.856–0.864 is converged in grid, slicing and step, but
  lower than EME's 0.936.
- **Bend loss and the PML**: with the default `pml_reflection` of 1e-8, instead of FDE's 1e-6, the 3D bend
  loss is 6 % lower, because the loss of a leaky mode depends on the truncation.

## Limits

- Every port must be normal to the same axis and lie on one of exactly two opposite planes, as for
  [EME](eme.md). Each port must point into the device.
- Unidirectional: reflections are neglected, and S entries between ports on the same plane are 0. Every
  result carries a warning that says so.
- At an abrupt junction BPM transmits by mode overlap, an error of the order of the mismatch.
- Raises an error when the cross-section grid exceeds `max_grid_cells`, or when a port supports fewer
  modes than its `num_modes`.
- Warns when Σ|S|² > 1.02 for an input (refine the grid or the step, or raise `pade_order`), and, for
  semi-vectorial 3D runs, that polarisation coupling is neglected.
- `auto` tries EME first, which accepts the same port layouts, so in practice BPM is used when you ask for
  it with `method="bpm"`.
