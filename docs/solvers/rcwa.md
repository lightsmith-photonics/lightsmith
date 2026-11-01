# RCWA

Backend `lightsmith-rcwa` (method `rcwa`, modules `lightsmith.backends.rcwa` and `lightsmith.backends._rcwa`).
Rigorous coupled-wave analysis, also called the Fourier modal method. It solves
[periodic structures](../problems/periodic.md) on 1D and 2D lattices, including non-rectangular lattices
and conical incidence, and also [layer stacks](../problems/layer-stack.md). It is differentiable and runs
on a GPU.

## Method

- **Fourier coefficients of ε and 1/ε.**
  - 1D: exact segment transforms.
  - 2D: closed-form transforms for rectangles, circles, ellipses and polygons when the shapes do not
    overlap. Otherwise a supersampled raster.
- **Factorisation.**
  - 1D: Li's inverse rule.
  - 2D: the normal-vector-field method (Schuster et al. 2007), which metals need. The normal field is a
    Gaussian-blurred structure-tensor field.
  - Optionally, the naive Laurent rule everywhere.
- **Layer eigenmodes.** Only the matrix functions K = √(PQ) and exp(i d K) are needed. Gradients use the
  Daleckii–Krein formula, so they stay exact even for degenerate modes.
- **Cascade.** Layers are combined in a stable Redheffer star-product S-matrix cascade (Rumpf 2011).
- **Outputs.**
  - R, T and A per diffraction order come from the Poynting flux of each harmonic.
  - The zeroth-order complex amplitudes follow the [TMM](tmm.md) conventions.
  - Truncation is rectangular: |m| ≤ M and |n| ≤ N.
- **Gradients.** `lightsmith.backends.rcwa.rcwa_tensors(problem, device, thicknesses=, eps_grids=)`
  differentiates with respect to thicknesses and pixelated permittivity maps.

## Options

`options["rcwa"]` or `options["lightsmith-rcwa"]`.

| Option | Default | Meaning |
|---|---|---|
| `harmonics` | `None` | Highest diffraction order kept along each lattice direction. An integer applies to both directions; a pair sets them separately. Default 25 for 1D gratings and 7 per direction for 2D lattices. |
| `factorization` | `"auto"` | `"auto"`, `"li"` and `"normal_vector"` all mean Li's rule in 1D and the normal-vector method in 2D. `"laurent"` uses the naive rule everywhere. |
| `fourier` | `"auto"` | 2D Fourier coefficients: `"analytic"` shape transforms, used by `"auto"` when shapes do not overlap, or `"raster"` |
| `raster_points` | `256` | Raster points per lattice direction |
| `normal_blur` | `0.1` | Gaussian smoothing of the 2D normal field, as a fraction of the shortest lattice vector |
| `precision` | `"auto"` | `"auto"` means double |
| `max_batch_bytes` | `2e9` | Wavelengths are solved in batches of about this much memory |

## Validation

All tests are in `tests/test_rcwa.py`.

| Test | Reference | Tolerance |
|---|---|---|
| `test_energy_conservation_lossless_1d_conical`: θ = 25°, φ = 35° | R + T = 1 | 1e-12 |
| `test_energy_conservation_lossless_2d_hexagonal_lattice` | A = 0 | 1e-11 with the Laurent rule; < 2e-3 at 6 harmonics with the normal-vector rule, shrinking with more harmonics |
| `test_analytic_and_raster_fourier_agree` | 512-point raster | 2e-3 |
| `test_rytov_effective_medium_benchmark`: ε = 12, fill factor 0.3, P → 0 | TMM on the Rytov effective uniaxial slab | 2e-5 (TE), 5e-5 (TM) |
| `test_li_factorization_converges_for_metallic_grating`: ε = −20 + 1.5i, TM | 80-harmonic result | error falls monotonically; < 4e-3 at 40 harmonics; Laurent more than 5× worse |
| `test_crossed_lattice_of_full_width_ridges_reduces_to_1d_li` | 1D result | 1e-10 |
| `test_guided_mode_resonance_in_hole_array_converges` | convergence in harmonics | peak moves < 1 nm between 3 and 5 harmonics |
| `test_grating_orders_follow_grating_equation` | grating equation | exact set of orders |
| `test_gradients_match_finite_differences`: degenerate C4-symmetric ε grid | central differences | 1e-4 (pixels), 1e-5 (thickness) |
| `test_rcwa_gpu_matches_cpu` (gpu) | CPU | 1e-9 |
| `tests/test_tmm.py::test_tmm_matches_rcwa_for_unpatterned_stack` | TMM | 1e-12 |

The [FDTD](fdtd.md) periodic tests also cross-check against RCWA.

## Limits

- No incoherent layers; use [TMM](tmm.md) for those.
- The normal-vector factorisation does not conserve energy exactly at finite truncation. A lossless
  structure with |A| > 1e-3 triggers a warning: increase `harmonics` or use `precision="double"`.
- Analytic 2D transforms are valid only for shapes that do not overlap. `"auto"` falls back to the
  raster, but forcing `"analytic"` with overlapping shapes is not checked.
- With an absorbing superstrate, efficiencies are normalised to the incident Poynting flux, and the
  result carries a warning saying so.
