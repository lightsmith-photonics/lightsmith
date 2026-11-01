# TMM

Backend `lightsmith-tmm` (method `tmm`, module `lightsmith.backends.tmm`). The transfer-matrix method for
[layer stacks](../problems/layer-stack.md). It handles coherent and incoherent layers, is exact,
differentiable, and runs on the CPU or a GPU.

## Method

- **Coherent layers.** A stable Airy recursion of r from the substrate upwards. It is equivalent to 2 × 2
  transfer matrices, but every phase factor is exp(i k_z d) with Im k_z ≥ 0, so thick absorbing or
  evanescent layers never overflow. The branch of k_z handles both absorption and total internal
  reflection.
- **Incoherent layers** (`coherent=False`). An intensity cascade over coherent groups (Katsidis & Siapkas
  2002).
- **Amplitude conventions.** Amplitudes follow Byrnes' `tmm` package. For s polarisation, r and t are
  ratios of E_y. For p polarisation, r is the ratio of tangential H, so r_p = −r_s at normal incidence,
  and t is the ratio of the full E field. Amplitudes are omitted when any layer is incoherent.
- **Gradients.** `lightsmith.backends.tmm.tmm_tensors(problem, device, thicknesses=, eps=)` is
  differentiable with respect to thicknesses and permittivities.
- **Thickness optimisation.** `optimize_thicknesses(problem, objective, ...)` runs Adam over layer
  thicknesses. It needs PyTorch.

## Options

`options["tmm"]` or `options["lightsmith-tmm"]`.

| Option | Default | Meaning |
|---|---|---|
| `precision` | `"auto"` | `"auto"` means double, because interference needs the digits |

## Validation

All tests are in `tests/test_tmm.py`.

| Test | Reference | Tolerance |
|---|---|---|
| `test_fresnel_single_interface` (0–89°) | Fresnel equations | amplitudes 1e-13 |
| `test_brewster_angle` | R_p = 0 | < 1e-28 |
| `test_total_internal_reflection` | R = 1, evanescent Fresnel amplitudes | 1e-14, 1e-13 |
| `test_frustrated_tir_and_energy_conservation`: random 12-layer stack | R + T = 1 (lossless); R + T + A = 1 | 1e-12; 1e-14 |
| `test_quarter_wave_ar_coating` | R = 0 at λ₀; single-film Airy formula | < 1e-28; 1e-14 |
| `test_bragg_mirror_stop_band`: 10 pairs | analytic admittance at the centre | 1e-12 |
| `test_incoherent_thick_slab_matches_closed_form`: 1 mm slab | closed form | 1e-10 |
| `test_incoherent_equals_phase_averaged_coherent` | coherent result averaged over thickness | 1e-6 |
| `test_tmm_matches_rcwa_for_unpatterned_stack` | [RCWA](rcwa.md) | R, T, A and amplitudes 1e-12 |
| `test_gradients_match_finite_differences` | central differences | 1e-6 relative |
| `test_optimize_ar_coating_finds_quarter_wave` | λ/(4n) | 0.5 nm |
| `test_tmm_gpu_matches_cpu` (gpu) | CPU | 1e-12 |

## Limits

- Layer stacks only. For patterned layers, use [RCWA](rcwa.md).
- With an absorbing superstrate, R and T are normalised to the incident Poynting flux at the first
  interface, and the result carries a warning saying so.
