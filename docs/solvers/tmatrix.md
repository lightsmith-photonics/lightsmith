# T-matrix

Backend `treams` (method `tmatrix`, module `lightsmith.backends.tmatrix_treams`). It computes exact
multiple scattering by clusters of spheres and layered spheres, and supports analytic orientation
averaging.

```bash
uv add "lightsmith[tmatrix]"     # treams (MIT); Python < 3.14, and treams pins scipy < 1.17
```

## Method

- **Single particles.** Each sphere's Mie T-matrix comes from
  [treams](https://github.com/tfp-photonics/treams) (`TMatrix.sphere`), in the parity basis with the
  `exp(-iωt)` convention.
- **Clusters.** Lightsmith couples the particles in a multiple-scattering solve in a local basis, using
  its own translation and rotation coupling matrix. The solve is rescaled (T = D Y D, with D = √|T0|), so
  high orders stay stable. For a close gold dimer, treams' own `interaction.solve()` loses accuracy from
  l ≈ 16.
- **Cross sections.** C_ext comes from the optical theorem. C_abs is summed per sphere from its exciting
  field, and C_sca = C_ext − C_abs.
- **Orientation averages.** Analytic, from trace formulas of the global T-matrix (Mishchenko 1991).
- **Fields.** The far field uses the asymptotic vector spherical harmonics. The near field comes from
  treams and is NaN inside spheres.

## Options

`options["tmatrix"]` or `options["treams"]`.

| Option | Default | Meaning |
|---|---|---|
| `lmax` | `None` | Multipole order of every particle. Default: Wiscombe's rule at the shortest wavelength, plus extra orders for close neighbours. |
| `extra_orders` | `0` | Orders added to the automatic per-particle `lmax` |
| `lmax_global` | `None` | Order of the cluster's global T-matrix, used for `orientation_average`. Default: Wiscombe's rule on the circumscribing sphere, plus 2. |
| `convergence_check` | `True` | For clusters, re-solve the peak-extinction wavelength at `lmax + 2` and warn if the result moves |
| `convergence_tol` | `1e-3` | Relative tolerance for that check |
| `max_modes` | `8000` | Refuse clusters with more modes than this; memory grows as modes² |

For a gap g smaller than the radius R, the automatic order adds min(25, ⌈4.5 (g/R)^−0.63 − 4.5⌉). This
rule was calibrated on a resonant gold-like dimer to about 1e-4 in C_ext.

The solver runs on the CPU in float64. There are no gradients.

## Validation

All tests are in `tests/test_tmatrix_treams.py`, and are skipped when treams is not installed.

| Test | Reference | Tolerance |
|---|---|---|
| `test_single_sphere_matches_mie`: off-origin r = 40 nm sphere, oblique incidence, dielectric and Drude | miepython | < 1e-6 (measured about 1e-12) |
| `test_layered_sphere_with_core_material_shell_equals_sphere` | 40 nm sphere | < 1e-6 |
| `test_small_core_shell_matches_quasistatic_polarizability` | Bohren & Huffman eq. 5.36 | 1e-2 |
| `test_distant_pair_doubles_single_sphere`: spheres 800 nm apart | 2 × single sphere | C_ext, C_abs < 1e-2; C_sca < 5e-2 (measured 3–4 %, from coherent interference) |
| `test_cluster_matches_treams_reference_path`: three-sphere cluster | treams `interaction.solve()` and `TMatrix.xs` | T-matrix < 1e-10; cross sections 1e-9 |
| `test_gold_dimer_redshifts_for_axial_polarization`: R = 20 nm, 5 nm gap | qualitative, plus convergence | axial peak ≥ 20 nm red of the monomer; lmax + 2 change < 1e-3 |
| `test_orientation_average_of_dimer_matches_quadrature` | 10 × 20 × 2-polarisation quadrature | 1e-6 |
| `test_far_field_integrates_to_c_sca` (sphere and dimer) | ∫ dC/dΩ = C_sca | 1e-6 |
| `test_tetrahedral_cluster_converges_and_conserves` (slow): four spheres, ε = 16 + 0.5i | far-field integral; `extra_orders=4` | 1e-5; change < 1e-3 |

## Limits

- Spheres and layered spheres only, and they must not overlap. Use [DDA](dda.md) for anything else.
- No substrate.
- Field outputs cannot be combined with `orientation_average`.
- Raises an error above `max_modes`.
- Warns "not converged in multipole order" when the `lmax + 2` check exceeds `convergence_tol`.
