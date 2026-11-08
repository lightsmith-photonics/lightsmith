# Lightsmith agent benchmark: oracle-no_estimate

- Run: 2026-11-07T21:47:14, 115 s, server `lightsmith 0.2.0` (17 tools: list_problem_kinds, get_schema, list_backends, check_environment, validate_problem, estimate, get_discretization, solve, start_solve, compare, get_job, list_jobs, cancel_job, get_result, search_materials, get_material, fit_drude_lorentz)
- lightsmith 0.2.0, commit eeafaf0, Windows 11, Python 3.12.2
- **Passed 32/43 (74%)**, mean score 0.934

| kind | tasks | passed | mean score |
|---|---:|---:|---:|
| bands | 4 | 1 | 0.804 |
| circuit | 5 | 5 | 1.000 |
| device | 2 | 0 | 0.708 |
| layer_stack | 8 | 8 | 1.000 |
| mode | 7 | 7 | 1.000 |
| optimization | 1 | 0 | 0.750 |
| periodic | 4 | 0 | 0.742 |
| scattering | 12 | 11 | 0.983 |

| category | tasks | passed | mean score |
|---|---:|---:|---:|
| pushback | 3 | 3 | 1.000 |
| standard | 23 | 13 | 0.885 |
| trap | 15 | 14 | 0.987 |
| warning | 2 | 2 | 1.000 |

| task | kind | category | pass | score | tool calls (errors) | turns | stop | time s | cost $ | failed checks |
|---|---|---|:-:|---:|---:|---:|---|---:|---:|---|
| sca-au40-water-peak | scattering | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 2.3 |  |  |
| sca-au80-vendor-size-abs | scattering | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| sca-au-medium-wavelength | scattering | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| sca-si-sphere-ev | scattering | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| sca-au-oil-peak | scattering | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.7 |  |  |
| sca-coreshell-nir | scattering | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 4.4 |  |  |
| sca-au-nanorod-peak | scattering | standard | ❌ | 0.80 | 3 (0) | 0 | submitted | 38.7 |  | estimate_before_solve |
| sca-quasistatic-too-big | scattering | warning | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| sca-ejwt-absorbing | scattering | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| sca-ejwt-gain-pushback | scattering | pushback | ✅ | 1.00 | 0 (0) | 0 | submitted | 0.0 |  |  |
| sca-au-midir-dataset | scattering | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| sca-ps-bead-wavenumber | scattering | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| ls-quarter-wave-ar | layer_stack | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.4 |  |  |
| ls-brewster-angle | layer_stack | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| ls-bragg-mirror | layer_stack | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| ls-gold-film-glass-side | layer_stack | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| ls-silver-film-ev | layer_stack | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| ls-tir-glass-air | layer_stack | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| ls-si-film-wavenumber | layer_stack | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| ls-thick-slide-incoherent | layer_stack | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| per-grating-first-order | periodic | standard | ❌ | 0.80 | 2 (0) | 0 | submitted | 4.5 |  | estimate_before_solve |
| per-subwavelength-orders | periodic | standard | ❌ | 0.67 | 2 (0) | 0 | submitted | 4.5 |  | estimate_before_solve |
| per-tio2-metasurface | periodic | standard | ❌ | 0.75 | 2 (0) | 0 | submitted | 4.5 |  | estimate_before_solve |
| per-grating-oblique-r0 | periodic | standard | ❌ | 0.75 | 2 (0) | 0 | submitted | 4.5 |  | estimate_before_solve |
| mode-si-strip-te0 | mode | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 1.7 |  |  |
| mode-si-strip-ng | mode | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 1.7 |  |  |
| mode-si-strip-tm0 | mode | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 1.7 |  |  |
| mode-sin-thz | mode | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 2.5 |  |  |
| mode-guided-count | mode | warning | ✅ | 1.00 | 2 (0) | 0 | submitted | 1.7 |  |  |
| mode-neff-above-core-pushback | mode | pushback | ✅ | 1.00 | 0 (0) | 0 | submitted | 0.0 |  |  |
| mode-fiber-fundamental | mode | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 1.1 |  |  |
| dev-mmi-1x2-2d | device | standard | ❌ | 0.75 | 2 (0) | 0 | submitted | 1.1 |  | estimate_before_solve |
| dev-width-step-2d | device | standard | ❌ | 0.67 | 2 (0) | 0 | submitted | 1.1 |  | estimate_before_solve |
| bands-rods-tm-gap | bands | standard | ❌ | 0.75 | 2 (0) | 0 | submitted | 3.5 |  | estimate_before_solve |
| bands-gap-wavelengths | bands | trap | ❌ | 0.80 | 2 (0) | 0 | submitted | 3.5 |  | estimate_before_solve |
| bands-triangular-holes-complete | bands | standard | ❌ | 0.67 | 2 (0) | 0 | submitted | 6.5 |  | estimate_before_solve |
| bands-no-contrast | bands | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 3.5 |  |  |
| circ-mzi-fsr | circuit | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 4.6 |  |  |
| circ-ring-fsr | circuit | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 3.7 |  |  |
| circ-splitter-tree | circuit | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.3 |  |  |
| circ-link-budget-units | circuit | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| circ-lossless-splitter-pushback | circuit | pushback | ✅ | 1.00 | 0 (0) | 0 | submitted | 0.0 |  |  |
| opt-bend-topology | optimization | standard | ❌ | 0.75 | 2 (0) | 0 | submitted | 6.6 |  | estimate_before_solve |
