# Lightsmith agent benchmark: oracle-no_estimate

- Run: 2026-11-08T10:20:21, 400 s, server `lightsmith 0.2.0` (17 tools: list_problem_kinds, get_schema, list_backends, check_environment, validate_problem, estimate, get_discretization, solve, start_solve, compare, get_job, list_jobs, cancel_job, get_result, search_materials, get_material, fit_drude_lorentz)
- lightsmith 0.2.0, commit 203b6df, Windows 11, Python 3.12.2
- **Passed 33/44 (75%)**, mean score 0.935

| kind | tasks | passed | mean score |
|---|---:|---:|---:|
| bands | 4 | 1 | 0.804 |
| circuit | 5 | 5 | 1.000 |
| device | 2 | 0 | 0.708 |
| emission | 1 | 1 | 1.000 |
| layer_stack | 8 | 8 | 1.000 |
| mode | 7 | 7 | 1.000 |
| optimization | 1 | 0 | 0.750 |
| periodic | 4 | 0 | 0.742 |
| scattering | 12 | 11 | 0.983 |

| category | tasks | passed | mean score |
|---|---:|---:|---:|
| pushback | 3 | 3 | 1.000 |
| standard | 23 | 13 | 0.885 |
| trap | 16 | 15 | 0.988 |
| warning | 2 | 2 | 1.000 |

| task | kind | category | pass | score | tool calls (errors) | turns | stop | time s | cost $ | failed checks |
|---|---|---|:-:|---:|---:|---:|---|---:|---:|---|
| sca-au40-water-peak | scattering | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 6.5 |  |  |
| sca-au80-vendor-size-abs | scattering | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| sca-au-medium-wavelength | scattering | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| sca-si-sphere-ev | scattering | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| sca-au-oil-peak | scattering | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 2.5 |  |  |
| sca-coreshell-nir | scattering | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 10.0 |  |  |
| sca-au-nanorod-peak | scattering | standard | ❌ | 0.80 | 4 (0) | 0 | submitted | 110.1 |  | estimate_before_solve |
| sca-quasistatic-too-big | scattering | warning | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| sca-ejwt-absorbing | scattering | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| sca-ejwt-gain-pushback | scattering | pushback | ✅ | 1.00 | 0 (0) | 0 | submitted | 0.0 |  |  |
| sca-au-midir-dataset | scattering | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| sca-ps-bead-wavenumber | scattering | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| em-dye-near-gold-sphere | emission | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| ls-quarter-wave-ar | layer_stack | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 1.8 |  |  |
| ls-brewster-angle | layer_stack | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| ls-bragg-mirror | layer_stack | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.2 |  |  |
| ls-gold-film-glass-side | layer_stack | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| ls-silver-film-ev | layer_stack | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| ls-tir-glass-air | layer_stack | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.3 |  |  |
| ls-si-film-wavenumber | layer_stack | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.1 |  |  |
| ls-thick-slide-incoherent | layer_stack | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.2 |  |  |
| per-grating-first-order | periodic | standard | ❌ | 0.80 | 2 (0) | 0 | submitted | 19.3 |  | estimate_before_solve |
| per-subwavelength-orders | periodic | standard | ❌ | 0.67 | 2 (0) | 0 | submitted | 19.4 |  | estimate_before_solve |
| per-tio2-metasurface | periodic | standard | ❌ | 0.75 | 2 (0) | 0 | submitted | 16.3 |  | estimate_before_solve |
| per-grating-oblique-r0 | periodic | standard | ❌ | 0.75 | 2 (0) | 0 | submitted | 16.9 |  | estimate_before_solve |
| mode-si-strip-te0 | mode | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 8.9 |  |  |
| mode-si-strip-ng | mode | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 11.3 |  |  |
| mode-si-strip-tm0 | mode | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 8.8 |  |  |
| mode-sin-thz | mode | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 8.8 |  |  |
| mode-guided-count | mode | warning | ✅ | 1.00 | 2 (0) | 0 | submitted | 6.7 |  |  |
| mode-neff-above-core-pushback | mode | pushback | ✅ | 1.00 | 0 (0) | 0 | submitted | 0.0 |  |  |
| mode-fiber-fundamental | mode | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 4.7 |  |  |
| dev-mmi-1x2-2d | device | standard | ❌ | 0.75 | 2 (0) | 0 | submitted | 3.6 |  | estimate_before_solve |
| dev-width-step-2d | device | standard | ❌ | 0.67 | 2 (0) | 0 | submitted | 3.6 |  | estimate_before_solve |
| bands-rods-tm-gap | bands | standard | ❌ | 0.75 | 2 (0) | 0 | submitted | 11.8 |  | estimate_before_solve |
| bands-gap-wavelengths | bands | trap | ❌ | 0.80 | 2 (0) | 0 | submitted | 12.0 |  | estimate_before_solve |
| bands-triangular-holes-complete | bands | standard | ❌ | 0.67 | 3 (0) | 0 | submitted | 33.8 |  | estimate_before_solve |
| bands-no-contrast | bands | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 13.9 |  |  |
| circ-mzi-fsr | circuit | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 13.9 |  |  |
| circ-ring-fsr | circuit | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 13.1 |  |  |
| circ-splitter-tree | circuit | standard | ✅ | 1.00 | 2 (0) | 0 | submitted | 1.1 |  |  |
| circ-link-budget-units | circuit | trap | ✅ | 1.00 | 2 (0) | 0 | submitted | 0.2 |  |  |
| circ-lossless-splitter-pushback | circuit | pushback | ✅ | 1.00 | 0 (0) | 0 | submitted | 0.0 |  |  |
| opt-bend-topology | optimization | standard | ❌ | 0.75 | 3 (0) | 0 | submitted | 27.9 |  | estimate_before_solve |
