# Lightsmith agent benchmark: oracle

- Run: 2026-11-08T10:09:15, 156 s, server `lightsmith 0.2.0` (17 tools: list_problem_kinds, get_schema, list_backends, check_environment, validate_problem, estimate, get_discretization, solve, start_solve, compare, get_job, list_jobs, cancel_job, get_result, search_materials, get_material, fit_drude_lorentz)
- lightsmith 0.2.0, commit 73e1508, Windows 11, Python 3.12.2
- **Passed 44/44 (100%)**, mean score 1.000

| kind | tasks | passed | mean score |
|---|---:|---:|---:|
| bands | 4 | 4 | 1.000 |
| circuit | 5 | 5 | 1.000 |
| device | 2 | 2 | 1.000 |
| emission | 1 | 1 | 1.000 |
| layer_stack | 8 | 8 | 1.000 |
| mode | 7 | 7 | 1.000 |
| optimization | 1 | 1 | 1.000 |
| periodic | 4 | 4 | 1.000 |
| scattering | 12 | 12 | 1.000 |

| category | tasks | passed | mean score |
|---|---:|---:|---:|
| pushback | 3 | 3 | 1.000 |
| standard | 23 | 23 | 1.000 |
| trap | 16 | 16 | 1.000 |
| warning | 2 | 2 | 1.000 |

| task | kind | category | pass | score | tool calls (errors) | turns | stop | time s | cost $ | failed checks |
|---|---|---|:-:|---:|---:|---:|---|---:|---:|---|
| sca-au40-water-peak | scattering | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 1.8 |  |  |
| sca-au80-vendor-size-abs | scattering | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| sca-au-medium-wavelength | scattering | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| sca-si-sphere-ev | scattering | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| sca-au-oil-peak | scattering | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 1.1 |  |  |
| sca-coreshell-nir | scattering | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 5.0 |  |  |
| sca-au-nanorod-peak | scattering | standard | ✅ | 1.00 | 4 (0) | 0 | submitted | 49.9 |  |  |
| sca-quasistatic-too-big | scattering | warning | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| sca-ejwt-absorbing | scattering | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| sca-ejwt-gain-pushback | scattering | pushback | ✅ | 1.00 | 0 (0) | 0 | submitted | 0.0 |  |  |
| sca-au-midir-dataset | scattering | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| sca-ps-bead-wavenumber | scattering | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| em-dye-near-gold-sphere | emission | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| ls-quarter-wave-ar | layer_stack | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.7 |  |  |
| ls-brewster-angle | layer_stack | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| ls-bragg-mirror | layer_stack | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| ls-gold-film-glass-side | layer_stack | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| ls-silver-film-ev | layer_stack | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| ls-tir-glass-air | layer_stack | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| ls-si-film-wavenumber | layer_stack | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| ls-thick-slide-incoherent | layer_stack | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| per-grating-first-order | periodic | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 6.6 |  |  |
| per-subwavelength-orders | periodic | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 5.6 |  |  |
| per-tio2-metasurface | periodic | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 6.6 |  |  |
| per-grating-oblique-r0 | periodic | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 7.6 |  |  |
| mode-si-strip-te0 | mode | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 3.6 |  |  |
| mode-si-strip-ng | mode | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 2.5 |  |  |
| mode-si-strip-tm0 | mode | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 2.5 |  |  |
| mode-sin-thz | mode | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 2.5 |  |  |
| mode-guided-count | mode | warning | ✅ | 1.00 | 3 (0) | 0 | submitted | 2.5 |  |  |
| mode-neff-above-core-pushback | mode | pushback | ✅ | 1.00 | 0 (0) | 0 | submitted | 0.0 |  |  |
| mode-fiber-fundamental | mode | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 1.7 |  |  |
| dev-mmi-1x2-2d | device | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 1.7 |  |  |
| dev-width-step-2d | device | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 1.1 |  |  |
| bands-rods-tm-gap | bands | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 5.6 |  |  |
| bands-gap-wavelengths | bands | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 5.6 |  |  |
| bands-triangular-holes-complete | bands | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 7.6 |  |  |
| bands-no-contrast | bands | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 5.6 |  |  |
| circ-mzi-fsr | circuit | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 5.6 |  |  |
| circ-ring-fsr | circuit | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 5.7 |  |  |
| circ-splitter-tree | circuit | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.5 |  |  |
| circ-link-budget-units | circuit | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| circ-lossless-splitter-pushback | circuit | pushback | ✅ | 1.00 | 0 (0) | 0 | submitted | 0.0 |  |  |
| opt-bend-topology | optimization | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 10.6 |  |  |
