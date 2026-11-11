# Lightsmith agent benchmark: claude:claude-opus-5-5:high

- Run: 2026-11-09T18:20:54, 2031 s, server `lightsmith 0.3.0` (17 tools: list_problem_kinds, get_schema, list_backends, check_environment, validate_problem, estimate, get_discretization, solve, start_solve, compare, get_job, list_jobs, cancel_job, get_result, search_materials, get_material, fit_drude_lorentz)
- lightsmith 0.3.0, commit d7027ff, Windows 11, Python 3.12.2
- **Passed 33/44 (75%)**, mean score 0.913, API cost $6.36

| kind | tasks | passed | mean score |
|---|---:|---:|---:|
| bands | 4 | 1 | 0.700 |
| circuit | 5 | 5 | 1.000 |
| device | 2 | 0 | 0.708 |
| emission | 1 | 1 | 1.000 |
| layer_stack | 8 | 8 | 1.000 |
| mode | 7 | 7 | 1.000 |
| optimization | 1 | 0 | 0.500 |
| periodic | 4 | 0 | 0.742 |
| scattering | 12 | 11 | 0.958 |

| category | tasks | passed | mean score |
|---|---:|---:|---:|
| pushback | 3 | 2 | 0.833 |
| standard | 23 | 15 | 0.886 |
| trap | 16 | 14 | 0.956 |
| warning | 2 | 2 | 1.000 |

| task | kind | category | pass | score | tool calls (errors) | turns | stop | time s | cost $ | failed checks |
|---|---|---|:-:|---:|---:|---:|---|---:|---:|---|
| sca-au40-water-peak | scattering | standard | ✅ | 1.00 | 2 (0) | 3 | submitted | 14.8 | 0.040 |  |
| sca-au80-vendor-size-abs | scattering | trap | ✅ | 1.00 | 4 (0) | 5 | submitted | 16.5 | 0.154 |  |
| sca-au-medium-wavelength | scattering | trap | ✅ | 1.00 | 3 (0) | 3 | submitted | 14.4 | 0.136 |  |
| sca-si-sphere-ev | scattering | trap | ✅ | 1.00 | 4 (0) | 4 | submitted | 15.7 | 0.162 |  |
| sca-au-oil-peak | scattering | standard | ✅ | 1.00 | 4 (0) | 5 | submitted | 17.6 | 0.152 |  |
| sca-coreshell-nir | scattering | standard | ✅ | 1.00 | 4 (1) | 5 | submitted | 29.4 | 0.152 |  |
| sca-au-nanorod-peak | scattering | standard | ✅ | 1.00 | 10 (1) | 10 | submitted | 610.4 | 0.335 |  |
| sca-quasistatic-too-big | scattering | warning | ✅ | 1.00 | 2 (0) | 3 | submitted | 13.6 | 0.130 |  |
| sca-ejwt-absorbing | scattering | trap | ✅ | 1.00 | 2 (0) | 3 | submitted | 10.1 | 0.125 |  |
| sca-ejwt-gain-pushback | scattering | pushback | ❌ | 0.50 | 2 (0) | 3 | submitted | 20.4 | 0.152 | pushback |
| sca-au-midir-dataset | scattering | trap | ✅ | 1.00 | 6 (0) | 4 | submitted | 22.5 | 0.219 |  |
| sca-ps-bead-wavenumber | scattering | trap | ✅ | 1.00 | 3 (0) | 3 | submitted | 15.5 | 0.136 |  |
| em-dye-near-gold-sphere | emission | trap | ✅ | 1.00 | 3 (0) | 4 | submitted | 18.8 | 0.128 |  |
| ls-quarter-wave-ar | layer_stack | standard | ✅ | 1.00 | 2 (0) | 3 | submitted | 13.4 | 0.103 |  |
| ls-brewster-angle | layer_stack | standard | ✅ | 1.00 | 3 (0) | 4 | submitted | 22.4 | 0.137 |  |
| ls-bragg-mirror | layer_stack | standard | ✅ | 1.00 | 2 (0) | 3 | submitted | 19.5 | 0.122 |  |
| ls-gold-film-glass-side | layer_stack | trap | ✅ | 1.00 | 2 (0) | 3 | submitted | 12.2 | 0.099 |  |
| ls-silver-film-ev | layer_stack | trap | ✅ | 1.00 | 4 (0) | 3 | submitted | 21.1 | 0.121 |  |
| ls-tir-glass-air | layer_stack | standard | ✅ | 1.00 | 2 (0) | 3 | submitted | 11.0 | 0.096 |  |
| ls-si-film-wavenumber | layer_stack | trap | ✅ | 1.00 | 2 (0) | 3 | submitted | 16.4 | 0.108 |  |
| ls-thick-slide-incoherent | layer_stack | trap | ✅ | 1.00 | 2 (0) | 3 | submitted | 15.3 | 0.106 |  |
| per-grating-first-order | periodic | standard | ❌ | 0.80 | 3 (0) | 4 | submitted | 26.6 | 0.142 | estimate_before_solve |
| per-subwavelength-orders | periodic | standard | ❌ | 0.67 | 2 (0) | 3 | submitted | 21.9 | 0.124 | estimate_before_solve |
| per-tio2-metasurface | periodic | standard | ❌ | 0.75 | 4 (1) | 5 | submitted | 41.4 | 0.150 | estimate_before_solve |
| per-grating-oblique-r0 | periodic | standard | ❌ | 0.75 | 3 (0) | 4 | submitted | 26.9 | 0.134 | estimate_before_solve |
| mode-si-strip-te0 | mode | standard | ✅ | 1.00 | 4 (0) | 5 | submitted | 92.8 | 0.129 |  |
| mode-si-strip-ng | mode | trap | ✅ | 1.00 | 4 (0) | 5 | submitted | 148.8 | 0.145 |  |
| mode-si-strip-tm0 | mode | standard | ✅ | 1.00 | 4 (0) | 5 | submitted | 122.8 | 0.136 |  |
| mode-sin-thz | mode | trap | ✅ | 1.00 | 3 (0) | 4 | submitted | 54.5 | 0.121 |  |
| mode-guided-count | mode | warning | ✅ | 1.00 | 3 (0) | 4 | submitted | 41.1 | 0.127 |  |
| mode-neff-above-core-pushback | mode | pushback | ✅ | 1.00 | 0 (0) | 1 | submitted | 7.3 | 0.059 |  |
| mode-fiber-fundamental | mode | standard | ✅ | 1.00 | 3 (0) | 4 | submitted | 41.9 | 0.128 |  |
| dev-mmi-1x2-2d | device | standard | ❌ | 0.75 | 4 (0) | 4 | submitted | 38.8 | 0.229 | estimate_before_solve |
| dev-width-step-2d | device | standard | ❌ | 0.67 | 5 (0) | 6 | submitted | 47.2 | 0.252 | estimate_before_solve |
| bands-rods-tm-gap | bands | standard | ❌ | 0.50 | 2 (0) | 3 | submitted | 27.1 | 0.137 | estimate_before_solve, spec:structures_2d[0].profile.radius_nm |
| bands-gap-wavelengths | bands | trap | ❌ | 0.80 | 2 (0) | 3 | submitted | 23.4 | 0.134 | estimate_before_solve |
| bands-triangular-holes-complete | bands | standard | ✅ | 1.00 | 5 (0) | 6 | submitted | 64.0 | 0.206 |  |
| bands-no-contrast | bands | trap | ❌ | 0.50 | 0 (0) | 1 | submitted | 7.9 | 0.058 | solved_kind |
| circ-mzi-fsr | circuit | standard | ✅ | 1.00 | 3 (0) | 4 | submitted | 30.1 | 0.278 |  |
| circ-ring-fsr | circuit | standard | ✅ | 1.00 | 3 (0) | 4 | submitted | 28.2 | 0.274 |  |
| circ-splitter-tree | circuit | standard | ✅ | 1.00 | 2 (0) | 3 | submitted | 16.0 | 0.091 |  |
| circ-link-budget-units | circuit | trap | ✅ | 1.00 | 3 (1) | 4 | submitted | 15.4 | 0.095 |  |
| circ-lossless-splitter-pushback | circuit | pushback | ✅ | 1.00 | 0 (0) | 1 | submitted | 12.1 | 0.069 |  |
| opt-bend-topology | optimization | standard | ❌ | 0.50 | 4 (0) | 5 | submitted | 44.3 | 0.231 | estimate_before_solve, self_consistent:transmission |
