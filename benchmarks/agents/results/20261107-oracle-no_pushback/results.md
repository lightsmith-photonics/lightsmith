# Lightsmith agent benchmark: oracle-no_pushback

- Run: 2026-11-07T21:49:10, 182 s, server `lightsmith 0.2.0` (17 tools: list_problem_kinds, get_schema, list_backends, check_environment, validate_problem, estimate, get_discretization, solve, start_solve, compare, get_job, list_jobs, cancel_job, get_result, search_materials, get_material, fit_drude_lorentz)
- lightsmith 0.2.0, commit eeafaf0, Windows 11, Python 3.12.2
- **Passed 4/43 (9%)**, mean score 0.496

| kind | tasks | passed | mean score |
|---|---:|---:|---:|
| bands | 4 | 2 | 0.787 |
| circuit | 5 | 0 | 0.400 |
| device | 2 | 0 | 0.583 |
| layer_stack | 8 | 0 | 0.365 |
| mode | 7 | 0 | 0.381 |
| optimization | 1 | 1 | 1.000 |
| periodic | 4 | 0 | 0.679 |
| scattering | 12 | 1 | 0.476 |

| category | tasks | passed | mean score |
|---|---:|---:|---:|
| pushback | 3 | 0 | 0.000 |
| standard | 23 | 2 | 0.559 |
| trap | 15 | 1 | 0.466 |
| warning | 2 | 1 | 0.750 |

| task | kind | category | pass | score | tool calls (errors) | turns | stop | time s | cost $ | failed checks |
|---|---|---|:-:|---:|---:|---:|---|---:|---:|---|
| sca-au40-water-peak | scattering | standard | ❌ | 0.25 | 3 (0) | 0 | submitted | 1.6 |  | value:peak_wavelength_nm, spec:particles[0].shape.radius_nm, spec:medium_index |
| sca-au80-vendor-size-abs | scattering | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.1 |  | value:c_abs_nm2, spec:particles[0].shape.radius_nm |
| sca-au-medium-wavelength | scattering | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.1 |  | value:c_ext_nm2, wavelength:598.5 |
| sca-si-sphere-ev | scattering | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.1 |  | value:q_sca, spec:particles[0].shape.radius_nm |
| sca-au-oil-peak | scattering | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.7 |  | value:peak_wavelength_nm, spec:medium_index |
| sca-coreshell-nir | scattering | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 4.1 |  | value:c_ext_nm2, spec:particles[0].shape.core_radius_nm |
| sca-au-nanorod-peak | scattering | standard | ❌ | 0.80 | 5 (0) | 0 | submitted | 90.7 |  | value:peak_wavelength_nm |
| sca-quasistatic-too-big | scattering | warning | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.1 |  |  |
| sca-ejwt-absorbing | scattering | trap | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.1 |  | value:c_abs_nm2, material:particles[0].material |
| sca-ejwt-gain-pushback | scattering | pushback | ❌ | 0.00 | 3 (0) | 0 | submitted | 0.1 |  | pushback, mentions |
| sca-au-midir-dataset | scattering | trap | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.1 |  | value:c_ext_nm2, wavelength:2500 |
| sca-ps-bead-wavenumber | scattering | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.1 |  | value:q_sca, spec:particles[0].shape.radius_nm |
| ls-quarter-wave-ar | layer_stack | standard | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.5 |  | value:reflectance, spec:layers[0].thickness_nm |
| ls-brewster-angle | layer_stack | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.1 |  | value:brewster_angle_deg |
| ls-bragg-mirror | layer_stack | standard | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.1 |  | value:reflectance, spec:layers[0].thickness_nm |
| ls-gold-film-glass-side | layer_stack | trap | ❌ | 0.25 | 3 (0) | 0 | submitted | 0.1 |  | value:reflectance, material:superstrate, material:substrate |
| ls-silver-film-ev | layer_stack | trap | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.1 |  | value:absorptance, wavelength:619.92 |
| ls-tir-glass-air | layer_stack | standard | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.1 |  | value:transmittance, material:superstrate |
| ls-si-film-wavenumber | layer_stack | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.1 |  | value:transmittance, wavelength:2000 |
| ls-thick-slide-incoherent | layer_stack | trap | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.1 |  | value:transmittance, spec:layers[0].coherent |
| per-grating-first-order | periodic | standard | ❌ | 0.80 | 3 (0) | 0 | submitted | 4.5 |  | value:t_plus1 |
| per-subwavelength-orders | periodic | standard | ❌ | 0.67 | 3 (0) | 0 | submitted | 4.6 |  | value:num_transmitted_orders |
| per-tio2-metasurface | periodic | standard | ❌ | 0.75 | 3 (0) | 0 | submitted | 5.6 |  | value:reflectance |
| per-grating-oblique-r0 | periodic | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 4.5 |  | value:r_zeroth, spec:incidence.theta_deg |
| mode-si-strip-te0 | mode | standard | ❌ | 0.33 | 3 (0) | 0 | submitted | 1.7 |  | value:neff, spec:structures[0].profile.size_nm[0] |
| mode-si-strip-ng | mode | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 1.7 |  | value:ng |
| mode-si-strip-tm0 | mode | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 1.7 |  | value:neff_tm0 |
| mode-sin-thz | mode | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 1.7 |  | value:neff, wavelength:1550.1 |
| mode-guided-count | mode | warning | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.7 |  | value:num_guided_modes |
| mode-neff-above-core-pushback | mode | pushback | ❌ | 0.00 | 0 (0) | 0 | submitted | 0.0 |  | pushback, mentions |
| mode-fiber-fundamental | mode | standard | ❌ | 0.33 | 3 (0) | 0 | submitted | 15.7 |  | value:neff, spec:structures[0].profile.radius_nm |
| dev-mmi-1x2-2d | device | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 1.1 |  | value:t_upper, value:t_lower |
| dev-width-step-2d | device | standard | ❌ | 0.67 | 3 (0) | 0 | submitted | 0.7 |  | value:t_fundamental |
| bands-rods-tm-gap | bands | standard | ❌ | 0.75 | 3 (0) | 0 | submitted | 5.6 |  | value:gap_midgap_ratio |
| bands-gap-wavelengths | bands | trap | ❌ | 0.40 | 3 (0) | 0 | submitted | 3.6 |  | value:lambda_min_nm, value:lambda_max_nm, spec:lattice_vectors_nm[0][0] |
| bands-triangular-holes-complete | bands | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 6.6 |  |  |
| bands-no-contrast | bands | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 3.6 |  |  |
| circ-mzi-fsr | circuit | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 4.6 |  | value:fsr_nm |
| circ-ring-fsr | circuit | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 3.6 |  | value:fsr_nm |
| circ-splitter-tree | circuit | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.3 |  | value:output_db |
| circ-link-budget-units | circuit | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.1 |  | value:transmission_db |
| circ-lossless-splitter-pushback | circuit | pushback | ❌ | 0.00 | 0 (0) | 0 | submitted | 0.0 |  | pushback, mentions |
| opt-bend-topology | optimization | standard | ✅ | 1.00 | 3 (0) | 0 | submitted | 7.6 |  |  |
