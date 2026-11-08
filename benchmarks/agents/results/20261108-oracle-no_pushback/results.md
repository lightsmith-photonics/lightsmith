# Lightsmith agent benchmark: oracle-no_pushback

- Run: 2026-11-08T10:27:05, 885 s, server `lightsmith 0.2.0` (17 tools: list_problem_kinds, get_schema, list_backends, check_environment, validate_problem, estimate, get_discretization, solve, start_solve, compare, get_job, list_jobs, cancel_job, get_result, search_materials, get_material, fit_drude_lorentz)
- lightsmith 0.2.0, commit 203b6df, Windows 11, Python 3.12.2
- **Passed 4/44 (9%)**, mean score 0.496

| kind | tasks | passed | mean score |
|---|---:|---:|---:|
| bands | 4 | 2 | 0.787 |
| circuit | 5 | 0 | 0.400 |
| device | 2 | 0 | 0.583 |
| emission | 1 | 0 | 0.500 |
| layer_stack | 8 | 0 | 0.365 |
| mode | 7 | 0 | 0.381 |
| optimization | 1 | 1 | 1.000 |
| periodic | 4 | 0 | 0.679 |
| scattering | 12 | 1 | 0.476 |

| category | tasks | passed | mean score |
|---|---:|---:|---:|
| pushback | 3 | 0 | 0.000 |
| standard | 23 | 2 | 0.559 |
| trap | 16 | 1 | 0.468 |
| warning | 2 | 1 | 0.750 |

| task | kind | category | pass | score | tool calls (errors) | turns | stop | time s | cost $ | failed checks |
|---|---|---|:-:|---:|---:|---:|---|---:|---:|---|
| sca-au40-water-peak | scattering | standard | ❌ | 0.25 | 3 (0) | 0 | submitted | 7.4 |  | value:peak_wavelength_nm, spec:particles[0].shape.radius_nm, spec:medium_index |
| sca-au80-vendor-size-abs | scattering | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.1 |  | value:c_abs_nm2, spec:particles[0].shape.radius_nm |
| sca-au-medium-wavelength | scattering | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.1 |  | value:c_ext_nm2, wavelength:598.5 |
| sca-si-sphere-ev | scattering | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.1 |  | value:q_sca, spec:particles[0].shape.radius_nm |
| sca-au-oil-peak | scattering | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 3.9 |  | value:peak_wavelength_nm, spec:medium_index |
| sca-coreshell-nir | scattering | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 13.4 |  | value:c_ext_nm2, spec:particles[0].shape.core_radius_nm |
| sca-au-nanorod-peak | scattering | standard | ❌ | 0.80 | 7 (0) | 0 | submitted | 201.0 |  | value:peak_wavelength_nm |
| sca-quasistatic-too-big | scattering | warning | ✅ | 1.00 | 3 (0) | 0 | submitted | 0.2 |  |  |
| sca-ejwt-absorbing | scattering | trap | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.2 |  | value:c_abs_nm2, material:particles[0].material |
| sca-ejwt-gain-pushback | scattering | pushback | ❌ | 0.00 | 3 (0) | 0 | submitted | 0.1 |  | pushback, mentions |
| sca-au-midir-dataset | scattering | trap | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.2 |  | value:c_ext_nm2, wavelength:2500 |
| sca-ps-bead-wavenumber | scattering | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.5 |  | value:q_sca, spec:particles[0].shape.radius_nm |
| em-dye-near-gold-sphere | emission | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.6 |  | value:purcell, value:efficiency, spec:particles[0].shape.radius_nm |
| ls-quarter-wave-ar | layer_stack | standard | ❌ | 0.33 | 3 (0) | 0 | submitted | 2.0 |  | value:reflectance, spec:layers[0].thickness_nm |
| ls-brewster-angle | layer_stack | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.1 |  | value:brewster_angle_deg |
| ls-bragg-mirror | layer_stack | standard | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.1 |  | value:reflectance, spec:layers[0].thickness_nm |
| ls-gold-film-glass-side | layer_stack | trap | ❌ | 0.25 | 3 (0) | 0 | submitted | 0.2 |  | value:reflectance, material:superstrate, material:substrate |
| ls-silver-film-ev | layer_stack | trap | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.3 |  | value:absorptance, wavelength:619.92 |
| ls-tir-glass-air | layer_stack | standard | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.3 |  | value:transmittance, material:superstrate |
| ls-si-film-wavenumber | layer_stack | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.3 |  | value:transmittance, wavelength:2000 |
| ls-thick-slide-incoherent | layer_stack | trap | ❌ | 0.33 | 3 (0) | 0 | submitted | 0.2 |  | value:transmittance, spec:layers[0].coherent |
| per-grating-first-order | periodic | standard | ❌ | 0.80 | 4 (0) | 0 | submitted | 23.6 |  | value:t_plus1 |
| per-subwavelength-orders | periodic | standard | ❌ | 0.67 | 4 (0) | 0 | submitted | 24.7 |  | value:num_transmitted_orders |
| per-tio2-metasurface | periodic | standard | ❌ | 0.75 | 4 (0) | 0 | submitted | 48.6 |  | value:reflectance |
| per-grating-oblique-r0 | periodic | standard | ❌ | 0.50 | 4 (0) | 0 | submitted | 25.7 |  | value:r_zeroth, spec:incidence.theta_deg |
| mode-si-strip-te0 | mode | standard | ❌ | 0.33 | 3 (0) | 0 | submitted | 12.5 |  | value:neff, spec:structures[0].profile.size_nm[0] |
| mode-si-strip-ng | mode | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 15.6 |  | value:ng |
| mode-si-strip-tm0 | mode | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 13.6 |  | value:neff_tm0 |
| mode-sin-thz | mode | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 13.7 |  | value:neff, wavelength:1550.1 |
| mode-guided-count | mode | warning | ❌ | 0.50 | 3 (0) | 0 | submitted | 5.3 |  | value:num_guided_modes |
| mode-neff-above-core-pushback | mode | pushback | ❌ | 0.00 | 0 (0) | 0 | submitted | 0.0 |  | pushback, mentions |
| mode-fiber-fundamental | mode | standard | ❌ | 0.33 | 8 (0) | 0 | submitted | 294.3 |  | value:neff, spec:structures[0].profile.radius_nm |
| dev-mmi-1x2-2d | device | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 4.9 |  | value:t_upper, value:t_lower |
| dev-width-step-2d | device | standard | ❌ | 0.67 | 3 (0) | 0 | submitted | 4.9 |  | value:t_fundamental |
| bands-rods-tm-gap | bands | standard | ❌ | 0.75 | 3 (0) | 0 | submitted | 15.4 |  | value:gap_midgap_ratio |
| bands-gap-wavelengths | bands | trap | ❌ | 0.40 | 3 (0) | 0 | submitted | 16.4 |  | value:lambda_min_nm, value:lambda_max_nm, spec:lattice_vectors_nm[0][0] |
| bands-triangular-holes-complete | bands | standard | ✅ | 1.00 | 4 (0) | 0 | submitted | 26.8 |  |  |
| bands-no-contrast | bands | trap | ✅ | 1.00 | 3 (0) | 0 | submitted | 18.7 |  |  |
| circ-mzi-fsr | circuit | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 20.0 |  | value:fsr_nm |
| circ-ring-fsr | circuit | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 17.8 |  | value:fsr_nm |
| circ-splitter-tree | circuit | standard | ❌ | 0.50 | 3 (0) | 0 | submitted | 1.9 |  | value:output_db |
| circ-link-budget-units | circuit | trap | ❌ | 0.50 | 3 (0) | 0 | submitted | 0.6 |  | value:transmission_db |
| circ-lossless-splitter-pushback | circuit | pushback | ❌ | 0.00 | 0 (0) | 0 | submitted | 0.0 |  | pushback, mentions |
| opt-bend-topology | optimization | standard | ✅ | 1.00 | 4 (0) | 0 | submitted | 30.7 |  |  |
