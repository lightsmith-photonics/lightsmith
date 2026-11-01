# Quasistatic

Backend `lightsmith-quasistatic` (method `quasistatic`, module `lightsmith.backends.quasistatic`). It
uses the dipole (Rayleigh–Gans) approximation for particles much smaller than the wavelength. `auto`
never picks it; request it with `method="quasistatic"`.

## Method

The polarisability is computed with the relative permittivity ε_p / n_m² (Bohren & Huffman, chapter 5):

- **Sphere:** α = 4πa³(ε − 1)/(ε + 2).
- **Layered sphere:** the recursive effective permittivity of Bohren & Huffman eq. 5.36.
- **Spheroid:** Gans theory with closed-form depolarisation factors.

From the polarisability:

- C_abs = k Im(e·α·e)
- C_sca = k⁴|α·e|²/(6π)

With any correction switched on, C_ext comes from the optical theorem instead. Orientation averages are
the mean over the three principal axes. The far field is that of the induced dipole.

## Options

`options["quasistatic"]` or `options["lightsmith-quasistatic"]`.

| Option | Default | Meaning |
|---|---|---|
| `corrections` | `"none"` | One of: `"none"` (static dipole); `"radiative"` (adds radiative damping); `"mlwa"` (modified long-wavelength approximation: dynamic depolarisation plus radiative damping); `"mie_dipole"` (the Mie dipole term exact to O(x⁴) plus damping, homogeneous spheres only) |

The solver runs on the CPU in float64. There are no gradients.

## Validation

All tests are in `tests/test_quasistatic.py`. They cover 400–800 nm in water, with the native Mie solver
as the reference.

| Test | Reference | Tolerance |
|---|---|---|
| `test_dielectric_5nm_sphere_matches_mie` (every correction mode) | Mie | C_ext, C_sca, C_abs < 1 % |
| `test_metal_5nm_sphere_matches_mie`: Drude sphere | Mie | `mie_dipole` C_ext < 1 %; error ordering mie_dipole < mlwa < none |
| `test_coated_sphere_matches_mie`: 3 nm core with a 2 nm shell | Mie | C_ext < 1 % |
| `test_depolarization_factors` | numerical quadrature | 1e-12 |
| `test_far_field_integrates_to_csca` | ∫ dC/dΩ = C_sca | 1e-12 |
| `test_energy_balance_with_radiative_damping`: lossless spheroid, `mlwa` | C_abs = 0 | < 1e-12 C_ext |

## Limits

- One particle only.
- Shapes: spheres, layered spheres and spheroids; cylinders are refused. `mie_dipole` accepts plain
  spheres only.
- No near field and no substrate. The far field cannot be combined with `orientation_average`.
- Warns when the size parameter x = 2π n_m a_max / λ_min exceeds 0.3. At that size, use [Mie](mie.md) or
  [DDA](dda.md) instead.
