# Mie

Exact solution for a single homogeneous or layered sphere. It is the reference for every other scattering
method. There are two backends, both under method `mie`.

| Backend | Module | Shapes | Fields | Device | Gradients |
|---|---|---|---|---|---|
| `miepython` | `lightsmith.backends.mie` | Sphere | — | CPU | — |
| `lightsmith-mie` | `lightsmith.backends.mie_native` | Sphere, LayeredSphere | near and far | CPU or GPU (NumPy or PyTorch) | through `mie_cross_sections` |

`lightsmith-mie` also solves [dipole emission](../problems/emission.md) next to one sphere (dipole outside it),
exactly: the decay rates of radial and tangential dipoles from the Mie coefficients (Chew, J. Chem. Phys. 87,
1355 (1987); Kim et al., Surf. Sci. 195, 1 (1988)), cos²-weighted for other orientations. Tests: a lossless
sphere radiates everything, the rates tend to 1 far away, and the quenching by a small lossy sphere tends to
the quasi-static result as (R/d)² (`tests/test_emission.py`). No radiation pattern yet.

## Method

**`miepython`** wraps [miepython](https://github.com/scottprahl/miepython) (MIT), a core dependency.
miepython uses the `n − ik` convention, so the index is conjugated before the call.

**`lightsmith-mie`** is a native Lorenz–Mie series that follows Bohren & Huffman's conventions.

- **Recursions.** It uses the log-derivative recursions of Yang (2003) and Peña & Pal (2009),
  vectorised over wavelengths. D1 comes from a downward recursion. D3 and the inner-layer ratios come
  from upward recursions, so no Riccati–Bessel function is formed for the cross sections.
- **Coefficients.** a_n and b_n are written in a form that keeps Re(a_n) accurate for tiny, weakly
  absorbing spheres.
- **Number of orders.** Wiscombe's rule for homogeneous spheres; Peña & Pal's rule for layered spheres.
- **Far field.** Computed from S1 and S2.
- **Near field.** Matches tangential E and H at every interface. It always runs in double precision.
- **Incidence.** Any direction, and orientation averaging, which is trivial for a sphere.
- **Gradients.** The functional API `lightsmith.backends.mie_native.mie_cross_sections(radii_nm, eps,
  wavelength_nm, medium_index, n_max=None)` is differentiable with PyTorch tensors, on a CPU or a GPU.

## Options

`options["mie"]` or `options["lightsmith-mie"]`. `miepython` has no options.

| Option | Default | Meaning |
|---|---|---|
| `precision` | `"double"` | `"single"` or `"double"` for the series. Single precision is typically about 1e-7 relative, but can degrade to about 1e-3 near zeros of ψ_n(x). Single is not faster on GPUs, because the series is memory-bound. Near fields always use double. |
| `n_max` | `None` | Number of multipole orders. Default: Wiscombe's or Peña & Pal's rule. |
| `wavelength_chunk` | `None` | Wavelengths per vectorised batch. Default: as many as fit in about 1 GiB of work arrays. |

## Validation

All tests are in `tests/test_mie_native.py` unless noted.

| Test | Reference | Tolerance |
|---|---|---|
| `test_matches_miepython`: n = 1.5 + 0.02i and a gold-like Drude sphere; r = 2–1000 nm; 450–900 nm | miepython | C_ext, C_sca < 1e-6 relative; \|ΔC_abs\|/C_ext < 1e-6 |
| `test_hard_cases_against_mpmath`: x = 0.028; x = 42 with \|mx\| = 126; a Drude sphere near ε ≈ 0 | 40-digit mpmath (Bohren & Huffman) | Q_ext, Q_sca to 1e-11 |
| `test_coated_sphere_against_mpmath`: 30/42 nm core-shell, three material pairs | 40-digit mpmath (Bohren & Huffman 8.1) | 1e-11 |
| `test_shell_of_core_material_reproduces_sphere` | 55 nm homogeneous sphere | cross sections and far field < 1e-10; near field 1e-7 |
| `test_far_field_integrates_to_csca`: r = 80 nm and a core-shell, oblique incidence | ∫ dC/dΩ = C_sca | 1e-10 |
| `test_near_field_boundary_conditions` | continuity of tangential E and normal D | < 1e-4 of the field scale |
| `test_autodiff_matches_finite_differences` | central differences | 1e-6 relative |
| `test_gpu_matches_cpu` (gpu) | CPU, double precision | C_ext and far field < 1e-12; near field 1e-10; single precision < 1e-3 |
| `tests/test_core.py::test_mie_rayleigh_limit` (`miepython`): a = 2 nm | Rayleigh formula | C_abs 5e-3 |

## Limits

- A single particle only. For clusters, use [T-matrix](tmatrix.md) or [DDA](dda.md).
- Spheres only: plain spheres for `miepython`, plain or layered for `lightsmith-mie`. Use DDA or FDTD for
  other shapes.
- No substrate.
- `miepython` refuses near and far fields, and ignores the device (it always runs on the CPU).
- `lightsmith-mie` warns about non-finite cross sections. If that happens, use `precision="double"` or a
  larger `n_max`.
