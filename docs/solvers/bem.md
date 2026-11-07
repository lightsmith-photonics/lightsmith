# BEM

Backend `lightsmith-bem` (method `bem`, modules `lightsmith.backends.bem` and `lightsmith.backends._bem`). A
boundary element (surface integral equation) solver for [scattering](../problems/scattering.md) by
spheres, spheroids, cylinders, layered spheres and clusters of disjoint particles in a homogeneous medium.
It runs on the CPU or a CUDA GPU.

Only the particle surfaces are meshed, so curved metal boundaries converge smoothly instead of being
staircased by a volume grid. For a gold sphere of radius 40 nm the error against Mie is below 0.1 % with 540 unknowns (see
[Convergence](#convergence)).

## Method

- **Formulation.** PMCHWT surface integral equations (Kern & Martin 2009): equivalent electric and magnetic
  currents J and M on every interface, expanded in RWG functions on flat triangles (Rao, Wilton & Glisson
  1982). The formulation covers several regions, so each interface of a layered sphere carries its own J
  and M.
- **Singular integrals.** For nearby triangle pairs, the 1/R and ∇(1/R) parts of the Green's function are
  integrated analytically over the source triangle (Graglia 1993) and the quadrature of the same terms is
  subtracted. These corrections do not depend on the wavelength and are computed once per mesh.
- **Meshes.** Geodesic spheres, arc-length-mapped spheroids, cylinders with graded rims, concentric
  spheres for layered particles, and any number of disjoint particles. By default each faceted surface is
  rescaled to the true volume of the shape (radially only for cylinders, whose flat ends are exact). This
  removes the leading O(h²) error of the inscribed polyhedron.
- **Linear solve.** Dense LU, or restarted GMRES with Jacobi preconditioning on the dense matrix.
- **Cross sections.**
  - C_ext comes from the optical theorem, −Re(bᴴx), whose error is quadratic in the current error;
  - C_sca is the far-field power flux over a Gauss–Legendre sphere;
  - C_abs = C_ext − C_sca.

  The Poynting flux through the particle surface is reported in `provenance["c_abs_flux_nm2"]` as an
  independent check only: with RWG currents it converges slowly.
- **Fields.** Far fields (dC_sca/dΩ) and near fields, inside and outside the particles. Near the surface
  the static parts of the field integrals are integrated analytically.
- **Orientation average.** Fibonacci-sphere incidence directions, each with two polarisations (the same set
  as [DDA](dda.md)). With the direct solver the matrix is factorised once per wavelength for all of them.

The matrix assembly uses the array API, so the same code runs on NumPy (CPU) and PyTorch (CPU or CUDA).

## Options

`options["bem"]` or `options["lightsmith-bem"]`.

| Option | Default | Meaning |
|---|---|---|
| `mesh_size_nm` | `None` | Target mean triangle edge. Default: the smaller of size / `elements_per_size` and λ_min / (\|n\|_max · `elements_per_wavelength`). |
| `elements_per_size` | `10` | Edges across the smallest particle dimension (for layered spheres, also twice the thinnest shell) |
| `elements_per_wavelength` | `10` | Edges per wavelength in the densest material |
| `volume_correction` | `True` | Rescale each faceted surface to the true volume |
| `quadrature` | `7` | Points of the triangle quadrature rule: `3`, `7` or `12` |
| `near_factor` | `2.0` | Triangle pairs closer than this times the sum of their circumradii get analytic singularity extraction |
| `solver` | `"auto"` | `"direct"` (dense LU), `"gmres"`, or `"auto"`: direct up to `direct_max_unknowns` |
| `direct_max_unknowns` | `20000` | Switch-over point for `"auto"` |
| `tol` | `1e-8` | GMRES relative residual target |
| `restart` | `100` | GMRES restart length |
| `max_iter` | `3000` | GMRES iteration cap |
| `precision` | `"double"` | `"single"` halves memory and is faster on consumer GPUs; conditioning worsens on very fine meshes |
| `max_unknowns` | `60000` | Refuse problems with more unknowns (2 × edges) |
| `orientations` | `32` | Incidence directions (× 2 polarisations) for `orientation_average` |
| `assembly_budget` | `None` | Kernel evaluations per assembly chunk. Default 2e6 on the CPU, 8e6 on a GPU. |

There are no gradients. `estimate` reports the mesh (triangles, edges, unknowns), the solver, the number of
linear solves and the memory, which is dominated by the dense matrix (16 bytes × unknowns² in double
precision).

## Validation

All tests are in `tests/test_bem.py`. Materials are Johnson–Christy gold and silver, in water unless stated.
Errors are relative errors of the cross sections.

| Test | Reference | Tolerance |
|---|---|---|
| `test_dielectric_sphere_matches_mie`: n = 1.5 + 0.02i, r = 40 nm, h = 12 nm | Mie | C_ext, C_sca, C_abs < 0.2 % |
| `test_metal_sphere_matches_mie_on_and_off_resonance`: Au and Ag, r = 40 nm, h = 12 nm | Mie | C_ext, C_abs < 0.3 %; no warnings |
| `test_mesh_convergence_without_volume_correction` | Mie | error falls by at least 30 % from h = 14 to 9 nm, and is < 3 % at 9 nm |
| `test_layered_sphere_matches_mie`: Au core 30 nm + 10 nm n = 1.45 shell | Mie | C_ext, C_abs < 0.5 % |
| `test_far_and_near_field_match_mie`: Au r = 30 nm, h = 8 nm | Mie | dC_sca/dΩ 0.2 %; \|E\|² 3 % away from the surface |
| `test_far_field_integrates_to_c_sca` | own C_sca | 0.1 % |
| `test_orientation_average_of_sphere` | fixed incidence | 0.2 % |
| `test_gmres_matches_direct` | dense LU | 1e-7 |
| `test_spheroid_sphere_limit_matches_mie`: spheroid with equal semi-axes | Mie | C_ext < 0.3 % |
| `test_spheroid_polarization_anisotropy`: Drude 16 × 48 nm rod | physics | longitudinal peak > 80 nm to the red of the transverse one |
| `test_dielectric_shapes_agree_with_dda`: 20 × 40 nm spheroid, 20 × 60 nm cylinder | [DDA](dda.md), 24 dipoles per diameter | C_ext < 3 % |
| `test_dimer_couples_and_is_symmetric`: two Au r = 20 nm spheres | single sphere; mirror image | bonding mode red-shifted and > 2× stronger at a 5 nm gap; 2 × single sphere within 3 % at a 1500 nm gap; mirror symmetric to 1e-6 |
| `test_gpu_matches_cpu` (gpu) | CPU | double 1e-9; single 2e-3 (measured about 2e-4) |
| `test_gold_sphere_fine_mesh_converges_below_1e4` (slow): h = 6 nm, 520–800 nm | Mie | C_ext, C_abs < 3e-4 |
| `test_gold_rod_converges_with_mesh` (slow): 20 × 60 nm Au rod, h = 4, 3, 2 nm | self-convergence | band-integrated C_abs: the error falls with h and is < 3 % at h = 3 nm |
| `test_small_rod_approaches_quasistatic_limit` (slow): 10 × 30 nm Drude rod, h = 1.5 nm | [quasistatic](quasistatic.md) with radiative correction | same peak wavelength; band-integrated C_abs within 3 % (measured 1 %) |

The test module states the measured accuracy behind these tolerances: C_ext and C_abs within about
0.05 % of Mie at a 12 nm mesh on a 40 nm gold or silver sphere, on and off the plasmon resonance.

### Convergence

`benchmarks/bem_convergence.py` solves a gold (Johnson–Christy) sphere of radius 40 nm in water at 520, 600,
700 and 800 nm on a ladder of mesh sizes. The error is the maximum over the four wavelengths of
|C/C_Mie − 1|. Results committed in `benchmarks/results/20261101-bem-convergence/` (RTX 5070):

| `mesh_size_nm` | Unknowns | C_ext error | C_abs error | Without volume correction (C_ext) |
|---|---|---|---|---|
| 16 | 540 | 0.056 % | 0.055 % | 11.7 % |
| 12 | 960 | 0.032 % | 0.033 % | 6.79 % |
| 10 | 1500 | 0.021 % | 0.022 % | 4.42 % |
| 8 | 2160 | 0.015 % | 0.016 % | 3.09 % |
| 6 | 3840 | 0.0081 % | 0.0091 % | 1.76 % |
| 4 | 8640 | 0.0035 % | 0.0040 % | 0.785 % |

Without volume correction the error falls as h². The same run solves the sphere with [DDA](dda.md) up to
64 dipoles per diameter and [FDTD](fdtd.md) down to 2 nm cells; their largest errors over the four
wavelengths are in `summary.md` of that directory. The wall times there were measured on a shared, loaded
machine and are indicative only.

## Limits

- No substrate: BEM uses the homogeneous-space Green's function. Use [DDA](dda.md) or [FDTD](fdtd.md).
- Particles must be disjoint. Overlapping or touching particles are refused, and so are particles so close
  that their volume-corrected meshes, which sit slightly outside the exact shapes, would overlap (about
  0.1 nm for 20 nm spheres at the default mesh). A gap smaller than the mesh size is accepted with a
  warning, because the field in the gap is not resolved: set `mesh_size_nm` below the gap.
- Near and far fields are not available together with `orientation_average`.
- Raises an error above `max_unknowns` (increase `mesh_size_nm`). The dense matrix limits practical
  meshes to some tens of thousands of unknowns.
- Warns when GMRES does not reach `tol`, and when the currents are not finite (try `precision="double"` or
  a finer mesh).
- Thin rods need about 10 edges across their diameter (the default `elements_per_size`). The
  `test_small_rod_approaches_quasistatic_limit` docstring records that at h = 2 nm the peak of a 10 nm
  diameter rod is still 4 nm to the blue.
