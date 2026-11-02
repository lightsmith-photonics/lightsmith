# FDTD roadmap

!!! info "Plan, not a feature list"
    This page lists what the [FDTD](../fdtd.md) solver lacks compared with Meep and commercial FDTD tools,
    and the order in which we are adding it. Work happens on branch `fdtd-advanced`. A feature is part of a
    release only when the [changelog](../../changelog.md) says so.

The goal is the most capable open FDTD solver: everything users expect from Meep and the commercial tools,
plus features none of them combine (GPU, dispersive conformal meshing, non-uniform grids, differentiable
runs, and error bars on every result).

## Where we stand

| Area | Lightsmith FDTD today | Meep | Commercial FDTD (Lumerical, Tidy3D) |
|---|---|---|---|
| Hardware | CPU or one CUDA GPU (CUDA graphs) | CPU, MPI | GPU (local or cloud), multi-GPU |
| Dimensions | 3D | 1D, 2D, 3D, cylindrical | 2D, 3D |
| Grid | Uniform, non-cubic cells | Uniform | Non-uniform, graded, override regions |
| Interfaces | Conformal (tensor) smoothing; dispersive media diagonal | Tensor subpixel smoothing (non-dispersive part) | Conformal meshing for dielectrics, metals and PEC |
| Sources | Plane wave (TFSF; one-sided in periodic cells), Gaussian beam, point dipole | Point, current, Gaussian beam, eigenmode, plane wave | All of these, plus focused beams and imported fields |
| Boundaries | CPML, periodic, Bloch | PML, absorber, periodic, Bloch, mirror symmetry | PML, periodic, Bloch, PEC, PMC, symmetry |
| Materials | Drude, Lorentz, fitted tables | + conductivity, anisotropic, μ, χ2/χ3, gain, gyrotropic | + 2D sheets, index perturbation, multi-coefficient fits |
| Outputs | Flux spectra, diffraction orders, near and far fields, resonances (Q), Purcell factor | + far field, Harminv, LDOS, forces, mode coefficients | + far field into substrates, Q analysis, S-parameters |
| Problem kinds | scattering, periodic, layer_stack, emission | — | + devices with ports |
| Gradients | None | Adjoint (CPU) | Adjoint |

## Conformal meshing

The commercial conformal methods are proprietary. We build the same modes from published methods and
check each against analytic references (Mie for spheres, TMM for films).

| Mode (`mesh_refinement`) | What it does | Method | Status |
|---|---|---|---|
| `staircase` | Each cell takes the material at its centre | — | Done |
| `volume_average` | Arithmetic average of ε over the cell | — | Done |
| `subpixel` | Diagonal anisotropic averaging, dispersive media included | Farjadpour et al., Opt. Lett. 31, 2972 (2006) | Done (the 0.1 default) |
| `conformal` | Full tensor averaging with symmetrised off-diagonal terms between non-dispersive cells; dispersive cells keep the diagonal | Kottke, Farjadpour & Johnson, PRE 77, 036611 (2008) | **Done, new default** |
| `conformal` for metals | Off-diagonal terms in dispersive cells | Tensor terms as one extra input to the two dispersive ADE paths | Tried: diverges for Drude metals (see below) |
| `conformal_metal` | Contour-path updates in metal partial cells | Mohammadi, Nadgaran & Agio, Opt. Express 13, 10367 (2005); Benkler, Chavannes & Kuster, IEEE TAP 54, 1843 (2006) for PEC | Planned (step 3) |

Fill fractions and surface normals come from the exact geometry where the shape provides it, and from
supersampling otherwise.

**Results so far** (r = 60 nm, n = 1.5 sphere, maximum C_ext error over 400–800 nm):

| dx | 6 nm | 4 nm | 3 nm | 2 nm |
|---|---|---|---|---|
| `subpixel` (0.1) | 1.1 % | 0.8 % | 0.6 % | 0.4 % |
| `conformal` | 0.04 % | 0.05 % | 0.07 % | 0.05 % |

Two lessons:

- **Symmetry matters.** Meep's one-sided average (E_c takes the mean of D_d over the four neighbouring
  E_d nodes, weighted by its own coefficient) made a 3 ps run of an n = 3.5 sphere grow without bound.
  Symmetrising the coupling, K(p, q) = (o(p) + o(q)) / 8 for both directions, keeps the discrete 1/ε
  operator symmetric, and the same run decays to 2e-19.
- **Metals need a different method.** On a Drude sphere (r = 40 nm in water, plasmon at 534 nm), off-diagonal
  terms in metal cells diverge even when symmetrised. ⟨ε⟩ of a metal–water cell crosses zero inside the
  band, so each such cell is a resonator. The other averages are no better: peak height errors at
  dx = 2 nm are about −11 % (diagonal, the default), −5 % (staircase, with a 0–4 nm shift),
  −16 % (volume average, 10 nm red shift) and −22 % (harmonic only, 10 nm red shift).
  `conformal_metal` (step 3) is the next attempt.

## Order of work

### Tier 1: what every user expects

1. Conformal meshing (above): tensor averaging, exact normals, refinement modes.
2. Point-dipole and Gaussian-beam sources.
3. Near-to-far-field projection (radiation patterns, far-field scattering).
4. Resonance analysis: frequencies, Q factors and amplitudes by filter diagonalisation (as Harminv).
5. Particles on substrates: a plane-wave source in layered media and a PML that ends inside the substrate.
6. Mirror symmetries (PEC/PMC planes), for 2×–8× smaller runs.
7. Device problems: waveguide mode sources and mode-expansion monitors from the FDE mode solver, giving
   S-parameters that feed [circuits](../../problems/index.md) directly.

### Tier 2: matching the commercial tools

8. Non-uniform, graded grids with refinement regions around small features.
9. Anisotropic ε (diagonal and rotated), μ ≠ 1, and conductivity.
10. 2D runs (TE/TM) and 2D material sheets (surface conductivity, e.g. graphene).
11. Broadband oblique incidence in periodic cells from one run per angle.
12. Hexagonal and oblique lattices.
13. Time-domain field snapshots, LDOS and Purcell factors.
14. Nonlinear media (χ2, χ3) and gain.

### Tier 3: ahead of everyone

15. Adjoint gradients on the GPU through PyTorch, with checkpointing so memory does not grow with run time.
16. Automatic convergence: runs at two or three resolutions and Richardson extrapolation, reported as error
    bars on every spectrum.
17. Higher-order (2,4) stencils for large photonic-circuit volumes.
18. Multi-GPU domain decomposition.

## Progress

Each item gets its options, validation numbers and limits on the main [FDTD](../fdtd.md) page when it is
merged. This table tracks the branch.

| Item | State |
|---|---|
| Roadmap | Written |
| 1. Conformal meshing (dielectrics) | Done: `mesh_refinement`, `conformal` default |
| 1. Conformal meshing (metals) | Open: tensor terms diverge in Drude cells; next try contour-path |
| 3. Far field | Done: `outputs.far_field` for scattering |
| 4. Resonances | Done: `outputs.resonances` for scattering; `lightsmith.resonance.harmonic_inversion` |
| 2. Dipole source | Done: `kind: "emission"` (FDTD and exact Mie); within 0.2 % of Mie for dielectrics, 10 %–2.4× near metals (interface cells) |
| 2. Gaussian-beam source | Done: `GaussianBeam` illumination; 0.13 % from Mie at the focus, 0.1 % from the beam profile off axis |
