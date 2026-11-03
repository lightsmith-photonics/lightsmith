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
| Interfaces | Conformal (tensor) smoothing, metals included | Tensor subpixel smoothing (non-dispersive part) | Conformal meshing for dielectrics, metals and PEC |
| Sources | Plane wave (TFSF; one-sided in periodic cells), Gaussian beam, point dipole | Point, current, Gaussian beam, eigenmode, plane wave | All of these, plus focused beams and imported fields |
| Boundaries | CPML, periodic, Bloch | PML, absorber, periodic, Bloch, mirror symmetry | PML, periodic, Bloch, PEC, PMC, symmetry |
| Materials | Drude, Lorentz, fitted tables | + conductivity, anisotropic, μ, χ2/χ3, gain, gyrotropic | + 2D sheets, index perturbation, multi-coefficient fits |
| Outputs | Flux spectra, diffraction orders, near and far fields, resonances (Q), Purcell factor | + far field, Harminv, LDOS, forces, mode coefficients | + far field into substrates, Q analysis, S-parameters |
| Problem kinds | scattering (also on a substrate), periodic, layer_stack, emission | — | + devices with ports |
| Gradients | None | Adjoint (CPU) | Adjoint |

## Conformal meshing

The commercial conformal methods are proprietary. We build the same modes from published methods and
check each against analytic references (Mie for spheres, TMM for films).

| Mode (`mesh_refinement`) | What it does | Method | Status |
|---|---|---|---|
| `staircase` | Each cell takes the material at its centre | — | Done |
| `volume_average` | Arithmetic average of ε over the cell | — | Done |
| `subpixel` | Diagonal anisotropic averaging, dispersive media included | Farjadpour et al., Opt. Lett. 31, 2972 (2006) | Done (the 0.1 default) |
| `conformal` | Full tensor averaging with symmetrised off-diagonal terms between non-dispersive cells | Kottke, Farjadpour & Johnson, PRE 77, 036611 (2008) | **Done, new default** |
| `conformal` for metals | Full dispersive tensor at grid vertices, as a sum of passive congruences over octant triads | Ours (below) | **Done** (part of `conformal`) |
| `conformal` for metals, first try | Off-diagonal terms in dispersive cells, node by node | Tensor terms as one extra input to the two dispersive ADE paths | Dropped: diverges for Drude metals |
| PEC | Partial-cell contour path | Benkler, Chavannes & Kuster, IEEE TAP 54, 1843 (2006) | Not planned until there is a PEC material |

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
- **Metals need the full tensor, built to be passive.** On a Drude sphere (r = 40 nm in water, plasmon at
  534 nm), off-diagonal terms added node by node diverge, even when symmetrised. The diagonal alone is
  stable but wrong: it gives every component part of the tangential average 1/⟨ε⟩, whose pole lies inside
  the band for cells about 20–40 % metal, so each such cell becomes a resonator. The C_ext error then
  ripples across the band and hardly falls with dx. The fix is to apply the tensor at grid vertices, as a
  sum of congruences Sᵀ η S over the octant triads of E nodes around each vertex. Each term is passive, so
  the sum is too, and in a uniform medium it is the plain update. RMS C_ext error over 480–600 nm:

  | dx | 3 nm | 2.5 nm | 2 nm | 1.6 nm | 1.33 nm |
  |---|---|---|---|---|---|
  | staircase | 22 % | 17 % | 15 % | 13 % | — |
  | diagonal (`subpixel`) | 7.5 % | 9.1 % | 6.6 % | 6.1 % | — |
  | vertex tensor (`conformal`) | 3.7 % | 3.5 % | 2.5 % | 2.3 % | 1.6 % |

  Peak-height errors fall from 5–13 % to 1.5–3.6 %. Emission 10 nm from the sphere: the tangential
  Purcell factor was up to 2.4× too high and is now within 11.5 % of Mie. Runs of 1.5 ps decay to 6e-22.
  What remains is first order: a node's diagonal is the mean of its two vertex cubes' tensors, not the
  tensor of its own cube, and 1/⟨ε⟩ is nonlinear in the fill fraction. A dielectric sphere through the
  vertex tensor errs by 1.0 % at dx = 6 nm, against 0.04 % with the node scheme, so dielectric interfaces
  keep that one. Larger vertex cubes (1.5×, 2×) do not remove the term. **Open:** a passive form whose node
  diagonals equal the node-cube tensor.

## Order of work

### Tier 1: what every user expects

1. Conformal meshing (above): tensor averaging, exact normals, refinement modes.
2. Point-dipole and Gaussian-beam sources.
3. Near-to-far-field projection (radiation patterns, far-field scattering).
4. Resonance analysis: frequencies, Q factors and amplitudes by filter diagonalisation (as Harminv).
5. Particles on substrates: a plane-wave source in layered media and a PML that ends inside the substrate.
   Done for non-absorbing substrates at any angle, including total internal reflection. Oblique incidence
   uses tables of the exact discrete background: per frequency a 1D discrete problem along z, then an
   inverse FFT. That is leak-free, unlike the interpolated fixed-angle grid of Capoglu & Smith (2008).
   Next: absorbing substrates (particle absorption and the reflectance change), and the far field into both
   half-spaces. The same tables now feed every plane wave and Gaussian-beam component in a uniform medium.
   They replace the 1D-fed source, which leaked about 1e-3 of the field off-axis and whose box flux did
   not cancel (up to 30 % of C_ext for a 20 nm sphere). Gaussian beams gain near fields.
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
| 1. Conformal meshing (metals) | Done: full tensor at grid vertices (passive by construction); C_ext RMS error 2.5 % against 6.6 % at dx = 2 nm |
| 3. Far field | Done: `outputs.far_field` for scattering |
| 4. Resonances | Done: `outputs.resonances` for scattering; `lightsmith.resonance.harmonic_inversion` |
| 2. Dipole source | Done: `kind: "emission"` (FDTD and exact Mie); within 0.2 % of Mie for dielectrics, 8–12 % 10 nm from a Drude sphere |
| 2. Gaussian-beam source | Done: `GaussianBeam` illumination; 0.13 % from Mie at the focus, 0.1 % from the beam profile off axis |
| 5. Substrates | Done for non-absorbing substrates, any angle incl. total internal reflection: leak-free layered TFSF (auxiliary column at normal incidence, exact discrete background tables at oblique incidence); C_sca within 0.2–0.5 % of a dipole reference above glass (1.7 % for TIR p). Open: absorbing substrates, far field |
