# Changelog

All notable changes to the `lightsmith` library are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/) (while the version is 0.x, minor releases may break the API).

## [Unreleased]

### Added

- **`lightsmith.discretization(problem, method)`**, `lightsmith discretization` and the MCP tool `get_discretization`: what a backend
  will discretise, before any solve, with a common envelope (`layout_kind`, a labelled `summary`, a `payload`, the
  estimate, warnings, and the library's refusal in `error`). FDTD: the Yee grid, TFSF and flux boxes, CPML,
  refinement regions and mirror planes; unit cells of periodic structures and layer stacks; device grids and
  ports. FDFD device grids and ports, inverse-design regions, the DDA dipole lattice, the BEM surface mesh, Mie
  and T-matrix multipole orders, RCWA harmonics, FDE grids, EME and BPM cells, PWE k-paths and plane waves. Each
  calls the backend's own planning code and never solves. `Backend.discretization` lets plugins answer too.
  See `docs/discretization.md`.

- **FDTD graded grids for device problems**: `mesh_override` boxes and `grid: "auto"` (each structure refined
  to `cells_per_wavelength` in its own material) now work for `kind: "device"`, in 2D and 3D. Ports sit in
  uniform cells along their normals; across their windows the cells may be graded, and the port modes come
  from the FDE problem on the same non-uniform cross-section (the FDE grid gained per-cell widths), so they stay
  exact for the grid: a straight guide still transmits |S21| = 1 with the grid's phase (1e-10 rad). A
  500 → 900 nm 2D step graded from 25 to 10 nm is within 2e-5 of a uniform 10 nm run with 63 % of the cells,
  and a 3D wire step graded from 40 to 15 nm within 8e-6 of a uniform 15 nm run with 45 % of the cells.
- **FDTD graded grids** (scattering): `grid: "auto"` refines each particle's bounding box to the spacing its
  material and size need, and `mesh_override: [{center_nm, size_nm, spacing_nm}]` adds refinement boxes. The
  cells grade back to the base spacing at most `grading_ratio` (1.25) per cell. The grid is a tensor product
  with independent per-axis widths. The Yee update uses primary and dual width arrays, so it conserves the
  same discrete energy as the uniform one, on the CPU and on CUDA with CUDA graphs. The CPML, TFSF tables
  (exact discrete background of the graded 1D problem, still leak-free), layered substrates, conformal
  smoothing (metals' vertex tensor included), flux and far-field surfaces, near fields, particle absorption
  and mirror planes all follow the widths. An n = 3.5, r = 20 nm sphere is within 0.11 % of a uniform 2 nm run, whose own error against Mie is
  0.7 %, with 153 k cells instead of 242 k. A Drude sphere of r = 10 nm at 0.5 nm cells needs 488 k cells
  instead of 1.74 M (2.1× faster in double precision). Its spectrum differs from the uniform run by at most
  7e-4 of the peak, against a 13 % discretisation error. Particles on glass or on an absorbing substrate
  agree within 0.34 %, and far fields within 0.1 % of the uniform run's. The coarsest cells must stay fine:
  any change of cell size h reflects about (k h)²/16 of a wave, so `grid: "auto"` uses 60 cells per
  background wavelength (`background_cells_per_wavelength`). A uniform grid gives today's run bit for bit.
  Scattering only, with plane waves along an axis (or graded only along the axes the wave does not vary along).
- **FDTD device problems** (`kind: "device"`, 3D and 2D): S-parameters between waveguide ports. Each port
  mode is an exact eigenmode of the FDTD grid itself: the FDE cross-section problem solved with the
  leapfrog frequency, the discrete propagation constant and each node's discrete material response, then
  interpolated over the band from Chebyshev nodes. So the TFSF mode sources launch one way only, and the
  monitors split the fields into forward and backward modes exactly. A straight silicon wire transmits with
  |S21| = 1 to 1e-6 and the grid's own phase. A 2D width step matches FDFD to 3e-5 (TE) and 3.4e-4 (TM) at
  dl = 10 nm, and a 3D one is converged to 1e-4 between dx = 25 and 15 nm. Device problems default to
  `subpixel` smoothing (port modes need a diagonal permittivity). `polarization` selects TE or TM in 2D.
- **FDTD mirror symmetries** (`symmetry: "auto"`, the default; `"off"` turns them off). A scattering run
  whose particles are mirror-symmetric about their centre, lit by a plane wave that travels in the mirror
  plane and is polarised in it (a PMC plane) or normal to it (a PEC plane), keeps half the grid per plane:
  2× or 4× fewer cells. The conformal tensors, metals' vertex tensor included, are folded across the
  planes, so cross sections, near fields and far fields match the full run to 1e-11 in double precision,
  over substrates and absorbing substrates too. On the GPU, runs of 1–5 M cells are 1.55–2.3× faster (3.7×
  in steady state at 15 M cells). The provenance and `estimate` report the planes used.
- **FDTD conformal meshing.** `mesh_refinement` (`staircase`, `volume_average`, `subpixel`, `conformal`). The
  new default, `conformal`, adds the off-diagonal terms of tensor subpixel smoothing, symmetrised for
  stability. Dielectric sphere C_ext error at dx = 6 nm falls from 1.1 % to 0.04 %. At metal interfaces
  the full dispersive tensor is applied at grid vertices as a sum of passive congruences, so it is stable
  by construction. On a Drude sphere's plasmon band it cuts the C_ext error 2–4× compared with diagonal
  smoothing (2.5 % against 6.6 % RMS at dx = 2 nm), and the error now falls as the grid is refined.
- **FDTD far field** (`outputs.far_field`): near-to-far-field projection of the scattered fields, within
  0.4 % of Mie.
- **Dipole emission** (`kind: "emission"`): Purcell factor, radiative enhancement, efficiency and
  radiation pattern of a point dipole near particles. Exact for one sphere in native Mie; any geometry in
  FDTD (within 0.2 % of Mie next to a dielectric sphere).
- **Gaussian beams** (`illumination: GaussianBeam`, FDTD): a focused beam as a sum of TFSF plane waves at
  fixed angles (the waist scales with the wavelength, as for a lens of fixed NA); cross sections per unit
  focal intensity. Within 0.13 % of Mie at the focus and 0.1 % of the beam profile off axis.
- **FDTD substrates** (`Simulation.substrate`): particles on or above a non-absorbing half-space, lit from
  above at any angle or through the substrate, including total internal reflection. The TFSF box takes
  the background field (incident, reflected and transmitted waves) from an auxiliary column at normal
  incidence. At oblique incidence it comes from tables of the exact discrete background (a 1D discrete
  problem along z per frequency, then an inverse FFT). Either way it does not leak where it cuts the
  interface. C_sca is within 0.2–0.5 % of a dipole reference for a sphere above glass (1.7 % for p
  under total internal reflection, where the dipole model is least exact). Near fields are supported;
  absorbing substrates and far fields are not yet.
- **FDTD absorbing substrates and far fields over substrates.** Particles over metals or other absorbing
  substrates. Over one, `c_abs` is the particles' own absorption, from the grid's exact discrete Poynting
  balance on their nodes; it equals the inflow through a closed box around the particle to 1e-6. `c_sca` is
  the power scattered upward, and `c_ext` the power taken from the reflected wave (the reflectance change).
  `outputs.far_field` now works over any substrate, into both half-spaces, by reciprocity with the grid's
  own layered background. Against a multipole reference, a sphere over an absorbing dielectric is within
  0.1 %, and one over a Drude metal within 0.14 %. Over glass the new quantities reproduce the box fluxes
  to 3e-4. `precision: "auto"` runs double over an absorbing substrate: in float32 a metal's long
  ring-down left rounding noise of up to 1.2 % in C_sca.
- **Leak-free plane waves and Gaussian beams at any angle** (FDTD scattering). Every plane wave, and every
  component of a Gaussian beam, is fed from tables of the grid's exact discrete plane wave (the oblique
  substrate method, with a uniform background), so the TFSF box does not leak at any direction. A sphere's
  C_ext agrees between directions to 4e-5 (0.35 % before). A 20 nm sphere is within 0.1 % of Mie: with the
  1D-fed source, the incident flux through the box did not cancel, giving errors up to 22 % at 800 nm. A beam
  on a 40 nm sphere is within 0.04 % of Mie at the focus and off axis (0.13 % before). Gaussian beams now
  support near-field output.
- **Resonances** (`outputs.resonances`): wavelengths, Q factors and amplitudes from the FDTD ring-down,
  through the new `lightsmith.resonance.harmonic_inversion` (multi-channel matrix pencil).

### Changed

- FDTD scattering stops when the scattered field has decayed, not the total field. The incident pulse
  no longer sets the peak, so weak, sharp resonances are no longer cut off: a 15 nm Drude sphere's
  spectrum rippled by ±10 %.
- FDTD `pml_cells` defaults to 16 with a substrate (10 otherwise).
- FDTD voxelisation evaluates the geometry once per point of the half-cell lattice, where it used to test each
  field position's 26 neighbours separately for every component. The result is the same, and it is 4× faster
  (45 s → 10 s for a 2.5 M-cell 3D MMI).
- FDTD scattering: the default TFSF padding and PML gap are now at least 3 % of the longest wavelength
  in the medium, so refining the grid no longer moves the monitors closer to the particle.
- FDTD: the table-fed plane-wave source updates all face slabs of a kind in one gather and scatter, and
  stops waiting once the pulse has passed (the wrapped edge of the table's inverse FFT had set it to the
  whole table), so small scattering runs are 2.3–2.8× faster with the same results.
- Documentation describes problems through the Python models first; saving and loading them as JSON
  files for the CLI and MCP server is covered in the quick start.

### Fixed

- Inverse design: `estimate` now applies the optimisation-level solver options (`OptimizationProblem.options`),
  merged over the device's key by key as the optimiser merges them (`inverse.backend.device_problem`); it used
  the device's options alone.
- FDTD options: the `pml_cells` description gives the device default (12), and the `mesh_override` description
  no longer names another product.
- FDTD: the time step is reduced where a dispersive (Drude–Lorentz) material would make the update unstable
  on a coarse grid, checked from the update's eigenvalues over the grid's spatial frequencies. The
  Johnson–Christy gold fit diverged at dx ≥ 20 nm, and lossy constants at dx near half a wavelength.
- **3D devices with ports facing y** (FDE ports, EME, BPM): the cross-section's local x axis was global z,
  so `"te"` selected the vertical-E mode and BPM's `bend_radius_nm` bent the guide out of the chip plane.
  Local x is now the in-plane transverse axis for ports facing x or y, as it already was in 2D. Results for
  ports facing x, and for all 2D devices, are unchanged.
- **`compare`**: device, circuit, mode and band results now report differences (they came back empty);
  without a method list the reference is the method `auto` would pick, not the alphabetically first
  backend; relative cross-section differences are floored at 0.1 % of the reference extinction, so a
  lossless particle's near-zero absorption no longer gives huge relative errors. Results that cannot be
  compared are listed under `not_compared`.
- **FDTD periodic and layer-stack transmission** into an absorbing substrate is referred to the interface,
  as in TMM and RCWA. Before, T was taken a few cells inside the substrate (Si at 420 nm read 0.744 instead
  of 0.815) and metal substrates reported T = 0.
- **BEM** checks particle overlap on the volume-corrected meshes the solve uses, so particles about 0.1 nm
  apart no longer produce interpenetrating meshes, and warns when a gap is smaller than the mesh size.
- **`auto`** skips a backend whose estimate exceeds its own size limit (BEM `max_unknowns`, DDA
  `max_dipoles`, T-matrix `max_modes`, ...) and falls through to the next method, instead of picking it and
  failing at solve time.
- **`Tabulated`** materials sort their data by wavelength (energy-ordered files are descending, which
  interpolation silently got wrong) and refuse duplicate wavelengths or mismatched column lengths.


## [0.2.0] - 2026-11-02

### Added

- **BEM** (`lightsmith-bem`): PMCHWT surface integral equations on RWG meshes for spheres, spheroids,
  cylinders, layered spheres and disjoint clusters, with near and far fields, on CPU or GPU. `auto` tries
  it after T-matrix for scattering.
- **BPM** (`lightsmith-bpm`): wide-angle Padé beam propagation for devices, 2D and 3D (semi- and
  full-vectorial), with bends, on CPU or GPU.
- **FDTD oblique and conical incidence** on layer stacks and periodic structures, up to 85° from the
  normal, with Bloch-periodic boundaries at fixed in-plane wavevector. Scattering accepts any incidence
  direction.
- **PDK flow** (`lightsmith.pdk`): technologies as JSON (`generic_soi` from open data), parametric cells
  with GDS output, simulation at the port reference planes, fitted and cached compact models, and
  `pdk:<cell>` instances in circuit netlists.
- **MCP server rewrite** on MCP SDK 2.x: jobs, budgets, resources, prompts, materials tools, and remote
  mode. The `mcp` extra is now `mcp>=2.3,<3`.
- **3D inverse design** on a full-vector 3D FDFD solver (`lightsmith-fdfd` with `dimension=3` devices):
  exact discrete port modes, TF/SF mode sources, batched BiCGStab on CPU or GPU, and adjoint gradients
  with warm starts across iterations.
- **Agent benchmark** (`benchmarks/agents/`): 43 graded tasks for agents using the MCP server.

### Changed

- The documentation moved to <https://docs.lightsmith.dev/>. The old GitHub Pages address redirects there.

### Fixed

- Multilayer Mie: avoid cancellation near zeros of ψₙ, which gave errors up to 12% for coated spheres
  whose interface sits on such a zero.
- RCWA: grazing diffraction orders (kz = 0) inside uniform layers no longer make the layer S-matrix
  singular.
- FDTD: runs whose fields grow without bound now raise, also on CPU in float64, where they used to stay
  finite and return cross sections off by many orders of magnitude.
- Topology optimisation: `OptimizationProblem.options` now merges into the device's options key by key
  within each method, instead of replacing the method's whole dictionary (which dropped `dl_nm`,
  `pml_cells` and the like).
- `device="auto"` falls back to CPU when CUDA is available but no device is visible
  (`CUDA_VISIBLE_DEVICES=""`), instead of failing with "Invalid device id".

## [0.1.0] - 2026-11-01

First public release.

### Added

- **One problem spec for every method.** Pydantic models with a `kind` discriminator: `scattering`,
  `layer_stack`, `periodic`, `mode`, `device`, `bands`, `circuit`, `optimization`. Any of them can be
  passed as a model, dict or JSON string. `lightsmith schema` prints the JSON Schema. Lengths are in nm,
  wavelengths are vacuum wavelengths, and the time convention is `exp(-iωt)`.
- **Entry points** `solve`, `estimate`, `compare`, `choose_method` and `doctor`. `method="auto"` picks
  the preferred applicable solver and explains why the others were skipped.
- **Solvers.** All are MIT, and all are native unless noted.
  - Scattering: Mie (native, homogeneous and layered spheres, GPU and autodiff), Mie through
    `miepython`, T-matrix through `treams` (`tmatrix` extra), DDA, FDTD, and the quasistatic dipole
    approximation.
  - Thin films and periodic structures: TMM (coherent and incoherent layers, differentiable), RCWA
    (1D/2D lattices, conical incidence, Li and normal-vector factorisation, differentiable), and
    FDTD for periodic and layer-stack problems at normal incidence.
  - Waveguides and devices: FDE (full-vector, bends, PML), EME, and 2D FDFD with adjoint gradients.
  - Photonic crystals: PWE (2D TE/TM and 3D, dense and LOBPCG solvers).
  - Circuits: S-matrix netlist solver with compact models and Touchstone I/O, differentiable.
  - Inverse design: density-based topology optimisation on FDFD (filters, projection, minimum
    feature size, MMA, Adam and L-BFGS-B), exported to polygons and GDS.
- **Materials.** Bundled CC0 datasets from refractiveindex.info (Au, Ag, Al, Cu, Si, SiO2, TiO2,
  water), online fetch of any database entry, and passive Drude–Lorentz fitting for time-domain solvers.
- **Layout.** Built-in GDSII reader and writer, extrusion of GDS layers into device problems, and an
  optional gdsfactory bridge (`layout` extra).
- **Interfaces.** The `lightsmith` CLI (JSON on stdout, JSON errors on stderr) and the `lightsmith-mcp`
  MCP server (`mcp` extra) for AI agents.
- **Plugins.** The `lightsmith.backends` entry-point group, so GPL-licensed solvers can live in separate
  packages (`lightsmith-adda`, `lightsmith-scattnlay`).
- **Typing.** A `py.typed` marker; the public API is annotated.
- **Documentation** site (MkDocs Material), `CITATION.cff` and `.zenodo.json`.

### Notes

- The `mcp` extra is pinned to `mcp>=1.2,<2`: the server uses the mcp 1.x `FastMCP` API, which mcp 2.x
  renamed.

[Unreleased]: https://github.com/lightsmith-photonics/lightsmith/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/lightsmith-photonics/lightsmith/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lightsmith-photonics/lightsmith/releases/tag/v0.1.0
