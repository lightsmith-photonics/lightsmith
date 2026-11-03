# Changelog

All notable changes to the `lightsmith` library are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/) (while the version is 0.x, minor releases may break the API).

## [Unreleased]

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
