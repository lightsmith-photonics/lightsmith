# Changelog

All notable changes to the `lightsmith` library are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/) (while the version is 0.x, minor releases may break the API).

## [Unreleased]

### Added

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
- FDTD scattering: the default TFSF padding and PML gap are now at least 3 % of the longest wavelength
  in the medium, so refining the grid no longer moves the monitors closer to the particle.

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

[Unreleased]: https://github.com/lightsmith-photonics/lightsmith/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lightsmith-photonics/lightsmith/releases/tag/v0.1.0
