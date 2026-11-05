# FDTD

Backend `lightsmith-fdtd` (method `fdtd`, modules `lightsmith.backends.fdtd` and
`lightsmith.backends._fdtd`). A 3D finite-difference time-domain solver for four problem kinds:

- [scattering](../problems/scattering.md), on a substrate too;
- [periodic structures](../problems/periodic.md) and [layer stacks](../problems/layer-stack.md), at normal
  incidence;
- [dipole emission](../problems/emission.md): Purcell factor, radiative enhancement, efficiency and radiation
  pattern.

It runs on the CPU or a CUDA GPU.

!!! note "Oblique incidence on periodic structures"
    Another branch is adding oblique incidence for periodic structures. See
    [FDTD oblique incidence](in-development/fdtd-oblique.md).

## Method

### Scattering

- **Grid and boundaries.** A Yee grid with a convolutional PML (Roden & Gedney 2000).
- **Source.** A total-field/scattered-field plane-wave source fed from tables of the grid's own discrete
  plane wave, at any direction (the method of the oblique substrate source below, with a uniform
  background). The 1D problem runs along the axis with the largest |k| component, relabelled to z by a
  cyclic axis permutation, which keeps the Yee staggering. The box does not leak at any angle: C_sca of a
  particle of the medium itself is about 1e-19 nm² in double precision. The interpolated 1D feed it
  replaces leaked ~1e-3 of the field off-axis. That feed's incident flux through the box also failed to
  cancel by ~1e-5 even on axis, which for a 20 nm sphere was up to 30 % of C_ext. C_abs subtracts the
  background's own flux through the box, and the intensity is n|E|²/2 of the incident wave. The grid's
  discrete Poynting flux differs from that by O((k dx)²), and differently with direction: about 1e-3
  along an axis and 4e-4 along a diagonal at k dx = 0.1. With n|E|²/2, a sphere's C_ext agrees between
  directions to 4e-5.
  Tables are stored every m steps, the largest m with ω h ≤ 0.2 at the top of the pulse band; m is about 3
  at 100 cells per wavelength. They are read with quintic interpolation, whose error (3e-7) only leaks.
- **Gaussian beams.** A `GaussianBeam` is its angular spectrum at fixed directions: Gauss–Legendre rings in
  the transverse wavevector and uniform azimuths, as many as the TFSF box needs (about 60–120 plane waves).
  Each is a table-fed plane wave, delayed so that all reach the focus in phase. Their tables are
  concatenated and scaled by the quadrature weights, and each face slab reads every component in one
  gather. A 65-component beam takes about 1 MB of tables per component at dx = 4 nm. Cross sections are per
  unit intensity at the focus. Each component's amplitude is known exactly, so the beam's background
  flux through the box is subtracted and near fields are available. Near fields are normalised by the
  beam's E at the focus along its polarisation.
- **Dispersion.** Drude and Lorentz auxiliary differential equations.
- **Interfaces.** Conformal meshing by tensor subpixel smoothing (Kottke, Farjadpour & Johnson 2008): each interface
  cell gets 1/ε = P⟨1/ε⟩ + (1 − P)/⟨ε⟩, P = n nᵀ, off-diagonal terms included. The off-diagonal coupling
  between neighbouring cells is symmetrised, which keeps long runs stable. See `mesh_refinement` below.
- **Metal interfaces.** Interfaces with a dispersive medium (metals) take the full tensor too, applied at
  grid vertices. Each vertex has the tensor of the cube around it, with ⟨1/ε⟩ and 1/⟨ε⟩ as dispersive ADE
  filters. It acts on the eight octant triads of E nodes around the vertex (one node per component), and
  each triad term is a congruence Sᵀ η S of a passive tensor. The sum is therefore passive, so it cannot
  grow, and in a uniform medium it reduces to the plain update. Adding the off-diagonal terms node by
  node, as for dielectrics, diverges for metals. Using the diagonal alone mixes the tangential average,
  whose pole lies inside the band, into every component, which turns each interface cell into a spurious
  resonator.
- **Spectra.** One broadband pulse covers every wavelength, and running DFTs give the spectra:
  - C_sca is the scattered flux leaving the TFSF box;
  - C_abs is the net total-field flux into the box;
  - C_ext = C_sca + C_abs.
- **Far field.** The scattered E and H on the outer flux box become equivalent surface currents, which are
  projected to the requested directions (near-to-far-field transformation) to give dC_sca/dΩ. Over a
  substrate the projection is by reciprocity (below).
- **Resonances.** E at probes around each particle is recorded after the pulse has passed, and a
  multi-channel matrix pencil (`lightsmith.resonance.harmonic_inversion`) gives each mode's wavelength,
  Q and amplitude, as Harminv does in Meep.
- **Stopping.** The run stops when the scattered-field energy on the outer flux box has decayed by
  `decay_by` from its peak, or at the time cap. The peak counts as at least 1e-6 of the total-field
  peak, so a particle that barely scatters still stops. Using the total field instead let the incident
  pulse set the peak, which cut off the ring-down of weak, sharp resonances (a 15 nm Drude sphere's
  spectrum rippled by ±10 %).
- **Substrates.** The background is the host above the plane and the substrate below, meshed layer by
  layer: a node's pixel holds a substrate fraction f, giving 1/⟨ε⟩ for E_x, E_y and ⟨1/ε⟩ for E_z, exact
  for a plane. Particle meshing only touches pixels that hold a particle. The TFSF box takes its incident
  field from an auxiliary column: a 3D engine 3 × 3 cells wide, periodic in x and y, with the main grid's
  z nodes, layered background, CPML and time step, driven by a current sheet outside the box. At normal
  incidence the background does not vary in x and y, so the column's fields (incident, reflected and
  transmitted waves) are the main grid's background exactly, and the box does not leak wherever it cuts
  the interface. A second column of the incidence medium alone gives the incident intensity.
  - **Oblique incidence.** At a fixed angle every frequency has the same transverse slowness
    s = n sinθ, so the background is F(z, t − s·r∥). Per frequency, the Yee equations for fields
    ∝ exp(iωs·r∥) reduce to a 1D discrete problem along z, solved with radiation conditions above and
    below by a QR-stabilised sweep:
    - transverse differences become iK with K = 2 sin(ωs dx/2)/dx;
    - the leapfrog becomes Ω = 2 sin(ω dt/2)/dt;
    - each node has its D-form ADE response.

    An inverse FFT turns the solutions into tables F_c(z, τ). Every TFSF point reads its component at its
    own height and delay, with quintic interpolation in τ. The tables satisfy the grid's own equations, so
    the box does not leak, including under total internal reflection: C_sca of a null particle is below
    1e-14 nm² in double precision. Two details keep the tables exact:
    - The pulse starts at e⁻²⁵ and is high-passed below the band. Near-static evanescent content under
      total internal reflection would otherwise ring for a very long time.
    - The DFT stride uses a 6 ω_half margin, since absorption is measured against the exact background.

  C_sca is
  the scattered flux leaving the outer box (both half-spaces). C_abs is the net total-field flux into the
  inner box minus the background's (from the column). The default PML is 16 cells with a substrate:
  steep waves scattered into the substrate need it (10 cells left 1.3 % at 800 nm, dx = 2 nm).
  - **Far field over a substrate.** The scattered field outside the outer box is radiated by its
    equivalent currents into the layered background. Reciprocity with a dipole far away in direction n̂
    gives the far-zone amplitude: p·f = (iω/4π) Σ (F_E·J − F_H·M) dA, where F is the background field
    (incident, reflected and transmitted waves) of the plane wave coming in from n̂. F is the grid's own
    discrete background at each sample's Yee position, from the same 1D solver as the tables. On each
    face the sum separates into the transverse phase and the z profile, so every face is two matrix
    products over all directions. Into an absorbing substrate there is no far field.
  - **Absorbing substrates.** The box fluxes would count the substrate's own absorption near the
    particle, so over an absorbing substrate the cross sections are defined without them:
    - C_abs is the particles' own absorption. The Yee update gives curl H = −iΩD at every E node exactly
      (Ω = 2 sin(ω dt/2)/dt), so a node absorbs (Ω/2) Im(E*·D) dV. E and D are recorded on the dispersive
      nodes whose pixel (for the vertex tensor, either of its vertices) holds a particle medium. That
      set is closed under the tensor's coupling, so its sum is the discrete Poynting inflow: it equals
      the inflow through a closed host box around a lifted sphere to 1e-6. Only where a particle touches
      the substrate does a node hold both, and it is split by fraction × Im ε.
    - C_sca is the power scattered into the upper half-space, from the far field. A Gauss–Legendre rule
      in cos θ is split at the critical angle when the other side is optically thinner.
    - C_ext is the power the particle takes from the reflected wave, by the optical theorem for that
      wave: (4π/(ω n_inc |E₀|²)) Im(f(k_r)·E_r*), with f in the reflected wave's own direction. Over a
      transparent substrate the same sum over the reflected and transmitted waves, with C_sca over both
      half-spaces and the volume C_abs, gives the box values to 3e-4, including total internal reflection.
    - C_ext − C_sca − C_abs is then the change of the substrate's own absorption, which can be negative.

    The incident wave must come from above (it would decay in the substrate). Every incidence then uses
    the tables, normal incidence included.
- **Materials.** A constant lossy permittivity becomes a Drude term matched exactly at the band centre,
  with a warning that it is not causal over a band. Tabulated and library materials are fitted with
  `fit_drude_lorentz`, with a warning that reports the RMS error.
- **Mirror symmetries.** When the particles map onto themselves under x → −x about their centre (and/or
  y, or z without a substrate), and the plane wave travels in that plane with its polarisation either in the
  plane or normal to it, the incident and scattered fields are even or odd about it. The run then keeps only
  one half of the grid per plane (2× or 4× fewer cells), as Meep's mirror symmetries do:
  - Polarisation in the plane: tangential E is even, a PMC plane (symmetric). Polarisation normal to the
    plane: tangential E is odd, a PEC plane (antisymmetric). A plane wave along z polarised along x has x
    antisymmetric and y symmetric.
  - The plane is node plane 1 of the half grid, with a ghost plane at index 0 and no PML on that side. A PEC
    plane is a wall like the grid's outer planes. A PMC plane steps tangential E on the plane, reading a
    ghost tangential H set to minus its image before every E update.
  - The conformal tensors are folded, so the half grid steps exactly as the symmetric field of the whole one:
    the off-diagonal rows' mirror images are added, and couplings to ghost-side nodes act through their
    stepped images with the parity sign. Vertex-tensor vertices on the plane read their ghost-side nodes
    the same way.
  - Flux boxes are open on the planes (no flux crosses them; the faces they cut keep half weight on the
    plane), and their fluxes and the particle absorption are doubled per plane. Far fields are the sum over
    the mirror images of the box's part in the grid, E(n̂) = Σ σ_g g E_part(g n̂). Near fields beyond a plane
    are the image's times the parity.

  The symmetric run matches the full one to 1e-11 or better in double precision (cross sections, near and
  far fields, metals, substrates). It is on by default (`symmetry: "auto"`). The provenance's `symmetry` lists
  the planes used (`mirrors`, `reduction`) and why the others were not (`skipped`). Gaussian beams run
  without it: their discrete angular spectrum is not always mirror-symmetric.
- **GPU.** On PyTorch CUDA, the time loop is replayed as CUDA graphs.

### Graded grids (scattering)

`grid: "auto"` or `mesh_override` make the grid non-uniform, as Lumerical's override regions and Tidy3D's
graded grids do.

- **Grid.** The grid is still a tensor product: along each axis the node planes are free, and the three axes
  are independent. A spacing function sets the cell widths. It takes the base spacing and refines each region
  [lo, hi] to that region's spacing s. Around a region the widths grow back no faster than
  h(x) = s + (r − 1) dist(x, region), with r the `grading_ratio`. Cells are marched outward from the scene's
  centre, and each takes the minimum of h over itself. Neighbouring cells therefore differ by at most r
  (`max_ratio` in the provenance).
- **Update.** Primary widths p_i = x_{i+1} − x_i separate nodes, and dual widths d_i = (p_{i−1} + p_i) / 2
  separate the midpoints around node i. The H update divides E differences along an axis by the primary
  widths. The E update divides H differences by the dual widths. Both use per-axis arrays through the array
  API, so CPU, CUDA and CUDA graphs share one path. Weighted by the cell volumes (E_c: p_c d_a d_b; H_c:
  d_c p_a p_b), the two curls are adjoint, so the scheme conserves the same discrete energy as the uniform
  one. The time step comes from the smallest cell per axis, dt = courant / √(Σ_a 1 / h_min,a²).
- **What follows the widths:**
  - CPML: uniform cells; the conductivity profile takes the end cells' width.
  - TFSF tables: the exact discrete background comes from the graded 1D problem along the incidence axis (one
    transfer matrix per cell, primary and dual widths). The box stays leak-free for incidence along an axis.
    Oblique incidence needs the grid uniform along the axes the wave varies along.
  - The layered background and its auxiliary columns use each node's own cell along z.
  - Subpixel and conformal smoothing: each node's pixel is its own cell (primary along its component, dual
    along the others), not a cube.
  - The vertex tensor of metal interfaces: each vertex's cell is its dual cell. Its D̃ along d is the mean
    of the two d-nodes, weighted by their half-cells p_{j−1}/2 and p_j/2 inside it, so the octant triads
    carry the volumes of the vertex's octants. The map stays a sum of passive congruences in the volume-weighted
    inner product. The conformal off-diagonal coupling between nodes p and q becomes O(p, q) / (2 V(p)): their
    overlap volume over p's volume, which keeps the operator symmetric in that inner product.
  - Flux boxes, near-to-far-field surfaces and the layered far field use the cells' own widths as area
    weights, and the samples' own positions.
  - DFT monitors, near fields and resonance probes use world coordinates.
  - Particle absorption over absorbing substrates uses each node's own volume.
  - Mirror planes: along a mirror axis the spacing function is symmetric about the centre, and the half grid
    keeps its node coordinates.
- **Layout.** Every box face (TFSF and both flux boxes), the gaps and the CPML sit at least three cells beyond
  the graded zone, where the grid is uniform along the face normal. The padding and gaps keep their physical
  lengths.
- **Grading reflects.** A change of cell size h reflects about (k h)²/16 of a wave, however gently it is
  graded. The discrete line is an LC ladder whose node impedance Z₀ / √(1 − (k h)²/4) depends on h. The
  graded zone is shorter than a wavelength, so its reflections add up to the full jump. With the base at
  20 cells per vacuum wavelength this was 0.6 % at 400 nm, and it moved C_ext by up to 2 %. `grid: "auto"`
  therefore sets the base to `background_cells_per_wavelength` (60) cells per shortest wavelength in the
  background (0.07 %). The cells grow from the particle's spacing to that base.
- **`grid: "auto"`.** Each particle's bounding box is refined to the uniform rule for that particle alone:
  `cells_per_wavelength` in its material and its smallest half-extent / `cells_per_radius`. The refinement
  extends a margin around the box for the near field: half the smallest half-extent, and at least two cells.
  When no particle needs cells finer than the base, the grid is today's uniform one. `mesh_override` boxes
  add their own spacings, with `spacing_nm` (or the uniform rule) as the base.

### Dipole emission

- **Source.** A point current along the dipole, spread over the eight E nodes around it by trilinear weights.
  Its time profile is the exact discrete derivative of a Gaussian pulse, so it leaves no net charge: a
  static dipole would keep a self-field ~1/(k dx)³ larger than the radiating field, and its DFT would
  depend on when the run stops.
- **Powers.** The emitted power is −½ Re Σ E·J* dV at the source; the radiated power is the flux through a
  closed box around the dipole and the particles. A second run of the dipole in the host alone gives P0, so
  the discretisation of the point source cancels in the ratios. That run's emitted and radiated powers agree
  to 1e-6 or better (`reference_balance` in the provenance).
- **Precision.** Double by default, even on the GPU: the emitted power is the small real part of the
  self-field, and single precision leaves 1–4 % errors. DFT sampling keeps pulse content above
  ω_c + 6 ω_half (e⁻³⁶) out of the band, for the same reason.
- **Radiation pattern.** The near-to-far-field projection of the fields on the box, normalised by P0.

### Periodic structures and layer stacks

- **Cell and boundaries.** One unit cell, with periodic x and y boundaries and a CPML along z. The cells
  are non-cubic, so each period holds a whole number of cells.
- **Source.** A one-sided plane-wave source in the superstrate.
- **Diffraction orders.** The reflected and transmitted DFT planes are split into orders by a 2D FFT. R
  and T are sums over the propagating orders, and A = 1 − R − T.
- **Layer stacks.** A layer stack at normal incidence runs once and serves both s and p.

## Options

`options["fdtd"]` or `options["lightsmith-fdtd"]`.

| Option | Default | Meaning |
|---|---|---|
| `spacing_nm` | `None` | Yee cell size. Overrides the two options below. |
| `cells_per_wavelength` | `20` | Cells per shortest wavelength in the densest material |
| `cells_per_radius` | `None` | Minimum cells across the smallest particle. Default 10, or 20 if a material is plasmonic (Re ε < 0) in the band. |
| `pml_cells` | `None` | CPML thickness in cells. Default: 10, or 16 with a substrate. |
| `pml_kappa` | `1.0` | CPML κ_max; values above 1 damp evanescent fields faster |
| `padding_nm` | `None` | Gap between the particles (or the stack) and the TFSF box. Default 4 cells, and for scattering at least 3 % of the longest wavelength in the medium. |
| `pml_gap_nm` | `None` | Gap between the flux monitors and the CPML. Default for scattering: 4 cells, at least 3 % of the longest wavelength in the medium. For periodic structures, 3 decay lengths of the slowest evanescent order (0.25–1.5 λ). |
| `courant` | `0.95` | Time step as a fraction of the 3D Courant limit |
| `run_time_fs` | `None` | Fixed simulated time; disables the decay shut-off |
| `decay_by` | `1e-5` | Stop when the monitor energy falls below this fraction of its peak |
| `max_time_fs` | `2000` | Hard cap on simulated time; high-Q resonances need picoseconds |
| `precision` | `"auto"` | `"auto"` means single on the GPU and double on the CPU, and double over an absorbing substrate |
| `subpixel` | `True` | Smooth interfaces; `False` means staircase. Superseded by `mesh_refinement`. |
| `mesh_refinement` | `None` | `staircase`, `volume_average` (⟨ε⟩), `subpixel` (diagonal tensor, the 0.1 behaviour) or `conformal` (full tensor; for metals at grid vertices). Default: `conformal`, or `staircase` when `subpixel` is false. |
| `subpixel_samples` | `8` | Supersamples per axis for interface cells (2–24) |
| `dft_stride` | `None` | Time steps between DFT samples. Default: aliasing-safe, from the source bandwidth. |
| `symmetry` | `"auto"` | Scattering: use the PEC/PMC mirror planes the particles and plane wave allow (2× or 4× smaller runs, the same results); `"off"` runs the whole grid |
| `grid` | `"uniform"` | Scattering: `"auto"` grades the grid, refining each particle (see Graded grids) |
| `mesh_override` | `[]` | Scattering: refinement boxes `{center_nm, size_nm, spacing_nm}` (one spacing or one per axis), graded to the base spacing around them |
| `grading_ratio` | `1.25` | Largest cell-width growth from one cell to the next (1.05–1.5) |
| `background_cells_per_wavelength` | `60` | `grid: "auto"`: base cells per shortest wavelength in the background |
| `cuda_graphs` | `True` | Replay time steps as CUDA graphs |
| `max_cells` | `300000000` | Refuse grids larger than this |

There are no gradients.

## Validation

### Scattering (`tests/test_fdtd.py`)

| Test | Reference | Tolerance |
|---|---|---|
| `test_fdtd_dielectric_sphere_matches_mie_cpu`: n = 1.5, r = 40 nm, dx = 6 nm | Mie | C_ext, C_sca < 3 %; \|C_abs\| < 1e-3 C_ext |
| `test_fdtd_gpu_matches_cpu` (gpu): GPU float32 with CUDA graphs | CPU float64 | 1e-4 |
| `test_fdtd_drude_sphere_matches_mie` (gpu): Drude r = 40 nm in water, dx = 2 nm | Mie | peak within 2 nm; peak height < 5 %; C_ext < 7 % and C_sca < 12 % everywhere |
| `test_fdtd_layered_sphere_near_field_and_oblique_incidence` (gpu) | `lightsmith-mie` | C_ext < 3 %; near-field \|E\|² RMS < 6 % |
| `test_fdtd_conformal_tensor_smoothing_beats_diagonal` (gpu): r = 60 nm, dx = 6 nm | Mie | conformal < 0.2 %, subpixel > 0.8 % |
| `test_fdtd_far_field_matches_mie` (gpu): r = 60 nm, dx = 4 nm | `lightsmith-mie` | dC_sca/dΩ median < 0.2 %, max < 1 %; integral = C_sca within 0.5 % |
| `test_fdtd_drude_sphere_resonance` (gpu): Drude r = 40 nm in water | Exact poles of Mie's a₁, a₂ | dipole within 3 nm, Q 7–10; quadrupole within 2 nm, Q 28–40 |
| `test_fdtd_emission_near_dielectric_sphere_matches_mie` (gpu): dipole 10 nm from n = 2, r = 60 nm, tilted 45° | `lightsmith-mie` | Purcell and radiative < 0.5 % (measured 0.07 % radial, 0.17 % tangential) |
| `test_fdtd_emission_near_drude_sphere_matches_mie` (gpu): dipole 10 nm from a Drude r = 40 nm sphere in water, tilted 45° | `lightsmith-mie` | Purcell < 15 % (measured 7.9 % radial, 11.5 % tangential) |
| `test_fdtd_lone_dipole_radiation_pattern` (gpu) | 3/(8π) sin²θ | < 0.2 % of the peak (measured 0.05 %); integral 1 within 0.2 % |
| `test_fdtd_oblique_tfsf_does_not_leak_cpu`: particle of the medium, direction (1, 2, 3) | Zero; the plane wave | C_sca < 1e-12 nm² (measured 1e-19); near field < 1e-3 (measured 3e-4, the output interpolation) |
| `test_fdtd_oblique_tfsf_sphere_is_direction_independent` (gpu): r = 60 nm, dx = 6 nm, (0, 0, 1), (1, 1, 1), (1, 2, 3) | Mie; the other directions | C_ext < 0.3 % (measured 0.08 %); spread < 5e-4 (measured 4e-5) |
| `test_fdtd_small_sphere_matches_mie` (gpu): r = 20 nm, dx = 2 nm, float64, axial and (1, 2, 3) | Mie | C_ext < 0.3 % (measured 0.04–0.08 %); \|C_abs\| < 1e-4 C_ext |
| `test_fdtd_gaussian_beam_focus_and_profile` (gpu): r = 40 nm sphere, 1 µm waist, at the focus and 333 nm off axis | Mie × the beam's |E|² | 0.2 % (measured 0.04 % at both; 0.13 % and 0.1 % with 1D-fed components) |
| `test_fdtd_gaussian_beam_near_field` (gpu): particle of the medium, 400 nm waist | The discretised beam | C_sca < 1e-8 nm²; near field < 2e-3 (measured 2e-4) |
| `test_fdtd_conformal_is_stable_at_late_times` (slow, gpu): n = 3.5 sphere, 3 ps | — | field decays below 1e-12 |
| `test_fdtd_dielectric_sphere_fine_grid` (slow, gpu): r = 60 nm, dx = 2 nm | Mie | C_ext < 0.2 % |
| `test_fdtd_drude_sphere_fine_grid` (slow, gpu): dx = 1.33 nm | Mie | peak within one 10 nm sample; height < 4 % |
| `test_fdtd_gold_library_material_via_drude_lorentz_fit` (slow, gpu): Johnson–Christy gold | Mie | peak within 5 nm; height < 3 %; C_ext < 7 % everywhere |
| `test_fdtd_metal_vertex_tensor_is_symmetric_and_bounded` (cpu) | Construction | the D → E map is symmetric, Rayleigh quotients within [1/ε∞,max, 1/ε_min] |
| `test_fdtd_metal_vertex_tensor_is_stable_at_late_times` (slow, gpu): Drude r = 40 nm, 1.5 ps | Decay | below 1e-15 of peak (measured 6e-22) |
| `test_fdtd_mirror_symmetry_matches_full_run_cpu`: sphere with near and far fields (2 planes), oblique spheroid (1 plane), Drude dimer (2 planes, vertex tensor), float64 | The full grid | 1e-6 relative (measured 1e-15–3e-11; 7e-9 for the dimer's 1e-4-smaller C_sca) |
| `test_fdtd_mirror_symmetry_detection` | Construction | planes only for symmetric particles and plane waves; none for beams, `"off"`, asymmetric overlaps |
| `test_fdtd_mirror_symmetry_speedup_gpu` (gpu): r = 60 nm, dx = 3 nm, float32, 2 planes | The full grid; Mie | 3e-5 (measured 5e-6); Mie 0.2 % (measured 0.063 % for both); time loop > 1.3× faster (measured 1.55×) |
| `test_fdtd_mirror_symmetry_speedup_fine_grid_gpu` (slow, gpu): the same at dx = 1.5 nm | The full grid; Mie | 3e-5 (measured 7e-6); > 1.8× faster (measured 2.3×) |

Measured in the test module docstring (RTX 5070, float32; CPU float64 agrees to about 1e-7):

- Dielectric r = 60 nm sphere, maximum C_ext error over 400–800 nm at dx = 6, 4, 3 and 2 nm:
  - `conformal`: 0.04 %, 0.05 %, 0.07 % and 0.05 %;
  - `subpixel` (diagonal only): 1.1 %, 0.8 %, 0.6 % and 0.4 % (first order).

  With the conformal default the remaining error is the source, monitor and PML floor, not the interfaces.
- Far field of the same sphere at dx = 4 nm: median 0.01–0.03 % from Mie, worst 0.4 % (at 90° in the
  E-plane, where dC_sca/dΩ is near zero).
- Drude r = 40 nm sphere in water, RMS C_ext error over 480–600 nm at dx = 3, 2.5, 2, 1.6 and 1.33 nm:
  - `conformal` (vertex tensor): 3.7 %, 3.5 %, 2.5 %, 2.3 % and 1.6 %; peak height within 1.5–3.6 %;
  - `subpixel` (diagonal): 7.5 %, 9.1 %, 6.6 % and 6.1 % (to 1.6 nm), rippling across the band;
  - `staircase`: 22 %, 17 %, 15 % and 13 %.

  Mirror symmetry gives the same numbers: in double precision the half or quarter grid matches the whole one
  to 1e-11 or better. On the GPU (float32), a quarter grid agrees with the full one to 5e-6. The speed-up
  grows with the grid, because the source's per-step cost (table interpolation on the TFSF faces) and kernel
  launches do not shrink as much as the volume. In steady state the quarter grid is 1.6× faster at 1e6 full
  cells, 2.5× at 5e6 and 3.7× at 1.5e7. Whole runs of the r = 60 nm sphere: 1.55× at dx = 3 nm, 1.84× at 2 nm,
  2.3× at 1.5 nm.

  The vertex tensor still has a first-order term: averaging the tensors of two vertex cubes is not the
  tensor of the node's own cube. Through it, a dielectric n = 1.5 sphere errs by 1.0 %, 0.65 % and 0.48 % at
  dx = 6, 4 and 3 nm, against 0.04–0.07 % with the dielectric scheme, so dielectric interfaces keep that one.
- Johnson–Christy gold (fitted), r = 40 nm, dx = 2 nm: peak at 550 nm as Mie, height within 0.7 %.
- Ring-down of the same Drude sphere: dipole 535.6 nm, Q 8.2 (exact pole 533.8 nm, Q 8.74); quadrupole
  488.4 nm, Q 31 (487.5 nm, Q 35.1). For a 10 nm sphere at dx = 0.5 nm: 500.6 nm, Q 31.9 (499.8 nm, Q 34.2).

### Graded grids (`tests/test_fdtd_graded.py`)

| Test | Reference | Tolerance |
|---|---|---|
| `test_graded_axis_is_bounded_symmetric_and_refined`, `test_graded_layout_boxes_lie_in_uniform_cells` | — | growth ≤ `grading_ratio`; region spacing inside; box faces, gaps and CPML in uniform cells |
| `test_graded_options_collapse_to_the_uniform_run` | uniform run | bit for bit |
| `test_graded_code_path_reproduces_uniform_grid_cpu`: dielectric (2 mirrors, near and far field), Drude (vertex tensor), glass, absorbing substrate | uniform run through the old code | 1e-12 (measured 3e-15 to 6e-14) |
| `test_graded_tfsf_does_not_leak_cpu`: n = 1 sphere, 10 → 2.5 nm | 0 | C_sca < 1e-10 πr² (measured 2e-15 nm²) |
| `test_graded_mirror_symmetry_matches_full_run_cpu` | full graded run | 1e-6 |
| `test_graded_engine_conserves_the_discrete_energy`: PEC box, ratio 1.4, 4000 steps | exact | 1e-10 |
| `test_graded_vertex_tensor_is_symmetric_and_bounded` | — | symmetric in the volume-weighted inner product; Rayleigh quotients within [1/ε_∞,max, 1/ε_min] |
| `test_graded_high_index_sphere_matches_uniform_and_mie` (gpu, slow): n = 3.5, r = 20 nm, 400–800 nm | uniform 2 nm run, converged domain | 0.2 % (measured 0.11 %, against the uniform run's 0.66–0.71 % error from Mie) |
| `test_graded_drude_sphere_matches_uniform_with_fewer_cells` (gpu, slow): Drude r = 10 nm in water, 0.5 nm | uniform 0.5 nm run | ΔC_ext < 5 % of the uniform run's error (measured 7e-4 of the peak against 13 %); ≥ 3× fewer cells |
| `test_graded_sphere_on_substrate_matches_uniform` (gpu, slow): n = 3.5, r = 20 nm touching glass (from below and above, column source) or an absorbing substrate (tables, layered far field) | uniform 2 nm run | 0.5 % (measured C_ext 0.16 %, 0.20 %, 0.34 %; C_sca 0.13 %, 0.12 %, 0.05 %) |
| `test_graded_far_and_near_fields_match_mie` (gpu): n = 3.5, r = 20 nm | `lightsmith-mie` | dC_sca/dΩ median < 1 %, max < 1.2 % (measured 0.75 %, 0.79 %; uniform 2 nm: 0.68 %, 0.72 %); near-field \|E\|² beyond 4 nm from the surface: RMS < 2 % (measured 0.47 %) |
| `test_graded_grid_is_stable_at_late_times` (gpu, slow): conformal n = 3.5 sphere (3 ps) and Drude sphere through the vertex tensor (1.5 ps), 3 nm graded to 10 and 8 nm | — | decay level < 1e-15 (measured 5e-20 and 9e-21; no growth at the transitions) |

Cells and run times (RTX 5070, CUDA graphs):

| Case | Uniform | Graded (`grid: "auto"`) |
|---|---|---|
| n = 3.5, r = 20 nm (2 nm cells), float64 | 242 k cells, 57 s | 153 k cells (6.7 nm base), 54 s |
| Drude r = 10 nm in water (0.5 nm cells), float64 | 1.74 M cells, 662 s | 488 k cells (5.6 nm base), 319 s |
| the same, float32 | 1.74 M cells, 305 s | 488 k cells, 269 s |

Both use the time step of the finest cells. The small runs are dominated by the source tables' fixed per-step
cost (one table interpolation per TFSF face slab; see the roadmap), so the time falls less than the cell count.
Timings on a shared GPU vary by up to 2×; the cell counts are exact.

The Drude sphere's off-peak values are sensitive to when the run stops. With the default `decay_by` (1e-5 of the
energy, 3e-3 of the amplitude), the plasmon's truncated ring-down leaks about 3e-4 of the peak into the tails,
and the two grids then differ by up to 8 % at 650 nm, where C_ext is 0.3 % of the peak. With `decay_by: 1e-8`
they agree within 0.55 % at every wavelength, against 13–18 % errors from Mie.

### Periodic structures (`tests/test_fdtd_periodic.py`)

Differences are maximum absolute differences in R, T and A.

| Test | Reference | Tolerance |
|---|---|---|
| `test_fdtd_layer_stack_matches_tmm_cpu`: 300 nm n = 2 slab; 30 nm Drude film | [TMM](tmm.md) | < 0.01 (slab); < 0.005 (film) |
| `test_fdtd_layer_stack_gpu_matches_cpu_and_metal_film` (gpu): 40 nm Ag film | TMM; CPU | < 0.01; GPU vs CPU 2e-5 |
| `test_fdtd_1d_grating_matches_rcwa` (gpu): lamellar grating | [RCWA](rcwa.md), 30 harmonics | total < 0.015; per order 0.01 |
| `test_fdtd_hole_array_guided_mode_resonance_matches_rcwa` (slow, gpu) | RCWA | resonance within 0.5 nm (measured 0.09 nm) |

### Substrates (`tests/test_fdtd_substrate.py`)

The reference treats a small sphere as a point dipole (Mie a₁ polarizability) driven by the incident and
reflected waves and coupled to its own reflected field through Sommerfeld integrals, exact in the dipole
limit.

| Test | Reference | Tolerance |
|---|---|---|
| `test_substrate_tfsf_does_not_leak_cpu`: sphere of the host medium on glass | Zero | C_sca < 1e-9 nm², C_abs < 1e-6 nm² (float64) |
| `test_sphere_above_glass_matches_dipole_reference` (gpu): n = 2, r = 20 nm, centre 40 and 100 nm above glass, dx = 2 nm | Dipole | C_sca < 0.5 % (measured 0.17 %, 0.23 %) |
| `test_sphere_lit_through_glass_matches_dipole_reference` (gpu): the same, lit through the glass | Dipole | C_sca < 0.5 % (measured 0.23 %) |
| `test_substrate_near_field_is_the_standing_wave_cpu`: sphere of the host medium on glass | |e^{−ikz} + r e^{ikz}|² above, |t|² below | < 0.5 % (measured 0.03 % at dx = 2 nm); uniform in x to 1e-9 |

| `test_discrete_layered_solver_matches_fresnel`: the discrete background solver, glass, 30° and 60° (TIR), s and p | Fresnel | \|r\| and phase < 2e-4 at dz = 1 nm |
| `test_oblique_substrate_tfsf_does_not_leak_cpu`: null particle on glass, 40° | Zero; Fresnel standing wave | C_sca < 1e-12 nm²; near field < 0.2 % (measured 2e-4) |
| `test_oblique_sphere_above_glass_matches_dipole_reference` (gpu): n = 2, r = 20 nm, 40 nm above glass | Dipole | 45° s / p from above: 0.17 % / 0.33 %; 30° p through the glass: 0.20 %; total internal reflection at 60°: s 0.47 %, p 0.7–1.7 % |
| `test_absorbing_substrate_null_particle_cpu`: null particle over an absorbing dielectric, normal and 40° | Zero | C_sca < 1e-12 nm²; C_abs exactly 0; \|C_ext\| < 1e-4 nm² (linear in the residual field); far field zero, exactly 0 into the substrate |
| `test_transparent_substrate_far_field_quantities_match_box_cpu`: lossy sphere on glass, 45° p, with the absorbing-substrate definitions | The box fluxes | 0.2 % (measured 3e-4 at dx = 2.5 nm) |
| `test_particle_absorption_equals_host_box_inflow_cpu`: lossy sphere lifted over an absorbing dielectric | Net inflow through a closed host box around it | 1e-6 (float64) |
| `test_sphere_above_absorber_matches_multipole_reference` (gpu): n = 2, r = 20 nm, 100 nm above an absorbing dielectric (ε ≈ 14 + 3i), normal, 45° s and p | Multipole | C_sca < 0.2 % (measured 0.01–0.07 %); C_ext within 2e-3 of its interference scale (measured 5e-4); far field 0.5 % |
| `test_lossy_sphere_above_absorber_matches_multipole_reference` (gpu): lossy sphere (ε ≈ 4.5 + 1i), the same geometry | Multipole, relative to the sphere in free space | C_abs 0.2 %, C_ext 0.4 %, C_sca 0.3 % (measured within 0.1 %) |
| `test_substrate_mirror_symmetry_matches_full_run_cpu`: sphere on glass at normal incidence (column, 2 planes) and 45° s (tables and layered far field, 1 plane); over an absorbing dielectric (particle absorption, far-field C_sca and C_ext) | The full grid | 1e-6 relative (measured 1e-15, 5e-12, 4e-11) |

Over absorbing substrates the reference is the sphere's full Mie response to the incident and reflected waves,
plus its dipole's coupling to its own reflected field, radiating directly and by specular reflection. The
dipole model alone is off by the b₁ and quadrupole terms (3 % of the dipole amplitude at x = 0.3). C_ext and
absorption feel these at first order, because over a reflector C_ext depends on the sphere's reactive response.
The lossy sphere's own discretisation at dx = 2 nm (C_abs −0.7 %, C_sca −1.4 %, the same in free space) is
divided out. At a 20 nm gap the agreement is 0.3 %: the reference leaves out the higher multipoles' own
coupling to the substrate. Over a gold-like Drude metal (ε ≈ −9.4 + 0.6i at 600 nm, sphere 100 nm above):
C_sca is within 0.11–0.14 % and C_ext within 2e-5 of its scale (5e-4 at 400 nm, where the metal barely
reflects), in double precision (`test_sphere_above_metal_matches_multipole_reference`, slow). In float32 the
metal's long ring-down (65 000 steps) left rounding noise in the scattered field: C_sca was off by up to 1.2 %,
and by 10 % with 100 nm more padding. So `precision: "auto"` runs double over an absorbing substrate, about 3×
slower on a consumer GPU. Over the absorbing dielectric, float32 was within 0.07 % as well.

At dx = 1.33 nm the normal-incidence spheres are within 0.11 % and 0.19 %. Under total internal reflection the
evanescent field varies across the sphere, which the dipole model misses, hence its larger p deviation.

## Limits

**Scattering**
- No orientation averaging. Near fields, far fields and resonances are supported.
- Substrates:
  - oblique incidence through the substrate needs a constant-index substrate (a fixed angle in a dispersive
    incidence medium has no fixed transverse slowness); grazing angles (more than 84°) are refused;
  - over an absorbing substrate: incidence from above only; C_ext ≠ C_sca + C_abs (the rest is the change of
    the substrate's own absorption); where a particle touches the substrate, the nodes holding both are split
    by fraction × Im ε, an O(dx) ambiguity at the contact;
  - far field with incidence through the substrate at normal incidence needs a constant-index substrate;
  - near-field points outside the source box, beyond the incident-field column's range (in the PML or above
    the source plane), are NaN;
  - absorption of a weak absorber is limited by float32 cancellation, because the box carries the full
    incident flux (~0.004 nm² for a 100 nm box); use `precision: "double"` for it.
- Mirror symmetry: planes through the particles' bounding-box centre, normal to x, y or z (not z over a
  substrate), for plane waves polarised in the plane or normal to it; not for Gaussian beams or diagonal
  planes. Resonances come from the probes on the grid's side (fewer, non-redundant channels), so the weaker
  fitted modes can differ slightly from a full run's; modes of the other parity are not excited anyway.
  The decay shut-off sees the energy of the box part, so a decaying run can stop one chunk apart from a full
  one (well below `decay_by`).
- Graded grids (`grid: "auto"`, `mesh_override`):
  - scattering only; emission, periodic and layer-stack problems refuse them;
  - plane waves only, not Gaussian beams;
  - oblique incidence needs the grid uniform along the axes the wave varies along (the tables are exact
    only for a wave that does not vary along a graded axis); axis-aligned incidence works on any graded
    grid, substrates included;
  - every change of cell size reflects about (k h)²/16 of a wave. Keep the coarsest cells fine enough
    (`background_cells_per_wavelength`, or `spacing_nm` with `mesh_override`): at 20 cells per wavelength the
    reflection is 0.6 %, and C_ext moved by up to 2 %;
  - the time step comes from the smallest cell, so a run is cheaper in proportion to its cells, not its steps.
    On a consumer GPU in float32, the per-step cost of the TFSF tables dominates below a few million cells,
    and the speed-up there is smaller than the cell ratio;
  - over a substrate, the layered far field's transverse grid wavenumbers are those of the base spacing, so
    samples on graded transverse cells carry an O((k h)²) phase error;
  - along a mirror axis the refinement is made symmetric about the centre.
- Gaussian beams: no substrate yet. Table memory grows with the number of components (about 1 MB each at
  dx = 4 nm); `estimate` includes it.
- Resonances of metal particles: the plasmon Q comes out 7–10 % low, and a weak extra mode can appear
  next to the plasmon (amplitude 0.19 at 503.7 nm for a 10 nm Drude sphere). Broad modes (Q below about 10)
  are the hardest to fit: allow `max_modes` of 10 or more.
- Emission near metals: a dipole 10 nm from a Drude sphere (r = 40 nm in water, dx = 2 nm) has its Purcell
  factor within 7.9 % (radial) and 11.5 % (tangential) of Mie over 480–600 nm. The radiative part is within
  about 10 %, except where it is small (41 % at 480 nm, where it is 4 against a Purcell factor of 300).
- Metal interfaces: the vertex tensor leaves a first-order error of about 2.5 % RMS in C_ext at r/20
  resolution (see validation). It falls as dx is refined.
- Warns about:
  - fields that have not decayed by the time cap;
  - negative absorption;
  - non-causal constant lossy materials;
  - fitted materials.

**Periodic structures and layer stacks**
- Normal incidence only. For oblique incidence, use [RCWA](rcwa.md) or [TMM](tmm.md).
- Rectangular lattices only: a1 along x and a2 along y.
- The superstrate must be lossless and non-dispersive.
- No incoherent layers.

**Both**
- Raises an error above `max_cells` (increase `spacing_nm`).
- Raises `FloatingPointError` on divergence: non-finite fields, or monitor energy above 1e8 times its
  peak while the pulse passed (float64 CPU runs stay finite long after diverging). Refine the mesh,
  lower `courant`, or give lossy materials as Drude/DrudeLorentz instead of a constant complex index.
