# Discretisation: what a solver will discretise

`lightsmith.discretization(problem, method="auto")` reports what a backend will discretise for a problem, before any
solve: the FDTD Yee grid and its boxes, a device grid and its ports, the DDA dipole lattice, the BEM surface mesh,
the Mie or T-matrix multipole order, the RCWA harmonics, an FDE cross-section grid, the EME or BPM cells along a
device, the PWE k-path and plane waves. It calls each backend's own planning code (the code `estimate` and
`solve` call) and never runs a solver, so it is as cheap as `estimate`. GUIs use it to draw the discretisation,
and agents use it to check a set-up.

```python
import lightsmith as ls

out = ls.discretization(problem, method="dda")
out["summary"]        # {"dipoles": 2176, "spacing (nm)": 4.98, "dipoles per diameter": 16, ...}
out["payload"]        # {"points_nm": [[x, y, z], ...], "spacing_nm": 4.98, ...}
```

```bash
lightsmith discretization spec.json --method fdtd
```

The MCP server has the same call as the read-only tool `get_discretization(problem, method)`.

## The envelope

Every backend answers with the same keys:

| Key | Meaning |
|---|---|
| `backend`, `method`, `problem_kind` | The backend that `method` resolves to (as for `solve`) and the problem kind |
| `layout_kind` | Which payload follows (table below) |
| `summary` | Labelled key figures, in display order |
| `payload` | The geometry, in nm and world coordinates |
| `estimate` | The backend's `estimate` for the same problem (on the CPU) |
| `warnings` | The planning code's and the estimate's warnings |
| `error` | The library's refusal of the problem, or `null`. The geometry is still returned where it can be computed, for example for a graded FDTD grid with a Gaussian beam. |

A backend the problem does not suit raises, as `solve` would. Backends with nothing to discretise (TMM,
quasistatic, circuits) return `layout_kind: "none"` and the estimate's figures in `summary`. Plugins can answer
by overriding `Backend.discretization(problem, device)`.

## Payloads

| `layout_kind` | Backends | Payload |
|---|---|---|
| `yee_grid` | FDTD scattering, emission | `grid` (`kind`, `shape`, `cells`, `nodes_nm` per axis, min/max spacing, `max_ratio`), `domain_nm`, `pml`, `boxes` (TFSF, total- and scattered-field flux; emission: the radiated-power box), `regions` (from `grid: "auto"` and `mesh_override`), `symmetry` (mirror planes, `planes_nm`). The grid is whole; the run steps one mirror image of it (`estimate.cells`). |
| `unit_cell` | FDTD periodic, layer stacks | Cell nodes `x_nm`, `y_nm`, `z_nm` (depth below the top of the stack), `planes_nm` (reflection monitor, source, stack top and bottom, transmission monitor), `pml_cells`, layer tops and thicknesses |
| `device_grid` | FDTD and FDFD devices, inverse design | `nodes_nm` per axis, PML inner faces, `ports` (`axis`, `direction`, `source_nm`, FDTD `monitor_nm`, `window_nm`, `num_modes`), `two_d`; inverse design: `design_regions` (shape, extent, filter radius in cells) |
| `dipole_lattice` | DDA | `points_nm` (occupied dipoles, subsampled by `stride` beyond 20 000), `material_slot`, `spacing_nm`, `dipoles_per_diameter`, `grid`, `n_dipoles`, `extrapolate` |
| `surface_mesh` | BEM | `verts_nm`, `tris` (subsampled beyond 40 000), `tri_surface`, `mesh_size_nm`, `n_triangles`, `n_edges` |
| `multipoles` | Mie (`lightsmith-mie`, `miepython`), T-matrix | Mie: `n_max` and `size_parameter` per wavelength and the rule used; T-matrix: `lmax` per particle, `modes`, `lmax_global` |
| `harmonics` | RCWA | `harmonics` (m, n), `orders_max`, `layers` (each uniform in z: the problem's own layers) |
| `cross_section_grid` | FDE | `x_nm`, `y_nm`, spacings, `pml_cells`, `symmetry`, `boundary`, `num_modes` |
| `cells` | EME, BPM | `axis`, `lo_nm`, `hi_nm`, `cells` (`start_nm`, `stop_nm`; BPM: `steps`, `dz_nm`), the cross-section grid; EME: `modes_per_cell`; BPM: `step_nm` |
| `plane_waves` | PWE | `k_reduced`, `k_distance`, `k_labels`, per polarisation the solver, plane-wave count, matrix size, FFT grid and (dense) G indices |
| `none` | TMM, quasistatic, circuits | `note` |
