# PDK flow

`lightsmith.pdk` takes photonic components from layout to circuit:

1. a **technology** describes the layer stack and its materials;
2. parametric **cells** build a layout (polygons on GDS layers) and its ports;
3. `flow.simulate` solves the component's [device problem](problems/device.md) with
   [`ls.solve`](api/core.md) and returns S-parameters at the component's ports;
4. `flow.component_model` fits those S-parameters, caches the fit on disk, and makes it available to
   [circuit](problems/circuit.md) netlists as `pdk:<cell>`.

It is a generic, open process design kit: every number in it comes from open data and textbook
dimensions, not from any foundry.

```python
import lightsmith as ls
from lightsmith.pdk import generic_soi, get_component
from lightsmith.pdk import flow

mmi = get_component("mmi1x2", length_mmi_nm=5400)
mmi.write_gds("mmi.gds", generic_soi())
res = flow.simulate(mmi, wavelengths=[1530, 1550, 1570])   # EME in 2D, at the port reference planes
print(res.ports, abs(res.s()[1, 1, 0]) ** 2)
```

## Technology

A `Technology` lists named layers, each with a GDS `(layer, datatype)`, a `kind` and a z range:

- `material`: polygons are filled with the layer's material between `z_min_nm` and `z_max_nm`;
- `etch`: polygons remove the core over the z range and are filled with the cladding;
- `drawing`: no optical effect (floorplan, labels).

Layers are listed in override order: where polygons overlap, later layers win. The technology also sets
the `cladding`, the buried oxide (`box`, below z = 0, `box_thickness_nm`), an optional `substrate`, the
`core_layer` that defines ports and mode solves, the default waveguide width and the polarisation family.
It is a pydantic model, so you can save it to a file and load it back: `load_technology("tech.json")` reads one,
`register_technology` makes it available by name, and `get_technology` accepts a name, a path, a dict or
a `Technology`.

`generic_soi()` is a generic 220 nm silicon-on-insulator stack:

| Layer | GDS | Kind | z (nm) |
|---|---|---|---|
| `SLAB` | (3, 0) | Si | 0–90 (rib slab) |
| `WG` | (1, 0) | Si (core layer) | 0–220 |
| `SHALLOW_ETCH` | (2, 0) | etch | 150–220 (70 nm, grating teeth) |
| `FLOORPLAN` | (99, 0) | drawing | — |

The cladding and the 2 µm buried oxide are SiO₂ (Malitson 1965) and the handle is Si. Silicon is the
bundled `Si/Li-293K` dataset (Li 1980), which covers 1.2–14 µm; both come from refractiveindex.info under
CC0. Default waveguide width 500 nm, smallest feature 100 nm, smallest bend radius 2 µm.

**2D and 3D.** Simulations are 2D plan views by default (`dimension=2`). Each patterned region then
becomes an effective-index material: the TE0 slab index of the vertical stack at that point
(`Technology.eim_material`). With `dimension=3`, polygons are extruded over their layers' z ranges.

## Cells

`get_component(name, **params)` validates the parameters and builds a `Component`. `list_components()`
returns every cell with its ports and the JSON Schema of its parameters. Lengths are in nm. Every cell
starts with port `o1` at the origin facing west and extends towards +x; port names follow gdsfactory and
the built-in [compact models](solvers/circuit.md#compact-models).

| Cell | Ports | Main parameters (defaults) | Solved by default with |
|---|---|---|---|
| `straight` | o1, o2 | `length_nm` 10000, `width_nm` 500 | mode solve |
| `bend_circular` | o1, o2 | `radius_nm` 5000, `angle_deg` 90 | FDFD |
| `bend_euler` | o1, o2 | as above, plus `p` 0.5 (fraction of the turn in clothoid sections) | FDFD |
| `taper` | o1, o2 | `length_nm` 10000, `width1_nm` 500, `width2_nm` 1000 | EME |
| `mmi1x2` | o1; o2 top, o3 bottom | `width_mmi_nm` 2500, `length_mmi_nm` 5400, `width_taper_nm` 1000, `gap_mmi_nm` 250 | EME |
| `mmi2x2` | o1, o2 west; o3, o4 east | `width_mmi_nm` 3000, `length_mmi_nm` 10500, `width_taper_nm` 800, `gap_mmi_nm` 200 | EME |
| `coupler` | o1, o2 west; o3, o4 east | `gap_nm` 200, `length_nm` 5000, `dx_nm` 8000, `dy_nm` 3000 | EME |
| `y_branch` | o1; o2 top, o3 bottom | `length_taper_nm` 2000, `length_branch_nm` 8000, `pitch_nm` 2000 | EME |
| `crossing` | o1 west, o2 north, o3 east, o4 south | `width_center_nm` 1200, `length_taper_nm` 500, `length_center_nm` 4500 | FDFD |
| `coupler_ring` | o1 bus west, o2, o3 ring, o4 bus east | `radius_nm` 5000, `gap_nm` 200, `length_x_nm` 0 | FDFD |
| `ring_single` | o1, o2 | `radius_nm` 5000, `gap_nm` 200 | circuit of its parts |
| `racetrack_single` | o1, o2 | as above, plus `length_x_nm` 5000 | circuit of its parts |
| `grating_coupler` | o1 waveguide, o2 fiber | `period_nm` 580, `fill_factor` 0.5, `num_periods` 20, `fiber_mfd_nm` 10400 | FDFD (2D side view) |
| `mzi` | o1, o2 | `delta_length_nm` 10000, `radius_nm` 3000, `splitter` `"mmi1x2"` | circuit of its parts |

For 2×2 devices, o1–o4 is the bar path and o2–o3 the cross path. The last column is what `method="auto"`
picks for the default 2D plan view: [EME](solvers/eme.md) when all ports lie on two opposite planes,
[FDFD](solvers/fdfd.md) otherwise (bends, crossings, ring couplers). The grating coupler is simulated as a
2D side view, with the fiber mode as port o2.

**Components and assemblies.** A `Component` holds polygons per layer, ports (`CPort`: name, centre,
orientation, width, layer) and, for composite cells, its instances and netlist.
`Component.write_gds(path, tech)` writes a hierarchical GDS, one cell per distinct component.
`Assembly` places components by connecting ports face to face (`add`, `connect`, `link`), names the
outer ports (`expose`), and `build` returns a composite component whose netlist the circuit solver can
evaluate.

## Simulation

`flow.simulate(component, tech=None, wavelengths=None, *, method="auto", device="auto", dimension=2,
options=None, direct=False, **params)` returns an `SParameterResult` with the component's port names and
order. `component` is a `Component` or a cell name plus `params`.

- **Straights** use a mode solve: S21 = exp(i k₀ n_eff L).
- **Composites** (`ring_single`, `racetrack_single`, `mzi`) solve the circuit of their parts.
- **Everything else**, and any component with `direct=True`, is built as a `DeviceProblem`
  (`Component.device_problem`) and solved with `ls.solve(problem, method, device)`. A device backend
  added to the `auto` order is used automatically when it accepts the problem.

**Reference planes.** The port centres are the S-parameter reference planes. Solver ports sit 100 nm
outside them, on straight input stubs that reach through the absorbing boundary, because FDFD's port
decomposition reads the lines just inside the port plane. `simulate` then moves S back onto the port
centres: with FDFD's own port-mode effective indices for FDFD results (so its numerical dispersion stays
out of the phase), and with the technology's mode index otherwise. The shift is recorded in
`provenance["reference_plane_shift_nm"]`.

## Compact models

`flow.component_model(cell, flow=None, *, refresh=False, **params)` simulates the cell over a band, fits
the S-parameters and returns a `FittedModel` with `evaluate(wavelengths_nm)` and `summary()`.

- **Fit.** A pole–residue model from vector fitting (Gustavsen & Semlyen 1999), after removing each
  entry's propagation delay, with poles kept in the causal half-plane. The pole count is the smallest that
  reproduces held-out samples (every other wavelength) to `fit_tol`. If none does, the model falls back
  to a table interpolated in magnitude and unwrapped phase. Straights from a mode solve keep a polynomial
  n_eff(λ) instead, so their phase is exact for any length.
- **Band.** Evaluating a model outside its simulated band raises an error that says to widen
  `wl_min_nm`/`wl_max_nm`.
- **Cache.** Models are stored as JSON under `LIGHTSMITH_PDK_CACHE`, or the Lightsmith cache directory's
  `pdk` folder. The key is a hash of the cell, its parameters, the technology, the wavelengths, the
  method, the dimension, the fit settings, the solver options and the library version. The device is not
  part of the key.

`FlowSettings`, which also become settings of every `pdk:` circuit instance:

| Setting | Default | Meaning |
|---|---|---|
| `technology` | `"generic_soi"` | Registered technology name or its JSON |
| `wl_min_nm`, `wl_max_nm` | `1500`, `1600` | Simulated band; the circuit must stay inside it |
| `num_points` | `11` | Simulated wavelengths across the band |
| `method` | `"auto"` | Device solver |
| `device` | `"auto"` | Where to simulate (not part of the cache key) |
| `dimension` | `2` | 2 = plan view with effective indices; 3 = full 3D |
| `waveguide_model` | `"mode"` | Straights: `"mode"` (mode solve) or `"device"` (a device solve with `method`, the same discretisation as a direct simulation of a larger layout) |
| `fit` | `"auto"` | `"auto"`, `"pole_residue"` or `"table"` |
| `fit_tol` | `5e-3` | Largest \|S_fit − S\| on held-out samples for a pole–residue fit |
| `sim_options` | `{}` | Per-method solver options, e.g. `{"fdfd": {"dl_nm": 25}}` |
| `cache` | `True` | Read and write the on-disk cache |

## Circuits

In a [circuit](problems/circuit.md) netlist, `pdk:<cell>` instantiates the fitted model of a cell. Its
settings are the cell's parameters plus any `FlowSettings`. A missing model is simulated on first use, and
later uses read the cache.

```python
from lightsmith.pdk import flow
from lightsmith.problems import CircuitProblem

band = {"num_points": 3}
prob = CircuitProblem(
    instances={"split": flow.instance("mmi1x2", band),
               "up": {"component": "pdk:straight", "settings": {"length_nm": 20_000.0, **band}},
               "down": flow.instance("straight", band, length_nm=10_000.0)},
    connections=[("split,o2", "up,o1"), ("split,o3", "down,o1")],
    ports={"in": "split,o1", "a": "up,o2", "b": "down,o2"},
    wavelengths={"start_nm": 1520, "stop_nm": 1580, "num": 4})
res = ls.solve(prob)
```

`flow.instance(cell, flow, **params)` validates the parameters before building the instance.
`flow.netlist_problem(component, wavelengths, flow)` turns a composite component, such as the `mzi` cell,
into a `CircuitProblem` of its parts.

`examples/pdk_mzi.py` runs the whole flow for an unbalanced MZI: GDS output, one simulation and fit per
distinct part, the circuit on a dense wavelength grid, and a direct FDFD simulation of the whole layout for
comparison.

!!! tip "Comparing a circuit with a direct simulation"
    Use the same solver and grid for the parts and for the whole layout:
    `FlowSettings(method="fdfd", waveguide_model="device", sim_options={"fdfd": {...}})`. At its default
    grid, FDFD's numerical dispersion is about 1e-2 in n_eff. With the default 4× subpixel sampling, where
    a waveguide's edges fall on the grid shifts n_eff by about 7e-3 between otherwise identical
    instances; use `"subsamples": 16`.

## Validation

All tests are in `tests/test_pdk.py`. Device simulations are 2D plan views of `generic_soi` at 1550 nm
unless stated.

| Test | Reference | Tolerance |
|---|---|---|
| `test_slab_effective_index_against_the_symmetric_slab_formula` | symmetric slab | 1e-9 |
| `test_cells_have_consistent_ports_and_write_gds`: every cell | GDS read back | polygon area 1e-3 (1 nm database grid) |
| `test_straight_phase_eme_vs_mode_solve`: 5 µm straight | mode solve | \|S21\| = 1 to 1e-4; phase 1e-2 rad |
| `test_fdfd_reference_planes_follow_the_component_not_the_grid`: straight shifted by half a cell | unshifted | 5e-3 rad (about 0.07 rad without the correction) |
| `test_mmi1x2_splits_50_50_with_low_loss` | symmetry | 50/50 to 1e-6; total > 0.97; reflection < 1e-3 |
| `test_mmi2x2_is_a_3db_90_degree_coupler` | 3 dB coupler | total > 0.92; imbalance < 1 dB; phase 90° ± 3° |
| `test_directional_coupler_follows_coupled_mode_theory` (slow) | supermode theory | change of coupling angle within 2 %; total > 0.97 |
| `test_bends_are_low_loss_and_euler_is_not_worse`: R = 3 µm | — | circular > 0.97; Euler no worse than circular − 2e-3 |
| `test_crossing_is_symmetric_with_low_crosstalk` | symmetry; reciprocity | through > 0.85; crosstalk < −20 dB; reciprocal 1e-3 |
| `test_y_branch_and_taper` | symmetry | Y-branch 50/50 to 1e-6, total > 0.9; 500 nm → 2 µm taper > 0.98 |
| `test_grating_coupler_peak_follows_phase_matching` (slow) | first-order Bragg condition | peak wavelength within 4 %; Σ\|S\|² < 1.02 |
| `test_fit_reproduces_samples_and_held_out_wavelengths`: MMI, 7 samples | direct EME between samples | transmission 5e-3; every entry 3e-2 and within 3× the fit's own held-out estimate |
| `test_vector_fit_recovers_a_known_pole_residue_model` | known model | fit error 1e-8; pole 1e-6 |
| `test_models_are_cached_by_parameter_hash` | — | second call reads the disk cache; other parameters or method give another key |
| `test_netlist_references_components_by_name_and_parameters` | product of the models | 1e-9; 20 µm straight phase 1e-2 rad |
| `test_ring_fsr_matches_the_group_index`: R = 3 µm | λ²/(n_g 2πR) | 2 % |
| `test_mzi_circuit_matches_direct_simulation` (slow): FDFD, 30 nm grid, 16× subpixel | direct FDFD of the whole MZI | \|S21\|² within 0.15 (measured 0.12) |

The MZI test docstring explains the remaining difference: the circuit forces the field through the
straight waveguide mode at every junction (about 0.03–0.06 rad and a few percent of power per arm), and
grid alignment still shifts n_eff by about 2e-3 between instances at 16× sampling. At 4× sampling the
mismatch was 0.19.

## Limits

- The default 2D plan view is an effective-index approximation of the 3D stack. `dimension=3` is
  supported but far more expensive.
- Only open data: `generic_soi` is not a foundry process, and its default cell dimensions were designed
  for it in the 2D effective-index model.
- A fitted model is valid only inside its simulated band.
- Ports carry the fundamental mode of the core-layer waveguide of their width.
