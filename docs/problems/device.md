# Devices

Model `lightsmith.problems.DeviceProblem`. A photonic device, such as a waveguide,
coupler, splitter or bend, characterised by its S-matrix between ports.

## Fields

| Field | Meaning |
|---|---|
| `dimension` | `3` (default), or `2` for a device in the xy plane that is invariant along z |
| `structures` | 3D structures (`prism`, `box`, `ball` with a material). Later structures override earlier ones. |
| `layout` | Optional `GdsLayout(path, cell, layer_map)`: each GDS `"layer/datatype"` is extruded with a material between `z_min_nm` and `z_max_nm`. Layout structures are added after `structures`, so they win where the two overlap. |
| `background` | Cladding material |
| `domain_center_nm`, `domain_size_nm` | The simulation domain |
| `ports` | `Port(name, center_nm, size_nm, direction, num_modes)`. `size_nm` is zero along the port normal; `direction` is the direction of propagation into the device. |
| `wavelengths` | Vacuum wavelengths in nm |
| `options` | Per-method solver options |

## Result

`SParameterResult`:

- `ports`: multimode ports get an `@<mode>` suffix, for example `o1@0`;
- `s_real` and `s_imag`, indexed `[wavelength][out][in]`;
- `.s()`, which returns the complex array.

## Solvers

`auto` order: [EME](../solvers/eme.md), [BPM](../solvers/bpm.md), [FDFD](../solvers/fdfd.md), then
[FDTD](../solvers/fdtd.md).

- EME needs every port on one of two opposite planes, and works in 2D and 3D.
- BPM has the same port requirement and works in 2D and 3D, on the CPU or a GPU. It is one-way (no
  reflections) and suits long, slowly varying devices. Since EME accepts the same devices, ask for BPM
  with `method="bpm"`.
- FDFD works in 2D and 3D, runs on a GPU, and gives adjoint gradients. It is the engine behind
  [inverse design](optimization.md).
- FDTD works in 3D and 2D with ports anywhere and in any orientation, and runs on a GPU. One broadband run per
  port mode gives that column of S at every wavelength. Its port modes are exact for its grid, so a straight
  guide transmits with |S21| = 1.

## Layout import

- GDS files are read by a built-in reader, with no extra dependency. It handles BOUNDARY, BOX and PATH
  elements and flattens SREF and AREF references.
- With the `layout` extra, `lightsmith.layout.from_gdsfactory_component` turns a gdsfactory component into
  a `DeviceProblem`, including its ports.
- `from_gdsfactory_netlist` turns a gdsfactory netlist into a [circuit](circuit.md).

## Example

```python
import lightsmith as ls
from lightsmith.problems import DeviceProblem, Port, Prism, Rectangle, Structure

si, sio2 = ls.ConstantIndex(n_real=3.48), ls.ConstantIndex(n_real=1.44)
guide = DeviceProblem(
    dimension=2, background=sio2,
    structures=[Structure(geometry=Prism(profile=Rectangle(size_nm=(6000, 500)), z_min_nm=-110, z_max_nm=110),
                          material=si)],
    domain_size_nm=(4000, 3000, 0),
    ports=[Port(name="o1", center_nm=(-1500, 0, 0), size_nm=(0, 2000, 1), direction="+"),
           Port(name="o2", center_nm=(1500, 0, 0), size_nm=(0, 2000, 1), direction="-")],
    wavelengths=[1550],
)
res = ls.solve(guide, method="fdfd")
abs(res.s()[0, 1, 0]) ** 2     # |S21|²
```
