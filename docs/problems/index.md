# Problem kinds

Each kind of problem has its own typed model. `solve`, `estimate` and `compare` accept any of them, and
validate every field when the problem is built, so a mistake is reported before anything runs. The CLI
and the MCP server take the same problems saved to a file (see
[Saving a problem](../quickstart.md#saving-a-problem-and-using-the-command-line)).

| Kind | Model | Describes | Result |
|---|---|---|---|
| `scattering` | `lightsmith.Simulation` | Particles in a homogeneous medium under a plane wave | `Result` (cross sections, near and far field) |
| `layer_stack` | `lightsmith.problems.LayerStackProblem` | Planar multilayers | `SpectrumResult` (R, T, A, amplitudes) |
| `periodic` | `lightsmith.problems.PeriodicProblem` | 1D/2D-periodic patterned layers | `SpectrumResult` (with diffraction orders) |
| `mode` | `lightsmith.problems.ModeProblem` | Waveguide cross-sections | `ModeResult` (neff, ng, loss, fields) |
| `device` | `lightsmith.problems.DeviceProblem` | Devices with ports | `SParameterResult` |
| `bands` | `lightsmith.problems.BandProblem` | Photonic-crystal band structures | `BandResult` |
| `circuit` | `lightsmith.problems.CircuitProblem` | Netlists of compact models | `SParameterResult` |
| `optimization` | `lightsmith.inverse.OptimizationProblem` | Topology optimisation of a device | `OptimizationResult` |
| `emission` | `lightsmith.problems.EmissionProblem` | A dipole emitter near particles | `EmissionResult` (Purcell factor, efficiency, radiation pattern) |

`parse_problem` loads any kind from a saved file's text or a dict, and works out which kind it is. A
saved problem with no `kind` is read as `scattering`, so files from version 0.1 still load.
`lightsmith schema` (or `lightsmith.problems.problem_json_schema()`) lists every field of every kind.

## Shared building blocks

- **Materials** go wherever a `material`, `background`, `superstrate` or `substrate` field appears. See
  [Materials](../materials.md).
- **Wavelengths** are either a list of vacuum wavelengths in nm, or
  `WavelengthRange(start_nm, stop_nm, num)`, which is evenly spaced and includes both ends.
- **2D profiles** are used for patterned layers, mode cross-sections and 2D crystals
  (all in `lightsmith.problems`):
  - `Rectangle(center_nm, size_nm)`
  - `Circle(center_nm, radius_nm)`
  - `Ellipse(center_nm, semi_axes_nm, angle_deg)`
  - `Polygon(vertices_nm)`
- **3D geometries** are used for devices and 3D crystals:
  - `Prism`: a 2D profile extruded from `z_min_nm` to `z_max_nm`, with an optional `sidewall_angle_deg`;
  - `Box(center_nm, size_nm)`;
  - `Ball(center_nm, radius_nm)`.
- **Structures** pair a geometry or profile with a material. Where structures overlap, the later one wins.
- **`options`** is a dict keyed by method or backend name. See [Conventions](../conventions.md#options).
