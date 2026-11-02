# Problem kinds

A problem is a pydantic model with a `kind` discriminator. That makes every problem plain JSON, and
`solve`, `estimate`, `compare`, the CLI and the MCP server all accept any kind. `lightsmith schema` (or
`lightsmith.problems.problem_json_schema()`) prints the JSON Schema of every kind.

| `kind` | Model | Describes | Result |
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

`parse_problem` reads any kind from a model, a dict or a JSON string. A dict without `kind` is read as a
`scattering` problem, which keeps 0.1 scattering specs valid.

## Shared building blocks

- **Materials** go wherever a `material`, `background`, `superstrate` or `substrate` field appears. See
  [Materials](../materials.md).
- **Wavelengths** are either a list of vacuum wavelengths in nm, or
  `{"start_nm": ..., "stop_nm": ..., "num": ...}`, which is evenly spaced and includes both ends.
- **2D profiles** are used for patterned layers, mode cross-sections and 2D crystals. They are written as
  `{"kind": ...}` with one of these kinds:
  - `rectangle` (`center_nm`, `size_nm`)
  - `circle` (`center_nm`, `radius_nm`)
  - `ellipse` (`center_nm`, `semi_axes_nm`, `angle_deg`)
  - `polygon` (`vertices_nm`)
- **3D geometries** are used for devices and 3D crystals:
  - `prism`: a 2D profile extruded from `z_min_nm` to `z_max_nm`, with an optional `sidewall_angle_deg`;
  - `box` (`center_nm`, `size_nm`);
  - `ball` (`center_nm`, `radius_nm`).
- **Structures** pair a geometry or profile with a material. Where structures overlap, the later one wins.
- **`options`** is a dict keyed by method or backend name. See [Conventions](../conventions.md#options).
