# Command line

The `lightsmith` command wraps the Python API. It is built for scripts and agents:

- every command prints JSON to stdout;
- every failure prints `{"error": "<ExceptionType>", "message": "..."}` to stderr and exits with code 1.

```text
lightsmith doctor
lightsmith backends
lightsmith schema
lightsmith estimate SPEC [--method auto] [--device auto]
lightsmith solve    SPEC [--method auto] [--device auto] [-o result.json]
lightsmith compare  SPEC [--methods mie,dda] [--device auto] [-o comparison.json]
```

| Command | Output |
|---|---|
| `doctor` | NumPy, PyTorch and CuPy versions, the CUDA GPUs, and the device `auto` would pick |
| `backends` | One entry per backend: `name`, `method`, `problem_kinds`, `license`, `gpu`, `differentiable`, `available`, `unavailable_reason`, `citation` |
| `schema` | The JSON Schema of every problem kind |
| `estimate` | Backend, device, problem size and memory, without solving |
| `discretization` | What the backend will discretise: grid, mesh, dipole lattice, harmonics, cells ([Discretisation](discretization.md)) |
| `solve` | The result model as JSON |
| `compare` | `reference`, `results` (one per method) and `relative_difference` |

`SPEC` is a JSON file of any [problem kind](problems/index.md), or `-` to read from stdin.

| Option | Values |
|---|---|
| `--device` | `auto`, `cpu`, `gpu`, `cuda:N` or `torch-cpu` |
| `--method` | `auto`, a method such as `dda`, or a backend name such as `lightsmith-mie` (see [Solvers](solvers/index.md)) |
| `--methods` | Comma-separated methods or backend names. Default: every backend that applies. |

## Examples

```bash
lightsmith estimate examples/drude_sphere.json --method dda --device gpu
lightsmith solve examples/drude_sphere.json -o result.json
cat spec.json | lightsmith solve - --method rcwa
lightsmith compare examples/drude_sphere.json --methods mie,dda,fdtd
```

The [MCP server](mcp.md) exposes the same operations to AI agents.
