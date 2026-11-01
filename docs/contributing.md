# Contributing

Contributions are welcome: solver backends, validation tests, materials and documentation. Lightsmith is
MIT-licensed. By contributing, you agree that your contribution is released under the same licence.

## Development setup

```bash
git clone https://github.com/lightsmith-photonics/lightsmith
cd lightsmith
uv sync --extra gpu --extra mcp
uv run pytest                      # GPU tests are marked `gpu` and skip without CUDA
uv run pytest -m slow              # long convergence checks
uv run --group docs mkdocs serve   # documentation preview
```

## Ground rules

- **Conventions are fixed.** Use nm, vacuum wavelengths and `exp(-iωt)`, and normalise efficiencies by
  π r_eq². See [Conventions](conventions.md). Changing any of these needs a schema version bump.
- **Every backend is validated.** A new backend needs a test against Mie or another reference, with a
  stated tolerance. Add the numbers to the backend's page under [Solvers](solvers/index.md).
- **One code path for CPU and GPU.** Native solvers use the array API (`device.namespace()`) and never
  call `numpy` or `torch` directly in hot loops.
- **The core stays MIT.** GPL-licensed solvers must live in separate plugin packages.
- **Optional dependencies are imported lazily**, inside methods, so that a missing extra never breaks
  `import lightsmith`.

## Writing a backend

A backend wraps one implementation of one method. Subclass `lightsmith.backends.Backend`:

```python
from typing import ClassVar

from lightsmith.backends import Backend, register, unsupported_outputs
from lightsmith.device import Device
from lightsmith.results import Result
from lightsmith.spec import Simulation


@register
class MySolver(Backend):
    name: ClassVar[str] = "my-solver"            # unique implementation id
    method: ClassVar[str] = "dda"                # the method family it implements
    license: ClassVar[str] = "MIT"
    citation: ClassVar[str] = "Author et al., Journal (year)"
    gpu: ClassVar[bool] = False
    differentiable: ClassVar[bool] = False
    problem_kinds: ClassVar[tuple[str, ...]] = ("scattering",)

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            import my_dependency  # noqa: F401  (import optional dependencies lazily)
        except ImportError:
            return False, "my_dependency is not installed (pip install my_dependency)"
        return True, ""

    @classmethod
    def supports(cls, sim: Simulation) -> tuple[bool, str]:
        why = unsupported_outputs(sim, near_field=False, far_field=False)
        return (not why), why

    @classmethod
    def estimate(cls, sim: Simulation, device: Device) -> dict:
        return {"unknowns": 0, "memory_bytes": 0}     # cheap; never solves

    @classmethod
    def solve(cls, sim: Simulation, device: Device) -> Result:
        ...
```

| Member | Role |
|---|---|
| `available()` | Whether the dependencies are installed, and if not, why. It is checked before `supports`. |
| `supports(problem)` | Whether this backend can solve the problem, and if not, an actionable reason. The reason appears in `auto` selection errors. |
| `estimate(problem, device)` | A cheap pre-flight report of size and memory. Optional, but expected for grid-based solvers. |
| `solve(problem, device)` | Returns the result model for the problem kind: `Result`, `SpectrumResult`, `ModeResult`, `SParameterResult` or `BandResult`. Fill in `provenance` and `warnings`. |

Options: read `options_for(problem, cls.name, cls.method)` from `lightsmith.problems` and validate it with a
pydantic model that uses `extra="forbid"`.

To make `method="auto"` consider a new method, add it to `AUTO_ORDER` in `lightsmith/api.py`. Methods
not listed there are tried after the listed ones.

## Plugin packages

A backend can live in its own package. This is required for GPL-licensed solvers. Register it through
the `lightsmith.backends` entry-point group in the plugin's `pyproject.toml`:

```toml
[project]
name = "lightsmith-mysolver"
dependencies = ["lightsmith>=0.1"]

[project.entry-points."lightsmith.backends"]
mysolver = "lightsmith_mysolver:MySolver"
```

Once the plugin is installed, `lightsmith backends` lists it, and `solve`, `compare` and the MCP server
can use it. `lightsmith-adda` and `lightsmith-scattnlay` are working examples.

## Documentation

- The documentation lives in `docs/` and builds with MkDocs Material: `uv run --group docs mkdocs build --strict`.
- The API reference is generated from docstrings by mkdocstrings.
- New pages must be added to `nav` in `mkdocs.yml`; the strict build fails on pages missing from it.

## Releases

The maintainers' checklist is `RELEASING.md` in the repository root.
