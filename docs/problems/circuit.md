# Circuits

Model `lightsmith.problems.CircuitProblem`. A netlist of components, each described by
an S-parameter model. Solving it gives the S-matrix of the whole circuit.

## Fields

| Field | Meaning |
|---|---|
| `instances` | Instance name → `Instance(component, settings)`. `component` is a built-in compact model (see [Circuit solver](../solvers/circuit.md#compact-models)) or `sparams:<file>`, which loads a Touchstone or JSON file. |
| `connections` | Pairs of `"instance,port"` |
| `ports` | External port name → `"instance,port"` |
| `wavelengths` | Vacuum wavelengths in nm |
| `options` | Per-method solver options |

## Result

`SParameterResult` over the external ports. `res.s()` gives the complex array indexed
`[wavelength][out][in]`.

## Example: Mach–Zehnder interferometer

```python
import lightsmith as ls
from lightsmith.problems import CircuitProblem

guide = {"neff": 2.4, "ng": 4.2, "wl0_nm": 1550.0, "loss_db_per_cm": 2.0}
mzi = CircuitProblem(
    instances={
        "split": {"component": "coupler", "settings": {"coupling": 0.5}},
        "combine": {"component": "coupler", "settings": {"coupling": 0.5}},
        "long": {"component": "waveguide", "settings": {"length_nm": 120_000.0, **guide}},
        "short": {"component": "waveguide", "settings": {"length_nm": 100_000.0, **guide}},
        "dump": {"component": "terminator"},
    },
    connections=[("split,o3", "long,o1"), ("long,o2", "combine,o2"),
                 ("split,o4", "short,o1"), ("short,o2", "combine,o1"),
                 ("split,o2", "dump,o1")],
    ports={"in": "split,o1", "bar": "combine,o4", "cross": "combine,o3"},
    wavelengths={"start_nm": 1500, "stop_nm": 1600, "num": 2001},
)
res = ls.solve(mzi)
s = res.s()
bar = abs(s[:, res.ports.index("bar"), res.ports.index("in")]) ** 2
```

The repository's `examples/` folder has full scripts for a Mach–Zehnder interferometer, a ring and a
splitter tree.
