# Installation

Lightsmith needs Python 3.10 or newer. The core install is pure Python and depends only on NumPy, SciPy,
pydantic, array-api-compat and miepython.

## CPU

=== "uv"

    ```bash
    uv add lightsmith
    ```

=== "pip"

    ```bash
    pip install lightsmith
    ```

Check the installation:

```bash
lightsmith doctor     # array frameworks and GPUs that the solvers can see
lightsmith backends   # every solver backend, and whether it is available
```

## Optional extras

| Extra | Adds | Needed for |
|---|---|---|
| `gpu` | PyTorch (`torch>=2.7`) | GPU solvers, every gradient path, [inverse design](solvers/inverse.md) |
| `mcp` | The MCP Python SDK (`mcp>=2.3,<3`) | The [`lightsmith-mcp`](mcp.md) server |
| `tmatrix` | [treams](https://github.com/tfp-photonics/treams) (Python < 3.14) | The [T-matrix](solvers/tmatrix.md) backend |
| `layout` | [gdsfactory](https://gdsfactory.github.io/gdsfactory/) (Python ≥ 3.11) | Importing gdsfactory components and netlists. The built-in GDS reader needs no extra. |
| `xarray` | xarray | Labelled-array conversions |

Combine extras as needed:

```bash
uv add "lightsmith[gpu,mcp,tmatrix]"
```

!!! note "Licences of optional dependencies"
    gdsfactory is MIT, but recent versions pull in KLayout (GPL-3.0-or-later) through kfactory. For that
    reason it is optional and imported lazily.

## GPU

The GPU solvers run on PyTorch with CUDA. Install a CUDA build of PyTorch that matches your driver.

- **Linux**: the PyPI `torch` wheels already include CUDA, so `lightsmith[gpu]` is enough.
- **Windows**: the PyPI `torch` wheels are CPU-only. Take PyTorch from the PyTorch index instead.
- **RTX 50xx (Blackwell)**: needs CUDA 12.8 wheels (`cu128`) or newer.

=== "uv project"

    Add this to your own `pyproject.toml` so that `uv` resolves `torch` from the CUDA index:

    ```toml
    [project]
    dependencies = ["lightsmith[gpu]"]

    [tool.uv.sources]
    torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'win32' or sys_platform == 'linux'" }]

    [[tool.uv.index]]
    name = "pytorch-cu128"
    url = "https://download.pytorch.org/whl/cu128"
    explicit = true
    ```

=== "pip"

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install "lightsmith[gpu]"
    ```

Then check that the GPU is visible:

```bash
lightsmith doctor
```

The output should show the GPU under `"cuda_gpus"`, and `"auto_device"` should name `cuda:0`. With
`device="auto"`, which is the default, solvers that support a GPU use it, unless `estimate` says the problem
will not fit in the GPU's free memory: then the solve runs on the CPU and says so in `warnings`. Pass `device="cpu"` to force the
CPU, `device="cuda:1"` to pick a specific GPU, or `device="torch-cpu"` to run the PyTorch code path on
the CPU, for example to get gradients without a GPU.

## From a clone (development)

```bash
git clone https://github.com/lightsmith-photonics/lightsmith
cd lightsmith
uv sync --extra gpu --extra mcp    # pulls CUDA 12.8 PyTorch wheels on Windows and Linux
uv run lightsmith doctor
uv run pytest
```

To build this documentation locally:

```bash
uv run --group docs mkdocs serve
```

## GPL plugins

GPL-licensed solvers are separate packages. Once installed, they register through the
`lightsmith.backends` entry-point group, and `lightsmith backends` lists them:

- `lightsmith-adda`: [ADDA](https://github.com/adda-team/adda) discrete-dipole solver, run as an
  external executable.
- `lightsmith-scattnlay`: [scattnlay](https://github.com/ovidiopr/scattnlay) multilayer-sphere Mie solver.
