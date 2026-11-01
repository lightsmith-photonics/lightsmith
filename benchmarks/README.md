# omniscatter cross-method benchmark

Scripts that regenerate every benchmark number and figure in the paper (`paper/`).
They measure accuracy against a reference and cost (wall time, peak memory) for every
installed backend, on CPU and GPU, across a ladder of resolutions.

| File | Purpose |
|---|---|
| `problems.py` | Generates the problem suites (`smoke`, `full`) as omniscatter specs → `suites/<suite>.json` (`--split` also writes one Simulation JSON per problem). |
| `run.py` | Runs a suite: problem × backend × device × variant × resolution. Discovers backends at run time. |
| `analyze.py` | Convergence fits, staircase noise, cheapest run per target accuracy (run automatically by `run.py`). |
| `plots.py` | Figures (matplotlib, not a project dependency). |
| `suites/` | Generated suites (checked in so the exact problems are reviewable). |
| `results/<timestamp>-<suite>/` | One directory per run. |

## Reproduce

From the repository root, in an environment set up with `uv sync --extra gpu --extra mcp`:

```bash
uv run --no-sync python benchmarks/problems.py --suite smoke --split   # optional: suites are built on the fly
uv run --no-sync python benchmarks/run.py --suite smoke --devices cpu,gpu
uv run --no-sync --with matplotlib python benchmarks/plots.py benchmarks/results/<dir> --paper
```

`--no-sync` keeps `uv` from touching the environment; drop it on a fresh clone.
`--paper` copies the PDFs into `paper/figures/`.

Useful options of `run.py`:

| Option | Meaning |
|---|---|
| `--suite smoke\|smoke-extended\|full\|path.json` | Suite to run. A suite may carry `options` (merged into every run, e.g. DDA `max_iter`). |
| `--devices cpu,gpu` | Devices (`cpu`, `gpu`, `cuda:N`, `torch-cpu`). GPU is skipped with a message if unavailable. Backends without GPU support run on CPU only. |
| `--methods mie,dda` | Restrict to methods or backend names. Default: every installed backend whose `supports()` accepts the problem. |
| `--problems sphere-au,dimer` | Restrict to problem ids (prefix match). |
| `--budget '{"gpu":{"max_dipoles":2e5}}'` | Override the suite's per-device budgets. |
| `--max-mem-gb 6` | Skip runs whose `estimate()` memory exceeds this. |
| `--extra-options '{"dda":{"precision":"double"}}'` | Options merged into every run (keyed by method or backend). |
| `--dry-run` | Plan and estimate only: writes `plan.csv` and `skips.csv`, solves nothing. |

## What a run records

`results/<timestamp>-<suite>/`:

| File | Content |
|---|---|
| `environment.json` | Command line, Python, OS, CPU, `omniscatter.doctor()` (torch/CUDA/GPU), backend list with versions and licences, git commit. |
| `suite.json` | The exact problems that were run. |
| `runs.csv` | One row per run: problem, backend, method, variant, device, framework, precision, resolution (semantic and as the backend option), status, wall time, solver runtime, peak memory and how it was measured, `estimate()` memory, dipoles, iterations, whether the run is the reference, relative errors (max and RMS over wavelengths) for C_ext, C_sca, C_abs, the headline error, warnings. |
| `runs.jsonl` | The same plus full spectra, the options passed, the `estimate()` output and full provenance. |
| `spectra.csv` | Long format: one row per run × wavelength, with the reference values alongside. |
| `references.csv` | The reference of each problem and its self-convergence. |
| `skips.csv` | Every backend/device/resolution not run, and why (unsupported geometry, not installed, budget). |
| `convergence.csv` | Per resolution ladder: fitted order p in err ∝ res^-p, staircase noise (std of log10 error about the fit, in decades), fraction of refinements that reduced the error. |
| `cheapest.csv` | Per problem, metric (headline, extinction only) and target accuracy (10%, 5%, 1%, 0.1%): the fastest run that meets it, excluding Mie and copies of the reference run on another device. |
| `summary.md` | Human-readable digest. |
| `figures/` | Written by `plots.py`. |

### Definitions

* **Reference.** Mie theory when a Mie backend supports the problem (homogeneous sphere; layered
  sphere with the native Mie backend). Otherwise the most accurate method present
  (T-matrix > DDA > FDTD) at its highest resolution in its default variant, preferring double
  precision, labelled `highest-resolution ...` in `runs.csv`/`references.csv`. Its
  `reference_self_change` (headline error of the next-lower resolution against it) tells you how
  far from converged it is. **For metals this reference can be poor; read it before trusting
  the errors of non-spherical problems.**
* **Headline error.** max over wavelengths of the relative error of C_ext and of C_abs (C_abs is
  dropped when the reference absorption is below 1e-6 of the extinction).
* **Resolutions** are semantic in the suite: `dipoles_per_diameter` (DDA), `mesh_nm` (FDTD),
  `multipole_order` (T-matrix). `run.py` maps them onto whatever option the installed backend's
  `*Options` model exposes (`RESOLUTION_ADAPTERS`), e.g. `mesh_nm` → `spacing_nm`. A backend with no
  matching option runs once with its defaults (logged in `skips.csv`).
* **DDA variants**: `center` (default), `weighted` (experimental), `extrapolate` (experimental;
  the resolution is the finest lattice of the ladder `[res/2, 3res/4, res]`).
* **Wall time** is measured around `omniscatter.solve` after a warm-up of each backend/device
  (JIT, CUDA context, cuFFT plans). GPU timings synchronise the device.
* **Peak memory**: `torch.cuda.max_memory_allocated` minus the allocation before the run on GPU;
  `tracemalloc` peak on CPU (NumPy buffers are traced). `tracemalloc` slows Python-level loops,
  so CPU wall times are slight overestimates.
* **Budgets**: each suite sets per-device limits on DDA dipoles (`max_dipoles`), FDTD cells
  (`max_cells`) and FDTD cells × time steps (`max_cell_steps`), checked with `estimate()` before
  solving. Skipped runs are listed in `skips.csv`.

## Suites

* **smoke** — 8 problems, one per family (dielectric sphere, Drude sphere with a dense DDA ladder,
  Au and Ag spheres with Johnson–Christy data, SiO2@Au core–shell, Au spheroid, Au rod, Au dimer),
  1–3 wavelengths, short ladders. Experimental DDA variants (`weighted`, `extrapolate`) only on the
  dielectric and Drude spheres; DDA `max_iter` 3000 (suite-level `options`). CPU budget 600 dipoles /
  1e9 FDTD cell-steps; GPU 60 000 dipoles / 1.2e11 cell-steps. 98 runs in 649 s (~11 min) on a Ryzen 5
  5600X + RTX 5070 shared with other jobs; most of it is the Drude ladder (capped `weighted` runs and
  2 nm FDTD). Kept run: `results/20261101-011025-smoke/`.
* **smoke-extended** — the first smoke definition: every DDA variant on every problem, `max_iter`
  10 000, CPU budget 1200 dipoles. 154 runs in 44 min, mostly `weighted` DDA stalling at the
  iteration cap. Kept run: `results/20261101-012216-smoke-extended/` (see its `NOTE.md`).
* **full** — 145 problems: dielectric spheres (x = 0.5–10, m = 1.2–3.5, lossless and lossy, vacuum
  and water), Au/Ag spheres (r = 10–100 nm, vacuum and water, 41 wavelengths), a dense Drude
  staircase study, 8 core–shell particles, Au/Ag spheroids (aspect ratio 1–5), rods (2–5) and
  dimers (gap 1–40 nm). Not yet run; see below.

### Running the full suite

Do a dry run first; it lists every planned run with its estimated size:

```bash
uv run --no-sync python benchmarks/run.py --suite full --devices cpu,gpu --dry-run
```

A partial dry run (2026-11-01, killed after 15 min) planned 1134 runs for the first 39 problems (dielectric spheres only; DDA up to 9.3e5 dipoles, plus FDTD, treams, Mie and quasistatic), so the whole suite is several thousand runs. The dry run itself is slow because DDA `estimate()` builds the full supersampled lattice for the `weighted` variant. Expect days on one consumer GPU, not hours: metal DDA needs ~10³ BiCGStab iterations per wavelength, the full suite has up
to 71 wavelengths per problem and lattices up to 2·10⁶ dipoles, and the GPU FDTD budget allows
runs of several minutes each. Recommended:

* run per family (`--problems sphere-au,sphere-ag`) and per device, each into its own directory;
* an idle machine (no other GPU jobs) for the timing figures;
* install `treams` so the T-matrix backend can serve as the reference for dimers;
* run `--extra-options '{"dda":{"precision":"double"}}'` on GPU once to separate precision from
  discretisation error.

## Adding a backend

Nothing here needs to change: a backend registered with `@register` (or via the
`omniscatter.backends` entry point) is discovered, filtered by `available()` and `supports()`,
sized with `estimate()` and solved. To take part in resolution sweeps its options model (a
pydantic class named `*Options` in the backend's module, or `Backend.options_model`) needs a field
listed in `RESOLUTION_ADAPTERS` in `run.py`; add a mapping there if it uses a new name.
