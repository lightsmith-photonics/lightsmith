# Solvers

Each solver is a *backend*, which wraps one implementation of one method. `lightsmith backends` lists them
all, with licence, GPU support, differentiability and availability. You can choose a backend in three ways:

- `method="auto"` (the default) picks the first backend in the problem kind's preference order that is
  installed and accepts the problem. If none does, the error lists why each one was skipped.
- `method="<method>"`, for example `"dda"`, picks the first *available* backend for that method. It does
  not check whether that backend accepts the problem.
- `method="<backend name>"`, for example `"lightsmith-mie"`, picks exactly that implementation.

| Method | Backend name(s) | Problem kinds | GPU | Gradients | Dependency |
|---|---|---|---|---|---|
| [`mie`](mie.md) | `miepython`, `lightsmith-mie` | scattering | native | native (functional API) | miepython (core) |
| [`tmatrix`](tmatrix.md) | `treams` | scattering | — | — | treams (`tmatrix` extra) |
| [`bem`](bem.md) | `lightsmith-bem` | scattering | ✅ | — | — |
| [`dda`](dda.md) | `lightsmith-dda` | scattering | ✅ | — | — |
| [`fdtd`](fdtd.md) | `lightsmith-fdtd` | scattering, periodic, layer_stack | ✅ | — | — |
| [`quasistatic`](quasistatic.md) | `lightsmith-quasistatic` | scattering | — | — | — |
| [`tmm`](tmm.md) | `lightsmith-tmm` | layer_stack | ✅ | ✅ | — |
| [`rcwa`](rcwa.md) | `lightsmith-rcwa` | periodic, layer_stack | ✅ | ✅ | — |
| [`fde`](fde.md) | `lightsmith-fde` | mode | — | — | — |
| [`eme`](eme.md) | `lightsmith-eme` | device | — | — | — |
| [`bpm`](bpm.md) | `lightsmith-bpm` | device | ✅ | — | — |
| [`fdfd`](fdfd.md) | `lightsmith-fdfd` | device (2D) | ✅ | ✅ (adjoint) | — |
| [`pwe`](pwe.md) | `lightsmith-pwe` | bands | ✅ | dense path | — |
| [`circuit`](circuit.md) | `lightsmith-circuit` | circuit | ✅ | ✅ | — |
| [`inverse`](inverse.md) | `lightsmith-inverse` | optimization | ✅ | — | PyTorch (`gpu` extra) |

Every backend is MIT-licensed. GPL solvers ship as separate plugin packages (see
[Installation](../install.md#gpl-plugins)).

## Automatic order

| Problem kind | `auto` tries, in order |
|---|---|
| scattering | mie → tmatrix → bem → dda → fdtd (then any other method, such as quasistatic) |
| layer_stack | tmm → rcwa → fdtd |
| periodic | rcwa → fdtd |
| mode | fde |
| device | eme → bpm → fdfd |
| bands | pwe |
| circuit | circuit |
| optimization | inverse |

For scattering by a single plain sphere with no field outputs, `auto` picks `miepython`. For a layered
sphere, or when near or far fields are requested, it picks `lightsmith-mie`. Non-spherical particles, and
clusters that T-matrix does not take, go to BEM unless there is a substrate, which BEM does not support.

For devices, BPM accepts the same port layouts as EME, so `auto` reaches it only when EME is unavailable or
refuses the problem. Ask for it with `method="bpm"`.

!!! tip "`method=\"mie\"` and layered spheres"
    `method="mie"` resolves to the first installed Mie backend, which is `miepython`, and miepython
    handles only homogeneous spheres. For layered spheres or field outputs, pass
    `method="lightsmith-mie"` or `method="auto"`.

## How each solver page is organised

- **Method**: what is actually implemented.
- **Options**: every key of `options[<method>]`, with its default.
- **Validation**: the accuracy checks in the test suite (`tests/`). The tolerances are the asserted
  bounds; "measured" figures come from comments and docstrings in the tests.
- **Limits**: what the solver refuses or warns about.

GPU-only and slow tests are marked; they run with `uv run pytest -m gpu` and `-m slow`.
