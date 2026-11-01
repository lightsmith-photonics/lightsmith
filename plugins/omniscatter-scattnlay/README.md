# omniscatter-scattnlay

An [omniscatter](../../README.md) backend for [scattnlay](https://github.com/ovidiopr/scattnlay), the
multilayer-sphere Mie code by Peña, Pal, Ladutenko and co-workers.

**License: GPL-3.0-or-later.** scattnlay is GPL-3 (or later), and this plugin imports it, so the plugin is
GPL too. It lives outside the MIT omniscatter core and is found through the `omniscatter.backends`
entry point. Installing it next to omniscatter does not change omniscatter's license. Redistributing
the combination does bring in GPL obligations.

## Install

```console
uv pip install ./plugins/omniscatter-scattnlay   # pulls scattnlay 2.4 from PyPI
omniscatter backends                             # lists "scattnlay" (method "mie")
```

scattnlay 2.4 ships wheels for CPython 3.11 and 3.12 on Windows and macOS-arm only. Other platforms build
it from the sdist, which needs a C++ compiler.

## What it does

- Backend `scattnlay`, method `mie`, for one `Sphere` or `LayeredSphere`. Layers are passed core first.
- Size parameters are x_i = 2π n_medium r_i / λ and the indices are relative, m_i = n_i / n_medium.
- Cross sections are `Q * π r_out²` in nm², so they match omniscatter's normalisation (r_eq = r_out).
- `orientation_average` is accepted, since a sphere looks the same from every direction.
- `far_field`: dC_sca/dΩ = (|S2|² cos²φ + |S1|² sin²φ) / k², with φ measured from the incident polarization.
- Sign convention: scattnlay uses exp(-iωt), so it absorbs for Im(m) > 0, the same as omniscatter. With
  scattnlay 2.4, m = 1.5 + 0.1i gives Q_abs > 0 and m = 1.5 − 0.1i gives Q_abs < 0. Indices are passed
  unconjugated.

Options (`Simulation.options["scattnlay"]`, strict):

- `multiprecision`: needs a scattnlay build with `scattnlay_mp`; the PyPI wheels do not have it.
- `n_max`
- `pec_layer`

From the method-level `options["mie"]`, `n_max` is honoured and other keys are ignored with a warning.

## Develop

```console
cd plugins/omniscatter-scattnlay
uv sync
uv run pytest
uv run omniscatter backends
```

The tests compare against `miepython` for plain spheres (agreement to about 1e-11 relative). They also check
that a layered sphere whose layers share one material reduces to Mie, and that the layer order is core
first. Finally, they check that the far field integrates to C_sca.

## Cite

O. Peña and U. Pal, Comput. Phys. Commun. 180, 2348 (2009); K. Ladutenko, U. Pal, A. Rivera and
O. Peña-Rodríguez, Comput. Phys. Commun. 214, 225 (2017).
