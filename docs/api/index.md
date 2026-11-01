# API reference

Generated from the docstrings. The most-used names are re-exported from the top-level package
(`import lightsmith as ls`):

- [Top level](core.md): `solve`, `estimate`, `compare`, `choose_method`, `doctor`
- [Problems](problems.md): every problem kind, plus `parse_problem` and `problem_json_schema`
- [Scattering spec](spec.md): `Simulation`, shapes, illumination, output requests
- [Materials](materials.md): material models, the bundled library, refractiveindex.info, Drude–Lorentz fits
- [Results](results.md): result models for every problem kind
- [Backends](backends.md): the `Backend` interface and registry
- [Circuits](circuits.md): compact models, netlists, Touchstone files
- [Inverse design](inverse.md): optimisation spec, driver, export
- [Layout](layout.md): GDS import and export, the gdsfactory bridge
