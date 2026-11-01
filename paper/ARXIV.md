# arXiv submission metadata

## Title

Lightsmith: an open, GPU-native, differentiable photonics suite for humans and AI agents

## Authors

Alexander Feghali (affiliation: TODO)

## Categories

- Primary: **physics.optics**
- Cross-lists: **physics.comp-ph**, **cs.MS**

## Licence

TODO: choose the arXiv licence (e.g. CC BY 4.0) at submission.

## Abstract (plain text; keep in sync with main.tex)

Photonics simulation is split between closed commercial suites and a fragmented landscape of open single-method codes, each with its own input format, units, sign conventions and licence. This makes cross-checking methods laborious for people and error-prone for the AI agents that are increasingly asked to set up simulations. We present Lightsmith, an MIT-licensed Python suite in which a problem is described once, as a typed and JSON-serialisable specification covering seven problem kinds (particle scattering, planar layer stacks, periodic structures, waveguide modes, devices, photonic bands and circuits), and solved by any applicable backend on CPU or GPU. Native solvers are written once against the Python array API, so the same code runs on NumPy and on PyTorch with CUDA, and several are differentiable end to end. The suite includes Mie theory for multilayer spheres, quasistatic models, a T-matrix interface, an FFT-accelerated discrete dipole approximation, FDTD, transfer-matrix and rigorous coupled-wave analysis, mode solvers, frequency-domain inverse design, plane-wave band structures and S-matrix circuits, with a CC0 database of measured optical constants and passive Drude-Lorentz fitting. Every result carries provenance, and every operation is exposed through a Python API, a JSON command-line interface and a Model Context Protocol server, with a cheap estimate step before every expensive solve. We report validation of each solver against analytic or independent references and a scripted cross-method benchmark of scattering by dielectric and plasmonic particles, which quantifies how slowly and irregularly volume-discretisation methods converge for metals. TODO: headline numbers from the full benchmark and performance study.

(arXiv limit: 1920 characters. The TODO sentence must be replaced, and the total re-checked, before submission.)

## Comments field (suggested)

TODO: "N pages, M figures. Code and benchmark data: <repository URL>"

## Upload checklist

- [ ] Remove every `\todo{}` (delete the macro definition in `main.tex` so any leftover fails to compile).
- [ ] Author confirms the AI-assistance disclosure in the Acknowledgements.
- [ ] Replace placeholder smoke-suite figures and numbers with full-suite results; re-run
      `benchmarks/plots.py ... --paper`.
- [ ] Re-run `uv run --no-sync python paper/tools/make_bib.py` and check that nothing is commented out.
- [ ] Compile with **pdflatex** (arXiv's default TeX Live): `pdflatex main && bibtex main && pdflatex main && pdflatex main`. Fix all warnings about undefined references or citations.
- [ ] Upload `main.tex`, the generated **`main.bbl`** (arXiv does not run BibTeX reliably from `.bib`; include the `.bbl` whose name matches `main.tex`), and `figures/*.pdf`. Do not upload `references.bib` only.
- [ ] Keep figures small: vector PDFs, no embedded high-resolution rasters; the whole submission should stay well under arXiv's size limit (individual files and total; currently 50 MB total, smaller is processed faster). Check `figures/` size before upload.
- [ ] Files referenced with exact case (arXiv's file system is case-sensitive); no absolute paths.
- [ ] Check the arXiv-generated PDF preview before announcing.
- [ ] **Endorsement**: a first submission to physics.optics (or by a new account) may require endorsement by an established arXiv author in that category. Arrange this before the planned submission date.
- [ ] Fill the licence and the comments field.
