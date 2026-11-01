# Landscape survey: open-source EM scattering solvers (Mie, T-matrix, DDA/GDM, FDTD)

*Compiled 2026-11-01 for PlasmonHub, an MIT-core, uv/pip-installable Python library with CPU and GPU support.*

## How the facts were gathered

| Tag | Source and method |
|---|---|
| **[PyPI]** | PyPI JSON API (`https://pypi.org/pypi/<pkg>/json`), queried 2026-11-01 for latest version, upload date, license field, wheel tags and `requires_dist`. |
| **[GH]** | GitHub REST API: `pushed_at`, SPDX license, stars, latest release. |
| **[GL]** | GitLab REST API, the same fields. |
| **[doc]** | The project's own README, docs or paper page, fetched on the survey date. |
| **unverified** | Could not be confirmed from a primary source. |

**Wheel tags.** `win` means a native Windows `win_amd64` wheel. `any` means a pure-Python wheel, which installs anywhere, including with `uv`. `sdist` means a source build; for Fortran or C++ packages that needs a local compiler, gfortran in particular, which is painful on Windows.

**uv.** Anything on PyPI with a matching wheel installs with uv. The caveats are PyTorch CUDA builds, which need the PyTorch package index, and conda-only packages.

---

## 0. Executive summary

1. **No direct competitor exists.** No maintained package puts Mie, T-matrix, DDA and FDTD behind one problem spec. The closest partial overlaps:
   - **HoloPy** (GPL-3) wraps Mie, multisphere T-matrix and ADDA, but it targets holography, and its last release was 2021.
   - **TorchGDM** (GPL-3) combines GDM with Mie and T-matrix effective models via treams.
   - **Tidy3D** is cloud-only for its solver.
   - **dda-bench** (GPL-3, 2026) cross-verifies three DDA codes only.
2. **A permissive GPU + autodiff FFT-DDA is a real gap.**
   - TorchGDM: GPL-3, dense matrices, about 10–15k dipoles, homogeneous environment only.
   - pyGDM2: GPL-3. It uses CuPy only for dense inversion.
   - CPDDA has CuPy FFT-DDA, but no stated license, no PyPI package and no autodiff.
   - ADDA and DDSCAT are GPL executables.
3. **The GPU-strategy decider: JAX still has no official native-Windows CUDA support** (late 2026). The `jax-cuda12/13-plugin` wheels are Linux-only, and WSL2 GPU is "experimental". The only native route is the unofficial `winjax` beta. That beta is single-GPU and cannot share a process with PyTorch.
   - PyTorch ships native Windows CUDA wheels; torch 2.15.0 has cu132 and cu134 builds. **PyTorch is therefore the only mainstream autodiff GPU stack that works on the Windows dev machine.**
   - CuPy also has Windows wheels, but it has no autodiff.
4. **Several "standard" backends have packaging rot. These are good upstream-contribution targets.**
   - **smuthi** (MIT) hard-pins `numpy<2` and `scipy<1.14`. It has a hard dependency on GPL-3 `pywigxjpf`, and it ships wheels only for CPython 3.8–3.10.
   - **pytmatrix** cannot build on Python ≥3.12 because it uses `numpy.distutils`.
   - **scattnlay** is v2.4 on PyPI (cp311/cp312, Windows and macOS-arm wheels only), while the README advertises v2.5.
   - **Meep** is still conda-only, with no native Windows support.
   - **pyMSTM** has Linux wheels only.
5. **FDTDX** (MIT, JAX) gained Drude/Lorentz dispersive materials only in May 2026 (v0.6.2). It has no published plasmonic validation, so our Mie-referenced benchmark would be a genuine service to that community.

---

## 1. Mie theory (sphere and layered-sphere ground truth)

| Library | License | Latest release | Activity | Wheels / uv | GPU | Autodiff | Computes | Notes and limitations | Citation |
|---|---|---|---|---|---|---|---|---|---|
| **miepython** ([GH](https://github.com/scottprahl/miepython), [docs](https://miepython.readthedocs.io)) | MIT [PyPI] | 3.3.1, 2026-10-28 [PyPI] | Very active; push 2026-10-28 [GH] | `any`, pure Python plus optional numba; uv OK on all OSes; Python ≥3.11 | No | No | Q_ext, Q_sca, Q_back, g, angular intensities, amplitude functions, Mueller matrices, near-field E/H [doc] | Homogeneous spheres only; no coated or layered spheres listed [doc]. Numba JIT gives 10–50× [doc]. | Prahl, Zenodo DOI 10.5281/zenodo.7949263 [doc] |
| **scattnlay** ([GH](https://github.com/ovidiopr/scattnlay)) | **GPL-3** [GH] | PyPI 2.4, 2024-04-10. The README cites 2.5 (Dec 2025), which is not on PyPI [PyPI][doc]. | Moderate; push 2026-01 [GH] | Win and macOS-arm wheels for cp311/cp312 only; Linux and other Pythons need an sdist (C++) build | No | No | Multilayer spheres; near and internal fields; optional Boost multiprecision; SIMD [doc] | GPL, so optional plugin only. The best reference for multilayer and near fields; arbitrary precision is useful for a ground-truth audit. | Peña & Pal, CPC 180, 2348 (2009); Ladutenko et al., CPC 214, 225 (2017) [doc] |
| **PyMieScatt** ([GH](https://github.com/bsumlin/PyMieScatt)) | MIT | 1.8.1.1, **2020-12-30** [PyPI] | GitHub push 2025-09, but no release since 2020 | sdist only, pure Python, so uv builds it | No | No | Homogeneous and core-shell spheres; inverse Mie; size distributions | Stale on PyPI. The Tidy3D docs use it as their Mie reference. | Sumlin et al., JQSRT 205, 127 (2018) (citation from memory, unverified) |
| **PyMieSim** ([GH](https://github.com/MartinPdeS/PyMieSim)) | MIT [PyPI] | 6.1.6, 2026-10-27 [PyPI] | Very active, about 120 releases | Win, Linux-x86_64 and macOS-arm wheels, cp311–cp314 | No | No | Spheres, core-shell, infinite cylinders; cross sections, far field, detector coupling, near-field example [doc] | **Hard-pins `numpy==2.2.6`, `matplotlib==3.10.9` and pandas [PyPI requires_dist]**, which is an environment-conflict hazard as a dependency. | Poinsinet de Sivry-Houle et al., Opt. Continuum 2(3), 520 (2023), doi:10.1364/OPTCON.473102 [doc] |
| **PyMieDiff** (`pymiediff`, [GH](https://github.com/UoS-Integrated-Nanophotonics-group/MieDiff)) | **GPL-3** [PyPI][GH] | 0.13, 2026-10-06 [PyPI] | Active (new in 2025) | `any`; Linux and Windows, Python 3.10–3.13 [doc] | Yes, PyTorch (CUDA) | **Yes**, torch autograd | Multilayer spheres; Q_sca, Q_ext; angular scattering; internal and near fields; refractiveindex.info YAML [doc] | GPL, so optional plugin. GPU pays off only for thousands of concurrent evaluations [doc]. | APL Photonics 11, 046114 (2026), doi:10.1063/5.0316814 [doc]; arXiv:2512.08614 |
| **treams sphere solver** ([GH](https://github.com/tfp-photonics/treams)) | MIT | 0.4.7, 2026-07-16 [PyPI] | Active; push 2026-10-16 [GH] | Win, Linux-x86_64 and macOS-arm wheels, cp39–cp313; **`<3.14`**; pins `scipy<1.17` [PyPI] | No | No | Sphere T-matrix in helicity or parity basis; multilayer spheres via `TMatrix.sphere` with several radii (from memory, **unverified**) | Gives a T-matrix-native Mie for consistency checks against the T-matrix backend. | Beutel, Fernandez-Corbaton, Rockstuhl, CPC 297, 109076 (2024) [doc] |
| **pymiecs** ([GL](https://gitlab.com/wiechapeter/pymiecs)) | GPL-3+ [PyPI] | 0.5, 2025-12-16 | Low activity | sdist (pure Python) | No | No | Core-shell spheres | Used as a test dependency of pyGDM2 and TorchGDM. | — |
| **MiePy** (`miepy`, [GH](https://github.com/johnaparker/miepy)) | **GPL-3** [GH] | 1.1.0, 2026-03-18 [PyPI] | Revived in 2026 | Win, Linux and macOS (x86 and arm) wheels, cp311–cp313 | No | No | GMMT clusters, i.e. multi-sphere T-matrix; also non-spherical particles via EBCM | Strictly a cluster T-matrix code (see §2). GPL. | — |

**Takeaway.** Mie is cheap, so it needs no GPU. The CPU ground truth should be **miepython** (MIT) for homogeneous spheres and **treams** (MIT) for layered spheres and T-matrix consistency. **scattnlay** (GPL) is useful as an optional high-precision cross-check. An MIT, PyTorch-batched, differentiable Mie is a small piece of code that does not exist permissively today. PyMieDiff is GPL, so writing our own is a cheap win.

---

## 2. T-matrix / multiple-scattering codes

| Library | License | Latest release | Activity | Wheels / uv | GPU | Autodiff | Computes | Notes and limitations | Citation |
|---|---|---|---|---|---|---|---|---|---|
| **treams** ([GH](https://github.com/tfp-photonics/treams), [docs](https://tfp-photonics.github.io/treams)) | MIT | 0.4.7, 2026-07-16 | Active (44 stars, 42 open issues) | Win, Linux-x86_64 and macOS-arm wheels, cp39–cp313 | No [doc] | No [doc] | Spherical- and cylindrical-wave T-matrices; clusters; 1D, 2D and 3D lattices; stratified media (S-matrices); band structures; helicity and parity bases; chiral materials [doc] | No arbitrary-shape (EBCM) T-matrix generator. Pins `python<3.14` and `scipy<1.17`. Same group (KIT) as the T-matrix HDF5 format and database. | CPC 297, 109076 (2024) |
| **smuthi** ([GL](https://gitlab.com/AmosEgel/smuthi), [docs](https://smuthi.readthedocs.io)) | MIT (but see notes) | 2.2.4, 2025-10-23 [PyPI] | Active on GitLab (2026-10-30) [GL]; master moving to a Meson build | 2.2.4 wheels are **Linux cp38–cp310 only**. 2.2.2 and 2.2.3 had Windows cp38–cp310 wheels. Anything newer means an sdist build with gfortran [PyPI]. | Optional PyCUDA (`extra cuda`), Linux-oriented | No | Multiple particles near or between planar interfaces; spheres, spheroids, cylinders and polyhedra via NFM-DS (Fortran); near and far fields; cross sections [doc] | **Hard dependencies: `numpy<2.0`, `scipy<1.14`, and `pywigxjpf` (GPL-3)**. Windows uses `pywigxjpf-win`, which only has cp36–cp39 wheels. Not installable cleanly in a modern Python 3.12 or 3.13 environment. | Egel et al., JQSRT 273, 107846 (2021), arXiv:2105.04259 |
| **pytmatrix** ([GH](https://github.com/jleinonen/pytmatrix)) | MIT | 0.3.3, 2023-05-10 | Dormant | sdist zip only. **`setup.py` uses `numpy.distutils` [GH source], so it does not build on Python ≥3.12.** | No | No | Mishchenko EBCM for spheroids, cylinders and Chebyshev particles; orientation averaging; radar quantities | Effectively uninstallable on modern Python without a Meson port. | Leinonen, Opt. Express 22, 1655 (2014) (from memory, unverified) |
| **MSTM v4** ([GH](https://github.com/dmckwski/MSTM)) and **pyMSTM** ([GH](https://github.com/arunoruto/pyMSTM)) | MSTM: MIT [GH]. pyMSTM: MIT, with bundled GCC runtime (GPL-3 + GCC exception, LGPL) [PyPI] | MSTM tag v4.0.0 (push 2024-02). pyMSTM 2.1.0, 2026-10-15. | MSTM slow; pyMSTM new (2026, 0 stars) | pyMSTM: **Linux-x86_64 wheels only**, cp311–cp314; other OSes build from sdist with gfortran [doc]. MSTM itself ships Windows .bat build scripts and MS-MPI. | No | No | Sphere clusters, spheres in layered media, periodic lattices, random orientation, T-matrices, Mueller matrices [pyMSTM README] | Spheres only. The MSTM v4 paper was not located (unverified). | Mackowski & Mishchenko, JQSRT 112, 2182 (2011) |
| **TERMS** ([GH](https://github.com/nano-optics/terms)) | **MPL-2.0** [GH] (file-level copyleft; can be used as an external executable) | v1.2.0, 2026-05-05 [GH] | Low but alive | Fortran source; not on PyPI | No | No | Superposition T-matrix for particle clusters; near and far field; orientation-averaged cross sections | Not a Python package; it would need a subprocess wrapper. | Schebarchov et al., arXiv:2111.12556 (user guide) |
| **SMARTIES** ([GH](https://github.com/nano-optics/smarties)) | **CC BY-NC 4.0, not OSI** [search; GH says NOASSERTION] | v1.2.0, 2024-07 | Dormant | MATLAB | No | No | Spheroids, with high accuracy at large aspect ratios | Exclude: the non-commercial license is incompatible. Use it only as a published reference. | Somerville, Auguié, Le Ru, JQSRT 174, 39 (2016); arXiv:1511.00798 |
| **CELES** ([GH](https://github.com/disordered-photonics/celes)) | BSD-3 [doc] | v2.2, 2021-04-13 | Stale (push 2023-06) | MATLAB + CUDA MEX | **CUDA**, via MATLAB; tested on Windows and Linux | No | Very large sphere ensembles (10k+ spheres) | Requires MATLAB, so not pip-installable. The only GPU T-matrix of note. | Egel et al., JQSRT 199, 103 (2017) [doc] |
| **MiePy** | GPL-3 | 1.1.0, 2026-03 | Revived | Wheels on all OSes | No | No | GMMT clusters, EBCM non-spherical particles, optical forces | GPL, so optional plugin. | — |
| **ScatterPy** | GPL-3+ | 0.1.1, 2020 | Dead | `any` | — | — | T-matrix | Ignore. | — |

**Community infrastructure.**
- **T-matrix HDF5 data format:** Asadova et al., JQSRT 333, 109310 (2025), [arXiv:2408.10727](https://arxiv.org/abs/2408.10727).
- **T-matrix database ("Daphona"):** [arXiv:2602.02101](https://arxiv.org/abs/2602.02101), [tmatrix.scc.kit.edu](https://tmatrix.scc.kit.edu/).

We should read and write the HDF5 format natively. It is the obvious interchange format between our T-matrix, DDA and FDTD backends: compute a T-matrix with DDA or FDTD, then compare it against treams.

**GPU / autodiff T-matrix.** No maintained, pip-installable GPU or differentiable T-matrix code was found. A 2024 paper built a JAX T-matrix model for time-varying nanostructures ([arXiv:2409.04551](https://arxiv.org/pdf/2409.04551)), but no package was released (unverified). This is a secondary gap: T-matrix cost is usually dominated by small dense solves, so the CPU is fine.

---

## 3. DDA / coupled-dipole / Green dyadic method

| Library | License | Latest release | Activity | Install | GPU | Autodiff | Computes | Notes and limitations | Citation |
|---|---|---|---|---|---|---|---|---|---|
| **ADDA** ([GH](https://github.com/adda-team/adda)) | **GPL-3** | v1.4.0, **2020-12-15** (asset `adda-1.4.0-win64.zip`) [GH] | Repo still active (push 2026-03-30, 157 open issues) | Prebuilt Windows binaries; build from source on Linux and macOS; **not on PyPI** (the PyPI `adda` package is unrelated) | **OpenCL** (works on Windows) [doc] | No | FFT-accelerated DDA; MPI; particles near a substrate; orientation averaging; many polarizability prescriptions (LDR, FCD, IGT); near fields | Accuracy reference for DDA. The executable can be called via subprocess as an optional plugin. | Yurkin & Hoekstra, JQSRT 112, 2234 (2011) [doc] |
| **DDSCAT** ([page](https://www.astro.princeton.edu/~draine/DDSCAT.7.3.html)) | **GPL** (v2 or v3 is not clear; unverified) | 7.3.3 per Draine's page; date unverified | Effectively frozen | Fortran source tarball | No | No | DDA with FFT; periodic targets (1D and 2D); orientation averaging; near fields | The historic standard. ScatPy (GPL-3, 2014) is a dead Python wrapper. | Draine & Flatau user guide, [arXiv:1305.6497](https://arxiv.org/abs/1305.6497) |
| **IFDDA** ([page](https://www.fresnel.fr/perso/chaumet/ifdda.html)) | Unverified | Unverified | Active (used in the 2026 dda-bench paper) | Fortran with a Qt GUI; hosted on GitLab | GPU per the dda-bench paper (unverified for IFDDA itself) | No | DDA; IF-DDAM variant for layered substrates | — | Chaumet et al. (see dda-bench) |
| **pyGDM2** ([GL](https://gitlab.com/wiechapeter/pyGDM2)) | **GPL-3+** | 1.1.12, 2025-11-20 [PyPI] | Moderate (2026-04) [GL] | `any` (numba-based now); uv OK on all OSes; pins `numpy<=2.3` | Optional **CuPy** dense matrix inversion (Windows OK) | No | GDM / coupled dipole; substrate via mirror-dipole approximation; 2D and 3D; near and far fields; spectra; EELS/CL; evolutionary optimization | Dense matrices, so N is limited to around 10⁴. | Wiecha, CPC 233, 167 (2018) (from memory); arXiv:1802.04071 |
| **TorchGDM** ([GL](https://gitlab.com/wiechapeter/torchgdm), [paper](https://scipost.org/SciPostPhysCodeb.60)) | **GPL-3+** [PyPI][GL] | 0.61, 2026-10-26 [PyPI] (GitLab tag v0.62) | Very active | `any`; Linux, Windows and macOS; `torch>=2.6` | **PyTorch** (CUDA, so native Windows works) | **Yes**, full autograd (except the treams and T-matrix import paths) | 3D and 2D GDM; mixes full discretization with effective electric/magnetic dipole (GPM) models; cross sections, near and far fields, LDOS, multipoles, chirality [paper] | **Dense full inversion, no FFT or iterative solver; memory grows as N²; about 10k–15k dipoles maximum. Homogeneous environment only** (layered media and periodic structures listed as future work) [arXiv:2505.09545v3](https://arxiv.org/html/2505.09545v3). | Ponomareva, Wiecha et al., SciPost Phys. Codebases 60 (2025), doi:10.21468/SciPostPhysCodeb.60 |
| **CPDDA** ([GL](https://gitlab.com/dmcxu1/cpdda), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11990789/)) | **Not stated** [paper][GL] | Paper 2025-03; repo last active 2025-04 | Likely dormant | Not on PyPI | **CuPy**, with 3D-FFT matrix-vector products; tested on Windows 10 with cupy-cuda12x | No | Cross sections; near and far fields; arbitrary shapes | About 26× GPU speed-up over CPU [paper]. The license is unclear, so it is unusable as a dependency, but it shows that CuPy FFT-DDA works on Windows. | Xu et al., Nanomaterials 15, 500 (2025), doi:10.3390/nano15070500 |
| **dda-bench** ([GH](https://github.com/Argencle/dda-bench)) | GPL-3 | 0.1.1, 2026-03-02 [PyPI] | New | `any` | Benchmarks the CPU and GPU builds of the wrapped codes | — | A harness that runs DDSCAT, ADDA and IFDDA and compares their outputs digit by digit | The DDA-only precedent for our benchmark; cite it. | Argentin, Chaumet, Gross & Yurkin, CPC 325, 110172 (2026), [arXiv:2603.02871](https://arxiv.org/abs/2603.02871) |
| **HoloPy** ([GH](https://github.com/manoharan-lab/holopy)) | GPL-3 | 3.5.0, 2021-05 [GH] (the PyPI `holopy` name belongs to an unrelated package) | Low | conda | No | No | Wraps Mie, multisphere T-matrix and ADDA | A historical multi-method wrapper (see §5). | — |

**No JAX DDA** was found by search. Every discoverable PyTorch or CuPy DDA is GPL or unlicensed.

**Gap confirmed: an MIT/BSD, FFT-accelerated (O(N log N) matrix-vector product), iterative-solver DDA** with:
- GPU support through PyTorch (native Windows CUDA, Linux CUDA/ROCm) and a CPU path through NumPy/SciPy;
- autodiff through implicit or adjoint differentiation of the linear solve, rather than unrolling the solver iterations;
- modern polarizabilities (LDR, FCD, IGT) and substrate support.

---

## 4. FDTD

| Library | License | Latest release | Activity | Install / Windows | GPU | Autodiff | Dispersive metals | Notes and limitations | Citation |
|---|---|---|---|---|---|---|---|---|---|
| **Meep** ([GH](https://github.com/NanoComp/meep)) | **GPL-2+** [GH] | v1.35.0, 2026-10-09 [GH] | Very active (1.8k stars) | **conda-forge only.** **No PyPI package** (the PyPI `meep` and `pymeep` names are unrelated or 404). **"Native Windows installation is currently unsupported"**; use WSL [doc](https://github.com/NanoComp/meep/blob/master/doc/docs/Installation.md). | No (CPU, MPI) | Adjoint solver (`meep.adjoint`, autograd/JAX wrappers) | Yes (Lorentz/Drude susceptibilities) | Subpixel smoothing; near-to-far field; Bloch periodic boundaries; the reference open FDTD. Optional plugin only, via a conda or WSL subprocess. | Oskooi et al., CPC 181, 687 (2010) |
| **FDTDX** ([GH](https://github.com/ymahlau/fdtdx), [docs](https://fdtdx.readthedocs.io)) | **MIT** [GH] (PyPI license field empty) | 0.6.3, 2026-10-09/12 | Very active (369 stars) | `any`, Python 3.11–3.14. Extras `[cuda12]`, `[rocm]` and `[tpu]` (since 0.4.3). Native Windows is CPU only, because JAX CUDA is Linux-only. **Depends on `tidy3d` (LGPL)** [PyPI]. | JAX: CUDA and ROCm (Linux), TPU, multi-GPU | **Yes**, JAX AD with time-reversal memory saving [doc] | **Drude/Lorentz ADE merged 2026-05-13 (v0.6.2+)** [GH commits]. The JOSS paper says non-dispersive only. Complex permittivity and conductivity added 2026-07. | PEC, PMC and Bloch boundaries since v0.6.2; GDS import. No published plasmonic or Mie validation found. | Mahlau et al., JOSS (2026), doi:10.21105/joss.08912 |
| **Tidy3D** ([GH](https://github.com/flexcompute/tidy3d)) | Client **LGPL-2.1**; solver proprietary and cloud-only | 2.12.2, 2026-10-28 [PyPI] | Very active | `any`; uv OK on all OSes. Runs need an API key and **paid FlexCredits**; there is an Education Free tier of 10 credits/month ([edu](https://flexcompute.com/tidy3d/educational-licenses), [billing](https://docs.flexcompute.com/projects/tidy3d/en/latest/faq/docs/faq/how-is-using-tidy3d-billed.html)). `tidy3d-extras` adds local functionality (its license field is empty). | Cloud GPU | Yes (autograd adjoint; `autograd` is a core dependency) | Yes; documented gold-sphere-vs-Mie notebook ([doc](https://docs.flexcompute.com/projects/tidy3d/en/stable/notebooks/PlasmonicNanoparticle.html)) | Best-in-class accuracy and UX, but not local and not free. **FlexAgent MCP server for AI agents** (Nov 2025). Optional cloud backend only. | — |
| **TorchFDTD** ([GH](https://github.com/hyoseokp/TorchFDTD)) | **MIT** [GH] | 1.1.7, 2026-10-06/07 | New (Sep 2026, 24 stars) | `any`. **Explicit native Windows + NVIDIA instructions** using the PyTorch cu126 index and CuPy kernels [PyPI readme]. | PyTorch + CuPy fused CUDA kernels; CPU fallback | Yes (discrete adjoint into torch autograd) | Unverified | Browser CAD workbench; validated against Meep (per its docs). Very young. | Park & Park, [arXiv:2609.30039](https://arxiv.org/abs/2609.30039) |
| **fdtd** (flaport, [GH](https://github.com/flaport/fdtd)) | MIT | 0.3.5, **2023-12-07** | Low (push 2025-09) | `any` | PyTorch `torch.cuda` backend (works on Windows) | Not documented | **No** Drude/Lorentz (conductivity and anisotropy only) [doc] | An educational code; not suitable for plasmonics without major work. | — |
| **fdtdz** ([GH](https://github.com/spinsphotonics/fdtdz)) | MIT | 1.1.3, 2023-10 | Dormant (push 2025-02) | sdist; JAX custom CUDA kernel; Linux | CUDA | JAX | No (dielectric photonics) | Specialised for thin-slab silicon photonics. | — |
| **ceviche** ([GH](https://github.com/fancompute/ceviche)) | MIT | 0.1.3, 2023-02 | Dormant | `any` | No | HIPS autograd | No | 2D FDFD/FDTD; teaching and inverse design. | Hughes et al., ACS Photonics (2019) |
| **Khronos.jl** ([GH](https://github.com/facebookresearch/Khronos.jl)) | MIT | — | **Archived** (2026) [GH] | Julia | KernelAbstractions: CUDA, ROCm, oneAPI, Metal | Yes | No polarizable materials (per a secondary listing) | Dead end, but a good design reference. | — |
| **Luminescent.jl** ([GH](https://github.com/paulxshen/Luminescent.jl)) / `luminescent` (PyPI 4.3.1) | **No license file on GitHub; PyPI license empty** (unverified, so treat as proprietary) | 4.3.1, 2026-10-28 | Active | Julia core with a Python wrapper | GPU | Yes | Unverified | License unclear, so avoid it. | — |
| **gprMax 4** ([GH](https://github.com/gprMax/gprMax)) | **GPL-3+** | 4.0.1, 2026-10-03 [PyPI] | Active | Wheels for Win, Linux and macOS (cp311–cp313) | CUDA, OpenCL, Apple Metal (per the v4 announcement; partly unverified) | No | Debye/Drude/Lorentz (GPR-oriented) | Built for ground-penetrating radar and antennas, not nano-optics. | — |
| **openEMS** ([GH](https://github.com/thliebig/openEMS)) | GPL-3 | v0.37.0, 2026-10-09 | Active | Windows builds; Python interface via `pyopenems` on PyPI (third party?) | No | No | Yes | RF-oriented EC-FDTD. Not a nano-optics fit. | — |

**Takeaway.**
- **No permissive, pip-installable, native-Windows-GPU FDTD has been validated for plasmonics.**
- FDTDX is the strongest MIT GPU candidate (JAX, Linux GPU), but its dispersive support is brand new.
- TorchFDTD is the only MIT FDTD with native Windows CUDA, but it is very young and its dispersion support is unverified.
- Meep remains the CPU accuracy reference, but it is GPL and conda-only, so it can only be an optional plugin.

---

## 5. Multi-method / unified toolkits (potential competitors)

| Project | What it unifies | How it differs from PlasmonHub |
|---|---|---|
| **HoloPy** ([GH](https://github.com/manoharan-lab/holopy)), GPL-3 | Mie (layered), multisphere T-matrix (SCSMFO) and ADDA behind one scatterer/theory API | Holography and inference focus; no FDTD; no GPU or autodiff; last release 2021; conda-only; GPL. |
| **TorchGDM** ([paper](https://scipost.org/SciPostPhysCodeb.60)), GPL-3 | GDM plus effective-dipole models from Mie and T-matrix (via treams) | One method family; dense and limited N; homogeneous environment; GPL. The closest "multi-scale" competitor. |
| **Tidy3D**, LGPL client with proprietary cloud solver | FDTD, mode solver, EME, heat/charge, RF | Commercial cloud; no Mie, T-matrix or DDA solvers, though Mie is used for validation; has an agent MCP. |
| **smuthi**, MIT | T-matrix (via NFM-DS for non-spherical particles) plus layered media | One method; packaging problems. |
| **PyMieSim**, MIT | Mie for spheres, core-shell and cylinders, with detectors and experiment sweeps | Mie only. |
| **dda-bench**, GPL-3 | DDSCAT, ADDA and IFDDA harness | DDA only; a benchmark, not a solver API. |
| **FermiLink** ([arXiv:2604.03460](https://arxiv.org/abs/2604.03460)) | LLM agent spanning about 50 simulation packages | An agent layer over existing codes, not a unified physics spec. |
| **ScattPort** ([scattport.org](https://scattport.org)), **Wikipedia list** ([DDA codes](https://en.wikipedia.org/wiki/Discrete_dipole_approximation_codes)) | Catalogues of scattering codes | Directories only. |

**Conclusion.** No project offers one declarative problem spec (particle + materials + environment + illumination + observables) dispatched to Mie, T-matrix, DDA and FDTD with consistent units, conventions and outputs, a validation harness, and agent-friendly schemas. That is the PlasmonHub niche.

---

## 6. Refractive-index data sources (programmatic)

| Source | License | Notes |
|---|---|---|
| **refractiveindex.info database** ([GH](https://github.com/polyanskiy/refractiveindex.info-database)) | **CC0-1.0** [GH] (public-domain dedication) | YAML; actively curated (push 2026-10-27; snapshot release 2026-05-24). Cite Polyanskiy, *Sci. Data* 11, 94 (2024), doi:10.1038/s41597-023-02898-2, plus the original dataset papers listed in each file [doc]. **Best choice: vendor or download a pinned snapshot.** |
| `refractiveindex` (toftul, [GH](https://github.com/toftul/refractiveindex)) | MIT | 1.3.1, 2026-10-27. Python interface to the database above. |
| `PyOptik` ([GH](https://github.com/MartinPdeS/PyOptik)) | MIT | 3.5.16, 2026-10-27. Unit-aware materials, Sellmeier and tabulated data, thin films. |
| `refidx` ([GL](https://gitlab.com/benvial/refidx)) | **Conflict:** `pyproject.toml` and PyPI say MIT, but GitLab's license detector reports GPL-3.0-only | 1.3.0, 2025-10. Clarify the license before depending on it. |
| `dispersion` (JCMwave, [GH](https://github.com/jcmwave/dispersion)) | MIT | 1.0.6, 2024. Database file handling. |
| Tidy3D material library and dispersion fitter | LGPL client | Useful for Drude-Lorentz pole fitting, which FDTD needs (fitter existence from memory; re-verify the API). |
| Meep materials library | GPL-2+ | Do not copy the code. The underlying literature fit parameters (e.g., Rakić 1998) are facts and can be cited and re-entered. |

**Recommendation.** Ship a small MIT module that:
- reads refractiveindex.info YAML (CC0);
- stores the provenance and citation of each dataset;
- provides analytic Drude-Lorentz / critical-point fits for Au, Ag, Al and Cu (e.g., Johnson & Christy, Rakić, Etchegoin), so that all four methods use **identical ε(λ)**. FDTD requires a fit while the others use tabulated data, so the benchmark should report the fit error separately.

Handbook compilations such as Palik may carry publisher copyright on the compilation; prefer primary-paper datasets (unverified legal nuance).

---

## 7. Published cross-method benchmarks for plasmonic and small particles

Status key:
- **V** = verified from the abstract, publisher page or arXiv page.
- **P** = citation verified, but the takeaway comes from abstracts or snippets.
- **U** = unverified.

### Classic comparisons

1. **Wriedt & Comberg (1998)**, *JQSRT* 60, 411, doi:10.1016/S0022-4073(98)00016-8. **P**
   - Compared DDA (DDSCAT), VIE, FDTD and EBCM on a dielectric cube.
   - FDTD is general but costly; DDA is fast but memory-hungry.
2. **Kahnert (2003)**, *JQSRT* 79–80, 775, and the 2016 tutorial, *JQSRT* 178, 22. **V**
   - Qualitative method reviews, not numerical benchmarks.
3. **Yurkin, Hoekstra, Brock & Lu (2007)**, *Opt. Express* 15, 17902, doi:10.1364/OE.15.017902. **P**
   - DDA vs FDTD for dielectric spheres (x ≤ 80, m ≤ 2).
   - DDA is faster at small m.
4. **Myroshnychenko et al. (2008)**, "Modelling the optical response of gold nanoparticles," *Chem. Soc. Rev.* 37, 1792, doi:10.1039/B711486A. **P**
   - A tutorial review with a critical comparison of methods, including FDTD and BEM.
   - The full method list and its findings are U.
5. **Hoffmann et al. (2009)**, *Proc. SPIE* 7390, 73900J, [arXiv:0907.3570](https://arxiv.org/abs/0907.3570). **V**
   - Compared FEM codes (JCMsuite, COMSOL, HFSS, CST) against MMP on gap-antenna near fields.
   - Time-domain results largely did not arrive in reasonable time.
6. **Smajic et al. (2009)**, *J. Comput. Theor. Nanosci.* 6, 763. **P**
   - Compared FEM and FDTD against MMP, MAS and BEM on 2D plasmonic wires.
   - Boundary methods win in 2D, and FDTD shows staircasing errors.
7. **Yurkin, de Kanter & Hoekstra (2010)**, *J. Nanophotonics* 4, 041585, doi:10.1117/1.3335329. **V**
   - ADDA on gold spheres, cubes and rods of 10–100 nm.
   - Keeping near-IR absorption error under 10% needs about 10⁷ dipoles per sphere.
   - Shape (staircase) error dominates, and cubes are about 10× better than spheres.
   - FCD beats LDR.
   - **This is the key motivation for a fast GPU DDA and for reporting dipole-count convergence.**
8. **Parsons, Burrows, Sambles & Barnes (2010)**, *J. Mod. Opt.* 57, 356, doi:10.1080/09500341003628702. **P**
   - Compared Mie, null-field T-matrix, DDA, FEM and FDTD on a gold sphere.
   - The closest classic analogue to our paper, but a one-off with no released code.
9. **Karamehmedović et al. (2011)**, *Opt. Express* 19, 8939. **P**
   - Compared NFM-DS, COMSOL and MMP on Ag spheres and spheroids in the near field.
10. **Loke et al. (2014)**, *Appl. Phys. B* 115, 237, doi:10.1007/s00340-013-5594-z. **P**
    - Compared DDA (including DDA-SI with a substrate) against FDTD, FEM, Mie and NFM-DS on Au spheres, free-standing and on a substrate.
11. **Calà Lesina et al. (2015)**, *Opt. Express* 23, 10481, doi:10.1364/OE.23.010481. **P**
    - Checked FDTD for Au and Ag spheres against Mie.
    - Needs a mesh of about 0.25–0.5 nm for about 1% error, a Drude plus two critical-point dispersion model, and **double precision**.
12. **Gallinet, Butet & Martin (2015)**, *Laser Photonics Rev.* 9, 577, doi:10.1002/lpor.201500122. **V** (abstract only)
    - Review of "standard problems" in nanophotonics.

### Recent work (2019–2026)

13. **Lalanne et al. (2019)**, *JOSA A* 36, 686, [arXiv:1811.11751](https://arxiv.org/abs/1811.11751). **V**
    - A multi-group benchmark of quasinormal-mode solvers, written to set standards.
    - A model for our paper's framing.
14. **Rassinfosse et al. (2024)**, *Optics Continuum* 3(6), doi:10.1364/OPTCON.511183. **P**
    - FDTD on periodic plasmonics.
    - "No general rule" for mesh size exists.
15. **Dieperink et al. (2024)**, *Nanophotonics* 13, 4647, doi:10.1515/nanoph-2024-0238, [arXiv:2410.08093](https://arxiv.org/abs/2410.08093). **P**
    - Compared DDA, FDTD, DGTD and BEM against Mie for a 50 nm Au sphere.
    - BEM was fastest by orders of magnitude, and DDA showed irregular convergence.
    - **The closest recent prior work**, but a single case with no harness.
16. **Argentin, Chaumet, Gross & Yurkin (2026)**, *CPC* 325, 110172, [arXiv:2603.02871](https://arxiv.org/abs/2603.02871), code: dda-bench. **V**
    - DDSCAT, ADDA and IFDDA agree to 10–15 digits when hidden parameters are aligned, and only to about 4 digits otherwise.
    - Fair benchmarking is hard because of defaults and stopping criteria.
    - Covers CPU and GPU energy use.
    - Test cases are dielectric only (no metals, no Mie reference).
17. **Tidy3D gold sphere vs Mie notebook.** **V**, but vendor material and not peer-reviewed.
    - Notes that subpixel averaging is off for metals.

**Novelty check.** No open, scripted, reproducible benchmark was found that runs Mie, T-matrix, DDA and FDTD **open-source codes** against Mie for **plasmonic** particles and reports both accuracy and cost.
- The precedents are each narrower: #16 is DDA-only and dielectric, #15 is a single case, and #8 and #10 are one-offs.
- Before submission:
  - search *JQSRT* 2025–26 and the ELS 2026 proceedings;
  - check the validation sections of the smuthi, treams and MNPBEM papers;
  - check anything citing arXiv:2603.02871.
- **Methodological lesson from #16:** expose and align the hidden parameters (polarizability prescription, iterative tolerance, stopping rule, precision), and report them.

---

## 8. LLM agents running photonics/EM simulations (2023–2026)

| Work | Simulator | What the agent does | Lessons | Status |
|---|---|---|---|---|
| **MetaChat**: Lupoiu … Fan, *Sci. Adv.* (Oct 2025), doi:10.1126/sciadv.adx8006, [arXiv:2503.20479](https://arxiv.org/abs/2503.20479) | DL surrogate Maxwell solvers | Multi-agent freeform metasurface design | Agents work with fast surrogates in the loop. | V |
| **Kim, Park & Shin**, *Nanophotonics* (2025), doi:10.1515/nanoph-2024-0674 | Transfer matrix (LLM-written code) | In-context design of multilayers and metasurfaces | — | V |
| **Huang et al.**, MCP-enabled LLM for meta-optics inverse design, [arXiv:2508.10277](https://arxiv.org/abs/2508.10277) | TorchRDIT (differentiable RCWA) via an MCP server with 5 APIs | Inverse design | **Structured prompts and a small API surface beat free-form natural language** on quality, cost and errors. | V |
| **PhIDO**: Sharma … Poon, [arXiv:2508.14123](https://arxiv.org/abs/2508.14123) | PIC layout and simulation | Natural language to PIC layout | Up to 91% success on single devices; about 57% pass@5 end-to-end. | V |
| **PICBench**, [arXiv:2502.03159](https://arxiv.org/abs/2502.03159) | Circuit simulator | Netlist generation | Iterative error feedback matters. | V |
| **PICopilot**, [arXiv:2608.01791](https://arxiv.org/abs/2608.01791) | Unstated | Multi-agent retrieval-augmented script generation | Solved 48/48 tasks vs 27/48 for a GPT-5 baseline with generic retrieval. | V |
| **Kharel, Khavasi, Chen & Hughes**, [arXiv:2606.00915](https://arxiv.org/abs/2606.00915) | Unstated (Tidy3D likely, U) | Propose, simulate, evaluate against acceptance criteria | — | V |
| **HALO** with HALO-Bench (52 tasks), [arXiv:2608.28877](https://arxiv.org/abs/2608.28877) | Nanophotonic solvers (unstated) | Compares a fixed workflow, a structured agent and a coding agent | **Typed specs plus a fixed workflow are the most reliable and token-efficient**; free-form coding adds operational failures. | V |
| **Biswas, De & Islam**, [arXiv:2606.22493](https://arxiv.org/abs/2606.22493) | Eigenmode solver + FDTD (MPB/Meep, U) | Directional-coupler design with convergence judgment | — | V |
| **Flexcompute FlexAgent MCP** ([news](https://flexcompute.com/news/ai-assisted-simulation-in-tidy3d-ushering-in-a-new-era-of-photonic-design), [docs](https://docs.flexcompute.com/projects/tidy3d/en/stable/ai/flex_agent.html)) | Tidy3D | Builds and modifies simulations, troubleshoots validation errors; works with Claude Code, Cursor and Codex | Vendor product with no metrics. Tidy3D now hard-depends on `fastmcp` [PyPI]. | V |
| **FEABench**, [arXiv:2504.06260](https://arxiv.org/abs/2504.06260) | COMSOL API | LLMs generate FEA models | Executable 88% of the time, but **no model fully solved any problem**: "runs" is not "correct". | P |
| **FermiLink**, [arXiv:2604.03460](https://arxiv.org/abs/2604.03460) | About 50 packages (a Meep variant, U) | Reproduces published figures | 74/132 figures reproduced (56%). | V |

**Design implications for PlasmonHub:**
- a Pydantic/JSON-Schema problem spec with a small typed API surface;
- an MCP server exposing schema, validate, run and compare;
- machine-readable errors with fix hints;
- built-in physics checks (an optical-theorem check, Mie agreement, a convergence flag), so that an agent can tell "ran" from "correct".

No agent or benchmark targets single-particle scattering codes with an analytic ground truth, which is a differentiator. The search was not exhaustive.

---

## 9. GPU array-framework realities on native Windows (late 2026)

| Framework | Native Windows GPU | Linux GPU | macOS GPU | Autodiff | FFT / complex128 | Install notes |
|---|---|---|---|---|---|---|
| **PyTorch** 2.15.0 (2026-10-28) | **Yes, NVIDIA CUDA.** Wheels for cu126, cu130, cu132 and cu134 (`win_amd64`) are on `download.pytorch.org` (checked index). AMD ROCm on Windows is a **preview** through AMD's own wheels (ROCm 7.2, Jan 2026; RX 7000/9000 and Ryzen AI only) ([AMD notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-2.html)). | CUDA and ROCm | MPS. No float64 on MPS (widely reported, e.g. [Apple forum](https://www.developer.apple.com/forums/thread/797778)); complex support partial and improving (2.15 adds complex64 LU on MPS, from secondary source). | Yes | `torch.fft` with complex128 on CUDA and CPU | The **PyPI `torch` Windows wheel is CPU-only**. uv needs a `[[tool.uv.index]]` entry for the PyTorch CUDA index with `explicit = true` plus `tool.uv.sources`. |
| **JAX** 0.12.0.1 (2026-10-21) | **No (official).** Native Windows is CPU-only ("experimental"). `jax-cuda12-plugin` and `jax-cuda13-plugin` ship **Linux wheels only** [PyPI]. The docs say pip CUDA installs "do not work with Windows" ([docs](https://docs.jax.dev/en/latest/installation.html)). WSL2 + NVIDIA is "experimental". | CUDA 12 and 13; ROCm | **Not supported**; `jax-metal` last released 2024-10 [PyPI] | Yes (best-in-class) | Yes | Unofficial **winjax** 0.12.0 ([GH](https://github.com/eterevsky/winjax), beta, 0 stars) loads a Windows-built PJRT CUDA plugin. Single GPU; **cannot coexist with PyTorch in one process**; no Pallas. Not a foundation to build on. |
| **CuPy** 14.2.0 | **Yes.** `cupy-cuda12x` and `cupy-cuda13x` Windows wheels; the `[ctk]` extra pulls the CUDA toolkit from PyPI ([docs](https://docs.cupy.dev/en/stable/install.html)) | CUDA; ROCm experimental (no official ROCm wheels from v13.4) | No | **No** | cuFFT, complex128 | Good for a fast non-differentiable path. Zero-copy with torch via DLPack. |
| **NVIDIA Warp** 1.18.0 | Windows wheels [PyPI] | CUDA | macOS-arm wheels (CPU) | Yes (kernel-level) | Limited | A kernel DSL, a poor fit for FFT-heavy DDA. |
| **nvmath-python** 1.0.0 | Windows wheels | Linux | — | No | cuFFT, cuBLAS with device-side callbacks | A possible accelerator under CuPy or torch. |

**Implications for the GPU strategy:**
1. **Use PyTorch as the primary GPU and autodiff backend for our own solvers** (DDA first, then the differentiable Mie). It is the only option that gives native Windows CUDA, Linux CUDA/ROCm, a CPU fallback, and autodiff in one install. Keep a pure NumPy/SciPy CPU reference path, and consider the Python Array API standard (`array-api-compat`) to share code.
2. **Treat JAX as a "Linux/WSL2 backend family"** that reaches FDTDX. Run it out of process (subprocess or worker), so that JAX and torch never share a process, which also sidesteps the winjax conflict.
3. **FP64 is a real concern.** Plasmonic accuracy studies call for double precision (#11), and consumer GeForce GPUs have very low FP64 throughput. Benchmark complex64 vs complex128 (and mixed-precision iterative refinement) explicitly, as both a paper result and a design decision.
4. On macOS, MPS has no float64, so default to the CPU there.

---

## 10. Recommendations

### Per-method backend choices (MIT-core-compatible first; GPL = optional plugin only)

| Method | Best CPU backend (pip/uv, permissive) | Best GPU backend (pip/uv, permissive) | Optional plugins (GPL or other caveats) | Our own work |
|---|---|---|---|---|
| **Mie** | **miepython** (MIT) for homogeneous spheres; **treams** (MIT) for layered spheres and T-matrix-consistent Mie | None needed for speed. For autodiff and batched GPU, **write our own MIT PyTorch Mie** (a few hundred lines; log-derivative recurrences). | scattnlay (GPL; multiprecision cross-check), PyMieDiff (GPL), PyMieScatt (MIT but stale) | A small, high-value MIT differentiable Mie; also the benchmark ground truth. Audit it against scattnlay's multiprecision mode. |
| **T-matrix** | **treams** (MIT; Windows, Linux and macOS-arm wheels; clusters, lattices, stratified media) | **None exists** that is pip-installable (CELES is MATLAB) | smuthi (MIT itself, but has a GPL `pywigxjpf` dependency and `numpy<2`), pyMSTM / MSTM (MIT, Linux wheels only), TERMS (MPL-2.0 executable), MiePy (GPL). Exclude SMARTIES (CC BY-NC). | Upstream work (§ below). A GPU T-matrix is a lower-priority gap. Read and write the KIT T-matrix HDF5 format. |
| **DDA / GDM** | **No permissive pip-installable option exists.** Use our own MIT FFT-DDA (NumPy/SciPy, CPU) | **Our own MIT FFT-DDA in PyTorch** (CUDA on Windows and Linux, ROCm on Linux) | ADDA (GPL-3 executable via subprocess; the accuracy reference, with OpenCL GPU), pyGDM2 and TorchGDM (GPL-3), DDSCAT (GPL) | **Highest-value gap** (see below). |
| **FDTD** | No permissive CPU code is validated for plasmonics. Short term, use **FDTDX on CPU** (MIT; runs on native Windows CPU, slowly). | **FDTDX** (MIT, JAX; Linux/WSL2 CUDA or ROCm). **TorchFDTD** (MIT; native Windows CUDA) is a watch/evaluate candidate. | **Meep** (GPL-2+, conda/WSL, CPU reference); **Tidy3D** (LGPL client + paid cloud, "cloud plugin"); flaport/fdtd (MIT but no dispersion) | Contribute plasmonic validation, and fixes if needed, to FDTDX's new Drude-Lorentz ADE. Our benchmark is that validation. |

### The most valuable gap to fill ourselves: a permissive GPU + autodiff FFT-DDA

Today's options are GPL (ADDA, DDSCAT, pyGDM2, TorchGDM), unlicensed (CPDDA), or dense and size-limited (TorchGDM, about 10–15k dipoles). Yet the literature (#7) says plasmonic accuracy needs 10⁶–10⁷ dipoles, which only FFT plus an iterative solver on a GPU makes interactive.

**Minimum viable feature set:**
- Polarizabilities: CM, LDR, FCD and IGT. Use FCD as the default for metals, per #7.
- FFT-accelerated interaction matvec in torch with complex64 and complex128.
- Krylov solvers: BiCGStab, QMR and GMRES. ADDA's defaults (QMR) are a good reference.
- Gradients via **implicit differentiation** (an adjoint solve with the transposed system) for memory-safe autodiff.
- Observables: C_ext, C_abs and C_sca; near and far fields; orientation averaging.
- Later: substrate via a Sommerfeld or image approximation, and periodic structures.
- Validate against ADDA bit-for-bit where possible, reusing dda-bench's aligned-parameter methodology, and against Mie.

### Upstream contributions (cheap, high goodwill, and they unblock our backends)

- **smuthi:**
  - release Meson-built wheels for cp311–cp313 on Windows and macOS;
  - lift the `numpy<2` and `scipy<1.14` pins;
  - make `pywigxjpf` (GPL) optional or replace it with a permissive Wigner-3j implementation. This also removes the license wrinkle.
- **pytmatrix:** a Meson/f2py port so it builds on Python ≥3.12.
- **scattnlay:** publish 2.5 to PyPI with cibuildwheel for Linux and more Python versions.
- **pyMSTM:** Windows and macOS wheels.
- **treams:** Python 3.14 support and relaxing the scipy cap.
- **FDTDX:** plasmonic test cases (Au and Ag spheres vs Mie, with a Drude-Lorentz fit); declare the license in the PyPI metadata (the field is currently empty).

### Packaging policy for the MIT core

- **Core dependencies:** numpy, scipy, pydantic, miepython, treams. These are all MIT/BSD with Windows wheels.
- **Extras:**
  - `[torch]` for the GPU DDA and Mie;
  - `[jax]` for FDTDX, Linux-recommended and run out-of-process;
  - `[tidy3d]` for the cloud backend;
  - `[gpl]` meta-extra for scattnlay, pyGDM2 and TorchGDM, documented as changing the license of the combined work if redistributed.
- ADDA and Meep are discovered as external executables or environments, never imported.
- Avoid depending on packages with exact version pins (PyMieSim) or GPL transitive dependencies (smuthi) in the core.

---

*Main sources not linked inline: PyPI JSON API pages `https://pypi.org/pypi/<name>/json` for every package named above, and the GitHub and GitLab REST APIs, all queried 2026-11-01.*
