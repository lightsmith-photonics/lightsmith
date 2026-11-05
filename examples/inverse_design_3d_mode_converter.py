"""Topology-optimise a 3D SOI TE0 -> TE1 mode converter (full-vector 3D FDFD + adjoint), validate with EME.

A 2.0 x 1.6 um design region in the 220 nm silicon layer (SiO2 cladding) joins a
500 nm input waveguide (port o1, TE0) to a 1000 nm output waveguide (port o2, whose
second mode o2@1 is TE1). The density is *extruded* through the 220 nm layer (one
vertical etch, as fabricated), with a conic filter, tanh projection with
beta-continuation and a 120 nm minimum feature size. We maximise
|S[o2@1 <- o1]|^2 at 1550 nm with MMA on the 3D FDFD solver (BiCGStab on the GPU,
warm-started forward and adjoint solves). The start is random: a mirror-symmetric
start cannot couple the even TE0 to the odd TE1 and has zero gradient.

The final binary design is exported to polygons and GDS, re-simulated with 3D FDFD
over 1530-1570 nm, and validated at 1550 nm by an independent method: eigenmode
expansion (``lightsmith-eme``: full-vector FDE cross sections + mode matching) on the
exported polygons. A strongly scattering topology-optimised device needs a large
modal basis in EME: on our design the TE0 -> TE1 power from EME was 0.20, 0.50,
0.74, 0.80 and 0.82 with 16, 30, 60, 100 and 150 modes per cell, approaching the
0.88 of 3D FDFD (40 nm and 25 nm grids), so the spec asks for 100 modes.

    uv run python examples/inverse_design_3d_mode_converter.py [--device gpu] [--iters 30]

The problem is read from ``inverse_design_3d_mode_converter.json`` next to this
script (``--write-spec`` regenerates it). Outputs land in ./inverse_3d_out/:
result.json, design.gds, checkpoint.npz (rerun to resume), validation.json and
design.png (if matplotlib is installed). Measured on an RTX 5070 and a CPU both
shared with other jobs: 73 minutes for 30 iterations (70 forward + adjoint
evaluations of 1.3 million unknowns), 3 minutes for the FDFD band check and about
35 minutes for the 100-mode EME solve on the CPU. Rerunning resumes from the
checkpoint, so only the validation is repeated.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import lightsmith as ls
from lightsmith import inverse
from lightsmith.problems import DeviceProblem, Polygon, Port, Prism, Rectangle, Structure

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "inverse_design_3d_mode_converter.json")
SI = ls.ConstantIndex(n_real=3.48)
SIO2 = ls.ConstantIndex(n_real=1.44)
L, W_DES = 2000.0, 1600.0  # design region length (x) and width (y)
W_IN, W_OUT = 500.0, 1000.0
XP = L / 2 + 400  # port planes at +-XP
BAND = [1530.0, 1550.0, 1570.0]  # 3D FDFD validation wavelengths
EME_BAND = [1550.0]  # EME validation wavelengths (100 modes per cell is slow on a CPU)


def waveguide(cx, w, length):
    return Structure(geometry=Prism(profile=Rectangle(center_nm=(cx, 0), size_nm=(length, w)),
                                    z_min_nm=-110, z_max_nm=110), material=SI)


def build_problem(iters: int = 30) -> inverse.OptimizationProblem:
    device = DeviceProblem(
        dimension=3, background=SIO2,
        structures=[waveguide(-L / 2 - 2500, W_IN, 5000), waveguide(L / 2 + 2500, W_OUT, 5000)],
        domain_size_nm=(2 * XP + 400, W_DES + 800, 1400), wavelengths=[1550.0],
        ports=[Port(name="o1", center_nm=(-XP, 0, 0), size_nm=(0, W_DES + 600, 1300), direction="+"),
               Port(name="o2", center_nm=(XP, 0, 0), size_nm=(0, W_DES + 600, 1300), direction="-",
                    num_modes=2)],
        options={"fdfd": {"dl_nm": 40.0, "pml_cells": 10, "tol": 1e-5},
                 "eme": {"grid_spacing_nm": 25.0, "num_modes": 100, "taper_cells": 1}},
    )
    return inverse.OptimizationProblem(
        device=device,
        design_regions=[inverse.DesignRegion(
            size_nm=(L, W_DES, 220), material_low=SIO2, material_high=SI, init="random", seed=1,
            filter_radius_nm=120.0, min_feature_nm=120.0, parametrization="extruded")],
        objective=[inverse.SParameterTerm(out_port="o2", out_mode=1, in_port="o1")],
        optimizer=inverse.OptimizerSettings(method="mma", max_iter=iters, beta_start=4, beta_max=64,
                                            beta_growth=2, beta_every=max(iters // 5, 1), move_limit=0.1),
    )


def snapped(structures, step: float) -> list[Structure]:
    """Polygon vertices rounded to a `step` raster along x, so EME slices the device into <= L/step cells.

    EME places a cell boundary at every vertex coordinate along the propagation axis;
    marching-squares contours have hundreds of distinct x values. Rounding moves vertices
    by at most step/2 (here 10 nm, a quarter of the optimisation grid).
    """
    out = []
    for s in structures:
        g = s.geometry
        if isinstance(g, Prism) and isinstance(g.profile, Polygon):
            v = [(round(x / step) * step, y) for x, y in g.profile.vertices_nm]
            v = [p for i, p in enumerate(v) if p != v[i - 1]]
            if len(v) >= 3:
                out.append(s.model_copy(update={"geometry": g.model_copy(
                    update={"profile": Polygon(vertices_nm=v)})}))
            continue
        out.append(s)
    return out


def te1_power(res: ls.results.SParameterResult) -> np.ndarray:
    S = res.s()
    return np.abs(S[:, res.ports.index("o2@1"), res.ports.index("o1")]) ** 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--iters", type=int, default=None, help="override optimizer.max_iter of the spec")
    ap.add_argument("--out", default="inverse_3d_out")
    ap.add_argument("--write-spec", action="store_true", help="rewrite the JSON spec from build_problem() and exit")
    args = ap.parse_args()
    if args.write_spec:
        with open(SPEC, "w", newline="\n") as fh:
            fh.write(build_problem().model_dump_json(indent=1))
        print("wrote", SPEC)
        return
    os.makedirs(args.out, exist_ok=True)
    with open(SPEC) as fh:
        spec = inverse.OptimizationProblem.model_validate_json(fh.read())
    if args.iters:
        spec = spec.model_copy(update={"optimizer": spec.optimizer.model_copy(update={
            "max_iter": args.iters, "beta_every": max(args.iters // 5, 1)})})
    print(json.dumps(ls.estimate(spec, method="inverse", device=args.device), default=str, indent=1))

    def progress(s: inverse.IterationState) -> None:
        print(f"it {s.iteration:3d}  objective {s.objective:+.4f}  beta {s.beta:5.1f}  "
              f"binary {s.binarization:.2f}  evals {s.evaluations}  {s.elapsed_s:7.1f}s", flush=True)

    t0 = time.perf_counter()
    res = inverse.optimize(spec, device=args.device, callback=progress,
                           checkpoint=os.path.join(args.out, "checkpoint.npz"), validate=False)
    t_opt = time.perf_counter() - t0
    print(f"optimised in {t_opt:.0f}s on {res.provenance.get('device')} {res.provenance.get('gpu_name', '')}, "
          f"{res.provenance['evaluations']} evaluations")
    print(f"objective: initial {res.objective_initial:.4f} -> final {res.objective_final:.4f} "
          f"(hard-thresholded {res.objective_binary:.4f})")
    for w in res.warnings:
        print("warning:", w)
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        fh.write(res.model_dump_json())

    d = res.designs[0]
    contours = inverse.marching_squares(np.asarray(d.density), np.asarray(d.x_nm), np.asarray(d.y_nm))
    gds = inverse.write_gds(os.path.join(args.out, "design.gds"), contours, layer=(1, 0))
    print("GDS:", gds)

    # Re-simulate the exported polygons over the band: 3D FDFD (same solver as the optimisation) ...
    band_dev = res.final_device.model_copy(update={"wavelengths": BAND})
    t1 = time.perf_counter()
    fd = ls.solve(band_dev, method="fdfd", device=args.device)
    t_fd = time.perf_counter() - t1
    # ... and EME (independent method), with polygon vertices snapped to 20 nm along x.
    t1 = time.perf_counter()
    eme = ls.solve(band_dev.model_copy(update={"structures": snapped(band_dev.structures, 20.0),
                                               "wavelengths": EME_BAND}), method="eme", device="cpu")
    t_eme = time.perf_counter() - t1
    fd_te1, eme_te1 = te1_power(fd), te1_power(eme)
    eme_te0 = np.abs(eme.s()[:, eme.ports.index("o2@0"), eme.ports.index("o1")]) ** 2
    print(f"validation on the exported polygons (FDFD {t_fd:.0f}s, EME {t_eme:.0f}s with "
          f"{eme.provenance['cells']} cells, {eme.provenance['modes_per_cell'][0]} modes per cell):")
    for lam, a in zip(BAND, fd_te1):
        line = f"  {lam:.0f} nm: TE0->TE1 |S|^2  3D FDFD {a:.4f} ({10 * np.log10(a):.2f} dB)"
        if lam in EME_BAND:
            k = EME_BAND.index(lam)
            line += f"  EME {eme_te1[k]:.4f} ({10 * np.log10(eme_te1[k]):.2f} dB, residual TE0 {eme_te0[k]:.1e})"
        print(line)
    for w in fd.warnings + eme.warnings:
        print("validation warning:", w)
    summary = {"objective_initial": res.objective_initial, "objective_final": res.objective_final,
               "objective_binary": res.objective_binary, "optimisation_s": t_opt, "fdfd_validation_s": t_fd,
               "eme_validation_s": t_eme, "device": res.provenance.get("device"),
               "gpu_name": res.provenance.get("gpu_name"), "evaluations": res.provenance["evaluations"],
               "grid": res.provenance["grid"], "wavelengths_nm": BAND, "eme_wavelengths_nm": EME_BAND,
               "eme_modes_per_cell": eme.provenance["modes_per_cell"],
               "fdfd_te1_power": fd_te1.tolist(), "eme_te1_power": eme_te1.tolist(),
               "eme_te0_power": eme_te0.tolist(), "eme_cells": eme.provenance["cells"], "gds": gds,
               "warnings": res.warnings + fd.warnings + eme.warnings}
    with open(os.path.join(args.out, "validation.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"validated figure of merit (EME on the exported GDS polygons, TE0->TE1 power at 1550 nm): "
          f"{eme_te1[EME_BAND.index(1550.0)]:.4f}; 3D FDFD band mean {fd_te1.mean():.4f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(np.asarray(d.density).T, origin="lower", cmap="gray_r",
                 extent=(d.x_nm[0], d.x_nm[-1], d.y_nm[0], d.y_nm[-1]))
    ax[0].set_title("optimised density (extruded, 220 nm Si)")
    ax[0].set_xlabel("x (nm)")
    ax[0].set_ylabel("y (nm)")
    ax[1].plot([h["objective"] for h in res.history])
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("|S(o2@1 <- o1)|^2 at 1550 nm (3D FDFD)")
    ax[2].plot(BAND, fd_te1, "o-", label="3D FDFD")
    ax[2].plot(EME_BAND, eme_te1, "s", label="EME (independent)")
    ax[2].set_xlabel("wavelength (nm)")
    ax[2].set_ylabel("TE0 -> TE1 power, exported GDS")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "design.png"), dpi=120)


if __name__ == "__main__":
    main()
