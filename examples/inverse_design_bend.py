"""Topology-optimise a compact silicon 90-degree waveguide bend (2D FDFD + adjoint).

A 2 x 2 um design region joins a horizontal input waveguide (port o1) to a vertical
output waveguide (port o2). We maximise the worst-case |S21|^2 over 1500-1600 nm,
with a conic filter, tanh projection with beta-continuation and a 100 nm minimum
feature size, using globally convergent MMA. The final binary design is exported
to lightsmith Structures (re-simulated for validation) and to GDS.

    uv run python examples/inverse_design_bend.py [--device auto|cpu|gpu] [--iters 60]

Outputs land in ./inverse_bend_out/: result.json (OptimizationResult), design.gds,
checkpoint.npz (rerun to resume), and design.png if matplotlib is installed.
Runs in about 1-3 minutes on a laptop CPU, faster on a CUDA GPU.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import lightsmith as ls
from lightsmith import inverse
from lightsmith.problems import DeviceProblem, Port, Prism, Rectangle, Structure

SI = ls.ConstantIndex(n_real=3.48)
SIO2 = ls.ConstantIndex(n_real=1.44)
W, HALF = 450.0, 1000.0  # waveguide width; half-size of the design region (nm)


def waveguide(cx, cy, sx, sy):
    return Structure(geometry=Prism(profile=Rectangle(center_nm=(cx, cy), size_nm=(sx, sy)),
                                    z_min_nm=-110, z_max_nm=110), material=SI)


def build_problem(iters: int) -> inverse.OptimizationProblem:
    device = DeviceProblem(
        dimension=2, background=SIO2,
        structures=[waveguide(-5000 - HALF, 0, 10000, W),   # input, along x
                    waveguide(0, 5000 + HALF, W, 10000)],   # output, along y
        domain_size_nm=(2 * HALF + 1200, 2 * HALF + 1200, 0),
        wavelengths=[1500.0, 1550.0, 1600.0],
        ports=[Port(name="o1", center_nm=(-HALF - 400, 0, 0), size_nm=(0, 1800, 1), direction="+"),
               Port(name="o2", center_nm=(0, HALF + 400, 0), size_nm=(1800, 0, 1), direction="-")],
        options={"fdfd": {"polarization": "te", "dl_nm": 25.0, "pml_cells": 14}},
    )
    return inverse.OptimizationProblem(
        device=device,
        design_regions=[inverse.DesignRegion(
            size_nm=(2 * HALF, 2 * HALF, 220), material_low=SIO2, material_high=SI, init=0.5,
            filter_radius_nm=100.0, min_feature_nm=100.0)],
        objective=[inverse.SParameterTerm(out_port="o2", in_port="o1", aggregate="min"),
                   inverse.SParameterTerm(out_port="o1", in_port="o1", goal="minimize", weight=0.5)],
        optimizer=inverse.OptimizerSettings(method="mma", max_iter=iters, beta_start=4, beta_max=64,
                                            beta_growth=2, beta_every=max(iters // 5, 1)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", default="inverse_bend_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    spec = build_problem(args.iters)
    print(ls.estimate(spec.device, method="fdfd", device=args.device))

    def progress(s: inverse.IterationState) -> None:
        print(f"it {s.iteration:3d}  objective {s.objective:+.4f}  beta {s.beta:5.1f}  "
              f"binary {s.binarization:.2f}  evals {s.evaluations}  {s.elapsed_s:6.1f}s", flush=True)

    t0 = time.perf_counter()
    res = inverse.optimize(spec, device=args.device, callback=progress,
                           checkpoint=os.path.join(args.out, "checkpoint.npz"))
    print(f"done in {time.perf_counter() - t0:.0f}s on {res.provenance.get('device')}")

    S = res.sparameters.s()
    t21 = np.abs(S[:, 1, 0]) ** 2
    r11 = np.abs(S[:, 0, 0]) ** 2
    print(f"objective: initial {res.objective_initial:+.3f} -> final {res.objective_final:+.3f} "
          f"(binary {res.objective_binary:+.3f})")
    for lam, t, r in zip(res.sparameters.wavelengths_nm, t21, r11):
        print(f"  exported polygons, {lam:.0f} nm: |S21|^2 = {t:.4f} ({10 * np.log10(t):.2f} dB), |S11|^2 = {r:.2e}")
    for w in res.warnings:
        print("warning:", w)

    with open(os.path.join(args.out, "result.json"), "w") as fh:
        fh.write(res.model_dump_json())
    d = res.designs[0]
    contours = inverse.marching_squares(np.asarray(d.density), np.asarray(d.x_nm), np.asarray(d.y_nm))
    print(inverse.write_gds(os.path.join(args.out, "design.gds"), contours, layer=(1, 0)))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].imshow(np.asarray(d.density).T, origin="lower", cmap="gray_r",
                 extent=(d.x_nm[0], d.x_nm[-1], d.y_nm[0], d.y_nm[-1]))
    ax[0].set_title("optimised density")
    ax[1].plot([h["objective"] for h in res.history])
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("objective")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "design.png"), dpi=120)


if __name__ == "__main__":
    main()
