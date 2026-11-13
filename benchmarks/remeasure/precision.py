"""Cost and accuracy of double versus single precision on the GPU (DDA and FDTD).

Same problem, same tolerance; only the precision option changes. Wall times are medians of three
runs after a warm-up. On consumer GPUs FP64 throughput is a small fraction of FP32, so this
quantifies what "double precision is slow" means for the two volume solvers.

    uv run --no-sync python benchmarks/remeasure/precision.py

Writes benchmarks/remeasure/results/precision.json.
"""

from __future__ import annotations

import datetime as _dt
import json
import statistics
import time
from pathlib import Path

import numpy as np

import lightsmith as ls

OUT = Path(__file__).resolve().parent / "results" / "precision.json"
AU = {"kind": "library", "name": "Au", "dataset": "Johnson-Christy"}
DRUDE = {"kind": "drude", "eps_inf": 9.5, "wp_ev": 9.0, "gamma_ev": 0.07}


def sim(material, r, wl, opts):
    return ls.Simulation.model_validate({
        "particles": [{"shape": {"kind": "sphere", "radius_nm": r}, "material": material}],
        "medium_index": 1.33, "wavelengths": wl, "options": opts})


def timed(s, backend, repeats=3):
    ls.solve(s, backend, "gpu")  # warm-up (JIT, cuFFT plans, CUDA graphs)
    walls, res = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = ls.solve(s, backend, "gpu")
        walls.append(time.perf_counter() - t0)
    return res, statistics.median(walls)


def main():
    from _machine import NOTE, state

    before = state()
    cases = [
        ("dda Au r=40 npd=32, 520/600 nm", "lightsmith-dda", AU, 40.0, [520.0, 600.0],
         lambda p: {"dda": {"dipoles_per_diameter": 32, "precision": p, "tol": 1e-5}}),
        ("dda Au r=40 npd=48, 520/600 nm", "lightsmith-dda", AU, 40.0, [520.0, 600.0],
         lambda p: {"dda": {"dipoles_per_diameter": 48, "precision": p, "tol": 1e-5}}),
        ("fdtd Drude r=40, 2 nm mesh", "lightsmith-fdtd", DRUDE, 40.0, [500.0, 600.0, 700.0],
         lambda p: {"fdtd": {"spacing_nm": 2.0, "precision": p}}),
    ]
    rows = []
    for label, backend, mat, r, wl, opts in cases:
        out = {"label": label}
        for prec in ("single", "double"):
            res, wall = timed(sim(mat, r, wl, opts(prec)), backend)
            out[prec] = {"wall_s": wall, "c_ext_nm2": res.c_ext_nm2, "c_abs_nm2": res.c_abs_nm2,
                         "iterations": res.provenance.get("iterations")}
        a, b = (np.array(out[p]["c_ext_nm2"] + out[p]["c_abs_nm2"]) for p in ("single", "double"))
        out["max_rel_diff"] = float(np.max(np.abs(a - b) / np.abs(b)))
        out["slowdown_double"] = out["double"]["wall_s"] / out["single"]["wall_s"]
        print(json.dumps({k: out[k] for k in ("label", "max_rel_diff", "slowdown_double")}), flush=True)
        rows.append(out)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"cases": rows, "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                               "note": NOTE, "machine_state": {"before": before, "after": state()},
                               "doctor": ls.doctor()}, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
