"""Re-measure the PWE "GPU vs. CPU" row: agreement (2D test config) and the 3D speedup ("~17x reported").

- Agreement: tests/test_pwe.py::test_gpu_matches_cpu (square lattice of eps=8.9 rods, r=0.2a, TE+TM,
  points_per_segment=4, 4 bands; TM dense in single precision on the GPU, TE LOBPCG in double).
- 3D timing: tests/test_pwe.py::test_diamond_lattice_complete_gap_3d (diamond lattice of eps=11.56
  spheres, X-U-L-Gamma-X-W-K path, 3 points per segment, 5 bands, LOBPCG at resolution 16), solved on
  the CPU and on the GPU; optional extra resolutions via --resolutions.

    uv run --no-sync python benchmarks/remeasure/pwe.py [--repeats 3] [--resolutions 16]

GPU and CPU load are recorded with the timings (nvidia-smi snapshots, machine_state).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import statistics
import subprocess
import time

import numpy as np
import torch

import lightsmith as ls
from lightsmith.backends.pwe import band_gaps
from lightsmith.problems import Ball, BandProblem, Circle, KPoint, Structure, Structure2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "pwe.json")
A = 400.0
AIR = ls.ConstantIndex(n_real=1.0)
SQ = [(A, 0, 0), (0, A, 0)]
SQ_PATH = [KPoint(label="Γ", reduced=(0, 0, 0)), KPoint(label="X", reduced=(0.5, 0, 0)),
           KPoint(label="M", reduced=(0.5, 0.5, 0)), KPoint(label="Γ", reduced=(0, 0, 0))]


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def gpu_snapshot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True).strip()
        u, m, t = (float(v) for v in out.split(","))
        snap = {"time": now(), "gpu_util_percent": u, "mem_used_mib": m, "mem_total_mib": t}
        try:  # the CPU is shared too (other jobs on the machine): record its load
            load = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                            "(Get-CimInstance Win32_Processor).LoadPercentage"], text=True,
                                           timeout=30).strip()
            snap["cpu_load_percent"] = float(load.split()[0])
        except Exception:
            pass
        return snap
    except Exception as e:  # pragma: no cover
        return {"time": now(), "error": str(e)}


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=HERE).strip()
    except Exception:
        return "unknown"


def rods(eps=8.9, r=0.2, pol="tm", pps=6, nb=4, **opts):
    """tests/test_pwe.py::rods"""
    return BandProblem(lattice_vectors_nm=SQ, background=AIR, k_path=SQ_PATH, points_per_segment=pps,
                       structures_2d=[Structure2D(profile=Circle(radius_nm=r * A),
                                                  material=ls.ConstantPermittivity(eps_real=eps))],
                       num_bands=nb, polarization=pol, options={"pwe": opts})


def diamond(resolution=16):
    """tests/test_pwe.py::test_diamond_lattice_complete_gap_3d"""
    ac = 500.0
    fcc = [(0, ac / 2, ac / 2), (ac / 2, 0, ac / 2), (ac / 2, ac / 2, 0)]
    pts = {"X": (0.5, 0, 0.5), "U": (0.625, 0.25, 0.625), "L": (0.5, 0.5, 0.5), "Γ": (0, 0, 0),
           "W": (0.5, 0.25, 0.75), "K": (0.375, 0.375, 0.75)}
    si = ls.ConstantPermittivity(eps_real=11.56)
    return BandProblem(dimension=3, lattice_vectors_nm=fcc, background=AIR, num_bands=5, points_per_segment=3,
                       k_path=[KPoint(label=lab, reduced=pts[lab]) for lab in ["X", "U", "L", "Γ", "X", "W", "K"]],
                       structures=[Structure(geometry=Ball(center_nm=(s * ac / 8,) * 3, radius_nm=0.25 * ac),
                                             material=si) for s in (1, -1)],
                       options={"pwe": {"resolution": resolution}})


def timed(fn, repeats, warmup=True):
    if warmup:
        fn()
    ts, out = [], None
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts, out


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main():
    from _machine import NOTE, state

    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--resolutions", default="16")
    args = ap.parse_args()
    have_gpu = torch.cuda.is_available()
    import scipy

    out = {"script": "benchmarks/remeasure/pwe.py", "timestamp": now(), "git_commit": git_rev(),
           "cpu": "AMD Ryzen 5 5600X (" + platform.processor() + ")",
           "gpu": torch.cuda.get_device_name(0) if have_gpu else None,
           "torch_cpu_threads": torch.get_num_threads(),
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "torch": torch.__version__},
           "repeats": args.repeats,
           "note": NOTE, "machine_state": {"before": state()}}
    if not have_gpu:
        out["error"] = "no CUDA device"
        json.dump(out, open(OUT, "w"), indent=1)
        return

    # --- 2D agreement (test config) -------------------------------------------------------
    p2 = rods(pol="both", pps=4)
    snap2 = gpu_snapshot()
    t_cpu2, cpu = timed(lambda: ls.solve(p2, "pwe", "cpu"), args.repeats)
    t_gpu2, gpu = timed(lambda: ls.solve(p2, "pwe", "gpu"), args.repeats)
    d_tm = float(np.max(np.abs(np.asarray(gpu.frequencies["tm"]) - np.asarray(cpu.frequencies["tm"]))))
    d_te = float(np.max(np.abs(np.asarray(gpu.frequencies["te"]) - np.asarray(cpu.frequencies["te"]))))
    agreement = {
        "configuration": "tests/test_pwe.py::test_gpu_matches_cpu: square lattice a=400 nm, eps=8.9 rods r=0.2a in "
                         "air, Gamma-X-M-Gamma, 4 points/segment (13 k), 4 bands, TE+TM, default options",
        "estimate_cpu": jsonable(ls.estimate(p2, method="pwe", device="cpu")),
        "estimate_gpu": jsonable(ls.estimate(p2, method="pwe", device="gpu")),
        "provenance_cpu": jsonable({k: cpu.provenance.get(k) for k in ("solver", "precision", "matrix_size",
                                                                       "num_plane_waves", "framework", "device")}),
        "provenance_gpu": jsonable({k: gpu.provenance.get(k) for k in ("solver", "precision", "matrix_size",
                                                                       "num_plane_waves", "framework", "device")}),
        "max_abs_diff_tm_a_over_lambda": d_tm,
        "max_abs_diff_te_a_over_lambda": d_te,
        "cpu_wall_s": t_cpu2, "gpu_wall_s": t_gpu2,
        "speedup_wall_median": statistics.median(t_cpu2) / statistics.median(t_gpu2),
        "gpu_snapshot": snap2,
    }
    print("2D agreement", d_tm, d_te, agreement["speedup_wall_median"], flush=True)

    # --- 3D timing ------------------------------------------------------------------------
    cases = []
    for res in (int(r) for r in args.resolutions.split(",")):
        p3 = diamond(res)
        est_c = ls.estimate(p3, method="pwe", device="cpu")
        est_g = ls.estimate(p3, method="pwe", device="gpu")
        rec = {"resolution": res, "estimate_cpu": jsonable(est_c), "estimate_gpu": jsonable(est_g)}
        try:
            # CPU: one untimed warm-up is skipped for long runs; first run timed separately.
            t_c, rc = timed(lambda: ls.solve(p3, "pwe", "cpu"), args.repeats, warmup=False)
            torch.cuda.empty_cache()
            snap = gpu_snapshot()
            t_g, rg = timed(lambda: ls.solve(p3, "pwe", "gpu"), args.repeats, warmup=True)
            fc, fg = np.asarray(rc.frequencies["all"]), np.asarray(rg.frequencies["all"])
            gc, gg = band_gaps(rc)["all"], band_gaps(rg)["all"]
            rec.update({
                "cpu_wall_s": t_c, "gpu_wall_s": t_g,
                "cpu_wall_median_s": statistics.median(t_c), "gpu_wall_median_s": statistics.median(t_g),
                "speedup_wall_median": statistics.median(t_c) / statistics.median(t_g),
                "max_abs_diff_a_over_lambda": float(np.max(np.abs(fg - fc))),
                "gap_cpu": jsonable(gc), "gap_gpu": jsonable(gg),
                "provenance_cpu": jsonable({k: v for k, v in rc.provenance.items() if not isinstance(v, (list,))}),
                "provenance_gpu": jsonable({k: v for k, v in rg.provenance.items() if not isinstance(v, (list,))}),
                "gpu_snapshot": snap,
            })
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:400]
        cases.append(rec)
        print("3D", res, {k: rec.get(k) for k in ("cpu_wall_median_s", "gpu_wall_median_s", "speedup_wall_median",
                                                  "max_abs_diff_a_over_lambda", "error")}, flush=True)
        out_tmp = dict(out, partial=True, cases_3d=cases)
        json.dump(jsonable(out_tmp), open(OUT, "w"), indent=1, default=str)

    ok = [c for c in cases if "speedup_wall_median" in c]
    main_case = ok[0] if ok else None
    entry = {
        "row": "GPU vs. CPU | CPU | 2e-4 (TM, single) and 1e-6 (TE) bound; ~17x faster in 3D reported",
        "configuration": "agreement: test_gpu_matches_cpu (2D); timing: test_diamond_lattice_complete_gap_3d "
                         "(3D diamond, eps=11.56, 19 k-points, 5 bands, LOBPCG double)",
        "measured": {
            "agreement_2d_tm_single": d_tm, "agreement_2d_te_double": d_te,
            "speedup_3d": {str(c["resolution"]): c["speedup_wall_median"] for c in ok},
            "basis_3d": {str(c["resolution"]): {"num_plane_waves": c["estimate_cpu"].get("num_plane_waves"),
                                                "matrix_size": c["estimate_cpu"].get("matrix_size"),
                                                "solver": c["estimate_cpu"].get("solver"),
                                                "precision_cpu": c["estimate_cpu"].get("precision"),
                                                "precision_gpu": c["estimate_gpu"].get("precision")} for c in cases},
            "max_abs_diff_3d": {str(c["resolution"]): c["max_abs_diff_a_over_lambda"] for c in ok},
        },
        "reference": "CPU (NumPy/SciPy) solve of the same problem",
        "previously_reported": {"speedup_3d": "~17x"},
        "reproduced": {"agreement_within_bounds": bool(d_tm <= 2e-4 and d_te <= 1e-6),
                       "speedup_~17x": bool(main_case and 12 <= main_case["speedup_wall_median"] <= 24)},
        "error_column": (f"{d_tm:.0e} (TM, single); {d_te:.0e} (TE); 3D " + ", ".join(
            f"{c['speedup_wall_median']:.3g}x (res {c['resolution']}, N={c['estimate_cpu'].get('matrix_size')})"
            for c in ok)),
    }
    out["machine_state"]["after"] = state()
    out.update(timestamp=now(), agreement_2d=agreement, cases_3d=cases, entries=[entry])
    json.dump(jsonable(out), open(OUT, "w"), indent=1, default=str)
    print(json.dumps(jsonable(entry), indent=1, default=str))


if __name__ == "__main__":
    main()
