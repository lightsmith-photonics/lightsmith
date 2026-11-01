"""Re-measure the "reported" values of the FDFD / inverse-design validation table.

Every configuration is copied from tests/test_fdfd.py, tests/test_inverse.py and
examples/inverse_design_bend.py (the full 2x2 um bend). Rows:

1. straight guide |S21|            (test_straight_waveguide_transmits)
2. closed guide power / reciprocity  (test_power_conservation_closed_guide)
3. adjoint gradient vs. FD          (test_adjoint_gradient_matches_finite_differences)
4. GPU block LU vs. CPU SuperLU     (test_gpu_matches_cpu, test_block_solver_matches_scipy_on_cpu, plus timings)
5. inverse-designed bend            (test_bend_topology_optimisation_improves_monotonically and the example)

    uv run --no-sync python benchmarks/remeasure/fdfd_inverse.py [--rows 1,2,3,4,5] [--repeats 3]

Results are merged into benchmarks/remeasure/results/fdfd_inverse.json (rows not rerun are kept).
GPU timings are provisional when another job shares the GPU (nvidia-smi snapshots are recorded).
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
import tempfile
import time

import numpy as np
import torch

import lightsmith as ls
from lightsmith import inverse
from lightsmith.backends._fdfd import linsolve
from lightsmith.backends.fdfd import FDFD
from lightsmith.problems import DeviceProblem, Polygon, Port, Prism, Rectangle, Structure  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "fdfd_inverse.json")
SI = ls.ConstantIndex(n_real=3.48)
SIO2 = ls.ConstantIndex(n_real=1.44)
SIN = ls.ConstantIndex(n_real=2.0)


# --- helpers ---------------------------------------------------------------------------


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


def fmax(a):
    return float(np.max(np.asarray(a)))


def prism(profile, material=SI):
    return Structure(geometry=Prism(profile=profile, z_min_nm=-110, z_max_nm=110), material=material)


def rect(cx, cy, sx, sy, material=SI):
    return prism(Rectangle(center_nm=(cx, cy), size_nm=(sx, sy)), material)


def straight(pol="te", dl=25.0, width=500.0, wavelengths=(1500.0, 1550.0, 1600.0), num_modes=1, length=2400.0):
    """tests/test_fdfd.py::straight"""
    return DeviceProblem(
        dimension=2, background=SIO2, structures=[rect(0, 0, 20000, width)],
        domain_size_nm=(length + 600, 2500, 0), wavelengths=list(wavelengths),
        ports=[Port(name="o1", center_nm=(-length / 2, 0, 0), size_nm=(0, 2000, 1), direction="+", num_modes=num_modes),
               Port(name="o2", center_nm=(length / 2, 0, 0), size_nm=(0, 2000, 1), direction="-", num_modes=num_modes)],
        options={"fdfd": {"polarization": pol, "dl_nm": dl}})


# --- row 1 -----------------------------------------------------------------------------


def row1():
    per_pol = {}
    worst = 0.0
    for pol in ("te", "tm"):
        prob = straight(pol)
        res = ls.solve(prob, method="fdfd", device="cpu")
        S = res.s()
        eng = FDFD.engine(prob)
        L = (eng.ports[1].i0 - eng.ports[0].i0) * eng.grid.dl
        phase = []
        for k, lam in enumerate(eng.wavelengths):
            beta_dl = eng.modes(eng.base_eps(lam), lam)[0][0].beta_dl.real
            phase.append(float(abs(np.angle(S[k, 1, 0] * np.exp(-1j * beta_dl * L / eng.grid.dl)))))
        e21 = np.abs(np.abs(S[:, 1, 0]) - 1)
        e12 = np.abs(np.abs(S[:, 0, 1]) - 1)
        worst = max(worst, fmax(e21), fmax(e12))
        per_pol[pol] = {"wavelengths_nm": res.wavelengths_nm, "abs_S21_minus_1": e21.tolist(),
                        "abs_S12_minus_1": e12.tolist(), "max_abs_S11_S22": fmax(np.abs(S[:, [0, 1], [0, 1]])),
                        "phase_error_rad": phase, "solver": res.provenance["solver"], "warnings": res.warnings,
                        "grid": res.provenance["grid"]}
    return {
        "row": "Straight guide, |S21| | 1 | 1e-4 bound; 2e-9 reported",
        "configuration": "tests/test_fdfd.py::test_straight_waveguide_transmits: Si (n=3.48) 500 nm guide in SiO2 "
                         "(n=1.44), port planes 2400 nm apart, domain 3000x2500 nm, dl=25 nm, 16 PML cells, TE and TM, "
                         "1500/1550/1600 nm, CPU SciPy SuperLU, double precision",
        "measured": {"max_abs_|S21|_minus_1_over_pols_wavelengths_and_directions": worst, "per_polarization": per_pol},
        "reference": "|S21| = 1 (lossless straight guide)",
        "previously_reported": 2e-9,
        "reproduced": bool(worst <= 2e-9 * 5),  # same order of magnitude counts as reproduced
        "error_column": f"{worst:.0e}",
    }


# --- row 2 -----------------------------------------------------------------------------


def row2():
    per_pol = {}
    pw, rc = 0.0, 0.0
    for pol in ("te", "tm"):
        prob = DeviceProblem(
            dimension=2, background=SIO2, structures=[rect(100, 150, 300, 400, SIN), rect(-150, -250, 200, 300, SI)],
            domain_size_nm=(2800, 1000, 0), wavelengths=[1200.0, 1300.0],
            ports=[Port(name="a", center_nm=(-1100, 0, 0), size_nm=(0, 1200, 1), direction="+", num_modes=2),
                   Port(name="b", center_nm=(1100, 0, 0), size_nm=(0, 1200, 1), direction="-", num_modes=2)],
            options={"fdfd": {"polarization": pol, "dl_nm": 20, "pml_cells": [16, 0]}})
        S = ls.solve(prob, method="fdfd", device="cpu").s()
        p = [fmax(np.abs((np.abs(Sk) ** 2).sum(axis=0) - 1)) for Sk in S]
        r = [fmax(np.abs(Sk - Sk.T)) for Sk in S]
        real = [bool(np.abs(Sk[2:, :2]).max() < 0.99 and np.abs(Sk[:2, :2]).max() > 0.05) for Sk in S]
        per_pol[pol] = {"wavelengths_nm": [1200.0, 1300.0], "power_error": p, "reciprocity_error": r,
                        "real_scattering_check": real}
        pw, rc = max(pw, *p), max(rc, *r)
    return {
        "row": "Closed guide with scatterers | power conservation; reciprocity | 1e-6; 1e-8 bound; 2e-8 (power) reported",
        "configuration": "tests/test_fdfd.py::test_power_conservation_closed_guide: SiO2 parallel-plate guide "
                         "(PEC/PMC walls in y, 16 PML cells in x), SiN 300x400 and Si 200x300 nm scatterers, "
                         "domain 2800x1000 nm, two 2-mode ports 2200 nm apart, dl=20 nm, TE and TM, 1200/1300 nm, CPU",
        "measured": {"power_conservation_max_|sum_i|S_ij|^2-1|": pw, "reciprocity_max_|S-S^T|": rc,
                     "per_polarization": per_pol},
        "reference": "S unitary (column power sums = 1) and symmetric",
        "previously_reported": {"power": 2e-8, "reciprocity": None},
        "reproduced": {"power": bool(pw <= 2e-8 * 5 and pw >= 2e-8 / 5)},
        "error_column": f"{pw:.0e} (power); {rc:.0e} (reciprocity)",
    }


# --- row 3 -----------------------------------------------------------------------------


def row3():
    per_pol = {}
    worst = 0.0
    hs = (1e-3, 1e-4, 1e-5)
    for pol in ("te", "tm"):
        prob = straight(pol, dl=40.0, wavelengths=[1500.0, 1600.0], length=1200.0)
        ctx = FDFD.prepare(prob)
        base = torch.as_tensor(FDFD.base_permittivity(prob))
        pert = torch.as_tensor(np.random.default_rng(1).uniform(0, 3, base.shape[1:]))
        mask = torch.zeros(base.shape[1:], dtype=torch.float64)
        mask[25:45, 20:50] = 1

        def objective(p):
            sol = FDFD.differentiable_solve(prob, base + p * mask, "cpu", sources=["o1"], **ctx)
            return (sol.sparam("o2", "o1").abs() ** 2).sum() + 0.5 * (sol.fields[:, 0, 35, 35].abs() ** 2).sum()

        x = pert.clone().requires_grad_(True)
        objective(x).backward()
        g = x.grad.numpy()
        pts = [(30, 30), (35, 37), (40, 25), (28, 45)]
        rel = {}
        for h in hs:
            rel[h] = []
            for i, j in pts:
                xp, xm = pert.clone(), pert.clone()
                xp[i, j] += h
                xm[i, j] -= h
                with torch.no_grad():
                    fd = (float(objective(xp)) - float(objective(xm))) / (2 * h)
                rel[h].append({"point": [i, j], "adjoint": float(g[i, j]), "fd": fd,
                               "rel_err": abs(g[i, j] - fd) / abs(fd)})
        m = max(r["rel_err"] for r in rel[1e-4])
        worst = max(worst, m)
        per_pol[pol] = {"max_rel_err_h1e-4": m,
                        "max_rel_err_by_h": {f"{h:g}": max(r["rel_err"] for r in rel[h]) for h in hs},
                        "points_h1e-4": rel[1e-4]}
    return {
        "row": "Adjoint gradient | finite differences | 1e-3 relative bound; 1e-6 reported",
        "configuration": "tests/test_fdfd.py::test_adjoint_gradient_matches_finite_differences: straight guide, "
                         "dl=40 nm, port planes 1200 nm apart, 1500/1600 nm, TE and TM; objective sum|S21|^2 + "
                         "0.5 sum|Hz or Ez(35,35)|^2; random pixel perturbation (seed 1) in mask [25:45, 20:50]; central "
                         "differences h=1e-4 at pixels (30,30),(35,37),(40,25),(28,45); CPU, double",
        "measured": {"max_rel_err_h1e-4": worst, "per_polarization": per_pol},
        "reference": "central finite differences (h = 1e-4; h = 1e-3 and 1e-5 also listed)",
        "previously_reported": 1e-6,
        "reproduced": bool(worst <= 1e-6 * 5),
        "error_column": f"{worst:.0e}",
    }


# --- row 4 -----------------------------------------------------------------------------


def _time(fn, repeats):
    fn()  # warm-up (CUDA context, kernels, caches)
    ts = []
    out = None
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts, out


def row4(repeats):
    have_gpu = torch.cuda.is_available()
    entry = {
        "row": "GPU block-tridiagonal LU vs. CPU SuperLU | CPU | 1e-10 bound; 2-5x faster on large grids reported",
        "previously_reported": {"agreement_bound": 1e-10, "speedup": "2-5x on large grids"},
        "reference": "CPU SciPy SuperLU (double)",
    }
    # (a) block LU vs SuperLU on CPU (test_block_solver_matches_scipy_on_cpu)
    import array_api_compat.torch as xp

    cpu_block = []
    for size in ((1600, 1000, 0), (1000, 1600, 0)):
        prob = DeviceProblem(dimension=2, background=SIO2, structures=[rect(0, 0, 400, 300)],
                             domain_size_nm=size, wavelengths=[1550.0],
                             ports=[Port(name="o1", center_nm=(-300, 0, 0), size_nm=(0, 600, 1), direction="+")],
                             options={"fdfd": {"dl_nm": 50, "pml_cells": 8}})
        eng = FDFD.engine(prob)
        eps = eng.base_eps(1550.0)
        d = eng.diagonals(xp, torch.as_tensor(eps), 1550.0, device=torch.device("cpu"))
        d = linsolve.Diagonals(*(torch.broadcast_to(v, eps.shape).contiguous() for v in d))
        B = torch.as_tensor(np.random.default_rng(0).normal(size=eps.shape + (2,)) + 0j)
        sc, bl = linsolve.ScipyFactor(d), linsolve.BlockFactor(d)
        a, b = sc.solve(B).numpy(), bl.solve(B).numpy()
        at, bt = sc.solve_t(B).numpy(), bl.solve_t(B).numpy()
        cpu_block.append({"domain_nm": list(size[:2]), "max_abs_diff_solve": fmax(np.abs(a - b)),
                          "max_rel_diff_solve": fmax(np.abs(a - b)) / fmax(np.abs(a)),
                          "max_abs_diff_solve_t": fmax(np.abs(at - bt))})
    entry["block_lu_on_cpu_vs_superlu"] = cpu_block
    if not have_gpu:
        entry["error"] = "no CUDA device"
        return entry
    entry["gpu_snapshot_before"] = gpu_snapshot()
    # (b) GPU vs CPU agreement (test_gpu_matches_cpu)
    prob = straight("te", dl=30, wavelengths=[1550.0], length=1200.0)
    s_cpu = ls.solve(prob, method="fdfd", device="cpu")
    s_gpu = ls.solve(prob, method="fdfd", device="cuda")
    s_diff = fmax(np.abs(s_gpu.s() - s_cpu.s()))
    base = torch.as_tensor(FDFD.base_permittivity(prob))
    grads = []
    for dev in ("cpu", "cuda"):
        x = (base.real * 0 + 0.5).to(dev).requires_grad_(True)
        sol = FDFD.differentiable_solve(prob, base.to(dev) + x, dev, sources=["o1"])
        (sol.sparam("o2", "o1").abs() ** 2).sum().backward()
        grads.append(x.grad.cpu().numpy())
    g_rel = fmax(np.abs(grads[1] - grads[0])) / fmax(np.abs(grads[0]))
    entry["agreement_test_config"] = {
        "configuration": "tests/test_fdfd.py::test_gpu_matches_cpu: straight TE guide, dl=30 nm, port planes 1200 nm "
                         "apart, 1550 nm; GPU solver " + str(s_gpu.provenance["solver"]) + ", double precision",
        "max_abs_S_gpu_minus_S_cpu": s_diff,
        "max_abs_grad_diff_over_max_abs_grad": g_rel,
        "grid": s_cpu.provenance["grid"],
    }
    # (c) timings on growing grids: straight TE guide (test geometry), one wavelength, double precision.
    timings = []
    for dl in (25.0, 15.0, 10.0, 7.5, 6.0):
        prob = straight("te", dl=dl, wavelengths=[1550.0])
        est = ls.estimate(prob, method="fdfd", device="cpu")
        rec = {"dl_nm": dl, "grid": est["grid"], "unknowns": est["unknowns"],
               "block_lu_memory_bytes_estimate": est["memory_bytes_block"]}
        try:
            tc, rc = _time(lambda: ls.solve(prob, method="fdfd", device="cpu"), repeats)
            rec["cpu_wall_s"] = tc
            rec["cpu_solver"] = rc.provenance["solver"]
            rec["cpu_solver_runtime_s"] = rc.provenance["runtime_s"]
            torch.cuda.empty_cache()
            snap = gpu_snapshot()
            tg, rg = _time(lambda: ls.solve(prob, method="fdfd", device="cuda"), repeats)
            rec["gpu_snapshot"] = snap
            rec["gpu_wall_s"] = tg
            rec["gpu_solver"] = rg.provenance["solver"]
            rec["gpu_solver_runtime_s"] = rg.provenance["runtime_s"]
            rec["cpu_wall_median_s"] = statistics.median(tc)
            rec["gpu_wall_median_s"] = statistics.median(tg)
            rec["speedup_wall_median"] = rec["cpu_wall_median_s"] / rec["gpu_wall_median_s"]
            rec["max_abs_S_gpu_minus_S_cpu"] = fmax(np.abs(rg.s() - rc.s()))
            rec["gpu_peak_mem_bytes"] = int(torch.cuda.max_memory_allocated())
        except Exception as e:  # e.g. CUDA OOM while the GPU is shared
            rec["error"] = f"{type(e).__name__}: {e}"[:400]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        timings.append(rec)
        print("  row4 timing", {k: rec.get(k) for k in ("dl_nm", "unknowns", "cpu_wall_median_s",
                                                         "gpu_wall_median_s", "speedup_wall_median", "error")},
              flush=True)
    entry["timings"] = {
        "configuration": "straight TE guide of tests/test_fdfd.py (3000x2500 nm domain + 16 PML cells), 1550 nm, "
                         "dl 25/15/10/7.5/6 nm; wall time of ls.solve (rasterisation, port modes, factorisation, "
                         "solves, S extraction); 1 warm-up + repeats, median",
        "repeats": repeats, "precision": "double (complex128) on both devices",
        "cpu": "AMD Ryzen 5 5600X, SciPy SuperLU (single-threaded factorisation)",
        "gpu": torch.cuda.get_device_name(0) + ", torch block-tridiagonal LU",
        "provisional": "GPU shared with a long background benchmark during the measurement; speedups provisional",
        "cases": timings,
    }
    entry["gpu_snapshot_after"] = gpu_snapshot()
    ok = [t for t in timings if "speedup_wall_median" in t]
    max_s_diff = max([s_diff] + [t["max_abs_S_gpu_minus_S_cpu"] for t in ok])
    entry["measured"] = {"max_abs_S_gpu_minus_S_cpu_all_cases": max_s_diff,
                         "speedup_by_unknowns": {str(t["unknowns"]): round(t["speedup_wall_median"], 3) for t in ok}}
    entry["configuration"] = "agreement: test_gpu_matches_cpu config + all timing grids; timings: see 'timings'"
    sp = [t["speedup_wall_median"] for t in ok]
    entry["reproduced"] = {"agreement_1e-10": bool(max_s_diff < 1e-10),
                           "speedup_2-5x_on_large_grids": bool(sp and max(sp) >= 2) if sp else False}
    entry["error_column"] = f"{max_s_diff:.0e}; speedup " + ", ".join(
        f"{t['speedup_wall_median']:.2g}x ({t['unknowns'] / 1e3:.0f}k)" for t in ok) + " (provisional)"
    return entry


# --- row 5 -----------------------------------------------------------------------------


def wg(cx, cy, sx, sy):
    return Structure(geometry=Prism(profile=Rectangle(center_nm=(cx, cy), size_nm=(sx, sy)), z_min_nm=-110,
                                    z_max_nm=110), material=SI)


def bend_spec_test(method="mma", max_iter=12, dl=40.0, half=600.0, **opt):
    """tests/test_inverse.py::bend_spec"""
    dev = DeviceProblem(
        dimension=2, background=SIO2, structures=[wg(-5000 - half, 0, 10000, 400), wg(0, 5000 + half, 400, 10000)],
        domain_size_nm=(2 * half + 1400, 2 * half + 1400, 0), wavelengths=[1550.0],
        ports=[Port(name="o1", center_nm=(-half - 450, 0, 0), size_nm=(0, 1500, 1), direction="+"),
               Port(name="o2", center_nm=(0, half + 450, 0), size_nm=(1500, 0, 1), direction="-")],
        options={"fdfd": {"dl_nm": dl, "pml_cells": 10}})
    return inverse.OptimizationProblem(
        device=dev,
        design_regions=[inverse.DesignRegion(size_nm=(2 * half, 2 * half, 220), material_low=SIO2, material_high=SI,
                                             filter_radius_nm=80)],
        objective=[inverse.SParameterTerm(out_port="o2", in_port="o1")],
        optimizer=inverse.OptimizerSettings(method=method, max_iter=max_iter, beta_every=1000, **opt))


def bend_spec_example(iters=60):
    """examples/inverse_design_bend.py::build_problem (2 x 2 um region, 100 nm minimum feature)."""
    W, HALF = 450.0, 1000.0
    device = DeviceProblem(
        dimension=2, background=SIO2,
        structures=[wg(-5000 - HALF, 0, 10000, W), wg(0, 5000 + HALF, W, 10000)],
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


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + 1e-9


def min_feature_estimate(binary, dl):
    """Largest disc diameter (pixels, odd 2r+1) whose morphological opening leaves solid (closing leaves void)
    unchanged; the design is edge-padded so features continuing outside the region are not penalised."""
    from scipy import ndimage

    out = {}
    for name, img in (("solid", binary), ("void", ~binary)):
        best, changed = 1, {}
        for r in range(1, 9):
            pad = np.pad(img, r + 1, mode="edge")
            op = ndimage.binary_opening(pad, structure=_disk(r))[r + 1:-r - 1, r + 1:-r - 1]
            n = int(np.sum(op != img))
            changed[f"{(2 * r + 1) * dl:g}nm"] = n
            if n == 0:
                best = 2 * r + 1
        out[name] = {"largest_disc_diameter_preserving_all_features_nm": best * dl,
                     "pixels_changed_by_opening_with_disc_diameter": changed}
    return out


def segment_monotonicity(history):
    """MMA is monotone in the loss (-objective + lengthscale penalty) at fixed beta; check per beta segment."""
    beta = [h["beta"] for h in history]
    obj = np.array([h["objective"] for h in history])
    loss = np.array([-h["objective"] + h.get("penalty", 0.0) for h in history])
    same = np.array(beta[1:]) == np.array(beta[:-1])
    d_obj, d_loss = np.diff(obj)[same], np.diff(loss)[same]
    return {"beta_history": beta, "penalty_history": [h.get("penalty", 0.0) for h in history],
            "objective_monotone_within_fixed_beta_segments": bool(np.all(d_obj >= -1e-9)),
            "loss_monotone_within_fixed_beta_segments": bool(np.all(d_loss <= 1e-9)),
            "largest_objective_drop_within_fixed_beta": float(-min(d_obj.min(), 0)) if d_obj.size else 0.0,
            "largest_loss_rise_within_fixed_beta": float(max(d_loss.max(), 0)) if d_loss.size else 0.0}


def run_bend(spec, label, tmp):
    seen = []
    ck = os.path.join(tmp, f"{label}.npz")
    t0 = time.perf_counter()
    res = inverse.optimize(spec, device="cpu", callback=lambda s: seen.append(s.objective), checkpoint=ck)
    runtime = time.perf_counter() - t0
    obj = [h["objective"] for h in res.history]
    S = res.sparameters.s()
    wls = res.sparameters.wavelengths_nm
    i1, i2 = res.sparameters.ports.index("o1"), res.sparameters.ports.index("o2")
    exported = {"wavelengths_nm": wls,
                "|S21|^2": (np.abs(S[:, i2, i1]) ** 2).tolist(),
                "|S11|^2": (np.abs(S[:, i1, i1]) ** 2).tolist(),
                "|S22|^2": (np.abs(S[:, i2, i2]) ** 2).tolist(),
                "reciprocity_|S21-S12|": np.abs(S[:, i2, i1] - S[:, i1, i2]).tolist()}
    # Pixel designs (final beta and binary beta=inf) re-solved for per-wavelength |S21|^2, |S11|^2.
    x, beta, _, _, _ = inverse.load_checkpoint(ck)
    topt = inverse.TopologyOptimizer(spec, "cpu")
    pixel = {}
    for name, b in (("final_beta", beta), ("binary", math.inf)):
        with torch.no_grad():
            eps, _, _ = topt.permittivity(torch.as_tensor(x, device=topt.tdev), b)
            sol = topt.solve(eps)
            s21 = sol.sparam("o2", "o1").cpu().numpy()
            s11 = sol.sparam("o1", "o1").cpu().numpy()
        pixel[name] = {"beta": b if math.isfinite(b) else "inf", "|S21|^2": (np.abs(s21) ** 2).tolist(),
                       "|S11|^2": (np.abs(s11) ** 2).tolist()}
    d = res.designs[0]
    dens = np.asarray(d.density)
    dl = float(d.x_nm[1] - d.x_nm[0])
    return res, {
        "label": label,
        "device": "cpu (SciPy SuperLU), double",
        "runtime_s": runtime,
        "grid": res.provenance.get("grid"), "dl_nm": res.provenance.get("dl_nm"),
        "evaluations": res.provenance.get("evaluations"), "final_beta": res.provenance.get("final_beta"),
        "iterations": len(obj),
        "objective_history": obj,
        "callback_matches_history": obj == seen,
        "min_step_objective_diff": float(np.min(np.diff(obj))) if len(obj) > 1 else None,
        "monotone_within_1e-9": bool(np.all(np.diff(obj) >= -1e-9)),
        **segment_monotonicity(res.history),
        "objective_initial": res.objective_initial, "objective_final": res.objective_final,
        "objective_binary": res.objective_binary,
        "final_terms": res.history[-1].get("terms"), "final_binarization": res.history[-1].get("binarization"),
        "pixel_design": pixel,
        "exported_polygons": exported,
        "abs_exported_|S21|^2_minus_objective_binary": (float(abs(np.abs(S[0, i2, i1]) ** 2 - res.objective_binary))
                                                         if len(wls) == 1 else None),
        "min_feature_setting_nm": spec.design_regions[0].min_feature_nm,
        "filter_radius_nm": spec.design_regions[0].filter_radius_nm,
        "measured_min_feature_pixel_estimate": min_feature_estimate(dens > 0.5, dl),
        "warnings": res.warnings,
    }


def row5(iters_full):
    out = {"row": "Inverse design: 2x2 um Si bend | --- | objective rises monotonically, final >0.8, exported polygons "
                  "within 0.15 (bound); |S21|^2=0.98 with 100 nm minimum features reported",
           "caption_note": "|S21|^2=0.987 (0.981-0.987 after export), |S11|^2<=2.5e-4 and reciprocity 8e-15 are "
                           "not recorded in the repository",
           "previously_reported": {"|S21|^2": 0.987, "|S21|^2_after_export_range": [0.981, 0.987],
                                   "|S11|^2_max": 2.5e-4, "reciprocity": 8e-15, "min_feature_nm": 100}}
    with tempfile.TemporaryDirectory() as tmp:
        _, test_run = run_bend(bend_spec_test(max_iter=12), "test_bend_1.2um", tmp)
        test_run["configuration"] = (
            "tests/test_inverse.py::test_bend_topology_optimisation_improves_monotonically: 1.2x1.2 um design "
            "region, 400 nm guides, 1550 nm only, TE, dl=40 nm, 10 PML cells, filter radius 80 nm, no minimum-"
            "feature constraint, MMA 12 iterations, beta fixed (beta_every=1000), objective |S21|^2, CPU")
        print("  row5 test run done", test_run["objective_final"], test_run["objective_binary"], flush=True)
        _, full_run = run_bend(bend_spec_example(iters_full), "example_bend_2um", tmp)
        full_run["configuration"] = (
            f"examples/inverse_design_bend.py (default --iters {iters_full}): 2x2 um design region, 450 nm guides, "
            "TE, 1500/1550/1600 nm, dl=25 nm, 14 PML cells, filter radius 100 nm, min_feature_nm=100, MMA "
            f"{iters_full} iterations, beta 4->64 (x2 every {max(iters_full // 5, 1)}), objective "
            "min_lambda |S21|^2 - 0.5 |S11|^2, CPU (the example defaults to --device auto)")
    out["test_configuration_run"] = test_run
    out["full_configuration_run"] = full_run
    ex = full_run["exported_polygons"]
    px = full_run["pixel_design"]
    out["configuration"] = "test (1.2 um, 12 it.) and full example (2 um, 3 wavelengths, 100 nm min feature)"
    out["measured"] = {
        "test_monotone": test_run["monotone_within_1e-9"], "test_objective_final": test_run["objective_final"],
        "test_objective_binary": test_run["objective_binary"],
        "test_exported_|S21|^2": test_run["exported_polygons"]["|S21|^2"][0],
        "test_|exported - binary|": test_run["abs_exported_|S21|^2_minus_objective_binary"],
        "full_pixel_final_beta_|S21|^2_per_wl": px["final_beta"]["|S21|^2"],
        "full_pixel_binary_|S21|^2_per_wl": px["binary"]["|S21|^2"],
        "full_exported_|S21|^2_per_wl": ex["|S21|^2"],
        "full_exported_|S11|^2_max": max(ex["|S11|^2"]),
        "full_exported_reciprocity_max": max(ex["reciprocity_|S21-S12|"]),
        "full_min_feature_setting_nm": full_run["min_feature_setting_nm"],
        "full_min_feature_pixel_estimate_nm": {k: v["largest_disc_diameter_preserving_all_features_nm"]
                                               for k, v in full_run["measured_min_feature_pixel_estimate"].items()},
    }
    m = out["measured"]
    out["reproduced"] = {
        "test_bounds (monotone, final>0.8, export within 0.15)": bool(
            m["test_monotone"] and m["test_objective_final"] > 0.8 and m["test_|exported - binary|"] < 0.15),
        "|S21|^2=0.987": bool(abs(max(px["final_beta"]["|S21|^2"]) - 0.987) < 0.005),
        "after export 0.981-0.987": bool(min(ex["|S21|^2"]) >= 0.976 and max(ex["|S21|^2"]) <= 0.992),
        "|S11|^2<=2.5e-4": bool(max(ex["|S11|^2"]) <= 2.5e-4),
        "reciprocity ~8e-15": bool(max(ex["reciprocity_|S21-S12|"]) < 1e-13),
        "objective rises monotonically (full run, within fixed-beta segments)":
            full_run["objective_monotone_within_fixed_beta_segments"],
    }
    out["reference"] = "test bounds; exported polygons re-solved with FDFD (CPU)"
    out["notes"] = ("min feature: 100 nm is a soft lengthscale penalty (weight 100, from beta>=16); the pixel check is "
                    "a coarse morphological opening on the dl grid (also counts rounded corners), not a proof. "
                    "Reciprocity of the open bend (PML, ports on orthogonal axes) is limited by discretisation; "
                    "8e-15 matches the closed-guide reciprocity (row 2), not the bend.")
    ms = out["measured"]
    out["error_column"] = (f"|S21|^2 = {full_run['objective_final']:.3f} "
                           f"(worst case over 1500-1600 nm; {min(ex['|S21|^2']):.3f}-{max(ex['|S21|^2']):.3f} after "
                           f"export), |S11|^2 <= {ms['full_exported_|S11|^2_max']:.1e}, reciprocity "
                           f"{ms['full_exported_reciprocity_max']:.0e}")
    return out


# --- main -------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,3,4,5")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iters", type=int, default=60, help="iterations of the full example bend (example default 60)")
    args = ap.parse_args()
    rows = [int(r) for r in args.rows.split(",")]
    data = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            data = json.load(fh)
    import scipy

    data.update({
        "script": "benchmarks/remeasure/fdfd_inverse.py",
        "git_commit": git_rev(),
        "cpu": "AMD Ryzen 5 5600X (" + platform.processor() + ")",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "double (complex128) unless stated",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "torch": torch.__version__, "lightsmith": getattr(ls, "__version__", "?")},
        "note": "Timings provisional: a long benchmark was running on the same GPU and the CPU was at ~100% load from other jobs (see snapshots: gpu_util_percent, cpu_load_percent); repeat on an idle machine before quoting speedups.",
    })
    data.setdefault("entries", {})
    fns = {1: row1, 2: row2, 3: row3, 4: lambda: row4(args.repeats), 5: lambda: row5(args.iters)}
    for r in rows:
        print(f"row {r} ...", flush=True)
        t0 = time.perf_counter()
        e = fns[r]()
        e["timestamp"] = now()
        e["script_runtime_s"] = time.perf_counter() - t0
        data["entries"][str(r)] = e
        data["timestamp"] = now()
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as fh:
            json.dump(data, fh, indent=1, default=float)
        print(f"row {r}: {e.get('error_column', '')} measured={json.dumps(e.get('measured'), default=float)[:600]}",
              flush=True)


if __name__ == "__main__":
    main()
