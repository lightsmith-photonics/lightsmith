"""Re-measure the TMM/RCWA "GPU vs. CPU" validation row.

Row: "GPU vs. CPU | CPU | 1e-12 (TMM), 1e-9 (RCWA) bound; GPU 2-8x faster reported".

For each problem: max |R_gpu - R_cpu| (the agreement the tests bound) and wall-clock timings on CPU
(NumPy) and GPU (PyTorch CUDA), after warm-up, as medians over repeats. Two timings are recorded:
  * end_to_end: ls.solve(problem, method, device), i.e. what a user sees (includes building the
    SpectrumResult and copying results to host lists);
  * kernel: tmm_tensors / rcwa_tensors with torch.cuda.synchronize(), i.e. the solver core.
The first two problems are exactly the configurations of tests/test_tmm.py::test_tmm_gpu_matches_cpu
and tests/test_rcwa.py::test_rcwa_gpu_matches_cpu; the rest are larger batched problems.

Run from the worktree root: uv run --no-sync python benchmarks/remeasure/tmm_rcwa.py [--threads N]

--threads N (default 6 = physical cores of the Ryzen 5 5600X) caps OpenBLAS/OpenMP/torch CPU threads. On the
shared, fully loaded machine of 2026-11-01 uncapped OpenBLAS oversubscribed badly (a 5 x 162 x 162 inverse took
minutes), so all timings use a fixed cap; pass --threads 0 to leave the libraries' defaults.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _threads_arg() -> int:
    if "--threads" in sys.argv:
        return int(sys.argv[sys.argv.index("--threads") + 1])
    return 6


THREADS = _threads_arg()
if THREADS > 0:  # must be set before numpy/torch load their BLAS
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(THREADS)

import numpy as np  # noqa: E402
import torch  # noqa: E402

if THREADS > 0:
    torch.set_num_threads(THREADS)

import lightsmith as ls  # noqa: E402
from lightsmith.backends.rcwa import RCWAOptions, rcwa_tensors
from lightsmith.backends.tmm import tmm_tensors
from lightsmith.device import resolve_device
from lightsmith.problems import (Circle, Incidence, Layer, LayerStackProblem, PatternedLayer, PeriodicProblem,
                                  Rectangle, Structure2D)

CI, CP = ls.ConstantIndex, ls.ConstantPermittivity
OUT = Path(__file__).resolve().parent / "results" / "tmm_rcwa.json"


# --- problem builders, copied from tests/test_tmm.py and tests/test_rcwa.py -----------------------

def stack(layers=(), sup=1.0, sub=1.5, wl=(500.0,), angles=(0.0,), pols=("s", "p"), **kw):
    mat = lambda n: n if not isinstance(n, (int, float, complex)) else CI(n_real=complex(n).real,  # noqa: E731
                                                                           n_imag=complex(n).imag)
    return LayerStackProblem(superstrate=mat(sup), substrate=mat(sub),
                             layers=[Layer(thickness_nm=d, material=mat(n), **kw) for n, d in layers],
                             wavelengths=list(wl), angles_deg=list(angles), polarizations=list(pols))


def lamellar(period, width, eps_ridge, eps_groove=1.0, h=300.0, wl=(1000.0,), sub=1.5, theta=0.0, phi=0.0,
             a2=None, ridge_len=1.0, **opts):
    ridge = CP(eps_real=complex(eps_ridge).real, eps_imag=complex(eps_ridge).imag)
    return PeriodicProblem(
        lattice_a1_nm=(period, 0), lattice_a2_nm=a2, superstrate=CI(n_real=1.0), substrate=CI(n_real=sub),
        layers=[PatternedLayer(thickness_nm=h, background=CP(eps_real=eps_groove),
                               shapes=[Structure2D(profile=Rectangle(size_nm=(width, ridge_len)), material=ridge)])],
        wavelengths=list(wl), incidence=Incidence(theta_deg=theta, phi_deg=phi), options={"rcwa": opts})


def hole_slab(m, wl, **opts):
    return PeriodicProblem(
        lattice_a1_nm=(400, 0), lattice_a2_nm=(0, 400), superstrate=CI(n_real=1.0), substrate=CI(n_real=1.45),
        layers=[PatternedLayer(thickness_nm=150, background=CI(n_real=2.0),
                               shapes=[Structure2D(profile=Circle(radius_nm=100), material=CI(n_real=1.0))])],
        wavelengths=list(wl), incidence=Incidence(polarizations=["s"]), options={"rcwa": {"harmonics": m, **opts}})


def random_stack(n_layers, wl, angles, seed=0):
    rng = np.random.default_rng(seed)
    return stack([(n, d) for n, d in zip(rng.uniform(1.3, 2.5, n_layers), rng.uniform(20, 200, n_layers))],
                 wl=wl, angles=angles)


# --- timing ------------------------------------------------------------------------------------

def sync():
    torch.cuda.synchronize()


def timed(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
        sync()
    ts = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(ts), "min_s": min(ts), "max_s": max(ts), "all_s": ts}


def gpu_snapshot():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        u, used, tot = (x.strip() for x in q.stdout.strip().split(","))
        return {"utilization_pct": float(u), "memory_used_mib": float(used), "memory_total_mib": float(tot)}
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}


def cpu_load_pct():
    try:
        q = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).LoadPercentage"],
                           capture_output=True, text=True, timeout=30)
        return float(q.stdout.strip())
    except Exception:
        return None


def run_case_retry(*a, **kw):
    """The machine is shared: retry a case if a transient out-of-memory hits (other workloads)."""
    import gc
    for attempt in range(4):
        try:
            r = run_case(*a, **kw)
            r["attempts"] = attempt + 1
            return r
        except MemoryError as e:
            print(f"   MemoryError ({e}); retrying in 60 s", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(60)
    raise RuntimeError("persistent MemoryError")


def run_case(name, method, problem, state, repeats, warmup, test_config=None, pols=("s", "p"), kernel=True):
    cpu_dev, gpu_dev = resolve_device("cpu"), resolve_device("gpu")
    print(f"[{name}] {state}", flush=True)
    t_case = time.perf_counter()
    snap_before = {**gpu_snapshot(), "system_cpu_load_pct": cpu_load_pct()}
    rc, rg = ls.solve(problem, method, "cpu"), ls.solve(problem, method, "gpu")
    assert rg.provenance["framework"] == "torch"
    diff = max(float(np.max(np.abs(np.asarray(rg.r[p]) - np.asarray(rc.r[p])))) for p in pols)
    if method == "tmm":
        k = lambda d: tmm_tensors(problem, d, precision="double")  # noqa: E731
    else:
        opts = RCWAOptions.model_validate(problem.options.get("rcwa", {}))
        k = lambda d: rcwa_tensors(problem, d, options=opts)  # noqa: E731
    t = {
        "cpu_end_to_end": timed(lambda: ls.solve(problem, method, "cpu"), repeats, warmup),
        "gpu_end_to_end": timed(lambda: ls.solve(problem, method, "gpu"), repeats, warmup),
    }
    if kernel:
        t["cpu_kernel"] = timed(lambda: k(cpu_dev), repeats, warmup)
        t["gpu_kernel"] = timed(lambda: k(gpu_dev), repeats, warmup)
    snap_after = {**gpu_snapshot(), "system_cpu_load_pct": cpu_load_pct()}
    sp_e2e = t["cpu_end_to_end"]["median_s"] / t["gpu_end_to_end"]["median_s"]
    sp_k = t["cpu_kernel"]["median_s"] / t["gpu_kernel"]["median_s"] if kernel else None
    bound = 1e-12 if method == "tmm" else 1e-9
    print(f"   max|dR|={diff:.2e} (bound {bound:g})  speedup e2e={sp_e2e:.2f}x kernel={sp_k}  "
          f"cpu={t['cpu_end_to_end']['median_s']:.4f}s gpu={t['gpu_end_to_end']['median_s']:.4f}s "
          f"(case {time.perf_counter() - t_case:.0f}s)", flush=True)
    return {"name": name, "method": method, "test_config": test_config, "state_size": state,
            "precision": rc.provenance.get("precision", "double"),
            "max_abs_diff_R_gpu_vs_cpu": diff, "agreement_bound": bound, "within_bound": diff < bound,
            "timings": t, "speedup_end_to_end": sp_e2e, "speedup_kernel": sp_k,
            "repeats": repeats, "warmup_runs": warmup,
            "warmup_note": "the CPU and GPU solves used for the agreement check run first and also warm up both "
                           "devices" + ("" if kernel else "; kernel-only timing skipped for this (slow) case, "
                                        "RCWA end-to-end and kernel times agree within a few % on the test cases"), "cpu_threads": THREADS or "library default",
            "machine_load_before": snap_before, "machine_load_after": snap_after}


def partial(cases):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".partial.json").write_text(json.dumps({"incomplete": True, "cases": cases}, indent=2))


def main():
    from _machine import NOTE, state

    assert torch.cuda.is_available(), "needs a CUDA GPU"
    before = state()
    cases = []
    done = {}
    if "--resume" in sys.argv and OUT.with_suffix(".partial.json").exists():
        done = {c["name"]: c for c in json.loads(OUT.with_suffix(".partial.json").read_text())["cases"]}
        print("resuming; reusing", list(done), flush=True)
    angles9, wl101 = np.linspace(0, 80, 9), np.linspace(400, 900, 101)

    # TMM. State size = wavelengths x angles x polarizations (independent 2x2 transfer-matrix chains).
    for name, nl, wl, ang, rep, test in [
        ("tmm_test_config", 20, wl101, angles9, 10, "tests/test_tmm.py::test_tmm_gpu_matches_cpu"),
        ("tmm_20L_2001wl_46ang", 20, np.linspace(400, 900, 2001), np.linspace(0, 89, 46), 5, None),
        ("tmm_100L_2001wl_91ang", 100, np.linspace(400, 900, 2001), np.linspace(0, 89, 91), 3, None),
    ]:
        p = random_stack(nl, wl, ang)
        st = {"layers": nl, "wavelengths": len(wl), "angles": len(ang), "polarizations": 2,
              "points": len(wl) * len(ang) * 2}
        cases.append(done.get(name) or run_case_retry(name, "tmm", p, st, rep, 1, test))
        partial(cases)

    # RCWA. State size = harmonics N, S-matrix/eigenproblem size 2N, batched over wavelengths.
    rcwa_cases = [
        ("rcwa_test_config_hole_slab_m4", hole_slab(4, np.linspace(590, 600, 5)), ("s",), 4, 5, 2, 3,
         "tests/test_rcwa.py::test_rcwa_gpu_matches_cpu (part 1)"),
        ("rcwa_test_config_lamellar_metal_m20",
         lamellar(700, 300, complex(-20, 1.5), h=100, wl=[700.0, 800.0], theta=10, harmonics=20), ("s", "p"),
         20, 2, 1, 3, "tests/test_rcwa.py::test_rcwa_gpu_matches_cpu (part 2)"),
        ("rcwa_hole_slab_m6_8wl", hole_slab(6, np.linspace(550, 650, 8)), ("s",), 6, 8, 2, 3, None),
        ("rcwa_lamellar_metal_m50_51wl",
         lamellar(700, 300, complex(-20, 1.5), h=100, wl=np.linspace(500, 1000, 51), theta=10, harmonics=50),
         ("s", "p"), 50, 51, 1, 3, None),
    ]
    for name, p, pols, m, nw, dim, rep, test in rcwa_cases:
        nh = (2 * m + 1) ** dim
        st = {"harmonic_order_M": m, "lattice_dim": dim, "harmonics_N": nh, "matrix_size_2N": 2 * nh,
              "wavelengths": nw, "polarizations": len(pols)}
        big_case = test is None
        cases.append(done.get(name) or run_case_retry(name, "rcwa", p, st, rep, 0 if big_case else 1, test, pols,
                                                      kernel=not big_case))
        partial(cases)

    sp = {c["name"]: round(c["speedup_end_to_end"], 2) for c in cases}
    spk = {c["name"]: round(c["speedup_kernel"], 2) for c in cases if c["speedup_kernel"] is not None}
    tmm_d = max(c["max_abs_diff_R_gpu_vs_cpu"] for c in cases if c["method"] == "tmm")
    rcwa_d = max(c["max_abs_diff_R_gpu_vs_cpu"] for c in cases if c["method"] == "rcwa")
    big = [c for c in cases if c["test_config"] is None]
    result = {
        "row": "GPU vs. CPU | CPU | 1e-12 (TMM), 1e-9 (RCWA) bound; GPU 2-8x faster reported",
        "reference": "CPU (NumPy) solution of the same problem",
        "previously_reported": "GPU 2-8x faster",
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": NOTE,
        "machine_state": {"before": before, "after": state()},
        "hardware": {"gpu": torch.cuda.get_device_name(0), "cpu": "AMD Ryzen 5 5600X (6C/12T)",
                     "platform": platform.platform()},
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                     "torch_cuda": torch.version.cuda, "lightsmith": getattr(ls, "__version__", None),
                     "cpu_backend": "numpy (OpenBLAS)", "gpu_backend": "torch cuda:0",
                     "cpu_threads": THREADS or "library default",
                     "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                     "torch_num_threads": torch.get_num_threads()},
        "precision": "double (complex128) on both devices",
        "timing_method": "time.perf_counter around the call, torch.cuda.synchronize before/after; warm-up runs "
                         "discarded; median over repeats",
        "measured": {
            "max_abs_diff_R_tmm": tmm_d, "max_abs_diff_R_rcwa": rcwa_d,
            "speedup_end_to_end_by_case": sp, "speedup_kernel_by_case": spk,
            "speedup_end_to_end_range_large_problems": [min(c["speedup_end_to_end"] for c in big),
                                                        max(c["speedup_end_to_end"] for c in big)],
            "speedup_kernel_range_large_problems": [min(c["speedup_kernel"] for c in big if c["speedup_kernel"]),
                                                    max(c["speedup_kernel"] for c in big if c["speedup_kernel"])],
        },
        "agreement_within_bounds": tmm_d < 1e-12 and rcwa_d < 1e-9,
        "cases": cases,
    }
    lo, hi = result["measured"]["speedup_end_to_end_range_large_problems"]
    result["reproduced"] = bool(result["agreement_within_bounds"] and lo >= 2 * 0.8 and hi <= 8 * 1.25)
    result["reproduced_note"] = ("agreement bounds are checked strictly; the 2-8x speedup claim is judged against "
                                 "the end-to-end range on the large (non-test) problems with 20% slack")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    OUT.with_suffix(".partial.json").unlink(missing_ok=True)
    print(json.dumps(result["measured"], indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
