"""Re-measure "Speed vs. treams' own path | --- | ~7x reported" (T-matrix table).

Times the wrapper's cluster solve (lightsmith.backends.tmatrix_treams._Solver.tmatrix: z-axis translations
plus rotations, half the blocks by symmetry, sqrt|T|-scaled solve) against treams' own path
(TMatrix.sphere for each particle, TMatrix.cluster(...).interaction.solve()), at the same per-particle
lmax, one wavelength, CPU. Both timings include building the single-sphere T-matrices. The wrapper's
per-wavelength cache is cleared before every repeat. Median of 7 (dimer, trimer) or 3 (tetramer)
repeats after one warm-up. BLAS limited to 1 thread by default (see THREADS below).

Clusters:
  dimer    tests/test_tmatrix_treams.py::test_gold_dimer_redshifts_for_axial_polarization
           (Drude gold, r=20 nm, gap 5 nm, plan lmax from 480 nm), at 520 nm
  trimer   test_cluster_matches_treams_reference_path (2 gold + 1 dielectric, unequal lmax), 520 nm
  tetramer test_tetrahedral_cluster_converges_and_conserves (eps=16+0.5i, r=75-110 nm), 1000 nm

Writes benchmarks/remeasure/results/tmatrix.json.
"""

from __future__ import annotations

import os

# BLAS threads, set before NumPy loads. Default 1: the machine is shared (CPU at 100% load from other jobs
# during the re-measurement), and oversubscribed multithreaded BLAS made the wrapper's dense solve erratic.
# Override with LIGHTSMITH_BENCH_THREADS=N.
THREADS = os.environ.get("LIGHTSMITH_BENCH_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = THREADS

import json
import platform
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import lightsmith as ls
from lightsmith.backends.tmatrix_treams import TreamsOptions, _Plan, _Solver, _treams, _treams_version

warnings.filterwarnings("ignore", category=DeprecationWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GOLD = ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)
DIELECTRIC = ls.ConstantIndex(n_real=1.5, n_imag=0.02)


def env() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    try:
        from threadpoolctl import threadpool_info
        blas = [{k: d.get(k) for k in ("internal_api", "num_threads", "version")} for d in threadpool_info()]
    except Exception:  # noqa: BLE001
        blas = "unknown"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "numpy": np.__version__, "treams": _treams_version(), "blas": blas,
            "lightsmith_commit": commit, "blas_threads_env": THREADS,
            "cpu_load_note": "shared machine: other jobs (incl. the full-paper benchmark) kept the CPU at ~100% load; "
                             "absolute times are inflated, ratios are the quantity of interest"}


def particle(material, radius, pos):
    return ls.Particle(shape=ls.Sphere(radius_nm=radius), material=material, position_nm=pos)


def sim_of(parts, wavelengths, **kw):
    return ls.Simulation(particles=parts, medium_index=1.33, wavelengths=list(wavelengths), **kw)


def clusters():
    r, gap = 20.0, 5.0
    dimer_wl = np.arange(480.0, 641.0, 10.0).tolist()
    dimer = sim_of([particle(GOLD, r, (-(r + gap / 2), 0.0, 0.0)), particle(GOLD, r, (r + gap / 2, 0.0, 0.0))],
                   dimer_wl, illumination=ls.PlaneWave(polarization=(1.0, 0.0, 0.0)))
    trimer = sim_of([particle(GOLD, 20.0, (-22.0, 0.0, 0.0)), particle(DIELECTRIC, 30.0, (10.0, 55.0, 10.0)),
                     particle(GOLD, 15.0, (0.0, -10.0, 45.0))], [520.0],
                    illumination=ls.PlaneWave(direction=(0.0, 1.0, 1.0)))
    mat = ls.ConstantPermittivity(eps_real=16.0, eps_imag=0.5)
    pos = (220 / np.sqrt(24)) * np.array([[np.sqrt(8), 0, -1], [-np.sqrt(2), np.sqrt(6), -1],
                                          [-np.sqrt(2), -np.sqrt(6), -1], [0, 0, 3]])
    tetra = sim_of([particle(mat, rr, tuple(p)) for rr, p in zip([110.0, 90.0, 80.0, 75.0], pos)], [1000.0])
    return [("dimer (test_gold_dimer_redshifts_for_axial_polarization)", dimer, 520.0, 7),
            ("trimer (test_cluster_matches_treams_reference_path)", trimer, 520.0, 7),
            ("tetramer (test_tetrahedral_cluster_converges_and_conserves)", tetra, 1000.0, 3)]


def treams_path(tr, sim, plan, lam):
    k0, emb = 2 * np.pi / lam, tr.Material(sim.medium_index**2)
    singles = [tr.TMatrix.sphere(lm, k0, p.shape.radius_nm, [tr.Material(p.material.eps(np.array([lam]))[0]), emb],
                                 poltype="parity") for p, lm in zip(sim.particles, plan.lmax)]
    return tr.TMatrix.cluster(singles, plan.positions).interaction.solve()


def wrapper_path(solver, lam):
    solver._cache = None
    return solver.tmatrix(lam)[0]


def timed(fn, n):
    fn()  # warm-up
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return ts


def main() -> None:
    tr = _treams()  # installs lightsmith's Windows/NumPy-2 integer shim, which treams' own path also needs here
    runs = []
    for label, sim, lam, n in clusters():
        plan = _Plan(sim, TreamsOptions())
        solver = _Solver(sim, plan)
        ours, ref = np.asarray(wrapper_path(solver, lam)), np.asarray(treams_path(tr, sim, plan, lam))
        diff = float(np.max(np.abs(ours - ref)) / np.max(np.abs(ref)))
        t_ours = timed(lambda: wrapper_path(solver, lam), n)
        t_ref = timed(lambda: treams_path(tr, sim, plan, lam), n)
        m_ours, m_ref = statistics.median(t_ours), statistics.median(t_ref)
        runs.append({"cluster": label, "particles": len(sim.particles), "wavelength_nm": lam, "lmax": plan.lmax,
                     "modes": plan.modes, "repeats": n, "tmatrix_rel_diff": diff,
                     "wrapper_s": t_ours, "treams_s": t_ref,
                     "wrapper_median_s": m_ours, "treams_median_s": m_ref, "speedup_median": m_ref / m_ours,
                     "speedup_min_time": min(t_ref) / min(t_ours)})
        print(f"{label:62s} lmax={plan.lmax} wrapper {m_ours*1e3:8.1f} ms  treams {m_ref*1e3:8.1f} ms  "
              f"x{m_ref/m_ours:5.2f}  (T diff {diff:.1e})", flush=True)
    dimer = runs[0]
    row = {
        "row": "Speed vs. treams' own path",
        "table": "T-matrix via treams (tests/test_tmatrix_treams.py)",
        "configuration": "cluster T-matrix at one wavelength incl. single-sphere T-matrices; wrapper _Solver.tmatrix "
                         "vs treams TMatrix.cluster(...).interaction.solve() at the same lmax; median of repeats "
                         "after a warm-up; CPU (NumPy/SciPy, double), BLAS threads = " + THREADS,
        "device": "cpu", "precision": "double",
        "measured_value": {r["cluster"].split()[0]: round(r["speedup_median"], 2) for r in runs},
        "measured_value_min_time_ratio": {r["cluster"].split()[0]: round(r["speedup_min_time"], 2) for r in runs},
        "headline_dimer_speedup": dimer["speedup_median"],
        "note": "repeat-to-repeat spread was up to ~7x under the shared CPU load, so the ratio is only "
                "known to within a factor ~2; re-run on an idle machine before quoting a single number",
        "reference": "treams' own cluster path (same lmax, same positions)",
        "previously_reported": "~7x",
        "reproduced": bool(5.0 <= dimer["speedup_median"] <= 9.0),
        "runs": runs,
    }
    out = {"script": "benchmarks/remeasure/tmatrix.py", "env": env(), "rows": [row]}
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / "tmatrix.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
