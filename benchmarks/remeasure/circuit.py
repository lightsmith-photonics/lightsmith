"""Re-measure the ring FSR quoted in the circuit validation table caption ("9.080 vs. 9.082 nm").

Configuration replicated from tests/test_circuit.py::test_ring_netlist_and_model_match_allpass_formula:
neff = 2.4, ng = 4.2 at 1550 nm (linear neff dispersion), R = 10 um (L = 2 pi R), coupling 0.1,
3 dB/cm, netlist of a coupler closed by a waveguide loop, 2001 wavelengths over 1540-1560 nm.
The ring_single compact model and examples/circuit_ring.py (4001 points, critical coupling) are
measured too.

FSR values:
- grid: difference of adjacent sampled transmission minima (the test's method, 0.01 nm grid);
- refined: resonances located to ~1e-12 nm as sign changes of Im S21 from the solver (S21 is real on
  resonance for the all-pass response), bracketed by the sampled minima;
- formula: lambda^2 / (ng L) evaluated at the midpoint of the two resonances (the test's formula),
  and the exact lambda_m lambda_{m+1} / (ng L), which holds for the linear neff model.

    uv run --no-sync python benchmarks/remeasure/circuit.py
"""

from __future__ import annotations

import datetime
import json
import math
import os
import platform
import subprocess
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar

import lightsmith as ls
from lightsmith.problems import CircuitProblem

NEFF, NG, WL0 = 2.4, 4.2, 1550.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "circuit.json")


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        return "unknown"


def wg(length, **kw):
    return {"component": "waveguide", "settings": {"length_nm": length, "neff": NEFF, "ng": NG, "wl0_nm": WL0, **kw}}


def ring_netlist(L, coupling, loss_db_per_cm, wl):
    return CircuitProblem(
        instances={"dc": {"component": "coupler", "settings": {"coupling": coupling}},
                   "loop": wg(L, loss_db_per_cm=loss_db_per_cm)},
        connections=[("dc,o3", "loop,o1"), ("loop,o2", "dc,o2")],
        ports={"in": "dc,o1", "out": "dc,o4"}, wavelengths=list(wl))


def ring_model(R, coupling, loss_db_per_cm, wl):
    return CircuitProblem(instances={"r": {"component": "ring_single", "settings": {
        "radius_nm": R, "coupling": coupling, "loss_db_per_cm": loss_db_per_cm, "neff": NEFF, "ng": NG,
        "wl0_nm": WL0}}}, ports={"in": "r,o1", "out": "r,o2"}, wavelengths=list(wl))


def s21(problem, wl):
    p = type(problem).model_validate({**problem.model_dump(), "wavelengths": [float(w) for w in np.atleast_1d(wl)]})
    return ls.solve(p, device="cpu").s()[:, 1, 0]


def transmission(problem, wl):
    return np.abs(s21(problem, wl)) ** 2


def measure(name, problem_fn, L, wl):
    p = problem_fn(wl)
    t0 = time.perf_counter()
    T = np.abs(ls.solve(p, device="cpu").s()[:, 1, 0]) ** 2
    runtime = time.perf_counter() - t0
    mins = wl[1:-1][(T[1:-1] < T[:-2]) & (T[1:-1] < T[2:])]
    dw = wl[1] - wl[0]
    refined = []
    for m in mins:
        # On resonance (phi = 2 pi m) the all-pass S21 = (t - a)/(1 - t a) is real: Im S21 changes sign.
        f = lambda x: float(s21(p, x)[0].imag)  # noqa: E731
        lo, hi = m - dw, m + dw
        if f(lo) * f(hi) < 0:
            refined.append(float(brentq(f, lo, hi, xtol=1e-13, rtol=4 * np.finfo(float).eps)))
        else:
            r = minimize_scalar(lambda x: float(transmission(p, x)[0]), bounds=(lo, hi), method="bounded",
                                options={"xatol": 1e-10})
            refined.append(float(r.x))
    refined = np.asarray(refined)
    # Exact resonances of the linear-neff model: L (ng/lambda + (neff - ng)/wl0) = m.
    c = L * (NEFF - NG) / WL0
    m_exact = np.round(NG * L / refined + c)
    exact = NG * L / (m_exact - c)
    fsr_grid = np.diff(mins)
    fsr_ref = np.diff(refined)
    fsr_mid = ((refined[:-1] + refined[1:]) / 2) ** 2 / (NG * L)
    fsr_mid_grid = ((mins[:-1] + mins[1:]) / 2) ** 2 / (NG * L)
    fsr_prod = refined[:-1] * refined[1:] / (NG * L)
    return {
        "case": name,
        "ring_length_nm": L,
        "wavelength_grid": {"start_nm": float(wl[0]), "stop_nm": float(wl[-1]), "num": int(len(wl)),
                            "step_nm": float(dw)},
        "resonances_grid_nm": mins.tolist(),
        "resonances_refined_nm": refined.tolist(),
        "resonances_exact_formula_nm": exact.tolist(),
        "resonance_order_m": m_exact.astype(int).tolist(),
        "max_refined_vs_exact_resonance_nm": float(np.max(np.abs(refined - exact))),
        "fsr_solver_grid_nm": fsr_grid.tolist(),
        "fsr_solver_refined_nm": fsr_ref.tolist(),
        "fsr_formula_lambda_mid_sq_over_ngL_nm": fsr_mid.tolist(),
        "fsr_formula_lambda_mid_sq_over_ngL_grid_minima_nm": fsr_mid_grid.tolist(),
        "fsr_formula_lambda_m_lambda_m1_over_ngL_nm": fsr_prod.tolist(),
        "rel_diff_solver_refined_vs_formula_mid": (np.abs(fsr_ref - fsr_mid) / fsr_mid).tolist(),
        "rel_diff_solver_grid_vs_formula_mid_grid": (np.abs(fsr_grid - fsr_mid_grid) / fsr_mid_grid).tolist(),
        "rel_diff_solver_refined_vs_exact_product": (np.abs(fsr_ref - fsr_prod) / fsr_prod).tolist(),
        "solve_runtime_s": runtime,
    }


def main():
    R = 10_000.0
    L = 2 * np.pi * R
    wl_test = np.linspace(1540, 1560, 2001)
    loss = 3.0
    cases = [
        measure("test_ring_netlist (coupler + waveguide loop, kappa=0.1, 3 dB/cm)",
                lambda wl: ring_netlist(L, 0.1, loss, wl), L, wl_test),
        measure("test_ring_model (ring_single, kappa=0.1, 3 dB/cm)",
                lambda wl: ring_model(R, 0.1, loss, wl), L, wl_test),
    ]
    a = 10 ** (-loss * L * 1e-7 / 20)
    kc = 1 - a**2
    wl_ex = np.linspace(1540, 1560, 4001)
    cases.append(measure(f"examples/circuit_ring.py (ring_single, critical coupling kappa={kc:.6f}, 3 dB/cm)",
                         lambda wl: ring_model(R, kc, loss, wl), L, wl_ex))

    main_case = cases[0]
    grid_fsr = main_case["fsr_solver_grid_nm"]
    ref_fsr = main_case["fsr_solver_refined_nm"]
    formula = main_case["fsr_formula_lambda_mid_sq_over_ngL_nm"]
    formula_grid = main_case["fsr_formula_lambda_mid_sq_over_ngL_grid_minima_nm"]
    entry = {
        "row": "Circuit table caption: ring FSR 9.080 vs. 9.082 nm (solver vs. lambda^2/(n_g L))",
        "configuration": "tests/test_circuit.py::test_ring_netlist_and_model_match_allpass_formula: neff=2.4, "
                         "ng=4.2 at 1550 nm, R=10 um (L=62.832 um), kappa=0.1, 3 dB/cm, coupler+waveguide netlist, "
                         "1540-1560 nm, 2001 points (0.01 nm step); one FSR (two resonances in the window)",
        "measured": {
            "fsr_solver_grid_minima_nm": grid_fsr,
            "fsr_solver_refined_minima_nm": ref_fsr,
            "fsr_formula_lambda_mid_sq_over_ngL_nm (refined minima)": formula,
            "fsr_formula_lambda_mid_sq_over_ngL_nm (grid minima, as in the test)": formula_grid,
            "rel_diff_refined": main_case["rel_diff_solver_refined_vs_formula_mid"],
            "rel_diff_grid_as_in_test": main_case["rel_diff_solver_grid_vs_formula_mid_grid"],
        },
        "reference": "lambda^2/(n_g L) at the midpoint of the two resonances (test tolerance rtol=0.01)",
        "previously_reported": {"fsr_solver_nm": 9.080, "fsr_formula_nm": 9.082},
        "reproduced": bool(abs(round(grid_fsr[0], 3) - 9.080) < 1e-9 and abs(round(formula_grid[0], 3) - 9.082) < 1e-9),
        "notes": "The grid FSR is quantised to the 0.01 nm wavelength step. With resonances refined by root finding the "
                 "solver FSR equals lambda_m lambda_(m+1)/(n_g L) to machine precision (exact for the linear-neff "
                 "model); lambda_mid^2/(n_g L) differs from it by (FSR/2)^2/(n_g L) ~ 8e-5 nm.",
    }
    import scipy
    import torch

    out = {
        "script": "benchmarks/remeasure/circuit.py",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_rev(),
        "device": "cpu (numpy)",
        "cpu": "AMD Ryzen 5 5600X (" + platform.processor() + ")",
        "precision": "double (complex128)",
        "repeats": 1,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "torch": torch.__version__},
        "entries": [entry],
        "cases": cases,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(entry, indent=1))


if __name__ == "__main__":
    main()
