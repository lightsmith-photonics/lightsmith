"""Drude-sphere DDA absorption plateau: is it staircase noise or a systematic error?

The smoke suite found DDA (center, LDR, single precision on the GPU) giving Q_abs ~ 0.275 for
npd = 24-32 against Mie 0.107. This script separates the candidate causes:

* precision / iterative tolerance: single vs double, tol 1e-5 vs 1e-10;
* polarizability prescription: LDR vs Clausius-Mossotti + radiative reaction;
* the absorption formula: C_abs from the dipole sum vs C_ext - C_sca with C_sca integrated from
  the DDA far field (32-point Gauss-Legendre in theta, 24-point trapezoid in phi);
* resolution: npd up to 64.

    uv run --no-sync python benchmarks/remeasure/drude_plateau.py

Writes benchmarks/remeasure/results/drude_plateau.json.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import time
from pathlib import Path

import numpy as np

import lightsmith as ls

OUT = Path(__file__).resolve().parent / "results" / "drude_plateau.json"
R, LAM, N_MED = 40.0, 700.0, 1.33
DRUDE = {"kind": "drude", "eps_inf": 9.5, "wp_ev": 9.0, "gamma_ev": 0.07}


def sim(opts, far=False):
    s = {"particles": [{"shape": {"kind": "sphere", "radius_nm": R}, "material": DRUDE}],
         "medium_index": N_MED, "wavelengths": [LAM], "options": {"dda": opts}}
    if far:
        x, _ = np.polynomial.legendre.leggauss(32)
        s["outputs"] = {"far_field": {"theta_deg": list(np.degrees(np.arccos(-x))),
                                      "phi_deg": list(np.arange(0.0, 360.0, 15.0)), "wavelengths_nm": [LAM]}}
    return ls.Simulation.model_validate(s)


def c_sca_far(res):
    x, w = np.polynomial.legendre.leggauss(32)  # nodes in cos(theta), theta = arccos(-x) ascending
    d = np.asarray(res.far_field[0].dcsca_domega_nm2)  # [theta][phi]
    return float(np.sum(w[:, None] * d) * (2 * math.pi / d.shape[1]))


def q(c):
    return c / (math.pi * R**2)


def run(label, opts, device, far=False):
    t0 = time.perf_counter()
    res = ls.solve(sim(opts, far), "lightsmith-dda", device)
    row = {"label": label, "device": device, "options": opts, "wall_s": round(time.perf_counter() - t0, 2),
           "dipoles": res.provenance.get("dipoles"), "iterations": res.provenance.get("iterations"),
           "precision": res.provenance.get("precision"),
           "q_ext": q(res.c_ext_nm2[0]), "q_abs": q(res.c_abs_nm2[0])}
    if far:
        csca = c_sca_far(res)
        row["q_sca_far"] = q(csca)
        row["q_abs_ext_minus_far"] = q(res.c_ext_nm2[0] - csca)
    print(json.dumps(row), flush=True)
    return row


def main():
    mie = ls.solve(sim({}).model_copy(update={"options": {}}), "miepython", "cpu")
    ref = {"q_ext": q(mie.c_ext_nm2[0]), "q_abs": q(mie.c_abs_nm2[0]), "q_sca": q(mie.c_sca_nm2[0])}
    print("Mie", ref, flush=True)
    rows = []
    # 1. precision and tolerance at the plateau
    for npd in (24, 32):
        rows.append(run(f"npd{npd} single tol1e-5", {"dipoles_per_diameter": npd, "precision": "single"}, "gpu"))
        rows.append(run(f"npd{npd} double tol1e-10", {"dipoles_per_diameter": npd, "precision": "double",
                                                       "tol": 1e-10}, "gpu"))
    # 2. polarizability prescription
    for npd in (16, 24, 32, 48):
        rows.append(run(f"npd{npd} cm_rr", {"dipoles_per_diameter": npd, "polarizability": "cm_rr",
                                            "max_iter": 20000}, "gpu"))
    # 3. absorption formula: dipole sum vs C_ext - far-field C_sca (double precision)
    for npd in (16, 24, 32):
        rows.append(run(f"npd{npd} double far-field", {"dipoles_per_diameter": npd, "precision": "double",
                                                        "tol": 1e-10}, "gpu", far=True))
    # 4. resolution beyond the plateau (LDR, center, single precision, as the benchmark)
    for npd in (36, 40, 44, 48, 56, 64):
        rows.append(run(f"npd{npd} single", {"dipoles_per_diameter": npd, "precision": "single",
                                             "max_iter": 20000}, "gpu"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"problem": {"radius_nm": R, "wavelength_nm": LAM, "medium_index": N_MED,
                                           "material": DRUDE}, "mie": ref, "runs": rows,
                               "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                               "doctor": ls.doctor()}, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
