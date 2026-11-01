"""Re-measure the DDA row "Dielectric sphere, n_pd=20 | Mie | 0.07% reported".

No test in tests/test_core.py or tests/test_dda.py uses n_pd = 20, so this measures the likely
configurations, all on CPU in double precision (default LDR polarizability, "center" discretisation,
tol 1e-8), against Mie (native, CPU):

  A  tests/test_core.py::sphere_sim(DIELECTRIC): n = 1.5+0.02i, r = 40 nm, water (1.33), 450/550/650 nm,
     at n_pd = 20 (and the test's own n_pd = 12). Same configuration as the old ADDA plugin test
     test_real_adda_matches_mie_for_dielectric_sphere (sphere_sim(npd=20)).
  B  tests/test_dda.py::sim(): same sphere at 500 nm, n_pd = 20 (and 12).
  C  benchmark problem sphere-diel-x1.5-m1.5-k0.01-water (paper Section "Dielectric sphere"):
     n = 1.995+0.01i, r = 107.7 nm, water, 450/550/700 nm, n_pd = 20, center and weighted.

Writes benchmarks/remeasure/results/dda.json.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import lightsmith as ls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIELECTRIC = ls.ConstantIndex(n_real=1.5, n_imag=0.02)
QS = ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2")


def env() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
            "numpy": np.__version__, "lightsmith_commit": commit}


def sphere_sim(material, radius, wavelengths, npd, medium=1.33, **dda):
    return ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=radius), material=material)],
                         medium_index=medium, wavelengths=list(wavelengths),
                         options={"dda": {"dipoles_per_diameter": npd, **dda}})


def measure(label, material, radius, wavelengths, npd, **dda) -> dict:
    sim = sphere_sim(material, radius, wavelengths, npd, **dda)
    ref = ls.solve(sim, "lightsmith-mie", "cpu")
    t = time.perf_counter()
    res = ls.solve(sim, "dda", "cpu")
    wall = time.perf_counter() - t
    rel = {q[2:5]: (np.abs(np.asarray(getattr(res, q)) - np.asarray(getattr(ref, q))) / np.abs(getattr(ref, q))).tolist()
           for q in QS}
    return {"config": label, "radius_nm": radius, "material": material.model_dump(), "medium_index": 1.33,
            "wavelengths_nm": list(wavelengths), "dipoles_per_diameter": npd, "dda_options": dda,
            "device": res.provenance.get("device"), "precision": res.provenance.get("precision"),
            "polarizability": res.provenance.get("polarizability"), "tol": res.provenance.get("tol"),
            "dipoles": res.provenance.get("dipoles"), "iterations": res.provenance.get("iterations"),
            "wall_s": wall, "warnings": res.warnings,
            "mie": {q: list(map(float, getattr(ref, q))) for q in QS},
            "dda": {q: list(map(float, getattr(res, q))) for q in QS},
            "rel_err_per_wavelength": rel,
            "max_rel_err": {k: max(v) for k, v in rel.items()},
            "rms_rel_err": {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in rel.items()}}


def main() -> None:
    runs = []
    for npd in (20, 12):
        runs.append(measure("A: test_core sphere_sim(DIELECTRIC)", DIELECTRIC, 40.0, (450.0, 550.0, 650.0), npd))
        runs.append(measure("B: test_dda sim()", DIELECTRIC, 40.0, (500.0,), npd))
    r_c = 1.5 * 600.0 / (2 * math.pi * 1.33)
    mat_c = ls.ConstantIndex(n_real=1.5 * 1.33, n_imag=0.01)
    for disc in ("center", "weighted"):
        runs.append(measure(f"C: benchmark sphere-diel-x1.5-m1.5-k0.01-water ({disc})", mat_c, r_c,
                            (450.0, 550.0, 700.0), 20, discretization=disc))
    a20 = runs[0]
    best = min((r for r in runs if r["dipoles_per_diameter"] == 20),
               key=lambda r: min(r["max_rel_err"].values()))
    row = {
        "row": "Dielectric sphere, n_pd=20",
        "table": "DDA (tests/test_core.py, tests/test_dda.py)",
        "configuration": "most likely: config A (test_core sphere_sim, n=1.5+0.02i, r=40 nm in water, 450/550/650 nm) "
                         "at n_pd=20; CPU double, LDR, center discretisation. No test uses n_pd=20; B and C are "
                         "the alternative readings.",
        "device": "cpu", "precision": "double",
        "measured_value": {"c_ext_max_rel_err": a20["max_rel_err"]["ext"], "c_abs_max_rel_err": a20["max_rel_err"]["abs"],
                           "c_sca_max_rel_err": a20["max_rel_err"]["sca"]},
        "smallest_max_error_of_any_npd20_config_and_quantity": {"config": best["config"],
                                                                 "max_rel_err": best["max_rel_err"]},
        "reference": "Mie (lightsmith-mie, CPU double)",
        "previously_reported": "0.07%",
        "reproduced": bool(any(abs(v - 7e-4) < 2e-4 for r in runs if r["dipoles_per_diameter"] == 20
                               for v in r["max_rel_err"].values())),
        "runs": runs,
    }
    out = {"script": "benchmarks/remeasure/dda.py", "env": env(), "rows": [row]}
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / "dda.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for r in runs:
        print(f"{r['config']:60s} npd={r['dipoles_per_diameter']:2d} "
              + " ".join(f"{k} max {v:.3%} rms {r['rms_rel_err'][k]:.3%}" for k, v in r["max_rel_err"].items())
              + f"  ({r['wall_s']:.1f}s)")
    print("wrote", path)


if __name__ == "__main__":
    main()
