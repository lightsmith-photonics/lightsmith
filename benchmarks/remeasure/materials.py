"""Re-measure the Drude-Lorentz fit accuracy rows of the materials validation table.

Exactly as tests/test_materials.py::test_fit_reproduces_johnson_christy sets it up: fit the bundled
Johnson-Christy data on BAND = linspace(450, 900, 91) with the automatic pole count, then evaluate
fit_quality on linspace(450, 900, 451) (between the fitted samples too).

Writes benchmarks/remeasure/results/materials.json.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

import lightsmith as ls
from lightsmith.materials_fit import fit_drude_lorentz, fit_quality

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BAND = np.linspace(450.0, 900.0, 91)
EVAL = np.linspace(450.0, 900.0, 451)


def env() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "lightsmith_commit": commit}


def row(name: str) -> dict:
    mat = {"kind": "library", "name": name, "dataset": "Johnson-Christy"}
    model, info = fit_drude_lorentz(mat, BAND, return_info=True)
    q = fit_quality(model, mat, EVAL)
    q_fit = fit_quality(model, mat, info.wavelengths_nm)
    return {
        "row": f"{name} (Johnson-Christy), 450-900 nm",
        "table": "Materials (tests/test_materials.py)",
        "configuration": "fit_drude_lorentz(JC data, linspace(450, 900, 91), automatic pole count); "
                         "fit_quality on linspace(450, 900, 451); error = |eps_fit - eps_JC| / |eps_JC| "
                         "(JC linearly interpolated in n, k)",
        "device": "cpu", "precision": "double",
        "n_poles": info.n_poles, "drude": info.drude,
        "model": model.model_dump(),
        "measured_rms_rel_error": q["rms_rel_error"],
        "measured_max_rel_error": q["max_rel_error"],
        "measured_rms_imag_error": q["rms_imag_error"],
        "measured_max_imag_error": q["max_imag_error"],
        "on_fit_samples": {k: v for k, v in q_fit.items() if k != "band_nm"},
        "reference": "tabulated Johnson-Christy data (bundled refractiveindex.info file)",
        "test_bound": {"rms_rel_error": 0.02, "max_rel_error": 0.05, "rms_imag_error": 0.03, "max_imag_error": 0.08},
        "previously_reported": "1.2% RMS",
        "reproduced": bool(abs(q["rms_rel_error"] - 0.012) < 0.0015),
    }


def main() -> None:
    out = {"script": "benchmarks/remeasure/materials.py", "env": env(), "lightsmith_version": ls.__version__,
           "rows": [row("Au"), row("Ag")]}
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / "materials.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for r in out["rows"]:
        print(r["row"], "poles", r["n_poles"], "rms %.4f max %.4f | imag rms %.4f max %.4f" % (
            r["measured_rms_rel_error"], r["measured_max_rel_error"], r["measured_rms_imag_error"],
            r["measured_max_imag_error"]))
    print("wrote", path)


if __name__ == "__main__":
    main()
