"""Re-measure the values quoted for lightsmith.optics and for the DDA's weighted and extrapolated modes.

Configurations replicated from tests/test_optics.py and tests/test_dda.py. Writes
results/optics_dda_modes.json with the configuration next to each number.

    uv run --no-sync python benchmarks/remeasure/optics_dda_modes.py
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

import lightsmith as ls
from lightsmith.optics import angular_spectrum, focal_spot

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results" / "optics_dda_modes.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def optics() -> dict:
    t = load(ROOT / "tests" / "test_optics.py", "test_optics")
    n, dx, lam, k = t.N, t.DX, t.LAM, t.K
    x, y = t._grid()
    out: dict = {"config": {"wavelength_nm": lam, "dx_nm": dx, "n": n}}

    # plane-wave components
    span = n * dx
    errs = {}
    for label, m, p, z in (("propagating", 37, -12, 3000.0), ("evanescent", 300, 0, 100.0)):
        kx, ky = 2 * np.pi * m / span, 2 * np.pi * p / span
        kz = np.sqrt(complex(k**2 - kx**2 - ky**2))
        f = np.exp(1j * (kx * (x - x[0, 0]) + ky * (y - y[0, 0])))
        errs[label] = float(np.max(np.abs(angular_spectrum(f, dx, lam, z, band_limit=False) - f * np.exp(1j * kz * z))))
    out["plane_wave_max_error"] = errs

    # disk, on the axis
    a = 10 * lam
    disk = t._disk(a).astype(complex)
    aperture = {}
    for zz in (1.0, 2.0):
        z = zz * a**2 / lam
        r = np.sqrt(z**2 + a**2)
        ref = abs(np.exp(1j * k * z) - z / r * np.exp(1j * k * r)) ** 2
        aperture[f"z={zz:g} a^2/lambda"] = float(abs(angular_spectrum(disk, dx, lam, z)[n // 2, n // 2]) ** 2 / ref - 1)
    out["disk_on_axis_relative_error"] = aperture

    # Gaussian beam
    w0 = 10 * lam
    zr = np.pi * w0**2 / lam
    gauss = np.exp(-(x**2 + y**2) / w0**2).astype(complex)
    beam = {}
    for zz in (0.5, 1.0, 2.0):
        spot = focal_spot(np.abs(angular_spectrum(gauss, dx, lam, zz * zr)) ** 2, dx)
        wz = w0 * np.sqrt(1 + zz**2)
        beam[f"z={zz:g} zR"] = {"peak": spot["peak"] / (w0 / wz) ** 2 - 1,
                                "fwhm": spot["fwhm_x_nm"] / (wz * np.sqrt(2 * np.log(2))) - 1}
    out["gaussian_relative_error"] = beam

    # ideal lens
    f = 40 * lam
    lens = t._disk(a) * np.exp(-1j * k * (np.sqrt(x**2 + y**2 + f**2) - f))
    u = angular_spectrum(lens, dx, lam, f)
    ra = np.sqrt(a**2 + f**2)
    ref = f**2 * ((1 / f - 1 / ra) ** 2 + k**2 * np.log(ra / f) ** 2)
    spot = focal_spot(np.abs(u) ** 2, dx)
    out["ideal_lens"] = {"na": float(a / ra), "focus_intensity_relative_error": float(abs(u[n // 2, n // 2]) ** 2 / ref - 1),
                         "fwhm_over_airy_minus_1": float(spot["fwhm_x_nm"] / (0.5145 * lam * ra / a) - 1)}

    # the metalens example
    lens_mod = load(ROOT / "examples" / "metalens_rcwa.py", "metalens_rcwa")
    out["metalens_example"] = {k_: v for k_, v in lens_mod.main(device="cpu").items() if k_ != "runtime_s"}
    return out


def dda_modes() -> dict:
    dielectric = ls.ConstantIndex(n_real=1.5, n_imag=0.02)

    def sim(**opts):
        return ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=40), material=dielectric)],
                             medium_index=1.33, wavelengths=[500.0], options={"dda": opts})

    mie = ls.solve(sim(), "mie", "cpu")

    def err(res) -> dict:
        return {"c_ext": res.c_ext_nm2[0] / mie.c_ext_nm2[0] - 1, "c_abs": res.c_abs_nm2[0] / mie.c_abs_nm2[0] - 1}

    out: dict = {"config": {"sphere_nm": 40, "n": "1.5+0.02i", "medium": 1.33, "wavelength_nm": 500, "device": "cpu"}}
    out["center"] = {str(n): err(ls.solve(sim(dipoles_per_diameter=n), "dda", "cpu")) for n in (12, 16, 24)}
    out["weighted"] = {str(n): err(ls.solve(sim(dipoles_per_diameter=n, discretization="weighted"), "dda", "cpu"))
                       for n in (12, 16, 24)}
    out["extrapolated_12_16_24"] = err(ls.solve(sim(extrapolate=[12, 16, 24]), "dda", "cpu"))
    return out


def main() -> None:
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    res = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "git": rev,
           "platform": platform.platform(), "lightsmith": ls.__version__, "optics": optics(), "dda_modes": dda_modes()}
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
