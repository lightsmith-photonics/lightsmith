"""Re-measure the FDE/EME validation rows whose values were "reported" but not asserted.

Each configuration is copied verbatim from tests/test_fde.py and tests/test_eme.py.

Run from the worktree root: uv run --no-sync python benchmarks/remeasure/fde_eme.py [--skip-fine-mmi]
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, jvp, kv, kvp

import lightsmith as ls
from lightsmith.problems import Box, Circle, DeviceProblem, ModeProblem, Polygon, Port, Prism, Rectangle, Structure, \
    Structure2D

OUT = Path(__file__).resolve().parent / "results" / "fde_eme.json"
SI, SIO2 = ls.ConstantIndex(n_real=3.476), ls.ConstantIndex(n_real=1.444)
CORE, CLAD = ls.ConstantIndex(n_real=2.85), ls.ConstantIndex(n_real=1.44)


def tic():
    return time.perf_counter()


# --- FDE: symmetric slab (tests/test_fde.py::test_slab_matches_dispersion_relation) -----------------

N1, N2, D, LAM = 2.0, 1.45, 600.0, 1550.0


def slab_neff(tm: bool, lam=LAM) -> float:
    k0 = 2 * np.pi / lam
    r = (N1 / N2) ** 2 if tm else 1.0

    def f(n):
        kap, gam = k0 * np.sqrt(N1**2 - n**2), k0 * np.sqrt(n**2 - N2**2)
        return kap * np.sin(kap * D / 2) - r * gam * np.cos(kap * D / 2)

    hi = N1 - 1e-12
    lo = max(N2 + 1e-12, np.sqrt(N1**2 - (np.pi / (k0 * D)) ** 2) + 1e-12)
    return brentq(f, lo, hi, xtol=1e-14)


def slab(dx, pol):
    core = Structure2D(profile=Rectangle(size_nm=(D, 1e9)), material=ls.ConstantIndex(n_real=N1))
    return ModeProblem(structures=[core],
                       background=ls.ConstantIndex(n_real=N2), window_size_nm=(4000, 0), wavelengths=[LAM],
                       num_modes=1, options={"fde": {"grid_spacing_nm": dx, "polarization": pol}})


def row_slab():
    t0 = tic()
    per = {}
    for tm in (False, True):
        pol = "te" if tm else "tm"  # test's mapping: physical TE = 'tm' in the te_fraction convention
        exact = slab_neff(tm)
        neff = {dx: ls.solve(slab(dx, pol), "fde").modes[0][0].neff_real for dx in (10.0, 2.5)}
        errs = {dx: neff[dx] - exact for dx in neff}
        order = float(np.log(abs(errs[10.0]) / abs(errs[2.5])) / np.log(4.0))
        per["TM" if tm else "TE"] = {
            "fde_polarization_option": pol, "exact_neff": exact,
            "neff": {str(k): v for k, v in neff.items()}, "error": {str(k): v for k, v in errs.items()},
            "abs_error_dx2.5": abs(errs[2.5]), "observed_order": order,
            "bound_pass": abs(errs[2.5]) < 1e-4 and abs(errs[2.5]) < abs(errs[10.0]) / 10}
    worst = max(v["abs_error_dx2.5"] for v in per.values())
    return {
        "row": "FDE: symmetric slab, TE and TM | dispersion relation | <1e-4 bound, second order; ~2e-6 reported",
        "test": "tests/test_fde.py::test_slab_matches_dispersion_relation",
        "configuration": {"n_core": N1, "n_clad": N2, "thickness_nm": D, "wavelength_nm": LAM,
                          "window_nm": 4000, "grid_spacing_nm": [10.0, 2.5], "mode": "fundamental even",
                          "solver": "ls.solve(..., 'fde') default options"},
        "reference": "exact symmetric-slab dispersion relation, brentq xtol 1e-14",
        "measured": per, "measured_value": worst, "measured_value_label": "max |dn_eff| over TE/TM at dx=2.5 nm",
        "previously_reported": 2e-6,
        # "~2e-6": reproduced only if every polarisation's error rounds to 2e-6 at one significant figure.
        "reproduced": bool(all(float(f"{v['abs_error_dx2.5']:.0e}") == 2e-6 for v in per.values())),
        "suggested_error_column": "TE {:.1e}, TM {:.1e} (dx = 2.5 nm); order {:.2f}/{:.2f}".format(
            per["TE"]["abs_error_dx2.5"], per["TM"]["abs_error_dx2.5"], per["TE"]["observed_order"],
            per["TM"]["observed_order"]),
        "runtime_s": tic() - t0}


# --- FDE: step-index fibre HE11 (tests/test_fde.py::test_step_index_fibre_he11) ----------------------

def row_fibre():
    t0 = tic()
    n1, n2, a, lam = 1.50, 1.45, 1500.0, 1550.0
    k0 = 2 * np.pi / lam

    def eq(n):
        u, w = a * k0 * np.sqrt(n1**2 - n**2), a * k0 * np.sqrt(n**2 - n2**2)
        J, K = jvp(1, u) / (u * jv(1, u)), kvp(1, w) / (w * kv(1, w))
        return (J + K) * (n1**2 * J + n2**2 * K) - n**2 * (1 / u**2 + 1 / w**2) ** 2

    ns = np.linspace(n2 + 1e-6, n1 - 1e-6, 2000)
    v = np.array([eq(x) for x in ns])
    i = max(j for j in range(len(ns) - 1) if np.sign(v[j]) != np.sign(v[j + 1]) and abs(v[j]) < 1e3)
    exact = brentq(eq, ns[i], ns[i + 1], xtol=1e-14)
    p = ModeProblem(structures=[Structure2D(profile=Circle(radius_nm=a), material=ls.ConstantIndex(n_real=n1))],
                    background=ls.ConstantIndex(n_real=n2), window_size_nm=(9000, 9000), wavelengths=[lam],
                    num_modes=1, options={"fde": {"grid_spacing_nm": 50, "symmetry": ["pec", "pmc"]}})
    m = ls.solve(p, "fde").modes[0][0]
    err = m.neff_real - exact
    return {
        "row": "FDE: step-index fibre HE11 | exact vector eigenvalue equation | 2e-6 bound; 5e-7 reported",
        "test": "tests/test_fde.py::test_step_index_fibre_he11",
        "configuration": {"n_core": n1, "n_clad": n2, "radius_nm": a, "wavelength_nm": lam,
                          "window_nm": [9000, 9000], "grid_spacing_nm": 50, "symmetry": ["pec", "pmc"]},
        "reference": "exact HE11 vector eigenvalue equation (Snyder & Love Table 12-3)",
        "measured": {"neff": m.neff_real, "exact_neff": exact, "error": err, "te_fraction": m.te_fraction},
        "measured_value": abs(err), "bound": 2e-6, "bound_pass": abs(err) < 2e-6 and m.te_fraction > 0.999,
        "previously_reported": 5e-7,
        "reproduced": bool(float(f"{abs(err):.0e}") == 5e-7),
        "runtime_s": tic() - t0}


# --- FDE: Si strip (tests/test_fde.py::test_si_strip_te0_and_grid_convergence / test_si_strip_fine_grid) ---

SI_STRIP_TE0 = 2.44523


def strip(dx, wl=(1550.0,), num_modes=1, symmetry=("pec", "pmc"), window=(3000, 2000)):
    opts = {"grid_spacing_nm": dx, "symmetry": list(symmetry)}
    return ModeProblem(structures=[Structure2D(profile=Rectangle(size_nm=(500, 220)), material=SI)], background=SIO2,
                       window_size_nm=window, wavelengths=list(wl), num_modes=num_modes, options={"fde": opts})


def row_strip():
    t0 = tic()
    per = {}
    for dx in (20, 10, 5):
        t1 = tic()
        m = ls.solve(strip(dx), "fde").modes[0][0]
        per[str(dx)] = {"neff": m.neff_real, "neff_5dp": round(m.neff_real, 5), "error": m.neff_real - SI_STRIP_TE0,
                        "te_fraction": m.te_fraction, "ng": m.ng, "runtime_s": tic() - t1}
    fine = per["5"]
    return {
        "row": "FDE: Si strip 500x220 nm, TE0, 1550 nm | SiPkit model, 2.44523 | 3e-4 bound; 2.4453 reported",
        "test": "tests/test_fde.py::test_si_strip_fine_grid (dx=5, slow, 3e-4 bound); dx=20/10 from "
                "test_si_strip_te0_and_grid_convergence (1e-3 bound)",
        "configuration": {"core": "Si n=3.476, 500x220 nm", "clad": "SiO2 n=1.444", "wavelength_nm": 1550,
                          "window_nm": [3000, 2000], "symmetry": ["pec", "pmc"], "grid_spacing_nm": 5,
                          "also_measured_grid_spacing_nm": [20, 10]},
        "reference": SI_STRIP_TE0,
        "measured": per, "measured_value": round(fine["neff"], 5), "error_vs_reference": fine["error"],
        "bound": 3e-4, "bound_pass": abs(fine["error"]) < 3e-4,
        "previously_reported": 2.4453,
        "reproduced": bool(round(fine["neff"], 4) == 2.4453),
        "runtime_s": tic() - t0}


# --- EME: linear taper (tests/test_eme.py::test_linear_taper_approaches_adiabatic_limit) --------------

def port2d(name, x, y=0.0, width=2000.0, direction="+"):
    return Port(name=name, center_nm=(x, y, 0), size_nm=(0, width, 1), direction=direction)


def taper(length, w1=500.0, w2=3000.0):
    x = length / 2 + 500
    st = [Structure(geometry=Box(center_nm=(-length / 2 - 1000, 0, 0), size_nm=(2000, w1, 1)), material=CORE),
          Structure(geometry=Box(center_nm=(length / 2 + 1000, 0, 0), size_nm=(2000, w2, 1)), material=CORE),
          Structure(geometry=Prism(profile=Polygon(vertices_nm=[(-length / 2, -w1 / 2), (length / 2, -w2 / 2),
                                                                 (length / 2, w2 / 2), (-length / 2, w1 / 2)]),
                                   z_min_nm=-1, z_max_nm=1), material=CORE)]
    return DeviceProblem(dimension=2, structures=st, background=CLAD, domain_size_nm=(2 * x, 6000, 0),
                         wavelengths=[1550],
                         ports=[port2d("o1", -x, width=6000), port2d("o2", x, width=6000, direction="-")],
                         options={"eme": {"num_modes": 30, "grid_spacing_nm": 10}})


def row_taper():
    t0 = tic()
    lengths = (1000, 3000, 10000, 30000)
    t, recip = [], []
    for L in lengths:
        s = ls.solve(taper(L), "eme").s()[0]
        t.append(float(abs(s[1, 0]) ** 2))
        recip.append(float(abs(s[1, 0] - s[0, 1])))
    t_arr = np.array(t)
    monotone = bool(np.all(np.diff(t_arr) > 0))
    reported = [0.39, 0.77, 0.97, 0.99]
    return {
        "row": "EME: linear taper, 1/3/10/30 um | adiabatic limit | monotone, <0.6 and >0.98 at the ends (bound); "
               "0.39/0.77/0.97/0.99 reported",
        "test": "tests/test_eme.py::test_linear_taper_approaches_adiabatic_limit",
        "configuration": {"dimension": 2, "core_n": 2.85, "clad_n": 1.44, "w_in_nm": 500, "w_out_nm": 3000,
                          "lengths_nm": list(lengths), "wavelength_nm": 1550, "num_modes": 30,
                          "grid_spacing_nm": 10, "domain_width_nm": 6000, "port_width_nm": 6000},
        "reference": "adiabatic limit (T -> 1, monotone in length)",
        "measured": {"transmission_fundamental": dict(zip([f"{L / 1000:g}um" for L in lengths], t)),
                     "transmission_2dp": [round(x, 2) for x in t], "reciprocity_max_abs": max(recip),
                     "monotone": monotone},
        "measured_value": [round(x, 2) for x in t],
        "bound_pass": monotone and t[0] < 0.6 and t[-1] > 0.98 and max(recip) < 1e-3,
        "previously_reported": reported,
        "reproduced": bool(all(round(x, 2) == r for x, r in zip(t, reported))),
        "runtime_s": tic() - t0}


# --- EME: 1x2 MMI (tests/test_eme.py::test_mmi_1x2_self_imaging_length) --------------------------------

NR, NC, LAMM, W, WA = 2.85, 1.44, 1550.0, 3000.0, 800.0
WE = W + LAMM / np.pi * (NC / NR) ** 2 / np.sqrt(NR**2 - NC**2)
L_TH = NR * WE**2 / (2 * LAMM)


def mmi(length):
    x = length / 2 + 1000
    st = [Structure(geometry=Box(center_nm=(-x, 0, 0), size_nm=(2 * x - length, WA, 1)), material=CORE),
          Structure(geometry=Box(size_nm=(length, W, 1)), material=CORE)]
    st += [Structure(geometry=Box(center_nm=(x, y, 0), size_nm=(2 * x - length, WA, 1)), material=CORE)
           for y in (-W / 4, W / 4)]
    return DeviceProblem(dimension=2, structures=st, background=CLAD, domain_size_nm=(2 * x, W + 3000, 0),
                         wavelengths=[LAMM], ports=[port2d("o1", -x, width=W / 2),
                                                    port2d("o2", x, -W / 4, W / 2, "-"),
                                                    port2d("o3", x, W / 4, W / 2, "-")],
                         options={"eme": {"num_modes": 40, "grid_spacing_nm": 10}})


def mmi_scan(fs):
    out, imb = [], []
    for f in fs:
        s = np.abs(ls.solve(mmi(f * L_TH), "eme").s()[0]) ** 2
        out.append(float(s[1, 0] + s[2, 0]))
        imb.append(float(abs(s[1, 0] - s[2, 0]) / s[2, 0]))
    return out, imb


def row_mmi(fine: bool):
    t0 = tic()
    fs = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
    out, imb = mmi_scan(fs)
    o = np.array(out)
    best = fs[int(np.argmax(o))]
    res = {
        "row": "EME: 1x2 MMI | symmetry; self-imaging length | 50/50 to 1e-6 bound; best length within 10% and "
               "transmission >0.9 (bound); 0.92 at 0.93x reported",
        "test": "tests/test_eme.py::test_mmi_1x2_self_imaging_length",
        "configuration": {"dimension": 2, "core_n": NR, "clad_n": NC, "wavelength_nm": LAMM, "mmi_width_nm": W,
                          "access_width_nm": WA, "outputs_y_nm": [-W / 4, W / 4], "num_modes": 40,
                          "grid_spacing_nm": 10, "L_theory_nm": L_TH,
                          "length_factors_test_grid": fs},
        "reference": "Soldano & Pennings self-imaging length L = n_r W_e^2 / (2 lambda) (1x2, centre-fed); "
                     "mirror symmetry for the split",
        "measured": {"total_transmission_by_factor": dict(zip([str(f) for f in fs], out)),
                     "split_rel_imbalance_by_factor": dict(zip([str(f) for f in fs], imb)),
                     "max_rel_imbalance": max(imb),
                     "best_factor_on_test_grid": best, "max_transmission_on_test_grid": float(o.max()),
                     "transmission_at_1.0x": out[fs.index(1.0)]},
        "bound_pass": bool(max(imb) < 1e-6 and 0.9 <= best <= 1.0 and o.max() > 0.9 and out[fs.index(1.0)] > 0.8
                           and o[0] < o.max() and o[-1] < o.max()),
        "previously_reported": {"max_transmission": 0.92, "best_factor": 0.93},
        "note": "0.93x is not on the test's length grid (0.85..1.10 step 0.05); a supplementary fine scan "
                "around the optimum is included to locate the true optimum.",
    }
    if fine:
        ff = [round(x, 3) for x in np.arange(0.88, 0.981, 0.01)]
        fo, fi = mmi_scan(ff)
        k = int(np.argmax(fo))
        # parabolic refinement of the peak
        if 0 < k < len(ff) - 1:
            y0, y1, y2 = fo[k - 1], fo[k], fo[k + 1]
            h = ff[1] - ff[0]
            dlt = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            f_pk, t_pk = ff[k] + dlt * h, y1 - 0.25 * (y0 - y2) * dlt
        else:
            f_pk, t_pk = ff[k], fo[k]
        res["measured"]["fine_scan"] = {"factors": ff, "total_transmission": fo, "max_rel_imbalance": max(fi),
                                        "best_factor_on_fine_grid": ff[k], "max_transmission_on_fine_grid": fo[k],
                                        "parabolic_peak_factor": float(f_pk), "parabolic_peak_transmission": float(t_pk)}
        res["measured_value"] = {"max_transmission": round(fo[k], 2), "best_factor": round(float(f_pk), 2)}
        res["reproduced"] = bool(round(fo[k], 2) == 0.92 and abs(f_pk - 0.93) <= 0.005)
        res["reproduced_note"] = (
            "Not reproduced as stated. On the test's grid the maximum is {:.3f} at {:g}x; on a 0.01-step fine scan "
            "the maximum is {:.3f} at {:g}x. The reported pair (0.92 at 0.93x) appears to combine the two. The "
            "fine-scan curve is not smooth (10 nm grid, 40 modes), so the fine optimum is itself grid-sensitive."
        ).format(float(o.max()), best, fo[k], ff[k])
        res["suggested_error_column"] = "50/50 to {:.0e}; best {:.2f} at {:g}x on test grid (0.85-1.10, step 0.05)"             .format(max(imb), float(o.max()), best)
    else:
        res["measured_value"] = {"max_transmission": round(float(o.max()), 2), "best_factor": best}
        res["reproduced"] = False
        res["reproduced_note"] = "fine scan skipped; cannot confirm the 0.93x optimum from the test grid alone"
    res["runtime_s"] = tic() - t0
    return res


def main():
    fine = "--skip-fine-mmi" not in sys.argv
    rows = {}
    for key, fn in [("fde_slab", row_slab), ("fde_fibre_he11", row_fibre), ("fde_si_strip", row_strip),
                    ("eme_taper", row_taper), ("eme_mmi_1x2", lambda: row_mmi(fine))]:
        print(f"[{key}] ...", flush=True)
        rows[key] = fn()
        print(f"   measured={rows[key]['measured_value']} reported={rows[key]['previously_reported']} "
              f"reproduced={rows[key]['reproduced']} ({rows[key]['runtime_s']:.1f}s)", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "device": "cpu (FDE/EME native solvers, double precision)",
            "precision": "double (float64/complex128)",
            "repeats": 1, "note": "deterministic eigen/S-matrix solves; one run per configuration. Background "
                                  "benchmark load affects runtimes only, not values.",
            "platform": platform.platform(), "python": sys.version.split()[0], "rows": rows}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
