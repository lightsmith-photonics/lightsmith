"""Re-measure the "reported" entries of the native-Mie validation table.

Row 1: native Mie (homogeneous + coated spheres, hard cases) vs a 40-digit mpmath reference,
       exactly the cases and formulas of tests/test_mie_native.py (test_hard_cases_against_mpmath,
       test_coated_sphere_against_mpmath). CPU, double precision.
Row 2: scattnlay plugin vs miepython, the configurations of
       lightsmith-scattnlay/tests/test_scattnlay.py. Needs the `scattnlay` package, e.g.

           uv run --no-sync --with scattnlay python benchmarks/remeasure/mie.py

       The plugin's source is imported read-only from ../../lightsmith-scattnlay/src (not installed,
       so its uv source pointing at another lightsmith checkout is not used) and registered by hand.

Writes benchmarks/remeasure/results/mie.json.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

import lightsmith as ls
from lightsmith.backends import mie_native
from lightsmith.backends.mie_native import mie_cross_sections

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLUGIN_SRC = ROOT.parents[1] / "lightsmith-scattnlay" / "src"  # C:/Users/.../ChatGPT/lightsmith-scattnlay/src

DRUDE = ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)


def env() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
            "numpy": np.__version__, "mpmath": mp.__version__, "lightsmith_commit": commit}


# --- row 1: 40-digit mpmath reference (as in tests/test_mie_native.py) ----------------------------


def mp_riccati():
    mp.mp.dps = 40
    psi = lambda n, z: mp.sqrt(mp.pi * z / 2) * mp.besselj(n + mp.mpf(1) / 2, z)  # noqa: E731
    chi = lambda n, z: -mp.sqrt(mp.pi * z / 2) * mp.bessely(n + mp.mpf(1) / 2, z)  # noqa: E731
    dpsi = lambda n, z: psi(n - 1, z) - n * psi(n, z) / z  # noqa: E731
    dchi = lambda n, z: chi(n - 1, z) - n * chi(n, z) / z  # noqa: E731
    return psi, chi, dpsi, dchi


def mp_efficiencies(ab, x, n_max):
    """(Q_ext, Q_sca) kept in 40-digit arithmetic (the test rounds them to float before comparing)."""
    ext = sca = mp.mpf(0)
    for n in range(1, n_max + 1):
        a, b = ab(n)
        ext += (2 * n + 1) * (a + b).real
        sca += (2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2)
    return 2 * ext / x**2, 2 * sca / x**2


def rel_mp(value: float, ref) -> float:
    return float(abs(mp.mpf(value) - ref) / abs(ref))


def homogeneous_case(n_part, radius, lam):
    psi, chi, dpsi, dchi = mp_riccati()
    med = 1.33
    eps = complex(DRUDE.eps(np.array([lam]))[0]) if n_part is None else complex(n_part) ** 2
    m, x = mp.mpc(np.sqrt(eps) / med), mp.mpf(2 * np.pi * med * radius / lam)

    def ab(n):
        xi, dxi = psi(n, x) - 1j * chi(n, x), dpsi(n, x) - 1j * dchi(n, x)
        d = dpsi(n, m * x) / psi(n, m * x)
        a = (dpsi(n, x) - d / m * psi(n, x)) / (dxi - d / m * xi)
        b = (dpsi(n, x) - m * d * psi(n, x)) / (dxi - m * d * xi)
        return a, b

    qe, qs = mp_efficiencies(ab, x, mie_native.n_max_wiscombe(float(x)) + 10)
    ce, cs, _ = mie_cross_sections(radius, eps, np.array([lam]), med)
    g = np.pi * radius**2
    return {"case": "homogeneous", "n_particle": "Drude(9.5, 9.0 eV, 0.07 eV)" if n_part is None else n_part,
            "eps": [eps.real, eps.imag], "radius_nm": radius, "wavelength_nm": lam, "medium_index": med,
            "x": float(x), "q_ext_mp": mp.nstr(qe, 25), "q_sca_mp": mp.nstr(qs, 25),
            "q_ext_native": float(ce[0] / g), "q_sca_native": float(cs[0] / g),
            "rel_err_ext": rel_mp(float(ce[0] / g), qe), "rel_err_sca": rel_mp(float(cs[0] / g), qs)}


def coated_case(core_eps, shell_eps, lam):
    psi, chi, dpsi, dchi = mp_riccati()
    med, rc, rs = 1.33, 30.0, 42.0
    metal = complex(DRUDE.eps(np.array([lam]))[0])
    ec, es = (metal if core_eps is None else core_eps), (metal if shell_eps is None else shell_eps)
    m1, m2 = mp.mpc(np.sqrt(complex(ec)) / med), mp.mpc(np.sqrt(complex(es)) / med)
    k = 2 * mp.pi * med / lam
    x, y = k * rc, k * rs

    def ab(n):
        A = ((m2 * psi(n, m2 * x) * dpsi(n, m1 * x) - m1 * dpsi(n, m2 * x) * psi(n, m1 * x))
             / (m2 * chi(n, m2 * x) * dpsi(n, m1 * x) - m1 * dchi(n, m2 * x) * psi(n, m1 * x)))
        B = ((m2 * psi(n, m1 * x) * dpsi(n, m2 * x) - m1 * psi(n, m2 * x) * dpsi(n, m1 * x))
             / (m2 * dchi(n, m2 * x) * psi(n, m1 * x) - m1 * dpsi(n, m1 * x) * chi(n, m2 * x)))
        xi, dxi = psi(n, y) - 1j * chi(n, y), dpsi(n, y) - 1j * dchi(n, y)
        fa, ga = dpsi(n, m2 * y) - A * dchi(n, m2 * y), psi(n, m2 * y) - A * chi(n, m2 * y)
        fb, gb = dpsi(n, m2 * y) - B * dchi(n, m2 * y), psi(n, m2 * y) - B * chi(n, m2 * y)
        a = (psi(n, y) * fa - m2 * dpsi(n, y) * ga) / (xi * fa - m2 * dxi * ga)
        b = (m2 * psi(n, y) * fb - dpsi(n, y) * gb) / (m2 * xi * fb - dxi * gb)
        return a, b

    qe, qs = mp_efficiencies(ab, y, mie_native.n_max_wiscombe(float(y)) + 10)
    ce, cs, _ = mie_cross_sections([rc, rs], [ec, es], np.array([lam]), med)
    g = np.pi * rs**2
    label = lambda e: "Drude" if e is None else e  # noqa: E731
    return {"case": "coated", "core_eps": str(label(core_eps)), "shell_eps": str(label(shell_eps)),
            "core_radius_nm": rc, "outer_radius_nm": rs, "wavelength_nm": lam, "medium_index": med,
            "q_ext_mp": mp.nstr(qe, 25), "q_sca_mp": mp.nstr(qs, 25),
            "q_ext_native": float(ce[0] / g), "q_sca_native": float(cs[0] / g),
            "rel_err_ext": rel_mp(float(ce[0] / g), qe), "rel_err_sca": rel_mp(float(cs[0] / g), qs)}


def row_native() -> dict:
    cases = [homogeneous_case(*c) for c in [(4.0, 1.0, 300.0), (4.0, 1500.0, 300.0), (None, 10.0, 422.5)]]
    cases += [coated_case(*c) for c in [(2.25, None, 600.0), (None, 2.1, 520.0), (12 + 0.5j, None, 500.0)]]
    errs = [max(c["rel_err_ext"], c["rel_err_sca"]) for c in cases]
    worst = int(np.argmax(errs))
    return {
        "row": "Homogeneous and coated spheres, hard cases",
        "table": "Native Mie (tests/test_mie_native.py)",
        "configuration": "the 3 homogeneous hard cases of test_hard_cases_against_mpmath and the 3 coated cases of "
                         "test_coated_sphere_against_mpmath; medium 1.33; mie_cross_sections on NumPy (CPU, float64); "
                         "reference: 40-digit mpmath Riccati-Bessel formulas copied from the test, n_max = Wiscombe + 10",
        "device": "cpu", "precision": "double",
        "quantity": "max relative error of Q_ext and Q_sca vs the mpmath value (kept at 40 digits)",
        "measured_max_rel_error": max(errs),
        "worst_case": cases[worst],
        "measured_max_rel_error_homogeneous": max(errs[:3]),
        "measured_max_rel_error_coated": max(errs[3:]),
        "cases": cases,
        "reference": "40-digit mpmath",
        "test_bound": 1e-11,
        "previously_reported": "~1e-15",
        "reproduced": bool(max(errs) < 1e-14),
    }


# --- row 2: scattnlay plugin vs miepython -------------------------------------------------------


def rel_max(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def row_scattnlay() -> dict:
    base = {"row": "scattnlay plugin (GPL, separate package)", "table": "Native Mie",
            "reference": "miepython", "test_bound": 1e-9, "previously_reported": "1e-11"}
    try:
        from importlib.metadata import version

        import scattnlay  # noqa: F401
        sys.path.insert(0, str(PLUGIN_SRC))
        from lightsmith.backends.base import register
        from lightsmith_scattnlay import ScattnlayBackend

        register(ScattnlayBackend)
        sn_version = version("scattnlay")
    except Exception as e:  # noqa: BLE001
        return {**base, "status": "not run", "reason": f"{type(e).__name__}: {e}", "reproduced": False,
                "how_to_run": "uv run --no-sync --with scattnlay python benchmarks/remeasure/mie.py"}

    import miepython

    AU = DRUDE
    DIEL = ls.ConstantIndex(n_real=1.5, n_imag=0.02)
    WL = ls.WavelengthRange(start_nm=400, stop_nm=800, num=41)

    def sphere(material, radius=40.0):
        return ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=radius), material=material)],
                              medium_index=1.33, wavelengths=WL)

    cases = []
    for name, mat in (("drude-gold", AU), ("lossy-dielectric", DIEL)):  # test_matches_miepython_for_plain_sphere
        sim = sphere(mat)
        ref = ls.solve(sim, method="miepython")
        res = ls.solve(sim, method="scattnlay")
        nat = ls.solve(sim, "lightsmith-mie", "cpu")
        assert res.backend == "scattnlay"
        cases.append({"case": f"plain sphere r=40 nm, {name}, 400-800 nm (41)",
                      "test": "test_matches_miepython_for_plain_sphere",
                      "precision": res.provenance.get("precision"),
                      **{f"rel_err_{q[2:5]}_vs_miepython": rel_max(getattr(res, q), getattr(ref, q))
                         for q in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2")},
                      **{f"rel_err_{q[2:5]}_vs_native_mie": rel_max(getattr(res, q), getattr(nat, q))
                         for q in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2")}})
    # test_layered_sphere_reduces_to_homogeneous: 30 nm Drude core + 10 nm Drude shell vs 40 nm sphere
    shape = ls.LayeredSphere(core_radius_nm=30.0, shells=[ls.Shell(thickness_nm=10.0, material=AU)])
    same = ls.Simulation(particles=[ls.Particle(shape=shape, material=AU)], medium_index=1.33, wavelengths=WL)
    ref = ls.solve(sphere(AU), method="miepython")
    res = ls.solve(same, method="scattnlay")
    cases.append({"case": "layered 30+10 nm, shell = core (Drude), 400-800 nm (41)",
                  "test": "test_layered_sphere_reduces_to_homogeneous", "precision": res.provenance.get("precision"),
                  "rel_err_ext_vs_miepython": rel_max(res.c_ext_nm2, ref.c_ext_nm2)})
    vs_mp = [v for c in cases for k, v in c.items() if k.endswith("_vs_miepython")]
    plain_ext_sca = [c[k] for c in cases[:2] for k in ("rel_err_ext_vs_miepython", "rel_err_sca_vs_miepython")]
    return {**base,
            "configuration": "the plugin's tests: Drude and n=1.5+0.02i spheres, r=40 nm in n=1.33, 400-800 nm "
                             "(41 points); C_ext, C_sca, C_abs; plus the layered (shell = core) case vs miepython",
            "device": "cpu", "precision": "double (scattnlay default, multiprecision off)",
            "scattnlay_version": sn_version, "miepython_version": miepython.__version__,
            "plugin_source": str(PLUGIN_SRC) + " (imported read-only, registered by hand)",
            "install": "uv run --no-sync --with scattnlay (ephemeral overlay env; project env untouched)",
            "measured_max_rel_error": max(vs_mp),
            "measured_max_rel_error_plain_ext_sca": max(plain_ext_sca),
            "note": "C_abs = C_ext - C_sca suffers cancellation for the weakly absorbing dielectric; "
                    "see per-quantity values",
            "cases": cases,
            "reproduced": bool(max(vs_mp) <= 1e-10)}


def main() -> None:
    out = {"script": "benchmarks/remeasure/mie.py", "env": env(), "rows": [row_native(), row_scattnlay()]}
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / "mie.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for r in out["rows"]:
        print(r["row"], "->", r.get("measured_max_rel_error", r.get("reason")))
    print("wrote", path)


if __name__ == "__main__":
    main()
