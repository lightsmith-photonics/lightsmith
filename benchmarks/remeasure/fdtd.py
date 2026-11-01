"""Re-measure the "reported" FDTD validation numbers of the paper (tests/test_fdtd.py).

Each entry replicates the configuration of the corresponding test exactly and records the
measured value, the reference, the previously reported value and whether it was reproduced.

Run from the repository root:  uv run --no-sync python benchmarks/remeasure/fdtd.py
Writes benchmarks/remeasure/results/fdtd.json (rewritten after every row).
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

import lightsmith as ls
from lightsmith.materials import LibraryMaterial
from lightsmith.spec import LayeredSphere, NearFieldRequest, Outputs, Shell

OUT = Path(__file__).resolve().parent / "results" / "fdtd.json"
DRUDE = ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)  # as in tests/test_fdtd.py


def sphere_sim(material, radius, wavelengths, medium=1.0, **fdtd):  # copied from tests/test_fdtd.py
    return ls.Simulation(
        particles=[ls.Particle(shape=ls.Sphere(radius_nm=radius), material=material)],
        medium_index=medium, wavelengths=wavelengths, options={"fdtd": fdtd})


def rel(a, b):
    return np.abs(np.asarray(a) / np.asarray(b) - 1)


def pct(x):
    return round(100 * float(x), 3)


def prov(res):
    keep = ("backend", "framework", "device", "precision", "spacing_nm", "grid", "steps", "dt_fs", "runtime_s",
            "decay_level", "cuda_graphs")
    return {k: res.provenance[k] for k in keep if k in res.provenance}


def timed_solve(sim, method, device=None):
    t0 = time.perf_counter()
    res = ls.solve(sim, method, device) if device else ls.solve(sim, method)
    return res, time.perf_counter() - t0


def env_info():
    info = {"timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "host": platform.node(), "python": platform.python_version(),
            "lightsmith_version": getattr(ls, "__version__", None)}
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                            cwd=Path(__file__).resolve().parent).stdout.strip()
        info["git_branch"] = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True,
                                            cwd=Path(__file__).resolve().parent).stdout.strip()
    except OSError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_runtime"] = torch.version.cuda
    except ImportError:
        info["cuda_available"] = False
    info["precision"] = "GPU runs: single (float32, FDTD precision='auto' on GPU); CPU runs: double (float64)"
    info["wall_time_note"] = ("PROVISIONAL: a long benchmark was running on the same GPU in the background "
                              "during these measurements, so all wall/runtime values are upper bounds and not "
                              "suitable for the paper without a re-run on an idle GPU.")
    return info


def row_dielectric():
    """Row 1: tests/test_fdtd.py::test_fdtd_dielectric_sphere_fine_grid, swept over dx = 6/4/3/2 nm."""
    wl = ls.WavelengthRange(start_nm=400, stop_nm=800, num=9)
    per_dx = []
    for dx in (6.0, 4.0, 3.0, 2.0):
        sim = sphere_sim(ls.ConstantIndex(n_real=1.5), 60, wl, spacing_nm=dx)
        res, wall = timed_solve(sim, "fdtd", "gpu")
        mie = ls.solve(sim, "mie")
        e = rel(res.c_ext_nm2, mie.c_ext_nm2)
        per_dx.append({"spacing_nm": dx, "max_rel_err_c_ext_pct": pct(e.max()),
                       "rel_err_c_ext_pct_per_wavelength": [pct(x) for x in e],
                       "wall_s": round(wall, 2), "provenance": prov(res)})
        print(f"[row1] dx={dx}: max C_ext err {pct(e.max())}%  ({wall:.1f} s)", flush=True)
    reported = {6.0: 1.1, 4.0: 0.8, 3.0: 0.6, 2.0: 0.4}
    # "Reproduced": agrees with the reported value when both are rounded to 0.1 percentage points.
    repro = all(round(d["max_rel_err_c_ext_pct"], 1) == reported[d["spacing_nm"]] for d in per_dx)
    return {
        "row": "Dielectric sphere, 6/4/3/2 nm mesh | Mie | 1.1/0.8/0.6/0.4% reported",
        "test": "tests/test_fdtd.py::test_fdtd_dielectric_sphere_fine_grid (spacing swept)",
        "configuration": {"material": "ConstantIndex n=1.5", "radius_nm": 60, "medium_index": 1.0,
                          "wavelengths_nm": "400-800, 9 points", "spacing_nm": [6, 4, 3, 2],
                          "device": "gpu", "other_fdtd_options": "defaults"},
        "quantity": "max over wavelengths of |C_ext,FDTD / C_ext,Mie - 1|",
        "reference": "Mie (ls.solve(sim, 'mie'))",
        "measured": per_dx,
        "measured_summary": "/".join(f"{d['max_rel_err_c_ext_pct']:.2f}" for d in per_dx) + "%",
        "previously_reported": "1.1/0.8/0.6/0.4% (and 0.40% measured, 0.6% bound at 2 nm)",
        "reproduced": repro,
        "reproduced_criterion": "each value equal to the reported one after rounding to 0.1 percentage points",
    }


def layered_sim(direction=(1, 0, 1), polarization=(1, 0, -1)):
    """Configuration of tests/test_fdtd.py::test_fdtd_layered_sphere_near_field_and_oblique_incidence."""
    return ls.Simulation(
        particles=[ls.Particle(shape=LayeredSphere(core_radius_nm=35, shells=[
            Shell(thickness_nm=20, material=ls.ConstantIndex(n_real=1.4))]), material=ls.ConstantIndex(n_real=2.0))],
        medium_index=1.33, wavelengths=[450.0, 550.0, 700.0], options={"fdtd": {"spacing_nm": 5.0}},
        illumination=ls.PlaneWave(direction=direction, polarization=polarization),
        outputs=Outputs(near_field=NearFieldRequest(plane="xz", half_width_nm=100, points=21, wavelengths_nm=[550.0])))


def measure_layered(sim):
    res, wall = timed_solve(sim, "fdtd", "gpu")
    mie = ls.solve(sim, "lightsmith-mie", "cpu")
    e = rel(res.c_ext_nm2, mie.c_ext_nm2)
    a, b = np.asarray(res.near_field[0].enhancement), np.asarray(mie.near_field[0].enhancement)
    ok = np.isfinite(b)
    nf = float(np.sqrt(np.mean((a[ok] / b[ok] - 1) ** 2)))
    return {"max_rel_err_c_ext_pct": pct(e.max()), "rel_err_c_ext_pct_per_wavelength": [pct(x) for x in e],
            "abs_over_max_ext": float(np.abs(res.c_abs_nm2).max() / max(res.c_ext_nm2)),
            "near_field_rms_rel_err_pct": pct(nf), "wall_s": round(wall, 2), "provenance": prov(res)}


def row_layered():
    """Row 2: the test's configuration (oblique), plus the same sphere at normal incidence for comparison."""
    oblique = measure_layered(layered_sim())
    print(f"[row2] oblique (test config): max C_ext err {oblique['max_rel_err_c_ext_pct']}%", flush=True)
    normal = measure_layered(layered_sim(direction=(0, 0, 1), polarization=(1, 0, 0)))
    print(f"[row2] normal incidence: max C_ext err {normal['max_rel_err_c_ext_pct']}%", flush=True)
    return {
        "row": "Layered sphere, oblique incidence, 5 nm mesh, C_ext | native Mie | <3% bound; "
               "0.6% (layered), ~1% (oblique) reported",
        "test": "tests/test_fdtd.py::test_fdtd_layered_sphere_near_field_and_oblique_incidence",
        "configuration": {"core": "n=2.0, r=35 nm", "shell": "n=1.4, 20 nm (outer r=55 nm)", "medium_index": 1.33,
                          "wavelengths_nm": [450, 550, 700], "spacing_nm": 5.0, "device": "gpu",
                          "illumination": "PlaneWave direction (1,0,1), polarization (1,0,-1)  [test config]",
                          "supplementary": "same sphere at normal incidence, direction (0,0,1), polarization (1,0,0)"},
        "quantity": "max over wavelengths of |C_ext,FDTD / C_ext,Mie - 1|",
        "reference": "native Mie (ls.solve(sim, 'lightsmith-mie', 'cpu'))",
        "measured": {"oblique_test_config": oblique, "normal_incidence_supplementary": normal},
        "measured_summary": f"{oblique['max_rel_err_c_ext_pct']:.2f}% (test config: layered + oblique); "
                            f"{normal['max_rel_err_c_ext_pct']:.2f}% (same sphere, normal incidence)",
        "previously_reported": "0.6% (layered), ~1% (oblique); near field 2.9% RMS",
        "reproduced": bool(round(oblique["max_rel_err_c_ext_pct"]) == 1
                           and round(normal["max_rel_err_c_ext_pct"], 1) == 0.6),
        "reproduced_criterion": "oblique rounds to ~1% and normal-incidence value equals 0.6% at 0.1 pp",
    }


def row_gold():
    """Row 3: tests/test_fdtd.py::test_fdtd_gold_library_material_via_drude_lorentz_fit."""
    wl = ls.WavelengthRange(start_nm=480, stop_nm=620, num=29)
    sim = sphere_sim(LibraryMaterial(name="Au", dataset="Johnson-Christy"), 40, wl, medium=1.33)
    res, wall = timed_solve(sim, "fdtd", "gpu")
    mie = ls.solve(sim, "mie")
    w = np.asarray(res.wavelengths_nm)
    i, j = int(np.argmax(mie.c_ext_nm2)), int(np.argmax(res.c_ext_nm2))
    h = float(res.c_ext_nm2[j] / mie.c_ext_nm2[i] - 1)
    offset = float(w[j] - w[i])
    print(f"[row3] peak FDTD {w[j]} nm vs Mie {w[i]} nm, height {pct(h)}%, wall {wall:.1f} s", flush=True)
    return {
        "row": "Au (Johnson-Christy fit), r=40 nm, 2 nm mesh | Mie | position <=10 nm, height <20% bound; "
               "peak at the Mie position, -3.9%, 16 s on an RTX 5070 reported",
        "test": "tests/test_fdtd.py::test_fdtd_gold_library_material_via_drude_lorentz_fit",
        "configuration": {"material": "LibraryMaterial Au, Johnson-Christy (Drude-Lorentz fit by the backend)",
                          "radius_nm": 40, "medium_index": 1.33, "wavelengths_nm": "480-620, 29 points (5 nm step)",
                          "spacing": "default (20 cells per radius for plasmonic -> 2 nm)", "device": "gpu"},
        "quantity": "peak wavelength offset (FDTD - Mie, on the 5 nm grid), signed peak height error, wall time",
        "reference": "Mie (ls.solve(sim, 'mie')) with the tabulated Johnson-Christy data",
        "measured": {"peak_fdtd_nm": float(w[j]), "peak_mie_nm": float(w[i]), "peak_offset_nm": offset,
                     "peak_height_err_pct": pct(h), "wall_s_including_fit": round(wall, 2),
                     "fdtd_runtime_s_provenance": res.provenance.get("runtime_s"),
                     "spacing_nm": res.provenance.get("spacing_nm"),
                     "fit_warnings": [m for m in res.warnings if "fit" in m],
                     "c_ext_fdtd_nm2": [float(x) for x in res.c_ext_nm2],
                     "c_ext_mie_nm2": [float(x) for x in mie.c_ext_nm2], "provenance": prov(res),
                     "wall_time_status": "provisional (GPU shared with a background benchmark)"},
        "measured_summary": f"peak offset {offset:+.0f} nm (FDTD {w[j]:.0f} vs Mie {w[i]:.0f} nm), "
                            f"height {100 * h:+.1f}%, {wall:.0f} s (provisional)",
        "previously_reported": "peak at the Mie position (550 nm), height -3.9%, 16 s on an RTX 5070",
        "reproduced": bool(offset == 0 and round(100 * h, 1) == -3.9),
        "reproduced_criterion": "peak offset 0 nm and height error -3.9% at 0.1 pp; wall time not judged "
                                "(provisional)",
    }


def row_gpu_cpu():
    """Row 4: tests/test_fdtd.py::test_fdtd_gpu_matches_cpu."""
    sim = sphere_sim(DRUDE, 20, [450.0, 520.0, 600.0], medium=1.33, spacing_nm=4.0, run_time_fs=5.0)
    cpu, wall_cpu = timed_solve(sim, "fdtd", "cpu")
    gpu, wall_gpu = timed_solve(sim, "fdtd", "gpu")
    d = {q: rel(getattr(gpu, q), getattr(cpu, q)) for q in ("c_ext_nm2", "c_abs_nm2", "c_sca_nm2")}
    worst = max(float(d["c_ext_nm2"].max()), float(d["c_abs_nm2"].max()))
    print(f"[row4] max rel diff GPU vs CPU (C_ext, C_abs): {worst:.2e}", flush=True)
    return {
        "row": "GPU (single) vs. CPU (double) | CPU | 1e-4 bound; ~1e-7 reported",
        "test": "tests/test_fdtd.py::test_fdtd_gpu_matches_cpu",
        "configuration": {"material": "Drude(eps_inf=9.5, wp=9.0 eV, gamma=0.07 eV)", "radius_nm": 20,
                          "medium_index": 1.33, "wavelengths_nm": [450, 520, 600], "spacing_nm": 4.0,
                          "run_time_fs": 5.0, "gpu": "torch CUDA, single", "cpu": "numpy, double"},
        "quantity": "max relative difference |GPU/CPU - 1| over wavelengths of C_ext and C_abs (the asserted "
                    "quantities); C_sca reported for information",
        "reference": "CPU double-precision FDTD",
        "measured": {"max_rel_diff_c_ext_and_c_abs": worst,
                     "max_rel_diff": {q: float(v.max()) for q, v in d.items()},
                     "per_wavelength": {q: [float(x) for x in v] for q, v in d.items()},
                     "gpu_provenance": prov(gpu), "cpu_provenance": prov(cpu),
                     "wall_s": {"cpu": round(wall_cpu, 2), "gpu": round(wall_gpu, 2)}},
        "measured_summary": f"{worst:.1e}",
        "previously_reported": "~1e-7",
        "reproduced": bool(worst < 1e-6),
        "reproduced_criterion": "same order of magnitude as ~1e-7 (i.e. < 1e-6)",
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {"script": "benchmarks/remeasure/fdtd.py", "environment": env_info(), "rows": []}

    def dump():
        OUT.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    dump()
    # Row 4 first (small), then 1 and 2; the Au timing (row 3) runs last on an already warmed-up GPU.
    for fn in (row_gpu_cpu, row_dielectric, row_layered, row_gold):
        try:
            data["rows"].append(fn())
        except Exception as exc:  # keep the other rows
            traceback.print_exc()
            data["rows"].append({"row_function": fn.__name__, "error": repr(exc), "reproduced": False})
        dump()
    data["environment"]["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    dump()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
