"""Measured values behind the BPM validation table: the configurations of tests/test_bpm.py, solved once each.

    uv run --no-sync python benchmarks/remeasure/bpm.py

Writes benchmarks/remeasure/results/bpm.json. Uses the test module's device builders so that the problems are
exactly those the tests assert on.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

import lightsmith as ls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
import test_bpm as T  # noqa: E402
from lightsmith.backends._fde import port_modes, solve_modes  # noqa: E402
from lightsmith.problems import Box, DeviceProblem, Port, Rectangle, Structure, Structure2D  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "bpm.json"


def t2(s, i, j):
    return float(abs(s[i, j]) ** 2)


def main():
    out, t0 = {"date": _dt.date.today().isoformat(), "lightsmith": ls.__version__}, time.perf_counter()

    # straight slab: |S21| and phase against EME
    for pol in ("te", "tm"):
        p = T.straight2d(pol=pol)
        s, e = ls.solve(p, "bpm", device="cpu").s(), ls.solve(p, "eme").s()
        out[f"straight_slab_{pol}"] = {"max_abs_S21_minus_1": float(np.abs(np.abs(s[:, 1, 0]) - 1).max()),
                                       "max_phase_vs_eme_rad": float(np.abs(np.angle(s[:, 1, 0] / e[:, 1, 0])).max())}

    # straight 3D strip, full vector
    L, w, h, d = 2000.0, 2400.0, 2000.0, 40.0
    p = DeviceProblem(structures=[Structure(geometry=Box(size_nm=(L + 1000, 500, 220)), material=T.SI)],
                      background=T.SIO2, domain_size_nm=(L, w, h), wavelengths=[1550],
                      ports=[Port(name="o1", center_nm=(-L / 2, 0, 0), size_nm=(0, w, h), direction="+", num_modes=2),
                             Port(name="o2", center_nm=(L / 2, 0, 0), size_nm=(0, w, h), direction="-", num_modes=2)],
                      options={"bpm": {"grid_spacing_nm": d, "pml_cells": 6, "precision": "double"}})
    s = ls.solve(p, "bpm", device=T.CPU_3D).s()[0]
    neff = port_modes(p, p.ports[0], 1550, grid_spacing_nm=d)[0].neff.real
    out["strip_3d_full_vector"] = {
        "max_deviation_from_identity": float(np.abs(np.abs(s[2:, :2]) - np.eye(2)).max()),
        "te0_phase_error_rad": float(abs(np.angle(s[2, 0] * np.exp(-2j * np.pi / 1550 * neff * L))))}

    # tapers against EME
    out["taper"] = {}
    for length in (3000, 10000):
        b = ls.solve(T.taper(length), "bpm", device="cpu").s()[0]
        e = ls.solve(T.taper(length), "eme").s()[0]
        out["taper"][str(length)] = {"bpm": t2(b, 1, 0), "eme": t2(e, 1, 0),
                                     "phase_diff_rad": float(abs(np.angle(b[1, 0] / e[1, 0])))}
    s = ls.solve(T.taper(30000, dl=10.0, directions="both"), "bpm", device="cpu").s()[0]
    e = ls.solve(T.taper(30000, dl=10.0), "eme").s()[0]
    out["taper"]["30000_10nm"] = {"bpm_S21": t2(s, 1, 0), "bpm_S12": t2(s, 0, 1), "eme": t2(e, 1, 0)}

    # MMI self-imaging length
    nr, nc, lam, W = 2.85, 1.44, 1550.0, 3000.0
    we = W + lam / np.pi * (nc / nr) ** 2 / np.sqrt(nr**2 - nc**2)
    l_th = nr * we**2 / (2 * lam)
    fs = [0.9, 0.95, 1.0, 1.05]
    mb, me = [], []
    for f in fs:
        b = np.abs(ls.solve(T.mmi(f * l_th), "bpm", device="cpu").s()[0]) ** 2
        e = np.abs(ls.solve(T.mmi(f * l_th), "eme").s()[0]) ** 2
        mb.append(float(b[1, 0] + b[2, 0]))
        me.append(float(e[1, 0] + e[2, 0]))
    out["mmi"] = {"length_factors": fs, "bpm_total": mb, "eme_total": me,
                  "max_abs_diff": float(np.max(np.abs(np.array(mb) - np.array(me))))}

    # Y-branch
    p = T.ybranch()
    s = np.abs(ls.solve(p, "bpm", device="cpu").s()[0]) ** 2
    e = np.abs(ls.solve(p, "eme").s()[0]) ** 2
    out["y_branch"] = {"bpm_total": float(s[1, 0] + s[2, 0]), "bpm_total_reverse": float(s[0, 1] + s[0, 2]),
                       "split_imbalance": float(abs(s[1, 0] / s[2, 0] - 1)), "eme_total": float(e[1, 0] + e[2, 0])}

    # 2D slab bend loss against the FDE bend mode
    co, cl, R = ls.ConstantIndex(n_real=1.60), ls.ConstantIndex(n_real=1.45), 20000.0
    ref = solve_modes([Structure2D(profile=Rectangle(size_nm=(800, 1e9)), material=co)], cl, 1550,
                      center_nm=(0, 0), size_nm=(8000, 0), grid_spacing_nm=20, boundary="pml", pml_cells=15,
                      bend_radius_nm=R, polarization="te")[0].loss_db_per_cm

    def bend2(L):
        return DeviceProblem(dimension=2, structures=[Structure(geometry=Box(size_nm=(L + 2000, 800, 1)), material=co)],
                             background=cl, domain_size_nm=(L, 8000, 0), wavelengths=[1550],
                             ports=[T.port2d("o1", -L / 2, width=8000), T.port2d("o2", L / 2, width=8000, direction="-")],
                             options={"bpm": {"grid_spacing_nm": 20, "bend_radius_nm": R, "directions": "forward"}})

    t = {L: abs(ls.solve(bend2(L), "bpm", device="cpu").s()[0][1, 0]) ** 2 for L in (20000, 30000)}
    loss = -10 * np.log10(t[30000] / t[20000]) / 10000 * 1e7
    out["slab_bend"] = {"bpm_db_per_cm": float(loss), "fde_db_per_cm": float(ref), "rel_diff": float(loss / ref - 1)}

    # 3D strip bend loss against the FDE bend mode
    sin, R = ls.ConstantIndex(n_real=2.0), 8000.0
    ref = solve_modes([Structure2D(profile=Rectangle(size_nm=(800, 400)), material=sin)], T.SIO2, 1550,
                      center_nm=(0, 0), size_nm=(5000, 2000), grid_spacing_nm=50, boundary="pml", pml_cells=10,
                      bend_radius_nm=R, group_index="none")[0].loss_db_per_cm

    def bend3(L):
        return DeviceProblem(structures=[Structure(geometry=Box(size_nm=(L + 2000, 800, 400)), material=sin)],
                             background=T.SIO2, domain_size_nm=(L, 5000, 2000), wavelengths=[1550],
                             ports=[Port(name="o1", center_nm=(-L / 2, 0, 0), size_nm=(0, 5000, 2000), direction="+"),
                                    Port(name="o2", center_nm=(L / 2, 0, 0), size_nm=(0, 5000, 2000), direction="-")],
                             options={"bpm": {"grid_spacing_nm": 50, "pml_cells": 10, "pml_reflection": 1e-6,
                                              "bend_radius_nm": R, "directions": "forward"}})

    t = {L: abs(ls.solve(bend3(L), "bpm", device=T.CPU_3D).s()[0][1, 0]) ** 2 for L in (10000, 20000)}
    loss = -10 * np.log10(t[20000] / t[10000]) / 10000 * 1e7
    out["strip_bend_3d"] = {"bpm_db_per_cm": float(loss), "fde_db_per_cm": float(ref), "rel_diff": float(loss / ref - 1)}

    out["wall_s"] = time.perf_counter() - t0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
