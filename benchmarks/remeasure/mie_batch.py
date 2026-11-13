"""Native Mie solver: CPU versus GPU wall time as a function of the number of wavelengths.

The benchmark solves nine wavelengths per problem, where kernel-launch overhead makes the GPU slower;
this measures where the GPU starts to pay off. Sphere r = 50 nm, n = 1.5 + 0.01i, in water. Wall times
are medians of three runs after a warm-up.

    uv run --no-sync python benchmarks/remeasure/mie_batch.py

Writes benchmarks/remeasure/results/mie_batch.json.
"""

from __future__ import annotations

import datetime as _dt
import json
import statistics
import time
from pathlib import Path

import numpy as np

import lightsmith as ls

OUT = Path(__file__).resolve().parent / "results" / "mie_batch.json"


def sim(n_wl):
    return ls.Simulation.model_validate({
        "particles": [{"shape": {"kind": "sphere", "radius_nm": 50.0},
                       "material": {"kind": "constant_index", "n_real": 1.5, "n_imag": 0.01}}],
        "medium_index": 1.33, "wavelengths": list(np.linspace(400.0, 800.0, n_wl))})


def timed(s, device, repeats=3):
    ls.solve(s, "lightsmith-mie", device)
    walls, res = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = ls.solve(s, "lightsmith-mie", device)
        walls.append(time.perf_counter() - t0)
    return res, statistics.median(walls)


def main():
    from _machine import NOTE, state

    before = state()
    rows = []
    for n_wl in (9, 10**3, 10**4, 10**5, 10**6):
        s = sim(n_wl)
        (rc, tc), (rg, tg) = timed(s, "cpu"), timed(s, "gpu")
        a, b = np.array(rc.c_ext_nm2), np.array(rg.c_ext_nm2)
        row = {"wavelengths": n_wl, "cpu_s": tc, "gpu_s": tg, "speedup": tc / tg,
               "max_rel_diff": float(np.max(np.abs(a - b) / np.abs(a)))}
        print(json.dumps(row), flush=True)
        rows.append(row)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                               "note": NOTE, "machine_state": {"before": before, "after": state()},
                               "doctor": ls.doctor()},
                              indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
