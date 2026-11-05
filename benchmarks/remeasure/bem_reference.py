"""BEM mesh ladders for the benchmark's spheroids and rods, to serve as their reference.

The full-paper suite has no exact method for spheroids and flat-ended cylinders, so their reference was
the finest DDA run, which is not converged. BEM meshes only the surface and converges smoothly (it is
within 1e-4 of Mie on every sphere of the suite at its default mesh). This script solves each spheroid
and rod problem of a results directory with BEM on the GPU in double precision at
elements_per_size = 8, 10 (the default), 12 and 14, skipping meshes whose estimate exceeds MAX_MEM_GB.
`benchmarks/rereference.py` then rescores the other methods against the finest BEM run.

    uv run --no-sync python benchmarks/remeasure/bem_reference.py benchmarks/results/20261104-full-paper

Writes benchmarks/remeasure/results/bem_reference.jsonl (one line per solve, appended as it goes; a
problem/mesh already in the file is not solved again).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

import lightsmith as ls

OUT = Path(__file__).resolve().parent / "results" / "bem_reference.jsonl"
FAMILIES = ("spheroid", "rod")
LADDER = (8, 10, 12, 14)
MAX_MEM_GB = 9.5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="benchmark results directory (its suite.json is used)")
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args(argv)
    suite = json.loads((args.results / "suite.json").read_text(encoding="utf-8"))
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["problem_id"], r["elements_per_size"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for p in suite["problems"]:
        if p["family"] not in FAMILIES:
            continue
        base = ls.Simulation.model_validate(p["simulation"])
        for eps in LADDER:
            if (p["id"], eps) in done:
                continue
            opts = {"elements_per_size": eps, "precision": "double"}
            sim = base.model_copy(update={"options": {**base.options, "lightsmith-bem": opts}})
            est = ls.estimate(sim, method="bem", device=args.device)
            est = est if isinstance(est, dict) else est.model_dump()
            mem_gb = (est.get("memory_bytes") or 0) / 1e9
            rec = {"problem_id": p["id"], "family": p["family"], "elements_per_size": eps,
                   "device": args.device, "unknowns": est.get("unknowns"), "solver": est.get("solver"),
                   "est_memory_gb": mem_gb, "date": _dt.date.today().isoformat()}
            if mem_gb > MAX_MEM_GB:
                rec.update(status="skipped", message=f"estimate {mem_gb:.1f} GB > {MAX_MEM_GB} GB")
            else:
                t0 = time.perf_counter()
                res = ls.solve(sim, "lightsmith-bem", args.device)
                rec.update(status="ok", wall_s=time.perf_counter() - t0, wavelengths_nm=res.wavelengths_nm,
                           c_ext_nm2=res.c_ext_nm2, c_sca_nm2=res.c_sca_nm2, c_abs_nm2=res.c_abs_nm2,
                           equivalent_radius_nm=res.equivalent_radius_nm, warnings=res.warnings,
                           mesh=res.provenance.get("mesh"), version=res.provenance.get("backend_version"))
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            print(f"[bem] {p['id']} eps={eps} {rec['status']} {rec.get('wall_s', 0):.1f}s "
                  f"unknowns={rec['unknowns']}", flush=True)


if __name__ == "__main__":
    main()
