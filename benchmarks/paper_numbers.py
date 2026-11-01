"""Every benchmark number quoted in the technical report, computed from one results directory.

    uv run --no-sync python benchmarks/paper_numbers.py benchmarks/results/<dir>

Reads runs.csv, runs.jsonl, references.csv, skips.csv, convergence.csv, cheapest.csv and totals.json
(run analyze.py first); writes <dir>/paper_numbers.json and prints it. Stdlib + numpy only.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import fineness, load_runs  # noqa: E402

METALS = ("metal_sphere", "core_shell", "spheroid", "rod", "dimer")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def stats(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()),
            "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))}


def finest(rows, **match):
    """Finest-resolution ok, non-reference run per (problem, backend, variant, device) matching `match`."""
    best = {}
    for r in rows:
        if r["status"] != "ok" or r["is_reference"] or any(r.get(k) != v for k, v in match.items()):
            continue
        k = (r["problem_id"], r["backend"], r["variant"], r["device"])
        if k not in best or fineness(r) > fineness(best[k]):
            best[k] = r
    return list(best.values())


def main(argv=None):
    d = Path((argv or sys.argv[1:])[0])
    rows = load_runs(d)
    for r in rows:
        r["est_memory_bytes"] = _f(r.get("est_memory_bytes"))
        r["unknowns"] = _f(r.get("unknowns"))
    fam = {r["problem_id"]: r["family"] for r in rows}
    with open(d / "references.csv", newline="", encoding="utf-8") as f:
        refs = {r["problem_id"]: r for r in csv.DictReader(f)}
    with open(d / "convergence.csv", newline="", encoding="utf-8") as f:
        conv = list(csv.DictReader(f))
    with open(d / "cheapest.csv", newline="", encoding="utf-8") as f:
        cheap = list(csv.DictReader(f))
    with open(d / "skips.csv", newline="", encoding="utf-8") as f:
        skips = list(csv.DictReader(f))
    totals = json.loads((d / "totals.json").read_text(encoding="utf-8"))
    out = {"directory": d.name, "totals": totals,
           "problems": len(refs), "problems_by_family": dict(Counter(fam[p] for p in refs if p in fam)),
           "runs_by_method": dict(Counter(r["method"] for r in rows)),
           "errors_by_backend": dict(Counter(r["backend"] for r in rows if r["status"] != "ok")),
           "skip_reasons": dict(Counter((s.get("reason") or "").split(":")[0] for s in skips)),
           "reference_kinds": dict(Counter(r["reference_kind"] for r in refs.values()))}

    # exact methods against miepython on every sphere
    mie = [r for r in rows if r["status"] == "ok" and r["reference_kind"] == "mie" and not r["is_reference"]]
    out["mie_native_vs_ref"] = stats([r["err_headline"] for r in mie if r["backend"] == "lightsmith-mie"])
    out["treams_vs_mie_finest"] = stats([r["err_headline"] for r in finest(mie, backend="treams")])
    out["mie_reference_backends"] = dict(Counter(r["reference_label"] for r in refs.values()
                                                 if r["reference_kind"] == "mie"))

    # finest-resolution errors of the volume methods, by family, device and variant
    fin = {}
    for method in ("dda", "fdtd", "quasistatic"):
        for family in sorted(set(fam.values())):
            for dev in ("gpu", "cpu"):
                for variant in ("center", "weighted", "extrapolate", "default"):
                    sel = [r for r in finest(rows, method=method, family=family, device=dev, variant=variant)]
                    if not sel:
                        continue
                    key = f"{method}/{variant}/{dev}/{family}"
                    fin[key] = {"headline": stats([r["err_headline"] for r in sel]),
                                "ext": stats([r["err_ext_max"] for r in sel]),
                                "abs": stats([r["err_abs_max"] for r in sel]),
                                "finest_resolution": stats([r["resolution"] for r in sel]),
                                "unknowns": stats([r["unknowns"] for r in sel])}
    out["finest_error"] = fin

    # convergence fits (headline), by method/variant/device/family, ladders with >= 4 points
    cv = defaultdict(lambda: {"p": [], "noise": [], "monotone": []})
    for c in conv:
        if c["metric"] != "err_headline" or int(c["n_resolutions"]) < 4:
            continue
        method = "fdtd" if "fdtd" in c["backend"] else ("dda" if "dda" in c["backend"] else c["backend"])
        k = f"{method}/{c['variant']}/{c['device']}/{fam.get(c['problem_id'], '?')}"
        cv[k]["p"].append(_f(c["order_p"]))
        cv[k]["noise"].append(_f(c["noise_sigma_dex"]))
        cv[k]["monotone"].append(_f(c["monotone_fraction"]))
    out["convergence"] = {k: {m: stats(v[m]) for m in v} for k, v in sorted(cv.items())}

    # cheapest run meeting each target: share of problems per family with any run meeting it
    tgt = defaultdict(lambda: Counter())
    for c in cheap:
        k = f"{c['metric']}/{c['family']}/{c['target']}"
        tgt[k]["problems"] += 1
        tgt[k]["met"] += c["backend"] != "none"
        tgt[k][f"by_{c['method']}"] += 1
    out["targets"] = {k: dict(v) for k, v in sorted(tgt.items())}

    # non-Mie references and their self-convergence
    out["non_mie_references"] = {p: {"label": r["reference_label"], "self_change": _f(r["reference_self_change"])}
                                 for p, r in sorted(refs.items()) if r["reference_kind"] != "mie"}

    # Drude staircase: Q_abs per npd (center, gpu)
    st = sorted((r for r in rows if r["problem_id"] == "sphere-drude-r40-water" and r["status"] == "ok"
                 and r["method"] == "dda"), key=lambda r: (r["variant"], r["device"], r["resolution"]))
    out["drude"] = defaultdict(dict)
    for r in st:
        out["drude"][f"{r['variant']}/{r['device']}"][f"{r['resolution']:g}"] = r["q_abs_first"]
    out["drude"]["fdtd"] = {f"{r['resolution']:g}/{r['device']}": r["q_abs_first"] for r in rows
                            if r["problem_id"] == "sphere-drude-r40-water" and r["method"] == "fdtd"
                            and r["status"] == "ok"}
    out["drude"]["mie"] = next((r["q_abs_first"] for r in rows if r["problem_id"] == "sphere-drude-r40-water"
                                and r["is_reference"]), None)

    # CPU vs GPU: matched runs (same problem, backend, variant, resolution)
    key = lambda r: (r["problem_id"], r["backend"], r["variant"], r["resolution"])  # noqa: E731
    by = defaultdict(dict)
    for r in rows:
        if r["status"] == "ok" and r["method"] in ("dda", "fdtd", "mie"):
            by[key(r)][r["device"]] = r
    spec = {}
    with open(d / "runs.jsonl", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if j.get("status") == "ok" and j.get("method") in ("dda", "fdtd", "mie"):
                spec[(j["problem_id"], j["backend"], j["variant"], _f(j["resolution"]), j["device"])] = j
    agree, speed = defaultdict(list), defaultdict(list)
    for k, dv in by.items():
        if "cpu" in dv and "gpu" in dv:
            m = dv["cpu"]["method"]
            a = spec.get((k[0], k[1], k[2], k[3], "cpu"))
            b = spec.get((k[0], k[1], k[2], k[3], "gpu"))
            if a and b:
                x, y = np.array(a["c_ext_nm2"]), np.array(b["c_ext_nm2"])
                agree[m].append(float(np.max(np.abs(x - y) / np.abs(x))))
            speed[m].append((dv["cpu"]["wall_s"] / dv["gpu"]["wall_s"], dv["gpu"]["unknowns"]))
    out["cpu_gpu_agreement_ext"] = {m: stats(v) for m, v in agree.items()}
    out["cpu_gpu_speedup"] = {m: stats([s for s, _ in v]) for m, v in speed.items()}
    out["cpu_gpu_speedup_largest_matched"] = {m: sorted(v, key=lambda t: t[1])[-3:] for m, v in speed.items()}

    # GPU throughput (unknowns per second) at the largest sizes
    big = defaultdict(list)
    for r in rows:
        if r["status"] == "ok" and r["device"] == "gpu" and r["method"] in ("dda", "fdtd"):
            big[r["method"]].append((r["unknowns"], r["wall_s"], r["n_wavelengths"] if "n_wavelengths" in r else ""))
    out["gpu_largest_runs"] = {m: sorted(v)[-3:] for m, v in big.items()}

    # memory: measured peak / estimate
    mem = defaultdict(list)
    for r in rows:
        if r["status"] == "ok" and r["method"] in ("dda", "fdtd") and r["est_memory_bytes"] > 0 \
                and r["peak_mem_bytes"] > 0:
            mem[f"{r['method']}/{r['device']}"].append(r["peak_mem_bytes"] / r["est_memory_bytes"])
    out["memory_measured_over_estimate"] = {k: stats(v) for k, v in mem.items()}

    # wall time by method and device
    wt = defaultdict(float)
    for r in rows:
        if np.isfinite(r["wall_s"]):
            wt[f"{r['method']}/{r['device']}"] += r["wall_s"]
    out["wall_s_by_method_device"] = dict(wt)

    (d / "paper_numbers.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
