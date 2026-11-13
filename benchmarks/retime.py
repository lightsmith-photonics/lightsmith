"""Re-time benchmark runs on a machine with no other compute jobs.

The full-paper benchmark (results/20261105-bem-reference) recorded wall times while other jobs shared
the GPU, and timed CPU runs under tracemalloc, which slows Python-level loops. This script repeats a
selection of its runs at the current library version with neither: no memory tracing, the GPU
synchronised before and after each solve, and every run started only when the GPU and CPU are idle
(utilisation sampled before each run and recorded with it).

Selection (each run rebuilt from the benchmark's suite.json and the options it recorded in runs.jsonl):
  pairs        every CPU/GPU pair of DDA, FDTD and Mie runs (same problem, backend, variant, resolution),
               and the BEM pairs at the default mesh: all spheroids and rods plus every --bem-every-th of
               the others in order of CPU wall time (BEM on the CPU took 11 h of the benchmark);
  largest      the three GPU-only DDA and FDTD runs with the most unknowns;
  cheapest     the run that reached 1 % at the lowest wall time for each problem (cheapest.csv).

CPU and GPU runs of a pair alternate. Runs recorded under 0.5 s are repeated five times, under 20 s
three times, others once; the median is reported. The C_ext of every run is compared with the
benchmark's, to confirm that the same computation was timed.

    uv run --no-sync python benchmarks/retime.py --dry-run
    uv run --no-sync python benchmarks/retime.py --out benchmarks/results/20261112-retime [--resume]
    uv run --no-sync python benchmarks/retime.py --summarize benchmarks/results/20261112-retime

Writes retime.jsonl (one line per run), environment.json and, with --summarize, summary.json.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lightsmith as ls  # noqa: E402

SRC = HERE / "results" / "20261105-bem-reference"
METHODS = ("dda", "fdtd", "mie", "bem")
IDLE_GPU_PCT = 10.0
IDLE_CPU_PCT = 25.0


def _res(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def key(r):
    return (r["problem_id"], r["backend"], r["variant"], _res(r["resolution"]))


def item_id(k, dev):
    return f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|{dev}"


def load(src: Path):
    suite = json.loads((src / "suite.json").read_text(encoding="utf-8"))
    problems = {p["id"]: p for p in suite["problems"]}
    recs = {}
    with open(src / "runs.jsonl", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if j.get("status") == "ok" and not j.get("is_reference_only"):
                recs[(*key(j), j["device"])] = j
    return problems, recs


def plan(problems, recs, bem_every: int):
    by = defaultdict(dict)
    for (p, b, v, res, dev), j in recs.items():
        if j["method"] in METHODS:
            by[(p, b, v, res)][dev] = j
    items = []  # (set, key, device, recorded record)
    bem = []
    for k, dv in by.items():
        if "cpu" not in dv or "gpu" not in dv:
            continue
        m = dv["cpu"]["method"]
        if m == "bem":
            if k[2] == "default":
                bem.append(k)
            continue
        items += [("pairs", k, "cpu", dv["cpu"]), ("pairs", k, "gpu", dv["gpu"])]
    fam = {pid: p["family"] for pid, p in problems.items()}
    chosen = sorted(bem, key=lambda k: by[k]["cpu"]["wall_s"])[::bem_every]
    for k in chosen:
        items += [("pairs", k, "cpu", by[k]["cpu"]), ("pairs", k, "gpu", by[k]["gpu"])]
    # the paper quotes BEM's GPU wall time at the default mesh on spheroids and rods: re-time all of them
    items += [("bem-gpu", k, "gpu", by[k]["gpu"]) for k in bem
              if k not in chosen and fam.get(k[0]) in ("spheroid", "rod")]
    for m in ("dda", "fdtd"):
        gpu_only = [(k, dv["gpu"]) for k, dv in by.items()
                    if "gpu" in dv and "cpu" not in dv and dv["gpu"]["method"] == m]
        gpu_only.sort(key=lambda t: (t[1].get("unknowns") or 0, t[1]["wall_s"]))
        items += [("largest", k, "gpu", j) for k, j in gpu_only[-3:]]
    seen = {(k, d) for _, k, d, _ in items}
    with open(SRC / "cheapest.csv", newline="", encoding="utf-8") as f:
        for c in csv.DictReader(f):
            if c["metric"] != "err_headline" or float(c["target"]) != 0.01:
                continue
            k = (c["problem_id"], c["backend"], c["variant"], _res(c["resolution"]))
            j = recs.get((*k, c["device"]))
            if j is None or j["method"] == "mie":
                continue
            if (k, c["device"]) in seen:
                items.append(("cheapest-dup", k, c["device"], j))
            else:
                items.append(("cheapest", k, c["device"], j))
                seen.add((k, c["device"]))
    return items, len(bem), len(chosen)


def repeats(recorded_s: float) -> int:
    return 5 if recorded_s < 0.5 else 3 if recorded_s < 20 else 1


def gpu_state():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        u, mem, clk, temp = (float(x) for x in out.strip().split(","))
        return {"gpu_util_pct": u, "gpu_mem_used_mib": mem, "gpu_sm_clock_mhz": clk, "gpu_temp_c": temp}
    except Exception as e:  # noqa: BLE001
        return {"gpu_query_error": str(e)}


def wait_idle(psutil, max_wait_s=600.0):
    t0 = time.perf_counter()
    while True:
        cpu = psutil.cpu_percent(interval=0.25)
        g = gpu_state()
        busy = cpu > IDLE_CPU_PCT or g.get("gpu_util_pct", 0.0) > IDLE_GPU_PCT
        waited = time.perf_counter() - t0
        if not busy or waited > max_wait_s:
            return {"cpu_pct": cpu, **g, "waited_s": round(waited, 1), "idle": not busy}
        time.sleep(5.0)


def build(problems, j):
    sim = ls.Simulation.model_validate(problems[j["problem_id"]]["simulation"])
    return sim.model_copy(update={"options": {**sim.options, j["backend"]: j.get("options") or {}}})


def timed(sim, backend, dev, torch):
    if dev != "cpu" and torch is not None:
        # release PyTorch's cached blocks from earlier runs: on Windows the driver may otherwise place a large
        # allocation in system memory (sysmem fallback) and slow the solve severalfold
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = ls.solve(sim, backend, dev)
        torch.cuda.synchronize()
        return res, time.perf_counter() - t0
    t0 = time.perf_counter()
    res = ls.solve(sim, backend, dev)
    return res, time.perf_counter() - t0


def environment(args):
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=HERE.parent).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    cpu = sh(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]) \
        if sys.platform == "win32" else platform.processor()
    try:
        import torch
        tv = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception:  # noqa: BLE001
        tv = {}
    return {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"), "command": sys.argv,
            "source": str(SRC.relative_to(HERE.parent)), "python": sys.version, "platform": platform.platform(),
            "cpu": cpu, "numpy": np.__version__, **tv, "lightsmith_version": ls.__version__,
            "git_commit": sh(["git", "rev-parse", "HEAD"]), "git_dirty_src": bool(sh(["git", "status", "--porcelain", "src"])),
            "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit",
                              "--format=csv,noheader"]),
            "idle_thresholds": {"gpu_util_pct": IDLE_GPU_PCT, "cpu_pct": IDLE_CPU_PCT},
            "bem_every": args.bem_every}


def rel_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.all(np.isfinite(a * b)):
        return math.inf
    return float(np.max(np.abs(a - b) / np.abs(b)))


def run(args):
    import psutil

    problems, recs = load(SRC)
    items, n_bem, n_bem_chosen = plan(problems, recs, args.bem_every)
    items = [it for it in items if it[0] != "cheapest-dup"]
    est = defaultdict(float)
    for s, k, d, j in items:
        est[f"{s}/{j['method']}/{d}"] += j["wall_s"] * repeats(j["wall_s"])
    print(f"[retime] {len(items)} runs; BEM pairs {n_bem_chosen} of {n_bem}; recorded wall x repeats:")
    for k_, v in sorted(est.items()):
        print(f"   {k_:28s} {v / 3600:6.2f} h")
    print(f"   total {sum(est.values()) / 3600:.2f} h")
    if args.dry_run:
        return
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log = out / "retime.jsonl"
    done = set()
    if args.resume and log.exists():
        with open(log, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    env = "environment.json" if not (out / "environment.json").exists() else f"environment-{len(list(out.glob('environment*.json')))}.json"
    (out / env).write_text(json.dumps(environment(args), indent=1, default=str), encoding="utf-8")
    try:
        import torch
        torch = torch if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        torch = None
    import run as bench

    bench.warmup(["cpu", "gpu"], [b for b in ls.all_backends() if b.method in METHODS])
    # pairs alternate CPU and GPU repeat by repeat
    groups = defaultdict(list)
    for s, k, d, j in items:
        groups[(s, k)].append((d, j))
    order = sorted(groups, key=lambda g: (g[0] != "pairs", groups[g][0][1]["method"], g[1][0]))
    if args.limit:  # smoke test: the cheapest few groups of each set and method
        per = defaultdict(list)
        for g in sorted(order, key=lambda g: sum(j["wall_s"] for _, j in groups[g])):
            per[(g[0], groups[g][0][1]["method"])].append(g)
        keep = {g for v in per.values() for g in v[:args.limit]}
        order = [g for g in order if g in keep]
    t_all = time.perf_counter()
    for gi, g in enumerate(order):
        s, k = g
        todo = [(d, j) for d, j in groups[g] if item_id(k, d) not in done]
        if not todo:
            continue
        sim = build(problems, todo[0][1])
        n = max(repeats(j["wall_s"]) for _, j in todo)
        walls = defaultdict(list)
        states = defaultdict(list)
        last = {}
        err = {}
        for _ in range(n):
            for d, j in todo:
                if d in err or len(walls[d]) >= repeats(j["wall_s"]):
                    continue
                states[d].append(wait_idle(psutil))
                try:
                    res, w = timed(sim, j["backend"], d, torch)
                except Exception as e:  # noqa: BLE001
                    err[d] = f"{type(e).__name__}: {e}"
                    continue
                walls[d].append(w)
                last[d] = res
        for d, j in todo:
            rec = {"id": item_id(k, d), "set": s, "problem_id": k[0], "family": j.get("family"),
                   "backend": k[1], "method": j["method"], "variant": k[2], "resolution": k[3], "device": d,
                   "unknowns": j.get("unknowns"), "n_wavelengths": j.get("n_wavelengths"),
                   "recorded_wall_s": j["wall_s"], "recorded_precision": j.get("precision"),
                   "idle_state": states[d]}
            if d in err:
                rec.update(status="error", message=err[d])
            else:
                r = last[d]
                rec.update(status="ok", wall_s=statistics.median(walls[d]), walls_s=walls[d],
                           solver_runtime_s=r.provenance.get("runtime_s"),
                           precision=r.provenance.get("precision", "double"),
                           c_ext_nm2=list(map(float, r.c_ext_nm2)),
                           c_ext_vs_recorded=rel_diff(r.c_ext_nm2, j["c_ext_nm2"]))
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            done.add(rec["id"])
        msg = " ".join(f"{d}={statistics.median(walls[d]):.3g}s(rec {j['wall_s']:.3g})" if d not in err else f"{d}=ERR"
                       for d, j in todo)
        print(f"[{gi + 1}/{len(order)} {time.perf_counter() - t_all:7.0f}s] {s} {todo[0][1]['method']} {k[0]} "
              f"{k[2]} {k[3]}: {msg}", flush=True)


def stats(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()),
            "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))}


def summarize(d: Path):
    rows = [json.loads(line) for line in open(d / "retime.jsonl", encoding="utf-8") if line.strip()]
    ok = [r for r in rows if r["status"] == "ok"]
    out = {"directory": str(d), "runs": len(rows), "ok": len(ok),
           "errors": [(r["id"], r["message"]) for r in rows if r["status"] != "ok"]}
    idle = [s for r in rows for s in r["idle_state"]]
    out["idle"] = {"checks": len(idle), "not_idle_after_wait": sum(not s.get("idle", True) for s in idle),
                   "cpu_pct": stats([s.get("cpu_pct") for s in idle]),
                   "gpu_util_pct": stats([s.get("gpu_util_pct") for s in idle]),
                   "waited_s_total": sum(s.get("waited_s", 0) for s in idle)}
    out["c_ext_vs_recorded"] = {
        f"{m}/{dev}": stats([r["c_ext_vs_recorded"] for r in ok if r["method"] == m and r["device"] == dev])
        for m in METHODS for dev in ("cpu", "gpu")}
    out["precision"] = {f"{m}/{dev}": sorted({str(r["precision"]) for r in ok if r["method"] == m and r["device"] == dev})
                        for m in METHODS for dev in ("cpu", "gpu")}
    # idle wall time over the benchmark's recorded wall time, same run
    out["idle_over_recorded"] = {f"{m}/{dev}": stats([r["wall_s"] / r["recorded_wall_s"] for r in ok
                                                      if r["method"] == m and r["device"] == dev])
                                 for m in METHODS for dev in ("cpu", "gpu")}
    allr = {"cpu": [], "gpu": []}
    for r in ok:
        allr[r["device"]].append(r["wall_s"] / r["recorded_wall_s"])
    out["idle_over_recorded_all"] = {dev: stats(v) for dev, v in allr.items()}
    # speed-ups
    pairs = defaultdict(dict)
    for r in ok:
        if r["set"] == "pairs":
            pairs[r["id"].rsplit("|", 1)[0]][r["device"]] = r
    sp = defaultdict(list)
    agree = defaultdict(list)
    for k, dv in pairs.items():
        if "cpu" in dv and "gpu" in dv:
            c, g = dv["cpu"], dv["gpu"]
            sp[c["method"]].append((c["wall_s"] / g["wall_s"], g.get("unknowns") or 0, c["problem_id"],
                                    c["wall_s"], g["wall_s"]))
            agree[c["method"]].append(rel_diff(g["c_ext_nm2"], c["c_ext_nm2"]))
    out["speedup"] = {m: stats([t[0] for t in v]) for m, v in sp.items()}
    out["cpu_gpu_agreement_ext"] = {m: stats(v) for m, v in agree.items()}
    out["speedup_recorded_same_pairs"] = {m: stats([dv["cpu"]["recorded_wall_s"] / dv["gpu"]["recorded_wall_s"]
                                                    for dv in pairs.values() if "cpu" in dv and "gpu" in dv
                                                    and dv["cpu"]["method"] == m]) for m in sp}
    f = sp.get("fdtd", [])
    out["fdtd_speedup_by_cells"] = {
        "below_5e4": stats([t[0] for t in f if t[1] < 5e4]),
        "5e4_to_1e5": stats([t[0] for t in f if 5e4 <= t[1] < 1e5]),
        "1e5_to_2e5": stats([t[0] for t in f if 1e5 <= t[1] < 2e5]),
        "above_2e5": stats([t[0] for t in f if t[1] >= 2e5]),
        "max_cells_matched": max((t[1] for t in f), default=None)}
    dd = sorted(sp.get("dda", []), key=lambda t: (t[1], t[0]))
    out["dda_speedup_largest_matched"] = [{"speedup": t[0], "dipoles": t[1], "problem": t[2]} for t in dd[-3:]]
    out["dda_speedup_by_dipoles"] = {"below_500": stats([t[0] for t in dd if t[1] < 500]),
                                     "500_and_more": stats([t[0] for t in dd if t[1] >= 500])}
    b = sp.get("bem", [])
    out["bem_speedup_by_family"] = {}
    fam = {r["problem_id"]: r.get("family") for r in ok}
    for fm in sorted({fam[t[2]] for t in b}):
        out["bem_speedup_by_family"][fm] = stats([t[0] for t in b if fam[t[2]] == fm])
    bem_gpu = [r for r in ok if r["method"] == "bem" and r["device"] == "gpu" and r["set"] in ("pairs", "bem-gpu")]
    out["bem_gpu_default_wall"] = {fm: {"idle": stats([r["wall_s"] for r in bem_gpu if r["family"] == fm]),
                                        "recorded": stats([r["recorded_wall_s"] for r in bem_gpu if r["family"] == fm])}
                                   for fm in ("spheroid", "rod")}
    out["largest_gpu_runs"] = [{"method": r["method"], "problem": r["problem_id"], "unknowns": r["unknowns"],
                                "n_wavelengths": r["n_wavelengths"], "wall_s": r["wall_s"],
                                "recorded_wall_s": r["recorded_wall_s"]} for r in ok if r["set"] == "largest"]
    # cheapest run to 1 % per problem: wall time now (cheapest set plus runs re-timed as pairs)
    by_id = {r["id"]: r for r in ok}
    groups = {"single and layered spheres": ("dielectric_sphere", "metal_sphere", "drude_sphere", "core_shell"),
              "dimers": ("dimer",)}
    cheap = defaultdict(list)
    cheap_rec = defaultdict(list)
    with open(SRC / "cheapest.csv", newline="", encoding="utf-8") as fh:
        for c in csv.DictReader(fh):
            if c["metric"] != "err_headline" or float(c["target"]) != 0.01 or c["method"] == "mie":
                continue
            rid = item_id((c["problem_id"], c["backend"], c["variant"], _res(c["resolution"])), c["device"])
            if rid in by_id:
                cheap[c["family"]].append(by_id[rid]["wall_s"])
                cheap_rec[c["family"]].append(float(c["wall_s"]))
    out["cheapest_1pct_wall_by_family"] = {fm: {"idle": stats(v), "recorded": stats(cheap_rec[fm])}
                                           for fm, v in cheap.items()}
    out["cheapest_1pct_groups"] = {g: {"idle": stats([w for fm in fams for w in cheap.get(fm, [])]),
                                       "recorded": stats([w for fm in fams for w in cheap_rec.get(fm, [])])}
                                   for g, fams in groups.items()}
    (d / "summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("runs", "ok", "errors", "idle", "speedup", "fdtd_speedup_by_cells",
                                          "dda_speedup_largest_matched", "idle_over_recorded_all")}, indent=1,
                     default=float))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=HERE / "results" / f"{_dt.date.today():%Y%m%d}-retime")
    ap.add_argument("--bem-every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: at most this many groups per set and method")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.summarize:
        summarize(args.summarize)
    else:
        run(args)


if __name__ == "__main__":
    main()
