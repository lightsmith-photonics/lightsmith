"""Cross-method benchmark runner for lightsmith.

For every problem x backend x device x variant x resolution it records cross
sections, relative error against a reference, wall time, peak memory and
provenance. Backends are discovered at runtime (`ls.all_backends()`,
`available()`, `supports(sim)`), so new backends join the benchmark without
changes here, provided they expose a resolution option this file knows how to
map (see RESOLUTION_ADAPTERS).

Reference: Mie when a Mie backend supports the problem (single sphere, or a
layered sphere if the installed Mie backend handles it). Otherwise the most
accurate method present (order: tmatrix > dda > fdtd > others) at its highest
resolution, in its default variant, labelled "highest-resolution". Its
self-convergence (relative change from the next-lower resolution) is recorded
as `reference_self_change` so readers can judge it.

Usage:
    uv run --no-sync python benchmarks/run.py --suite smoke --devices cpu,gpu
    uv run --no-sync python benchmarks/run.py --suite full --devices gpu --methods mie,dda --dry-run

Outputs go to benchmarks/results/<timestamp>-<suite>/:
    environment.json   machine, versions, backends, command line
    suite.json         the exact problems run
    runs.csv           one row per run (tidy)
    runs.jsonl         the same plus full spectra, provenance and warnings
    spectra.csv        long format: one row per run x wavelength, with reference values
    references.csv     which run is the reference for each problem, and its self-convergence
    skips.csv          backend/problem/device combinations not run, and why
    convergence.csv, cheapest.csv   written by analyze.py (run automatically at the end)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import platform
import subprocess
import sys
import time
import tracemalloc
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lightsmith as ls  # noqa: E402
from problems import build_suite  # noqa: E402

# Method preference when no Mie reference exists (most accurate first).
REFERENCE_ORDER = ("mie", "tmatrix", "dda", "fdtd")
QUANTITIES = ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2")
NEGLIGIBLE = 1e-6  # a quantity whose reference max is below NEGLIGIBLE * max(C_ext) gets no relative error


# --- mapping semantic resolutions onto backend options ----------------------

# quantity -> list of (option field name, converter(value, sim) -> option value), first match wins.
def _per_um(v, sim):
    return 1000.0 / v


def _per_nm(v, sim):
    return 1.0 / v


def _per_wavelength(v, sim):
    # cells per (shortest wavelength in the densest material)
    wl = sim.wavelength_array()
    n_max = max(float(np.abs(m.n(wl)).max()) for p in sim.particles for m in p.region_materials())
    return max(1, int(round(wl.min() / max(n_max, sim.medium_index) / v)))


_ident = lambda v, sim: v  # noqa: E731

RESOLUTION_ADAPTERS = {
    "dipoles_per_diameter": [("dipoles_per_diameter", lambda v, s: int(v))],
    "mesh_nm": [(f, _ident) for f in ("spacing_nm", "mesh_nm", "resolution_nm", "grid_nm", "dx_nm", "cell_nm", "cell_size_nm",
                                      "step_nm", "grid_spacing_nm", "spacing_nm")]
    + [(f, _per_um) for f in ("resolution", "pixels_per_um", "cells_per_um")]
    + [(f, _per_nm) for f in ("cells_per_nm", "pixels_per_nm")]
    + [(f, _per_wavelength) for f in ("cells_per_wavelength", "points_per_wavelength", "ppw")],
    "multipole_order": [(f, lambda v, s: int(v)) for f in ("lmax", "l_max", "multipole_order", "order", "n_max",
                                                           "nmax")],
}


def options_model(backend):
    """The pydantic options model of a backend, if it has one (by convention `*Options`)."""
    from pydantic import BaseModel

    mod = sys.modules.get(backend.__module__)
    explicit = getattr(backend, "options_model", None) or getattr(backend, "Options", None)
    if isinstance(explicit, type) and issubclass(explicit, BaseModel):
        return explicit
    cands = [v for k, v in vars(mod).items() if isinstance(v, type) and issubclass(v, BaseModel)
             and v is not BaseModel and k.endswith("Options")]
    if len(cands) > 1:  # prefer one named after the method, e.g. DDAOptions for method "dda"
        named = [c for c in cands if c.__name__.lower().startswith(backend.method.lower())]
        cands = named or cands
    return cands[0] if cands else None


def resolution_knob(backend, quantity):
    """(field, converter) for this backend and semantic quantity, or None."""
    model = options_model(backend)
    if model is None:
        return None
    fields = model.model_fields
    for field, conv in RESOLUTION_ADAPTERS.get(quantity, []):
        if field in fields:
            return field, conv
    return None


def variant_options(backend, variant: str, field: str | None, value):
    """Extra options for a named variant. Unknown variants are passed as {'discretization': name}."""
    if backend.method != "dda" or variant == "default":
        return {field: value} if field else {}
    if variant == "extrapolate":
        model = options_model(backend)
        disc = model.model_fields.get("discretization") if model else None
        ladder = sorted({max(4, int(round(value / 2))), max(5, int(round(3 * value / 4))), int(value)})
        opts = {"extrapolate": ladder}
        # Newer dda.py may expose extrapolation as a discretization mode.
        if disc is not None and "extrapolate" in str(disc.annotation):
            opts["discretization"] = "extrapolate"
        return opts
    return {field: value, "discretization": variant}


# --- devices & measurement --------------------------------------------------


def device_list(arg: str) -> list[str]:
    out = []
    for d in arg.split(","):
        d = d.strip()
        if not d:
            continue
        if d == "gpu":
            try:
                ls.resolve_device("gpu")
            except Exception as e:  # noqa: BLE001
                print(f"[run] gpu requested but unavailable: {e}", file=sys.stderr)
                continue
        out.append(d)
    return out


def _torch():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def measure(fn, device: str):
    """Run fn(); return (result, wall_s, peak_bytes, mem_kind)."""
    torch = _torch() if device != "cpu" else None
    if torch is not None:
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        res = fn()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        return res, wall, int(torch.cuda.max_memory_allocated() - base), "cuda_max_memory_allocated"
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        res = fn()
    finally:
        wall = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return res, wall, int(peak), "tracemalloc_peak"


def warmup(devices, backends):
    """Pay one-off costs (numba JIT, CUDA context, cuFFT plans, CUDA graphs) before timing.
    Only tiny problems: anything whose estimate is not tiny is skipped."""
    sim = ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=10),
                                                 material=ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07))],
                         wavelengths=[500.0], medium_index=1.33,
                         options={"dda": {"dipoles_per_diameter": 6}, "fdtd": {"spacing_nm": 4.0}})
    tiny = {"max_dipoles": 2000, "max_cells": 2e5, "max_cell_steps": 3e8}
    for b in backends:
        for dev in devices if b.gpu else ["cpu"]:
            try:
                if not (b.available()[0] and b.supports(sim)[0]):
                    continue
                lim = tiny if dev == "cpu" else {**tiny, "max_cells": 2e6, "max_cell_steps": 5e9}
                if budget_violation(ls.estimate(sim, b.name, dev), lim):
                    continue
                t0 = time.perf_counter()
                ls.solve(sim, b.name, dev)
                print(f"[warmup] {b.name}/{dev} {time.perf_counter() - t0:.1f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[warmup] {b.name}/{dev} failed: {type(e).__name__}: {e}", flush=True)


# --- errors -----------------------------------------------------------------


def rel_errors(res: dict, ref: dict) -> dict:
    out = {}
    ext_scale = max(np.max(np.abs(ref["c_ext_nm2"])), 1e-300)
    for q in QUANTITIES:
        x, y = np.asarray(res[q], float), np.asarray(ref[q], float)
        key = q.split("_")[1]
        if np.max(np.abs(y)) < NEGLIGIBLE * ext_scale:
            out[f"err_{key}_max"] = out[f"err_{key}_rms"] = float("nan")
            continue
        rel = np.abs(x - y) / np.maximum(np.abs(y), 1e-300)
        out[f"err_{key}_max"] = float(rel.max())
        out[f"err_{key}_rms"] = float(np.sqrt(np.mean(rel**2)))
    vals = [out[k] for k in ("err_ext_max", "err_abs_max") if not math.isnan(out[k])]
    out["err_headline"] = max(vals) if vals else float("nan")
    return out


# --- planning ---------------------------------------------------------------


def budget_violation(est: dict, budget: dict) -> str | None:
    """Budgets per device: max_dipoles (DDA), max_cells and max_cell_steps (FDTD = cells x time steps)."""
    checks = [("max_dipoles", est.get("dipoles")), ("max_cells", est.get("cells")),
              ("max_cell_steps", (est.get("cells") or 0) * (est.get("steps_estimate") or 0) or None)]
    for key, val in checks:
        if val and budget.get(key) and val > budget[key]:
            return f"budget: {key.removeprefix('max_')} {val:.3g} > {key} {budget[key]:.3g}"
    return None


def plan_problem(problem, backends, devices, budgets, max_mem_gb, extra_options):
    """List of run dicts (not yet executed) and skips for one problem."""
    sim = ls.Simulation.model_validate(problem["simulation"])
    runs, skips = [], []
    for b in backends:
        ok, why = b.available()
        if ok:
            ok, why = b.supports(sim)
        if not ok:
            skips.append({"backend": b.name, "device": "*", "reason": why})
            continue
        sweep = problem.get("sweeps", {}).get(b.method)
        knob = resolution_knob(b, sweep["quantity"]) if sweep else None
        if sweep and knob is None:
            skips.append({"backend": b.name, "device": "*", "reason":
                          f"no option maps sweep quantity {sweep['quantity']!r}; running defaults only"})
        values = sweep["values"] if (sweep and knob) else [None]
        variant_spec = problem.get("variants", {}).get(b.method) or {"default": None}
        for dev in (devices if b.gpu else ["cpu"]):
            budget = budgets.get("gpu" if dev != "cpu" else "cpu", {})
            for variant, subset in variant_spec.items():
                for v in values:
                    if subset is not None and v is not None and v not in subset:
                        continue
                    field = knob[0] if knob else None
                    val = knob[1](v, sim) if knob else None
                    opts = {**extra_options.get(b.method, {}), **extra_options.get(b.name, {}),
                            **variant_options(b, variant, field, val)}
                    run_sim = sim.model_copy(update={"options": {**sim.options, b.name: opts}})
                    try:
                        est = ls.estimate(run_sim, b.name, dev)
                    except Exception as e:  # noqa: BLE001
                        skips.append({"backend": b.name, "device": dev, "variant": variant, "resolution": v,
                                      "reason": f"estimate failed: {type(e).__name__}: {e}"})
                        continue
                    reason = budget_violation(est, budget)
                    mem = est.get("memory_bytes")
                    if reason:
                        pass
                    elif mem and max_mem_gb and mem > max_mem_gb * 2**30:
                        reason = f"budget: estimated {mem / 2**30:.2f} GiB > {max_mem_gb} GiB"
                    elif est.get("fits") is False:
                        reason = "backend estimate says it does not fit"
                    if reason:
                        skips.append({"backend": b.name, "device": dev, "variant": variant, "resolution": v,
                                      "reason": reason})
                        continue
                    runs.append({"backend": b, "device": dev, "variant": variant,
                                 "resolution_quantity": sweep["quantity"] if (sweep and knob) else "",
                                 "resolution": v, "option_field": field or "", "option_value": val,
                                 "sim": run_sim, "estimate": est})
    return runs, skips


# --- execution --------------------------------------------------------------


def _flatten(x):
    if isinstance(x, (list, tuple)):
        for y in x:
            yield from _flatten(y)
    elif isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        yield int(x)


def execute(run) -> dict:
    b, dev = run["backend"], run["device"]
    rec = {"backend": b.name, "method": b.method, "variant": run["variant"], "device": dev,
           "resolution_quantity": run["resolution_quantity"], "resolution": run["resolution"],
           "option_field": run["option_field"], "option_value": run["option_value"],
           "estimate": run["estimate"], "options": run["sim"].options.get(b.name, {})}
    try:
        res, wall, peak, kind = measure(lambda: ls.solve(run["sim"], b.name, dev), dev)
    except Exception as e:  # noqa: BLE001
        rec.update(status="error", message=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=3))
        return rec
    prov = res.provenance
    flat = list(_flatten(prov.get("iterations")))
    rec.update(
        status="ok", message="", wall_s=wall, peak_mem_bytes=peak, mem_kind=kind,
        solver_runtime_s=prov.get("runtime_s"), framework=prov.get("framework", "numpy" if dev == "cpu" else ""),
        precision=prov.get("precision", "double"),
        dipoles=(max(prov["dipoles"]) if isinstance(prov.get("dipoles"), list) else prov.get("dipoles")),
        iterations_total=int(sum(flat)) if flat else None, iterations_max=int(max(flat)) if flat else None,
        wavelengths_nm=res.wavelengths_nm, c_ext_nm2=res.c_ext_nm2, c_sca_nm2=res.c_sca_nm2,
        c_abs_nm2=res.c_abs_nm2, equivalent_radius_nm=res.equivalent_radius_nm,
        provenance=prov, warnings=res.warnings,
    )
    return rec


def pick_reference(records, problem):
    ok = [r for r in records if r["status"] == "ok"]
    mie = [r for r in ok if r["method"] == "mie"]
    if mie:
        mie.sort(key=lambda r: (r["backend"] != "miepython", r["backend"]))
        r = mie[0]
        return r, "mie", f"Mie ({r['backend']})", float("nan")
    for method in REFERENCE_ORDER[1:] + tuple(sorted({r["method"] for r in ok} - set(REFERENCE_ORDER))):
        cands = [r for r in ok if r["method"] == method]
        if not cands:
            continue
        default_variant = next(iter(problem.get("variants", {}).get(method) or {"default": None}))
        pref = [r for r in cands if r["variant"] == default_variant] or cands

        def key(r):
            res = r["resolution"]
            # higher resolution first; for mesh sizes smaller is finer
            fine = -res if r["resolution_quantity"] == "mesh_nm" else res
            return (fine if res is not None else -math.inf, r["precision"] == "double", r["device"] == "cpu")

        pref.sort(key=key, reverse=True)
        ref = pref[0]
        # self-convergence: same backend/variant/device, next-lower resolution
        same = [r for r in pref if r["device"] == ref["device"] and r is not ref and r["resolution"] is not None]
        same.sort(key=key, reverse=True)
        change = float("nan")
        if same:
            change = rel_errors(same[0], ref)["err_headline"]
        label = (f"highest-resolution {ref['backend']} ({ref['variant']}) @ {ref['resolution_quantity']}="
                 f"{ref['resolution']}, {ref['device']}")
        return ref, "highest_resolution", label, change
    return None, "none", "no successful run", float("nan")


RUN_COLUMNS = [
    "suite", "problem_id", "family", "backend", "method", "variant", "device", "framework", "precision",
    "resolution_quantity", "resolution", "option_field", "option_value", "n_wavelengths", "status", "message",
    "wall_s", "solver_runtime_s", "peak_mem_bytes", "mem_kind", "est_memory_bytes", "dipoles", "unknowns",
    "iterations_total", "iterations_max", "is_reference", "reference_kind", "reference_label",
    "err_ext_max", "err_ext_rms", "err_sca_max", "err_sca_rms", "err_abs_max", "err_abs_rms", "err_headline",
    "q_ext_first", "q_abs_first", "n_warnings", "warnings",
]


class Writer:
    def __init__(self, out: Path):
        self.out = out
        self.files = {}
        self.writers = {}

    def row(self, name, columns, row):
        if name not in self.writers:
            f = open(self.out / name, "w", newline="", encoding="utf-8")
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            self.files[name], self.writers[name] = f, w
        self.writers[name].writerow({k: _fmt(row.get(k)) for k in columns})
        self.files[name].flush()

    def jsonl(self, name, obj):
        with open(self.out / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=_json_default) + "\n")

    def close(self):
        for f in self.files.values():
            f.close()


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.10g}"
    if isinstance(v, (list, tuple)):
        return json.dumps(v)
    return v


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def environment(args, backends) -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=HERE.parent).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    cpu = platform.processor()
    if sys.platform == "win32":
        cpu = sh(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]) or cpu
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "command": sys.argv, "args": vars(args) | {"out": str(args.out)},
        "python": sys.version, "platform": platform.platform(), "cpu": cpu,
        "lightsmith_version": ls.__version__, "doctor": ls.doctor(),
        "backends": [b.info() for b in backends],
        "git_commit": sh(["git", "rev-parse", "HEAD"]) or "uncommitted",
        "git_dirty": bool(sh(["git", "status", "--porcelain"])),
        "numpy": np.__version__,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="smoke", help="smoke | full | path to a suite JSON")
    ap.add_argument("--devices", default="cpu,gpu", help="comma list: cpu, gpu, cuda:N, torch-cpu")
    ap.add_argument("--methods", default="", help="comma list of methods or backend names (default: all)")
    ap.add_argument("--problems", default="", help="comma list of problem ids or id prefixes (default: all)")
    ap.add_argument("--max-mem-gb", type=float, default=6.0, help="skip runs whose estimate exceeds this")
    ap.add_argument("--budget", default="", help='JSON overriding suite budgets, e.g. {"cpu":{"max_dipoles":5000}}')
    ap.add_argument("--extra-options", default="{}", help="JSON of options merged into every run, keyed by "
                    "method or backend name")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="plan and estimate only; no solves")
    ap.add_argument("--no-analyze", action="store_true")
    args = ap.parse_args(argv)

    suite = build_suite(args.suite)
    budgets = suite.get("budgets", {})
    if args.budget:
        for k, v in json.loads(args.budget).items():
            budgets[k] = {**budgets.get(k, {}), **v}
    extra = suite.get("options", {})  # suite-wide solver options, overridden by --extra-options
    for k, v in json.loads(args.extra_options).items():
        extra[k] = {**extra.get(k, {}), **v}
    wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
    backends = [b for b in ls.all_backends() if not wanted or b.method in wanted or b.name in wanted]
    devices = device_list(args.devices)
    problems = suite["problems"]
    if args.problems:
        pats = [p.strip() for p in args.problems.split(",") if p.strip()]
        problems = [p for p in problems if any(p["id"].startswith(q) for q in pats)]

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or HERE / "results" / f"{stamp}-{suite['suite']}{'-dryrun' if args.dry_run else ''}"
    out.mkdir(parents=True, exist_ok=True)
    args.out = out
    (out / "suite.json").write_text(json.dumps({**suite, "problems": problems}, indent=1), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(environment(args, backends), indent=1, default=str),
                                          encoding="utf-8")
    print(f"[run] suite={suite['suite']} problems={len(problems)} devices={devices} "
          f"backends={[b.name for b in backends]} -> {out}", flush=True)

    w = Writer(out)
    if not args.dry_run:
        warmup(devices, backends)
    t_suite = time.perf_counter()
    totals = {"runs": 0, "ok": 0, "error": 0, "skipped": 0}
    for pi, problem in enumerate(problems):
        runs, skips = plan_problem(problem, backends, devices, budgets, args.max_mem_gb, extra)
        for s in skips:
            w.row("skips.csv", ["problem_id", "backend", "device", "variant", "resolution", "reason"],
                  {"problem_id": problem["id"], **s})
        totals["skipped"] += len(skips)
        if args.dry_run:
            for r in runs:
                e = r["estimate"]
                w.row("plan.csv", ["problem_id", "backend", "device", "variant", "resolution", "dipoles",
                                   "memory_bytes", "solves"],
                      {"problem_id": problem["id"], "backend": r["backend"].name, "device": r["device"],
                       "variant": r["variant"], "resolution": r["resolution"], "dipoles": e.get("dipoles"),
                       "memory_bytes": e.get("memory_bytes"), "solves": e.get("solves")})
            totals["runs"] += len(runs)
            print(f"[plan] {problem['id']}: {len(runs)} runs, {len(skips)} skips", flush=True)
            continue

        t_p = time.perf_counter()
        records = []
        for r in runs:
            rec = execute(r)
            records.append(rec)
            totals["runs"] += 1
            totals[rec["status"]] += 1
            tag = f"{rec['backend']}/{rec['variant']}/{rec['device']}@{rec['resolution']}"
            print(f"  {problem['id']} {tag}: {rec['status']} "
                  f"{rec.get('wall_s', 0):.2f}s {rec.get('message', '')[:120]}", flush=True)
        ref, ref_kind, ref_label, ref_change = pick_reference(records, problem)
        w.row("references.csv", ["problem_id", "family", "reference_kind", "reference_label",
                                 "reference_self_change"],
              {"problem_id": problem["id"], "family": problem["family"], "reference_kind": ref_kind,
               "reference_label": ref_label, "reference_self_change": ref_change})
        for i, rec in enumerate(records):
            row = {"suite": suite["suite"], "problem_id": problem["id"], "family": problem["family"], **rec,
                   "est_memory_bytes": rec["estimate"].get("memory_bytes"),
                   "unknowns": rec.get("dipoles") or rec["estimate"].get("dipoles") or rec["estimate"].get("cells"),
                   "is_reference": rec is ref, "reference_kind": ref_kind, "reference_label": ref_label}
            if rec["status"] == "ok":
                row["n_wavelengths"] = len(rec["wavelengths_nm"])
                geo = math.pi * rec["equivalent_radius_nm"] ** 2
                row["q_ext_first"] = rec["c_ext_nm2"][0] / geo
                row["q_abs_first"] = rec["c_abs_nm2"][0] / geo
                row["n_warnings"] = len(rec["warnings"])
                row["warnings"] = " | ".join(rec["warnings"])[:500]
                if ref is not None:
                    row.update(rel_errors(rec, ref))
                for j, wl in enumerate(rec["wavelengths_nm"]):
                    w.row("spectra.csv", ["problem_id", "run_index", "backend", "variant", "device", "resolution",
                                          "wavelength_nm", "c_ext_nm2", "c_sca_nm2", "c_abs_nm2",
                                          "ref_c_ext_nm2", "ref_c_sca_nm2", "ref_c_abs_nm2", "is_reference",
                                          "equivalent_radius_nm"],
                          {"problem_id": problem["id"], "run_index": i, "backend": rec["backend"],
                           "variant": rec["variant"], "device": rec["device"], "resolution": rec["resolution"],
                           "wavelength_nm": wl, "c_ext_nm2": rec["c_ext_nm2"][j], "c_sca_nm2": rec["c_sca_nm2"][j],
                           "c_abs_nm2": rec["c_abs_nm2"][j],
                           "ref_c_ext_nm2": ref["c_ext_nm2"][j] if ref else None,
                           "ref_c_sca_nm2": ref["c_sca_nm2"][j] if ref else None,
                           "ref_c_abs_nm2": ref["c_abs_nm2"][j] if ref else None,
                           "is_reference": rec is ref, "equivalent_radius_nm": rec["equivalent_radius_nm"]})
            w.row("runs.csv", RUN_COLUMNS, row)
            w.jsonl("runs.jsonl", {k: v for k, v in row.items() if k not in ("traceback",)}
                    | {"traceback": rec.get("traceback")})
        print(f"[run] ({pi + 1}/{len(problems)}) {problem['id']}: {len(records)} runs in "
              f"{time.perf_counter() - t_p:.1f}s; reference = {ref_label}", flush=True)

    totals["suite_wall_s"] = time.perf_counter() - t_suite
    (out / "totals.json").write_text(json.dumps(totals, indent=1), encoding="utf-8")
    w.close()
    print(f"[run] done: {json.dumps(totals)}", flush=True)
    if not args.dry_run and not args.no_analyze:
        import analyze

        analyze.main([str(out)])
    return out


if __name__ == "__main__":
    main()
