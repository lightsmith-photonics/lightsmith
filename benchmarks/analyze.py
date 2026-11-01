"""Post-process a benchmark run: convergence fits, staircase noise, cheapest method per target.

    uv run --no-sync python benchmarks/analyze.py benchmarks/results/<dir>

Reads runs.csv; writes
    convergence.csv  per (problem, backend, variant, device) resolution ladder:
                     fitted order p in err ~ C * res^-p, staircase noise sigma (std of log10
                     residuals about that fit, in decades), fraction of refinements that
                     reduced the error, finest error
    cheapest.csv     per problem and target accuracy: the fastest run whose headline error
                     (max relative error of C_ext and C_abs over wavelengths) meets the target,
                     excluding Mie and copies of the reference run on another device
    summary.md       short human-readable digest
Pure stdlib + numpy (no pandas) so it runs in the project environment.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGETS = (0.10, 0.05, 0.01, 0.001)


def _f(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return math.nan


def load_runs(path: Path) -> list[dict]:
    with open(path / "runs.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("resolution", "wall_s", "peak_mem_bytes", "err_headline", "err_ext_max", "err_abs_max",
                  "err_sca_max", "dipoles", "iterations_total", "q_abs_first", "q_ext_first"):
            r[k] = _f(r.get(k))
        r["is_reference"] = r.get("is_reference") in ("True", "true", "1")
    return rows


def fineness(r):
    """Larger = finer, for both dipoles-per-diameter and mesh sizes; runs without a resolution = 0."""
    res = r["resolution"]
    if math.isnan(res):
        return 0.0
    return -res if r["resolution_quantity"] == "mesh_nm" else res


def convergence(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["status"] != "ok" or math.isnan(r["resolution"]) or r["is_reference"]:
            continue
        groups[(r["problem_id"], r["backend"], r["variant"], r["device"])].append(r)
    out = []
    for (pid, backend, variant, device), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["resolution"])
        if rs[0]["resolution_quantity"] == "mesh_nm":
            rs.reverse()  # coarse -> fine
        for metric in ("err_headline", "err_ext_max", "err_abs_max"):
            res = np.array([r["resolution"] for r in rs])
            err = np.array([r[metric] for r in rs])
            keep = np.isfinite(err) & (err > 0)
            if keep.sum() < 2:
                continue
            res, err = res[keep], err[keep]
            x = np.log10(1.0 / res) if rs[0]["resolution_quantity"] == "mesh_nm" else np.log10(res)
            y = np.log10(err)
            row = {"problem_id": pid, "backend": backend, "variant": variant, "device": device, "metric": metric,
                   "n_resolutions": int(len(res)), "resolutions": " ".join(f"{v:g}" for v in res),
                   "errors": " ".join(f"{v:.3g}" for v in err), "finest_error": float(err[-1]),
                   "min_error": float(err.min()),
                   "monotone_fraction": float(np.mean(np.diff(err) < 0)),
                   "order_p": math.nan, "noise_sigma_dex": math.nan}
            if len(res) >= 3:
                b, a = np.polyfit(x, y, 1)
                row["order_p"] = float(-b)
                row["noise_sigma_dex"] = float(np.std(y - (a + b * x), ddof=2) if len(res) > 2 else 0.0)
            out.append(row)
    return out


METRICS = ("err_headline", "err_ext_max")


def cheapest(rows):
    return [row for m in METRICS for row in _cheapest(rows, m)]


def _cheapest(rows, metric):
    by_p = defaultdict(list)
    for r in rows:
        by_p[r["problem_id"]].append(r)
    out = []
    for pid, rs in by_p.items():
        ref = next((r for r in rs if r["is_reference"]), None)
        # Exclude exact methods (Mie: it *is* the reference where it applies) and copies of the reference run
        # on another device (same backend, variant and resolution), which would trivially "converge".
        same_as_ref = (lambda r: ref is not None and r["backend"] == ref["backend"]
                       and r["variant"] == ref["variant"] and r["resolution"] == ref["resolution"])
        cand = [r for r in rs if r["status"] == "ok" and not r["is_reference"] and np.isfinite(r[metric])
                and r["method"] != "mie" and not same_as_ref(r)]
        for t in TARGETS:
            ok = [r for r in cand if r[metric] <= t]
            row = {"problem_id": pid, "family": rs[0]["family"], "metric": metric, "target": t,
                   "reference_kind": rs[0]["reference_kind"],
                   "reference_backend": ref["backend"] if ref else "",
                   "reference_resolution": ref["resolution"] if ref else math.nan}
            if ok:
                best = min(ok, key=lambda r: r["wall_s"])
                row.update({"backend": best["backend"], "method": best["method"], "variant": best["variant"],
                            "device": best["device"], "resolution": best["resolution"], "wall_s": best["wall_s"],
                            "peak_mem_bytes": best["peak_mem_bytes"], "error": best[metric]})
            else:
                row.update({"backend": "none", "method": "none"})
            out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) and np.isfinite(v) else
                            ("" if isinstance(v, float) else v)) for k, v in r.items()})


def summary(rows, conv, cheap) -> str:
    lines = ["# Benchmark summary", ""]
    n_ok = sum(r["status"] == "ok" for r in rows)
    lines.append(f"{len(rows)} runs ({n_ok} ok). Headline error = max relative error over wavelengths of "
                 "C_ext and C_abs against the reference.")
    lines.append("")
    lines.append("| problem | reference | backend / variant / device | finest res | error | wall s |")
    lines.append("|---|---|---|---|---|---|")
    best = {}
    for r in rows:
        if r["status"] != "ok" or r["is_reference"]:
            continue
        k = (r["problem_id"], r["backend"], r["variant"], r["device"])
        if k not in best or fineness(r) > fineness(best[k]):
            best[k] = r
    refs = {r["problem_id"]: r["reference_label"] for r in rows}
    for k, r in sorted(best.items()):
        lines.append(f"| {k[0]} | {refs[k[0]]} | {k[1]} / {k[2]} / {k[3]} | {r['resolution']:g} | "
                     f"{r['err_headline']:.3g} | {r['wall_s']:.2f} |")
    lines += ["", "## Convergence (headline error)", "",
              "| problem | backend / variant / device | resolutions | order p | noise (dex) | monotone frac |",
              "|---|---|---|---|---|---|"]
    for c in conv:
        if c["metric"] != "err_headline":
            continue
        lines.append(f"| {c['problem_id']} | {c['backend']} / {c['variant']} / {c['device']} | {c['resolutions']} |"
                     f" {c['order_p']:.2f} | {c['noise_sigma_dex']:.2f} | {c['monotone_fraction']:.2f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    rows = load_runs(path)
    conv = convergence(rows)
    cheap = cheapest(rows)
    write_csv(path / "convergence.csv", conv)
    write_csv(path / "cheapest.csv", cheap)
    (path / "summary.md").write_text(summary(rows, conv, cheap), encoding="utf-8")
    print(f"[analyze] wrote convergence.csv ({len(conv)}), cheapest.csv ({len(cheap)}), summary.md in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
