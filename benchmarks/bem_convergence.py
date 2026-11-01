"""BEM vs DDA vs FDTD: error against Mie for a gold sphere, versus cost.

Gold (Johnson-Christy) sphere in water, on the plasmon resonance and off it,
solved by every method on a ladder of resolutions. Error = max over wavelengths
of the relative error of C_ext and of C_abs against Mie; cost = wall time of
`lightsmith.solve` (after one warm-up solve per method and device) and the
number of unknowns.

    uv run --no-sync python benchmarks/bem_convergence.py --device gpu
    uv run --no-sync python benchmarks/bem_convergence.py --quick            # coarse ladders only
    uv run --no-sync --with matplotlib python benchmarks/bem_convergence.py --plot benchmarks/results/<dir>

Writes benchmarks/results/<timestamp>-bem-convergence/{runs.csv, runs.jsonl, summary.md}
and, with matplotlib installed, figures/bem_convergence.pdf (error vs time and vs
resolution). Timings on a shared GPU/CPU are indicative only.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import lightsmith as ls

AU = ls.LibraryMaterial(name="Au", dataset="Johnson-Christy")
WATER = 1.33
RADIUS = 40.0
WAVELENGTHS = [520.0, 600.0, 700.0, 800.0]  # resonance at ~530 nm; the rest off resonance

LADDERS = {
    # method: (option, values, extra options per variant)
    "bem": ("mesh_size_nm", [16.0, 12.0, 10.0, 8.0, 6.0, 4.0],
            {"default": {}, "no-volume-correction": {"volume_correction": False}}),
    "dda": ("dipoles_per_diameter", [8, 12, 16, 24, 32, 48, 64], {"center": {}}),
    "fdtd": ("spacing_nm", [8.0, 5.0, 4.0, 3.0, 2.0], {"default": {}}),
}
QUICK = {"bem": [16.0, 10.0, 6.0], "dda": [8, 16, 32], "fdtd": [8.0, 4.0]}


def simulation(method: str | None = None, options: dict | None = None) -> ls.Simulation:
    return ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=RADIUS), material=AU)],
                         medium_index=WATER, wavelengths=WAVELENGTHS,
                         options={method: options} if method else {})


def errors(res: ls.Result, ref: ls.Result) -> dict[str, float]:
    out = {}
    for q in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"):
        rel = np.abs(np.asarray(getattr(res, q)) / np.asarray(getattr(ref, q)) - 1)
        out[f"err_{q[:5]}_max"] = float(rel.max())
        out[f"err_{q[:5]}_per_wl"] = rel.tolist()
    out["err_headline"] = max(out["err_c_ext_max"], out["err_c_abs_max"])
    return out


def size_of(method: str, res: ls.Result) -> int | None:
    p = res.provenance
    if method == "bem":
        return p.get("unknowns")
    if method == "dda":
        d = p.get("dipoles")
        return d[-1] if isinstance(d, list) else d
    if method == "fdtd":
        g = p.get("grid") or p.get("grid_shape")
        return int(np.prod(g)) if g else p.get("cells")
    return None


def run(args) -> Path:
    out = Path(args.out or f"benchmarks/results/{datetime.now():%Y%m%d-%H%M%S}-bem-convergence")
    out.mkdir(parents=True, exist_ok=True)
    ref = ls.solve(simulation(), "lightsmith-mie", "cpu")
    env = {"command": " ".join(sys.argv), "python": platform.python_version(), "platform": platform.platform(),
           "doctor": ls.doctor(), "lightsmith": ls.__version__, "problem": json.loads(simulation().to_json()),
           "reference": {"backend": ref.backend, "c_ext_nm2": ref.c_ext_nm2, "c_abs_nm2": ref.c_abs_nm2}}
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    rows = []
    for method in args.methods:
        option, values, variants = LADDERS[method]
        if args.quick:
            values = QUICK[method]
        device = args.device if ls.get_backend(method, "scattering").gpu else "cpu"
        ls.solve(simulation(method, {option: values[0], **next(iter(variants.values()))}), method, device)  # warm-up
        for variant, extra in variants.items():
            for v in values:
                s = simulation(method, {option: v, **extra})
                est = ls.estimate(s, method, device)
                if est.get("memory_bytes", 0) > args.max_mem_gb * 1e9:
                    print(f"skip {method}/{variant} {option}={v}: estimate {est['memory_bytes'] / 1e9:.1f} GB")
                    continue
                t = time.perf_counter()
                try:
                    res = ls.solve(s, method, device)
                except Exception as e:  # record and continue the ladder
                    print(f"{method}/{variant} {option}={v}: failed: {e}")
                    continue
                wall = time.perf_counter() - t
                row = {"method": method, "backend": res.backend, "variant": variant, "device": device,
                       "resolution_option": option, "resolution": v, "size": size_of(method, res), "wall_s": wall,
                       **errors(res, ref), "warnings": res.warnings}
                rows.append(row)
                print(f"{method:5s} {variant:22s} {option}={v:<6g} size={row['size']}  {wall:7.1f} s  "
                      f"C_ext err {100 * row['err_c_ext_max']:.3f} %  C_abs err {100 * row['err_c_abs_max']:.3f} %",
                      flush=True)
                with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({**row, "c_ext_nm2": res.c_ext_nm2, "c_abs_nm2": res.c_abs_nm2,
                                        "provenance": res.provenance}) + "\n")
    keys = ["method", "backend", "variant", "device", "resolution_option", "resolution", "size", "wall_s",
            "err_c_ext_max", "err_c_sca_max", "err_c_abs_max", "err_headline"]
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary(rows, out)
    return out


def summary(rows: list[dict], out: Path) -> None:
    lines = [f"# BEM convergence: Au (JC) sphere r = {RADIUS:g} nm in water, wavelengths {WAVELENGTHS} nm", "",
             "Error = max over wavelengths of |C/C_Mie - 1|.", "",
             "| method | variant | resolution | size | wall (s) | C_ext err | C_abs err |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['variant']} | {r['resolution_option']}={r['resolution']:g} | {r['size']} | "
                     f"{r['wall_s']:.1f} | {100 * r['err_c_ext_max']:.3g} % | {100 * r['err_c_abs_max']:.3g} % |")
    lines += ["", "Cheapest run per target (headline error = max of C_ext and C_abs errors):", ""]
    for target in (0.1, 0.05, 0.01, 0.001):
        best = {}
        for r in rows:
            if r["err_headline"] <= target and (r["method"] not in best or r["wall_s"] < best[r["method"]]["wall_s"]):
                best[r["method"]] = r
        cells = ", ".join(f"{m}: {b['wall_s']:.1f} s ({b['resolution_option']}={b['resolution']:g})"
                          for m, b in sorted(best.items())) or "none"
        lines.append(f"- {100 * target:g} %: {cells}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot(out: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    style = {"bem": "o-", "dda": "s-", "fdtd": "^-"}
    for (method, variant) in dict.fromkeys((r["method"], r["variant"]) for r in rows):
        rs = [r for r in rows if r["method"] == method and r["variant"] == variant]
        label = method.upper() + ("" if variant in ("default", "center") else f" ({variant})")
        fmt = style[method] if variant in ("default", "center") else style[method].replace("-", "--")
        for ax, q in zip(axes, ("err_c_ext_max", "err_c_abs_max")):
            ax.loglog([r["wall_s"] for r in rs], [max(r[q], 1e-6) for r in rs], fmt, label=label, ms=4)
    for ax, title in zip(axes, ("extinction", "absorption")):
        ax.set_xlabel("wall time (s)")
        ax.set_ylabel(f"max relative error, {title}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8, loc="center left")
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "bem_convergence.pdf")
    fig.savefig(out / "figures" / "bem_convergence.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="gpu", help="cpu | gpu | cuda:N | torch-cpu (CPU-only backends ignore it)")
    ap.add_argument("--methods", default="bem,dda,fdtd", type=lambda s: s.split(","))
    ap.add_argument("--quick", action="store_true", help="three resolutions per method")
    ap.add_argument("--max-mem-gb", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None, metavar="DIR", help="only plot an existing results directory")
    args = ap.parse_args()
    if args.plot:
        plot(Path(args.plot))
        return
    if args.device == "gpu":
        try:
            ls.resolve_device("gpu")
        except RuntimeError as e:
            print(f"{e}; using cpu")
            args.device = "cpu"
    out = run(args)
    try:
        plot(out)
    except ImportError:
        print("matplotlib not installed; run with --plot to draw the figure")


if __name__ == "__main__":
    main()
