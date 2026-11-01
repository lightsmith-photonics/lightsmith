"""Figures for a benchmark run.

    uv run --no-sync --with matplotlib python benchmarks/plots.py benchmarks/results/<dir> [--paper]

Writes <dir>/figures/*.pdf and *.png; --paper also copies the PDFs to paper/figures/.
Figures:
    spectra.pdf         Q_ext and Q_abs spectra: reference vs each backend at its finest resolution
    error_vs_cost.pdf   headline error vs wall time, one panel per problem, lines join resolution ladders
    convergence.pdf     headline error vs resolution (DDA dipoles per diameter, FDTD cells per nm)
    staircase.pdf       Q_abs vs dipoles per diameter for the Drude sphere (or any 1-wavelength sphere)
    regime_map.pdf      cheapest non-reference run meeting each target accuracy, per problem
    family_maps.pdf     the same at one target, on each family's own parameter axes (needs >= 2 problems)
    gpu_speedup.pdf     CPU/GPU wall time for matching runs, and measured vs estimated memory
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import TARGETS, load_runs  # noqa: E402

METHOD_COLORS = {"mie": "#000000", "quasistatic": "#7f7f7f", "dda": "#1f77b4", "fdtd": "#ff7f0e",
                 "tmatrix": "#2ca02c", "none": "#ffffff"}
EXTRA_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
VARIANT_STYLE = {"center": "-", "weighted": "--", "extrapolate": ":", "default": "-"}
DEVICE_MARKER = {"cpu": "o", "gpu": "s"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "figure.dpi": 150,
                     "savefig.bbox": "tight"})


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def color(method):
    if method not in METHOD_COLORS:
        METHOD_COLORS[method] = EXTRA_COLORS[len(METHOD_COLORS) % len(EXTRA_COLORS)]
    return METHOD_COLORS[method]


def marker(device):
    return DEVICE_MARKER.get(device, "^")


def label(r):
    v = "" if r["variant"] in ("default", "") else f"/{r['variant']}"
    return f"{r['backend'].replace('lightsmith-', '')}{v}/{r['device']}"


def load_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _grid(n, w=3.4, h=2.9, ncols=4):
    ncols = min(ncols, max(n, 1))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"[plots] {name}")


def spectra(rows, spec, out):
    by_run = defaultdict(list)
    for s in spec:
        by_run[(s["problem_id"], int(s["run_index"]))].append(s)
    problems = sorted({s["problem_id"] for s in spec if
                       len({x["wavelength_nm"] for x in spec if x["problem_id"] == s["problem_id"]}) > 1})
    if not problems:
        return
    runs_by_p = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            runs_by_p[r["problem_id"]].append(r)
    fig, axes = _grid(len(problems))
    for ax, pid in zip(axes, problems):
        rs = runs_by_p[pid]
        # run_index is the position within the problem's runs as written by run.py
        idx = {i: r for i, r in enumerate([r for r in rows if r["problem_id"] == pid])}
        finest = {}
        for i, r in idx.items():
            if r["status"] != "ok":
                continue
            k = (r["backend"], r["variant"], r["device"])
            res = r["resolution"]
            fine = (-res if r["resolution_quantity"] == "mesh_nm" else res) if not math.isnan(res) else 0
            if k not in finest or fine > finest[k][0]:
                finest[k] = (fine, i)
        ref_drawn = False
        for k, (_, i) in sorted(finest.items()):
            pts = sorted((p for p in by_run.get((pid, i), []) if p["c_ext_nm2"] not in ("", "nan")),
                         key=lambda s: float(s["wavelength_nm"]))  # NaN rows: solver breakdown
            if not pts:
                continue
            wl = np.array([float(p["wavelength_nm"]) for p in pts])
            geo = math.pi * float(pts[0]["equivalent_radius_nm"]) ** 2
            r = idx[i]
            if not ref_drawn:
                ax.plot(wl, [float(p["ref_c_ext_nm2"]) / geo for p in pts], "k-", lw=1.2, label="ref Q_ext")
                ax.plot(wl, [float(p["ref_c_abs_nm2"]) / geo for p in pts], "k:", lw=1.2, label="ref Q_abs")
                ref_drawn = True
            if r["is_reference"]:
                continue
            ax.plot(wl, [float(p["c_ext_nm2"]) / geo for p in pts], marker(r["device"]), ls="none", ms=3,
                    mfc="none" if r["variant"] == "weighted" else color(r["method"]), color=color(r["method"]),
                    label=label(r))
        ax.set_title(pid, fontsize=6.5)
        ax.set_xlabel("vacuum wavelength (nm)")
        ax.set_ylabel("Q")
        ax.legend(loc="best", fontsize=5.5)
    fig.tight_layout()
    save(fig, out, "spectra")


def _ladders(rows):
    lad = defaultdict(list)
    for r in rows:
        if r["status"] == "ok" and not r["is_reference"] and np.isfinite(r["err_headline"]):
            lad[(r["problem_id"], r["backend"], r["method"], r["variant"], r["device"])].append(r)
    return lad


def error_vs_cost(rows, out):
    lad = _ladders(rows)
    problems = sorted({k[0] for k in lad})
    if not problems:
        return
    fig, axes = _grid(len(problems))
    for ax, pid in zip(axes, problems):
        for k, rs in sorted(lad.items()):
            if k[0] != pid:
                continue
            if k[2] == "mie":  # exact; would squash the axis to 1e-12
                continue
            fine = lambda r: (-r["resolution"] if r["resolution_quantity"] == "mesh_nm" else r["resolution"])  # noqa: E731
            rs = sorted(rs, key=lambda r: fine(r) if np.isfinite(r["resolution"]) else 0)
            ax.loglog([r["wall_s"] for r in rs], [max(r["err_headline"], 1e-4) for r in rs], marker=marker(k[4]),
                      ls=VARIANT_STYLE.get(k[3], "-"), color=color(k[2]), ms=3, lw=0.8, label=label(rs[0]))
        for t in (0.01, 0.1):
            ax.axhline(t, color="0.8", lw=0.5, zorder=0)
        ax.set_title(pid, fontsize=6.5)
        ax.set_xlabel("wall time (s)")
        ax.set_ylabel("max rel. error (ext, abs)")
        ax.legend(loc="best", fontsize=5.5)
    fig.tight_layout()
    save(fig, out, "error_vs_cost")


def convergence(rows, out):
    lad = _ladders(rows)
    problems = sorted({k[0] for k in lad})
    if not problems:
        return
    fig, axes = _grid(len(problems))
    for ax, pid in zip(axes, problems):
        any_line = False
        for k, rs in sorted(lad.items()):
            if k[0] != pid or len(rs) < 2:
                continue
            q = rs[0]["resolution_quantity"]
            x = np.array([r["resolution"] for r in rs])
            if q == "mesh_nm":
                x = 1.0 / x  # cells per nm, so "right = finer" for both methods
            o = np.argsort(x)
            ax.loglog(x[o], np.array([r["err_headline"] for r in rs])[o], marker=marker(k[4]),
                      ls=VARIANT_STYLE.get(k[3], "-"), color=color(k[2]), ms=3, lw=0.8, label=label(rs[0]))
            any_line = True
        ax.set_title(pid, fontsize=6.5)
        ax.set_xlabel("resolution (DDA: dipoles/diam.; FDTD: cells/nm)")
        ax.set_ylabel("max rel. error (ext, abs)")
        if any_line:
            ax.legend(loc="best", fontsize=5.5)
    fig.tight_layout()
    save(fig, out, "convergence")


def staircase(rows, out):
    """Q_abs vs dipoles per diameter for single-wavelength sphere problems (the staircase study)."""
    pids = sorted({r["problem_id"] for r in rows if r["status"] == "ok" and r.get("n_wavelengths") == "1"
                   and r["family"] in ("metal_sphere", "dielectric_sphere")})
    for pid in pids:
        rs = [r for r in rows if r["problem_id"] == pid and r["status"] == "ok"]
        ref = next((r for r in rs if r["is_reference"]), None)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [2, 1]})
        groups = defaultdict(list)
        for r in rs:
            if r["method"] == "dda":
                groups[(r["variant"], r["device"])].append(r)
        for (v, d), g in sorted(groups.items()):
            g.sort(key=lambda r: r["resolution"])
            a1.plot([r["resolution"] for r in g], [r["q_abs_first"] for r in g], marker=marker(d),
                    ls=VARIANT_STYLE.get(v, "-"), color={"center": "#1f77b4", "weighted": "#d62728",
                                                         "extrapolate": "#2ca02c"}.get(v, "k"),
                    ms=3, lw=0.8, label=f"DDA {v} ({d})")
        if ref is not None:
            a1.axhline(ref["q_abs_first"], color="k", lw=1, label=f"reference ({ref['backend']})")
            a2.axhline(ref["q_abs_first"], color="k", lw=1)
        core = [r["q_abs_first"] for r in rs if r["method"] == "dda" and r["variant"] != "extrapolate"
                and np.isfinite(r["q_abs_first"])]
        if core:
            top = 1.25 * max(core + [ref["q_abs_first"] if ref else 0])
            a1.set_ylim(min(0, min(core)) - 0.05 * top, top)
            for (v, d), g in groups.items():  # mark off-scale points (e.g. extrapolation) with arrows
                for r in g:
                    q = r["q_abs_first"]
                    if np.isfinite(q) and (q > top or q < a1.get_ylim()[0]):
                        y = top * 0.97 if q > top else a1.get_ylim()[0] + 0.03 * top
                        a1.annotate(f"{q:.2g}", (r["resolution"], y), fontsize=6, ha="center",
                                    va="top" if q > top else "bottom", color="#2ca02c")
        a1.set_xlabel("dipoles per diameter")
        a1.set_ylabel(r"$Q_\mathrm{abs}$")
        a1.set_title(pid, fontsize=7)
        a1.legend(loc="best")
        f = sorted([r for r in rs if r["method"] == "fdtd"], key=lambda r: -r["resolution"])
        for d in sorted({r["device"] for r in f}):
            fd = [r for r in f if r["device"] == d]
            a2.plot([r["resolution"] for r in fd], [r["q_abs_first"] for r in fd], marker=marker(d), color=color("fdtd"),
                    ms=3, lw=0.8, label=f"FDTD ({d})")
        others = [r for r in rs if r["method"] not in ("dda", "fdtd", "mie") and not r["is_reference"]]
        for r in others:
            a2.axhline(r["q_abs_first"], color=color(r["method"]), lw=0.6, ls="--", label=label(r))
        a2.set_xscale("log")
        if f:
            ticks = sorted({r["resolution"] for r in f})
            a2.set_xticks(ticks, [f"{t:g}" for t in ticks])
            a2.minorticks_off()
        a2.invert_xaxis()
        a2.set_xlabel("FDTD mesh (nm)")
        a2.legend(loc="best")
        save(fig, out, f"staircase_{pid}")


def regime_map(cheap_all, out, metric="err_headline", name="regime_map"):
    cheap = [c for c in cheap_all if c.get("metric", "err_headline") == metric]
    if not cheap:
        return
    pids = list(dict.fromkeys(c["problem_id"] for c in cheap))
    targets = sorted({float(c["target"]) for c in cheap}, reverse=True)
    fig, ax = plt.subplots(figsize=(1.5 * len(targets) + 2.4, 0.42 * len(pids) + 0.9))
    for i, pid in enumerate(pids):
        for j, t in enumerate(targets):
            c = next(c for c in cheap if c["problem_id"] == pid and float(c["target"]) == t)
            m = c.get("method", "none") or "none"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color(m), alpha=0.35 if m != "none" else 1, lw=0))
            if m == "none":
                txt = "none"
            else:
                res = c.get("resolution", "")
                res = f"@{float(res):g}" if res not in ("", None) else ""
                txt = f"{c['backend'].replace('lightsmith-', '')}{res}\n{c['device']} {float(c['wall_s']):.2g}s"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=5.5)
    ax.set_xlim(0, len(targets))
    ax.set_ylim(len(pids), 0)
    ax.set_xticks(np.arange(len(targets)) + 0.5, [f"{t:g}" for t in targets])
    def ylab(pid):
        c = next(c for c in cheap if c["problem_id"] == pid)
        if c.get("reference_kind") == "mie":
            return pid
        res = c.get("reference_resolution", "")
        res = f"@{float(res):g}" if res not in ("", None, "nan") else ""
        backend = c.get("reference_backend", "").replace("lightsmith-", "")
        return pid + "\n" + f"(ref: {backend}{res})"

    ax.set_yticks(np.arange(len(pids)) + 0.5, [ylab(p) for p in pids], fontsize=6)
    what = "C_ext and C_abs" if metric == "err_headline" else "C_ext"
    ax.set_xlabel(f"target max relative error of {what}; cheapest run by wall time (Mie excluded)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color(m), alpha=0.35) for m in METHOD_COLORS if m != "none"]
    ax.legend(handles, [m for m in METHOD_COLORS if m != "none"], loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, out, name)


def family_maps(cheap, suite, out, target=0.05):
    params = {p["id"]: p for p in suite.get("problems", [])}
    fams = defaultdict(list)
    for c in cheap:
        if float(c["target"]) == target and c["problem_id"] in params and c.get("metric", "err_headline") == "err_headline":
            fams[c["family"]].append(c)
    fams = {f: cs for f, cs in fams.items() if len(cs) >= 2 and params[cs[0]["problem_id"]].get("map_axes")}
    if not fams:
        return
    fig, axes = _grid(len(fams), w=3.2, h=2.6, ncols=3)
    for ax, (fam, cs) in zip(axes, sorted(fams.items())):
        xk, yk = params[cs[0]["problem_id"]]["map_axes"]
        for c in cs:
            p = params[c["problem_id"]]["params"]
            x, y = p.get(xk), p.get(yk)
            m = c.get("method") or "none"
            ax.scatter([x], [y], s=60, color=color(m), edgecolor="k", lw=0.4, alpha=0.8, label=m)
        h, lab = ax.get_legend_handles_labels()
        uniq = dict(zip(lab, h))
        ax.legend(uniq.values(), uniq.keys(), loc="best")
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title(f"{fam}: cheapest method at {target:g} error", fontsize=7)
    save(fig, out, "family_maps")


def gpu_speedup(rows, out):
    ok = [r for r in rows if r["status"] == "ok" and r["method"] in ("dda", "fdtd")]
    key = lambda r: (r["problem_id"], r["backend"], r["variant"], r["resolution"])  # noqa: E731
    cpu = {key(r): r for r in ok if r["device"] == "cpu"}
    pairs = [(cpu[key(r)], r) for r in ok if r["device"] != "cpu" and key(r) in cpu]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for c, g in pairs:
        n = g["dipoles"] if np.isfinite(g["dipoles"]) else _f(g.get("unknowns"))
        a1.scatter(n, c["wall_s"] / g["wall_s"], color=color(g["method"]), s=10,
                   marker="o" if g["variant"] != "weighted" else "^")
    a1.axhline(1, color="0.7", lw=0.5)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("unknowns (DDA dipoles, FDTD cells)")
    a1.set_ylabel("CPU time / GPU time")
    a1.set_title("GPU speed-up (matching runs)")
    pts = []
    for r in ok:
        e = _f(r.get("est_memory_bytes"))
        if np.isfinite(e) and e > 0 and np.isfinite(r["peak_mem_bytes"]) and r["peak_mem_bytes"] > 0:
            pts.append((e / 2**20, r["peak_mem_bytes"] / 2**20, r))
    for e, m, r in pts:
        a2.scatter(e, m, color=color(r["method"]), s=8, marker=marker(r["device"]))
    if pts:
        lo = min(min(p[0], p[1]) for p in pts) / 2
        hi = max(max(p[0], p[1]) for p in pts) * 2
        a2.plot([lo, hi], [lo, hi], color="0.7", lw=0.5)
        a2.set_xlim(lo, hi)
        a2.set_ylim(lo, hi)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("estimate() memory (MiB)")
    a2.set_ylabel("measured peak (MiB)")
    a2.set_title("estimate vs measured (o CPU tracemalloc, s GPU)")
    save(fig, out, "gpu_speedup")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--paper", action="store_true", help="copy PDFs to paper/figures/")
    args = ap.parse_args(argv)
    d = args.results
    rows = load_runs(d)
    jl = d / "runs.jsonl"
    if rows and "unknowns" not in rows[0] and jl.exists():  # older runs.csv: recover sizes from estimates
        recs = [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines() if line.strip()]
        for r, j in zip(rows, recs):
            e = j.get("estimate") or {}
            r["unknowns"] = j.get("dipoles") or e.get("dipoles") or e.get("cells")
    spec = load_csv(d / "spectra.csv")
    cheap = load_csv(d / "cheapest.csv")
    suite = json.loads((d / "suite.json").read_text(encoding="utf-8")) if (d / "suite.json").exists() else {}
    out = d / "figures"
    out.mkdir(exist_ok=True)
    for fn in (lambda: spectra(rows, spec, out), lambda: error_vs_cost(rows, out), lambda: convergence(rows, out),
               lambda: staircase(rows, out), lambda: regime_map(cheap, out),
               lambda: regime_map(cheap, out, "err_ext_max", "regime_map_ext"),
               lambda: family_maps(cheap, suite, out), lambda: gpu_speedup(rows, out)):
        try:
            fn()
        except Exception as e:  # noqa: BLE001  keep going: one bad figure should not lose the others
            import traceback

            traceback.print_exc()
            print(f"[plots] figure failed: {e}", file=sys.stderr)
    if args.paper:
        dest = HERE.parent / "paper" / "figures"
        dest.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.pdf"):
            shutil.copy2(f, dest / f.name)
        (dest / "SOURCE.txt").write_text(f"Placeholder figures copied from {d.resolve()}\n", encoding="utf-8")
        print(f"[plots] copied PDFs to {dest}")


if __name__ == "__main__":
    main()
