"""Replace one method's runs in a benchmark results directory with those of a later run of that method alone.

    uv run --no-sync python benchmarks/splice.py <base dir> <new dir> <out dir> --method fdtd
    uv run --no-sync python benchmarks/analyze.py <out dir>

For repeating one solver's part of a long suite after a change to that solver only. The new runs keep the base
run's references (which must come from other methods) and get their errors recomputed against them, exactly as
run.py computes them. Writes runs.jsonl, runs.csv, spectra.csv, skips.csv, references.csv, suite.json,
totals.json, the base and new environment files, and splice.json (what came from where).
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from run import RUN_COLUMNS, Writer, rel_errors

SPECTRA_COLUMNS = ["problem_id", "run_index", "backend", "variant", "device", "resolution", "wavelength_nm",
                   "c_ext_nm2", "c_sca_nm2", "c_abs_nm2", "ref_c_ext_nm2", "ref_c_sca_nm2", "ref_c_abs_nm2",
                   "is_reference", "equivalent_radius_nm"]
SKIP_COLUMNS = ["problem_id", "backend", "device", "variant", "resolution", "reason"]
ERROR_KEYS = ("err_ext_max", "err_ext_rms", "err_sca_max", "err_sca_rms", "err_abs_max", "err_abs_rms", "err_headline")


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _csv(path: Path) -> list[dict]:
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(out: Path, problem_ids: list[str], rows_by_problem: dict[str, list[dict]], refs: dict[str, dict]):
    """runs.jsonl, runs.csv and spectra.csv as run.py writes them (run_index: position within the problem)."""
    w = Writer(out)
    for pid in problem_ids:
        ref = refs.get(pid)
        for i, row in enumerate(rows_by_problem.get(pid, [])):
            if row["status"] == "ok":
                for j, wl in enumerate(row["wavelengths_nm"]):
                    w.row("spectra.csv", SPECTRA_COLUMNS,
                          {"problem_id": pid, "run_index": i, "backend": row["backend"], "variant": row["variant"],
                           "device": row["device"], "resolution": row["resolution"], "wavelength_nm": wl,
                           "c_ext_nm2": row["c_ext_nm2"][j], "c_sca_nm2": row["c_sca_nm2"][j],
                           "c_abs_nm2": row["c_abs_nm2"][j],
                           "ref_c_ext_nm2": ref["c_ext_nm2"][j] if ref else None,
                           "ref_c_sca_nm2": ref["c_sca_nm2"][j] if ref else None,
                           "ref_c_abs_nm2": ref["c_abs_nm2"][j] if ref else None,
                           "is_reference": row["is_reference"], "equivalent_radius_nm": row["equivalent_radius_nm"]})
            w.row("runs.csv", RUN_COLUMNS, row)
            w.jsonl("runs.jsonl", row)
    w.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--method", required=True, help="the method whose runs `new` replaces, e.g. fdtd")
    args = ap.parse_args(argv)
    m = args.method
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("runs.jsonl", "runs.csv", "spectra.csv", "skips.csv"):
        (args.out / name).unlink(missing_ok=True)

    suite = json.loads((args.base / "suite.json").read_text(encoding="utf-8"))
    pids = [p["id"] for p in suite["problems"]]
    base, new = _jsonl(args.base / "runs.jsonl"), _jsonl(args.new / "runs.jsonl")
    refs = {r["problem_id"]: r for r in base if r["is_reference"]}
    bad = [p for p, r in refs.items() if r["method"] == m]
    if bad:
        raise SystemExit(f"base references come from {m} for {bad}: the new runs cannot be scored against them")
    ref_info = {r["problem_id"]: (r["reference_kind"], r["reference_label"]) for r in base}
    backends = {r["backend"] for r in base + new if r["method"] == m}

    rows = defaultdict(list)
    for r in base:
        if r["method"] != m:
            rows[r["problem_id"]].append(r)
    n_new = 0
    for r in new:
        if r["method"] != m or r["problem_id"] not in ref_info:
            continue
        r = {**r, "is_reference": False, **{k: None for k in ERROR_KEYS}}
        r["reference_kind"], r["reference_label"] = ref_info[r["problem_id"]]
        ref = refs.get(r["problem_id"])
        if r["status"] == "ok" and ref is not None:
            r.update(rel_errors(r, ref))
        rows[r["problem_id"]].append(r)
        n_new += 1
    write_rows(args.out, pids, rows, refs)

    skips = [s for s in _csv(args.base / "skips.csv") if s["backend"] not in backends]
    skips += [s for s in _csv(args.new / "skips.csv") if s["backend"] in backends]
    w = Writer(args.out)
    for s in skips:
        w.row("skips.csv", SKIP_COLUMNS, s)
    w.close()
    for name in ("references.csv", "suite.json", "environment.json"):
        shutil.copyfile(args.base / name, args.out / name)
    shutil.copyfile(args.new / "environment.json", args.out / f"environment-{m}.json")

    flat = [r for p in pids for r in rows.get(p, [])]
    t_base, t_new = (json.loads((d / "totals.json").read_text(encoding="utf-8")) for d in (args.base, args.new))
    wall_base_m = sum(r["wall_s"] or 0 for r in base if r["method"] == m)
    totals = {"runs": len(flat), "ok": sum(r["status"] == "ok" for r in flat),
              "error": sum(r["status"] == "error" for r in flat), "skipped": len(skips),
              # the base run's wall time without its own runs of the method, plus the whole new run
              "suite_wall_s": t_base["suite_wall_s"] - wall_base_m + t_new["suite_wall_s"]}
    (args.out / "totals.json").write_text(json.dumps(totals, indent=1), encoding="utf-8")
    (args.out / "splice.json").write_text(json.dumps(
        {"base": str(args.base), "new": str(args.new), "method": m, "backends": sorted(backends),
         "runs_replaced": sum(r["method"] == m for r in base), "runs_added": n_new,
         "base_wall_s_of_method": wall_base_m, "new_suite_wall_s": t_new["suite_wall_s"]}, indent=1), encoding="utf-8")
    print(f"[splice] {m}: {totals['runs']} runs ({n_new} new), {len(skips)} skips -> {args.out}")


if __name__ == "__main__":
    main()
