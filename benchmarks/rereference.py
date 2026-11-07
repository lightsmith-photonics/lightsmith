"""Give some families of a benchmark results directory a new reference: the finest run of a later mesh ladder.

    uv run --no-sync python benchmarks/rereference.py <base dir> <ladder.jsonl> <out dir>
    uv run --no-sync python benchmarks/analyze.py <out dir>

The ladder file is written by `benchmarks/remeasure/bem_reference.py`: BEM solves of the spheroids and rods at
several values of elements_per_size. For every problem in it, the ladder's runs are added to the problem's
runs (method bem, resolution = elements_per_size), the finest successful one becomes the reference, its
change from the next coarser rung is recorded as `reference_self_change`, and every other run of the problem
is rescored against it with run.py's `rel_errors`. The old reference stays as an ordinary run. Other problems
are copied unchanged. Ladder runs get variant "ladder", so they can be told apart from the main run's BEM
solves at the default mesh. `reference_uncertainty` in rereference.json is the error of the finest rung
estimated by assuming first-order convergence in the element size h = 1/elements_per_size,
self_change * h_finest / (h_next - h_finest); the ladders converge with order 1 to 2 where monotone. Writes the same files as splice.py, plus rereference.json (what changed).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

from run import Writer, rel_errors
from splice import ERROR_KEYS, _csv, _jsonl, write_rows

REF_COLUMNS = ["problem_id", "family", "reference_kind", "reference_label", "reference_self_change"]


def ladder_record(r: dict, suite_name: str) -> dict:
    r_eq = r["equivalent_radius_nm"]
    area = math.pi * r_eq**2
    return {
        "suite": suite_name, "problem_id": r["problem_id"], "family": r["family"], "backend": "lightsmith-bem",
        "method": "bem", "variant": "ladder", "device": r["device"], "framework": "torch", "precision": "double",
        "resolution_quantity": "elements_per_size", "resolution": float(r["elements_per_size"]),
        "option_field": "elements_per_size", "option_value": r["elements_per_size"], "options":
        {"elements_per_size": r["elements_per_size"], "precision": "double"}, "status": "ok", "message": "",
        "wall_s": r["wall_s"], "solver_runtime_s": r["wall_s"], "peak_mem_bytes": None, "mem_kind": "",
        "est_memory_bytes": r["est_memory_gb"] * 1e9, "unknowns": r["unknowns"], "dipoles": None,
        "iterations_total": None, "iterations_max": None, "wavelengths_nm": r["wavelengths_nm"],
        "c_ext_nm2": r["c_ext_nm2"], "c_sca_nm2": r["c_sca_nm2"], "c_abs_nm2": r["c_abs_nm2"],
        "equivalent_radius_nm": r_eq, "provenance": {"mesh": r.get("mesh"), "source": "bem_reference.jsonl",
                                                     "date": r.get("date")},
        "warnings": r.get("warnings") or [], "n_warnings": len(r.get("warnings") or []),
        "n_wavelengths": len(r["wavelengths_nm"]), "q_ext_first": r["c_ext_nm2"][0] / area,
        "q_abs_first": r["c_abs_nm2"][0] / area, "is_reference": False,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("ladder", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("runs.jsonl", "runs.csv", "spectra.csv", "skips.csv", "references.csv"):
        (args.out / name).unlink(missing_ok=True)

    suite = json.loads((args.base / "suite.json").read_text(encoding="utf-8"))
    pids = [p["id"] for p in suite["problems"]]
    base = _jsonl(args.base / "runs.jsonl")
    ladder = defaultdict(list)
    for r in _jsonl(args.ladder):
        if r["status"] == "ok":
            ladder[r["problem_id"]].append(ladder_record(r, suite["suite"]))

    rows = defaultdict(list)
    for r in base:
        rows[r["problem_id"]].append(r)
    refs = {r["problem_id"]: r for r in base if r["is_reference"]}
    ref_rows = {r["problem_id"]: r for r in _csv(args.base / "references.csv")}
    changed = {}
    for pid, lad in ladder.items():
        lad.sort(key=lambda r: r["resolution"])
        ref = lad[-1]
        change = rel_errors(lad[-2], ref)["err_headline"] if len(lad) > 1 else float("nan")
        h_f, h_c = 1 / ref["resolution"], 1 / lad[-2]["resolution"] if len(lad) > 1 else float("nan")
        uncertainty = change * h_f / (h_c - h_f)
        label = f"finest BEM @ elements_per_size={ref['resolution']:g}, {ref['device']}"
        old_label = ref_rows[pid]["reference_label"]
        ref.update(is_reference=True, reference_kind="highest_resolution", reference_label=label,
                   **{k: None for k in ERROR_KEYS})
        new_rows = []
        for r in rows[pid] + lad:
            if r is not ref:
                r = {**r, "is_reference": False, "reference_kind": "highest_resolution", "reference_label": label,
                     **{k: None for k in ERROR_KEYS}}
                if r["status"] == "ok":
                    r.update(rel_errors(r, ref))
            new_rows.append(r)
        rows[pid] = new_rows
        refs[pid] = ref
        ref_rows[pid] = {**ref_rows[pid], "reference_kind": "highest_resolution", "reference_label": label,
                         "reference_self_change": change}
        changed[pid] = {"old_reference": old_label, "new_reference": label, "self_change": change,
                        "reference_uncertainty": uncertainty, "ladder": [r["resolution"] for r in lad]}

    write_rows(args.out, pids, rows, refs)
    w = Writer(args.out)
    for pid in pids:
        if pid in ref_rows:
            w.row("references.csv", REF_COLUMNS, ref_rows[pid])
    for s in _csv(args.base / "skips.csv"):
        w.row("skips.csv", ["problem_id", "backend", "device", "variant", "resolution", "reason"], s)
    w.close()
    for name in ("suite.json", "environment.json", "environment-fdtd.json", "splice.json"):
        if (args.base / name).exists():
            shutil.copyfile(args.base / name, args.out / name)
    totals = json.loads((args.base / "totals.json").read_text(encoding="utf-8"))
    flat = [r for p in pids for r in rows.get(p, [])]
    totals.update(runs=len(flat), ok=sum(r["status"] == "ok" for r in flat))
    (args.out / "totals.json").write_text(json.dumps(totals, indent=1), encoding="utf-8")
    (args.out / "rereference.json").write_text(json.dumps(
        {"base": str(args.base), "ladder": str(args.ladder), "problems": changed}, indent=1), encoding="utf-8")
    print(f"[rereference] {len(changed)} problems get a BEM reference -> {args.out}")


if __name__ == "__main__":
    main()
