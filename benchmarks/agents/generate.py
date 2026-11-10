"""Recompute the reference results of the agent benchmark and write ``tasks/<id>.json``.

Every task in ``taskdefs.py`` with a reference spec is solved in process with the
installed lightsmith; each target's extractor is applied to the result and the
value stored next to the spec, together with method, backend, device, runtime,
warnings and the library version. Pushback tasks (no reference spec) are written
as they are.

    uv run --no-sync python benchmarks/agents/generate.py            # all tasks
    uv run --no-sync python benchmarks/agents/generate.py --only sca-  # id prefix
    uv run --no-sync python benchmarks/agents/generate.py --check      # recompute, compare, write nothing
    uv run --no-sync python benchmarks/agents/generate.py --keep-reference --only bands-  # checks/prompts only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract import ExtractError, extract  # noqa: E402
from taskdefs import TASKS  # noqa: E402

TASK_DIR = HERE / "tasks"


def full_spec(t: dict) -> dict | None:
    """Reference spec with the reference solver options merged in (what the oracle plays)."""
    spec = t["reference_spec"]
    if spec is None:
        return None
    opts = t.get("reference_options")
    if opts:
        spec = {**spec, "options": {**spec.get("options", {}), **opts}}
    return spec


def oracle_extractors(t: dict) -> dict[str, dict]:
    """Extractor per answer key: targets first, then self-consistency checks, then oracle overrides."""
    ex = {k: v["extract"] for k, v in t["targets"].items()}
    for c in t["checks"]:
        if c["type"] == "self_consistent":
            ex.setdefault(c["key"], c["extract"])
    ex.update(t.get("oracle", {}).get("extract", {}))
    return ex


def solve_reference(t: dict) -> dict:
    import lightsmith as ls

    spec = full_spec(t)
    t0 = time.perf_counter()
    res = ls.solve(spec, t["reference_method"], t["reference_device"])
    runtime = time.perf_counter() - t0
    out = res.model_dump(mode="json")
    values = {}
    for key, tgt in t["targets"].items():
        values[key] = extract(tgt["extract"], out)
    info = {k: out.get(k) for k in ("method", "backend")}
    prov = out.get("provenance", {})
    if t["kind"] == "optimization":
        try:
            values_info = {k: extract(e, out) for k, e in oracle_extractors(t).items()}
        except ExtractError:
            values_info = {}
        info["oracle_values"] = values_info
    return {"values": values, **info, "device": prov.get("device"), "runtime_s": round(runtime, 3),
            "warnings": out.get("warnings", []), "lightsmith_version": ls.__version__}


def task_record(t: dict, reference: dict | None) -> dict:
    rec = {k: v for k, v in t.items() if k not in ("reference_spec", "reference_options")}
    rec["reference_spec"] = full_spec(t)
    rec["reference"] = reference
    rec["oracle_extract"] = oracle_extractors(t)
    return rec


def within(a: float, b: float, tol: dict) -> bool:
    lim = tol.get("abs", 0.0) + tol.get("rel", 0.0) * abs(b)
    return abs(a - b) <= lim


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated id prefixes")
    ap.add_argument("--check", action="store_true", help="recompute and compare with the stored values; write nothing")
    ap.add_argument("--keep-reference", action="store_true",
                    help="rewrite task files from taskdefs.py but keep the stored reference results (no solving; "
                         "for edits to prompts, checks or tolerances)")
    args = ap.parse_args(argv)
    prefixes = [p for p in args.only.split(",") if p]
    TASK_DIR.mkdir(exist_ok=True)
    failures = 0
    for t in TASKS:
        if prefixes and not any(t["id"].startswith(p) for p in prefixes):
            continue
        path = TASK_DIR / f"{t['id']}.json"
        if args.keep_reference and not args.check:
            if not path.exists():
                print(f"{t['id']:36s} no task file to keep the reference of; run without --keep-reference")
                failures += 1
                continue
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("reference_spec") != full_spec(t):
                print(f"{t['id']:36s} reference spec changed; run without --keep-reference")
                failures += 1
                continue
            path.write_text(json.dumps(task_record(t, stored["reference"]), indent=1, ensure_ascii=False) + "\n",
                            encoding="utf-8", newline="\n")
            print(f"{t['id']:36s} {t['kind']:12s} rewritten, reference kept")
            continue
        ref = solve_reference(t) if t["reference_spec"] is not None else None
        line = f"{t['id']:36s} {t['kind']:12s}"
        if ref:
            vals = ", ".join(f"{k}={v:.6g}" for k, v in ref["values"].items())
            line += f" {ref['backend']:22s} {ref['runtime_s']:7.2f} s  {vals}"
            if ref["warnings"]:
                line += f"  [warnings: {len(ref['warnings'])}]"
        print(line, flush=True)
        if args.check:
            old = json.loads(path.read_text(encoding="utf-8"))["reference"] if path.exists() else None
            for k, v in (ref or {}).get("values", {}).items():
                o = (old or {}).get("values", {}).get(k)
                tol = {kk: vv / 10 for kk, vv in t["targets"][k]["tol"].items()}
                if o is None or not within(v, o, tol):
                    failures += 1
                    print(f"    MISMATCH {k}: stored {o}, now {v}")
            continue
        path.write_text(json.dumps(task_record(t, ref), indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                        newline="\n")
    if not args.check and not prefixes:
        index = [{"id": t["id"], "kind": t["kind"], "category": t["category"], "traps": t["traps"]} for t in TASKS]
        (TASK_DIR / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n",
                                             encoding="utf-8", newline="\n")
        stale = {p.stem for p in TASK_DIR.glob("*.json")} - {t["id"] for t in TASKS} - {"index"}
        for s in sorted(stale):
            (TASK_DIR / f"{s}.json").unlink()
            print(f"removed stale task file {s}.json")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
