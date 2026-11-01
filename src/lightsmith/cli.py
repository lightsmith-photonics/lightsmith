"""Command-line interface. Every command prints JSON to stdout; errors go to
stderr as JSON with a non-zero exit code, so agents and scripts can parse both.

    lightsmith doctor
    lightsmith backends
    lightsmith schema
    lightsmith estimate spec.json [--method dda] [--device gpu]
    lightsmith solve spec.json [--method auto] [--device auto] [-o result.json]
    lightsmith compare spec.json [--methods mie,dda]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import api
from .backends import all_backends
from .device import doctor
from .problems import parse_problem, problem_json_schema


def _load(path: str):
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return parse_problem(text)


def _out(obj, path: str | None = None) -> None:
    text = json.dumps(obj, indent=2, default=str)
    if path:
        Path(path).write_text(text, encoding="utf-8")
        print(json.dumps({"written": path}))
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lightsmith", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="report CPU/GPU frameworks and devices")
    sub.add_parser("backends", help="list solver backends and their availability")
    sub.add_parser("schema", help="print the JSON Schema of every problem kind")
    for name in ("estimate", "solve", "compare"):
        s = sub.add_parser(name)
        s.add_argument("spec", help="simulation JSON file, or - for stdin")
        s.add_argument("--device", default="auto", help="auto | cpu | gpu | cuda:N | torch-cpu")
        if name == "compare":
            s.add_argument("--methods", help="comma-separated methods/backends (default: all applicable)")
        else:
            s.add_argument("--method", default="auto", help="auto | <method> (see `backends`) | <backend name>")
        if name != "estimate":
            s.add_argument("-o", "--output", help="write JSON here instead of stdout")
    a = p.parse_args(argv)

    try:
        if a.cmd == "doctor":
            _out(doctor())
        elif a.cmd == "backends":
            _out([b.info() for b in all_backends()])
        elif a.cmd == "schema":
            _out(problem_json_schema())
        elif a.cmd == "estimate":
            _out(api.estimate(_load(a.spec), a.method, a.device))
        elif a.cmd == "solve":
            _out(api.solve(_load(a.spec), a.method, a.device).model_dump(), a.output)
        elif a.cmd == "compare":
            methods = a.methods.split(",") if a.methods else None
            c = api.compare(_load(a.spec), methods, a.device)
            c["results"] = {k: v.model_dump() for k, v in c["results"].items()}
            _out(c, a.output)
    except Exception as e:  # report every failure as structured JSON
        print(json.dumps({"error": type(e).__name__, "message": str(e)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
