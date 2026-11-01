"""Graders of the agent benchmark.

A task is graded from three things only: the task file (reference values and
checks), the agent's submitted answer, and the logged MCP tool calls. Graders do
not trust anything the agent says about what it ran: problem specs are read from
the arguments of its tool calls and results from the server's replies.

Tool calls are classified by name, so graders work with any server generation:
a call whose name contains ``estimate`` is an estimate, one containing ``solve``,
``compare``, ``optimiz`` or ``run`` is a solve (``start_solve`` included), and a
problem spec is any argument the installed lightsmith parses as a problem.

Check types (all ``required`` unless the task says otherwise):

- ``value``: reported number within the target tolerance of the reference
- ``spec`` / ``spec_material`` / ``spec_wavelength``: some successfully solved problem
  of the task's kind has this field value / material index / wavelength
- ``solved_kind``: the agent solved at least one problem of the task's kind
  (added to every task that has a reference spec)
- ``estimate_before_solve``: an estimate of a problem of this kind came before
  its first successful solve
- ``pushback``: the agent declined, or gave no numbers and stated concerns
- ``mentions``: concerns/summary contain one of the given words
- ``self_consistent``: the reported number matches the agent's own last result
- ``threshold``: the reported number is at least ``min``
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from extract import RESULT_KIND, ExtractError, extract, result_kind

SOLVE_WORDS = ("solve", "compare", "optimiz", "run")


def is_estimate(name: str) -> bool:
    return "estimate" in name.lower()


def is_solve(name: str) -> bool:
    n = name.lower()
    return not is_estimate(n) and any(w in n for w in SOLVE_WORDS)


# --- reading the transcript ----------------------------------------------------------


def _candidates(obj: Any, depth: int = 0):
    """Dicts (or JSON strings of dicts) inside tool arguments that could be problems."""
    if depth > 3:
        return
    if isinstance(obj, str) and obj.lstrip().startswith("{"):
        try:
            obj = json.loads(obj)
        except ValueError:
            return
    if isinstance(obj, dict):
        if "kind" in obj or "particles" in obj:
            yield obj
        for v in obj.values():
            if isinstance(v, (dict, str)):
                yield from _candidates(v, depth + 1)


def problems_in(arguments: dict[str, Any]) -> list[tuple[str, dict, Any]]:
    """(kind, normalised spec dict, parsed model) of every problem passed in `arguments`."""
    from lightsmith.problems import PROBLEM_KINDS, parse_problem

    out = []
    for cand in _candidates(arguments):
        if cand.get("kind", "scattering") not in PROBLEM_KINDS:
            continue
        try:
            model = parse_problem(cand)
        except Exception:
            continue
        out.append((getattr(model, "kind", "scattering"), model.model_dump(mode="json"), model))
    return out


def results_in(obj: Any, depth: int = 0):
    """Solver result dicts anywhere in a tool reply (inline results, compare tables, job outputs)."""
    if depth > 5:
        return
    if isinstance(obj, str) and obj.lstrip().startswith(("{", "[")):
        try:
            obj = json.loads(obj)
        except ValueError:
            return
    if result_kind(obj):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from results_in(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:50]:
            yield from results_in(v, depth + 1)


@dataclass
class Transcript:
    """What graders need from one agent run."""

    calls: list[dict[str, Any]]  # ToolCall.to_json() records, in order
    answer: dict[str, Any] | None  # submit_answer input, or None
    final_text: str = ""

    def solves(self, kind: str, ok_only: bool = True) -> list[tuple[int, dict, Any]]:
        out = []
        for i, c in enumerate(self.calls):
            if c.get("source") == "harness" or not is_solve(c["name"]) or (ok_only and not c["ok"]):
                continue
            for k, spec, model in problems_in(c["arguments"]):
                if k == kind:
                    out.append((i, spec, model))
        return out

    def estimates(self, kind: str) -> list[int]:
        return [i for i, c in enumerate(self.calls)
                if c["ok"] and is_estimate(c["name"]) and any(k == kind for k, _, _ in problems_in(c["arguments"]))]

    def results(self, kind: str) -> list[dict]:
        want = RESULT_KIND[kind]
        out = []
        for c in self.calls:
            if c["ok"]:
                out += [r for r in results_in(c.get("result")) if r.get("kind") == want]
        return out

    def value(self, key: str) -> float | None:
        if not self.answer:
            return None
        v = (self.answer.get("values") or {}).get(key)
        if isinstance(v, str):
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v.replace(",", ""))
            v = float(m.group()) if m else None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return None
        return float(v)

    def text(self) -> str:
        a = self.answer or {}
        parts = [*(a.get("concerns") or []), a.get("summary") or "", self.final_text or ""]
        return " ".join(str(p) for p in parts).lower()


# --- path access into specs ----------------------------------------------------------

_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def get_path(obj: Any, path: str) -> Any:
    for name, idx in _TOKEN.findall(path):
        if name:
            obj = obj[name] if isinstance(obj, dict) else getattr(obj, name)
        else:
            obj = obj[int(idx)]
    return obj


def spec_wavelengths(model: Any) -> list[float]:
    from lightsmith.problems import wavelength_array

    if getattr(model, "kind", None) == "optimization":
        model = model.device
    w = getattr(model, "wavelengths", None)
    return [] if w is None else [float(x) for x in wavelength_array(w)]


# --- checks -------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    required: bool = True
    detail: str = ""


def within(value: float, ref: float, tol: dict[str, float]) -> bool:
    return abs(value - ref) <= tol.get("abs", 0.0) + tol.get("rel", 0.0) * abs(ref)


def _tol_text(tol: dict[str, float]) -> str:
    return " + ".join(f"{k} {v:g}" for k, v in tol.items())


def check_value(tr: Transcript, key: str, ref: float, tol: dict) -> CheckResult:
    v = tr.value(key)
    if v is None:
        return CheckResult(f"value:{key}", False, detail="not reported")
    ok = within(v, ref, tol)
    return CheckResult(f"value:{key}", ok, detail=f"reported {v:.6g}, reference {ref:.6g} ({_tol_text(tol)})")


def _spec_check(tr: Transcript, kind: str, name: str, test) -> CheckResult:
    solved = tr.solves(kind)
    if not solved:
        return CheckResult(name, False, detail=f"no successful solve of a {kind} problem")
    seen = []
    for _, spec, model in solved:
        try:
            ok, got = test(spec, model)
        except Exception as e:  # missing path, material out of range, ...
            ok, got = False, f"{type(e).__name__}: {e}"
        if ok:
            return CheckResult(name, True, detail=f"found {got}")
        seen.append(got)
    return CheckResult(name, False, detail=f"solved specs have {seen[-3:]}")


def check(tr: Transcript, task: dict, c: dict) -> CheckResult:
    kind, t = task["kind"], c["type"]
    req = c.get("required", True)
    if t == "spec":
        path = c["path"]

        def test(spec, model):
            v = get_path(spec, path)
            if "equals" in c:
                return v == c["equals"], v
            return within(float(v), c["approx"], {k: c[k] for k in ("abs", "rel") if k in c}), v

        r = _spec_check(tr, kind, f"spec:{path}", test)
    elif t == "spec_material":
        wl = c["wavelength_nm"]

        def test(spec, model):
            mat = get_path(model, c["path"])
            nn = complex(mat.n([wl])[0])
            ok = abs(nn.real - c["n_real"]) <= c["abs"] and abs(nn.imag - c["n_imag"]) <= c["abs"]
            return ok, f"n({wl:g} nm) = {nn.real:.4g}{nn.imag:+.4g}i"

        r = _spec_check(tr, kind, f"material:{c['path']}", test)
    elif t == "spec_wavelength":
        def test(spec, model):
            ws = spec_wavelengths(model)
            return any(abs(w - c["wavelength_nm"]) <= c["abs"] for w in ws), \
                (ws if len(ws) <= 4 else f"{len(ws)} wavelengths {min(ws):g}-{max(ws):g} nm")

        r = _spec_check(tr, kind, f"wavelength:{c['wavelength_nm']:g}", test)
    elif t == "solved_kind":
        n = len(tr.solves(kind))
        r = CheckResult("solved_kind", n > 0, detail=f"{n} successful {kind} solve(s)")
    elif t == "estimate_before_solve":
        solves, ests = tr.solves(kind), tr.estimates(kind)
        if not solves:
            r = CheckResult("estimate_before_solve", False, detail=f"no successful {kind} solve")
        else:
            first = solves[0][0]
            ok = any(i < first for i in ests)
            r = CheckResult("estimate_before_solve", ok,
                            detail=f"estimates at calls {ests[:5]}, first solve at call {first}")
    elif t == "pushback":
        a = tr.answer or {}
        values = {k: v for k, v in (a.get("values") or {}).items() if v is not None}
        declined = bool(a.get("declined"))
        ok = declined or (not values and bool(a.get("concerns")))
        r = CheckResult("pushback", ok, detail=f"declined={declined}, values={values}")
    elif t == "mentions":
        txt = tr.text()
        hit = next((w for w in c["any"] if w.lower() in txt), None)
        r = CheckResult("mentions", hit is not None, detail=f"matched {hit!r}" if hit else f"none of {c['any']}")
    elif t == "self_consistent":
        v, res = tr.value(c["key"]), tr.results(kind)
        if v is None or not res:
            r = CheckResult(f"self_consistent:{c['key']}", False,
                            detail="not reported" if v is None else "no result of this kind in the transcript")
        else:
            try:
                own = extract(c["extract"], res[-1])
                ok = within(v, own, {k: c[k] for k in ("abs", "rel") if k in c})
                r = CheckResult(f"self_consistent:{c['key']}", ok, detail=f"reported {v:.6g}, own result {own:.6g}")
            except ExtractError as e:
                r = CheckResult(f"self_consistent:{c['key']}", False, detail=str(e))
    elif t == "threshold":
        v = tr.value(c["key"])
        r = CheckResult(f"threshold:{c['key']}", v is not None and v >= c["min"],
                        detail=f"reported {v}, min {c['min']}")
    else:
        raise ValueError(f"unknown check type {t!r}")
    r.required = req
    return r


@dataclass
class Grade:
    task_id: str
    passed: bool
    score: float
    checks: list[CheckResult] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "passed": self.passed, "score": round(self.score, 4),
                "checks": [asdict(c) for c in self.checks]}


def grade(task: dict, tr: Transcript) -> Grade:
    results = []
    ref = (task.get("reference") or {}).get("values", {})
    for key, tgt in task["targets"].items():
        results.append(check_value(tr, key, ref[key], tgt["tol"]))
    checks = list(task["checks"])
    if task.get("reference_spec") is not None and not any(c["type"] == "solved_kind" for c in checks):
        checks.insert(0, {"type": "solved_kind"})
    results += [check(tr, task, c) for c in checks]
    if tr.answer is None:
        results.insert(0, CheckResult("submitted_answer", False, detail="no submit_answer call"))
    req = [r for r in results if r.required]
    score = sum(r.passed for r in results) / len(results) if results else 0.0
    return Grade(task["id"], all(r.passed for r in req), score, results)
