"""Tests of the agent benchmark: task files, graders, and the harness end to end over MCP.

The oracle plays every reference spec through a real stdio MCP server and must
pass every task; deliberately wrong behaviour (wrong numbers, careless specs,
skipped estimates, missing pushback, unreported warnings, no answer) must fail.

    uv run --no-sync pytest benchmarks/agents/tests

The server is launched with ``legacy_server.py`` (runs whichever
``lightsmith.mcp_server`` is installed, on MCP SDK 1.x or 2.x); set
AGENTBENCH_SERVER_CMD to test another command, e.g. ``lightsmith-mcp``.
AGENTBENCH_TASKS (comma-separated id prefixes) restricts the end-to-end runs.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from pathlib import Path

import pytest

import run as harness
from agents import OracleAgent
from extract import ExtractError, extract
from grading import Transcript, grade, problems_in, results_in
from taskdefs import TASKS

HERE = Path(__file__).resolve().parents[1]
SERVER_CMD = os.environ.get("AGENTBENCH_SERVER_CMD") or f'"{sys.executable}" "{HERE / "legacy_server.py"}"'
ONLY = os.environ.get("AGENTBENCH_TASKS", "")
FILES = {t["id"]: t for t in harness.load_tasks()}


# --- task files ------------------------------------------------------------------------


def test_task_files_match_definitions():
    assert [t["id"] for t in TASKS] == list(FILES), "run generate.py after editing taskdefs.py"
    assert 30 <= len(TASKS) <= 50
    kinds = {t["kind"] for t in TASKS}
    assert kinds == {"scattering", "layer_stack", "periodic", "mode", "device", "bands", "circuit", "optimization"}
    assert {t["category"] for t in TASKS} == {"standard", "trap", "pushback", "warning"}
    for t in TASKS:
        f = FILES[t["id"]]
        assert f["prompt"] == t["prompt"] and f["checks"] == t["checks"]
        keys = {a["key"] for a in t["answer_fields"]}
        assert set(t["targets"]) <= keys
        if t["reference_spec"] is None:
            assert t["category"] == "pushback" and f["reference"] is None
            continue
        assert set(f["reference"]["values"]) == set(t["targets"])
        assert set(f["oracle_extract"]) == keys or t["category"] == "warning", t["id"]


def test_prompts_do_not_leak_reference_specs():
    for t in TASKS:
        assert "lightsmith" not in t["prompt"].lower()
        assert '"kind"' not in t["prompt"] and "_nm" not in t["prompt"]


# --- graders on hand-made transcripts ---------------------------------------------------

SPHERE = {"particles": [{"shape": {"kind": "sphere", "radius_nm": 40.0},
                         "material": {"kind": "library", "name": "Au"}}], "medium_index": 1.33, "wavelengths": [532.0]}


def _result(c_abs=18256.9):
    return {"kind": "cross_sections", "method": "mie", "backend": "miepython", "wavelengths_nm": [532.0],
            "c_ext_nm2": [2e4], "c_sca_nm2": [2e4 - c_abs], "c_abs_nm2": [c_abs], "equivalent_radius_nm": 40.0}


def _call(name, args, result=None, ok=True, source="mcp"):
    return {"name": name, "arguments": args, "ok": ok, "result": result, "text": "", "source": source}


def _answer(**values):
    return {"values": values, "declined": False, "concerns": [], "summary": ""}


def test_problem_detection_with_old_and_new_argument_names():
    for args in ({"simulation": SPHERE}, {"problem": SPHERE}, {"problem": json.dumps(SPHERE)},
                 {"problem": {**SPHERE, "kind": "scattering"}, "method": "mie"}):
        found = problems_in(args)
        assert len(found) == 1 and found[0][0] == "scattering"
    assert problems_in({"job_id": "abc"}) == []
    assert problems_in({"problem": {"kind": "scattering", "particles": []}}) == []  # invalid spec


def test_results_found_inline_and_inside_job_outputs():
    r = _result()
    assert list(results_in(r)) == [r]
    assert list(results_in({"job_id": "x", "status": "succeeded", "result": r})) == [r]
    assert list(results_in(json.dumps({"results": {"mie": r}}))) == [r]
    assert list(results_in({"summary": {"peak": 1}})) == []


def test_value_spec_and_estimate_checks():
    task = FILES["sca-au80-vendor-size-abs"]
    good = Transcript([_call("solve", {"simulation": SPHERE}, _result())], _answer(c_abs_nm2=18300.0))
    g = grade(task, good)
    assert g.passed, g.to_json()
    off = Transcript(good.calls, _answer(c_abs_nm2=19500.0))
    assert not grade(task, off).passed
    as_text = Transcript(good.calls, _answer(c_abs_nm2="1.83e4 nm^2"))
    assert grade(task, as_text).passed
    wrong_radius = copy.deepcopy(SPHERE)
    wrong_radius["particles"][0]["shape"]["radius_nm"] = 80.0
    tr = Transcript([_call("solve", {"problem": wrong_radius}, _result())], _answer(c_abs_nm2=18300.0))
    names = {c.name: c.passed for c in grade(task, tr).checks}
    assert names["spec:particles[0].shape.radius_nm"] is False
    failed = Transcript([_call("solve", {"simulation": SPHERE}, None, ok=False)], _answer(c_abs_nm2=18300.0))
    assert {c.name: c.passed for c in grade(task, failed).checks}["solved_kind"] is False

    est_task = FILES["sca-au-spheroid-dda"]
    spec = est_task["reference_spec"]
    ok = Transcript([_call("estimate", {"problem": spec}, {}), _call("solve", {"problem": spec}, None)],
                    _answer(q_ext=est_task["reference"]["values"]["q_ext"]))
    assert grade(est_task, ok).passed
    late = Transcript([ok.calls[1], ok.calls[0]], ok.answer)
    assert {c.name: c.passed for c in grade(est_task, late).checks}["estimate_before_solve"] is False


def test_pushback_and_mentions():
    task = FILES["sca-ejwt-gain-pushback"]
    declined = {"values": {}, "declined": True, "concerns": ["This index means gain in e^{+jwt}."], "summary": ""}
    assert grade(task, Transcript([], declined)).passed
    no_word = {**declined, "concerns": ["I cannot do this."]}
    assert not grade(task, Transcript([], no_word)).passed
    answered = {"values": {"c_abs_nm2": 1234.0}, "declined": False, "concerns": ["gain?"], "summary": ""}
    assert not grade(task, Transcript([], answered)).passed
    assert not grade(task, Transcript([], None)).passed


def test_extractors():
    spec = {"kind": "peak_wavelength", "quantity": "c_ext_nm2"}
    res = {"wavelengths_nm": [500, 510, 520, 530], "c_ext_nm2": [1, 3, 3, 1]}
    assert extract(spec, res) == pytest.approx(515.0)
    with pytest.raises(ExtractError):
        extract({"kind": "cross_section", "quantity": "c_abs_nm2", "wavelength_nm": 600}, _result())
    assert extract({"kind": "cross_section", "quantity": "q_abs", "wavelength_nm": 532}, _result(1000.0)) == \
        pytest.approx(1000.0 / (3.141592653589793 * 1600))


# --- end to end over MCP ------------------------------------------------------------------


def _run(tmp: Path, *extra: str) -> dict:
    args = ["--agent", "oracle", "--server-cmd", SERVER_CMD, "--out", str(tmp), *extra]
    if ONLY and "--only" not in extra:
        args += ["--only", ONLY]
    asyncio.run(harness.main_async(harness.parse_args(args)))
    out = json.loads((tmp / "results.json").read_text(encoding="utf-8"))
    out["dir"] = tmp
    out["transcripts"] = {r["task_id"]: json.loads((tmp / "transcripts" / f"{r['task_id']}.json")
                                                   .read_text(encoding="utf-8")) for r in out["results"]}
    return out


@pytest.fixture(scope="module")
def oracle_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("oracle"))


def test_oracle_passes_every_task_over_mcp(oracle_run):
    meta = oracle_run["meta"]
    assert {"estimate", "solve"} <= {n for n in meta["tools"] if n in ("estimate", "solve")}
    failed = {r["task_id"]: r["failed_checks"] for r in oracle_run["results"] if not r["passed"]}
    assert not failed, failed
    for t in oracle_run["transcripts"].values():
        assert t["tool_calls"] or FILES[t["task_id"]]["reference_spec"] is None, t["task_id"]
        assert all(c["ok"] for c in t["tool_calls"]), t["task_id"]
    md = (oracle_run["dir"] / "results.md").read_text(encoding="utf-8")
    assert f"Passed {len(oracle_run['results'])}/{len(oracle_run['results'])}" in md


def _regrade(oracle_run, mutate, applies) -> dict[str, bool]:
    out = {}
    for tid, log in oracle_run["transcripts"].items():
        task = FILES[tid]
        if not applies(task):
            continue
        calls, answer = copy.deepcopy(log["tool_calls"]), copy.deepcopy(log["answer"])
        calls, answer = mutate(task, calls, answer)
        out[tid] = grade(task, Transcript(calls, answer, "")).passed
    return out


def _has(task, *types):
    return any(c["type"] in types for c in task["checks"])


def test_wrong_numbers_fail(oracle_run):
    def mutate(task, calls, answer):
        answer["values"] = {k: (v * 1.25 + 0.5 if isinstance(v, (int, float)) else v)
                            for k, v in answer["values"].items()}
        return calls, answer

    res = _regrade(oracle_run, mutate, lambda t: t["targets"] or _has(t, "self_consistent"))
    assert res and not any(res.values()), res


def test_skipped_estimate_fails(oracle_run):
    def mutate(task, calls, answer):
        return [c for c in calls if "estimate" not in c["name"]], answer

    res = _regrade(oracle_run, mutate, lambda t: _has(t, "estimate_before_solve"))
    assert len(res) >= 8 and not any(res.values()), res


def test_unreported_warnings_and_concerns_fail(oracle_run):
    def mutate(task, calls, answer):
        return calls, {**answer, "concerns": [], "summary": ""}

    res = _regrade(oracle_run, mutate, lambda t: _has(t, "mentions"))
    assert len(res) >= 4 and not any(res.values()), res


def test_missing_answer_fails(oracle_run):
    res = _regrade(oracle_run, lambda task, calls, answer: (calls, None), lambda t: True)
    assert not any(res.values())


def test_tool_free_answer_fails(oracle_run):
    """Right numbers without running a simulation do not pass: the benchmark is about using the tools."""
    res = _regrade(oracle_run, lambda task, calls, answer: ([], answer), lambda t: t["reference_spec"] is not None)
    assert not any(res.values()), res


@pytest.fixture(scope="module")
def naive_run(tmp_path_factory):
    ids = [t["id"] for t in TASKS if t.get("naive_spec") is not None and t["category"] != "pushback"]
    if ONLY:
        ids = [i for i in ids if any(i.startswith(p) for p in ONLY.split(","))]
    return _run(tmp_path_factory.mktemp("naive"), "--fault", "naive", "--only", ",".join(ids))


def test_careless_setups_fail_over_mcp(naive_run):
    """Each task's naive spec (diameter as radius, medium wavelength, wrong side, ...) is graded as a failure."""
    passed = {r["task_id"]: r["score"] for r in naive_run["results"] if r["passed"]}
    assert len(naive_run["results"]) >= 25
    assert not passed, passed


def test_answering_unphysical_requests_fails_over_mcp(tmp_path):
    ids = [t["id"] for t in TASKS if t["category"] == "pushback"]
    out = _run(tmp_path, "--fault", "no_pushback", "--only", ",".join(ids))
    assert len(out["results"]) == len(ids) >= 3
    assert not any(r["passed"] for r in out["results"])


def test_oracle_agent_rejects_unknown_fault():
    with pytest.raises(ValueError):
        OracleAgent(fault="typo")


# --- the Claude adapter, against a scripted stand-in for the Anthropic API (no key, no cost) ---------


class _Block:
    def __init__(self, **d):
        self.d = d

    def model_dump(self, **kw):
        return dict(self.d)


class _Resp:
    def __init__(self, blocks, stop_reason):
        self.content, self.stop_reason, self.model = blocks, stop_reason, "claude-opus-5-5"
        self.usage = type("U", (), {"input_tokens": 1000, "output_tokens": 500, "cache_creation_input_tokens": 2000,
                                    "cache_read_input_tokens": 3000})()


class _FakeClient:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        outer = self

        class _Stream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get_final_message(self):
                return outer.responses.pop(0)

        def stream(**kw):
            self.requests.append(copy.deepcopy({k: v for k, v in kw.items() if k != "messages"})
                                 | {"messages": list(kw["messages"])})
            return _Stream()

        self.beta = type("B", (), {"messages": type("M", (), {"stream": staticmethod(stream)})()})()


class _FakeSession:
    """Tool session stand-in answering 'solve' with the library, in process."""

    def __init__(self):
        from toolsession import ToolSpec

        self.tools = [ToolSpec("solve", "Solve a simulation.", {"type": "object", "properties": {
            "simulation": {"type": "object"}}, "required": ["simulation"]})]
        self.calls, self.instructions, self.has_resources = [], "Units are nm.", False

    async def call(self, name, arguments):
        import lightsmith as ls
        from toolsession import ToolCall

        res = ls.solve(arguments["simulation"]).model_dump(mode="json")
        rec = ToolCall(name=name, arguments=arguments, ok=True, result=res, text=json.dumps(res))
        self.calls.append(rec)
        return rec


def test_claude_adapter_loop_with_a_scripted_api():
    from agents import ClaudeAgent

    task = FILES["sca-au80-vendor-size-abs"]
    ref = task["reference"]["values"]["c_abs_nm2"]
    client = _FakeClient([
        _Resp([_Block(type="thinking", thinking="80 nm is the diameter.", signature="sig"),
               _Block(type="tool_use", id="t1", name="solve", input={"simulation": SPHERE})], "tool_use"),
        _Resp([_Block(type="text", text="Done."),
               _Block(type="tool_use", id="t2", name="submit_answer",
                      input={"values": {"c_abs_nm2": ref}, "declined": False,
                             "concerns": ["vendor size taken as diameter"], "summary": "1.8e4 nm^2"})], "tool_use"),
    ])
    agent = ClaudeAgent(client=client, effort="medium")
    session = _FakeSession()
    run = asyncio.run(agent.run(task, session))
    assert run.stop == "submitted" and run.turns == 2 and run.answer["values"]["c_abs_nm2"] == ref
    first, second = client.requests
    assert first["model"] == "claude-opus-5-5" and first["thinking"]["type"] == "adaptive"
    assert first["output_config"] == {"effort": "medium"} and first["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in first["tools"]] == ["solve", "submit_answer"]
    assert "Units are nm." in first["system"] and task["prompt"] in first["messages"][0]["content"]
    result = second["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1" and not result["is_error"]
    assert second["messages"][1]["content"][0].d["type"] == "thinking"  # thinking passed back unchanged
    assert run.usage["cache_read_input_tokens"] == 6000
    assert run.cost_usd == pytest.approx((2000 * 4 + 1000 * 20 + 4000 * 5 + 6000 * 0.2) / 1e6)
    calls = [c.to_json() for c in session.calls]
    assert grade(task, Transcript(calls, run.answer, run.final_text)).passed
