"""Agents for the benchmark: the scripted oracle and Claude via the Anthropic API.

Both see the server only through a :class:`toolsession.ToolSession` and finish by
calling the harness tool ``submit_answer``. The answer format they are given is
the same (:func:`answer_instructions`).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from extract import ExtractError, extract
from grading import is_estimate, is_solve, results_in
from toolsession import ToolCall, ToolSession

SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": "Give your final answer to the user. Call exactly once, when you are done.",
    "input_schema": {
        "type": "object",
        "properties": {
            "values": {"type": "object", "additionalProperties": {"type": ["number", "null"]},
                       "description": "The requested quantities, keyed by the names listed in the request, as plain "
                                      "numbers in the stated units. Leave out (or set null) any value you cannot or "
                                      "should not give."},
            "declined": {"type": "boolean",
                         "description": "true if you are not giving the requested numbers because the request cannot "
                                        "be answered as asked."},
            "concerns": {"type": "array", "items": {"type": "string"},
                         "description": "Anything the user must know: assumptions you made, problems with the "
                                        "request, warnings from the tools. Empty if none."},
            "summary": {"type": "string", "description": "Your answer to the user in a few sentences."},
        },
        "required": ["values", "declined", "concerns", "summary"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are a photonics simulation assistant. You can run electromagnetic simulations with the \
Lightsmith tools (an MCP server). Answer the user's question by setting up and running the appropriate \
simulation with these tools, then report the result.

The user will not answer follow-up questions: make reasonable assumptions and state them. When you are done, \
call submit_answer exactly once; the user reads only that."""


def answer_instructions(task: dict) -> str:
    lines = ["", "---", "Report your result by calling submit_answer with these values:"]
    for f in task["answer_fields"]:
        lines.append(f"- `{f['key']}` ({f['unit']}): {f['description']}")
    return "\n".join(lines)


def user_message(task: dict) -> str:
    return task["prompt"] + "\n" + answer_instructions(task)


@dataclass
class AgentRun:
    answer: dict[str, Any] | None = None
    final_text: str = ""
    stop: str = "submitted"  # submitted | no_answer | max_turns | budget | refusal | error
    error: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)  # model-side transcript (Claude)
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float | None = None
    turns: int = 0
    notes: list[str] = field(default_factory=list)


def _problem_arg(session: ToolSession, tool: str) -> str:
    props = session.schema_of(tool).get("properties", {})
    for name in ("problem", "simulation", "spec"):
        if name in props:
            return name
    return next(iter(props), "problem")


def _fit_args(session: ToolSession, tool: str, **kw) -> dict[str, Any]:
    props = session.schema_of(tool).get("properties")
    return {k: v for k, v in kw.items() if props is None or k in props}


def _find_key(obj: Any, key: str, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], (str, int)):
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key, depth + 1)
            if found is not None:
                return found
    return None


async def full_result(session: ToolSession, call: ToolCall, kind: str | None = None) -> dict | None:
    """The complete result behind a solve call: inline, or fetched by job id (jobs API servers)."""
    for r in results_in(call.result):
        if kind is None or r.get("kind") == kind:
            return r
    job = _find_key(call.result, "job_id")
    if job is None:
        return None
    status = _find_key(call.result, "status")
    get_job = session.tool("job", exclude=("list", "cancel", "start"))
    deadline = time.monotonic() + 1800
    while status in ("queued", "running", "pending") and get_job and time.monotonic() < deadline:
        c = await session.call(get_job, _fit_args(session, get_job, job_id=job, wait_s=60))
        status = _find_key(c.result, "status")
        for r in results_in(c.result):
            return r
    getter = session.tool("result", exclude=("list",))
    if getter:
        c = await session.call(getter, _fit_args(session, getter, job_id=job, format="json", max_chars=5_000_000))
        for r in results_in(c.result):
            return r
    if session.has_resources:
        c = await session.read_resource(f"lightsmith://jobs/{job}/result")
        for r in results_in(c.result):
            return r
    return None


# --- the oracle ------------------------------------------------------------------------

FAULTS = ("wrong_values", "naive", "no_estimate", "no_pushback", "silent")


class OracleAgent:
    """Plays each task's reference spec through MCP and reports the extracted values.

    ``fault`` makes it misbehave on purpose, to prove the graders catch it:
    wrong_values (numbers off by 25% + 0.5), naive (solves the task's naive_spec),
    no_estimate (skips estimate), no_pushback (answers unphysical requests),
    silent (reports no concerns).
    """

    def __init__(self, fault: str | None = None):
        if fault is not None and fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}; choose from {FAULTS}")
        self.fault = fault
        self.name = "oracle" + (f"-{fault}" if fault else "")
        self.model = None

    async def run(self, task: dict, session: ToolSession) -> AgentRun:
        run = AgentRun()
        oracle = task.get("oracle", {})
        concerns = [] if self.fault in ("silent", "naive", "no_pushback") else list(oracle.get("concerns", []))
        spec = task.get("reference_spec")
        extractors = dict(task.get("oracle_extract", {}))
        if self.fault in ("naive", "no_pushback") and task.get("naive_spec") is not None:
            spec = task["naive_spec"]
            extractors.update(oracle.get("naive_extract", {}))
        if spec is None or (self.fault != "no_pushback" and oracle.get("decline")):
            if self.fault == "no_pushback":
                run.answer = {"values": {f["key"]: 1.0 for f in task["answer_fields"]}, "declined": False,
                              "concerns": [], "summary": "Done."}
            else:
                run.answer = {"values": {}, "declined": True, "concerns": concerns, "summary": " ".join(concerns)}
            return run
        method, device = task.get("reference_method", "auto"), task.get("reference_device", "auto")
        solve = session.tool("solve", exclude=("start",)) or next((t.name for t in session.tools if is_solve(t.name)),
                                                                 None)
        estimate = next((t.name for t in session.tools if is_estimate(t.name)), None)
        if solve is None:
            run.stop, run.error = "error", "server has no solve tool"
            return run
        if estimate and self.fault != "no_estimate":
            await session.call(estimate, _fit_args(session, estimate, **{_problem_arg(session, estimate): spec},
                                                   method=method, device=device))
        call = await session.call(solve, _fit_args(session, solve, **{_problem_arg(session, solve): spec},
                                                   method=method, device=device))
        if not call.ok:
            run.stop, run.error = "error", f"solve failed: {call.error}"
            run.answer = {"values": {}, "declined": False, "concerns": [call.error or ""], "summary": "Solve failed."}
            return run
        res = await full_result(session, call)
        values: dict[str, float | None] = {}
        for key, ex in extractors.items():
            try:
                values[key] = extract(ex, res) if res is not None else None
            except ExtractError as e:
                values[key] = None
                run.notes.append(f"{key}: {e}")
        if self.fault == "wrong_values":
            values = {k: (v * 1.25 + 0.5 if v is not None else None) for k, v in values.items()}
        if res is not None and res.get("warnings") and self.fault != "silent":
            concerns += [f"solver warning: {w}" for w in res["warnings"]]
        run.answer = {"values": values, "declined": False, "concerns": concerns,
                      "summary": json.dumps(values)}
        return run


# --- Claude via the Anthropic API -----------------------------------------------------

# USD per million tokens: input, output, cache write (5 min), cache read.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 5.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 6.25, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 2.5, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
    "claude-fable-5-1": (10.0, 50.0, 12.5, 0.25),
}


def cost_usd(model: str, usage: dict[str, int]) -> float | None:
    p = PRICES.get(model)
    if p is None:
        return None
    return (usage.get("input_tokens", 0) * p[0] + usage.get("output_tokens", 0) * p[1]
            + usage.get("cache_creation_input_tokens", 0) * p[2] + usage.get("cache_read_input_tokens", 0) * p[3]) / 1e6


def _block_dump(b: Any) -> dict:
    return b.model_dump(mode="json", exclude_none=True) if hasattr(b, "model_dump") else dict(b)


class ClaudeAgent:
    """A manual tool-use loop over the Messages API (streamed), with the MCP tools listed at run time.

    The API key is read by the SDK from ANTHROPIC_API_KEY (or the other credential sources the SDK
    supports); this class never reads, logs or stores it.
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", max_turns: int = 40,
                 max_cost_usd: float = 5.0, max_tool_chars: int = 60_000, fallbacks: bool = True,
                 thinking_display: str = "summarized", max_tokens: int = 64_000, client: Any = None):
        if client is None:
            import anthropic  # optional dependency: uv run --with anthropic ...

            client = anthropic.AsyncAnthropic()
        self.client = client
        self.model, self.effort = model, effort
        self.max_turns, self.max_cost_usd, self.max_tool_chars = max_turns, max_cost_usd, max_tool_chars
        self.fallbacks, self.thinking_display, self.max_tokens = fallbacks, thinking_display, max_tokens
        self.name = f"claude:{model}:{effort}"

    def _tools(self, session: ToolSession, resources: bool) -> list[dict]:
        tools = []
        for t in session.tools:
            schema = t.input_schema if t.input_schema.get("type") == "object" else {"type": "object", **t.input_schema}
            tools.append({"name": t.name, "description": t.description[:4000] or t.name, "input_schema": schema})
        if resources:
            tools.append({"name": "list_resources", "description": "List the MCP resources (and resource "
                          "templates) the Lightsmith server offers.", "input_schema": {"type": "object",
                                                                                      "properties": {}}})
            tools.append({"name": "read_resource", "description": "Read an MCP resource of the Lightsmith "
                          "server by URI.", "input_schema": {"type": "object", "properties": {
                              "uri": {"type": "string"}}, "required": ["uri"]}})
        return tools + [SUBMIT_TOOL]

    def _tool_result(self, block_id: str, rec: ToolCall) -> dict:
        text = rec.text or ("(empty result)" if rec.ok else "(error)")
        if len(text) > self.max_tool_chars:
            text = (text[:self.max_tool_chars] + f"\n[truncated by the client: {len(rec.text)} characters, "
                    f"showing {self.max_tool_chars}]")
        content: list[dict] = [{"type": "text", "text": text}]
        content += [{"type": "image", "source": {"type": "base64", "media_type": i["mime_type"], "data": i["data"]}}
                    for i in rec.images]
        return {"type": "tool_result", "tool_use_id": block_id, "content": content, "is_error": not rec.ok}

    async def run(self, task: dict, session: ToolSession) -> AgentRun:
        try:
            from anthropic import APIConnectionError, APIStatusError
        except ImportError:  # a test double instead of the SDK
            APIConnectionError = APIStatusError = type("_NeverRaised", (Exception,), {})

        run = AgentRun()
        resources = False
        if session.has_resources:
            probe = await session.list_resources()
            session.calls.remove(probe)  # harness probe, not an agent action
            resources = probe.ok and bool(probe.result)
        tools = self._tools(session, resources)
        system = SYSTEM_PROMPT + (f"\n\nInstructions from the Lightsmith server:\n{session.instructions}"
                                  if session.instructions else "")
        messages: list[dict] = [{"role": "user", "content": user_message(task)}]  # sent to the API
        log: list[dict] = [dict(messages[0])]  # the same, JSON-serialisable
        usage = {k: 0 for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                "cache_read_input_tokens")}
        nudged = False
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=system, tools=tools,
                                      thinking={"type": "adaptive", "display": self.thinking_display},
                                      output_config={"effort": self.effort}, cache_control={"type": "ephemeral"})
        if self.fallbacks:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        while True:
            if run.turns >= self.max_turns:
                run.stop = "max_turns"
                break
            run.turns += 1
            try:
                async with self.client.beta.messages.stream(messages=messages, **kwargs) as stream:
                    resp = await stream.get_final_message()
            except APIStatusError as e:
                run.stop, run.error = "error", f"API error {e.status_code}: {e.message}"
                break
            except APIConnectionError as e:
                run.stop, run.error = "error", f"connection error: {e}"
                break
            for k in usage:
                usage[k] += getattr(resp.usage, k, 0) or 0
            content = [_block_dump(b) for b in resp.content]
            messages.append({"role": "assistant", "content": resp.content})  # unchanged, thinking blocks included
            log.append({"role": "assistant", "content": content, "stop_reason": resp.stop_reason,
                        "model": resp.model})
            if any(b.get("type") == "fallback" for b in content):
                run.notes.append(f"turn {run.turns}: served by fallback model {resp.model}")
            texts = [b.get("text", "") for b in content if b.get("type") == "text"]
            if texts:
                run.final_text = "\n".join(texts)
            run.cost_usd = cost_usd(self.model, usage)
            if resp.stop_reason == "refusal":
                run.stop = "refusal"
                break
            if resp.stop_reason == "pause_turn":
                continue
            uses = [b for b in content if b.get("type") == "tool_use"]
            if not uses:
                if run.answer is None and not nudged:
                    nudged = True
                    messages.append({"role": "user", "content": "Please call submit_answer now with your result."})
                    log.append(messages[-1])
                    continue
                break
            results = []
            for b in uses:
                if b["name"] == "submit_answer":
                    run.answer = b["input"]
                    results.append({"type": "tool_result", "tool_use_id": b["id"], "content": "Answer recorded."})
                    continue
                if b["name"] == "list_resources" and resources:
                    rec = await session.list_resources()
                elif b["name"] == "read_resource" and resources:
                    rec = await session.read_resource(str(b["input"].get("uri", "")))
                else:
                    rec = await session.call(b["name"], b["input"])
                results.append(self._tool_result(b["id"], rec))
            messages.append({"role": "user", "content": results})
            log.append({"role": "user", "content": [_loggable(r) for r in results]})
            if run.answer is not None:
                break
            if run.cost_usd is not None and run.cost_usd > self.max_cost_usd:
                run.stop = "budget"
                break
        run.messages, run.usage = log, usage
        run.cost_usd = cost_usd(self.model, usage)
        if run.answer is None and run.stop == "submitted":
            run.stop = "no_answer"
            run.answer = _answer_from_text(run.final_text)
        return run


def _loggable(block: dict) -> dict:
    """A tool_result block with image data left out of the transcript."""
    if not isinstance(block.get("content"), list):
        return block
    return {**block, "content": [{**c, "source": {**c["source"], "data": "<omitted>"}} if c.get("type") == "image"
                                 else c for c in block["content"]]}


def _answer_from_text(text: str) -> dict | None:
    """Last resort: a JSON object with "values" in the final text."""
    start = text.rfind('{"values"')
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


def make_agent(name: str, **kw) -> Any:
    if name == "oracle":
        return OracleAgent(fault=kw.get("fault"))
    if name == "claude":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE")):
            print("note: ANTHROPIC_API_KEY is not set; the SDK will try its other credential sources")
        return ClaudeAgent(**{k: v for k, v in kw.items() if k != "fault" and v is not None})
    raise ValueError(f"unknown agent {name!r}")

