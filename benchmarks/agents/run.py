"""Run an agent on the Lightsmith agent benchmark over MCP and grade it.

    uv run --no-sync python benchmarks/agents/run.py --agent oracle
    uv run --no-sync --with anthropic python benchmarks/agents/run.py --agent claude --model claude-opus-5-5
    uv run --no-sync python benchmarks/agents/run.py --regrade benchmarks/agents/results/<run>   # no model calls

Writes ``benchmarks/agents/results/<timestamp>-<agent>/``: ``results.json`` (run
metadata, per-task grades, usage and cost), ``results.md`` (tables) and
``transcripts/<task>.json`` (prompt, every tool call with arguments and
replies, the model-side messages, the answer and the grade).

``--regrade <dir>`` grades a finished run again from its saved transcripts with the
current graders and task files, and writes ``<dir>-regraded-<timestamp>/`` (the
original is left as it is); ``results.json`` gains ``meta.regraded`` listing every
task whose grade changed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agents import FAULTS, AgentRun, make_agent, user_message  # noqa: E402
from grading import Transcript, grade  # noqa: E402
from toolsession import ToolSession, default_server_command  # noqa: E402

TASK_DIR = HERE / "tasks"
# never handed to the MCP server process
SECRET_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "LIGHTSMITH_API_KEY", "LIGHTSMITH_CLOUD_API_KEY",
              "LIGHTSMITH_CLOUD_URL")


def load_tasks(only: str = "", categories: str = "", kinds: str = "") -> list[dict]:
    index = json.loads((TASK_DIR / "index.json").read_text(encoding="utf-8"))
    pre = [p for p in only.split(",") if p]
    cats = {c for c in categories.split(",") if c}
    ks = {k for k in kinds.split(",") if k}
    out = []
    for row in index:
        if pre and not any(row["id"].startswith(p) for p in pre):
            continue
        if cats and row["category"] not in cats:
            continue
        if ks and row["kind"] not in ks:
            continue
        out.append(json.loads((TASK_DIR / f"{row['id']}.json").read_text(encoding="utf-8")))
    return out


def server_env(data_dir: Path, extra: dict[str, str]) -> dict[str, str]:
    """Environment for the server process: inherited, minus secrets, with an isolated job store."""
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
    env["LIGHTSMITH_MCP_DATA_DIR"] = str(data_dir)
    env.update(extra)
    return env


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


async def run_task(agent, task: dict, session: ToolSession, timeout_s: float) -> tuple[AgentRun, list[dict], float]:
    start = len(session.calls)
    t0 = time.perf_counter()
    try:
        run = await asyncio.wait_for(agent.run(task, session), timeout_s)
    except asyncio.TimeoutError:
        run = AgentRun(stop="timeout", error=f"agent exceeded {timeout_s:g} s")
    except Exception as e:  # an agent crash is a failed task, not a failed benchmark
        run = AgentRun(stop="error", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    wall = time.perf_counter() - t0
    calls = [c.to_json() for c in session.calls[start:]]
    return run, calls, wall


def write_task_log(out: Path, task: dict, agent, run: AgentRun, calls: list[dict], g, wall: float) -> None:
    log = {"task_id": task["id"], "agent": agent.name, "prompt": user_message(task), "answer": run.answer,
           "final_text": run.final_text, "stop": run.stop, "error": run.error, "notes": run.notes,
           "turns": run.turns, "usage": run.usage, "cost_usd": run.cost_usd, "wall_s": round(wall, 2),
           "grade": g.to_json(), "reference": task.get("reference"), "tool_calls": calls, "messages": run.messages}
    (out / "transcripts" / f"{task['id']}.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")


def row_for(task: dict, run: AgentRun, calls: list[dict], g, wall: float) -> dict:
    est = sum(1 for c in calls if "estimate" in c["name"].lower())
    return {"task_id": task["id"], "kind": task["kind"], "category": task["category"], "traps": task["traps"],
            "passed": g.passed, "score": round(g.score, 4),
            "failed_checks": [c.name for c in g.checks if not c.passed and c.required],
            "tool_calls": len(calls), "tool_errors": sum(1 for c in calls if not c["ok"]), "estimate_calls": est,
            "turns": run.turns, "stop": run.stop, "wall_s": round(wall, 2), "usage": run.usage,
            "cost_usd": None if run.cost_usd is None else round(run.cost_usd, 4), "notes": run.notes}


def markdown(meta: dict, rows: list[dict]) -> str:
    n, passed = len(rows), sum(r["passed"] for r in rows)
    mean = sum(r["score"] for r in rows) / max(n, 1)
    cost = sum(r["cost_usd"] or 0 for r in rows)
    lines = [f"# Lightsmith agent benchmark: {meta['agent']}", "",
             f"- Run: {meta['started']}, {meta['wall_s']:.0f} s, server `{meta['server']}` "
             f"({len(meta['tools'])} tools: {', '.join(meta['tools'])})",
             f"- lightsmith {meta['lightsmith_version']}, commit {meta['git_commit']}, {meta['platform']}",
             f"- **Passed {passed}/{n} ({100 * passed / max(n, 1):.0f}%)**, mean score {mean:.3f}"
             + (f", API cost ${cost:.2f}" if meta.get("model") else ""), ""]
    for by in ("kind", "category"):
        agg: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            agg[r[by]].append(r)
        lines += [f"| {by} | tasks | passed | mean score |", "|---|---:|---:|---:|"]
        lines += [f"| {k} | {len(v)} | {sum(x['passed'] for x in v)} | {sum(x['score'] for x in v) / len(v):.3f} |"
                  for k, v in sorted(agg.items())]
        lines.append("")
    if meta.get("regraded"):
        rg = meta["regraded"]
        lines[5:5] = [f"- Regraded {rg['at']} from `{rg['from']}` at commit {rg['git_commit']}: "
                      f"{len(rg['changed'])} task grade(s) changed", ""]
    lines += ["| task | kind | category | pass | score | tool calls (errors) | turns | stop | time s | cost $ "
              "| failed checks |",
              "|---|---|---|:-:|---:|---:|---:|---|---:|---:|---|"]
    for r in rows:
        cost = "" if r["cost_usd"] is None else f"{r['cost_usd']:.3f}"
        lines.append(f"| {r['task_id']} | {r['kind']} | {r['category']} | {'✅' if r['passed'] else '❌'} | "
                     f"{r['score']:.2f} | {r['tool_calls']} ({r['tool_errors']}) | {r['turns']} | {r['stop']} | "
                     f"{r['wall_s']:.1f} | {cost} | {', '.join(r['failed_checks'])} |")
    return "\n".join(lines) + "\n"


def run_summary(meta: dict, rows: list[dict]) -> dict:
    return {"tasks": len(rows), "passed": sum(r["passed"] for r in rows),
            "mean_score": sum(r["score"] for r in rows) / len(rows),
            "cost_usd": sum(r["cost_usd"] or 0 for r in rows) if meta["model"] else None,
            "usage": {k: sum((r["usage"] or {}).get(k, 0) for r in rows)
                      for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                "cache_read_input_tokens")}}


def regrade(src: Path, out: Path | None = None) -> Path:
    """Grade a finished run again from its transcripts; write a new results directory."""
    src = Path(src).resolve()
    old = json.loads((src / "results.json").read_text(encoding="utf-8"))
    meta, rows = old["meta"], old["results"]
    files = {t["id"]: t for t in load_tasks()}
    missing = [r["task_id"] for r in rows if r["task_id"] not in files]
    if missing:
        raise SystemExit(f"no task file for {missing}")
    stamp = datetime.now()
    out = Path(out) if out else src.with_name(f"{src.name}-regraded-{stamp.strftime('%Y%m%d-%H%M%S')}")
    (out / "transcripts").mkdir(parents=True, exist_ok=False)
    changed = []
    for row in rows:
        tid = row["task_id"]
        log = json.loads((src / "transcripts" / f"{tid}.json").read_text(encoding="utf-8"))
        calls = log["tool_calls"]
        g = grade(files[tid], Transcript(calls=calls, answer=log["answer"], final_text=log.get("final_text") or ""))
        before = {"passed": row["passed"], "score": row["score"], "failed_checks": row["failed_checks"]}
        row.update(passed=g.passed, score=round(g.score, 4),
                   failed_checks=[c.name for c in g.checks if not c.passed and c.required])
        if (before["passed"], before["score"]) != (row["passed"], row["score"]):
            changed.append({"task_id": tid, "before": before, "after": {k: row[k] for k in before}})
        log["original_grade"], log["grade"] = log["grade"], g.to_json()
        (out / "transcripts" / f"{tid}.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
        mark = "PASS" if g.passed else "FAIL"
        delta = f"  (was {before['score']:.2f})" if before["score"] != row["score"] else ""
        print(f"{mark} {tid:36s} score {g.score:.2f}{delta}  {', '.join(row['failed_checks'])}", flush=True)
    meta["regraded"] = {"from": src.name, "at": stamp.isoformat(timespec="seconds"), "git_commit": _git_commit(),
                        "original_summary": {k: old["summary"][k] for k in ("tasks", "passed", "mean_score")},
                        "changed": changed}
    summary = run_summary(meta, rows)
    (out / "results.json").write_text(json.dumps({"meta": meta, "summary": summary, "results": rows}, indent=1,
                                                 default=str), encoding="utf-8")
    (out / "results.md").write_text(markdown(meta, rows), encoding="utf-8")
    o = old["summary"]
    print(f"\npassed {o['passed']}/{o['tasks']} -> {summary['passed']}/{summary['tasks']}, mean score "
          f"{o['mean_score']:.3f} -> {summary['mean_score']:.3f}, {len(changed)} task(s) changed\n-> {out}")
    return out


async def main_async(args) -> int:
    import lightsmith

    tasks = load_tasks(args.only, args.category, args.kind)
    if not tasks:
        print("no tasks selected")
        return 2
    agent = make_agent(args.agent, model=args.model, effort=args.effort, max_turns=args.max_turns,
                       max_cost_usd=args.max_cost_per_task, fault=args.fault,
                       **({"fallbacks": False} if args.no_fallbacks else {}))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else HERE / "results" / f"{stamp}-{agent.name.replace(':', '_')}"
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    command = ([a.strip('"') for a in shlex.split(args.server_cmd, posix=os.name != "nt")] if args.server_cmd
               else default_server_command())
    extra_env = dict(kv.split("=", 1) for kv in args.server_env)
    data_root = Path(tempfile.mkdtemp(prefix="lightsmith-agentbench-"))
    fresh = args.server_per_task if args.server_per_task is not None else args.agent != "oracle"

    def new_session(tag: str) -> ToolSession:
        d = data_root / tag
        d.mkdir(parents=True, exist_ok=True)
        return ToolSession(transport=args.transport, command=command, env=server_env(d, extra_env), cwd=args.server_cwd)

    meta = {"agent": agent.name, "model": getattr(agent, "model", None), "fault": args.fault,
            "started": datetime.now().isoformat(timespec="seconds"), "server_command": command,
            "transport": args.transport, "server_per_task": fresh, "lightsmith_version": lightsmith.__version__,
            "git_commit": _git_commit(), "platform": f"{platform.system()} {platform.release()}, Python "
            f"{platform.python_version()}", "tasks": [t["id"] for t in tasks]}
    rows, t_start = [], time.perf_counter()
    shared = None
    try:
        if not fresh:
            shared = await new_session("shared").__aenter__()
        for i, task in enumerate(tasks, 1):
            session = shared or await new_session(task["id"]).__aenter__()
            try:
                meta.setdefault("server", session.server_name)
                meta.setdefault("tools", [t.name for t in session.tools])
                meta.setdefault("server_instructions", session.instructions)
                run, calls, wall = await run_task(agent, task, session, args.task_timeout)
            finally:
                if session is not shared:
                    await session.__aexit__(None, None, None)
            g = grade(task, Transcript(calls=calls, answer=run.answer, final_text=run.final_text))
            write_task_log(out, task, agent, run, calls, g, wall)
            rows.append(row_for(task, run, calls, g, wall))
            mark = "PASS" if g.passed else "FAIL"
            extra = f"  ${run.cost_usd:.3f}" if run.cost_usd is not None else ""
            print(f"[{i:2d}/{len(tasks)}] {mark} {task['id']:36s} score {g.score:.2f}  {len(calls):2d} calls  "
                  f"{wall:6.1f} s{extra}  {', '.join(rows[-1]['failed_checks'])}", flush=True)
    finally:
        if shared is not None:
            await shared.__aexit__(None, None, None)
    meta["wall_s"] = time.perf_counter() - t_start
    meta.setdefault("server", None)
    meta.setdefault("tools", [])
    summary = run_summary(meta, rows)
    (out / "results.json").write_text(json.dumps({"meta": meta, "summary": summary, "results": rows}, indent=1,
                                                 default=str), encoding="utf-8")
    (out / "results.md").write_text(markdown(meta, rows), encoding="utf-8")
    print(f"\npassed {summary['passed']}/{summary['tasks']}, mean score {summary['mean_score']:.3f}"
          + (f", cost ${summary['cost_usd']:.2f}" if summary["cost_usd"] is not None else "") + f"\n-> {out}")
    return 0 if summary["passed"] == summary["tasks"] or not args.strict else 1


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", choices=["oracle", "claude"], default="oracle")
    ap.add_argument("--model", default=None, help="Claude model (default claude-opus-5-5)")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude effort (default high)")
    ap.add_argument("--max-turns", type=int, default=None, help="Claude: model turns per task (default 40)")
    ap.add_argument("--max-cost-per-task", type=float, default=None, help="Claude: stop a task above this many USD "
                    "(default 5)")
    ap.add_argument("--no-fallbacks", action="store_true", help="Claude: no server-side refusal fallback")
    ap.add_argument("--fault", choices=FAULTS, default=None, help="oracle: misbehave on purpose (grader tests)")
    ap.add_argument("--only", default="", help="comma-separated task id prefixes")
    ap.add_argument("--category", default="", help="comma-separated categories (standard,trap,pushback,warning)")
    ap.add_argument("--kind", default="", help="comma-separated problem kinds")
    ap.add_argument("--server-cmd", default=None, help="MCP server command (default: lightsmith-mcp of this env)")
    ap.add_argument("--server-cwd", default=None)
    ap.add_argument("--server-env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra environment for the server (repeatable)")
    ap.add_argument("--transport", choices=["stdio", "inproc"], default="stdio")
    sp = ap.add_mutually_exclusive_group()
    sp.add_argument("--server-per-task", dest="server_per_task", action="store_true", default=None,
                    help="fresh server for every task (default for claude)")
    sp.add_argument("--shared-server", dest="server_per_task", action="store_false",
                    help="one server for the whole run (default for the oracle)")
    ap.add_argument("--task-timeout", type=float, default=3600.0, help="seconds per task")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--strict", action="store_true", help="exit 1 unless every task passes")
    ap.add_argument("--regrade", default=None, metavar="DIR",
                    help="grade a finished run again from its transcripts (no agent, no server); writes "
                         "DIR-regraded-<timestamp>/ or --out")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.regrade:
        regrade(Path(args.regrade), args.out)
        return 0
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
