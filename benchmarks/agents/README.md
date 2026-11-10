# Lightsmith agent benchmark

This benchmark asks whether an AI agent can answer a natural-language photonics question by using the Lightsmith MCP server (`lightsmith-mcp`) to set up and solve the right simulation.
The technical report lists this as the *agent evaluation protocol*.

Each task gives the agent a question written the way a user would write it.
The agent can use the MCP server's tools and must finish by calling `submit_answer`.
An automatic grader then compares the answer with a reference result that the library computed from a hidden reference problem.
The grader also inspects the tool calls themselves: which problem was actually solved, whether `estimate` ran before an expensive solve, and whether the agent pushed back on requests that cannot be answered as asked.

| File | Purpose |
|---|---|
| `taskdefs.py` | Task definitions: prompt, answer fields, hidden reference spec, targets with tolerances, extra checks, a "naive" spec showing what a careless agent would solve, and oracle behaviour. |
| `generate.py` | Solves every reference spec in process and writes `tasks/<id>.json` with the reference values, plus `tasks/index.json`. |
| `tasks/` | Generated task files, checked in so reference values can be reviewed. |
| `extract.py` | Extractors that read a quantity from a result dict (cross-section at λ, peak wavelength, R/T/A, diffraction order, n_eff/n_g, \|S\|², FSR, band gap, …). The generator, the oracle and the graders all use them. |
| `grading.py` | Graders. They read only the task file, the submitted answer and the logged MCP tool calls. |
| `toolsession.py` | Logged MCP client (stdio or in-process). Tools and resources are listed at run time. |
| `agents.py` | `OracleAgent` (a scripted agent that plays the reference specs) and `ClaudeAgent` (Anthropic API, manual tool-use loop). |
| `run.py` | Harness: runs an agent over the tasks and writes results as JSON and Markdown, plus one transcript per task. |
| `legacy_server.py` | Runs the original six-tool server on MCP SDK 2.x (see [Servers](#servers)). |
| `tests/` | Grader unit tests, an oracle run over stdio MCP, and fault-injection runs. |

## Task set

44 tasks cover every problem kind:

| Kind | Tasks |
|---|---:|
| scattering | 12 |
| emission | 1 |
| layer_stack | 8 |
| mode | 7 |
| circuit | 5 |
| periodic | 4 |
| bands | 4 |
| device | 2 |
| optimization | 1 |

By category: 23 standard, 16 trap, 3 pushback and 2 warning tasks.

- **Standard**: a well-posed question. Examples: the plasmon peak of a gold sphere in water, an AR coating, a Bragg mirror, grating efficiencies, Si/SiN waveguide modes, a 2D MMI, the TM gap of a photonic crystal, an MZI or ring FSR, and a small topology optimisation.
- **Trap**: the question is answerable, but the obvious setup is wrong. The traps are:
  - diameter vs radius ("80 nm gold nanoparticles")
  - an emitter's distance given from the particle's surface, not its centre
  - wavelength given *in the medium*
  - photon energy (eV), wavenumber (cm⁻¹), frequency (THz) or µm units
  - n = 2.0 − 0.3j in the e^{+jωt} convention, where loss must become Im n = +0.3
  - light incident from the substrate side
  - a 1 mm slide that must be an incoherent layer
  - group index vs effective index
  - normalised frequency a/λ that must be turned into wavelengths
  - a material dataset that does not cover the wavelength
  - an index-matched lattice that has no gap
  - cm vs nm lengths in a link budget
- **Pushback**: the request is unphysical and the agent should decline and explain why. The three cases are:
  - n = 1.6 + 0.02j in e^{+jωt}, which is gain, not loss
  - n_eff = 3.8 above the core index
  - a lossless 1×2 splitter with 70% in each arm
- **Warning**: the tools warn about the answer and the agent must pass the warning on. Examples: the quasistatic approximation used for a 150 nm sphere; counting guided modes when the solver flags modes below cladding index.

11 tasks use grid-based solvers (BEM, RCWA, EME, PWE, inverse design) and also grade `estimate_before_solve`.

Reference values come from the installed library at the resolution in `taskdefs.py`.
Tolerances are physics tolerances, not reproducibility tolerances: a sensible agent at default resolution should land inside them, and a wrong setup should not.
The naive specs in `taskdefs.py` show what lands outside.

To regenerate after a library change:

```bash
uv run --no-sync python benchmarks/agents/generate.py           # rewrite tasks/*.json
uv run --no-sync python benchmarks/agents/generate.py --check   # recompute and compare, write nothing
```

## Protocol

1. **Isolation.** By default the harness starts a fresh server for every task (`--server-per-task`; the oracle shares one).
   Each server gets a scratch `LIGHTSMITH_MCP_DATA_DIR`, so jobs from earlier tasks are invisible.
   `ANTHROPIC_API_KEY`, `LIGHTSMITH_API_KEY` and `LIGHTSMITH_CLOUD_URL` are removed from the server's environment, so solves stay local.
2. **What the agent sees:**
   - every tool the server lists, with its own description and schema
   - the server's `instructions`, if it sends any
   - `list_resources` / `read_resource` helper tools, when the server has resources
   - `submit_answer`
   - the system prompt (`agents.SYSTEM_PROMPT`). It says the user will not answer follow-up questions and that the agent must finish with `submit_answer`. It does not mention traps, estimates or pushback.
   - the user message: the task prompt plus the list of answer keys with units (`agents.answer_instructions`)
3. **Answer.** The agent calls `submit_answer` with these fields:
   - `values`: numbers in the stated units
   - `declined`: true or false
   - `concerns`: a list of strings
   - `summary`
4. **Transcript.** Every MCP call is logged with its arguments, the parsed reply, any error and its timing, together with the model-side messages (thinking summaries, text, tool calls).

## Scoring

A task **passes** when every required check passes.
The **score** is the fraction of all checks that pass, which shows partial credit.

| Check | Passes when |
|---|---|
| `value:<key>` | The reported number is within `abs + rel·|ref|` of the reference. |
| `solved_kind` | At least one solve of a problem of the task's kind succeeded. This is added to every task with a reference spec, so a correct number without a simulation does not pass. |
| `spec:<path>`, `material:<path>`, `wavelength:<λ>` | Some successfully solved problem has this field value, material index at λ, or wavelength. These are read from the tool-call arguments, not from the agent's claims, and they diagnose *why* a value check failed. |
| `spec_ratio:<path>/<over>` | Some successfully solved problem has this ratio of two fields (the length, if `<over>` is a vector). Used where the prompt gives sizes only relative to a lattice constant, e.g. a rod radius of 0.2a. |
| `estimate_before_solve` | An `estimate` call for a problem of this kind came before its first successful solve. |
| `pushback` | `declined` is true, or no values were given and concerns were stated. |
| `mentions` | The concerns or summary contain one of the expected words (e.g. "gain", "unreliable", "energy conservation"). |
| `self_consistent`, `threshold` | For the optimisation task: the reported transmission matches the agent's own final result, and that transmission is ≥ 0.5. The final result is the last full result in the transcript or, if the agent answered from the job summary that `solve` returns, that summary's `validated_sparameters`. |

The graders are independent of server generation:
- An "estimate" is any tool whose name contains `estimate`.
- A "solve" is any tool whose name contains `solve`, `compare`, `optimiz` or `run`. This covers `start_solve`.
- A problem is any argument that `lightsmith.problems.parse_problem` accepts, whether passed as `simulation=` or `problem=`.
- Results are found inline or inside job outputs.

`results/<timestamp>-<agent>/` contains these files:
- `results.json`: run metadata (server, tool list, versions, commit) and one row per task with pass/score, failed checks, tool calls and errors, turns, stop reason, wall time, token usage and cost
- `results.md`: tables by kind, by category and by task
- `transcripts/<task>.json`: the full log of each task

`results/` is git-ignored because transcripts are large.
Commit a run deliberately (`git add -f`) when the report cites it.

## Running it

The harness needs the environment from `uv sync --extra gpu --extra mcp`.

The oracle needs no API key. It plays each reference spec over MCP and must pass every task:

```bash
uv run --no-sync python benchmarks/agents/run.py --agent oracle
```

To run Claude, set your key in your own shell first. The harness only lets the Anthropic SDK read `ANTHROPIC_API_KEY` from the environment: it never prints, logs or stores it, and never passes it to the MCP server.

```bash
uv run --no-sync --with anthropic python benchmarks/agents/run.py --agent claude
```

Useful options:

| Option | Meaning |
|---|---|
| `--model` | Model to run. Default `claude-opus-5-5`. Prices are known for Opus 5.5 / Opus 5 / Sonnet 5.5 / Haiku 5.5 / Fable 5.1. |
| `--effort` | `low` … `max`. Default `high`. |
| `--only sca-,ls-` | Restrict to task id prefixes. |
| `--category trap,pushback` | Restrict by category. |
| `--kind mode` | Restrict by problem kind. |
| `--max-turns 40` | Model turns per task. |
| `--max-cost-per-task 5` | Stop a task once it has cost more than this many USD. |
| `--task-timeout 3600` | Seconds per task. |
| `--server-cmd "..."` | Server command. Default: the `lightsmith-mcp` of this environment. |
| `--server-env K=V` | Extra environment for the server. Repeatable. |
| `--shared-server` / `--server-per-task` | Share one server across tasks, or start a fresh one per task. |
| `--no-fallbacks` | Turn off the server-side refusal fallback. |
| `--strict` | Exit 1 unless every task passes. |
| `--regrade <dir>` | Grade a finished run again from its saved transcripts with the current graders and task files. No model or server calls. Writes `<dir>-regraded-<timestamp>/` (or `--out`); the original run is left as it is, and `results.json` lists the tasks whose grade changed under `meta.regraded`. |

**Model call settings.**
The Claude adapter uses adaptive thinking with summarised thinking display, `output_config.effort`, streaming, and automatic prompt caching (top-level `cache_control`).
It also enables server-side refusal fallbacks (`fallbacks: "default"`).
When a fallback model serves a turn, the transcript's `notes` record it, and that task did not run purely on the named model.
Use `--no-fallbacks` to measure one model strictly.

### Servers

The harness only needs some server that speaks MCP over stdio.
- By default it starts `lightsmith-mcp`.
- `--server-cmd` points it at any other command, such as a development checkout of the MCP rewrite.

Main still has the original six-tool server, which imports `FastMCP` from the MCP SDK 1.x.
`uv.lock` pins MCP 2.3.0, where that class was renamed `MCPServer`, so `lightsmith-mcp` on main does not start until the rewrite lands.
In the meantime, `legacy_server.py` runs it unchanged by aliasing the old name and keeping tool error messages visible:

```bash
uv run --no-sync python benchmarks/agents/run.py --agent oracle --server-cmd "python benchmarks/agents/legacy_server.py"
```

With a server that already runs on SDK 2.x, the launcher just starts it.

The harness has also been run against the in-progress rewrite (jobs, resources, `get_result`).
With `--server-env LIGHTSMITH_MCP_INLINE_WAIT_S=2`, long solves return a job ID, and the oracle follows it to the full result.
11 tasks were run this way and all passed.

## Cost of a Claude run (estimate)

These figures are an estimate from the task shapes, not a measurement; no paid run has been made yet.
The adapter records actual token usage and cost per task in `results.json`.

Assumptions for one task with `claude-opus-5-5` at effort `high`:
- 6–12 model turns
- about 5–15k tokens of tool definitions and schema at the start (the full JSON Schema of every kind is ~47k characters ≈ 12k tokens; the rewrite serves it per kind)
- a context that grows to 20–40k tokens

With caching, the input side is about $0.03 in cache reads plus about $0.10–0.20 in cache writes ($5/MTok).
Output, including thinking, is about 5–15k tokens at $20/MTok, so $0.10–0.30.

| Run | Estimated cost |
|---|---|
| One task | about $0.25–0.50 |
| All 44 tasks, effort `high` | about **$10–25** |
| All 44 tasks, effort `medium` | roughly half |
| All 44 tasks, `claude-sonnet-5-5` | roughly half |

`--max-cost-per-task` caps runaway tasks at $5 each by default.

Wall time is dominated by the solves: a few minutes for most tasks.
The default-settings 2D RCWA metasurface takes minutes; the reference uses Laurent's rule for speed.

## Tests

```bash
uv run --no-sync pytest benchmarks/agents/tests
```

The tests start the server through `legacy_server.py`, which works with both server generations.
Two environment variables change that:
- `AGENTBENCH_SERVER_CMD=lightsmith-mcp` tests the installed server directly.
- `AGENTBENCH_TASKS=sca-,ls-` restricts the end-to-end runs.

The tests check:
- The task files match `taskdefs.py`.
- The graders work on hand-made transcripts. These cover old and new argument names, job-wrapped results, numbers given as strings, pushback and mentions.
- The oracle passes all 44 tasks over stdio MCP, with no failed tool call.
- The Claude adapter's loop works against a scripted stand-in for the API, with no key and no cost. This covers tools, thinking blocks passed back, cache and effort settings, and cost accounting.
- Deliberately wrong behaviour fails:
  - numbers off by 25%
  - skipped estimates
  - dropped concerns
  - no answer
  - the right numbers without any simulation
  - every naive spec, solved over MCP
  - answering the unphysical requests, over MCP

## Limitations

- Keyword `mentions` checks can be fooled by an agent that hedges with every keyword, and can miss an unusual phrasing. The pushback tasks therefore also require `declined` or no numbers.
- The tolerances assume default-resolution solves. An agent that picks a very coarse grid fails, and that is intended.
- Only one optimisation task: inverse design is graded on self-consistency plus a performance floor, not on a unique reference.
- The tasks are single-turn. The agent cannot ask the user a clarifying question; assumptions go into `concerns`.
