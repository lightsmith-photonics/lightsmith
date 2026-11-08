# Agent benchmark

The agent benchmark (`benchmarks/agents/`) asks whether an AI agent can answer a photonics question,
written the way a user would write it, by using the [MCP server](../mcp.md) to set up and solve the right
simulation. An automatic grader compares the answer with a reference result and also inspects the tool
calls: which problem was actually solved, whether `estimate` ran before an expensive solve, and whether the
agent pushed back on requests that cannot be answered as asked.

!!! note "No model results yet"
    The repository contains the tasks, the graders, the harness and its tests. No run of a language model
    has been committed, so there are no agent scores to report. Runs are written to
    `benchmarks/agents/results/`, which is git-ignored because transcripts are large.

## Tasks

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

They fall into four categories:

- **Standard** (23): a well-posed question, such as the plasmon peak of a gold sphere in water, an
  antireflection coating, a Bragg mirror, grating efficiencies, waveguide modes, a 2D MMI, the TM gap of a
  photonic crystal, the FSR of an MZI or a ring, or a small topology optimisation.
- **Trap** (16): answerable, but the obvious setup is wrong. Examples: a diameter given where the schema
  wants a radius; an emitter's distance given from a particle's surface; a wavelength given in the medium; energies, wavenumbers, frequencies or µm instead of
  nm; a complex index in the e^{+jωt} convention; light incident from the substrate; a 1 mm slide that must
  be an incoherent layer; group index versus effective index; normalised frequency a/λ; a material
  dataset that does not cover the wavelength; cm versus nm in a link budget.
- **Pushback** (3): the request is unphysical, and the agent should decline and explain why: a
  "lossy" index that is gain in the library's convention, an effective index above the core index, and a
  lossless 1×2 splitter with 70 % in each arm.
- **Warning** (2): the tools warn about the answer and the agent must pass the warning on.

11 tasks use grid-based solvers (DDA, RCWA, EME, PWE, inverse design) and are also graded on whether the
agent estimated before solving.

Each task file in `benchmarks/agents/tasks/` holds the prompt, the answer fields with units, a hidden
reference problem, its reference values, the targets with tolerances, and a "naive" problem showing what a
careless agent would solve. Reference values come from the installed library at the resolution in
`taskdefs.py`. The tolerances are physics tolerances: a sensible agent at default resolution should land
inside them, and the naive setup should not. `generate.py` rewrites the task files after a library change,
and `generate.py --check` recomputes them without writing.

## Protocol

1. **Isolation.** By default each task gets a fresh server with its own `LIGHTSMITH_MCP_DATA_DIR`, so
   jobs from earlier tasks are invisible. Cloud and API-key variables are removed from the server's
   environment, so solves stay local.
2. **What the agent sees.** Every tool the server lists, with its own description and schema; the
   server's instructions; helper tools to list and read resources; a `submit_answer` tool; a short
   system prompt; and the task prompt with the list of answer keys and units. The system prompt says the
   user will not answer follow-up questions. It does not mention traps, estimates or pushback.
3. **Answer.** The agent calls `submit_answer` with `values` (numbers in the stated units), `declined`,
   `concerns` and a `summary`.
4. **Transcript.** Every MCP call is logged with its arguments, reply, error and timing, together with the
   model's messages.

## Scoring

A task **passes** when every required check passes. The **score** is the fraction of all checks that pass,
which gives partial credit.

| Check | Passes when |
|---|---|
| `value:<key>` | The reported number is within abs + rel × \|reference\| of the reference value. |
| `solved_kind` | At least one solve of a problem of the task's kind succeeded, so a correct number without a simulation does not pass. |
| `spec:<path>`, `material:<path>`, `wavelength:<λ>` | Some successfully solved problem has this field value, material index or wavelength. These are read from the tool-call arguments, not from the agent's claims, and show *why* a value check failed. |
| `estimate_before_solve` | An `estimate` for a problem of this kind came before its first successful solve. |
| `pushback` | `declined` is true, or no values were given and concerns were stated. |
| `mentions` | The concerns or summary contain one of the expected words. |
| `self_consistent`, `threshold` | Optimisation task: the reported transmission matches the agent's own result, and is at least 0.5. |

The graders depend only on the task file, the submitted answer and the logged tool calls. They classify
tools by name (any tool containing `estimate`; any containing `solve`, `compare`, `optimiz` or `run`) and
accept a problem passed under any argument name, so they work with any server that speaks MCP.

## Running it

From a clone, in the environment of `uv sync --extra gpu --extra mcp`. The oracle is a scripted agent that
plays the reference problems over MCP; it needs no API key and must pass every task:

```bash
uv run --no-sync python benchmarks/agents/run.py --agent oracle
```

The Claude agent reads `ANTHROPIC_API_KEY` from your shell through the Anthropic SDK. The harness never
prints, logs or stores the key, and never passes it to the MCP server:

```bash
uv run --no-sync --with anthropic python benchmarks/agents/run.py --agent claude
```

Useful options: `--model`, `--effort`, `--only <task id prefixes>`, `--category`, `--kind`,
`--max-turns`, `--max-cost-per-task` (USD), `--task-timeout`, `--server-cmd`, `--server-env K=V`,
`--shared-server` / `--server-per-task`, and `--strict` (exit 1 unless every task passes). Each run writes
`results.json` (metadata and one row per task with pass, score, failed checks, tool calls, turns, wall
time, token usage and cost), `results.md` and one transcript per task. `benchmarks/agents/README.md`
describes every option.

## Tests

```bash
uv run --no-sync pytest benchmarks/agents/tests
```

The tests check that the task files match `taskdefs.py`, that the graders work on hand-made transcripts,
that the oracle passes all 44 tasks over stdio MCP with no failed tool call, and that the Claude adapter's
loop works against a scripted stand-in for the API (no key, no cost). They also check that deliberately
wrong behaviour fails: numbers off by 25 %, skipped estimates, dropped concerns, no answer, the right
numbers without a simulation, every naive problem, and answers to the unphysical requests.

## Limitations

- Keyword `mentions` checks can be fooled by an agent that hedges with every keyword, and can miss an
  unusual phrasing; the pushback tasks therefore also require `declined` or no numbers.
- The tolerances assume default-resolution solves: an agent that picks a very coarse grid fails, by
  design.
- There is only one optimisation task, graded on self-consistency and a performance floor rather than a
  unique reference.
- The tasks are single-turn: the agent cannot ask a clarifying question, so assumptions go into
  `concerns`.
