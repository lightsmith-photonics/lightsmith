# MCP server

`lightsmith-mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server. It lets AI agents
build, check, solve and compare Lightsmith problems through the same JSON interface as the Python API and
the [CLI](cli.md).

```bash
uv add "lightsmith[mcp]"
lightsmith-mcp            # stdio transport
```

To use it, register the `lightsmith-mcp` command as a stdio server in your MCP client.

## What the server offers

The MCP server is being expanded, so this page describes it only in general terms. The full reference
will be on its own page when that work is merged. In this release the server covers:

- **Discovery**: the JSON Schema of every problem kind, the installed backends, and the CPU and GPU
  environment.
- **Pre-flight**: estimates of size, device and memory, without solving.
- **Solving**: one method, or several methods with relative differences against a reference.

Results are the same JSON models the Python API returns, including `provenance` and `warnings`.

## Recommended agent workflow

1. Read the schema before building a problem.
2. Use nm and vacuum wavelengths, with `Im(ε) > 0` for loss (see [Conventions](conventions.md)).
3. Estimate before solving grid-based problems.
4. Solve, then check `warnings` before reporting numbers.
5. Cross-check important results with a second method.

The repository's [AGENTS.md](https://github.com/lightsmith-photonics/lightsmith/blob/main/AGENTS.md) and
[llms.txt](https://github.com/lightsmith-photonics/lightsmith/blob/main/llms.txt) give the same guidance in
compact form.
