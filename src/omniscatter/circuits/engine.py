"""S-matrix netlist solver.

Each instance contributes a block S_k[w, out, in] (from :mod:`.models`). All
instance ports are split into external ports (e) and internal, connected ports
(i). Connections are a symmetric permutation C on the internal ports (outgoing
wave of one port = incoming wave of its partner), so with incoming waves a and
outgoing waves b = S a:

    b_i = (I - S_ii C)^{-1} S_ie a_e
    S_ext = S_ee + S_ei C (I - S_ii C)^{-1} S_ie

which is solved for all wavelengths at once with one batched linear solve
(``algorithm="dense"``). C is never formed: multiplying by C permutes columns.
For large netlists ``algorithm="reduce"`` instead merges instances one
connection at a time with the two-port "innerconnect" formula (sub-circuit
reduction, the formula of scikit-rf's ``innerconnect``), whose cost grows
with the size of the merged blocks rather than the whole netlist.

Everything is written against the Python array API, so the same code runs on
NumPy, PyTorch (CPU/CUDA) or CuPy, and with PyTorch gradients flow from S_ext
back to any model setting passed as a tensor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError

from ..device import Device, resolve_device
from ..problems import CircuitProblem, parse_problem, wavelength_array
from .models import Ctx, get_model


class NetlistError(ValueError):
    """The netlist is inconsistent (unknown/duplicate/unconnected ports, unknown models, bad settings)."""


def split_ref(ref: str) -> tuple[str, str]:
    inst, sep, port = str(ref).partition(",")
    if not sep or not inst.strip() or not port.strip():
        raise NetlistError(f"port reference {ref!r} must look like 'instance,port'")
    return inst.strip(), port.strip()


@dataclass
class Block:
    """One evaluated instance (or a merged group of instances)."""

    ports: list[str]  # 'instance,port' labels
    s: Any  # (W, P, P)


@dataclass
class CircuitSolution:
    ports: list[str]  # external port names, order of S
    wavelengths_nm: np.ndarray
    s: Any  # (W, E, E) array in the device's namespace (a torch tensor with gradients, if requested)
    device: Device
    info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def numpy(self) -> np.ndarray:
        return Device.to_numpy(self.s)

    def __getitem__(self, key: tuple[str, str]):
        """``sol["out", "in"]`` -> (W,) transmission amplitude from external port `in` to `out`."""
        o, i = key
        return self.s[:, self.ports.index(o), self.ports.index(i)]


# --- evaluation ----------------------------------------------------------------


def _settings_values(inst_name: str, component: str, raw: dict[str, Any], overrides: dict[str, Any]):
    model = get_model(component)
    unknown = set(overrides) - set(model.settings.model_fields)
    if unknown:
        raise NetlistError(f"instance {inst_name!r} ({component}): unknown parameter(s) {sorted(unknown)}; "
                           f"settings are {sorted(model.settings.model_fields)}")
    placeholder = {k: (float(v.detach().cpu().reshape(-1)[0]) if hasattr(v, "detach") else
                       float(np.asarray(v).reshape(-1)[0]) if hasattr(v, "shape") else v)
                   for k, v in overrides.items()}
    try:
        values = model.settings.model_validate({**raw, **placeholder}).model_dump()
    except ValidationError as e:
        why = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'settings'}: {err['msg']}" for err in e.errors())
        raise NetlistError(f"instance {inst_name!r} ({component}): invalid settings: {why}") from None
    values.update(overrides)  # keep tensors (and their gradients)
    return model, values


def evaluate_instances(problem: CircuitProblem, ctx: Ctx, params: dict[str, dict[str, Any]] | None = None,
                       base_dir: str | Path | None = None) -> dict[str, Block]:
    """Evaluate every instance's S-matrix on ctx's wavelengths."""
    params = params or {}
    unknown = set(params) - set(problem.instances)
    if unknown:
        raise NetlistError(f"params given for unknown instance(s) {sorted(unknown)}; "
                           f"instances are {sorted(problem.instances)}")
    blocks: dict[str, Block] = {}
    errors = []
    for name, inst in problem.instances.items():
        if "," in name:
            errors.append(f"instance name {name!r} must not contain ','")
            continue
        component = inst.component
        if component.startswith("sparams:") and base_dir is not None:
            p = Path(component[len("sparams:"):].strip()).expanduser()
            if not p.is_absolute():
                component = "sparams:" + str(Path(base_dir) / p)
        try:
            model, values = _settings_values(name, component, dict(inst.settings), params.get(name, {}))
        except KeyError as e:
            errors.append(f"instance {name!r}: {e.args[0]}")
            continue
        except NetlistError as e:
            errors.append(str(e))
            continue
        try:
            ports, s = model.evaluate(ctx, values)
        except (FileNotFoundError, ValueError) as e:
            errors.append(f"instance {name!r} ({inst.component}): {e}")
            continue
        blocks[name] = Block([f"{name},{p}" for p in ports], s)
    if errors:
        raise NetlistError("; ".join(errors))
    return blocks


# --- netlist checks ----------------------------------------------------------------


def check_netlist(problem: CircuitProblem, ports_of: dict[str, list[str]], unconnected: str = "error"
                  ) -> tuple[list[tuple[str, str]], dict[str, str], list[str], list[str]]:
    """Validate references; return (connections, external map, ignored ports, warnings).

    Raises :class:`NetlistError` naming every unknown, duplicate or unconnected port."""
    errors: list[str] = []
    uses: dict[str, list[str]] = {}

    def ref(r: str, where: str) -> str | None:
        try:
            inst, port = split_ref(r)
        except NetlistError as e:
            errors.append(f"{where}: {e}")
            return None
        if inst not in ports_of:
            errors.append(f"{where}: unknown instance {inst!r} (instances: {', '.join(sorted(ports_of))})")
            return None
        if port not in ports_of[inst]:
            comp = problem.instances[inst].component
            errors.append(f"{where}: instance {inst!r} ({comp}) has no port {port!r} "
                          f"(ports: {', '.join(ports_of[inst])})")
            return None
        key = f"{inst},{port}"
        uses.setdefault(key, []).append(where)
        return key

    conns = []
    for n, (a, b) in enumerate(problem.connections):
        where = f"connection #{n} ({a!r} <-> {b!r})"
        ra, rb = ref(a, where), ref(b, where)
        if ra is not None and ra == rb:
            errors.append(f"{where}: a port cannot connect to itself")
            continue
        if ra is not None and rb is not None:
            conns.append((ra, rb))
    external = {}
    for name, r in problem.ports.items():
        k = ref(r, f"external port {name!r}")
        if k is not None:
            external[name] = k
    dup = {k: w for k, w in uses.items() if len(w) > 1}
    if dup:
        errors.append("duplicate port use (each port connects once, or is external once): " + "; ".join(
            f"{k!r} in {' and '.join(w)}" for k, w in dup.items()))
    all_ports = [f"{i},{p}" for i, ps in ports_of.items() for p in ps]
    loose = [p for p in all_ports if p not in uses]
    warnings: list[str] = []
    if loose and unconnected == "error":
        errors.append(f"unconnected port(s): {', '.join(repr(p) for p in loose)}. Connect them, expose them in "
                      "'ports', end them with a 'terminator' instance, or set options.circuit.unconnected='ignore' "
                      "to treat them as matched (non-reflecting) ends")
    elif loose:
        warnings.append(f"unconnected port(s) treated as matched ends: {', '.join(loose)}")
    if not problem.ports:
        errors.append("the circuit has no external ports")
    if errors:
        raise NetlistError("invalid netlist: " + "; ".join(errors))
    return conns, external, loose, warnings


# --- interconnection -------------------------------------------------------------


def _block_diag(xp, blocks: list[Block], like) -> Block:
    if len(blocks) == 1:
        return blocks[0]
    sizes = [len(b.ports) for b in blocks]
    n = sum(sizes)
    W = blocks[0].s.shape[0]
    s = xp.zeros((W, n, n), dtype=like.dtype, device=like.device)
    o = 0
    for b, m in zip(blocks, sizes):
        s[:, o:o + m, o:o + m] = b.s
        o += m
    return Block([p for b in blocks for p in b.ports], s)


def _dense(xp, blocks: list[Block], conns, ext_refs: list[str], like):
    full = _block_diag(xp, blocks, like)
    pos = {p: i for i, p in enumerate(full.ports)}
    e = [pos[r] for r in ext_refs]
    i_ports = [p for c in conns for p in c]
    i = [pos[p] for p in i_ports]
    ipos = {p: k for k, p in enumerate(i_ports)}
    partner = [0] * len(i_ports)
    for a, b in conns:
        partner[ipos[a]], partner[ipos[b]] = ipos[b], ipos[a]
    dev = full.s.device

    def sub(rows, cols):
        r = xp.asarray(rows, dtype=xp.int64, device=dev)
        c = xp.asarray(cols, dtype=xp.int64, device=dev)
        return xp.take(xp.take(full.s, r, axis=1), c, axis=2)

    see = sub(e, e)
    if not i:
        return see
    perm = [i[k] for k in partner]  # columns of S_.i C
    sii_c, sei_c, sie = sub(i, perm), sub(e, perm), sub(i, e)
    eye = xp.eye(len(i), dtype=like.dtype, device=dev)
    x = xp.linalg.solve(eye[None, ...] - sii_c, sie)
    return see + xp.matmul(sei_c, x)


def _innerconnect(xp, s, k: int, l: int):
    """Connect ports k and l of S[w, out, in] (scikit-rf ``innerconnect_s``); returns S without them."""
    skk, sll, skl, slk = s[:, k, k], s[:, l, l], s[:, k, l], s[:, l, k]
    den = (1 - skl) * (1 - slk) - skk * sll
    row_k, row_l = s[:, k, :][:, None, :], s[:, l, :][:, None, :]  # S_kj, S_lj
    col_k, col_l = s[:, :, k][:, :, None], s[:, :, l][:, :, None]  # S_ik, S_il
    num = (row_k * col_l * (1 - slk)[:, None, None] + row_l * col_k * (1 - skl)[:, None, None]
           + row_k * col_k * sll[:, None, None] + row_l * col_l * skk[:, None, None])
    out = s + num / den[:, None, None]
    keep = [m for m in range(s.shape[1]) if m not in (k, l)]
    idx = xp.asarray(keep, dtype=xp.int64, device=s.device)
    return xp.take(xp.take(out, idx, axis=1), idx, axis=2)


def _reduce(xp, blocks: dict[str, Block], conns, ext_refs: list[str], like):
    group_of: dict[str, int] = {}
    groups: dict[int, Block] = {}
    for g, b in enumerate(blocks.values()):
        groups[g] = b
        for p in b.ports:
            group_of[p] = g
    for a, b in conns:
        ga, gb = group_of[a], group_of[b]
        if ga != gb:
            merged = _block_diag(xp, [groups[ga], groups[gb]], like)
            del groups[gb]
            for p in merged.ports:
                group_of[p] = ga
            groups[ga] = merged
        g = groups[ga]
        k, l_ = g.ports.index(a), g.ports.index(b)
        groups[ga] = Block([p for m, p in enumerate(g.ports) if m not in (k, l_)], _innerconnect(xp, g.s, k, l_))
    keep = [g for g in groups.values() if any(p in ext_refs for p in g.ports)]
    full = _block_diag(xp, keep, like)
    pos = {p: i for i, p in enumerate(full.ports)}
    idx = xp.asarray([pos[r] for r in ext_refs], dtype=xp.int64, device=full.s.device)
    return xp.take(xp.take(full.s, idx, axis=1), idx, axis=2)


def interconnect(blocks: dict[str, Block], conns: list[tuple[str, str]], ext_refs: list[str], ctx: Ctx,
                 algorithm: Literal["auto", "dense", "reduce"] = "auto") -> tuple[Any, str]:
    """External S-matrix of the connected blocks; returns (S_ext, algorithm used)."""
    xp = ctx.xp
    like = ctx.zeros()
    # Drop instances that touch neither an external port nor a connection (e.g. all-ignored ports).
    used = {p for c in conns for p in c} | set(ext_refs)
    blocks = {n: b for n, b in blocks.items() if any(p in used for p in b.ports)}
    # Ports that are neither connected nor external are matched ends: remove their rows/columns.
    trimmed = {}
    for n, b in blocks.items():
        keep = [m for m, p in enumerate(b.ports) if p in used]
        if len(keep) < len(b.ports):
            idx = xp.asarray(keep, dtype=xp.int64, device=b.s.device)
            b = Block([b.ports[m] for m in keep], xp.take(xp.take(b.s, idx, axis=1), idx, axis=2))
        trimmed[n] = b
    if algorithm == "auto":
        algorithm = "dense" if 2 * len(conns) <= 600 else "reduce"
    if algorithm == "dense":
        return _dense(xp, list(trimmed.values()), conns, ext_refs, like), "dense"
    return _reduce(xp, trimmed, conns, ext_refs, like), "reduce"


# --- top-level ----------------------------------------------------------------------


def circuit_sparameters(problem: CircuitProblem | dict | str, *, params: dict[str, dict[str, Any]] | None = None,
                        device: str | Device = "cpu", wavelengths_nm=None,
                        algorithm: Literal["auto", "dense", "reduce"] = "auto",
                        precision: Literal["double", "single"] = "double",
                        unconnected: Literal["error", "ignore"] = "error",
                        base_dir: str | Path | None = None) -> CircuitSolution:
    """Solve a circuit and return its external S-matrix as an array on `device`.

    ``params`` overrides instance settings, e.g. ``{"ring": {"neff": t}}``; with
    ``device="torch-cpu"`` (or a CUDA device) and ``t`` a tensor with
    ``requires_grad=True``, gradients flow from the result back to ``t``.
    ``wavelengths_nm`` overrides the problem's wavelengths.
    """
    t0 = time.perf_counter()
    problem = parse_problem(problem)
    if not isinstance(problem, CircuitProblem):
        raise TypeError(f"expected a circuit problem, got kind {getattr(problem, 'kind', '?')!r}")
    dev = device if isinstance(device, Device) else resolve_device(device)
    wl = np.asarray(wavelengths_nm if wavelengths_nm is not None else wavelength_array(problem.wavelengths),
                    dtype=float).reshape(-1)
    if wl.size == 0 or np.any(wl <= 0):
        raise ValueError("wavelengths must be positive and non-empty")
    ctx = Ctx(wl, dev, single=precision == "single")
    blocks = evaluate_instances(problem, ctx, params, base_dir)
    ports_of = {n: [p.split(",", 1)[1] for p in b.ports] for n, b in blocks.items()}
    conns, external, loose, warnings = check_netlist(problem, ports_of, unconnected)
    names = list(external)
    s, used = interconnect(blocks, conns, [external[n] for n in names], ctx, algorithm)
    info = {"instances": len(blocks), "instance_ports": sum(len(b.ports) for b in blocks.values()),
            "internal_ports": 2 * len(conns), "external_ports": len(names), "algorithm": used,
            "precision": precision, "unconnected_ports": loose, "runtime_s": time.perf_counter() - t0,
            "models": sorted({i.component for i in problem.instances.values()})}
    return CircuitSolution(names, wl, s, dev, info, warnings)
