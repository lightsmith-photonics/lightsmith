"""Objective terms on a :class:`DifferentiableSolution` (torch scalars to maximise)."""

from __future__ import annotations

import numpy as np

from .spec import FieldIntensityTerm, SParameterTerm


def port_label(sol_labels: list[str], port: str, mode: int) -> str:
    if port in sol_labels and mode == 0:
        return port
    lab = f"{port}@{mode}"
    if lab not in sol_labels:
        raise KeyError(f"port mode {lab!r} not found; available: {sol_labels}")
    return lab


def required_sources(terms, labels: list[str]) -> list[str]:
    """Port-mode labels that must be excited to evaluate `terms`."""
    out: list[str] = []
    for t in terms:
        lab = (port_label(labels, t.in_port, t.in_mode) if isinstance(t, SParameterTerm)
               else port_label(labels, t.source_port, t.source_mode))
        if lab not in out:
            out.append(lab)
    return out


def _aggregate(values, term):
    import torch

    if term.aggregate == "mean":
        if term.wavelength_weights is not None:
            w = torch.as_tensor(term.wavelength_weights, dtype=values.dtype, device=values.device)
            if len(w) != len(values):
                raise ValueError("wavelength_weights must have one entry per wavelength")
            return (w * values).sum() / w.sum()
        return values.mean()
    # Smooth min/max (log-sum-exp with a sharpness that keeps it within ~1% of the hard value).
    s = 50.0
    if term.aggregate == "min":
        return -torch.logsumexp(-s * values, 0) / s + np.log(len(values)) / s
    return torch.logsumexp(s * values, 0) / s - np.log(len(values)) / s


def term_values(term, sol):
    """Per-wavelength values (n_wl,) of one term."""
    if isinstance(term, SParameterTerm):
        s = sol.sparam(port_label(sol.outputs, term.out_port, term.out_mode),
                       port_label(sol.outputs, term.in_port, term.in_mode))
        return {"power": s.abs() ** 2, "real": s.real, "imag": s.imag}[term.quantity]
    if isinstance(term, FieldIntensityTerm):
        if sol.fields is None:
            raise ValueError("field objectives need with_fields=True")
        src = sol.inputs.index(port_label(sol.outputs, term.source_port, term.source_mode))
        sl = []
        axes = (sol.grid.x, sol.grid.y, sol.grid.z) if sol.fields.ndim == 6 else (sol.grid.x, sol.grid.y)
        for a, coords in enumerate(axes):
            c, h = term.center_nm[a], term.size_nm[a] / 2
            idx = np.nonzero(np.abs(coords - c) <= h + 1e-9)[0]
            if len(idx) == 0:
                idx = np.array([int(np.argmin(np.abs(coords - c)))])
            sl.append(slice(int(idx[0]), int(idx[-1]) + 1))
        if sol.fields.ndim == 6:  # 3D: (n_wl, n_src, 3, Nx, Ny, Nz), intensity |E|^2 summed over components
            f = sol.fields[:, src, :, sl[0], sl[1], sl[2]]
            return (f.abs() ** 2).sum(1).reshape(f.shape[0], -1).mean(-1)
        f = sol.fields[:, src, sl[0], sl[1]]
        return (f.abs() ** 2).reshape(f.shape[0], -1).mean(-1)
    raise TypeError(f"unknown objective term {type(term).__name__}")


def evaluate(terms, sol):
    """(total to maximise, {term_i: aggregated value}) as torch scalars."""
    total = 0.0
    parts = {}
    for i, t in enumerate(terms):
        v = _aggregate(term_values(t, sol), t)
        parts[f"{i}:{t.kind}"] = v
        if t.goal == "maximize":
            total = total + t.weight * v
        elif t.goal == "minimize":
            total = total - t.weight * v
        else:
            total = total - t.weight * (v - t.target) ** 2
    return total, parts
