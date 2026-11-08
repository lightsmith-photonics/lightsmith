"""What a backend will discretise for a problem, before any solve: ``lightsmith.discretization(problem, method)``.

Every backend can answer ``Backend.discretization(problem, device)`` with a common envelope

    {"backend", "method", "problem_kind", "layout_kind", "summary": {...}, "payload": {...},
     "estimate": {...}, "warnings": [...], "error": None | str}

``summary`` is an ordered map of labelled key figures; ``payload`` depends on ``layout_kind``:

=====================  =========================================  ==================================================
layout_kind            backends                                   payload
=====================  =========================================  ==================================================
``yee_grid``           FDTD scattering, emission                  node planes per axis, TFSF and flux boxes, CPML,
                                                                  refinement regions, mirror planes
``unit_cell``          FDTD periodic, layer stacks                cell nodes, z planes (source, monitors, stack), CPML
``device_grid``        FDTD and FDFD devices, inverse design      node planes, PML, ports (and design regions)
``dipole_lattice``     DDA                                        occupied dipole positions, spacing
``surface_mesh``       BEM                                        vertices and triangles
``multipoles``         Mie, T-matrix                              multipole order per wavelength or per particle
``harmonics``          RCWA                                       Fourier orders, layers (each uniform in z)
``cross_section_grid`` FDE                                        node planes, PML, boundary
``cells``              EME (modes per cell), BPM (steps per cell)  cells along the propagation axis
``plane_waves``        PWE                                        k-path, plane waves per polarisation
``none``               TMM, quasistatic, circuits                 a note; the summary holds the estimate's figures
=====================  =========================================  ==================================================

The functions below call each backend's own planning code (the code ``estimate`` and ``solve`` call), never a
solver. Lengths are nm in world coordinates. Large arrays are capped: node planes are per axis, never per cell;
dipole positions and triangles are subsampled evenly beyond ``MAX_POINTS`` / ``MAX_TRIS``, with the true counts and
the ``stride`` reported.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

MAX_POINTS = 20000
MAX_TRIS = 40000
_ROUND = 6


def _r(x, nd: int = _ROUND) -> list:
    return [round(float(v), nd) for v in np.asarray(x, dtype=float).ravel()]


def _cpu():
    from .device import resolve_device

    return resolve_device("cpu")


def _done(kind: str, summary: dict, payload: dict, warnings=(), error: str | None = None) -> dict:
    return {"layout_kind": kind, "summary": summary, "payload": payload, "warnings": list(warnings), "error": error}


# --- FDTD ---------------------------------------------------------------------------------------------------------


def _box(lay, name: str, label: str, lo, hi) -> dict:
    return {"name": name, "label": label, "lo_nm": _r(lay.world(np.asarray(lo, dtype=float))),
            "hi_nm": _r(lay.world(np.asarray(hi, dtype=float)))}


def fdtd_open(problem, backend) -> dict:
    """FDTD scattering and emission: the whole grid (before mirror halving), boxes, CPML, regions, mirrors."""
    from .backends._fdtd import emission, scattering, symmetry
    from .backends._fdtd.setup import graded_layout, scattering_layout
    from .problems import EmissionProblem

    opts = backend._options(problem)
    warnings: list[str] = []
    error = None
    regions, n_override = [], 0
    if isinstance(problem, EmissionProblem):
        p = emission.plan(problem, opts, single=False, warnings=warnings)
        lay = p["layout"]
        sym = {"mirrors": {}, "reduction": 1, "skipped": "emission runs use the whole grid"}
        boxes = [_box(lay, "flux_radiated", "radiated-power box", *lay.sca)]
    else:
        mirror, sym = symmetry.find(problem, opts)
        n_override = len(opts.mesh_override)
        if scattering.graded(opts):
            why = scattering.graded_unsupported(problem)
            if why:
                error = f"FDTD: {why}; use grid='uniform' without mesh_override"
            base, regions = scattering.grid_regions(problem, opts)
            lay = graded_layout(problem, opts, base, regions, scattering._bbox(problem), mirror, opts.grading_ratio)
        else:
            lay = scattering_layout(problem, opts, scattering.spacing(problem, opts), scattering._bbox(problem))
        boxes = [_box(lay, "tfsf", "TFSF source box", *lay.tfsf),
                 _box(lay, "flux_total", "total-field flux box (C_abs)", *lay.tot),
                 _box(lay, "flux_scattered", "scattered-field flux box (C_sca, far field)", *lay.sca)]
    n_auto = len(regions) - n_override
    regs = []
    for i, r in enumerate(regions):
        src = {"source": "auto", "particle": i} if i < n_auto else {"source": "mesh_override", "index": i - n_auto}
        regs.append({**src, "lo_nm": _r(r.lo), "hi_nm": _r(r.hi), "spacing_nm": _r(r.spacing)})
    if lay.grid is not None:
        nodes = [_r(lay.grid.nodes[a]) for a in range(3)]
    else:
        nodes = [_r(lay.origin[a] + np.arange(lay.shape[a]) * lay.dx) for a in range(3)]
    widths = [np.diff(n) for n in nodes]
    grid = {"kind": "graded" if lay.grid is not None else "uniform", "base_spacing_nm": float(lay.dx),
            "shape": [int(v) for v in lay.shape], "cells": int(lay.cells), "nodes_nm": nodes,
            "spacing_min_nm": [float(w.min()) for w in widths], "spacing_max_nm": [float(w.max()) for w in widths]}
    if lay.grid is not None:
        grid["max_ratio"] = float(lay.grid.max_ratio())
    n = lay.pml
    payload = {"grid": grid, "domain_nm": {"lo_nm": [v[0] for v in nodes], "hi_nm": [v[-1] for v in nodes]},
               "pml": {"cells": int(n), "inner_lo_nm": [v[n] for v in nodes], "inner_hi_nm": [v[-1 - n] for v in nodes]},
               "boxes": boxes, "regions": regs, "symmetry": sym}
    summary = {"grid": f"{grid['kind']} {' × '.join(map(str, grid['shape']))}", "cells": grid["cells"],
               "cell size (nm)": f"{min(grid['spacing_min_nm']):.4g}–{max(grid['spacing_max_nm']):.4g}",
               "CPML cells": int(n),
               "mirror planes": ", ".join(f"{a} {m.split()[0]}" for a, m in sym["mirrors"].items()) or "none"}
    return _done("yee_grid", summary, payload, warnings, error)


def _ports_payload(frames, nodes) -> list[dict]:
    out = []
    for f in frames:
        win = {}
        for t, lo_i, n_i in ((f.t1, f.lo[0], f.n[0]), (f.t2, f.lo[1], f.n[1])):
            win["xyz"[t]] = [nodes[t][lo_i], nodes[t][min(lo_i + n_i, len(nodes[t]) - 1)]]
        out.append({"name": f.name, "axis": "xyz"[f.a], "direction": "+" if f.d > 0 else "-",
                    "source_nm": nodes[f.a][f.k_p], "monitor_nm": nodes[f.a][f.k_m], "window_nm": win,
                    "num_modes": int(f.num_modes)})
    return out


def fdtd_device(problem, backend) -> dict:
    """FDTD devices: the grid (uniform or graded), the PML and the port source and monitor planes."""
    from .backends._fdtd import device as dv

    opts = backend._options(problem)
    p = dv.plan(problem, opts, single=False)
    g, frames = p["grid"], p["frames"]
    nodes = [_r([g.coord(a, i) for i in range(g.shape[a])]) for a in range(3)]
    lo = [nodes[a][g.interior[a][0]] for a in range(3)]
    hi = [nodes[a][g.interior[a][1]] for a in range(3)]
    pol = opts.polarization or "te"
    widths = [np.diff(n) for n in nodes if len(n) > 1]
    summary = {"grid": f"{'graded' if g.graded is not None else 'uniform'} {' × '.join(map(str, g.shape))}",
               "cells": int(g.cells),
               "cell size (nm)": (round(float(g.dx), 4) if g.graded is None else
                                  f"{min(float(w.min()) for w in widths):.4g}–{max(float(w.max()) for w in widths):.4g}"),
               "PML cells": int(g.n_pml), "runs (port modes)": int(p["runs"]),
               "dimension": "2D (z invariant)" if g.two_d else "3D"}
    if g.two_d:
        summary["polarization"] = f"{pol} ({'E in the xy plane' if pol == 'te' else 'E along z'})"
    payload = {"nodes_nm": nodes, "pml_inner_lo_nm": lo, "pml_inner_hi_nm": hi, "ports": _ports_payload(frames, nodes),
               "two_d": g.two_d, "graded": g.graded is not None, "domain_lo_nm": lo, "domain_hi_nm": hi,
               "note": "the domain includes a buffer of cells per side beyond domain_size_nm, so ports may sit on its faces"}
    return _done("device_grid", summary, payload, p["warnings"])


def fdtd_cell(problem, backend) -> dict:
    """FDTD periodic structures and layer stacks: one unit cell, CPML along z, source and monitor planes."""
    from .backends._fdtd import periodic

    opts = backend._options(problem)
    c = periodic.make_cell(problem, opts)
    dx, dy, dz = c.spacing
    xs = _r(-c.periods[0] / 2 + np.arange(c.n_cells[0] + 1) * dx)
    ys = _r(-c.periods[1] / 2 + np.arange(c.n_cells[1] + 1) * dy)
    z = lambda k: round(float((k - c.k_top) * dz), _ROUND)
    zs = [z(k) for k in range(c.shape[2])]
    tops, d = [], 0.0
    for lay in problem.layers:
        tops.append(round(d, _ROUND))
        d += lay.thickness_nm
    planes = {"reflection monitor": z(c.k_r), "source": z(c.k_src), "stack top": 0.0, "stack bottom": round(c.thickness, _ROUND),
              "transmission monitor": z(c.k_t)}
    payload = {"x_nm": xs, "y_nm": ys, "z_nm": zs, "z_axis": "depth below the top of the stack (nm, downwards)",
               "periods_nm": list(c.periods), "n_cells": list(c.n_cells), "spacing_nm": [dx, dy, dz],
               "pml_cells": int(c.pml), "pml_inner_z_nm": [zs[c.pml], zs[-1 - c.pml]], "planes_nm": planes,
               "layer_tops_nm": tops, "layer_thicknesses_nm": [lay.thickness_nm for lay in problem.layers],
               "invariant": [c.n_cells[0] == 1, c.n_cells[1] == 1]}
    summary = {"cell": f"{c.n_cells[0]} × {c.n_cells[1]} × {c.shape[2]}", "cells": int(c.cells),
               "cell size (nm)": f"{dx:.4g} × {dy:.4g} × {dz:.4g}", "CPML cells (z)": int(c.pml),
               "layers": len(problem.layers)}
    return _done("unit_cell", summary, payload)


# --- FDFD and inverse design ------------------------------------------------------------------------------


def fdfd(problem, backend, design_regions=None) -> dict:
    eng = backend.engine(problem)
    g = eng.grid
    axes = [g.x, g.y] + ([g.z] if hasattr(g, "z") else [])
    nodes = [_r(a) for a in axes]
    pml_lo, pml_hi = [], []
    for a in range(len(axes)):
        i0, i1 = g.interior(a)
        pml_lo.append(nodes[a][i0])
        pml_hi.append(nodes[a][max(i0, i1 - 1)])
    ports = []
    for pg in eng.ports:
        ax = pg.axis
        port = {"name": pg.name, "axis": "xyz"[ax], "direction": "+" if pg.d > 0 else "-",
                "source_nm": nodes[ax][pg.i0], "num_modes": int(pg.num_modes)}
        if len(axes) == 2:
            t = 1 - ax
            port["window_nm"] = {"xyz"[t]: [nodes[t][pg.lo], nodes[t][min(pg.hi, len(nodes[t]) - 1)]]}
        elif hasattr(pg, "window"):
            port["window_nm"] = {"window": _r(pg.window)}
        ports.append(port)
    opts = backend._options(problem)
    summary = {"grid": " × ".join(str(len(a)) for a in axes), "cells": int(np.prod([len(a) for a in axes])),
               "cell size (nm)": round(float(g.dl), 4), "PML cells": int(opts.pml_cells),
               "dimension": "3D" if len(axes) == 3 else "2D"}
    if len(axes) == 2:
        summary["polarization"] = opts.polarization
    payload = {"nodes_nm": nodes, "pml_inner_lo_nm": pml_lo, "pml_inner_hi_nm": pml_hi, "ports": ports,
               "two_d": len(axes) == 2}
    if design_regions:
        payload["design_regions"] = design_regions(g)
        summary["design regions"] = ", ".join(f"{d['name']} {' × '.join(map(str, d['shape']))}"
                                              for d in payload["design_regions"])
    return _done("device_grid", summary, payload)


def inverse(problem, backend) -> dict:
    from .backends import get_backend
    from .inverse.backend import device_problem
    from .inverse.parametrization import region_map
    from .problems import wavelength_array

    dev = device_problem(problem)
    fd = get_backend(problem.backend, kind="device")
    if not hasattr(fd, "design_grid"):
        return _done("none", {"discretisation": f"{fd.name} has no design grid to show"}, {})
    wl = float(np.mean(wavelength_array(dev.wavelengths)))

    def regions(grid):
        out = []
        for r in problem.design_regions:
            m = region_map(r.name, r.center_nm, r.size_nm, grid, r.material_low.eps(wl), r.material_high.eps(wl),
                           parametrization=r.parametrization, filter_radius_nm=r.filter_radius_nm, eta=r.eta,
                           min_feature_nm=r.min_feature_nm, symmetry=tuple(r.symmetry))
            out.append({"name": r.name, "shape": [int(s) for s in m.shape], "x_nm": [float(m.x_nm[0]), float(m.x_nm[-1])],
                        "y_nm": [float(m.y_nm[0]), float(m.y_nm[-1])],
                        "filter_radius_cells": round(r.filter_radius_nm / float(m.dl_nm), 3), "pixels": int(np.prod(m.shape))})
        return out

    return fdfd(dev, fd, regions)


# --- particles: DDA, BEM, Mie, T-matrix -------------------------------------------------------------------------


def dda(problem, backend) -> dict:
    from .backends import dda as mod

    opts = backend._options(problem)
    npd = opts.dipoles_per_diameter or mod._default_npd(problem)
    npds = sorted({int(npd), *[int(n) for n in (opts.extrapolate or [])]})
    lat = mod._lattice(problem, max(npds), opts.discretization)
    occ = lat.occupied
    pts = lat.pts[occ]
    slot = np.argmax(lat.frac[:, occ], axis=0) if lat.frac.ndim == 4 else np.zeros(len(pts), int)
    stride = max(1, math.ceil(len(pts) / MAX_POINTS))
    payload = {"spacing_nm": float(lat.d), "dipoles_per_diameter": int(lat.npd), "grid": [int(v) for v in lat.n],
               "n_dipoles": len(pts), "stride": stride, "points_nm": [_r(p, 3) for p in pts[::stride]],
               "material_slot": [int(s) for s in slot[::stride]], "discretization": opts.discretization,
               "extrapolate": npds if len(npds) > 1 else None}
    summary = {"dipoles": len(pts), "spacing (nm)": round(float(lat.d), 4), "dipoles per diameter": int(lat.npd),
               "lattice": " × ".join(str(int(v)) for v in lat.n), "discretization": opts.discretization}
    if len(npds) > 1:
        summary["extrapolation lattices"] = ", ".join(map(str, npds))
    return _done("dipole_lattice", summary, payload)


def bem(problem, backend) -> dict:
    from .backends._bem.mesh import build_mesh

    opts = backend._options(problem)
    h = backend._mesh_size(problem, opts)
    mesh = build_mesh(problem, h, opts.volume_correction)
    tris = np.asarray(mesh.tris)
    stride = max(1, math.ceil(len(tris) / MAX_TRIS))
    payload = {"mesh_size_nm": float(h), "verts_nm": [_r(v, 3) for v in mesh.verts], "tris": tris[::stride].tolist(),
               "tri_surface": np.asarray(mesh.tri_surface)[::stride].tolist(), "n_triangles": len(tris),
               "n_edges": int(mesh.n_edges), "stride": stride, "surfaces": len(mesh.surfaces)}
    summary = {"triangles": len(tris), "edges": int(mesh.n_edges), "unknowns": 2 * int(mesh.n_edges),
               "target element size (nm)": round(float(h), 4), "mean edge (nm)": round(float(mesh.mean_edge_length()), 4),
               "surfaces": len(mesh.surfaces)}
    return _done("surface_mesh", summary, payload)


def mie(problem, backend) -> dict:
    from .backends import mie_native as mn
    from .problems import EmissionProblem

    if isinstance(problem, EmissionProblem):
        return _done("none", {"multipoles": "orders up to convergence at the dipole distance"},
                     {"note": "Mie emission sums the multipoles until the series converges at the dipole"})
    wl = problem.wavelength_array()
    if backend.name == mn.NativeMie.name:
        _, x, m = mn.NativeMie._setup(problem, wl)
        per = [int(mn._n_max([xi[j:j + 1] for xi in x], [mi[j:j + 1] for mi in m])) for j in range(len(wl))]
        fixed = mn.NativeMie._options(problem).n_max
        xs, rule, layers = np.max(np.abs(np.asarray(x)), axis=0), "Wiscombe's rule (layered spheres add margin)", len(x)
    else:  # miepython: its own rule, on the size parameter it computes (x = π d n_medium / λ)
        from miepython.core import wiscombe_terms

        r = problem.particles[0].shape.radius_nm
        xs = 2 * np.pi * r * problem.medium_index / wl
        per = [int(wiscombe_terms(float(v))) for v in xs]
        fixed, rule, layers = None, "miepython's wiscombe_terms", 1
    payload = {"wavelengths_nm": _r(wl, 4), "n_max": per, "size_parameter": _r(xs, 5), "fixed_n_max": fixed, "rule": rule}
    summary = {"multipole order n_max": f"{min(per)}–{max(per)}" if min(per) != max(per) else max(per),
               "rule": rule, "layers": layers, "largest size parameter": round(float(np.max(xs)), 4)}
    if fixed:
        summary["n_max (option)"] = fixed
    return _done("multipoles", summary, payload)


def tmatrix(problem, backend) -> dict:
    from .backends import tmatrix_treams as tt

    opts = backend._options(problem)
    plan = tt._Plan(problem, opts)
    payload = {"lmax": [int(v) for v in plan.lmax], "modes": int(plan.modes),
               "lmax_global": int(plan.lmax_global) if plan.lmax_global is not None else None,
               "positions_nm": [_r(p, 4) for p in np.asarray(plan.positions)],
               "r_circumscribing_nm": float(plan.r_circumscribing)}
    summary = {"lmax per particle": ", ".join(map(str, payload["lmax"])), "modes (unknowns)": payload["modes"]}
    if payload["lmax_global"] is not None:
        summary["lmax (global)"] = payload["lmax_global"]
    return _done("multipoles", summary, payload)


# --- periodic, modes, bands, devices -------------------------------------------------------------------------


def rcwa(problem, backend) -> dict:
    from .backends import rcwa as mod
    from .backends._rcwa.fourier import Lattice, harmonic_set
    from .problems import LayerStackProblem

    opts = backend._options(problem)
    M, N = mod._orders(problem, opts)
    if isinstance(problem, LayerStackProblem):
        dim, layers = 0, [{"thickness_nm": float(lay.thickness_nm), "patterned": False} for lay in problem.layers]
    else:
        dim = Lattice(problem.lattice_a1_nm, problem.lattice_a2_nm).dim
        layers = [{"thickness_nm": float(lay.thickness_nm), "patterned": bool(lay.shapes), "shapes": len(lay.shapes)}
                  for lay in problem.layers]
    h = harmonic_set(max(dim, 1), M, N) if dim else np.zeros((1, 2), int)
    payload = {"orders_max": [M, N], "harmonics": h.tolist(), "dim": dim, "layers": layers,
               "lattice_a1_nm": getattr(problem, "lattice_a1_nm", None),
               "lattice_a2_nm": getattr(problem, "lattice_a2_nm", None),
               "note": "each layer is uniform in z: the layers are the problem's own layers (no automatic slicing)"}
    summary = {"harmonics": len(h), "orders": f"|m| ≤ {M}" + (f", |n| ≤ {N}" if dim == 2 else ""),
               "layers": len(layers), "patterned layers": sum(1 for lay in layers if lay["patterned"])}
    return _done("harmonics", summary, payload)


def _fde_grid_payload(grid) -> dict:
    return {"x_nm": _r(grid.nodes(0, False)), "y_nm": _r(grid.nodes(1, False)), "dx_nm": float(grid.dx),
            "dy_nm": float(grid.dy), "pml_cells": [list(map(int, p)) for p in grid.pml],
            "symmetry": list(grid.symmetry), "bc_min": list(grid.bc_min)}


def fde(problem, backend) -> dict:
    from .backends import fde as mod

    opts = backend._options(problem)
    grid, boundary = mod._grid(problem, opts)
    payload = {**_fde_grid_payload(grid), "boundary": boundary, "num_modes": int(problem.num_modes)}
    summary = {"grid": f"{grid.nx} × {grid.ny}", "cells": int(grid.size),
               "cell size (nm)": f"{grid.dx:.4g} × {grid.dy:.4g}", "boundary": boundary,
               "PML cells": str(grid.pml), "modes": int(problem.num_modes)}
    return _done("cross_section_grid", summary, payload)


def cells(problem, backend) -> dict:
    """EME and BPM: the cells along the propagation axis (exactly the solve's slicing), modes or steps per cell."""
    import sys

    from .backends._eme import slice_device

    mod = sys.modules[backend.__module__]
    p = mod._resolved(problem) if hasattr(mod, "_resolved") else problem
    bpm = backend.method == "bpm"
    if bpm:
        st = backend._setup(p)
        opts, grid, axis, lo, hi = st["opts"], st["grid"], st["axis"], st["lo"], st["hi"]
        cs = slice_device(p.structures, p.background, axis, lo, hi, grid, p.dimension, p.domain_center_nm[2], None, None,
                          opts.subsamples, opts.subpixel, cells_per_step=opts.taper_grid_cells_per_slice,
                          taper_cells_range=(1, opts.max_taper_slices))
        step = float(st["step"])
    else:
        opts, axis, lo, hi, _, grid, _pol, num_modes = backend._setup(p)
        cs = slice_device(p.structures, p.background, axis, lo, hi, grid, p.dimension, p.domain_center_nm[2],
                          opts.taper_cells, opts.max_cell_length_nm, opts.subsamples, opts.subpixel)
    out = []
    for c in cs:
        d = {"start_nm": round(float(c.start), 4), "stop_nm": round(float(c.stop), 4)}
        if bpm:
            n = max(1, math.ceil(c.length / step - 1e-9))
            d.update(steps=int(n), dz_nm=round(float(c.length / n), 4))
        out.append(d)
    payload = {"axis": "xyz"[axis], "lo_nm": float(lo), "hi_nm": float(hi), "cells": out,
               "cross_section": _fde_grid_payload(grid)}
    summary = {"propagation axis": "xyz"[axis], "length (nm)": round(float(hi - lo), 3), "cells": len(out),
               "cross-section grid": f"{grid.nx} × {grid.ny}"}
    if bpm:
        payload["step_nm"] = step
        summary["step (nm)"] = round(step, 4)
        summary["steps"] = sum(c["steps"] for c in out)
    else:
        payload["modes_per_cell"] = int(num_modes)
        summary["modes per cell"] = int(num_modes)
    return _done("cells", summary, payload)


def pwe(problem, backend) -> dict:
    from .backends import pwe as mod
    from .backends._pwe.engine import make_plan

    opts = backend._options(problem)
    warnings: list[str] = []
    cell, _ = mod._cell(problem, opts, warnings)
    red, _cart, dist, labels = mod.k_path(problem)
    per = {}
    for key, pol in mod._pols(problem).items():
        plan = make_plan(cell, pol, opts, _cpu())
        entry = {"solver": plan.solver, "num_plane_waves": int(plan.num_plane_waves),
                 "matrix_size": int(plan.matrix_size), "grid": list(plan.grid) if plan.grid is not None else None}
        if plan.idx is not None:
            entry["g_indices"] = np.asarray(plan.idx).tolist()[:4000]
        per[key] = entry
    payload = {"k_reduced": [_r(k, 5) for k in red], "k_distance": _r(dist, 5),
               "k_labels": {str(k): v for k, v in labels.items()}, "polarizations": per,
               "lattice_vectors_nm": [list(map(float, v)) for v in problem.lattice_vectors_nm[:problem.dimension]]}
    summary = {"k-points": len(red),
               **{f"plane waves ({k})": f"{v['num_plane_waves']} ({v['solver']})" for k, v in per.items()}}
    return _done("plane_waves", summary, payload, warnings)


# --- dispatch ---------------------------------------------------------------------------------------------


NOTHING = {
    "tmm": "the transfer matrices of the problem's layers: exact, nothing to discretise",
    "quasistatic": "closed-form polarisabilities: nothing to discretise",
    "circuit": "the netlist's compact models: nothing to discretise",
}


def plan_for(backend, problem) -> dict | None:
    """The layout of `problem` for `backend` (layout_kind, summary, payload, warnings, error), or None."""
    kind, method = problem.kind, backend.method
    if not backend.__module__.startswith("lightsmith."):  # third-party: override Backend.discretization to answer
        return None
    if method == "fdtd":
        if kind == "device":
            return fdtd_device(problem, backend)
        if kind in ("periodic", "layer_stack"):
            return fdtd_cell(problem, backend)
        return fdtd_open(problem, backend)
    table = {"fdfd": fdfd, "inverse": inverse, "dda": dda, "bem": bem, "mie": mie, "tmatrix": tmatrix, "rcwa": rcwa,
             "fde": fde, "eme": cells, "bpm": cells, "pwe": pwe}
    fn = table.get(method)
    return fn(problem, backend) if fn else None


def _jsonable(obj: Any):
    return json.loads(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def discretization(problem, method: str = "auto") -> dict[str, Any]:
    """What the backend `method` (as solve resolves it) will discretise for `problem`. Never solves.

    Returns the envelope described in this module's docstring. The library's refusal of a problem goes to
    ``error`` while the geometry is still returned where it can be computed."""
    from . import api
    from .problems import parse_problem

    problem = parse_problem(problem)
    backend = api._resolve(problem, method)
    out: dict[str, Any] = {"backend": backend.name, "method": backend.method, "problem_kind": problem.kind,
                           "layout_kind": "none", "summary": {}, "payload": {}, "estimate": None, "warnings": [],
                           "error": None}
    try:
        out["estimate"] = _jsonable(api.estimate(problem, method=backend.name, device="cpu"))
    except Exception as e:  # noqa: BLE001 (the library's reason is what the caller needs)
        out["error"] = f"{type(e).__name__}: {e}"
    try:
        planned = backend.discretization(problem, _cpu())
    except Exception as e:  # noqa: BLE001
        planned, out["error"] = None, out["error"] or f"{type(e).__name__}: {e}"
    if planned:
        err = planned.pop("error", None)
        out.update(planned)
        out["error"] = err or out["error"]
    else:
        note = NOTHING.get(backend.method, "no layout for this backend; the summary holds its estimate")
        skip = {"backend", "method", "framework", "device", "gpu_name", "warnings", "notes", "fits"}
        figures = {k: v for k, v in (out["estimate"] or {}).items()
                   if k not in skip and (isinstance(v, (int, float, str)) or (
                       isinstance(v, list) and len(v) <= 4 and all(isinstance(x, (int, float)) for x in v)))}
        out["summary"] = {"discretisation": note, **figures}
        out["payload"] = {"note": note}
    out["warnings"] = list(dict.fromkeys([*out["warnings"], *((out["estimate"] or {}).get("warnings") or [])]))
    return _jsonable(out)
