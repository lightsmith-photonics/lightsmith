"""Periodic structures and planar stacks at normal incidence: R, T, A and diffraction orders.

The unit cell is the computational domain: x and y are periodic (ghost planes, real fields, so
normal incidence only), z ends in CPML. Cells may be non-cubic so that each period holds a whole
number of cells. A one-sided TFSF plane in the superstrate launches the pulse; everything above
it is the reflected field alone. Running DFTs of the tangential fields on a plane above the source
(reflection) and just below the stack (transmission), split into diffraction orders by a 2D FFT
over the cell, give per-order efficiencies; their sum is the total flux.

Geometry (the conventions of omniscatter.problems): layers stacked along +z from the superstrate
side, the first layer's top face at z = 0; the unit cell spans [-P/2, P/2) along each lattice axis;
s polarisation has E along (-sin phi, cos phi, 0), p along (cos phi, sin phi, 0).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ...device import Device
from ...problems import LayerStackProblem, PeriodicProblem, wavelength_array
from ...raster import inside_profile
from ...results import DiffractionOrder, SpectrumResult
from .engine import DFTMonitor, Engine, PMLSpec, flat_index
from .media import Medium, to_medium
from .options import FDTDOptions
from .setup import FS, run_until_decayed, voxelise
from .tfsf import TFSFPlaneWave

BIG = 10**9


# --- support checks -------------------------------------------------------------


def _const_real_eps(material, wl) -> float | None:
    try:
        e = np.asarray(material.eps(wl), dtype=complex)
    except Exception:  # noqa: BLE001
        return None
    if np.all(np.abs(e.imag) < 1e-12) and np.ptp(e.real) < 1e-9 * max(1.0, abs(e.real[0])) and e.real[0] >= 1:
        return float(e.real[0])
    return None


def supports(problem) -> tuple[bool, str]:
    wl = wavelength_array(problem.wavelengths)
    if _const_real_eps(problem.superstrate, wl) is None:
        return False, ("FDTD needs a lossless, non-dispersive superstrate (constant real eps >= 1) "
                       "for its plane-wave source")
    if isinstance(problem, LayerStackProblem):
        if any(abs(a) > 1e-12 for a in problem.angles_deg):
            return False, "FDTD solves layer stacks at normal incidence only (angles_deg = [0])"
        if any(not lay.coherent for lay in problem.layers):
            return False, "FDTD treats every layer coherently; use TMM for incoherent layers"
        return True, ""
    if abs(problem.incidence.theta_deg) > 1e-12:
        return False, "FDTD periodic support is normal incidence only (theta_deg = 0); use 'rcwa' for oblique"
    a1, a2 = np.asarray(problem.lattice_a1_nm, float), problem.lattice_a2_nm
    if abs(a1[1]) > 1e-9 * abs(a1[0]) or a1[0] <= 0:
        return False, "FDTD needs lattice_a1_nm along +x, e.g. (period, 0)"
    if a2 is not None and (abs(a2[0]) > 1e-9 * abs(a2[1]) or a2[1] <= 0):
        return False, "FDTD needs a rectangular lattice: lattice_a2_nm along +y, e.g. (0, period)"
    return True, ""


# --- set-up ------------------------------------------------------------------------


@dataclass
class _Cell:
    periods: tuple[float, float]  # nm; for a 1D grating / layer stack the cell width is nominal
    n_cells: tuple[int, int]  # real cells along x, y
    spacing: tuple[float, float, float]
    shape: tuple[int, int, int]  # including ghosts (x, y) and PEC planes (z)
    origin: np.ndarray
    k_r: int  # reflection plane (E nodes)
    k_src: int  # TFSF plane
    k_top: int  # node index of z = 0
    k_t: int  # transmission plane
    thickness: float
    pml: int

    @property
    def cells(self) -> int:
        return int(np.prod(self.shape))


def _materials(problem) -> list:
    out = [problem.superstrate, problem.substrate]
    for lay in problem.layers:
        if isinstance(problem, LayerStackProblem):
            out.append(lay.material)
        else:
            out.append(lay.background)
            out.extend(s.material for s in lay.shapes)
    return out


def _band(wl):
    om = 2 * np.pi / np.asarray(wl, float)
    omega_c = 0.5 * (om.min() + om.max())
    return omega_c, max(0.5 * (om.max() - om.min()), 0.15 * omega_c), om.max()


def make_cell(problem, opts: FDTDOptions) -> _Cell:
    wl = wavelength_array(problem.wavelengths)
    n_max = 1.0
    for m in _materials(problem):
        try:
            n_max = max(n_max, float(np.abs(np.sqrt(np.asarray(m.eps(wl), dtype=complex))).max()))
        except Exception:  # noqa: BLE001 (range errors surface later, with context)
            pass
    d0 = opts.spacing_nm or wl.min() / (n_max * opts.cells_per_wavelength)
    thick = [lay.thickness_nm for lay in problem.layers if lay.thickness_nm > 0]
    dz = min(d0, min(thick) / 2) if (thick and not opts.spacing_nm) else d0
    periods, ncell = [d0, d0], [1, 1]
    if isinstance(problem, PeriodicProblem):
        periods[0] = float(problem.lattice_a1_nm[0])
        ncell[0] = max(1, math.ceil(periods[0] / d0 - 1e-9))
        if problem.lattice_a2_nm is not None:
            periods[1] = float(problem.lattice_a2_nm[1])
            ncell[1] = max(1, math.ceil(periods[1] / d0 - 1e-9))
    dx, dy = periods[0] / ncell[0], periods[1] / ncell[1]
    H = float(sum(lay.thickness_nm for lay in problem.layers))
    n_pad = max(4, math.ceil((opts.padding_nm or 0) / dz))
    gap = opts.pml_gap_nm if opts.pml_gap_nm is not None else _default_gap(problem, wl, periods, ncell)
    n_gap = max(4, math.ceil(gap / dz))
    k_r = opts.pml_cells + n_gap
    k_src = k_r + 2
    k_top = k_src + n_pad
    k_bot = k_top + math.ceil(H / dz - 1e-9)
    k_t = k_bot + 3
    nz = k_t + n_gap + opts.pml_cells + 1
    shape = (ncell[0] + 2, ncell[1] + 2, nz)
    origin = np.array([-periods[0] / 2 - dx, -periods[1] / 2 - dy, -k_top * dz])
    return _Cell((periods[0], periods[1]), (ncell[0], ncell[1]), (dx, dy, dz), shape, origin, k_r, k_src, k_top,
                 k_t, H, opts.pml_cells)


def _default_gap(problem, wl, periods, ncell) -> float:
    """Monitor-to-PML distance: 3 decay lengths of the slowest evanescent diffraction order in the
    band (CPML does not absorb evanescent waves), clipped to [0.25, 1.5] x the longest wavelength."""
    eps = [np.real(np.asarray(m.eps(wl), dtype=complex)) for m in (problem.superstrate, problem.substrate)]
    lam_med = wl.max() / math.sqrt(max(1.0, float(np.min(eps[0]))))
    k0 = 2 * np.pi / wl
    kappa_min = np.inf
    for m in range(-ncell[0] // 2, ncell[0] // 2 + 1):
        for n in range(-ncell[1] // 2, ncell[1] // 2 + 1):
            if not (m or n):
                continue
            g2 = (2 * np.pi * m / periods[0]) ** 2 + (2 * np.pi * n / periods[1]) ** 2
            for e in eps:
                q2 = g2 - np.maximum(e, 0) * k0**2
                if np.any(q2 > 0):
                    kappa_min = min(kappa_min, float(np.sqrt(q2[q2 > 0].min())))
    if not np.isfinite(kappa_min):
        return 0.25 * lam_med
    return float(np.clip(3 / kappa_min, 0.25 * lam_med, 1.5 * wl.max()))


def _slot_fn(problem, slot_of_layer, slot_sup, slot_sub, cell: _Cell):
    """pts -> medium index. Patterned layers: later shapes override earlier ones (periodic images included)."""
    px, py = cell.periods
    tops = np.concatenate([[0.0], np.cumsum([lay.thickness_nm for lay in problem.layers])])
    one_d = isinstance(problem, LayerStackProblem) or problem.lattice_a2_nm is None
    shifts = [(sx, sy) for sx in (-px, 0.0, px) for sy in ((0.0,) if one_d else (-py, 0.0, py))]

    def fn(pts):
        x = (pts[..., 0] + px / 2) % px - px / 2
        # 1D gratings are invariant along y: profiles are evaluated on the line y = 0 (as RCWA does)
        y = np.zeros_like(pts[..., 1]) if one_d else (pts[..., 1] + py / 2) % py - py / 2
        z = pts[..., 2]
        out = np.full(z.shape, slot_sup, dtype=int)
        out[z >= tops[-1]] = slot_sub
        for i, layer in enumerate(problem.layers):
            sel = (z >= tops[i]) & (z < tops[i + 1])
            if not sel.any():
                continue
            bg, shapes = slot_of_layer[i]
            vals = np.full(int(sel.sum()), bg, dtype=int)
            xs, ys = x[sel], y[sel]
            for prof, s in shapes:
                hit = np.zeros(xs.shape, dtype=bool)
                for sx, sy in shifts:
                    hit |= inside_profile(prof, xs - sx, ys - sy)
                vals[hit] = s
            out[sel] = vals
        return out
    return fn


def _media(problem, wl, warnings):
    eps_sup = _const_real_eps(problem.superstrate, wl)
    media = [Medium(eps_sup, (), "superstrate")]
    keys = {problem.superstrate.model_dump_json(): 0}

    def slot(m):
        k = m.model_dump_json()
        if k not in keys:
            keys[k] = len(media)
            media.append(to_medium(m, wl, warnings))
        return keys[k]

    layers = []
    for lay in problem.layers:
        if isinstance(problem, LayerStackProblem):
            layers.append((slot(lay.material), []))
        else:
            layers.append((slot(lay.background), [(s.profile, slot(s.material)) for s in lay.shapes]))
    return media, layers, slot(problem.superstrate), slot(problem.substrate)


def plan(problem, opts: FDTDOptions, single: bool) -> dict:
    wl = wavelength_array(problem.wavelengths)
    cell = make_cell(problem, opts)
    media = [Medium(1.0)]
    for m in _materials(problem):
        try:
            e = np.asarray(m.eps(wl), dtype=complex)
            media.append(Medium(float(max(1.0, e.real.max()))))  # eps_inf >= 1 for every ADE medium used
        except Exception:  # noqa: BLE001
            pass
    dt = _time_step(cell, opts)
    omega_c, omega_half, omega_max = _band(wl)
    stride = opts.dft_stride or max(1, int(2 * np.pi / (omega_c + 4 * omega_half + omega_max) / dt))
    max_t = (opts.run_time_fs or opts.max_time_fs) * FS
    max_steps = int(math.ceil(max_t / dt / stride)) * stride
    n_pts = 2 * 2 * cell.n_cells[0] * cell.n_cells[1]
    mem = (4 if single else 8) * (12 * cell.cells + 4 * n_pts * len(wl))
    return {"cell": cell, "dt": dt, "stride": stride, "max_steps": max_steps, "memory_bytes": int(mem),
            "omega_c": omega_c, "omega_half": omega_half}


def _time_step(cell: _Cell, opts) -> float:
    inv2 = 1 / cell.spacing[2] ** 2
    for a in (0, 1):
        if cell.n_cells[a] > 1:
            inv2 += 1 / cell.spacing[a] ** 2
    return opts.courant / math.sqrt(inv2)


def _plane(cell: _Cell, k: int):
    """Flat indices for S_z = Ex Hy - Ey Hx on node plane k over the real cells (term 1, then term 2)."""
    i = np.arange(1, cell.n_cells[0] + 1)
    j = np.arange(1, cell.n_cells[1] + 1)
    gi, gj = np.meshgrid(i, j, indexing="ij")
    gi, gj = gi.ravel(), gj.ravel()
    e = np.concatenate([flat_index(0, gi, gj, k, cell.shape), flat_index(1, gi, gj, k, cell.shape)])
    h1 = np.concatenate([flat_index(1, gi, gj, k - 1, cell.shape), flat_index(0, gi, gj, k - 1, cell.shape)])
    h2 = np.concatenate([flat_index(1, gi, gj, k, cell.shape), flat_index(0, gi, gj, k, cell.shape)])
    return e, h1, h2


def _orders(e: np.ndarray, h: np.ndarray, cell: _Cell):
    """Per-order z flux (unnormalised): dict (m, n) -> (n_f,) of 0.5 Re(E1 H1* - E2 H2*) summed over the cell."""
    nx, ny = cell.n_cells
    n = nx * ny
    nf = e.shape[0]
    E1, E2 = e[:, :n].reshape(nf, nx, ny), e[:, n:].reshape(nf, nx, ny)
    H1, H2 = h[:, :n].reshape(nf, nx, ny), h[:, n:].reshape(nf, nx, ny)
    f = lambda a: np.fft.fft2(a, axes=(1, 2))  # noqa: E731
    flux = 0.5 * np.real(f(E1) * np.conj(f(H1)) - f(E2) * np.conj(f(H2))) / n * cell.spacing[0] * cell.spacing[1]
    mx = np.fft.fftfreq(nx, 1 / nx).astype(int)
    my = np.fft.fftfreq(ny, 1 / ny).astype(int)
    return flux, mx, my


# --- solve -----------------------------------------------------------------------------


def solve(problem, opts: FDTDOptions, device: Device) -> SpectrumResult:
    t_start = time.perf_counter()
    ok, why = supports(problem)
    if not ok:
        raise ValueError(why)
    warnings: list[str] = []
    wl = wavelength_array(problem.wavelengths)
    single = opts.precision == "single" or (opts.precision == "auto" and device.is_gpu)
    media, layer_slots, s_sup, s_sub = _media(problem, wl, warnings)
    p = plan(problem, opts, single)
    cell, stride = p["cell"], p["stride"]
    dt = _time_step(cell, opts) * math.sqrt(min([1.0] + [m.eps_inf for m in media]))
    max_steps = int(math.ceil((opts.run_time_fs or opts.max_time_fs) * FS / dt / stride)) * stride
    if cell.cells > opts.max_cells:
        raise ValueError(f"grid {cell.shape} has {cell.cells} cells > max_cells={opts.max_cells}")
    lam_max = wl.max()

    t0 = time.perf_counter()
    roi_lo = np.array([1, 1, max(1, cell.k_top - 3)])
    roi_hi = np.array([cell.shape[0] - 1, cell.shape[1] - 1, cell.shape[2] - 1])
    inv_eps, groups, vstats = voxelise(_slot_fn(problem, layer_slots, s_sup, s_sub, cell), media, cell.shape,
                                       cell.origin, cell.spacing, roi_lo, roi_hi, subpixel=opts.subpixel,
                                       samples=opts.subpixel_samples)
    t_vox = time.perf_counter() - t0

    eps_sup = media[0].eps_inf
    om_c = p["omega_c"]
    eps_sub_ref = float(np.real(media[s_sub].eps(np.array([om_c]))[0]))
    pml = PMLSpec(cells=((0, 0), (0, 0), (cell.pml, cell.pml)), kappa_max=opts.pml_kappa,
                  alpha_max=0.05 * 2 * np.pi * math.sqrt(eps_sup) / lam_max, eps_ref=eps_sup,
                  eps_ref_high=max(1.0, eps_sub_ref))
    if isinstance(problem, LayerStackProblem):
        pols, phi, angles = list(problem.polarizations), 0.0, list(problem.angles_deg)
    else:
        pols, phi, angles = list(problem.incidence.polarizations), np.radians(problem.incidence.phi_deg), [0.0]
    # a uniform stack at normal incidence is polarisation independent: one run serves both
    runs = {pol: pol for pol in pols}
    if isinstance(problem, LayerStackProblem) and pols:
        runs = {pol: pols[0] for pol in pols}

    omega = 2 * np.pi / wl
    area = cell.n_cells[0] * cell.spacing[0] * cell.n_cells[1] * cell.spacing[1]
    out: dict[str, dict] = {}
    prov_runs = []
    r0 = cell.origin + np.array([1, 1, cell.k_top]) * np.asarray(cell.spacing)
    for pol in dict.fromkeys(runs.values()):
        ehat = np.array([-np.sin(phi), np.cos(phi), 0.0]) if pol == "s" else np.array([np.cos(phi), np.sin(phi), 0.0])
        src = TFSFPlaneWave(shape=cell.shape, origin=cell.origin, dx=cell.spacing, dt=dt, r0=r0,
                            lo=np.array([-BIG, -BIG, cell.k_src]), hi=np.array([BIG, BIG, BIG]),
                            khat=np.array([0.0, 0.0, 1.0]), ehat=ehat, eps_b=eps_sup, omega_c=om_c,
                            omega_half=p["omega_half"], max_steps=max_steps + 4 * stride, periodic=(True, True, False))
        mr, mt = _plane(cell, cell.k_r), _plane(cell, cell.k_t)
        mon_r = DFTMonitor(omega, *mr, energy=True)
        mon_t = DFTMonitor(omega, *mt, energy=True)
        mon_i = DFTMonitor(omega, *src.probe(), source=src)
        eng = Engine(device, shape=cell.shape, dx=cell.spacing, dt=dt, single=single, inv_eps=inv_eps,
                     groups=groups, pml=pml, sources=[src], monitors=[mon_r, mon_t, mon_i], dft_stride=stride,
                     periodic=(True, True, False))
        transit = cell.shape[2] * cell.spacing[2] * max(math.sqrt(abs(m.eps_inf)) for m in media)
        t_loop = time.perf_counter()
        steps, level, graphs = run_until_decayed(
            eng, opts, max_steps=max_steps, min_steps=int((src.t_passed + transit) / dt),
            window=math.ceil(lam_max / (stride * dt)), warnings=warnings)
        t_loop = time.perf_counter() - t_loop
        e_i, h_i = mon_i.results()
        inc = 0.5 * np.real(e_i[:, 0] * np.conj(h_i[:, 0])) * area
        fr, mx, my = _orders(*mon_r.results(), cell)
        ft, _, _ = _orders(*mon_t.results(), cell)
        out[pol] = {"r_ord": -fr / inc[:, None, None], "t_ord": ft / inc[:, None, None]}
        prov_runs.append({"polarization": pol, "steps": steps, "decay_level": level, "time_loop_s": t_loop,
                          "cuda_graphs": graphs})
        del eng

    # orders propagating in the superstrate or substrate at some wavelength
    k0 = omega
    eps_sub_band = np.real(np.asarray(problem.substrate.eps(wl), dtype=complex))
    n2 = np.maximum(eps_sup, eps_sub_band)
    gx, gy = 2 * np.pi / cell.periods[0], 2 * np.pi / cell.periods[1]
    one_d = isinstance(problem, LayerStackProblem) or problem.lattice_a2_nm is None
    orders = []
    for ii, m in enumerate(mx):
        for jj, n in enumerate(my):
            if isinstance(problem, LayerStackProblem) and (m or n):
                continue
            if one_d and n:
                continue
            kpar2 = (m * gx) ** 2 + (n * gy) ** 2
            if np.any(kpar2 < n2 * k0**2):
                orders.append(((int(m), int(n)), ii, jj))
    # evanescent orders carry no power; what flux the DFT finds in them is PML/near-field noise
    for o in out.values():
        for (m, n), ii, jj in orders:
            kpar2 = (m * gx) ** 2 + (n * gy) ** 2
            o["r_ord"][kpar2 >= eps_sup * k0**2, ii, jj] = 0.0
            o["t_ord"][kpar2 >= eps_sub_band * k0**2, ii, jj] = 0.0
    r_tot, t_tot, a_tot = {}, {}, {}
    for pol in pols:
        o = out[runs[pol]]
        R = sum(o["r_ord"][:, ii, jj] for _, ii, jj in orders) if orders else np.zeros(len(wl))
        T = sum(o["t_ord"][:, ii, jj] for _, ii, jj in orders) if orders else np.zeros(len(wl))
        r_tot[pol] = [R.tolist()] * len(angles)
        t_tot[pol] = [T.tolist()] * len(angles)
        a_tot[pol] = [(1 - R - T).tolist()] * len(angles)
    diff_orders = None
    if isinstance(problem, PeriodicProblem):
        diff_orders = [DiffractionOrder(order=mn, r={pol: [out[runs[pol]]["r_ord"][:, ii, jj].tolist()] for pol in pols},
                                        t={pol: [out[runs[pol]]["t_ord"][:, ii, jj].tolist()] for pol in pols})
                       for mn, ii, jj in orders]
    lossless = all(np.all(np.abs(np.imag(np.asarray(m.eps(wl), dtype=complex))) < 1e-12) for m in _materials(problem))
    if lossless:
        a_max = max(float(np.max(np.abs(a_tot[pol][0]))) for pol in pols) if pols else 0.0
        if a_max > 1e-2:
            warnings.append(f"energy balance off by {a_max:.1e} for a lossless structure; refine the grid "
                            "or lower decay_by")
    prov = {**device.describe(), "grid": list(cell.shape), "real_cells": list(cell.n_cells) + [cell.shape[2]],
            "cells": cell.cells, "spacing_nm": list(cell.spacing), "dt_fs": dt / FS, "dft_stride": stride,
            "precision": "single" if single else "double", "subpixel": opts.subpixel, "pml_cells": cell.pml,
            "runs": prov_runs, "steps": sum(r["steps"] for r in prov_runs),
            "decay_level": max(r["decay_level"] for r in prov_runs) if prov_runs else None,
            "materials": [m.describe() for m in media], **vstats, "voxelise_s": t_vox,
            "runtime_s": time.perf_counter() - t_start,
            "notes": "normal incidence; R and T are fluxes on planes above the source and below the stack"}
    return SpectrumResult(method="fdtd", backend="omniscatter-fdtd", wavelengths_nm=wl.tolist(), angles_deg=angles,
                          polarizations=pols, r=r_tot, t=t_tot, a=a_tot, orders=diff_orders, provenance=prov,
                          warnings=warnings)
