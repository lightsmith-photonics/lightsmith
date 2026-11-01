"""Plane-wave scattering by particles: set-up, run and cross sections."""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from ...device import Device
from ...results import NearFieldData
from .engine import DFTMonitor, Engine, PMLSpec, flat_index
from .media import Medium, to_medium
from .options import FDTDOptions
from .setup import FS, flux, flux_points, particle_slot_fn, scattering_layout, time_step, voxelise
from .tfsf import TFSFPlaneWave


def _media(sim, warnings) -> tuple[list[Medium], list[list[int]]]:
    wl = sim.wavelength_array()
    if sim.outputs.near_field is not None:
        wl = np.concatenate([wl, sim.outputs.near_field.wavelengths_nm])
    media = [Medium(sim.medium_index**2, (), "medium")]
    keys: dict[str, int] = {}
    slot_of = []
    for p in sim.particles:
        slots = []
        for m in p.region_materials():
            key = m.model_dump_json()
            if key not in keys:
                keys[key] = len(media)
                media.append(to_medium(m, wl, warnings))
            slots.append(keys[key])
        slot_of.append(slots)
    return media, slot_of


def spacing(sim, opts: FDTDOptions) -> float:
    if opts.spacing_nm:
        return float(opts.spacing_nm)
    wl = sim.wavelength_array()
    n_max, metal = sim.medium_index, False
    for p in sim.particles:
        for m in p.region_materials():
            try:
                eps = m.eps(wl)
            except Exception:  # noqa: BLE001 (range errors surface later, with context)
                continue
            n_max = max(n_max, float(np.abs(np.sqrt(eps.astype(complex))).max()))
            metal |= bool(np.any(eps.real < 0))
    cpr = opts.cells_per_radius or (20.0 if metal else 10.0)
    feature = min(float(np.min(p.shape.bounding_half_extent())) for p in sim.particles)
    return min(wl.min() / (n_max * opts.cells_per_wavelength), feature / cpr)


def _band(sim):
    wl = sim.wavelength_array()
    if sim.outputs.near_field is not None:
        wl = np.concatenate([wl, sim.outputs.near_field.wavelengths_nm])
    om = 2 * np.pi / wl
    omega_c = 0.5 * (om.min() + om.max())
    omega_half = max(0.5 * (om.max() - om.min()), 0.15 * omega_c)
    return omega_c, omega_half, om.max()


def plan(sim, opts: FDTDOptions, single: bool, warnings: list[str] | None = None, media=None):
    """Everything about the run that is cheap to compute (used by estimate and solve)."""
    warnings = [] if warnings is None else warnings
    dx = spacing(sim, opts)
    lay = scattering_layout(sim, opts, dx)
    if media is None:
        media, _ = _media(sim, warnings)
    dt = time_step(dx, opts.courant, media)
    omega_c, omega_half, omega_max = _band(sim)
    # sampling every `stride` steps aliases source content above omega_c + 4 omega_half (amplitude
    # < e^-16) onto the highest requested frequency, never below it
    stride = opts.dft_stride or max(1, int(2 * np.pi / (omega_c + 4 * omega_half + omega_max) / dt))
    max_t = (opts.run_time_fs or opts.max_time_fs) * FS
    max_steps = int(math.ceil(max_t / dt / stride)) * stride
    # rough run-length guess: pulse + transit + slowest material damping
    tau = 2 / omega_half
    t_pulse = 7 * tau + (lay.shape[0] * dx) * sim.medium_index
    gammas = [p.gamma for m in media for p in m.poles if p.sigma > 0]
    t_ring = math.log(1 / opts.decay_by) / max(min(gammas), 1e-6) if gammas else 4 * t_pulse / 7
    est_steps = max_steps if opts.run_time_fs else min(max_steps, int((t_pulse + t_ring) / dt))
    n_cells = lay.cells
    bytes_ = 4 if single else 8
    n_flux = 2 * 12 * int(np.prod(lay.sca[1] - lay.sca[0] + 1) ** (2 / 3))
    n_f = len(sim.wavelength_array())
    mem = bytes_ * (12 * n_cells + 4 * n_flux * n_f)
    return {"dx": dx, "layout": lay, "dt": dt, "stride": stride, "max_steps": max_steps, "est_steps": est_steps,
            "omega_c": omega_c, "omega_half": omega_half, "media": media, "memory_bytes": int(mem),
            "warnings": warnings}


def solve(sim, opts: FDTDOptions, device: Device):
    t_start = time.perf_counter()
    warnings: list[str] = []
    single = opts.precision == "single" or (opts.precision == "auto" and device.is_gpu)
    media, slot_of = _media(sim, warnings)
    p = plan(sim, opts, single, warnings, media)
    lay, dx, dt, stride = p["layout"], p["dx"], p["dt"], p["stride"]
    if lay.cells > opts.max_cells:
        raise ValueError(f"grid {lay.shape} has {lay.cells} cells > max_cells={opts.max_cells}; "
                         "increase spacing_nm or raise max_cells")
    khat, ehat = sim.illumination.unit_vectors()
    if int(np.sum(np.abs(khat) > 1e-12)) != 1:
        warnings.append("oblique incidence: TFSF source is interpolated; expect ~0.1% incident-field leakage")
    eps_b = sim.medium_index**2

    t0 = time.perf_counter()
    inv_eps, groups, vstats = voxelise(particle_slot_fn(sim, slot_of), media, lay.shape, lay.origin, dx,
                                       lay.roi[0], lay.roi[1], subpixel=opts.subpixel,
                                       samples=opts.subpixel_samples)
    t_vox = time.perf_counter() - t0

    src = TFSFPlaneWave(shape=lay.shape, origin=lay.origin, dx=dx, dt=dt, r0=lay.r0, lo=lay.tfsf[0],
                        hi=lay.tfsf[1], khat=khat, ehat=ehat, eps_b=eps_b, omega_c=p["omega_c"],
                        omega_half=p["omega_half"], max_steps=p["max_steps"] + 4 * stride)
    wl = sim.wavelength_array()
    omega = 2 * np.pi / wl
    sca = flux_points(*lay.sca, lay.shape, dx)
    tot = flux_points(*lay.tot, lay.shape, dx)
    mon_sca = DFTMonitor(omega, sca[0], sca[1], sca[2], energy=True)
    mon_tot = DFTMonitor(omega, tot[0], tot[1], tot[2], energy=True)
    pe, ph1, ph2 = src.probe()
    mon_inc = DFTMonitor(omega, pe, ph1, ph2, source=src)
    monitors = [mon_sca, mon_tot, mon_inc]
    near = _near_monitor(sim, lay) if sim.outputs.near_field is not None else None
    if near is not None:
        near["inc"] = DFTMonitor(near["omega"], pe, ph1, ph2, source=src)
        monitors += [near["mon"], near["inc"]]

    lam_max = wl.max()
    pml = PMLSpec(cells=((lay.pml, lay.pml),) * 3, kappa_max=opts.pml_kappa,
                  alpha_max=0.05 * 2 * np.pi * sim.medium_index / lam_max, eps_ref=eps_b)
    eng = Engine(device, shape=lay.shape, dx=dx, dt=dt, single=single, inv_eps=inv_eps, groups=groups, pml=pml,
                 sources=[src], monitors=monitors, dft_stride=stride)
    del inv_eps
    run, graphs = eng.make_runner(opts.cuda_graphs)

    # time loop with decay shutoff
    min_steps = int(src.t_passed / dt)
    window = deque(maxlen=max(3, int(math.ceil(2 * np.pi / (2 * np.pi / lam_max) / (stride * dt)))))
    peak, level, steps = 0.0, float("nan"), 0
    t_loop = time.perf_counter()
    try:
        while steps < p["max_steps"]:
            run()
            steps += stride
            e = float(Device.to_numpy(eng.energy)[0])  # one sync per chunk
            if not np.isfinite(e):
                raise FloatingPointError(f"FDTD diverged after {steps} steps (non-finite fields); "
                                         "lower courant or check the material model")
            window.append(e)
            peak = max(peak, e)
            level = max(window) / peak if peak > 0 else float("nan")
            if (not opts.run_time_fs and steps >= min_steps and len(window) == window.maxlen
                    and level < opts.decay_by):
                break
        else:
            if not opts.run_time_fs:
                warnings.append(f"fields had decayed only to {level:.1e} of peak at the time cap "
                                f"({opts.max_time_fs} fs); raise max_time_fs or decay_by")
    finally:
        eng.release()
    t_loop = time.perf_counter() - t_loop

    # cross sections
    e_inc, h_inc = mon_inc.results()
    intensity = 0.5 * np.real(e_inc[:, 0] * np.conj(h_inc[:, 0]))
    es, hs = mon_sca.results()
    et, ht = mon_tot.results()
    c_sca = flux(es, hs, sca[3]) / intensity
    c_abs = -flux(et, ht, tot[3]) / intensity
    near_data = _near_results(sim, lay, near) if near is not None else None

    prov = {**device.describe(), "grid": list(lay.shape), "cells": lay.cells, "spacing_nm": dx, "dt_fs": dt / FS,
            "steps": steps, "simulated_time_fs": steps * dt / FS, "dft_stride": stride,
            "precision": "single" if single else "double", "cuda_graphs": graphs, "subpixel": opts.subpixel,
            "pml_cells": lay.pml, "tfsf_box_nm": float((lay.tfsf[1] - lay.tfsf[0])[0] * dx),
            "decay_level": level, "materials": [m.describe() for m in media], **vstats,
            "runtime_s": time.perf_counter() - t_start, "time_loop_s": t_loop, "voxelise_s": t_vox}
    if np.any(c_abs < -1e-3 * np.max(np.abs(c_sca + c_abs))):
        warnings.append("negative absorption beyond 0.1% of extinction: increase resolution or decay time")
    return {"c_sca": c_sca, "c_abs": c_abs, "c_ext": c_sca + c_abs, "near_field": near_data,
            "provenance": prov, "warnings": warnings}


# --- near fields -------------------------------------------------------------------

_PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def _near_monitor(sim, lay):
    req = sim.outputs.near_field
    a1, a2 = _PLANES[req.plane]
    nrm = 3 - a1 - a2
    lo_w = lay.origin
    rng = []
    for a in range(3):
        if a == nrm:
            lo_v, hi_v = req.offset_nm - lay.dx, req.offset_nm + lay.dx
        else:
            lo_v, hi_v = -req.half_width_nm - lay.dx, req.half_width_nm + lay.dx
        i0 = max(1, int(math.floor((lo_v - lo_w[a]) / lay.dx)) - 1)
        i1 = min(lay.shape[a] - 2, int(math.ceil((hi_v - lo_w[a]) / lay.dx)) + 1)
        if i1 <= i0:
            raise ValueError("near_field plane lies outside the FDTD grid; reduce half_width_nm or offset_nm")
        rng.append(np.arange(i0, i1 + 1))
    I, J, K = np.meshgrid(*rng, indexing="ij")
    idx = np.concatenate([flat_index(c, I, J, K, lay.shape).ravel() for c in range(3)])
    om = 2 * np.pi / np.asarray(req.wavelengths_nm, dtype=float)
    return {"mon": DFTMonitor(om, idx), "inc": None, "ranges": rng, "omega": om}


def _near_results(sim, lay, near):
    from scipy.interpolate import RegularGridInterpolator

    req = sim.outputs.near_field
    a1, a2 = _PLANES[req.plane]
    nrm = 3 - a1 - a2
    rng = near["ranges"]
    om = near["omega"]
    e_all, _ = near["mon"].results()
    shp = tuple(len(r) for r in rng)
    e_all = e_all.reshape(len(om), 3, *shp)
    u = np.linspace(-req.half_width_nm, req.half_width_nm, req.points)
    g1, g2 = np.meshgrid(u, u, indexing="ij")
    pts = np.zeros(g1.shape + (3,))
    pts[..., a1], pts[..., a2], pts[..., nrm] = g1, g2, req.offset_nm
    khat, ehat = sim.illumination.unit_vectors()
    lo, hi = lay.tfsf
    out = []
    inc0 = near["inc"].results()[0][:, 0]
    for f, lam in enumerate(req.wavelengths_nm):
        k = sim.medium_index * om[f]
        comps = []
        for c in range(3):
            off = np.zeros(3)
            off[c] = 0.5
            axes = [lay.origin[a] + (rng[a] + off[a]) * lay.dx for a in range(3)]
            P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
            U = np.meshgrid(*[rng[a] + off[a] for a in range(3)], indexing="ij")
            inside = np.all([(U[a] >= lo[a]) & (U[a] <= hi[a]) for a in range(3)], axis=0)
            vals = e_all[f, c] + np.where(inside, 0.0, ehat[c] * inc0[f] *
                                          np.exp(1j * k * ((P - lay.r0) @ khat)))
            interp = RegularGridInterpolator(axes, vals, bounds_error=False, fill_value=np.nan)
            comps.append(interp(pts))
        e = np.stack(comps, axis=-1) * np.exp(1j * k * (lay.r0 @ khat)) / inc0[f]
        enh = np.sum(np.abs(e) ** 2, axis=-1)
        out.append(NearFieldData(
            wavelength_nm=float(lam), plane=req.plane, offset_nm=req.offset_nm, axis1_nm=u.tolist(),
            axis2_nm=u.tolist(), enhancement=enh.tolist(),
            e_real=np.moveaxis(e.real, -1, 0).tolist(), e_imag=np.moveaxis(e.imag, -1, 0).tolist()))
    return out

