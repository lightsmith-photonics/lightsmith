"""Plane-wave scattering by particles: set-up, run and cross sections."""

from __future__ import annotations

import itertools
import math
import time

import numpy as np

from ...device import Device
from ...resonance import harmonic_inversion
from ...results import FarFieldData, NearFieldData, ResonanceData
from .beam import NA_MAX, BeamSource, plane_wave_components
from .engine import DFTMonitor, Engine, PMLSpec, TimeMonitor, flat_index
from .media import HBARC_EV_NM, Medium, fit_module, needs_fit, to_medium
from .options import FDTDOptions
from .setup import (FS, far_field, flux, flux_points, mesh_mode, particle_bbox, particle_slot_fn,
                    run_until_decayed, scattering_layout, surface_samples, time_step, voxelise)
from .substrate import LayeredPlaneWave, layered_background
from .tfsf import TFSFPlaneWave


def _media(sim, warnings) -> tuple[list[Medium], list[list[int]]]:
    """ADE media and particle slots; a substrate, if any, is the last medium."""
    wl = sim.wavelength_array()
    if sim.outputs.near_field is not None:
        wl = np.concatenate([wl, sim.outputs.near_field.wavelengths_nm])
    if sim.outputs.far_field is not None:
        wl = np.concatenate([wl, sim.outputs.far_field.wavelengths_nm])
    media, slot_of = particle_media(sim.particles, sim.medium_index, wl, warnings)
    if sim.substrate is not None:
        media.append(to_medium(sim.substrate.material, wl, warnings))
    return media, slot_of


def substrate_unsupported(sim) -> str:
    """Why FDTD cannot run this substrate problem (empty if it can)."""
    k, _ = sim.illumination.unit_vectors()
    if sim.illumination.kind != "plane_wave" or abs(k[0]) > 1e-9 or abs(k[1]) > 1e-9:
        return ("with a substrate, FDTD supports plane waves at normal incidence (direction (0, 0, +-1)) only so far; "
                "oblique incidence in layered media is on the roadmap")
    if sim.outputs.far_field is not None:
        return "does not support far_field output with a substrate (yet): it needs the layered-medium far field"
    mat = sim.substrate.material
    if needs_fit(mat) and fit_module() is None:
        return "tabulated/library substrate materials need lightsmith.materials_fit"
    eps = np.asarray(mat.eps(sim.wavelength_array()), dtype=complex)
    if np.any(np.abs(eps.imag) > 1e-3 * np.abs(eps)):
        return ("does not support absorbing substrates (yet): over one, the absorption near the particle mixes "
                "with the background's own, so c_abs and c_ext are not defined by box fluxes; "
                "planned: particle absorption and the reflectance change")
    return ""


def _slot_fn(sim, slot_of, n_media: int):
    fn = particle_slot_fn(sim, slot_of)
    if sim.substrate is None:
        return fn
    z_int, s_sub = sim.substrate.interface_z_nm, n_media - 1

    def with_substrate(pts):
        out = fn(pts)
        return np.where((out == 0) & (pts[..., 2] < z_int), s_sub, out)
    return with_substrate


def _bbox(sim):
    """Particles' bounding box, stretched along z to include the substrate interface."""
    lo, hi = particle_bbox(sim)
    if sim.substrate is not None:
        z = sim.substrate.interface_z_nm
        lo[2], hi[2] = min(lo[2], z), max(hi[2], z)
    return lo, hi


def particle_media(particles, medium_index: float, wl, warnings) -> tuple[list[Medium], list[list[int]]]:
    """ADE media (0 = the host medium) and each particle's region -> medium slots."""
    media = [Medium(medium_index**2, (), "medium")]
    keys: dict[str, int] = {}
    slot_of = []
    for p in particles:
        slots = []
        for m in p.region_materials():
            key = m.model_dump_json()
            if key not in keys:
                keys[key] = len(media)
                media.append(to_medium(m, wl, warnings))
            slots.append(keys[key])
        slot_of.append(slots)
    return media, slot_of


def spacing(sim, opts: FDTDOptions, feature_nm: float | None = None) -> float:
    """Cell size from the options, the wavelengths, the densest material and the smallest feature (the
    smallest particle half-extent / cells_per_radius, and `feature_nm` / 4 when given)."""
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
    dx = wl.min() / (n_max * opts.cells_per_wavelength)
    if sim.particles:
        dx = min(dx, min(float(np.min(p.shape.bounding_half_extent())) for p in sim.particles) / cpr)
    return min(dx, feature_nm / 4) if feature_nm else dx


def _band(sim):
    wl = sim.wavelength_array()
    if sim.outputs.near_field is not None:
        wl = np.concatenate([wl, sim.outputs.near_field.wavelengths_nm])
    if sim.outputs.far_field is not None:
        wl = np.concatenate([wl, sim.outputs.far_field.wavelengths_nm])
    om = 2 * np.pi / wl
    omega_c = 0.5 * (om.min() + om.max())
    omega_half = max(0.5 * (om.max() - om.min()), 0.15 * omega_c)
    return omega_c, omega_half, om.max()


def plan(sim, opts: FDTDOptions, single: bool, warnings: list[str] | None = None, media=None):
    """Everything about the run that is cheap to compute (used by estimate and solve)."""
    warnings = [] if warnings is None else warnings
    dx = spacing(sim, opts)
    lay = scattering_layout(sim, opts, dx, _bbox(sim))
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
    beam = sim.illumination if sim.illumination.kind == "gaussian_beam" else None
    if beam is None and int(np.sum(np.abs(khat) > 1e-12)) != 1:
        warnings.append("oblique incidence: TFSF source is interpolated; expect ~0.1% incident-field leakage")
    eps_b = sim.medium_index**2
    layered = sim.substrate is not None

    t0 = time.perf_counter()
    base, background = None, ()
    if layered:
        base = layered_background(lay.shape, lay.origin, dx, sim.substrate.interface_z_nm, media[0], media[-1])
        background = (0, len(media) - 1)
    inv_eps, groups, vstats, offdiag = voxelise(_slot_fn(sim, slot_of, len(media)), media, lay.shape, lay.origin, dx,
                                                lay.roi[0], lay.roi[1], mode=mesh_mode(opts),
                                                samples=opts.subpixel_samples, base=base, background=background)
    del base
    t_vox = time.perf_counter() - t0
    lam_max = float(sim.wavelength_array().max())
    pml = PMLSpec(cells=((lay.pml, lay.pml),) * 3, kappa_max=opts.pml_kappa,
                  alpha_max=0.05 * 2 * np.pi * sim.medium_index / lam_max, eps_ref=eps_b)

    tfsf = dict(shape=lay.shape, origin=lay.origin, dx=dx, dt=dt, lo=lay.tfsf[0], hi=lay.tfsf[1], eps_b=eps_b,
                omega_c=p["omega_c"], omega_half=p["omega_half"], max_steps=p["max_steps"] + 4 * stride)
    wl = sim.wavelength_array()
    omega = 2 * np.pi / wl
    om_ff = (2 * np.pi / np.asarray(sim.outputs.far_field.wavelengths_nm, dtype=float)
             if sim.outputs.far_field is not None else np.zeros(0))
    beam_info = None
    if layered:
        down = khat[2] < 0
        k_launch = int(lay.tfsf[1][2] + 1 if down else lay.tfsf[0][2] - 1)
        src = LayeredPlaneWave(shape=lay.shape, origin=lay.origin, spacing=dx, dt=dt, lo=lay.tfsf[0], hi=lay.tfsf[1],
                               down=down, ehat=ehat, inv_eps=inv_eps, z_int=sim.substrate.interface_z_nm,
                               host=media[0], sub=media[-1], pml=pml, k_launch=k_launch,
                               k_ref=k_launch - 2 if down else k_launch + 2, omega_c=p["omega_c"],
                               omega_half=p["omega_half"], max_steps=tfsf["max_steps"])
    elif beam is None:
        src = TFSFPlaneWave(r0=lay.r0, khat=khat, ehat=ehat, **tfsf)
    else:
        src, beam_info = _beam_source(sim, beam, lay, tfsf, warnings)
        beam_mon = src.monitor(np.concatenate([omega, om_ff]))
    sca = flux_points(*lay.sca, lay.shape, dx)
    tot = flux_points(*lay.tot, lay.shape, dx)
    mon_sca = DFTMonitor(omega, sca[0], sca[1], sca[2], energy=True)
    mon_tot = DFTMonitor(omega, tot[0], tot[1], tot[2], energy=2)
    if layered:
        mon_inc = DFTMonitor(omega, *src.plane_probe(src.k_ref), source=src.ref_view)
        mon_bg = [DFTMonitor(omega, *src.plane_probe(int(k)), source=src.col_view)
                  for k in (lay.tot[0][2], lay.tot[1][2])]
        monitors = [mon_sca, mon_tot, mon_inc, *mon_bg]
    elif beam is None:
        pe, ph1, ph2 = src.probe()
        mon_inc = DFTMonitor(omega, pe, ph1, ph2, source=src)
        monitors = [mon_sca, mon_tot, mon_inc]
    else:
        monitors = [mon_sca, mon_tot, beam_mon]
    near = _near_monitor(sim, lay) if sim.outputs.near_field is not None else None
    if near is not None and layered:
        near["inc"] = DFTMonitor(near["omega"], *src.plane_probe(src.k_ref), source=src.ref_view)
        near["bg"] = DFTMonitor(near["omega"], src.column_e(), source=src.col_view)
        near["layered"] = src
        monitors += [near["mon"], near["inc"], near["bg"]]
    elif near is not None:
        near["inc"] = DFTMonitor(near["omega"], pe, ph1, ph2, source=src)
        monitors += [near["mon"], near["inc"]]
    far = None
    if sim.outputs.far_field is not None:
        surf = surface_samples(*lay.sca, lay.shape, dx)
        far = {"surf": surf, "omega": om_ff, "mon": DFTMonitor(om_ff, surf.e_idx, surf.h1, surf.h2)}
        monitors.append(far["mon"])
        if beam is None:
            far["inc"] = DFTMonitor(om_ff, pe, ph1, ph2, source=src)
            monitors.append(far["inc"])
    probes = ()
    if sim.outputs.resonances is not None:
        probes = (TimeMonitor(_resonance_probes(sim, lay), p["max_steps"] // stride + 4),)

    eng = Engine(device, shape=lay.shape, dx=dx, dt=dt, single=single, inv_eps=inv_eps, groups=groups, pml=pml,
                 sources=[src], monitors=monitors, dft_stride=stride, offdiag=offdiag, time_monitors=probes)
    del inv_eps
    t_loop = time.perf_counter()
    steps, level, graphs = run_until_decayed(
        eng, opts, max_steps=p["max_steps"], min_steps=int(src.t_passed / dt),
        window=math.ceil(lam_max / (stride * dt)), warnings=warnings)
    t_loop = time.perf_counter() - t_loop

    # cross sections, per unit incident intensity (for a beam: at the focus)
    bg_out = 0.0
    if layered:
        intensity = np.abs(src.s_z(*mon_inc.results()))
        # net outward flux of the background through the total-field box (absorption in a lossy substrate)
        area = float(np.prod((lay.tot[1] - lay.tot[0])[:2])) * dx**2
        bg_out = area * (src.s_z(*mon_bg[1].results()) - src.s_z(*mon_bg[0].results()))
    elif beam is None:
        e_inc, h_inc = mon_inc.results()
        intensity = 0.5 * np.real(e_inc[:, 0] * np.conj(h_inc[:, 0]))
        inc_e2_ff = np.abs(far["inc"].results()[0][:, 0]) ** 2 if far is not None else None
    else:
        e2 = np.sum(np.abs(src.focus_field(beam_mon)) ** 2, axis=-1)
        intensity, inc_e2_ff = 0.5 * sim.medium_index * e2[:len(omega)], e2[len(omega):]
    es, hs = mon_sca.results()
    et, ht = mon_tot.results()
    c_sca = flux(es, hs, sca[3]) / intensity
    c_abs = -(flux(et, ht, tot[3]) - bg_out) / intensity
    near_data = _near_results(sim, lay, near, warnings) if near is not None else None
    far_data = _far_results(sim, lay, far, inc_e2_ff) if far is not None else None
    res_data = None
    if probes:
        n_skip = int(math.ceil(src.t_passed / (stride * dt)))
        res_data = _resonances(sim, probes[0].results(steps // stride)[:, n_skip:], stride * dt, warnings)

    prov = {**device.describe(), "grid": list(lay.shape), "cells": lay.cells, "spacing_nm": dx, "dt_fs": dt / FS,
            "steps": steps, "simulated_time_fs": steps * dt / FS, "dft_stride": stride,
            "precision": "single" if single else "double", "cuda_graphs": graphs,
            "subpixel": mesh_mode(opts) != "staircase",
            "pml_cells": lay.pml, "tfsf_box_nm": float((lay.tfsf[1] - lay.tfsf[0])[0] * dx),
            "decay_level": level, "materials": [m.describe() for m in media], **vstats,
            **({"gaussian_beam": beam_info} if beam_info else {}),
            **({"substrate": {"interface_z_nm": sim.substrate.interface_z_nm,
                              "incidence": "from above" if khat[2] < 0 else "through the substrate"}}
               if layered else {}),
            "runtime_s": time.perf_counter() - t_start, "time_loop_s": t_loop, "voxelise_s": t_vox}
    if np.any(c_abs < -1e-3 * np.max(np.abs(c_sca + c_abs))):
        warnings.append("negative absorption beyond 0.1% of extinction: increase resolution or decay time")
    return {"c_sca": c_sca, "c_abs": c_abs, "c_ext": c_sca + c_abs, "near_field": near_data,
            "far_field": far_data, "resonances": res_data, "provenance": prov, "warnings": warnings}


# --- resonances ----------------------------------------------------------------------

# probe directions: the 6 axes and the 8 cube diagonals, so that a probe sits off the nodes of any low-order mode
_DIRS = np.concatenate([np.eye(3), -np.eye(3),
                        np.array([[a, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)]) / np.sqrt(3)])


def _resonance_probes(sim, lay, max_particles: int = 4) -> np.ndarray:
    """Flat E indices: all three components at each particle's centre and at 14 points two cells outside it."""
    pts = []
    for p in sim.particles[:max_particles]:
        c = np.asarray(p.position_nm, dtype=float)
        h = p.shape.bounding_half_extent()
        pts += [c] + [c + d * (h + 2 * lay.dx) for d in _DIRS]
    idx = []
    for pt in pts:
        for comp in range(3):
            off = np.zeros(3)
            off[comp] = 0.5
            i = np.clip(np.rint((pt - lay.origin) / lay.dx - off).astype(int), 1, np.asarray(lay.shape) - 2)
            idx.append(flat_index(comp, *i, lay.shape))
    return np.unique(np.asarray(idx))


def _resonances(sim, x: np.ndarray, dt: float, warnings: list[str]) -> list[ResonanceData]:
    wl = sim.wavelength_array()
    if x.shape[1] < 64:
        warnings.append("resonances: the field decayed too fast after the pulse for harmonic inversion; "
                        "no resonance found (lower decay_by or set run_time_fs)")
        return []
    modes = harmonic_inversion(x, dt, 2 * np.pi / wl.max(), 2 * np.pi / wl.min(),
                               max_modes=sim.outputs.resonances.max_modes)
    if not modes:
        return []
    a0 = modes[0].amplitude
    return [ResonanceData(wavelength_nm=2 * np.pi / m.omega.real, energy_ev=HBARC_EV_NM * m.omega.real, q=m.q,
                          amplitude=m.amplitude / a0) for m in modes]


# --- far field ---------------------------------------------------------------------


def _beam_source(sim, beam, lay, tfsf, warnings):
    """BeamSource for a GaussianBeam, discretised for the TFSF box (see beam.py)."""
    focus = np.asarray(beam.focus_nm, dtype=float)
    k0, _ = beam.unit_vectors()
    corners = lay.origin + np.array(list(itertools.product(*zip(lay.tfsf[0], lay.tfsf[1])))) * lay.dx - focus
    rho = float(np.max(np.linalg.norm(corners - np.outer(corners @ k0, k0), axis=1)))
    wl = sim.wavelength_array()
    wl_ref = beam.waist_wavelength_nm or 0.5 * (wl.min() + wl.max())
    comps, info = plane_wave_components(beam, sim.medium_index, wl_ref, rho)
    if info["truncated"]:
        warnings.append(f"gaussian_beam: a {beam.waist_nm:g} nm waist needs angles beyond NA {NA_MAX}; its angular "
                        "spectrum is cut there, so the focus is non-paraxial (wider than the Gaussian, elongated "
                        "along the polarisation)")
    warnings.append(f"gaussian_beam: {info['plane_waves']} oblique TFSF plane waves; expect ~0.1% incident-field "
                    "leakage")
    return BeamSource(comps, r0=focus, **tfsf), info


def _far_results(sim, lay, far, inc_e2) -> list[FarFieldData]:
    """dC_sca/dOmega from the scattered fields on the flux box (near-to-far-field transformation).

    theta is measured from the incident direction and phi from the polarization, as in the other backends."""
    req = sim.outputs.far_field
    khat, ehat = sim.illumination.unit_vectors()
    hhat = np.cross(khat, ehat)
    th = np.radians(np.asarray(req.theta_deg, dtype=float))[:, None]
    ph = np.radians(np.asarray(req.phi_deg, dtype=float))[None, :]
    nhat = (np.cos(th)[..., None] * khat + np.sin(th)[..., None]
            * (np.cos(ph)[..., None] * ehat + np.sin(ph)[..., None] * hhat))
    surf = far["surf"]
    r = lay.origin + surf.pos * lay.dx - lay.r0
    e_all, h_all = far["mon"].results()
    nb = sim.medium_index
    out = []
    for f, lam in enumerate(req.wavelengths_nm):
        ef = far_field(surf, e_all[f], h_all[f], r, nhat.reshape(-1, 3), nb * far["omega"][f], 1 / nb)
        dcs = np.sum(np.abs(ef) ** 2, axis=-1).reshape(nhat.shape[:2]) / inc_e2[f]
        out.append(FarFieldData(wavelength_nm=float(lam), theta_deg=list(req.theta_deg), phi_deg=list(req.phi_deg),
                                dcsca_domega_nm2=dcs.tolist()))
    return out


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
    gi, gj, gk = np.meshgrid(*rng, indexing="ij")
    idx = np.concatenate([flat_index(c, gi, gj, gk, lay.shape).ravel() for c in range(3)])
    om = 2 * np.pi / np.asarray(req.wavelengths_nm, dtype=float)
    return {"mon": DFTMonitor(om, idx), "inc": None, "ranges": rng, "omega": om}


def _near_results(sim, lay, near, warnings=None):
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
    layered = near.get("layered")
    if layered is None:
        inc0 = near["inc"].results()[0][:, 0]
    else:
        # incident wave alone (reference column) referred to z = 0, and the background (column) per z node
        e_ref, h_ref = near["inc"].results()
        n_inc = np.sqrt((layered.host if layered.down else layered.sub).eps(om).astype(complex)).real
        z_ref = lay.origin[2] + layered.k_ref * lay.dx
        inc0 = (e_ref[:, 0] * ehat[0] + e_ref[:, 1] * ehat[1]) * np.exp(-1j * n_inc * om * khat[2] * z_ref)
        bg = near["bg"].results()[0].reshape(len(om), 2, -1)
        # the column holds the background between its PML and the source sheet
        nz = lay.shape[2]
        k_ok = ((lay.pml + 1, layered.k_launch - 1) if layered.down else (layered.k_launch + 1, nz - lay.pml - 2))
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
            if layered is None:
                inc = ehat[c] * inc0[f] * np.exp(1j * k * ((P - lay.r0) @ khat))
            else:
                kz = np.floor(U[2] + 1e-9).astype(int)
                ok = (kz >= k_ok[0]) & (kz <= k_ok[1])
                inc = np.where(ok, bg[f, c][np.clip(kz, 0, nz - 1)] if c < 2 else 0.0, np.nan)
            vals = e_all[f, c] + np.where(inside, 0.0, inc)
            interp = RegularGridInterpolator(axes, vals, bounds_error=False, fill_value=np.nan)
            comps.append(interp(pts))
        ref = inc0[f] if layered is not None else inc0[f] * np.exp(-1j * k * (lay.r0 @ khat))
        e = np.stack(comps, axis=-1) / ref
        if layered is not None and warnings is not None and np.isnan(e).any() and f == 0:
            warnings.append("near_field: points outside the source box beyond the incident-field column's range "
                            "(in the PML or above the source plane) are NaN")
        enh = np.sum(np.abs(e) ** 2, axis=-1)
        out.append(NearFieldData(
            wavelength_nm=float(lam), plane=req.plane, offset_nm=req.offset_nm, axis1_nm=u.tolist(),
            axis2_nm=u.tolist(), enhancement=enh.tolist(),
            e_real=np.moveaxis(e.real, -1, 0).tolist(), e_imag=np.moveaxis(e.imag, -1, 0).tolist()))
    return out

