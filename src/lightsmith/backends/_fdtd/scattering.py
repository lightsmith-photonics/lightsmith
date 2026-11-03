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
from ...materials import ConstantIndex, ConstantPermittivity
from .media import HBARC_EV_NM, Medium, fit_module, needs_fit, to_medium
from .options import FDTDOptions
from .setup import (FS, far_field, flux, flux_points, mesh_mode, particle_bbox, particle_slot_fn,
                    run_until_decayed, scattering_layout, surface_samples, time_step, voxelise)
from .absorption import ParticleAbsorption
from .layered import OMEGA_H, ObliqueLayeredPlaneWave
from .layered_far import LayeredFarField, hemisphere, outgoing_waves
from .substrate import LayeredPlaneWave, layered_background
from . import symmetry


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
    if sim.illumination.kind != "plane_wave":
        return "with a substrate, FDTD supports plane-wave illumination only (not Gaussian beams yet)"
    if abs(k[2]) < 0.1:
        return "with a substrate, the plane wave must not graze the interface (|direction_z| >= 0.1)"
    oblique = abs(k[0]) > 1e-9 or abs(k[1]) > 1e-9
    if oblique and k[2] > 0 and not isinstance(sim.substrate.material, (ConstantIndex, ConstantPermittivity)):
        return ("oblique incidence through the substrate needs a constant-index substrate (a fixed angle in a "
                "dispersive incidence medium has no fixed transverse slowness)")
    mat = sim.substrate.material
    if needs_fit(mat) and fit_module() is None:
        return "tabulated/library substrate materials need lightsmith.materials_fit"
    if k[2] > 0 and _absorbing(sim):
        return "the plane wave cannot come in through an absorbing substrate (it decays there); illuminate from above"
    if (k[2] > 0 and not oblique and sim.outputs.far_field is not None
            and not isinstance(mat, (ConstantIndex, ConstantPermittivity))):
        return ("far_field output with incidence through the substrate needs a constant-index substrate (the far field "
                "uses the tabled source, which needs a non-dispersive incidence medium)")
    return ""


def uses_single(sim, opts: FDTDOptions, device: Device) -> bool:
    """float32 for "auto" on a GPU, except over an absorbing substrate: a metal's background rings for long, and
    over such runs float32 rounding in the total-field region leaves noise in the scattered field (C_sca of a sphere
    over a Drude metal off by up to 1.2 %, against 0.13 % in double)."""
    if opts.precision != "auto":
        return opts.precision == "single"
    return device.is_gpu and not _absorbing(sim)


def _absorbing(sim) -> bool:
    """Whether the substrate absorbs in the band (Im eps > 1e-4 |eps| anywhere)."""
    if sim.substrate is None:
        return False
    eps = np.asarray(sim.substrate.material.eps(_band_wavelengths(sim)), dtype=complex)
    return bool(np.any(np.abs(eps.imag) > 1e-4 * np.abs(eps)))


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


def _band_wavelengths(sim):
    wl = sim.wavelength_array()
    if sim.outputs.near_field is not None:
        wl = np.concatenate([wl, sim.outputs.near_field.wavelengths_nm])
    if sim.outputs.far_field is not None:
        wl = np.concatenate([wl, sim.outputs.far_field.wavelengths_nm])
    return wl


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


def _tabled(sim) -> bool:
    """Whether the source is fed from tables of its exact discrete field (layered.py, beam.py): every plane
    wave and beam, except a plane wave at normal incidence on a transparent substrate without far-field output
    (an auxiliary column, substrate.py)."""
    if sim.substrate is None:
        return True
    k, _ = sim.illumination.unit_vectors()
    return abs(k[0]) > 1e-9 or abs(k[1]) > 1e-9 or _absorbing(sim) or sim.outputs.far_field is not None


def plan(sim, opts: FDTDOptions, single: bool, warnings: list[str] | None = None, media=None):
    """Everything about the run that is cheap to compute (used by estimate and solve)."""
    warnings = [] if warnings is None else warnings
    dx = spacing(sim, opts)
    mirror, sym = symmetry.find(sim, opts)
    lay = symmetry.halve(scattering_layout(sim, opts, dx, _bbox(sim)), mirror)
    if media is None:
        media, _ = _media(sim, warnings)
    dt = time_step(dx, opts.courant, media)
    omega_c, omega_half, omega_max = _band(sim)
    # sampling every `stride` steps aliases source content above omega_c + 4 omega_half (amplitude
    # < e^-16) onto the highest requested frequency, never below it. An oblique wave from tables is
    # compared with its exact background (absorption), which needs the e^-36 margin of 6 omega_half.
    margin = 6 if _tabled(sim) else 4
    stride = opts.dft_stride or max(1, int(2 * np.pi / (omega_c + margin * omega_half + omega_max) / dt))
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
    mem = bytes_ * (12 * n_cells + 4 * n_flux * n_f) + _table_bytes(sim, lay, dt, omega_c, omega_half, media, bytes_)
    if _absorbing(sim):  # DFTs of E and D on the particles' nodes, and of the surface for the far field
        n_nodes = 3 * sum(p.shape.volume() for p in sim.particles) / dx**3 * 1.3
        mem += bytes_ * 4 * n_f * (n_nodes + n_flux)
    return {"dx": dx, "layout": lay, "dt": dt, "stride": stride, "max_steps": max_steps, "est_steps": est_steps,
            "omega_c": omega_c, "omega_half": omega_half, "media": media, "memory_bytes": int(mem),
            "warnings": warnings, "symmetry": sym}


def _table_bytes(sim, lay, dt, omega_c, omega_half, media, bytes_) -> int:
    """Rough size of the source tables (layered.py): per plane wave (one, or every beam component), 6 field
    components x the box's rows x the pulse duration sampled every m steps."""
    if not _tabled(sim):
        return 0
    beam = sim.illumination if sim.illumination.kind == "gaussian_beam" else None
    n_src = len(_beam_components(sim, beam, lay)[0]) if beam is not None else 1
    ext = lay.tfsf[1] - lay.tfsf[0]
    n_k = int(ext.max()) + 5
    n_max = max(np.sqrt(m.eps_inf) for m in (media[0], media[-1]))
    tau_p = 2 / omega_half
    t0 = max(5 * tau_p, 30 / (2 * np.pi / float(np.max(_band_wavelengths(sim)))))
    t = 2 * t0 + 4 * tau_p + (4 * n_k + 2 * float(np.linalg.norm(ext))) * lay.dx * n_max
    n_t = 1 << int(np.ceil(np.log2(t / dt)))
    m = max(1, int(OMEGA_H / ((omega_c + 3 * omega_half) * dt)))
    return int(bytes_ * n_src * 6 * n_k * (n_t // m + 8))


def solve(sim, opts: FDTDOptions, device: Device):
    t_start = time.perf_counter()
    warnings: list[str] = []
    single = uses_single(sim, opts, device)
    media, slot_of = _media(sim, warnings)
    p = plan(sim, opts, single, warnings, media)
    lay, dx, dt, stride = p["layout"], p["dx"], p["dt"], p["stride"]
    if lay.cells > opts.max_cells:
        raise ValueError(f"grid {lay.shape} has {lay.cells} cells > max_cells={opts.max_cells}; "
                         "increase spacing_nm or raise max_cells")
    khat, ehat = sim.illumination.unit_vectors()
    beam = sim.illumination if sim.illumination.kind == "gaussian_beam" else None
    eps_b = sim.medium_index**2
    layered = sim.substrate is not None
    absorbing = _absorbing(sim)

    t0 = time.perf_counter()
    base, background = None, ()
    if layered:
        base = layered_background(lay.shape, lay.origin, dx, sim.substrate.interface_z_nm, media[0], media[-1])
        background = (0, len(media) - 1)
    inv_eps, groups, vstats, offdiag = voxelise(_slot_fn(sim, slot_of, len(media)), media, lay.shape, lay.origin, dx,
                                                lay.roi[0], lay.roi[1], mode=mesh_mode(opts),
                                                samples=opts.subpixel_samples, base=base, background=background,
                                                mirror=lay.mirror)
    del base
    t_vox = time.perf_counter() - t0
    lam_max = float(sim.wavelength_array().max())
    pml = PMLSpec(cells=symmetry.pml_cells(lay), kappa_max=opts.pml_kappa,
                  alpha_max=0.05 * 2 * np.pi * sim.medium_index / lam_max, eps_ref=eps_b)

    src_args = dict(shape=lay.shape, origin=lay.origin, spacing=dx, dt=dt, lo=lay.tfsf[0], hi=lay.tfsf[1],
                    inv_eps=inv_eps, host=media[0], omega_c=p["omega_c"], omega_half=p["omega_half"],
                    max_steps=p["max_steps"] + 4 * stride,
                    omega_min=2 * np.pi / float(np.max(_band_wavelengths(sim))))
    wl = sim.wavelength_array()
    omega = 2 * np.pi / wl
    om_ff = (2 * np.pi / np.asarray(sim.outputs.far_field.wavelengths_nm, dtype=float)
             if sim.outputs.far_field is not None else np.zeros(0))
    beam_info = None
    tabled = beam is None and _tabled(sim)  # a plane wave from tables
    if tabled:
        sub = dict(z_int=sim.substrate.interface_z_nm, sub=media[-1]) if layered else {}
        src = ObliqueLayeredPlaneWave(khat=khat, ehat=ehat, **src_args, **sub, mirror=lay.mirror)
    elif beam is not None:
        src, beam_info = _beam_source(sim, beam, lay, src_args, warnings)
        beam_mon = src.monitor(np.concatenate([omega, om_ff]))
    else:
        down = khat[2] < 0
        k_launch = int(lay.tfsf[1][2] + 1 if down else lay.tfsf[0][2] - 1)
        src = LayeredPlaneWave(shape=lay.shape, origin=lay.origin, spacing=dx, dt=dt, lo=lay.tfsf[0], hi=lay.tfsf[1],
                               down=down, ehat=ehat, inv_eps=inv_eps, z_int=sim.substrate.interface_z_nm,
                               host=media[0], sub=media[-1], pml=pml, k_launch=k_launch,
                               k_ref=k_launch - 2 if down else k_launch + 2, omega_c=p["omega_c"],
                               omega_half=p["omega_half"], max_steps=src_args["max_steps"], mirror=lay.mirror)
    # with mirror planes: boxes open on the planes, fluxes and absorption times n_sym, far fields unfolded
    n_sym, open_lo = symmetry.factor(lay.mirror), symmetry.open_faces(lay)
    sca = flux_points(*symmetry.cut_box(lay.sca, lay), lay.shape, dx, open_lo)
    tot = flux_points(*symmetry.cut_box(lay.tot, lay), lay.shape, dx, open_lo)
    mon_sca = DFTMonitor(omega, sca[0], sca[1], sca[2], energy=True)
    mon_tot = DFTMonitor(omega, tot[0], tot[1], tot[2], energy=2)
    if tabled:
        mon_inc = DFTMonitor(omega, np.arange(3), source=src.view)
        monitors = [mon_sca, mon_tot, mon_inc]
    elif beam is not None:
        monitors = [mon_sca, mon_tot, beam_mon]
    else:
        mon_inc = DFTMonitor(omega, *src.plane_probe(src.k_ref), source=src.ref_view)
        mon_bg = [DFTMonitor(omega, *src.plane_probe(int(k)), source=src.col_view)
                  for k in (lay.tot[0][2], lay.tot[1][2])]
        monitors = [mon_sca, mon_tot, mon_inc, *mon_bg]
    near = _near_monitor(sim, lay) if sim.outputs.near_field is not None else None
    if near is not None and tabled:
        near["inc"] = DFTMonitor(near["omega"], np.arange(3), source=src.view)
        near["oblique"] = src
        monitors += [near["mon"], near["inc"]]
    elif near is not None and beam is not None:
        near["inc"] = src.monitor(near["omega"])
        near["beam"] = src
        monitors += [near["mon"], near["inc"]]
    elif near is not None:
        near["inc"] = DFTMonitor(near["omega"], *src.plane_probe(src.k_ref), source=src.ref_view)
        near["bg"] = DFTMonitor(near["omega"], src.column_e(), source=src.col_view)
        near["layered"] = src
        monitors += [near["mon"], near["inc"], near["bg"]]
    far = None
    if sim.outputs.far_field is not None:
        surf = surface_samples(*symmetry.cut_box(lay.sca, lay), lay.shape, dx, open_lo)
        far = {"surf": surf, "omega": om_ff, "mon": DFTMonitor(om_ff, surf.e_idx, surf.h1, surf.h2)}
        monitors.append(far["mon"])
        if tabled:
            far["inc"] = DFTMonitor(om_ff, np.arange(3), source=src.view)
            monitors.append(far["inc"])
        if layered:
            far["lff"] = symmetry.UnfoldedFarField(LayeredFarField(
                surf, origin=lay.origin, dx=dx, dt=dt, z_int=sim.substrate.interface_z_nm, host=media[0],
                sub=media[-1]), lay.mirror)
    # over an absorbing substrate: the particles' own absorption, and the far field for C_sca and C_ext
    pabs = None
    if absorbing:
        pabs = ParticleAbsorption(groups, media[1:-1], omega, dt)
        for _, me, _, w in pabs.parts:  # nodes on a symmetric plane are shared with their image
            w *= symmetry.plane_weights(me.e_idx, lay.shape, lay.mirror)[None, :]
        surf_cs = surface_samples(*symmetry.cut_box(lay.sca, lay), lay.shape, dx, open_lo)
        mon_cs = DFTMonitor(omega, surf_cs.e_idx, surf_cs.h1, surf_cs.h2)
        monitors += [*pabs.monitors, mon_cs]
    probes = ()
    if sim.outputs.resonances is not None:
        probes = (TimeMonitor(_resonance_probes(sim, lay), p["max_steps"] // stride + 4),)

    eng = Engine(device, shape=lay.shape, dx=dx, dt=dt, single=single, inv_eps=inv_eps, groups=groups, pml=pml,
                 sources=[src], monitors=monitors, dft_stride=stride, offdiag=offdiag, time_monitors=probes,
                 mirror=lay.mirror)
    del inv_eps
    if pabs is not None:
        pabs.bind(eng)
    t_loop = time.perf_counter()
    steps, level, graphs = run_until_decayed(
        eng, opts, max_steps=p["max_steps"], min_steps=int(src.t_passed / dt),
        window=math.ceil(lam_max / (stride * dt)), warnings=warnings)
    t_loop = time.perf_counter() - t_loop

    # cross sections, per unit incident intensity (for a beam: at the focus)
    if tabled:
        # incident wave alone at the reference point: I = n |E|^2 / 2. (The grid's discrete Poynting flux
        # differs by O((k dx)^2) and with direction, 1e-3 axial and 4e-4 diagonal at k dx = 0.1, where with
        # n |E|^2 / 2 a sphere's C_ext is the same for every direction to 4e-5.)
        e_ref = mon_inc.results()[0]
        intensity = 0.5 * src.n_inc * np.sum(np.abs(e_ref) ** 2, axis=1)
        # the background's own flux through the total-field box, from the same flux formula (the averaged-H
        # flux is not exactly conserved by the grid, ~1e-6 of the box flux)
        unit = src.incident_reference(omega)
        amp = np.sum(e_ref * np.conj(unit), axis=1) / np.sum(np.abs(unit) ** 2, axis=1)
        e_bg = amp[:, None] * src.background(omega, tot[0], "E")
        h_bg = amp[:, None] * 0.5 * (src.background(omega, tot[1], "H") + src.background(omega, tot[2], "H"))
        bg_out = n_sym * flux(e_bg, h_bg, tot[3])
        if far is not None:
            inc_e2_ff = np.sum(np.abs(far["inc"].results()[0]) ** 2, axis=1)
    elif beam is not None:
        om_all = np.concatenate([omega, om_ff])
        amp = src.amplitudes(beam_mon, om_all)
        e2 = np.sum(np.abs(src.focus_field(amp, om_all)) ** 2, axis=-1)
        intensity, inc_e2_ff = 0.5 * sim.medium_index * e2[:len(omega)], e2[len(omega):]
        bg_out = src.box_flux(amp[:len(omega)], omega, tot)
    else:
        intensity = np.abs(src.s_z(*mon_inc.results()))
        # net outward flux of the background through the total-field box (absorption in a lossy substrate)
        area = float(np.prod((lay.tot[1] - lay.tot[0])[:2])) * dx**2  # the whole box's (lo is mirrored)
        bg_out = area * (src.s_z(*mon_bg[1].results()) - src.s_z(*mon_bg[0].results()))
    es, hs = mon_sca.results()
    et, ht = mon_tot.results()
    c_sca = n_sym * flux(es, hs, sca[3]) / intensity
    c_abs = -(n_sym * flux(et, ht, tot[3]) - bg_out) / intensity
    c_ext = c_sca + c_abs
    if absorbing:
        # the box fluxes would count the substrate's absorption near the particle: C_abs is the particles' own, C_sca
        # the far field (into the upper half-space), C_ext the power taken from the reflected wave
        c_abs = n_sym * pabs.power(dx**3) / intensity
        lff = symmetry.UnfoldedFarField(LayeredFarField(
            surf_cs, origin=lay.origin, dx=dx, dt=dt, z_int=sim.substrate.interface_z_nm, host=media[0],
            sub=media[-1]), lay.mirror)
        c_sca, c_ext = _layered_cross_sections(sim, lay, lff, src, omega, *mon_cs.results(), amp,
                                               np.sum(np.abs(e_ref) ** 2, axis=1))
    near_data = _near_results(sim, lay, near, warnings) if near is not None else None
    far_data = _far_results(sim, lay, far, inc_e2_ff) if far is not None else None
    res_data = None
    if probes:
        n_skip = int(math.ceil(getattr(src, "t_off", src.t_passed) / (stride * dt)))  # once the source is off
        res_data = _resonances(sim, probes[0].results(steps // stride)[:, n_skip:], stride * dt, warnings)

    prov = {**device.describe(), "grid": list(lay.shape), "cells": lay.cells, "spacing_nm": dx, "dt_fs": dt / FS,
            "steps": steps, "simulated_time_fs": steps * dt / FS, "dft_stride": stride,
            "precision": "single" if single else "double", "cuda_graphs": graphs,
            "subpixel": mesh_mode(opts) != "staircase",
            "pml_cells": lay.pml, "tfsf_box_nm": float((lay.tfsf[1] - lay.tfsf[0])[0] * dx),
            "decay_level": level, "materials": [m.describe() for m in media], **vstats, "symmetry": p["symmetry"],
            **({"gaussian_beam": beam_info} if beam_info else {}),
            **({"tfsf_table_stride": src.m} if tabled or beam is not None else {}),
            **({"substrate": {"interface_z_nm": sim.substrate.interface_z_nm,
                              "incidence": "from above" if khat[2] < 0 else "through the substrate",
                              "source": "layered tables" if tabled else "column", "absorbing": absorbing,
                              **({"particle_absorption_nodes": pabs.nodes} if pabs is not None else {})}}
               if layered else {}),
            "runtime_s": time.perf_counter() - t_start, "time_loop_s": t_loop, "voxelise_s": t_vox}
    if np.any(c_abs < -1e-3 * np.max(np.abs(c_sca + c_abs))):
        warnings.append("negative absorption beyond 0.1% of extinction: increase resolution or decay time")
    return {"c_sca": c_sca, "c_abs": c_abs, "c_ext": c_ext, "near_field": near_data,
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
            idx.append(flat_index(comp, *symmetry.node_index(pt, comp, lay), lay.shape))
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


def _beam_components(sim, beam, lay):
    """The beam's plane waves, discretised for the TFSF box (see beam.py)."""
    focus = np.asarray(beam.focus_nm, dtype=float)
    k0, _ = beam.unit_vectors()
    corners = lay.origin + np.array(list(itertools.product(*zip(lay.tfsf[0], lay.tfsf[1])))) * lay.dx - focus
    rho = float(np.max(np.linalg.norm(corners - np.outer(corners @ k0, k0), axis=1)))
    wl = sim.wavelength_array()
    wl_ref = beam.waist_wavelength_nm or 0.5 * (wl.min() + wl.max())
    return plane_wave_components(beam, sim.medium_index, wl_ref, rho)


def _beam_source(sim, beam, lay, src_args, warnings):
    """BeamSource for a GaussianBeam (see beam.py)."""
    comps, info = _beam_components(sim, beam, lay)
    focus = np.asarray(beam.focus_nm, dtype=float)
    if info["truncated"]:
        warnings.append(f"gaussian_beam: a {beam.waist_nm:g} nm waist needs angles beyond NA {NA_MAX}; its angular "
                        "spectrum is cut there, so the focus is non-paraxial (wider than the Gaussian, elongated "
                        "along the polarisation)")
    return BeamSource(comps, focus=focus, **src_args), info


def _layered_cross_sections(sim, lay, lff, src, omega, es, hs, amp, e0sq):
    """C_sca (the far field over every half-space it reaches) and C_ext (the optical theorem of each outgoing
    background wave) over a substrate, from the layered far field (layered_far.py); e0sq = |E_inc|^2, the incident
    intensity n_inc e0sq / 2 in the incidence medium."""
    n1, n_inc = sim.medium_index, src.n_inc
    R = 0.5 * float(np.linalg.norm(lay.sca[1] - lay.sca[0])) * lay.dx
    waves = outgoing_waves(src, omega, amp)
    c_sca, c_ext = np.zeros(len(omega)), np.zeros(len(omega))
    for f, w in enumerate(omega):
        n_sub = lff.side_index(w, False)
        for up in (True, False):
            n_this = n1 if up else n_sub
            if n_this is None:
                continue
            n_other = lff.side_index(w, not up)
            kr = n_this * w * R
            dirs, wts = hemisphere(n_this, n_other if n_other is not None else 1j, up, int(1.5 * kr) + 10,
                                   2 * int(1.5 * kr) + 20)
            ff = lff.amplitudes(w, es[f], hs[f], dirs)
            c_sca[f] += (n_this / n_inc) * np.sum(wts * np.sum(np.abs(ff) ** 2, axis=1)) / e0sq[f]
        for wave in waves:
            if wave is not None:
                khat, eb, _ = wave
                fb = lff.amplitudes(w, es[f], hs[f], khat[f:f + 1])[0]
                c_ext[f] += 4 * np.pi * np.imag(fb @ np.conj(eb[f])) / (w * n_inc * e0sq[f])
    return c_sca, c_ext


def _far_results(sim, lay, far, inc_e2) -> list[FarFieldData]:
    """dC_sca/dOmega from the scattered fields on the flux box (near-to-far-field transformation; over a substrate,
    by reciprocity with the layered background, zero into an absorbing substrate).

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
    lff = far.get("lff")
    for f, lam in enumerate(req.wavelengths_nm):
        w = far["omega"][f]
        if lff is None:
            ef = symmetry.unfold(lambda n: far_field(surf, e_all[f], h_all[f], r, n, nb * w, 1 / nb),
                                 nhat.reshape(-1, 3), lay.mirror)
            ratio = 1.0
        else:
            ef = lff.amplitudes(w, e_all[f], h_all[f], nhat.reshape(-1, 3))  # per unit intensity n_inc |E|^2 / 2
            n_inc = nb if khat[2] < 0 else lff.side_index(w, False)
            ratio = np.where(nhat.reshape(-1, 3)[:, 2] >= 0, nb, lff.side_index(w, False) or 0.0) / n_inc
        dcs = (ratio * np.sum(np.abs(ef) ** 2, axis=-1)).reshape(nhat.shape[:2]) / inc_e2[f]
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
        lo_v, hi_v = symmetry.fold_interval(lo_v, hi_v, a, lay)
        i0 = max(1, int(math.floor((lo_v - lo_w[a]) / lay.dx)) - 1)
        if lay.mirror[a] and i0 <= 2:  # from the ghost plane, filled from the images (symmetry.fill_ghosts)
            i0 = 0
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
    symmetry.fill_ghosts(e_all, rng, lay)
    u = np.linspace(-req.half_width_nm, req.half_width_nm, req.points)
    g1, g2 = np.meshgrid(u, u, indexing="ij")
    pts = np.zeros(g1.shape + (3,))
    pts[..., a1], pts[..., a2], pts[..., nrm] = g1, g2, req.offset_nm
    pts, par = symmetry.fold_points(pts, lay)  # E beyond a mirror plane: parity times E at the image
    khat, ehat = sim.illumination.unit_vectors()
    lo, hi = lay.tfsf
    out = []
    layered = near.get("layered")
    oblique = near.get("oblique")
    bsrc = near.get("beam")
    if oblique is not None:
        e_mon = near["inc"].results()[0]  # incident wave alone at the reference point, monitor units
        unit = oblique.incident_reference(om)
        amp = np.sum(e_mon * np.conj(unit), axis=1) / np.sum(np.abs(unit) ** 2, axis=1)
        # amp: the incident amplitude along ehat (the unit solution has tangential E = ehat_t at the reference
        # node); e_mon @ ehat would mix in E_z from half a cell away
        inc0 = amp * np.exp(-1j * oblique.n_inc * om * (khat @ oblique.r_ref))
    elif bsrc is not None:  # normalised by the beam at the focus, along its polarisation
        amps = bsrc.amplitudes(near["inc"], om)
        inc0 = bsrc.focus_field(amps, om) @ ehat
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
            if oblique is not None:
                ijk = [np.floor(U[a] + 1e-9).astype(int).ravel() for a in range(3)]
                flat = flat_index(c, *ijk, lay.shape)
                inc = (amp[f] * oblique.background(om[f:f + 1], flat, "E")[0]).reshape(U[0].shape)
                if layered is None and sim.substrate is None:  # beyond the tables: the continuum plane wave
                    inc = np.where(np.isnan(inc), ehat[c] * inc0[f] * np.exp(1j * k * (P @ khat)), inc)
            elif bsrc is not None:
                ijk = [np.floor(U[a] + 1e-9).astype(int).ravel() for a in range(3)]
                flat = flat_index(c, *ijk, lay.shape)
                inc = bsrc.background(amps[f:f + 1], om[f:f + 1], flat, "E")[0].reshape(U[0].shape)
                inc = np.where(np.isnan(inc), bsrc.continuum(amps[f:f + 1], om[f:f + 1], P)[0][..., c], inc)
            else:
                kz = np.floor(U[2] + 1e-9).astype(int)
                ok = (kz >= k_ok[0]) & (kz <= k_ok[1])
                inc = np.where(ok, bg[f, c][np.clip(kz, 0, nz - 1)] if c < 2 else 0.0, np.nan)
            vals = e_all[f, c] + np.where(inside, 0.0, inc)
            interp = RegularGridInterpolator(axes, vals, bounds_error=False, fill_value=np.nan)
            comps.append(par[..., c] * interp(pts))
        e = np.stack(comps, axis=-1) / inc0[f]
        if warnings is not None and np.isnan(e).any() and f == 0:
            warnings.append("near_field: points outside the source box, beyond the range where the background field "
                            "is known (the PML, above the source plane, or beyond the oblique-incidence tables), are NaN")
        enh = np.sum(np.abs(e) ** 2, axis=-1)
        out.append(NearFieldData(
            wavelength_nm=float(lam), plane=req.plane, offset_nm=req.offset_nm, axis1_nm=u.tolist(),
            axis2_nm=u.tolist(), enhancement=enh.tolist(),
            e_real=np.moveaxis(e.real, -1, 0).tolist(), e_imag=np.moveaxis(e.imag, -1, 0).tolist()))
    return out

