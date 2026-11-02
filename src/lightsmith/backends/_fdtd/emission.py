"""Dipole emission near particles (``EmissionProblem``): Purcell factor, radiative enhancement, efficiency and
radiation pattern.

A point current J = j(t) p_hat, spread over the eight E_c nodes around the dipole by trilinear weights,
radiates a broadband pulse. Running DFTs give

    P_total(w) = -1/2 Re sum_i E_i(w) J_i(w)* dV     (the power the dipole emits, at the source)
    P_rad(w)   = outward flux of E x H* through a closed box around the dipole and the particles

and a second run of the same dipole in the medium alone (same grid, time step and source sampling) gives
P0(w). Then purcell = P_total / P0, radiative = P_rad / P0 and efficiency = P_rad / P_total; the
discretisation of the point source cancels in the ratios. The radiation pattern projects the fields on the
flux box to the far field (as for scattering) and is normalised by P0.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ...device import Device
from ...problems import EmissionProblem
from ...results import EmissionResult, RadiationPatternData
from .engine import DFTMonitor, Engine, PMLSpec, flat_index
from .media import fit_module, needs_fit
from .options import FDTDOptions
from .scattering import particle_media, spacing
from .setup import (FS, e_offset, far_field, flux, flux_points, mesh_mode, particle_slot_fn, run_until_decayed,
                    scattering_layout, surface_samples, time_step, voxelise)


def supports(prob: EmissionProblem) -> tuple[bool, str]:
    if any(needs_fit(m) for p in prob.particles for m in p.region_materials()) and fit_module() is None:
        return False, "tabulated/library materials need lightsmith.materials_fit (Drude-Lorentz fitting)"
    return True, ""


def _wavelengths(prob) -> np.ndarray:
    wl = prob.wavelength_array()
    if prob.radiation_pattern is not None:
        wl = np.concatenate([wl, prob.radiation_pattern.wavelengths_nm])
    return wl


def _gap(prob) -> float | None:
    """Distance from the dipole to the nearest particle's bounding sphere (None if there is none outside)."""
    r = np.asarray(prob.dipole.position_nm, dtype=float)
    gaps = [np.linalg.norm(r - np.asarray(p.position_nm)) - float(np.max(p.shape.bounding_half_extent()))
            for p in prob.particles]
    gaps = [g for g in gaps if g > 0]
    return min(gaps) if gaps else None


def _bbox(prob, dx: float):
    r = np.asarray(prob.dipole.position_nm, dtype=float)
    lo, hi = r - 2 * dx, r + 2 * dx
    for p in prob.particles:
        c, h = np.asarray(p.position_nm, dtype=float), p.shape.bounding_half_extent()
        lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
    return lo, hi


def uses_single(opts: FDTDOptions) -> bool:
    """Emission runs in double precision unless single is asked for (see the warning in solve)."""
    return opts.precision == "single"


def plan(prob: EmissionProblem, opts: FDTDOptions, single: bool, warnings: list[str] | None = None, media=None):
    warnings = [] if warnings is None else warnings
    wl = _wavelengths(prob)
    if media is None:
        media, _ = particle_media(prob.particles, prob.medium_index, wl, warnings)
    dx = spacing(prob, opts, _gap(prob))
    lay = scattering_layout(prob, opts, dx, bbox=_bbox(prob, dx))
    dt = time_step(dx, opts.courant, media)
    om = 2 * np.pi / wl
    omega_c = 0.5 * (om.min() + om.max())
    omega_half = max(0.5 * (om.max() - om.min()), 0.15 * omega_c)
    # sampling every `stride` steps folds pulse content above omega_c + 6 omega_half (< e^-36) onto the band:
    # the scattering margin (e^-16) is not enough here, the self-field at the source amplifies it ~1/(k dx)^3
    stride = opts.dft_stride or max(1, int(2 * np.pi / (omega_c + 6 * omega_half + om.max()) / dt))
    max_t = (opts.run_time_fs or opts.max_time_fs) * FS
    max_steps = int(math.ceil(max_t / dt / stride)) * stride
    tau = 2 / omega_half
    t_pulse = 7 * tau + lay.shape[0] * dx * prob.medium_index
    gammas = [p.gamma for m in media for p in m.poles if p.sigma > 0]
    t_ring = math.log(1 / opts.decay_by) / max(min(gammas), 1e-6) if gammas else 4 * t_pulse / 7
    est_steps = max_steps if opts.run_time_fs else min(max_steps, int((t_pulse + t_ring) / dt))
    n_flux = 2 * 12 * int(np.prod(lay.sca[1] - lay.sca[0] + 1) ** (2 / 3))
    mem = (4 if single else 8) * (12 * lay.cells + 4 * n_flux * len(wl))
    return {"dx": dx, "layout": lay, "dt": dt, "stride": stride, "max_steps": max_steps, "est_steps": est_steps,
            "omega_c": omega_c, "omega_half": omega_half, "memory_bytes": int(mem), "runs": 2, "warnings": warnings}


class PointDipole:
    """A point current source J = j(t) p_hat at `pos`, spread over the eight E_c nodes around it.

    E_i -= dt (1/eps)_i w_i j(t) in the E update (after the curl, before the dispersive groups, so D-form
    cells see it as a current). j is a Gaussian-modulated sine, applied at t = (n + 1/2) dt in step n."""

    def __init__(self, *, shape, origin, dx: float, dt: float, pos, pdir, inv_eps: np.ndarray, omega_c: float,
                 omega_half: float, max_steps: int):
        idx, w = [], []
        for c in range(3):
            if abs(pdir[c]) < 1e-12:
                continue
            f = (np.asarray(pos, float) - origin) / dx - e_offset(c)
            i0 = np.floor(f).astype(int)
            t = f - i0
            for corner in range(8):
                bits = np.array([(corner >> a) & 1 for a in range(3)])
                wt = float(np.prod(np.where(bits, t, 1 - t)))
                if wt > 1e-12:
                    idx.append(int(flat_index(c, *(i0 + bits), shape)))
                    w.append(wt * pdir[c])
        self.idx, self.w = np.asarray(idx), np.asarray(w)
        self.coef = dt * inv_eps.reshape(-1)[self.idx] * self.w
        # j = dg/dt as the exact difference (g((n+1) dt) - g(n dt)) / dt, so sum_n j dt = g(end) - g(0) ~ 1e-11:
        # a current with any net charge leaves a static dipole whose self-field (~1/(k dx)^3 times the radiating
        # field) never decays and spoils -Re(E J*) by whenever the run stops
        self.tau = 2.0 / omega_half
        self.t0 = 5.0 * self.tau
        t = np.arange(max_steps + 3) * dt
        g = np.exp(-(((t - self.t0) / self.tau) ** 2)) * np.sin(omega_c * (t - self.t0))
        self.wave = np.diff(g) / (omega_c * dt)
        self.dt = dt

    def bind(self, eng):
        self._idx, self._coef, self._wave = eng.index(self.idx), eng.asarray(self.coef), eng.asarray(self.wave)

    def correct_H(self, eng):
        pass

    def advance_H(self, eng):
        pass

    def correct_E(self, eng):
        eng.E_flat[self._idx] -= self._coef * self._wave[eng.n]

    def advance_E(self, eng):
        pass

    def spectrum(self, omega: np.ndarray, stride: int, n_samples: int) -> np.ndarray:
        """j(w) sampled as the DFT monitors sample E: at the steps N = stride, 2 stride, ..., j applied at
        (N - 1/2) dt in step N - 1."""
        N = stride * np.arange(1, n_samples + 1)
        t = (N - 0.5) * self.dt
        return np.exp(1j * np.outer(omega, t)) @ self.wave[N - 1]


def _run(prob, opts, device, p, single, media, slot_of, omega, scene: bool, pattern, warnings):
    lay, dx, dt, stride = p["layout"], p["dx"], p["dt"], p["stride"]
    if scene and prob.particles:
        inv_eps, groups, vstats, offdiag = voxelise(
            particle_slot_fn(prob, slot_of), media, lay.shape, lay.origin, dx, lay.roi[0], lay.roi[1],
            mode=mesh_mode(opts), samples=opts.subpixel_samples)
    else:
        inv_eps, groups, vstats, offdiag = np.full((3,) + lay.shape, 1 / media[0].eps_inf), [], {}, None
    src = PointDipole(shape=lay.shape, origin=lay.origin, dx=dx, dt=dt, pos=prob.dipole.position_nm,
                      pdir=prob.dipole.unit(), inv_eps=inv_eps, omega_c=p["omega_c"], omega_half=p["omega_half"],
                      max_steps=p["max_steps"] + 4 * stride)
    if scene and np.any(inv_eps.reshape(-1)[src.idx] != 1 / media[0].eps_inf):
        warnings.append("the dipole sits in a cell that is not pure host medium; its emission is that of a "
                        "dipole inside the material, normalised by one in the host medium")
    box = flux_points(*lay.sca, lay.shape, dx)
    mon_src = DFTMonitor(omega, src.idx)
    mon_box = DFTMonitor(omega, box[0], box[1], box[2], energy=True)
    monitors = [mon_src, mon_box]
    ff = None
    if scene and pattern is not None:
        surf = surface_samples(*lay.sca, lay.shape, dx)
        om_ff = 2 * np.pi / np.asarray(pattern.wavelengths_nm, dtype=float)
        ff = {"surf": surf, "omega": om_ff, "mon": DFTMonitor(om_ff, surf.e_idx, surf.h1, surf.h2)}
        monitors.append(ff["mon"])
    pml = PMLSpec(cells=((lay.pml, lay.pml),) * 3, kappa_max=opts.pml_kappa,
                  alpha_max=0.05 * prob.medium_index * omega.min(), eps_ref=media[0].eps_inf)
    eng = Engine(device, shape=lay.shape, dx=dx, dt=dt, single=single, inv_eps=inv_eps, groups=groups, pml=pml,
                 sources=[src], monitors=monitors, dft_stride=stride, offdiag=offdiag)
    del inv_eps
    transit = float(np.max(lay.sca[1] - lay.sca[0])) * dx * prob.medium_index
    steps, level, graphs = run_until_decayed(
        eng, opts, max_steps=p["max_steps"], min_steps=int((2 * src.t0 + transit) / dt),
        window=math.ceil(2 * np.pi / omega.min() / (stride * dt)), warnings=warnings)
    j = src.spectrum(omega, stride, steps // stride)
    e_src, _ = mon_src.results()
    p_tot = -0.5 * dx**3 * np.real((e_src @ src.w) * np.conj(j))
    eb, hb = mon_box.results()
    p_rad = flux(eb, hb, box[3])
    out = {"p_tot": p_tot, "p_rad": p_rad, "steps": steps, "decay_level": level, "cuda_graphs": graphs, **vstats}
    if ff is not None:
        ef, hf = ff["mon"].results()
        jf = src.spectrum(ff["omega"], stride, steps // stride)
        out["pattern"] = (ff, ef, hf, jf)
    return out


def _pattern(prob, lay, ff, ef, hf, p0) -> list[RadiationPatternData]:
    req = prob.radiation_pattern
    th = np.radians(np.asarray(req.theta_deg, dtype=float))[:, None]
    ph = np.radians(np.asarray(req.phi_deg, dtype=float))[None, :]
    nhat = np.stack(np.broadcast_arrays(np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)), axis=-1)
    surf = ff["surf"]
    r = lay.origin + surf.pos * lay.dx - np.asarray(prob.dipole.position_nm, dtype=float)
    nb = prob.medium_index
    out = []
    for f, lam in enumerate(req.wavelengths_nm):
        e_far = far_field(surf, ef[f], hf[f], r, nhat.reshape(-1, 3), nb * ff["omega"][f], 1 / nb)
        dp = 0.5 * nb * np.sum(np.abs(e_far) ** 2, axis=-1).reshape(nhat.shape[:2]) / p0[f]
        out.append(RadiationPatternData(wavelength_nm=float(lam), theta_deg=list(req.theta_deg),
                                        phi_deg=list(req.phi_deg), dp_domega=dp.tolist()))
    return out


def solve(prob: EmissionProblem, opts: FDTDOptions, device: Device) -> EmissionResult:
    t_start = time.perf_counter()
    warnings: list[str] = []
    single = uses_single(opts)
    if single:
        warnings.append("single precision: the dipole's power is the small real part of a self-field ~1/(k dx)^3 "
                        "larger, so expect errors of a few percent; precision='double' (the default) avoids them")
    wl_all = _wavelengths(prob)
    media, slot_of = particle_media(prob.particles, prob.medium_index, wl_all, warnings)
    p = plan(prob, opts, single, warnings, media)
    lay = p["layout"]
    if lay.cells > opts.max_cells:
        raise ValueError(f"grid {lay.shape} has {lay.cells} cells > max_cells={opts.max_cells}; "
                         "increase spacing_nm or raise max_cells")
    wl = prob.wavelength_array()
    n_wl = len(wl)
    omega = 2 * np.pi / wl_all  # results first, then the pattern wavelengths (for their P0)
    scene = _run(prob, opts, device, p, single, media, slot_of, omega, True, prob.radiation_pattern, warnings)
    ref = _run(prob, opts, device, p, single, media[:1], [], omega, False, None, warnings)
    p0 = ref["p_tot"]
    balance = float(np.max(np.abs(ref["p_rad"][:n_wl] / p0[:n_wl] - 1)))
    if balance > 0.02:
        warnings.append(f"reference run: radiated and emitted power differ by {balance:.1%}; refine the grid or "
                        "widen pml_gap_nm")
    purcell = scene["p_tot"][:n_wl] / p0[:n_wl]
    radiative = scene["p_rad"][:n_wl] / p0[:n_wl]
    pattern = None
    if "pattern" in scene:
        ff, ef, hf, jf = scene["pattern"]
        # the pattern runs share the source pulse, so P0 at those wavelengths is the reference's p_tot
        pattern = _pattern(prob, lay, ff, ef, hf, p0[n_wl:])
    prov = {**device.describe(), "grid": list(lay.shape), "cells": lay.cells, "spacing_nm": p["dx"],
            "dt_fs": p["dt"] / FS, "steps": scene["steps"], "reference_steps": ref["steps"],
            "dft_stride": p["stride"], "precision": "single" if single else "double", "cuda_graphs": scene["cuda_graphs"],
            "mesh_refinement": mesh_mode(opts), "decay_level": scene["decay_level"],
            "reference_balance": balance, "materials": [m.describe() for m in media],
            **{k: v for k, v in scene.items() if k in ("interface_cells", "dispersive_cells", "offdiagonal_cells", "tensor_vertices")},
            "runtime_s": time.perf_counter() - t_start}
    return EmissionResult(method="fdtd", backend="lightsmith-fdtd", wavelengths_nm=wl.tolist(),
                          purcell=purcell.tolist(), radiative=radiative.tolist(),
                          efficiency=(radiative / purcell).tolist(), radiation_pattern=pattern,
                          provenance=prov, warnings=warnings)

