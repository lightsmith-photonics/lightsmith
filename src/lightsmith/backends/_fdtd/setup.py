"""Host-side (NumPy) set-up shared by FDTD problem kinds: grid layout, material
voxelisation with subpixel smoothing, and flux-box monitors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .engine import DispersiveGroup, flat_index
from .media import Medium

FS = 299.792458  # normalised time units (nm/c) per femtosecond
DIVERGED_GROWTH = 1e8  # monitor energy / its peak during the pulse beyond which a run counts as diverged


def e_offset(c: int) -> np.ndarray:
    o = np.zeros(3)
    o[c] = 0.5
    return o


def time_step(dx: float, courant: float, media: list[Medium]) -> float:
    """Courant-stable dt; media with eps_inf < 1 shrink it."""
    eps_min = min([1.0] + [m.eps_inf for m in media])
    return courant * dx * math.sqrt(eps_min) / math.sqrt(3)


# --- voxelisation ---------------------------------------------------------------


_TEST = np.stack(np.meshgrid(*[np.array([-0.5, 0.0, 0.5])] * 3, indexing="ij"), axis=-1).reshape(-1, 3)


def voxelise(slot_fn, media: list[Medium], shape, origin, dx, roi_lo, roi_hi, *, subpixel: bool,
             samples: int = 8, chunk: int = 2048):
    """Material on the Yee E grid.

    slot_fn(points[..., 3] in nm) -> int medium index (0 = background) per point.
    Only nodes in [roi_lo, roi_hi) (per axis) are examined; elsewhere is background.
    Returns (inv_eps[3, *shape] with 1/eps for non-dispersive cells and 1 for D-form cells,
    list of DispersiveGroup, stats dict)."""
    shape = tuple(shape)
    dx = np.broadcast_to(np.asarray(dx, dtype=float), (3,))  # cell size per axis
    S = len(media)
    eps0 = media[0].eps_inf
    inv_eps = np.full((3,) + shape, 1.0 / eps0)
    disp = np.array([m.dispersive for m in media])
    eps_inf = np.array([m.eps_inf for m in media])
    pure_idx = {s: [] for s in range(S) if disp[s]}
    mixed_idx, mixed_f, mixed_w = [], [], []
    n_mixed = 0
    axes_idx = [np.arange(roi_lo[a], roi_hi[a]) for a in range(3)]
    sub = ((np.arange(samples) + 0.5) / samples - 0.5)
    sub3 = np.stack(np.meshgrid(sub, sub, sub, indexing="ij"), axis=-1).reshape(-1, 3)
    for c in range(3):
        idx3 = np.stack(np.meshgrid(*axes_idx, indexing="ij"), axis=-1).reshape(-1, 3)
        pos = origin + (idx3 + e_offset(c)) * dx
        center = slot_fn(pos)
        if subpixel:
            mixed = np.zeros(len(pos), dtype=bool)
            for o in _TEST:
                if np.any(o):
                    mixed |= slot_fn(pos + o * dx) != center
        else:
            mixed = np.zeros(len(pos), dtype=bool)
        # pure cells
        flat = flat_index(c, idx3[:, 0], idx3[:, 1], idx3[:, 2], shape)
        for s in range(1, S):
            sel = (~mixed) & (center == s)
            if not sel.any():
                continue
            if disp[s]:
                pure_idx[s].append(flat[sel])
                inv_eps.reshape(-1)[flat[sel]] = 1.0
            else:
                inv_eps.reshape(-1)[flat[sel]] = 1.0 / eps_inf[s]
        # interface cells: supersample
        mi = np.nonzero(mixed)[0]
        n_mixed += len(mi)
        for a in range(0, len(mi), chunk):
            sel = mi[a:a + chunk]
            pts = pos[sel][:, None, :] + sub3[None, :, :] * dx
            sl = slot_fn(pts)  # (n, samples^3)
            frac = np.stack([(sl == s).mean(axis=1) for s in range(S)])  # (S, n)
            dom = np.argmax(frac, axis=0)
            chi = (sl == dom[:, None]).astype(float)
            mom = (chi - chi.mean(axis=1, keepdims=True)) @ sub3 / dx  # (n, 3) first moment ~ normal
            nrm = np.linalg.norm(mom, axis=1)
            w = np.where(nrm > 1e-12, (mom[:, c] / np.where(nrm > 0, nrm, 1)) ** 2, 1 / 3)
            has_disp = (frac[disp] > 0).any(axis=0) if disp.any() else np.zeros(len(sel), bool)
            f_idx = flat[sel]
            # non-dispersive interface cells: 1/eps_cc = w <1/eps> + (1-w)/<eps>
            nd = ~has_disp
            if nd.any():
                harm = (frac[:, nd] / eps_inf[:, None]).sum(axis=0)
                arit = (frac[:, nd] * eps_inf[:, None]).sum(axis=0)
                inv_eps.reshape(-1)[f_idx[nd]] = w[nd] * harm + (1 - w[nd]) / arit
            if has_disp.any():
                mixed_idx.append(f_idx[has_disp])
                mixed_f.append(frac[:, has_disp])
                mixed_w.append(w[has_disp])
                inv_eps.reshape(-1)[f_idx[has_disp]] = 1.0
    groups = []
    for s, lst in pure_idx.items():
        if lst:
            groups.append(DispersiveGroup(np.concatenate(lst), [media[s]], np.ones((1, sum(map(len, lst)))), None))
    if mixed_idx:
        f = np.concatenate(mixed_f, axis=1)
        used = [s for s in range(S) if np.any(f[s] > 0)]
        groups.append(DispersiveGroup(np.concatenate(mixed_idx), [media[s] for s in used], f[used],
                                      np.concatenate(mixed_w)))
    stats = {"interface_cells": int(n_mixed), "dispersive_cells": int(sum(len(g.idx) for g in groups))}
    return inv_eps, groups, stats


# --- flux boxes -------------------------------------------------------------------


def flux_points(lo, hi, shape, dx: float):
    """Closed box with faces on E-node planes lo[d], hi[d].

    Returns (e_idx, h_idx1, h_idx2, coeff): outward flux = 0.5 Re sum coeff * E * conj(H),
    with E_a1 H_a2 and E_a2 H_a1 sampled at coincident in-plane points and H averaged
    across the face (trapezoid weights along node directions)."""
    E_i, H1, H2, W = [], [], [], []
    for d in range(3):
        a1, a2 = (d + 1) % 3, (d + 2) % 3
        half = np.arange(lo[a1], hi[a1])
        nod2 = np.arange(lo[a2], hi[a2] + 1)
        w2 = np.ones(len(nod2))
        w2[[0, -1]] = 0.5
        nod1 = np.arange(lo[a1], hi[a1] + 1)
        w1 = np.ones(len(nod1))
        w1[[0, -1]] = 0.5
        half2 = np.arange(lo[a2], hi[a2])
        for face, sign in ((lo[d], -1.0), (hi[d], 1.0)):
            for e_comp, h_comp, i1, i2, ww, tsign in ((a1, a2, half, nod2, w2, 1.0), (a2, a1, nod1, half2, w1, -1.0)):
                A, B = np.meshgrid(i1, i2, indexing="ij")
                wgt = np.broadcast_to(ww[None, :] if tsign > 0 else ww[:, None], A.shape)
                idx = [None] * 3
                idx[a1], idx[a2] = A.ravel(), B.ravel()
                idx[d] = np.full(A.size, face)
                E_i.append(flat_index(e_comp, *idx, shape))
                idx[d] = np.full(A.size, face - 1)
                H1.append(flat_index(h_comp, *idx, shape))
                idx[d] = np.full(A.size, face)
                H2.append(flat_index(h_comp, *idx, shape))
                W.append(sign * tsign * dx**2 * wgt.ravel())
    return np.concatenate(E_i), np.concatenate(H1), np.concatenate(H2), np.concatenate(W)


def flux(e: np.ndarray, h: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """0.5 Re sum coeff E H*, per frequency (rows)."""
    return 0.5 * np.real((e * np.conj(h)) @ coeff)


# --- scattering layout --------------------------------------------------------------


@dataclass
class Layout:
    dx: float
    shape: tuple[int, int, int]
    origin: np.ndarray  # world position of node (0, 0, 0)
    center: np.ndarray  # index of the reference node r0
    tfsf: tuple[np.ndarray, np.ndarray]
    tot: tuple[np.ndarray, np.ndarray]
    sca: tuple[np.ndarray, np.ndarray]
    roi: tuple[np.ndarray, np.ndarray]
    pml: int

    @property
    def cells(self) -> int:
        return int(np.prod(self.shape))

    @property
    def r0(self) -> np.ndarray:
        return self.origin + self.center * self.dx


def particle_bbox(sim) -> tuple[np.ndarray, np.ndarray]:
    lo = np.min([np.asarray(p.position_nm) - p.shape.bounding_half_extent() for p in sim.particles], axis=0)
    hi = np.max([np.asarray(p.position_nm) + p.shape.bounding_half_extent() for p in sim.particles], axis=0)
    return lo, hi


def scattering_layout(sim, opts, dx: float) -> Layout:
    lo, hi = particle_bbox(sim)
    ctr = (lo + hi) / 2
    half_cells = np.ceil((hi - lo) / 2 / dx - 1e-9).astype(int) + 1
    n_pad = max(4, math.ceil((opts.padding_nm or 0) / dx))
    n_gap = max(4, math.ceil((opts.pml_gap_nm or 0) / dx))
    half_tfsf = half_cells + n_pad
    half_tot = half_tfsf + 2 + n_gap + opts.pml_cells + 1
    shape = tuple(int(v) for v in 2 * half_tot + 1)
    c = half_tot.astype(int)
    origin = ctr - c * dx
    roi_lo = np.maximum(c - half_cells - 2, 1)
    roi_hi = np.minimum(c + half_cells + 3, np.asarray(shape) - 1)
    return Layout(dx=dx, shape=shape, origin=origin, center=c, tfsf=(c - half_tfsf, c + half_tfsf),
                  tot=(c - half_tfsf + 2, c + half_tfsf - 2), sca=(c - half_tfsf - 2, c + half_tfsf + 2),
                  roi=(roi_lo, roi_hi), pml=opts.pml_cells)


def particle_slot_fn(sim, slot_of: list[list[int]]):
    """pts -> medium index; the first particle listed wins overlaps (as in DDA)."""
    def fn(pts):
        out = np.zeros(pts.shape[:-1], dtype=int)
        for p, slots in zip(sim.particles, slot_of):
            region = p.region(pts - np.asarray(p.position_nm))
            for r, s in enumerate(slots):
                out[(region == r) & (out == 0)] = s
        return out
    return fn


# --- time loop ---------------------------------------------------------------------


def run_until_decayed(eng, opts, *, max_steps: int, min_steps: int, window: int, warnings: list[str]):
    """Run chunks until the monitor energy has fallen below opts.decay_by of its peak (over a sliding
    window of `window` chunks, after `min_steps`), or until max_steps / opts.run_time_fs.
    Raises FloatingPointError when the run diverges (non-finite or runaway monitor energy).
    Returns (steps, decay_level, used_cuda_graphs)."""
    from collections import deque

    from ...device import Device

    run, graphs = eng.make_runner(opts.cuda_graphs)
    win = deque(maxlen=max(3, int(window)))
    peak, ref, level, steps = 0.0, 0.0, float("nan"), 0
    try:
        while steps < max_steps:
            run()
            steps += eng.stride
            e = float(Device.to_numpy(eng.energy)[0])  # one sync per chunk
            # float32 overflows to inf, but float64 (CPU) stays finite to ~1e300: also stop on growth far
            # above the energy seen while the pulse passed (ref), which a passive structure cannot reach.
            if not np.isfinite(e) or (steps > min_steps and e > DIVERGED_GROWTH * ref > 0):
                why = "non-finite fields" if not np.isfinite(e) else \
                    f"monitor energy grew to {e / ref:.0e} times its peak while the pulse passed"
                raise FloatingPointError(
                    f"FDTD diverged after {steps} steps ({why}); the results would be meaningless. "
                    "Use a finer mesh (smaller spacing_nm or larger cells_per_wavelength), lower courant, "
                    "or give lossy/dispersive materials as Drude/DrudeLorentz")
            win.append(e)
            peak = max(peak, e)
            if steps <= min_steps:
                ref = peak
            level = max(win) / peak if peak > 0 else float("nan")
            if not opts.run_time_fs and steps >= min_steps and len(win) == win.maxlen and level < opts.decay_by:
                break
        else:
            if not opts.run_time_fs:
                warnings.append(f"fields had decayed only to {level:.1e} of peak at the time cap "
                                f"({opts.max_time_fs} fs); raise max_time_fs or decay_by")
    finally:
        eng.release()
    return steps, level, graphs
