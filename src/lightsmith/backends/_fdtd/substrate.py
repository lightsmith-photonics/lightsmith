"""Particles on a substrate: the layered background and a plane-wave source in it (normal incidence).

The background is the host above the plane z = z_int and the substrate below. It is meshed layer by layer
on every grid: a node's pixel holds a substrate fraction f, giving 1/<eps> for the tangential components
(E_x, E_y) and <1/eps> for the normal one (E_z), exact for a planar interface and identical in the main grid
and in the auxiliary columns (dispersive media through the same D-form groups).

LayeredPlaneWave is a TFSF box whose incident field (incident, reflected and transmitted waves) comes from
an auxiliary column: a 3D engine 3 x 3 cells wide, periodic in x and y, with the main grid's z nodes, layered
background, CPML and time step, driven by a current sheet outside the box. At normal incidence the background
does not vary in x and y, so the column's fields are the main grid's background field exactly and the box
does not leak, wherever it cuts the interface. A second column of the incidence medium alone, driven by the
same sheet, gives the incident wave by itself (its intensity normalises the cross sections).
"""

from __future__ import annotations

import numpy as np

from .engine import DispersiveGroup, Engine, PMLSpec, e_region, flat_index
from .media import Medium
from .tfsf import tfsf_slabs


def layered_background(shape, origin, spacing, z_int: float, host: Medium, sub: Medium,
                       periodic=(False, False, False)):
    """(inv_eps, groups) of host above z_int and substrate below, on a grid of `shape` (the host must be
    non-dispersive; inv_eps is 1 at D-form nodes)."""
    shape = tuple(shape)
    dz = float(np.broadcast_to(np.asarray(spacing, dtype=float), (3,))[2])
    inv_eps = np.full((3,) + shape, 1.0 / host.eps_inf)
    pure, mixed, mixed_f, mixed_w = [], [], [], []
    for c in range(3):
        R = e_region(c, shape, periodic)
        ks = np.arange(R[2].start, R[2].stop)
        z = origin[2] + (ks + (0.5 if c == 2 else 0.0)) * dz
        f = np.clip((z_int - (z - dz / 2)) / dz, 0.0, 1.0)
        if sub.dispersive:
            for sel, out in ((f >= 1.0, pure), ((f > 0) & (f < 1), mixed)):
                if not sel.any():
                    continue
                gi, gj, gk = np.meshgrid(np.arange(R[0].start, R[0].stop), np.arange(R[1].start, R[1].stop), ks[sel],
                                      indexing="ij")
                idx = flat_index(c, gi.ravel(), gj.ravel(), gk.ravel(), shape)
                out.append(idx)
                inv_eps[c][:, :, ks[sel]] = 1.0
                if out is mixed:
                    fk = np.broadcast_to(f[sel], gk.shape).ravel()
                    mixed_f.append(np.stack([1 - fk, fk]))
                    mixed_w.append(np.full(len(fk), 1.0 if c == 2 else 0.0))
        else:
            e_h, e_s = host.eps_inf, sub.eps_inf
            harm, arit = f / e_s + (1 - f) / e_h, 1 / (f * e_s + (1 - f) * e_h)
            inv_eps[c][:, :, ks] = (harm if c == 2 else arit)[None, None, :]
    groups = []
    if pure:
        idx = np.concatenate(pure)
        groups.append(DispersiveGroup(idx, [sub], np.ones((1, len(idx))), None))
    if mixed:
        groups.append(DispersiveGroup(np.concatenate(mixed), [host, sub], np.concatenate(mixed_f, axis=1),
                                      np.concatenate(mixed_w)))
    return inv_eps, groups


class _View:
    """E_flat / H_flat of a column, for DFTMonitor(source=...)."""

    E_flat = H_flat = None


class LayeredPlaneWave:
    """TFSF box [lo, hi] (node-index units) for a plane wave along -z (down=True) or +z in a layered background.

    ehat: in-plane polarisation. inv_eps: the main grid's 1/eps (scales the E corrections, 1 at D-form nodes).
    The sheet sits at node plane k_launch (outside the box, on the incidence side); the incident-only column
    is probed at k_ref."""

    def __init__(self, *, shape, origin, spacing, dt, lo, hi, down: bool, ehat, inv_eps, z_int: float,
                 host: Medium, sub: Medium, pml: PMLSpec, k_launch: int, k_ref: int, omega_c, omega_half,
                 max_steps: int):
        self.shape, self.dt = tuple(shape), float(dt)
        self.spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,)).copy()
        self.origin = np.asarray(origin, dtype=float)
        nz = self.shape[2]
        self.col_shape = (3, 3, nz)
        per = (True, True, False)
        self.z_int, self.host, self.sub, self.down = float(z_int), host, sub, bool(down)
        self.col_mat = layered_background(self.col_shape, self.origin, self.spacing, z_int, host, sub, per)
        far = 1e30 if not down else -1e30  # incidence medium everywhere
        self.ref_mat = layered_background(self.col_shape, self.origin, self.spacing, far, host, sub, per)
        self.pml = PMLSpec(cells=((0, 0), (0, 0), pml.cells[2]), m=pml.m, kappa_max=pml.kappa_max,
                           alpha_max=pml.alpha_max, eps_ref=pml.eps_ref, eps_ref_high=pml.eps_ref_high)
        self.k_launch, self.k_ref = int(k_launch), int(k_ref)
        self.amp = (float(ehat[0]), float(ehat[1]))
        self.ehat = np.asarray(ehat, dtype=float)
        # corrections: component kind_c in `region` += coef * F_g(q), F from the column at q's z index
        self.slabs = []
        for kind, c, region, g, coef, q in tfsf_slabs(self.shape, lo, hi, dt, self.spacing):
            if g == 2:  # E_z and H_z of the background vanish at normal incidence
                continue
            zq = np.floor(q[..., 2] + 1e-9).astype(np.int64)
            cidx = flat_index(g, 1, 1, zq, self.col_shape)
            shape_ = tuple(r.stop - r.start for r in region)
            arr = coef * (inv_eps[c][region] if kind == "E" else np.ones(shape_))
            self.slabs.append((kind, c, region, arr, cidx.reshape(-1), shape_))
        self.tau = 2.0 / omega_half
        self.t0 = 3.5 * self.tau
        t = (np.arange(max_steps + 2) + 1) * dt
        self.wave = np.exp(-(((t - self.t0) / self.tau) ** 2)) * np.sin(omega_c * (t - self.t0))
        n_max = max(np.sqrt(host.eps_inf), np.sqrt(max(sub.eps_inf, 1.0)))
        self.t_passed = 2 * self.t0 + nz * self.spacing[2] * n_max
        self.col_view, self.ref_view = _View(), _View()

    # --- monitor index sets (flat indices into a column) -----------------------------------------
    def plane_probe(self, k: int):
        """(e, h1, h2) for S_z = E_x H_y - E_y H_x on node plane k, H averaged over k - 1/2 and k + 1/2."""
        n = self.col_shape
        e = np.array([flat_index(0, 1, 1, k, n), flat_index(1, 1, 1, k, n)])
        h1 = np.array([flat_index(1, 1, 1, k - 1, n), flat_index(0, 1, 1, k - 1, n)])
        h2 = np.array([flat_index(1, 1, 1, k, n), flat_index(0, 1, 1, k, n)])
        return e, h1, h2

    def column_e(self):
        """Flat indices of E_x and E_y at every z node of the scene column, (2 * nz,)."""
        k = np.arange(self.col_shape[2])
        return np.concatenate([flat_index(0, 1, 1, k, self.col_shape), flat_index(1, 1, 1, k, self.col_shape)])

    @staticmethod
    def s_z(e, h):
        """Time-averaged S_z per frequency from a plane_probe monitor's results."""
        return 0.5 * np.real(e[:, 0] * np.conj(h[:, 0]) - e[:, 1] * np.conj(h[:, 1]))

    # --- engine hooks ---------------------------------------------------------------------------
    def bind(self, eng):
        def column(mat):
            inv, groups = mat
            return Engine(eng.device, shape=self.col_shape, dx=self.spacing, dt=self.dt, single=eng.single,
                          inv_eps=inv, groups=groups, pml=self.pml, sources=[], monitors=[], dft_stride=1,
                          periodic=(True, True, False))

        self.col, self.ref = column(self.col_mat), column(self.ref_mat)
        del self.col_mat, self.ref_mat
        for v, c in ((self.col_view, self.col), (self.ref_view, self.ref)):
            v.E_flat, v.H_flat = c.E_flat, c.H_flat
        self._wave = eng.asarray(self.wave)
        self._slabs = [(kind, c, region, eng.asarray(arr), eng.index(cidx), shape_)
                       for kind, c, region, arr, cidx, shape_ in self.slabs]

    def _apply(self, eng, kind: str):
        target = eng.E if kind == "E" else eng.H
        src = self.col.H_flat if kind == "E" else self.col.E_flat
        for k, c, region, arr, cidx, shape_ in self._slabs:
            if k == kind:
                target[c][region] += arr * eng.xp.reshape(src[cidx], shape_)

    def correct_H(self, eng):
        self._apply(eng, "H")

    def correct_E(self, eng):
        self._apply(eng, "E")

    def advance_H(self, eng):
        self.col._update_H()
        self.ref._update_H()

    def advance_E(self, eng):
        w = self._wave[eng.n]
        k = self.k_launch
        for col in (self.col, self.ref):
            col._update_E()
            col.E[0][1:2, 1:2, k:k + 1] += self.amp[0] * w
            col.E[1][1:2, 1:2, k:k + 1] += self.amp[1] * w
            for g in col.groups:
                g.step(col)
