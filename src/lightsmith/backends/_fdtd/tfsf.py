"""Total-field/scattered-field plane-wave source fed by a 1D auxiliary FDTD grid.

The incident field is E = e f(s, t), H = h n_b f(s, t) with s = k.(r - r0) and
h = k x e. A 1D Yee grid along s (same dt, spacing dx) carries the scalar E and
H; for axis-aligned incidence its nodes coincide with the 3D Yee positions, so
the 3D grid sees an exactly consistent plane wave and the TFSF boundary does
not leak. Oblique incidence interpolates the 1D grid linearly, with its
permittivity tuned so its numerical phase velocity matches the 3D grid along k
(leakage then is typically 1e-3 of the incident field).

Corrections: a component at p updated from a neighbour q on the other side of
the box gets  +/- coef * F_inc(q) * (T(p) - T(q)), T = 1 inside the box
(boundary included), applied as thin slab slices of the 3D arrays.
"""

from __future__ import annotations

import numpy as np

from .engine import Engine, e_region, flat_index, h_region

PML1D = 48


def _offsets(kind: str, c: int) -> np.ndarray:
    o = np.zeros(3)
    if kind == "E":
        o[c] = 0.5
    else:
        o[:] = 0.5
        o[c] = 0.0
    return o


def _matched_eps_1d(khat, omega, dt, dx, eps_b) -> float:
    """1D permittivity whose Yee phase velocity along khat matches the 3D grid's at omega."""
    lhs = eps_b * (np.sin(omega * dt / 2) / dt) ** 2

    def f(k):
        return np.sum((np.sin(k * khat * dx / 2) / dx) ** 2) - lhs

    lo, hi = 0.0, np.pi / dx
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    k = 0.5 * (lo + hi)
    return float((np.sin(k * dx / 2) / dx) ** 2 / (np.sin(omega * dt / 2) / dt) ** 2)


class TFSFPlaneWave:
    def __init__(self, *, shape, origin, dx, dt, r0, lo, hi, khat, ehat, eps_b, omega_c, omega_half,
                 max_steps: int, periodic=(False, False, False), m0: int | None = None):
        """dx: cell size or (dx, dy, dz). lo/hi: TFSF box in node-index units; faces outside the grid
        (e.g. +-1e9 along periodic axes) produce no corrections, giving a one-sided plane source.
        m0: put s = 0 at least this far along the 1D grid (sources that must reach r0 at the same time)."""
        spacing = np.broadcast_to(np.asarray(dx, dtype=float), (3,)).copy()
        self.shape, self.dx, self.dt = tuple(shape), float(spacing[0]), float(dt)
        self.khat, self.ehat = np.asarray(khat, float), np.asarray(ehat, float)
        self.hhat = np.cross(self.khat, self.ehat)
        self.eps_b = float(eps_b)
        self.axis_aligned = int(np.sum(np.abs(self.khat) > 1e-12)) == 1
        if not self.axis_aligned and len(set(spacing)) > 1:
            raise ValueError("oblique TFSF incidence needs cubic cells")
        self.eps_1d = self.eps_b if self.axis_aligned else _matched_eps_1d(self.khat, omega_c, dt, spacing[0], eps_b)
        origin, r0 = np.asarray(origin, float), np.asarray(r0, float)
        lo, hi = np.asarray(lo), np.asarray(hi)

        # slabs: (kind, comp, region, coef, s-coordinate of the incident sample, projection)
        raw = []
        for kind in ("E", "H"):
            proj = self.hhat if kind == "E" else self.ehat  # E corrections need H_inc and vice versa
            for c in range(3):
                R = e_region(c, shape, periodic) if kind == "E" else h_region(c, shape, periodic)
                off = _offsets(kind, c)
                d1, d2 = (c + 1) % 3, (c + 2) % 3
                for d, g, s in ((d1, d2, 1.0), (d2, d1, -1.0)):
                    if abs(proj[g]) < 1e-12:
                        continue
                    if kind == "H":
                        s = -s
                    trans = {}
                    for a in range(3):
                        if a == d:
                            continue
                        idx = np.arange(R[a].start, R[a].stop)
                        ok = (idx + off[a] >= lo[a]) & (idx + off[a] <= hi[a])
                        trans[a] = idx[ok]
                    if any(len(v) == 0 for v in trans.values()):
                        continue
                    idx_d = np.arange(R[d].start, R[d].stop)
                    u = idx_d + off[d]

                    def inside(x, d=d):
                        return ((x >= lo[d]) & (x <= hi[d])).astype(int)

                    for sign_q, du in ((1.0, 0.5), (-1.0, -0.5)):
                        delta = inside(u) - inside(u + du)
                        for i in np.nonzero(delta)[0]:
                            region = [None] * 3
                            region[d] = slice(int(idx_d[i]), int(idx_d[i]) + 1)
                            axes = []
                            for a in range(3):
                                if a != d:
                                    region[a] = slice(int(trans[a][0]), int(trans[a][-1]) + 1)
                                    axes.append(trans[a] + off[a])
                                else:
                                    axes.append(np.array([u[i] + du]))
                            q = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)  # index coords of q
                            sq = (origin + q * spacing - r0) @ self.khat
                            coef = s * sign_q * delta[i] * (dt / spacing[d]) * proj[g]
                            if kind == "E":
                                coef /= eps_b
                            raw.append((kind, c, tuple(region), coef, sq))

        s_all = np.concatenate([r[4].ravel() for r in raw] + [np.zeros(1)])
        ds = float(spacing[int(np.argmax(np.abs(self.khat)))]) if self.axis_aligned else float(spacing[0])
        self.m_src = PML1D + 2
        self.m0 = max(int(self.m_src + 4 + np.ceil(-s_all.min() / ds)), m0 or 0)
        m_max = self.m0 + int(np.ceil(s_all.max() / ds)) + 4
        self.M = m_max + PML1D + 2
        self.slabs = []
        for kind, c, region, coef, sq in raw:
            f = sq.ravel() / ds + self.m0 - (0.5 if kind == "E" else 0.0)  # E corrections sample H1d
            i0 = np.floor(f + 1e-9).astype(np.int64)
            w = f - i0
            w[np.abs(w) < 1e-9] = 0.0
            shape_ = tuple(r.stop - r.start for r in region)
            self.slabs.append((kind, c, region, float(coef), i0, w if np.any(w) else None, shape_))
        self.s_far = float(s_all.max())

        # 1D grid coefficients with graded absorbing layers at both ends
        sig_max = (3 + 1) * np.log(1e9) / (2 * np.sqrt(self.eps_1d) * PML1D * ds)

        def sig(x):
            rho = np.clip(np.maximum((PML1D - x) / PML1D, (x - (self.M - 1 - PML1D)) / PML1D), 0, 1)
            return sig_max * rho**3

        se = sig(np.arange(self.M, dtype=float)) * dt / 2
        sh = sig(np.arange(self.M, dtype=float) + 0.5) * dt / 2
        self.ca, self.cb = (1 - se) / (1 + se), (dt / (self.eps_1d * ds)) / (1 + se)
        self.da, self.db = (1 - sh) / (1 + sh), (dt / ds) / (1 + sh)

        # Gaussian-modulated sine; E1d^{n+1} gets wave[n] = g((n+1) dt) at m_src
        self.tau = 2.0 / omega_half
        self.t0 = 3.5 * self.tau
        t = (np.arange(max_steps + 2) + 1) * dt
        self.wave = np.exp(-(((t - self.t0) / self.tau) ** 2)) * np.sin(omega_c * (t - self.t0))
        v = 1 / np.sqrt(self.eps_1d)
        self.t_passed = 2 * self.t0 + (self.s_far + (self.m0 - self.m_src) * ds) / v

    # incident-field probe at s = 0: E1d[m0] and H averaged over m0-1/2, m0+1/2
    def probe(self):
        return np.array([self.m0]), np.array([self.m0 - 1]), np.array([self.m0])

    def bind(self, eng):
        self.E_flat = eng.zeros((self.M,))
        self.H_flat = eng.zeros((self.M,))
        a = eng.asarray
        self._ca, self._cb = a(self.ca[1:-1]), a(self.cb[1:-1])
        self._da, self._db = a(self.da[:-1]), a(self.db[:-1])
        self._wave = a(self.wave)
        self._slabs = [(kind, c, region, coef, eng.index(i0), None if w is None else a(w), shape_)
                       for kind, c, region, coef, i0, w, shape_ in self.slabs]

    def _apply(self, eng, kind: str):
        target = eng.E if kind == "E" else eng.H
        src = self.H_flat if kind == "E" else self.E_flat
        for k, c, region, coef, i0, w, shape_ in self._slabs:
            if k != kind:
                continue
            vals = src[i0]
            if w is not None:
                vals = vals + w * (src[i0 + 1] - vals)
            target[c][region] += coef * eng.xp.reshape(vals, shape_)

    def correct_H(self, eng):
        self._apply(eng, "H")

    def correct_E(self, eng):
        self._apply(eng, "E")

    def advance_H(self, eng):
        E, H = self.E_flat, self.H_flat
        H[:-1] *= self._da
        H[:-1] -= self._db * (E[1:] - E[:-1])

    def advance_E(self, eng):
        E, H = self.E_flat, self.H_flat
        E[1:-1] *= self._ca
        E[1:-1] -= self._cb * (H[1:-1] - H[:-2])
        m = self.m_src
        E[m:m + 1] += self._wave[eng.n]


class BlochPlaneSource:
    """One-sided TFSF plane (normal +z) for a Bloch-periodic cell at fixed in-plane wavevector (kx, ky).

    The incident field is computed by an auxiliary column: a 3D engine one cell wide in x and y with the
    same Bloch phases, z spacing, CPML and time step, filled with the background medium and driven by a
    current sheet at node ``k_launch`` above the plane. Its fields times exp(i (kx x + ky y)) satisfy the
    discrete 3D equations of the main grid exactly, so the TFSF plane does not leak, whatever the angle
    and polarisation (which, at fixed (kx, ky), vary with frequency).

    The sheet is along s = (-Ky, Kx) or p = (Kx, Ky), with K = 2 sin(k d / 2) / d the grid wavenumbers,
    so it launches a pure s (TE) or p (TM) wave on the grid. Total field for z-nodes >= k_src.
    """

    def __init__(self, *, shape, spacing, dt, k_src, k_launch, kpar, pol: str, eps_b, pml, omega_c,
                 omega_half, max_steps: int):
        self.shape, self.spacing, self.dt = tuple(shape), tuple(float(d) for d in spacing), float(dt)
        self.k_src, self.k_launch = int(k_src), int(k_launch)
        self.kpar = (float(kpar[0]), float(kpar[1]))
        self.eps_b, self.pml = float(eps_b), pml
        dx, dy = self.spacing[0], self.spacing[1]
        K = np.array([2 * np.sin(self.kpar[a] * self.spacing[a] / 2) / self.spacing[a] for a in (0, 1)])
        u = K / np.linalg.norm(K)
        a = np.array([-u[1], u[0]]) if pol == "s" else u
        # sheet samples at the Ex (x + dx/2) and Ey (y + dy/2) positions of the column cell
        self.amp = (a[0] * np.exp(0.5j * self.kpar[0] * dx), a[1] * np.exp(0.5j * self.kpar[1] * dy))
        nx, ny = shape[0] - 2, shape[1] - 2
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        self.phase = np.exp(1j * (self.kpar[0] * i * dx + self.kpar[1] * j * dy))[:, :, None]
        self.tau = 2.0 / omega_half
        self.t0 = 3.5 * self.tau
        t = (np.arange(max_steps + 2) + 1) * dt
        self.wave = np.exp(-(((t - self.t0) / self.tau) ** 2)) * np.sin(omega_c * (t - self.t0))
        self.t_passed = 2 * self.t0

    def probe(self, k: int):
        """Flat indices (e, h1, h2) into the column's E/H for S_z = Ex Hy - Ey Hx on node plane k."""
        n = (3, 3, self.shape[2])
        e = np.array([flat_index(0, 1, 1, k, n), flat_index(1, 1, 1, k, n)])
        h1 = np.array([flat_index(1, 1, 1, k - 1, n), flat_index(0, 1, 1, k - 1, n)])
        h2 = np.array([flat_index(1, 1, 1, k, n), flat_index(0, 1, 1, k, n)])
        return e, h1, h2

    def bind(self, eng):
        nz = self.shape[2]
        col = Engine(eng.device, shape=(3, 3, nz), dx=self.spacing, dt=self.dt, single=eng.single,
                     inv_eps=np.full((3, 3, 3, nz), 1 / self.eps_b), groups=[], pml=self.pml, sources=[],
                     monitors=[], dft_stride=1, periodic=(True, True, False), bloch=(*self.kpar, 0.0))
        self.col = col
        self.E_flat, self.H_flat = col.E_flat, col.H_flat
        self._phase = eng.asarray(self.phase, dtype=eng.cdt)
        self._amp = [eng.asarray(np.array([v]), dtype=eng.cdt) for v in self.amp]
        self._wave = eng.asarray(self.wave)
        nx, ny = self.shape[0] - 2, self.shape[1] - 2
        self._xy = (slice(1, nx + 1), slice(1, ny + 1))
        self._cE = self.dt / (self.eps_b * self.spacing[2])
        self._cH = self.dt / self.spacing[2]

    def _inc(self, F, c, k):
        return self._phase * F[c][1:2, 1:2, k:k + 1]

    def correct_H(self, eng):
        # H just above the plane (scattered) was updated from total E on the plane: remove E_inc
        k, (X, Y) = self.k_src, self._xy
        E = self.col.E
        eng.H[0][X, Y, k - 1:k] -= self._cH * self._inc(E, 1, k)
        eng.H[1][X, Y, k - 1:k] += self._cH * self._inc(E, 0, k)

    def correct_E(self, eng):
        # E on the plane (total) was updated from scattered H above it: add H_inc
        k, (X, Y) = self.k_src, self._xy
        H = self.col.H
        eng.E[0][X, Y, k:k + 1] += self._cE * self._inc(H, 1, k - 1)
        eng.E[1][X, Y, k:k + 1] -= self._cE * self._inc(H, 0, k - 1)

    def advance_H(self, eng):
        self.col._update_H()

    def advance_E(self, eng):
        self.col._update_E()
        w = self._wave[eng.n]
        k = self.k_launch
        self.col.E[0][1:2, 1:2, k:k + 1] += self._amp[0] * w
        self.col.E[1][1:2, 1:2, k:k + 1] += self._amp[1] * w
