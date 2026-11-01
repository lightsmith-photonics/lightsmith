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

from .engine import e_region, h_region

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
                 max_steps: int):
        self.shape, self.dx, self.dt = tuple(shape), float(dx), float(dt)
        self.khat, self.ehat = np.asarray(khat, float), np.asarray(ehat, float)
        self.hhat = np.cross(self.khat, self.ehat)
        self.eps_b = float(eps_b)
        self.axis_aligned = int(np.sum(np.abs(self.khat) > 1e-12)) == 1
        self.eps_1d = self.eps_b if self.axis_aligned else _matched_eps_1d(self.khat, omega_c, dt, dx, eps_b)
        origin, r0 = np.asarray(origin, float), np.asarray(r0, float)
        lo, hi = np.asarray(lo), np.asarray(hi)

        # slabs: (kind, comp, region, coef, s-coordinate of the incident sample, projection)
        raw = []
        for kind in ("E", "H"):
            proj = self.hhat if kind == "E" else self.ehat  # E corrections need H_inc and vice versa
            for c in range(3):
                R = e_region(c, shape) if kind == "E" else h_region(c, shape)
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
                            sq = (origin + q * dx - r0) @ self.khat
                            coef = s * sign_q * delta[i] * (dt / dx) * proj[g]
                            if kind == "E":
                                coef /= eps_b
                            raw.append((kind, c, tuple(region), coef, sq))

        s_all = np.concatenate([r[4].ravel() for r in raw] + [np.zeros(1)])
        ds = dx
        self.m_src = PML1D + 2
        self.m0 = int(self.m_src + 4 + np.ceil(-s_all.min() / ds))
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
