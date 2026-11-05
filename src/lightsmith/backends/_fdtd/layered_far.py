"""Far field of a particle on a substrate, by reciprocity with the layered background.

Outside the scattered-field box the scattered field is radiated by the equivalent currents J = n x H, M = -n x E
on the box into the layered background (host above z_int, substrate below). Reciprocity with a dipole p far away
in direction nhat gives the far-zone amplitude f (E_sca ~ f e^{i k r} / r, k in the medium on nhat's side):

    p . f = (i omega / 4 pi) sum_S (F_E . J - F_H . M) dA,

where (F_E, F_H) is the background field (incident, reflected and transmitted waves) of the plane wave
p e^{-i k nhat . r} coming in from nhat. In a uniform medium this is the usual near-to-far-field transform. Here
F is the grid's own discrete background (layered.solve) at each sample's Yee position, with the incident wave
referred to the world origin, the phase origin of f. The sums are separable on each face (the transverse phase
along x and y, the 1D solution along z), so every face is a pair of matrix products over all directions.

dC_sca / dOmega = (n_side / n_inc) |f|^2 / |E_0|^2; into an absorbing substrate there is no far field. Each
outgoing background plane wave E_b (the reflected wave, and the transmitted one into a transparent substrate)
loses (2 pi / omega) Im(f(k_b) . E_b^*) of power by interference with the scattered field (the optical theorem
for that wave, f in its own direction); over all of them this is the extinction.
"""

from __future__ import annotations

import numpy as np

from .layered import discrete_eps, graded_rows, layer_bounds, layer_eta, solve
from .media import Medium


def _kz(Om, eta, Kx, Ky, dz, sign):
    """Discrete z wavenumber of a plane wave with transverse K in a uniform medium 1/eta (sign: direction)."""
    Kz = np.sqrt((Om**2 / eta - Kx**2 - Ky**2).astype(complex))
    return sign * (2 / dz) * np.arcsin(Kz * dz / 2)


def basis(nhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two real unit polarisations perpendicular to each direction (n, 3): horizontal (TE) and the other (TM)."""
    nhat = np.asarray(nhat, dtype=float)
    par = np.linalg.norm(nhat[:, :2], axis=1)
    te = np.zeros_like(nhat)
    ok = par > 1e-12
    te[ok, 0], te[ok, 1] = -nhat[ok, 1] / par[ok], nhat[ok, 0] / par[ok]
    te[~ok, 1] = 1.0
    tm = np.cross(te, nhat)
    return te, tm


def hemisphere(n_this: complex, n_other: complex, up: bool, n_theta: int, n_phi: int):
    """Directions (n, 3) and solid-angle weights over one half-space. cos(theta) is split at the critical
    angle (where the far-field pattern has a square-root edge) when the other medium is optically thinner, and
    each piece is mapped by x = a + (b - a)(1 - cos(pi t)) / 2 with Gauss-Legendre in t, which smooths such edges."""
    n_this, n_other = complex(n_this), complex(n_other)
    cuts = [0.0, 1.0]
    if abs(n_other.imag) < 1e-9 and n_other.real < n_this.real:
        cuts = [0.0, float(np.sqrt(1 - (n_other.real / n_this.real) ** 2)), 1.0]
    t, wt = np.polynomial.legendre.leggauss(n_theta)
    t, wt = 0.5 * (t + 1), 0.5 * wt
    xs, ws = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        xs.append(a + (b - a) * (1 - np.cos(np.pi * t)) / 2)
        ws.append(wt * (b - a) * np.pi / 2 * np.sin(np.pi * t))
    ct, w_ct = np.concatenate(xs), np.concatenate(ws)
    ph = 2 * np.pi * np.arange(n_phi) / n_phi
    st = np.sqrt(1 - ct**2)
    d = np.stack([np.outer(st, np.cos(ph)), np.outer(st, np.sin(ph)), np.outer(ct, np.ones(n_phi))], axis=-1)
    if not up:
        d[..., 2] *= -1
    return d.reshape(-1, 3), np.repeat(w_ct * 2 * np.pi / n_phi, n_phi)


class LayeredFarField:
    """Far-zone amplitudes from the DFT samples of a Surface (setup.surface_samples) on the layered grid."""

    def __init__(self, surf, *, origin, dx: float, dt: float, z_int: float, host: Medium, sub: Medium, grid=None):
        """grid: a graded grid (grid.Grid). The z profiles then come from the graded 1D problem and the transverse
        phases from the samples' own positions; the transverse grid wavenumbers K are those of the base spacing,
        where the faces lie, so samples on graded transverse cells carry an O((k h)^2) phase error."""
        self.origin, self.dx, self.dt = np.asarray(origin, dtype=float), float(dx), float(dt)
        self.z_int, self.host, self.sub = float(z_int), host, sub
        self.surf, self.grid = surf, grid
        lo = min(min(f - 1, i1.min(), i2.min()) for _, d, f, _, _, i1, i2 in surf.blocks)
        hi = max(max(f, i1.max(), i2.max()) for _, d, f, _, _, i1, i2 in surf.blocks)
        self.k0 = int(lo) - 2
        self.nk = int(hi) + 2 - self.k0 + 1
        rows = self.k0 + np.arange(self.nk)
        if grid is None:
            self.z_nodes = self.origin[2] + rows * self.dx
            self.dz_prim = self.dz_dual = None
        else:
            self.z_nodes, self.dz_prim, self.dz_dual = graded_rows(grid, 2, rows)
        # per block and lookup (E at the sample, H at face - 1 and at face): per axis the index array and the
        # Yee offset; the field is P_comp(row along z) exp(i omega s . r_par)
        self.blocks = []
        for b, (start, d, face, e_c, h_c, i1, i2) in enumerate(surf.blocks):
            a1, a2 = (d + 1) % 3, (d + 2) % 3
            n = len(i1) * len(i2)
            js = float(surf.jdir[start, e_c])
            ms = float(surf.mdir[start, h_c])
            looks = []
            for kind, comp, dpos in (("E", e_c, face), ("H", h_c, face - 1), ("H", h_c, face)):
                off = np.zeros(3)
                if kind == "E":
                    off[comp] = 0.5
                else:
                    off[:] = 0.5
                    off[comp] = 0.0
                idx = {a1: np.asarray(i1), a2: np.asarray(i2), d: np.array([dpos])}
                looks.append((kind + "xyz"[comp], idx, off))
            area = surf.area[start:start + n].reshape(len(i1), len(i2))
            self.blocks.append((start, n, d, a1, a2, js, ms, looks, area))

    def side_index(self, omega: float, up: bool) -> float | None:
        """Refractive index of the host (up) or substrate at omega; None for an absorbing substrate."""
        eps = complex(discrete_eps(self.host if up else self.sub, np.array([float(omega)]), self.dt)[0])
        return None if abs(eps.imag) > 1e-6 * abs(eps) else float(np.sqrt(eps.real))

    def _factor(self, axis, idx, off, prof, w, s):
        """(B, len(idx)) factor of the field along one axis: the z profile, or the transverse phase."""
        if axis == 2:
            return prof[:, idx - self.k0]
        x = self.origin[axis] + (idx + off) * self.dx if self.grid is None else self.grid.coord(axis, idx + off)
        return np.exp(1j * w * np.outer(s[:, axis], x))

    def amplitudes(self, omega: float, e: np.ndarray, h: np.ndarray, nhat: np.ndarray) -> np.ndarray:
        """f (n, 3) in directions nhat (n, 3) at one frequency, from the surface DFT samples e, h (zero for
        directions into an absorbing substrate)."""
        nhat = np.asarray(nhat, dtype=float)
        out = np.zeros((len(nhat), 3), dtype=complex)
        w = float(omega)
        for up in (True, False):
            sel = np.nonzero((nhat[:, 2] >= 0) == up)[0]
            if not len(sel):
                continue
            n_m = self.side_index(w, up)
            if n_m is None:
                continue
            te, tm = basis(nhat[sel])
            pol = np.concatenate([te, tm])  # (2m, 3)
            nh = np.concatenate([nhat[sel], nhat[sel]])
            s = -n_m * nh[:, :2]  # transverse slowness of the wave coming in from nhat
            A = self._reciprocal(w, s, pol, up, e, h)
            m = len(sel)
            out[sel] = A[:m, None] * te + A[m:, None] * tm
        return out

    def _reciprocal(self, w, s, pol, from_top, e, h):
        """(i omega / 4 pi) sum_S (F_E . J - F_H . M) dA for each incoming wave (transverse slowness s (B, 2),
        polarisation pol (B, 3), tangential part given)."""
        dz = self.dx
        B = len(s)
        Kx = 2 * np.sin(w * s[:, 0] * dz / 2) / dz
        Ky = 2 * np.sin(w * s[:, 1] * dz / 2) / dz
        Om = np.full(B, 2 * np.sin(w * self.dt / 2) / self.dt)
        wv = np.full(B, w)
        if self.grid is None:
            eta_t = layer_eta(self.z_nodes, False, wv, self.dt, dz, self.z_int, self.host, self.sub)
            eta_z = layer_eta(self.z_nodes + dz / 2, True, wv, self.dt, dz, self.z_int, self.host, self.sub)
            f = solve(eta_t, eta_z, Kx, Ky, Om, dz, pol[:, :2], from_top)
        else:
            zt, p = self.z_nodes, self.dz_prim
            eta_t = layer_eta(zt, False, wv, self.dt, dz, self.z_int, self.host, self.sub,
                              bounds=layer_bounds(zt, p, self.dz_dual))
            eta_z = layer_eta(zt + p / 2, True, wv, self.dt, dz, self.z_int, self.host, self.sub, bounds=(zt, zt + p))
            f = solve(eta_t, eta_z, Kx, Ky, Om, self.dz_dual, pol[:, :2], from_top, dzp=p)
        end = self.nk - 1 if from_top else 0
        kz = _kz(Om, eta_t[:, end], Kx, Ky, dz, -1 if from_top else 1)
        ref = np.exp(1j * kz * self.z_nodes[end])  # the incident wave is pol at the origin
        s3 = np.concatenate([s, np.zeros((B, 1))], axis=1)
        tot = np.zeros(B, dtype=complex)
        for start, n, d, a1, a2, js, ms, looks, area in self.blocks:
            n1, n2 = area.shape
            se = (e[start:start + n].reshape(n1, n2) * area)
            sh = (h[start:start + n].reshape(n1, n2) * area)
            for li, (name, idx, off) in enumerate(looks):
                prof = f[name]
                C = self._factor(d, idx[d], off[d], prof, w, s3)[:, 0]
                U = self._factor(a1, idx[a1], off[a1], prof, w, s3)
                V = self._factor(a2, idx[a2], off[a2], prof, w, s3)
                S, coef = (sh, js) if li == 0 else (se, -0.5 * ms)
                tot += coef * C * np.sum(U * (S @ V.T).T, axis=1)
        return (1j * w / (4 * np.pi)) * ref * tot


def outgoing_waves(src, omega: np.ndarray, amp: np.ndarray):
    """Background plane waves leaving the interface (the reflected one, and the transmitted one into a
    transparent substrate) for the run's tabled source `src` (layered.ObliqueLayeredPlaneWave with a substrate),
    each as (khat (n_omega, 3), E at the origin (n_omega, 3), n_medium (n_omega,)) for the incident wave amp x
    the unit solution; None for a wave that is evanescent or absorbed."""
    w = np.asarray(omega, dtype=float)
    f = src._solve(w)
    dz, dt = src.spacing[2], src.dt
    Kx = 2 * np.sin(w * src.s[0] * src.spacing[0] / 2) / src.spacing[0]
    Ky = 2 * np.sin(w * src.s[1] * src.spacing[1] / 2) / src.spacing[1]
    Om = 2 * np.sin(w * dt / 2) / dt
    transverse = np.exp(-1j * w * (src.r_c @ src.s))
    out = []
    for top in (True, False):
        k = src.nk - 1 if top else 0
        med = src.host if top else src.sub
        eps = discrete_eps(med, w, dt)
        if np.any(np.abs(eps.imag) > 1e-6 * np.abs(eps)):
            out.append(None)
            continue
        n_m = np.sqrt(eps.real)
        sz2 = 1 - (src.s @ src.s) / n_m**2
        if np.any(sz2 <= 0):
            out.append(None)
            continue
        u = np.stack([f["Ex"][:, k], f["Ey"][:, k], f["Ez"][:, k]], axis=1)
        if top == src.down:  # the incidence side: remove the incident wave
            u = u - f["inc"]
        kz = _kz(Om, 1 / eps, Kx, Ky, dz, 1 if top else -1)
        u[:, 2] *= np.exp(-1j * kz * dz / 2)  # E_z from z_k + dz / 2 to z_k
        u = u * (np.exp(-1j * kz * src.z_nodes[k]) * transverse * amp)[:, None]
        khat = np.concatenate([np.outer(1 / n_m, src.s), (1 if top else -1) * np.sqrt(sz2)[:, None]], axis=1)
        out.append((khat, u, n_m))
    return out
