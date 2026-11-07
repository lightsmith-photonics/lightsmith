"""Waveguide ports of FDTD device problems: exact discrete port modes, a one-sided TFSF mode source, modal monitors.

A port is a window on a node plane normal to axis ``a``. In a straight, uniform section of the grid a field
proportional to exp(-i omega t) and exp(i beta x_a) obeys the 3D Yee equations exactly when its transverse
profile is an eigenmode of the 2D Yee cross-section problem (Zhu & Brown, the FDE engine ``_fde``) with

    k0 -> Omega = 2 sin(omega dt / 2) / dt        (leapfrog in time)
    beta -> B = 2 sin(beta dx / 2) / dx            (central differences along the normal)

and every node's permittivity the discrete response of its update at omega (``layered.discrete_eps``, the two
smoothing paths of mixed D-form cells). E_t and H_n then sit on node planes and H_t, E_n half a cell later,
which is where the FDE grid puts them. The FDE walls are the window's edges (PEC), so windows must hold the mode.

Mode tables: per port the modes are solved at Chebyshev-Lobatto frequencies over the source band (nested
sets of 9, 17, 33, ... until the interpolant agrees with the next set), each tracked by overlap with the mode
at the band centre and normalised to unit discrete power. Between nodes they are interpolated (barycentric).

Source: a one-sided TFSF plane on the port window. The incident field at the plane is the forward mode times a
band-limited pulse; per frequency it is an exact solution of the grid, so nothing is launched backwards and no
other mode is excited (to the interpolation error). In time, the incident E_t (on the plane) and H_t (on the half
plane outside it) are a few real basis profiles times scalar time series: the basis spans the mode profiles over
the band (an SVD of the table nodes), the time series are inverse FFTs of the interpolated coefficients.

Monitors: DFTs of E_t on a node plane and H_t on the next half plane (towards local +s). With the discrete
modes' bi-orthogonality under the unconjugated product <e, h> = sum (e x h) . s dA (co-located Yee pairs),
p1 = <E, h_m> and p2 = <e_m, H> give the forward and backward amplitudes exactly:
X = (u^-1/2 p1 + p2) / (2 cos(beta dx/2) N), Y = (u^1/2 p1 - p2) / (2 cos(beta dx/2) N), u = exp(i beta dx),
N = <e_m, h_m>. Unit power: 1/2 <e_m, h_m*> cos(beta dx / 2) = 1, the exact discrete Poynting flux of a mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .._fde.grid import Grid
from .._fde.modes import solve_eigenmodes
from .engine import TensorMesh, flat_index
from .layered import discrete_eps


@dataclass(frozen=True)
class PortFrame:
    """A port on the grid. Local axes (t1, t2, s) are right-handed with s = sgn * x_a; the FDE window spans
    primal nodes lo .. lo + n along t1 and t2 (both ends are its PEC walls; n[1] = 1 for an invariant axis)."""

    name: str
    a: int
    t1: int
    t2: int
    sgn: int
    d: int  # +1 / -1: into the device along +x_a / -x_a
    k_p: int  # node plane of the TFSF source (E_t there is total field)
    k_m: int  # node plane of the monitor's E_t
    lo: tuple[int, int]
    n: tuple[int, int]
    s_ref: float  # local s of the reference plane (the port centre)
    s_m: float  # local s of the monitor plane
    num_modes: int
    shape: tuple[int, int, int]
    # graded grids: the window's primary widths along t1 and t2 (n[0] and n[1] cells), None where uniform
    widths: tuple = (None, None)
    ds: float = 0.0  # cell width along the normal around the port (uniform there)

    @property
    def delta(self) -> int:
        """+1 when the incoming wave travels along local +s (the FDE forward mode)."""
        return self.sgn * self.d

    def labels(self) -> list[str]:
        return [self.name] if self.num_modes == 1 else [f"{self.name}@{m}" for m in range(self.num_modes)]

    def nodes(self, c: int, k: int) -> np.ndarray:
        """Flat indices of component c at index k along a, over the window, (n1 * n2,) in (i, j) order."""
        i, j = np.meshgrid(np.arange(self.n[0]) + self.lo[0], np.arange(self.n[1]) + self.lo[1], indexing="ij")
        idx = [None, None, None]
        idx[self.t1], idx[self.t2], idx[self.a] = i.ravel(), j.ravel(), np.full(i.size, k)
        return np.asarray(flat_index(c, *idx, self.shape), dtype=np.int64)

    def half(self, k: int) -> int:
        """Index along a of the half plane at local s + 1/2 from node plane k (H_t, E_a there)."""
        return k if self.sgn > 0 else k - 1

    def e_nodes(self, k: int) -> np.ndarray:
        return np.concatenate([self.nodes(self.t1, k), self.nodes(self.t2, k)])

    def h_nodes(self, k: int) -> np.ndarray:
        return np.concatenate([self.nodes(self.t1, self.half(k)), self.nodes(self.t2, self.half(k))])

    def grid(self, spacing: float) -> Grid:
        """The FDE window; `spacing` is the uniform transverse spacing (the widths override it per axis)."""
        if self.n[1] == 1:
            return Grid(0.0, 0.0, spacing, 1.0, self.n[0], 1, widths=(self.widths[0], None))
        return Grid(0.0, 0.0, spacing, spacing, self.n[0], self.n[1], widths=tuple(self.widths))


def node_eta(inv_eps: np.ndarray, groups: list, flat: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """1/eps (n_omega, n) of E nodes `flat` at omega as the update sees it: 1/eps of plain cells, the discrete ADE
    response of D-form cells (mixed cells through both smoothing paths). Vertex-tensor nodes are not diagonal."""
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    eta = np.repeat(inv_eps.reshape(-1)[flat][None, :].astype(complex), len(omega), axis=0)
    for g in groups:
        if isinstance(g, TensorMesh):
            if np.isin(flat, g.idx).any():
                raise ValueError("a port plane crosses vertex-tensor cells")
            continue
        hit = np.isin(flat, g.idx)
        if not hit.any():
            continue
        order = np.argsort(g.idx)
        loc = order[np.searchsorted(g.idx, flat[hit], sorter=order)]
        eps = np.stack([discrete_eps(m, omega, dt) for m in g.media])  # (S, n_omega)
        if g.w is None:
            eta[:, hit] = 1.0 / eps[0][:, None]
            continue
        f, w = g.frac[:, loc], g.w[loc]
        harm = np.einsum("sn,sw->wn", f, 1.0 / eps)
        arit = np.einsum("sn,sw->wn", f, eps)
        eta[:, hit] = w[None, :] * harm + (1 - w[None, :]) / arit
    return eta


# --- mode tables ---------------------------------------------------------------------------------


def _lobatto(n: int, lo: float, hi: float) -> np.ndarray:
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(np.pi * np.arange(n) / (n - 1))


def _bary(nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Barycentric interpolation matrix (len(x), len(nodes)) on Chebyshev-Lobatto nodes."""
    n = len(nodes)
    w = (-1.0) ** np.arange(n)
    w[0] *= 0.5
    w[-1] *= 0.5
    x = np.asarray(x, dtype=float)
    diff = x[:, None] - nodes[None, :]
    exact = np.abs(diff) < 1e-14 * max(1.0, float(np.abs(nodes).max()))
    diff[exact] = 1.0
    M = w[None, :] / diff
    M /= M.sum(axis=1, keepdims=True)
    rows = exact.any(axis=1)
    M[rows] = exact[rows].astype(float)
    return M


@dataclass
class ModeTable:
    """Port modes over [omega_a, omega_b]: unit-power transverse profiles, discrete beta, smoothly tracked."""

    frame: PortFrame
    spacing: float
    dt: float
    omega_a: float
    omega_b: float
    eps_fn: object  # omega (n,) -> eps (n, 3, n1, n2) for local (t1, t2, s)
    polarization: str | None = None  # invariant (2D) windows: "x" (E along t1) or "y" (E along t2)
    tol: float = 1e-7
    max_nodes: int = 65
    nodes: np.ndarray = field(default=None, repr=False)
    e: np.ndarray = field(default=None, repr=False)  # (n_nodes, n_modes, 2, n1, n2) e_t1, e_t2
    h: np.ndarray = field(default=None, repr=False)  # (n_nodes, n_modes, 2, n1, n2) h_t1, h_t2
    beta: np.ndarray = field(default=None, repr=False)  # (n_nodes, n_modes)
    neff: np.ndarray = field(default=None, repr=False)  # FDE neff at the nodes (B / Omega)
    interp_error: float = float("nan")
    confinement: np.ndarray = field(default=None, repr=False)  # (n_nodes, n_modes) field fraction at the walls

    def __post_init__(self):
        g = self.frame.grid(self.spacing)
        self._grid = g
        wa = g.weights(0, True)[:, None] * g.weights(1, False)[None, :]
        wb = g.weights(0, False)[:, None] * g.weights(1, True)[None, :]
        self.wa, self.wb = wa, wb
        self._cache: dict[float, tuple] = {}
        self.reference = None
        self._build()

    def overlap(self, e, h) -> np.ndarray:
        """<e, h> = sum wa e_t1 h_t2 - wb e_t2 h_t1 (unconjugated), over the trailing (2, n1, n2) axes."""
        return (np.sum(self.wa * e[..., 0, :, :] * h[..., 1, :, :], axis=(-2, -1))
                - np.sum(self.wb * e[..., 1, :, :] * h[..., 0, :, :], axis=(-2, -1)))

    def _solve(self, w: float):
        key = float(w)
        if key in self._cache:
            return self._cache[key]
        Om = 2 * math.sin(w * self.dt / 2) / self.dt
        eps = self.eps_fn(np.array([w]))[0]
        nm = self.frame.num_modes
        modes = solve_eigenmodes(self._grid, eps, 2 * math.pi / Om, nm + 2, polarization=self.polarization)
        es, hs, betas, neffs, conf = [], [], [], [], []
        for m in modes:
            B = Om * m.neff
            beta = 2 / self.spacing * np.arcsin(B * self.spacing / 2 + 0j)
            c = np.cos(beta * self.spacing / 2)
            s = np.sqrt(2 / c)  # FDE modes have <e, h> = 1: now 1/2 <e, h> cos(beta dx/2) = 1
            es.append(np.stack([m.E[0], m.E[1]]) * s)
            hs.append(np.stack([m.H[0], m.H[1]]) * s)
            betas.append(beta)
            neffs.append(m.neff)
            e2 = np.abs(m.E[0]) ** 2 + np.abs(m.E[1]) ** 2 + np.abs(m.E[2]) ** 2
            rim = np.zeros(e2.shape, dtype=bool)
            rim[:2], rim[-2:] = True, True
            if e2.shape[1] > 1:
                rim[:, :2], rim[:, -2:] = True, True
            conf.append(float(e2[rim].sum() / max(e2.sum(), 1e-300)))
        out = (np.array(es), np.array(hs), np.array(betas), np.array(neffs), np.array(conf))
        self._cache[key] = out
        return out

    def _tracked(self, w: float):
        """The table's modes at w, in the reference order, each with its phase aligned to the reference."""
        es, hs, betas, neffs, conf = self._solve(w)
        nm = self.frame.num_modes
        if self.reference is None:
            if len(betas) < nm:
                raise ValueError(f"port {self.frame.name!r}: only {len(betas)} mode(s) found")
            self.reference = (es[:nm], hs[:nm])
        re, rh = self.reference
        ov = np.array([[0.5 * (self.overlap(re[i], hs[j]) + self.overlap(es[j], rh[i])) for j in range(len(betas))]
                       for i in range(nm)])
        pick = []
        for i in range(nm):
            j = int(np.argmax(np.abs(ov[i]) * np.array([k not in pick for k in range(len(betas))])))
            pick.append(j)
        ph = np.array([np.conj(ov[i, j]) / abs(ov[i, j]) if abs(ov[i, j]) > 0 else 1.0 for i, j in enumerate(pick)])
        return (es[pick] * ph[:, None, None, None], hs[pick] * ph[:, None, None, None], betas[pick], neffs[pick],
                conf[pick], np.abs(ov[np.arange(nm), pick]))

    def _build(self):
        mid = 0.5 * (self.omega_a + self.omega_b)
        self._tracked(mid)  # sets the reference
        n = 9
        prev = None
        while True:
            nodes = _lobatto(n, self.omega_b, self.omega_a)  # descending cos: from omega_b to omega_a
            data = [self._tracked(w) for w in nodes]
            e = np.array([d[0] for d in data])
            h = np.array([d[1] for d in data])
            beta = np.array([d[2] for d in data])
            if prev is not None:
                # the previous (coarser) interpolant at the new nodes in between
                pn, pe, ph, pb = prev
                M = _bary(pn, nodes[1::2])
                err_e = np.abs(np.tensordot(M, pe, axes=(1, 0)) - e[1::2]).max() / np.abs(e).max()
                err_h = np.abs(np.tensordot(M, ph, axes=(1, 0)) - h[1::2]).max() / np.abs(h).max()
                err_b = np.abs(M @ pb - beta[1::2]).max() / np.abs(beta).max()
                self.interp_error = float(max(err_e, err_h, err_b))
                if self.interp_error < self.tol or 2 * n - 1 > self.max_nodes:
                    break
            prev = (nodes, e, h, beta)
            n = 2 * n - 1
        self.nodes, self.e, self.h, self.beta = nodes, e, h, beta
        self.neff = np.array([d[3] for d in data])
        self.confinement = np.array([d[4] for d in data])
        self.track_overlap = float(np.min([d[5] for d in data]))

    def at(self, omega) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(e, h, beta) interpolated at omega: (n_w, n_modes, 2, n1, n2) twice and (n_w, n_modes)."""
        M = _bary(self.nodes, np.atleast_1d(omega))
        return (np.tensordot(M, self.e, axes=(1, 0)), np.tensordot(M, self.h, axes=(1, 0)), M @ self.beta)

    def amplitudes(self, omega, E: np.ndarray, H: np.ndarray, mode: int) -> tuple[np.ndarray, np.ndarray]:
        """(incoming, outgoing) amplitudes of `mode` at the reference plane, from monitor DFTs E, H (n_w, 2 n1 n2)."""
        fr = self.frame
        sh = (len(omega), 2) + fr.n
        E, H = E.reshape(sh), H.reshape(sh)
        e, h, beta = self.at(omega)
        e, h, beta = e[:, mode], h[:, mode], beta[:, mode]
        N = self.overlap(e, h)
        p1, p2 = self.overlap(E, h), self.overlap(e, H)
        u2 = np.exp(0.5j * beta * self.spacing)
        c = np.cos(0.5 * beta * self.spacing)
        X = (p1 / u2 + p2) / (2 * c * N)
        Y = (u2 * p1 - p2) / (2 * c * N)
        ds = fr.s_ref - fr.s_m
        fwd, bwd = X * np.exp(1j * beta * ds), Y * np.exp(-1j * beta * ds)
        return (fwd, bwd) if fr.delta > 0 else (bwd, fwd)


# --- the source ----------------------------------------------------------------------------------


def _basis(V: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Real orthonormal basis (n, r) of the complex columns of V (n, m)."""
    R = np.concatenate([V.real, V.imag], axis=1)
    U, s, _ = np.linalg.svd(R, full_matrices=False)
    r = int(np.sum(s > rtol * s[0])) if s[0] > 0 else 1
    return U[:, :max(r, 1)]


class ModeSource:
    """One-sided TFSF plane on a port window launching mode `mode` of `table` into the device.

    The pulse spectrum is a Gaussian of width sigma about omega_c, cut off with a raised-cosine taper at the
    table's band edges. Total field on the device side of node plane k_p."""

    def __init__(self, table: ModeTable, mode: int, *, inv_eps: np.ndarray, dt: float, omega_c: float,
                 sigma: float, n_steps: int):
        fr = table.frame
        self.table, self.mode, self.dt = table, mode, float(dt)
        dx = table.spacing
        nw = fr.n[0] * fr.n[1]
        # node basis: incident e on the plane, h on the half plane outside it (H_inc = delta h u^-1/2)
        e_nodes = table.e[:, mode].reshape(len(table.nodes), -1).T  # (2 nw, n_nodes)
        h_nodes = table.h[:, mode].reshape(len(table.nodes), -1).T
        self.UE, self.UH = _basis(e_nodes), _basis(h_nodes)
        cE_nodes = self.UE.T @ e_nodes  # (rE, n_nodes) complex
        cH_nodes = self.UH.T @ h_nodes
        self.rank = (self.UE.shape[1], self.UH.shape[1])
        # time tables by inverse FFT (numpy's e^{+i w t}: the e^{-i w t} response enters conjugated)
        lo, hi = table.omega_a, table.omega_b
        self.t0 = 9.0 / sigma  # the envelope exp(-(sigma t / 2)^2) starts at e^-20
        N = 1 << int(math.ceil(math.log2(4 * self.t0 / dt)))
        while True:
            om = 2 * np.pi * np.arange(N // 2 + 1) / (N * dt)
            keep = (om > lo) & (om < hi)
            w = om[keep]
            taper = np.ones(len(w))
            edge = 0.5 * sigma
            for d_ in (w - lo, hi - w):
                t = np.clip(d_ / edge, 0, 1)
                taper *= 0.5 * (1 - np.cos(np.pi * t))
            X = np.exp(-(((w - omega_c) / sigma) ** 2)) * taper * np.exp(-1j * w * self.t0)
            M = _bary(table.nodes, w)
            cE = M @ cE_nodes.T  # (n_w, rE)
            cH = (M @ cH_nodes.T) * (fr.delta * np.exp(-0.5j * (M @ table.beta[:, mode]) * dx))[:, None]
            spec_E = np.zeros((self.rank[0], len(om)), dtype=complex)
            spec_H = np.zeros((self.rank[1], len(om)), dtype=complex)
            spec_E[:, keep] = (np.conj(cE) * X[:, None]).T
            spec_H[:, keep] = (np.conj(cH) * (X * np.exp(0.5j * w * dt))[:, None]).T  # H at (n + 1/2) dt
            tE = np.fft.irfft(spec_E, n=N, axis=-1)
            tH = np.fft.irfft(spec_H, n=N, axis=-1)
            env = np.abs(tE).max(axis=0)
            if env[-N // 8:].max() < 1e-7 * env.max() or N >= 1 << 22:
                break
            N *= 2
        self.t_off = float(np.nonzero(env > 1e-6 * env.max())[0][-1] * dt)  # the pulse has passed (a 1e-7 tail remains)
        L = max(N, n_steps + 2)
        self.tab_E = np.zeros((L, self.rank[0]))
        self.tab_H = np.zeros((L, self.rank[1]))
        self.tab_E[:N], self.tab_H[:N] = tE.T, tH.T
        self.spectrum_peak = float(np.abs(X).max()) if len(X) else 0.0
        # targets: H_t2 += delta (dt/dx) E_inc,t1, H_t1 -= delta (dt/dx) E_inc,t2 on the outside half plane;
        # E_t1 += delta (dt/dx) eta H_inc,t2, E_t2 -= delta (dt/dx) eta H_inc,t1 on the plane
        k = fr.k_p
        k_out = k - 1 if fr.d > 0 else k  # H index of the half plane outside the plane
        ht1 = fr.nodes(fr.t1, k_out)
        ht2 = fr.nodes(fr.t2, k_out)
        et1, et2 = fr.nodes(fr.t1, k), fr.nodes(fr.t2, k)
        c = fr.delta * dt / dx
        UE1, UE2 = self.UE[:nw], self.UE[nw:]
        UH1, UH2 = self.UH[:nw], self.UH[nw:]
        self.h_idx = np.concatenate([ht2, ht1])
        self.mH = np.concatenate([c * UE1, -c * UE2])  # (2 nw, rE): H targets from E_inc
        eta = np.concatenate([inv_eps.reshape(-1)[et1], inv_eps.reshape(-1)[et2]])
        self.e_idx = np.concatenate([et1, et2])
        self.mE = np.concatenate([c * UH2, -c * UH1]) * eta[:, None]  # (2 nw, rH): E targets from H_inc

    def bind(self, eng):
        a = eng.asarray
        self._mH, self._mE = a(self.mH), a(self.mE)
        self._tE, self._tH = a(self.tab_E), a(self.tab_H)
        self._h, self._e = eng.index(self.h_idx), eng.index(self.e_idx)

    def correct_H(self, eng):
        eng.H_flat[self._h] += eng.xp.sum(self._mH * self._tE[eng.n], axis=1)

    def correct_E(self, eng):
        eng.E_flat[self._e] += eng.xp.sum(self._mE * self._tH[eng.n], axis=1)

    def advance_H(self, eng):
        pass

    def advance_E(self, eng):
        pass
