"""The transverse BPM operator: the FDE engine's Yee-grid operators, as array-API stencils.

The FDE eigenproblem (see ``_fde.modes``) is Q P E_t = n^2 E_t with h = P E_t / n,
so the transverse magnetic field (h = Z0 H) of every mode satisfies
P Q h = n^2 h. BPM marches h_t: d h_t/dw = i k0 sqrt(N) h_t with N = P Q, and FDE
modes are exact eigenvectors of the discrete propagator. H is used rather than
E because tangential H is continuous across every interface (mu = 1): when a
staircased taper moves a sidewall by one pixel between slices, the field
carried across the slice boundary stays smooth, whereas E normal to the wall
would jump by eps_core / eps_clad (the H-field formulation of TM-slab BPM).

With k0-normalised differences (forward D.f on E, backward D.b on H):

    e = Q h:   g = ezz^-1 (Dxb hy - Dyb hx),   ex = myy hy + Dxf g,   ey = -mxx hx + Dyf g
    P e:       hz = mzz^-1 (Dxf ey - Dyf ex),  (P e)_x = -eyy ey - Dxb hz,   (P e)_y = exx ex - Dyb hz

Semi-vectorial BPM keeps one diagonal block of N: quasi-TE (Ex-dominant, so
hy) N_yy = exx myy + exx Dxf ezz^-1 Dxb + Dyb mzz^-1 Dyf myy, quasi-TM (hx)
likewise. In a slab (invariant y) these are the exact in-plane-E (hy) and
out-of-plane-E (hx) operators.

Fields are (..., C, nx, ny) with C the propagated H components (0 = hx,
1 = hy); for the solves they are regrouped into lines along the longer
transverse axis, in which N is block tridiagonal with blocks of
C * (short axis) unknowns. The blocks are read off by probing N with
3 * nb comb vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._fde import Grid


@dataclass(frozen=True)
class Layout:
    """Fields (k, C, nx, ny) <-> lines (n, nb, k) along the longer transverse axis."""

    comps: tuple[int, ...]  # 0 = hx, 1 = hy
    nx: int
    ny: int

    @property
    def along_x(self) -> bool:
        return self.nx >= self.ny

    @property
    def n(self) -> int:
        return self.nx if self.along_x else self.ny

    @property
    def nb(self) -> int:
        return len(self.comps) * (self.ny if self.along_x else self.nx)

    def to_lines(self, xp, f):
        k = f.shape[0]
        if self.along_x:
            g = xp.permute_dims(f, (2, 1, 3, 0))
        else:
            g = xp.permute_dims(f, (3, 1, 2, 0))
        return xp.reshape(g, (self.n, self.nb, k))

    def from_lines(self, xp, g):
        k, c = g.shape[-1], len(self.comps)
        if self.along_x:
            g = xp.reshape(g, (self.nx, c, self.ny, k))
            return xp.permute_dims(g, (3, 1, 0, 2))
        g = xp.reshape(g, (self.ny, c, self.nx, k))
        return xp.permute_dims(g, (3, 1, 2, 0))


def keep_masks(grid: Grid) -> np.ndarray:
    """(2, nx, ny) bool: hx/hy unknowns that are free (normal H vanishes on a PEC min wall, as in the FDE solver)."""
    keep = np.ones((2, *grid.shape), bool)
    if grid.bc_min[0] == "pec":
        keep[0, 0, :] = False
    if grid.bc_min[1] == "pec" and not grid.invariant_y:
        keep[1, :, 0] = False
    return keep


class Operator:
    """M on one cross-section and wavelength, on an array-API device."""

    def __init__(self, grid: Grid, eps: np.ndarray, mu: np.ndarray | None, wavelength_nm: float,
                 comps: tuple[int, ...], xp, asarray, n_pml: float, pml_reflection: float):
        self.grid, self.comps, self.xp = grid, comps, xp
        k0 = 2 * np.pi / wavelength_nm
        self.layout = Layout(comps, grid.nx, grid.ny)
        coef = {}
        for a, (d, n) in enumerate(((grid.dx, grid.nx), (grid.dy, grid.ny))):
            if n == 1:
                coef[a] = None
                continue
            sp = grid.stretch(a, False, k0, n_pml, pml_reflection)
            sd = grid.stretch(a, True, k0, n_pml, pml_reflection)
            cf, cb = 1.0 / (k0 * d * sd), 1.0 / (k0 * d * sp)
            mf, qb = np.ones(n), np.ones(n)
            if grid.bc_min[a] == "pec":
                mf[0] = 0.0
            else:
                qb[0] = 2.0
            shape = (n, 1) if a == 0 else (1, n)
            coef[a] = tuple(asarray(np.reshape(v, shape)) for v in (cf, mf * cf, qb * cb, cb))
        self.coef = coef
        one = np.ones(grid.shape)
        mu = np.stack([one, one, one]) if mu is None else mu
        self.exx, self.eyy, self.ezi = asarray(eps[0]), asarray(eps[1]), asarray(1.0 / eps[2])
        self.mxx, self.myy, self.mzi = asarray(mu[0] * one), asarray(mu[1] * one), asarray(1.0 / mu[2] * one)
        self.keep = asarray(keep_masks(grid)[list(comps)].astype(float))
        wx = [grid.weights(0, d) for d in (False, True)]
        wy = [grid.weights(1, d) for d in (False, True)]
        self.wa = asarray(wx[1][:, None] * wy[0][None, :])  # at Ex / Hy
        self.wb = asarray(wx[0][:, None] * wy[1][None, :])  # at Ey / Hx
        # Power is integrated over the physical cross-section only: in the PML the stretched operators make
        # <E, H*> meaningless (radiation entering it can count as negative power).
        inside = [~grid.pml_mask(a, d) for a, d in ((0, True), (1, False), (0, False), (1, True))]
        self.wa_in = asarray(wx[1][:, None] * wy[0][None, :] * inside[0][:, None] * inside[1][None, :])
        self.wb_in = asarray(wx[0][:, None] * wy[1][None, :] * inside[2][:, None] * inside[3][None, :])
        self.asarray = asarray

    # --- differences (k0-normalised, stretched), matching _fde.modes._diff_1d -----------------

    def _shift(self, u, axis, forward):
        xp = self.xp
        if axis == 0:
            z = xp.zeros_like(u[..., :1, :])
            return xp.concat([u[..., 1:, :], z], axis=-2) if forward else xp.concat([z, u[..., :-1, :]], axis=-2)
        z = xp.zeros_like(u[..., :1])
        return xp.concat([u[..., 1:], z], axis=-1) if forward else xp.concat([z, u[..., :-1]], axis=-1)

    def fwd(self, u, axis):
        c = self.coef[axis]
        if c is None:
            return self.xp.zeros_like(u)
        return self._shift(u, axis, True) * c[0] - u * c[1]

    def bwd(self, v, axis):
        c = self.coef[axis]
        if c is None:
            return self.xp.zeros_like(v)
        return v * c[2] - self._shift(v, axis, False) * c[3]

    def _split(self, f):
        z = self.xp.zeros_like(f[..., 0, :, :])
        hx = f[..., self.comps.index(0), :, :] if 0 in self.comps else z
        hy = f[..., self.comps.index(1), :, :] if 1 in self.comps else z
        return hx, hy

    def q(self, hx, hy):
        """e = Q h (E from H, times n)."""
        g = self.ezi * (self.bwd(hy, 0) - self.bwd(hx, 1))
        return self.myy * hy + self.fwd(g, 0), -self.mxx * hx + self.fwd(g, 1)

    def p(self, ex, ey):
        """h = P e (H from E, times n)."""
        hz = self.mzi * (self.fwd(ey, 0) - self.fwd(ex, 1))
        return -self.eyy * ey - self.bwd(hz, 0), self.exx * ex - self.bwd(hz, 1)

    def apply(self, f):
        """N f = P Q f for f of shape (..., C, nx, ny) in this operator's components (masked)."""
        f = f * self.keep
        hx, hy = self.p(*self.q(*self._split(f)))
        out = [hx if c == 0 else hy for c in self.comps]
        return self.xp.stack(out, axis=-3) * self.keep

    def rayleigh(self, f):
        """(num, den, flux) per right-hand side of f (k, C, nx, ny); n^2 ~ num / den, power ~ flux / n.

        num / den = Re<h, S N h> / <h, S h> with S = diag(1 / eps) of each H component's partner E
        (eyy for hx, exx for hy). In-plane-E slabs have N = exx A with A symmetric, so this is the exact
        generalized Rayleigh quotient (stationary at modes, and its derivative terms sum by parts, so
        a field whose kinks sit at the previous slice's walls gives no spurious spikes); out-of-plane-E
        slabs and 3D are close to it. flux = Re sum(wa conj(hy) ex - wb conj(hx) ey), e = Q h, is n times
        the power <E, H> of a field whose E is Q h / n; it holds first differences only.
        """
        xp = self.xp
        hx, hy = self._split(f * self.keep)
        ex, ey = self.q(hx, hy)
        nx_, ny_ = self.p(ex, ey)
        cx, cy = xp.conj(hx), xp.conj(hy)
        sx, sy = self.wb / self.eyy, self.wa / self.exx
        num = xp.sum(sx * cx * nx_ + sy * cy * ny_, axis=(-2, -1))
        den = xp.sum(sx * cx * hx + sy * cy * hy, axis=(-2, -1))
        flux = xp.sum(self.wa * cy * ex - self.wb * cx * ey, axis=(-2, -1))
        return xp.real(num), xp.real(den), xp.real(flux)

    def power(self, f, y):
        """Re sum(wa conj(hy) (Q y)_x - wb conj(hx) (Q y)_y) outside the PML, per right-hand side: the power
        <E, H*> of the field H = f when y = N^-1/2 f (then E = Q y)."""
        xp = self.xp
        hx, hy = self._split(f * self.keep)
        ex, ey = self.q(*self._split(y * self.keep))
        return xp.real(xp.sum(self.wa_in * xp.conj(hy) * ex - self.wb_in * xp.conj(hx) * ey, axis=(-2, -1)))

    def blocks(self, chunk: int = 96):
        """Block-tridiagonal (A, B, C) of M in the line layout, each (n, nb, nb), by probing."""
        xp, lay = self.xp, self.layout
        n, nb = lay.n, lay.nb
        dtype = self.exx.dtype
        A, B, C = (xp.zeros((n, nb, nb), dtype=dtype, device=_device(self.exx)) for _ in range(3))
        color = np.arange(n) % 3
        for j0 in range(0, nb, chunk):
            js = np.arange(j0, min(nb, j0 + chunk))
            pr = np.zeros((n, nb, 3, len(js)))
            t = np.arange(len(js))[None, :]
            for c in range(3):
                pr[np.flatnonzero(color == c)[:, None], js[None, :], c, t] = 1.0
            pr = self.asarray(pr.reshape(n, nb, 3 * len(js)))
            out = lay.to_lines(xp, self.apply(lay.from_lines(xp, pr)))
            out = xp.reshape(out, (n, nb, 3, len(js)))
            for c in range(3):
                sa, sc = (c + 1) % 3, (c + 2) % 3
                B[c::3, :, j0:j0 + len(js)] = out[c::3, :, c, :]
                A[sa::3, :, j0:j0 + len(js)] = out[sa::3, :, c, :]
                C[sc::3, :, j0:j0 + len(js)] = out[sc::3, :, c, :]
        A[0] = 0
        C[n - 1] = 0
        return A, B, C


def _device(a):
    return getattr(a, "device", "cpu")
