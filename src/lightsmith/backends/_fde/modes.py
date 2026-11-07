"""Full-vector finite-difference eigenmodes on a 2D Yee grid.

Fields vary as exp(i(beta z - omega t)); with h = Z0 H, n = beta / k0 and
derivatives D = d/(k0 dx), Maxwell's curl equations reduce to

    n [hx, hy] = P [Ex, Ey],    n [Ex, Ey] = Q [hx, hy],    (Q P) E = n^2 E

    P = [[ Dxb mzz^-1 Dyf,       -(eyy + Dxb mzz^-1 Dxf)],
         [ exx + Dyb mzz^-1 Dyf, -Dyb mzz^-1 Dxf        ]]
    Q = [[-Dxf ezz^-1 Dyb,        myy + Dxf ezz^-1 Dxb  ],
         [-(mxx + Dyf ezz^-1 Dyb), Dyf ezz^-1 Dxb       ]]

with forward differences acting on E and backward differences on H, i.e. the
Yee-cell scheme of Zhu & Brown, Opt. Express 10, 853 (2002), which is the
same discretisation as the H-field solver of Fallahkhair, Li & Murphy,
J. Lightwave Technol. 26, 1423 (2008) in E-field form. Diagonal mu enters
only through bends: the conformal map rho = R + x of a bend about the y axis
is exactly an anisotropic medium with eps' = eps * diag(h, h, 1/h) and
mu' = diag(h, h, 1/h), h = 1 + (x - x_c)/R (transformation optics). PMLs are
complex coordinate stretches. Eigenpairs come from shift-invert Arnoldi
(ARPACK through scipy) around n_target^2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .grid import Grid

DB = 10 / math.log(10)


def _scipy():
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    return sp, spla


def _diff_1d(n: int, d, wall: str, s_primal: np.ndarray, s_dual: np.ndarray, k0: float, d_dual=None):
    """Forward (primal -> dual) and backward (dual -> primal) differences, scaled by 1/(k0 d s). d: the spacing,
    or the primary widths (n,) of a non-uniform axis, with d_dual the dual widths (n,) of its primal nodes."""
    sp, _ = _scipy()
    if n == 1:
        z = sp.csr_matrix((1, 1), dtype=complex)
        return z, z
    f = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], format="lil", dtype=complex)
    b = sp.diags([np.ones(n), -np.ones(n - 1)], [0, -1], format="lil", dtype=complex)
    if wall == "pec":
        f[0, 0] = 0.0  # tangential E vanishes on the min wall
    else:
        b[0, 0] = 2.0  # PMC: tangential H is odd about the min wall
    f = sp.diags(1.0 / (k0 * d * s_dual)) @ f.tocsr()
    b = sp.diags(1.0 / (k0 * (d if d_dual is None else d_dual) * s_primal)) @ b.tocsr()
    return f.tocsr(), b.tocsr()


@dataclass
class Operators:
    grid: Grid
    k0: float
    dxf: object
    dxb: object
    dyf: object
    dyb: object
    P: object
    Q: object
    keep: np.ndarray  # indices of free DOFs in [Ex, Ey]


def build_operators(grid: Grid, eps: np.ndarray, mu: np.ndarray | None, wavelength_nm: float,
                    pml_reflection: float = 1e-6, n_pml: float = 1.0) -> Operators:
    sp, _ = _scipy()
    k0 = 2 * math.pi / wavelength_nm
    st = [(grid.stretch(a, False, k0, n_pml, pml_reflection), grid.stretch(a, True, k0, n_pml, pml_reflection))
          for a in range(2)]
    d = [((grid.dx, grid.dy)[a], None) if grid.widths[a] is None else (grid.prim_widths(a), grid.dual_widths(a))
         for a in range(2)]
    fx, bx = _diff_1d(grid.nx, d[0][0], grid.bc_min[0], st[0][0], st[0][1], k0, d[0][1])
    fy, by = _diff_1d(grid.ny, d[1][0], grid.bc_min[1], st[1][0], st[1][1], k0, d[1][1])
    ix, iy = sp.identity(grid.nx, format="csr"), sp.identity(grid.ny, format="csr")
    dxf, dxb = sp.kron(fx, iy, format="csr"), sp.kron(bx, iy, format="csr")
    dyf, dyb = sp.kron(ix, fy, format="csr"), sp.kron(ix, by, format="csr")
    D = lambda a: sp.diags(np.ravel(a))  # noqa: E731
    exx, eyy, ezz = (D(e) for e in eps)
    ezi = D(1.0 / eps[2])
    if mu is None:
        mxx = myy = sp.identity(grid.size, format="csr")
        mzi = mxx
    else:
        mxx, myy, mzi = D(mu[0]), D(mu[1]), D(1.0 / mu[2])
    P = sp.bmat([[dxb @ mzi @ dyf, -(eyy + dxb @ mzi @ dxf)],
                 [exx + dyb @ mzi @ dyf, -(dyb @ mzi @ dxf)]], format="csr")
    Q = sp.bmat([[-(dxf @ ezi @ dyb), myy + dxf @ ezi @ dxb],
                 [-(mxx + dyf @ ezi @ dyb), dyf @ ezi @ dxb]], format="csr")
    keep_ex = np.ones(grid.shape, bool)
    keep_ey = np.ones(grid.shape, bool)
    if grid.bc_min[0] == "pec":
        keep_ey[0, :] = False
    if grid.bc_min[1] == "pec" and not grid.invariant_y:
        keep_ex[:, 0] = False
    keep = np.flatnonzero(np.concatenate([keep_ex.ravel(), keep_ey.ravel()]))
    return Operators(grid, k0, dxf, dxb, dyf, dyb, P, Q, keep)


@dataclass
class Mode:
    """One eigenmode. Fields are on their Yee positions, (3, nx, ny), H in units of E (Z0 H).

    Normalised so that the unconjugated overlap ∫(E x H)·z dA = 1 over the whole window
    (mirror images included), with dA in nm^2 (nm for invariant-y slabs). For lossless
    guided modes this is ½ Re∫(E x H*)·z dA = ½.
    """

    neff: complex
    wavelength_nm: float
    grid: Grid
    E: np.ndarray
    H: np.ndarray
    eps: np.ndarray
    mu: np.ndarray | None = None
    residual: float = 0.0
    ng: float | None = None
    meta: dict = field(default_factory=dict)

    # --- integrals ---------------------------------------------------------------

    def _w(self, dual_x: bool, dual_y: bool) -> np.ndarray:
        g = self.grid
        return g.weights(0, dual_x)[:, None] * g.weights(1, dual_y)[None, :]

    def weights(self):
        """Quadrature weights at Ex/Hy, Ey/Hx, Ez and Hz positions."""
        return self._w(True, False), self._w(False, True), self._w(False, False), self._w(True, True)

    def overlap(self, other: "Mode") -> complex:
        """Unconjugated ∫(E_self x H_other)·z dA on a shared grid."""
        wa, wb, _, _ = self.weights()
        return complex(np.sum(wa * self.E[0] * other.H[1]) - np.sum(wb * self.E[1] * other.H[0]))

    def power(self) -> float:
        """½ Re∫(E x H*)·z dA (Z0 = 1 units)."""
        wa, wb, _, _ = self.weights()
        return float(0.5 * np.real(np.sum(wa * self.E[0] * np.conj(self.H[1]))
                                   - np.sum(wb * self.E[1] * np.conj(self.H[0]))))

    @property
    def te_fraction(self) -> float:
        wa, wb, _, _ = self.weights()
        ax, ay = np.sum(wa * np.abs(self.E[0]) ** 2), np.sum(wb * np.abs(self.E[1]) ** 2)
        return float(ax / (ax + ay)) if ax + ay > 0 else 0.5

    @property
    def loss_db_per_cm(self) -> float:
        k0 = 2 * math.pi / self.wavelength_nm
        return float(DB * 2 * k0 * self.neff.imag * 1e7)

    def collocated(self) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """All six components averaged to cell centres (dual x, dual y)."""
        g = self.grid

        def up(a, axis):  # average node i and i+1 (zero beyond the max PEC wall)
            if axis == 1 and g.invariant_y:
                return a
            nxt = np.zeros_like(a)
            if axis == 0:
                nxt[:-1] = a[1:]
            else:
                nxt[:, :-1] = a[:, 1:]
            return 0.5 * (a + nxt)

        E, H = self.E, self.H
        comp = {"Ex": up(E[0], 1), "Ey": up(E[1], 0), "Ez": up(up(E[2], 0), 1),
                "Hx": up(H[0], 0), "Hy": up(H[1], 1), "Hz": H[2]}
        y = g.nodes(1, True) if not g.invariant_y else g.nodes(1, False)
        return g.nodes(0, True), y, comp

    @property
    def effective_area_um2(self) -> float | None:
        """(∫|E|²)² / ∫|E|⁴ over the whole window (None for slabs)."""
        g = self.grid
        if g.invariant_y:
            return None
        _, _, c = self.collocated()
        e2 = sum(np.abs(c[k]) ** 2 for k in ("Ex", "Ey", "Ez"))
        f = 2 ** sum(g.symmetry)  # mirror images
        a = g.dx * g.dy
        return float(f * (e2.sum() * a) ** 2 / ((e2**2).sum() * a) / 1e6)

    def pml_fraction(self) -> float:
        g = self.grid
        m = g.pml_mask(0, False)[:, None] | g.pml_mask(1, False)[None, :]
        e2 = sum(np.abs(c) ** 2 for c in self.E)
        tot = e2.sum()
        return float(e2[m].sum() / tot) if tot > 0 else 0.0

    def group_index_perturbation(self, eps_group: np.ndarray) -> float:
        """c/v_g = ∫(eps_g|E|² + mu|H|²) / (2 Re∫E x H*), exact for the lossless discrete system."""
        wa, wb, wz, wh = self.weights()
        we = (wa, wb, wz)
        wm = (wb, wa, wh)  # Hx with Ey, Hy with Ex, Hz at dual-dual
        mu = self.mu if self.mu is not None else np.ones_like(self.eps.real)
        u = sum(np.sum(w * eg.real * np.abs(e) ** 2) for w, eg, e in zip(we, eps_group, self.E))
        u += sum(np.sum(w * m.real * np.abs(h) ** 2) for w, m, h in zip(wm, mu, self.H))
        flux = 2 * self.power()
        return float(u / (2 * flux))

    def fix_phase_and_normalise(self) -> None:
        wa, wb, _, _ = self.weights()
        i = int(np.argmax([np.max(np.abs(self.E[0]) * (wa > 0)), np.max(np.abs(self.E[1]) * (wb > 0))]))
        c = self.E[i].ravel()[np.argmax(np.abs(self.E[i]).ravel() * ((wa, wb)[i] > 0).ravel())]
        ph = np.conj(c) / abs(c) if abs(c) > 0 else 1.0
        self.E *= ph
        self.H *= ph
        nrm = np.sqrt(self.overlap(self))
        if nrm != 0:
            self.E /= nrm
            self.H /= nrm


def _branch(n2: np.ndarray) -> np.ndarray:
    """n = sqrt(n²) on the forward branch: decaying (Im n > 0), or Re n > 0 when (nearly) lossless."""
    n = np.sqrt(n2.astype(complex))
    flip = (n.imag < -1e-12 * np.abs(n)) | ((np.abs(n.imag) <= 1e-12 * np.abs(n)) & (n.real < 0))
    return np.where(flip, -n, n)


def solve_eigenmodes(grid: Grid, eps: np.ndarray, wavelength_nm: float, num_modes: int,
                     target_neff: complex | None = None, mu: np.ndarray | None = None,
                     polarization: str | None = None, pml_reflection: float = 1e-6,
                     n_pml: float | None = None, dense_below: int = 64) -> list[Mode]:
    """Modes nearest `target_neff` (default: the highest index in the window), by decreasing Re(neff).

    `polarization` ("x" or "y") restricts invariant-y (slab) problems to one decoupled block.
    """
    sp, spla = _scipy()
    if n_pml is None:
        n_pml = float(np.sqrt(np.abs(eps[2][-1, -1])))
    ops = build_operators(grid, eps, mu, wavelength_nm, pml_reflection, n_pml)
    keep = ops.keep
    N = grid.size
    if polarization is not None:
        if not grid.invariant_y:
            raise ValueError("polarization restriction is only exact for invariant (slab) cross-sections")
        keep = keep[keep < N] if polarization == "x" else keep[keep >= N]
    A = (ops.Q @ ops.P).tocsr()[keep][:, keep]
    if target_neff is None:
        target_neff = float(np.sqrt(np.max(eps.real)))
    sigma = complex(target_neff) ** 2 * (1 + 1e-7) + 1e-9
    ndof = len(keep)
    k = max(1, min(num_modes, ndof - 2))
    if ndof <= dense_below:
        vals, vecs = np.linalg.eig(A.toarray())
        order = np.argsort(np.abs(vals - sigma))[:k]
        vals, vecs = vals[order], vecs[:, order]
    else:
        # Shift-invert by hand: minimum-degree ordering on A + A^T roughly halves the LU fill
        # of scipy's default COLAMD for these stencils.
        lu = spla.splu((A - sigma * sp.identity(ndof, format="csc")).tocsc(), permc_spec="MMD_AT_PLUS_A")
        op = spla.LinearOperator((ndof, ndof), matvec=lu.solve, dtype=complex)
        v0 = np.ones(ndof, dtype=complex)
        mu_, vecs = spla.eigs(op, k=k, which="LM", v0=v0, ncv=min(ndof - 1, max(2 * k + 1, 20)))
        vals = sigma + 1.0 / mu_
    neffs = _branch(vals)
    order = np.argsort(-neffs.real)
    modes = []
    for i in order:
        x = vecs[:, i]
        res = float(np.linalg.norm(A @ x - vals[i] * x) / max(np.linalg.norm(vals[i] * x), 1e-300))
        modes.append(_fields(ops, eps, mu, x, keep, neffs[i], wavelength_nm, res))
    return modes


def _fields(ops: Operators, eps, mu, x, keep, neff, wavelength_nm, residual) -> Mode:
    g = ops.grid
    N = g.size
    e = np.zeros(2 * N, dtype=complex)
    e[keep] = x
    h = (ops.P @ e) / neff
    ex, ey = e[:N], e[N:]
    hx, hy = h[:N], h[N:]
    ez = 1j * (ops.dxb @ hy - ops.dyb @ hx) / eps[2].ravel()
    mzz = mu[2].ravel() if mu is not None else 1.0
    hz = -1j * (ops.dxf @ ey - ops.dyf @ ex) / mzz
    E = np.stack([ex, ey, ez]).reshape(3, *g.shape)
    H = np.stack([hx, hy, hz]).reshape(3, *g.shape)
    if g.bc_min[0] == "pec":  # quantities on the x-min PEC wall
        E[1, 0, :] = E[2, 0, :] = H[0, 0, :] = 0
    if g.bc_min[1] == "pec" and not g.invariant_y:
        E[0, :, 0] = E[2, :, 0] = H[1, :, 0] = 0
    m = Mode(complex(neff), float(wavelength_nm), g, E, H, eps, mu, residual)
    m.fix_phase_and_normalise()
    return m


def bend_transform(grid: Grid, eps: np.ndarray, radius_nm: float, x_center_nm: float):
    """Conformal (transformation-optics) map of a bend about the y axis with radius R at x_center.

    Positive R puts the centre of curvature at x_center - R (the mode leaks towards +x).
    """
    hp = 1 + (grid.nodes(0, False) - x_center_nm) / radius_nm
    hd = 1 + (grid.nodes(0, True) - x_center_nm) / radius_nm
    if hp.min() <= 0 or hd.min() <= 0:
        raise ValueError("the window reaches the bend's centre of curvature; shrink it or increase bend_radius_nm")
    hp, hd = hp[:, None], hd[:, None]
    eps_t = np.stack([eps[0] * hd, eps[1] * hp, eps[2] / hp])
    one = np.ones(grid.shape)
    mu = np.stack([one * hp, one * hd, one / hd])  # mu_xx at Hx (primal x), mu_yy at Hy (dual x), mu_zz at Hz (dual x)
    return eps_t, mu


def overlap_matrix(a: list[Mode], b: list[Mode]) -> np.ndarray:
    """Symmetrised power-coupling amplitude ½(<E_i, H_j> + <E_j, H_i>), shape (len(a), len(b))."""
    return np.array([[0.5 * (ma.overlap(mb) + mb.overlap(ma)) for mb in b] for ma in a])
