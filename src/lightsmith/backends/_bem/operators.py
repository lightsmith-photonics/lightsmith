"""PMCHWT system assembly on RWG functions, written against the array API.

Fields radiated by surface currents in a homogeneous region (exp(-i omega t),
G = e^{ikR}/(4 pi R), lengths in nm, impedances relative to vacuum)::

    E = eta L[J] - K[M],     H = K[J] + L[M]/eta
    L[X] = ik int G X dS' + (i/k) grad int G div'X dS',    K[X] = curl int G X dS'

With J' = eta_0 J (eta_0: host impedance), tangential continuity of E and H on
every interface, Galerkin-tested with the RWG functions, gives for each pair of
surfaces (m, n) and each region r they both bound (s = +1 if r is on a
surface's normal side, -1 otherwise)::

    [ (eta_r/eta_0) L_r   -K_r            ] [J'_n]   = -[<f_m, E_inc>         ]
    [  K_r                (eta_0/eta_r) L_r] [M_n ]      [<f_m, eta_0 H_inc>  ]   (times s_m s_n, summed over r)

the PMCHWT formulation (Poggio & Miller 1973; Chang & Harrington 1977; Wu & Tsai
1977) generalised to several regions, which stays well conditioned for metals
(negative Re eps) because no operator is inverted on its own; cf. Kern & Martin,
JOSA A 26, 732 (2009). The +-1/2 jump terms of K cancel between the two sides of
each interface, so K is taken as a principal value.

Matrix elements, with RWG f = sigma l/(2A) (r - v_i) on a triangle (`coef` = sigma l/(2A))::

    <f_m, L f_n> = c_m c_n [ik int int (r - v_i).(r' - v_j) G - (4i/k) int int G]
    <f_m, K f_n> = c_m c_n int int grad G(r - r') . ((r' - v_j) x (r - v_i))
                 = c_m c_n int int grad G(r - r') . ((r - v_j) x (r - v_i))

(the last form because grad G is parallel to r - r'). Inner integrals use a
triangle quadrature; for nearby triangle pairs the static part 1/(4 pi R) (and its
gradient) is integrated analytically and its quadrature subtracted
(singularity extraction, `singular.py`). Those corrections do not depend on k and
are computed once per mesh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mesh import Mesh
from .singular import static_integrals

# Dunavant rules on the triangle: barycentric points, weights summing to 1.
_A7, _B7 = 0.059715871789770, 0.470142064105115
_C7, _D7 = 0.797426985353087, 0.101286507323456
_A12, _B12 = 0.501426509658179, 0.249286745170910
_C12, _D12 = 0.873821971016996, 0.063089014491502
_E12, _F12, _G12 = 0.053145049844817, 0.310352451033784, 0.636502499121399
QUADRATURE = {
    1: (np.array([[1 / 3, 1 / 3, 1 / 3]]), np.array([1.0])),
    3: (np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]), np.full(3, 1 / 3)),
    7: (np.array([[1 / 3, 1 / 3, 1 / 3], [_A7, _B7, _B7], [_B7, _A7, _B7], [_B7, _B7, _A7],
                  [_C7, _D7, _D7], [_D7, _C7, _D7], [_D7, _D7, _C7]]),
        np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3)),
    12: (np.array([[_A12, _B12, _B12], [_B12, _A12, _B12], [_B12, _B12, _A12],
                   [_C12, _D12, _D12], [_D12, _C12, _D12], [_D12, _D12, _C12],
                   [_E12, _F12, _G12], [_E12, _G12, _F12], [_F12, _E12, _G12],
                   [_F12, _G12, _E12], [_G12, _E12, _F12], [_G12, _F12, _E12]]),
         np.array([0.116786275726379] * 3 + [0.050844906370207] * 3 + [0.082851075618374] * 6)),
}

FOUR_PI = 4 * math.pi


def cross(a, b, xp):
    return xp.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def pair_blocks(G0, G1, Gd, u, w, delta, a, b, xp):
    """Triangle-pair Galerkin blocks from inner integrals at the outer quadrature points.

    Inner integrals over the source triangle q at outer point r_o (axis -1 of G0, -2 of G1/Gd):
    G0 = int G, G1 = int G (r' - c_q), Gd = int grad_r G. u = r_o - c_p, w: outer weights,
    delta = c_p - c_q, a/b: vertices minus centroid of the test/source triangle ([i or j, xyz]).
    Returns RR[..., i, j] = int int (r - v_i).(r' - v_j) G, M0 = int int G and
    KK[..., i, j] = int int grad G . ((r - v_j) x (r - v_i)).
    """
    wg = w * G0
    M0 = xp.sum(wg, axis=-1)
    Mu = xp.sum(wg[..., None] * u, axis=-2)
    M1 = xp.sum(w[..., None] * G1, axis=-2)
    Mu1 = xp.sum(w * xp.sum(u * G1, axis=-1), axis=-1)
    wd = w[..., None] * Gd
    ND = xp.sum(wd, axis=-2)
    NC = xp.sum(cross(u, wd, xp), axis=-2)
    NB = xp.sum(cross(wd, u + delta[..., None, :], xp), axis=-2)
    NA = xp.sum(xp.sum(wd * cross(delta[..., None, :], u, xp), axis=-1), axis=-1)
    ai, bj = a[..., :, None, :], b[..., None, :, :]
    RR = (Mu1[..., None, None] - xp.sum(Mu[..., None, None, :] * bj, axis=-1)
          - xp.sum(ai * M1[..., None, None, :], axis=-1) + xp.sum(ai * bj, axis=-1) * M0[..., None, None])
    KK = (NA[..., None, None] - xp.sum(ai * NB[..., None, None, :], axis=-1)
          - xp.sum(bj * NC[..., None, None, :], axis=-1) + xp.sum(cross(bj, ai, xp) * ND[..., None, None, :], axis=-1))
    return RR, M0, KK


def close_pairs(x: np.ndarray, y: np.ndarray, rx, ry, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i, j) with |x_i - y_j| < rx_i + ry_j (chunked brute force; cheap next to assembly)."""
    rx = np.broadcast_to(np.asarray(rx, dtype=float), (len(x),))
    ry = np.broadcast_to(np.asarray(ry, dtype=float), (len(y),))
    out_i, out_j = [], []
    for s in range(0, len(x), chunk):
        d2 = np.sum((x[s:s + chunk, None, :] - y[None, :, :]) ** 2, axis=-1)
        i, j = np.nonzero(d2 < (rx[s:s + chunk, None] + ry[None, :]) ** 2)
        out_i.append(i + s)
        out_j.append(j)
    return np.concatenate(out_i), np.concatenate(out_j)


@dataclass
class Region:
    index: int
    tris: np.ndarray  # global triangle indices of its bounding surfaces
    edges: np.ndarray  # global edge indices (sorted)
    sign: np.ndarray  # per local edge: +1 if the region is on the normal side
    runs: list  # contiguous (global start, stop, local start) ranges of `edges`
    tri_edge: np.ndarray  # (T_r, 3) local edge index
    col_plus: np.ndarray  # per local edge: flattened (local tri * 3 + loc) of its plus half
    col_minus: np.ndarray
    corr: tuple  # (rows, cols, A, B, C) local sparse singular corrections


class Geometry:
    """Everything about a mesh that does not depend on wavelength (NumPy)."""

    def __init__(self, mesh: Mesh, order: int = 7, near_factor: float = 2.0):
        self.mesh = mesh
        bary, wq = QUADRATURE[order]
        V = mesh.tri_vertices()
        self.V = V
        self.c = V.mean(axis=1)
        nrm = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
        self.area = 0.5 * np.linalg.norm(nrm, axis=1)
        self.normal = nrm / (2 * self.area[:, None])
        self.a = V - self.c[:, None, :]
        self.coef = mesh.tri_sign * mesh.edge_len[mesh.tri_edge] / (2 * self.area[:, None])
        self.X = np.einsum("qk,tkx->tqx", bary, V)
        self.w = self.area[:, None] * wq[None, :]
        self.rad = np.linalg.norm(self.a, axis=2).max(axis=1)
        self.near_factor = near_factor
        self._corrections()
        self.regions = [self._region(r) for r in range(mesh.n_regions)]

    # -- singular corrections ------------------------------------------------

    def near_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Ordered triangle pairs (p, q), p == q included, closer than near_factor (rad_p + rad_q)."""
        return close_pairs(self.c, self.c, self.near_factor * self.rad, self.near_factor * self.rad)

    def static_correction(self, X, q):
        """Analytic minus quadrature static inner integrals at points X (n, Qo, 3) over triangles q (n,)."""
        Vq = self.V[q][:, None]
        i0, irho, grad = static_integrals(X, Vq)
        nq = self.normal[q][:, None, :]
        rho = X - np.sum((X - Vq[..., 0, :]) * nq, axis=-1, keepdims=True) * nq
        cq = self.c[q][:, None, :]
        g0 = i0 / FOUR_PI
        g1 = (irho + (rho - cq) * i0[..., None]) / FOUR_PI
        gd = grad / FOUR_PI
        Y = self.X[q][:, None]  # (n, 1, Qi, 3)
        D = X[:, :, None, :] - Y
        R = np.linalg.norm(D, axis=-1)
        zero = R == 0
        inv = np.where(zero, 0.0, 1 / np.where(zero, 1.0, R))
        wy = self.w[q][:, None, :] / FOUR_PI
        g0 -= np.sum(wy * inv, axis=-1)
        g1 -= np.sum((wy * inv)[..., None] * (Y - cq[:, :, None, :]), axis=-2)
        gd -= np.sum((wy * -inv**3)[..., None] * D, axis=-2)
        return g0, g1, gd

    def _corrections(self, batch: int = 20000):
        p_all, q_all = self.near_pairs()
        N = self.mesh.n_edges
        acc_idx, acc_val = [], []
        for s in range(0, len(p_all), batch):
            p, q = p_all[s:s + batch], q_all[s:s + batch]
            g0, g1, gd = self.static_correction(self.X[p], q)
            u = self.X[p] - self.c[p][:, None, :]
            RR, M0, KK = pair_blocks(g0, g1, gd, u, self.w[p], self.c[p] - self.c[q], self.a[p], self.a[q], np)
            cc = self.coef[p][:, :, None] * self.coef[q][:, None, :]
            rows = np.broadcast_to(self.mesh.tri_edge[p][:, :, None], cc.shape)
            cols = np.broadcast_to(self.mesh.tri_edge[q][:, None, :], cc.shape)
            acc_idx.append((rows * N + cols).reshape(-1))
            acc_val.append(np.stack([cc * RR, cc * 4 * M0[:, None, None], cc * KK], axis=-1).reshape(-1, 3))
        lin = np.concatenate(acc_idx)
        val = np.concatenate(acc_val)
        uniq, inv = np.unique(lin, return_inverse=True)
        summed = np.stack([np.bincount(inv, weights=val[:, i], minlength=len(uniq)) for i in range(3)], axis=1)
        self.corr_rows, self.corr_cols = uniq // N, uniq % N
        self.corr_vals = summed  # columns: A (with ik), B (with -i/k), C (K operator)
        self.n_near_pairs = len(p_all)

    # -- regions ----------------------------------------------------------------

    def _region(self, r: int) -> Region:
        mesh = self.mesh
        surf_sign = dict(mesh.region_surfaces(r))
        in_tri = np.isin(mesh.tri_surface, list(surf_sign))
        tris = np.nonzero(in_tri)[0]
        edges = np.nonzero(np.isin(mesh.edge_surface, list(surf_sign)))[0]
        g2l = np.full(mesh.n_edges, -1)
        g2l[edges] = np.arange(len(edges))
        t2l = np.full(len(mesh.tris), -1)
        t2l[tris] = np.arange(len(tris))
        sign = np.array([surf_sign[s] for s in mesh.edge_surface[edges]])
        runs = []
        if len(edges):
            breaks = np.nonzero(np.diff(edges) != 1)[0] + 1
            for lo, hi in zip(np.r_[0, breaks], np.r_[breaks, len(edges)]):
                runs.append((int(edges[lo]), int(edges[hi - 1]) + 1, int(lo)))
        et, el = mesh.edge_tri[edges], mesh.edge_loc[edges]
        rows, cols = self.corr_rows, self.corr_cols
        m = (g2l[rows] >= 0) & (g2l[cols] >= 0)
        corr = (g2l[rows[m]], g2l[cols[m]], *self.corr_vals[m].T)
        return Region(r, tris, edges, sign, runs, g2l[mesh.tri_edge[tris]],
                      t2l[et[:, 0]] * 3 + el[:, 0], t2l[et[:, 1]] * 3 + el[:, 1], corr)


class DeviceGeometry:
    """Per-region arrays moved to one device/precision, reused across wavelengths."""

    def __init__(self, geo: Geometry, device, single: bool):
        self.geo, self.device = geo, device
        xp = self.xp = device.namespace()
        self.rdt = xp.float32 if single else xp.float64
        self.cdt = xp.complex64 if single else xp.complex128
        self.regions = []
        for reg in geo.regions:
            t = reg.tris
            mesh = geo.mesh
            lt = reg.tri_edge  # (T_r, 3) local edges
            sgn = mesh.tri_sign[t]
            h = np.arange(3 * len(t)).reshape(len(t), 3)
            d = dict(
                X=device.asarray(geo.X[t], dtype=self.rdt), w=device.asarray(geo.w[t], dtype=self.rdt),
                c=device.asarray(geo.c[t], dtype=self.rdt), a=device.asarray(geo.a[t], dtype=self.rdt),
                coef=device.asarray(geo.coef[t], dtype=self.rdt),
                col_plus=device.asarray(reg.col_plus, dtype=xp.int64),
                col_minus=device.asarray(reg.col_minus, dtype=xp.int64),
                lt=lt, sgn=sgn, h=h,
                corr_r=device.asarray(reg.corr[0], dtype=xp.int64), corr_c=device.asarray(reg.corr[1], dtype=xp.int64),
                corr=[device.asarray(v, dtype=self.rdt) for v in reg.corr[2:]],
            )
            self.regions.append(d)

    def region_operators(self, r: int, k: complex, budget: int):
        """Dense (L, K) over the edges of region r for wavenumber k.

        Both Galerkin matrices are symmetric, so only triangle pairs q >= p are integrated
        (the diagonal pairs with weight 1/2) and the result is added to its transpose.
        """
        xp, dev = self.xp, self.device
        d = self.regions[r]
        n_e = len(self.geo.regions[r].edges)
        X, w, c, a, coef = d["X"], d["w"], d["c"], d["a"], d["coef"]
        T, Q = X.shape[0], X.shape[1]
        L = xp.zeros((n_e, n_e), dtype=self.cdt, device=X.device) if dev.framework != "cupy"             else xp.zeros((n_e, n_e), dtype=self.cdt)
        K = xp.zeros_like(L)
        if T == 0:
            return L, K
        ik = 1j * complex(k)
        uy = X - c[:, None, :]  # (T, Qi, 3) source points about their centroid
        uy2 = xp.sum(uy * uy, axis=-1)
        uyT = xp.permute_dims(uy, (0, 2, 1))
        wc = xp.astype(w, self.cdt)[..., None]  # (T, Qi, 1)
        wuy = xp.astype(w[..., None] * uy, self.cdt)  # (T, Qi, 3)
        # coincident points (self pairs): R^2 from |u|^2 + |v|^2 - 2u.v carries round-off ~eps rad^2;
        # distinct quadrature points are >= ~0.2 rad apart
        tol2 = ((1e-2 if self.rdt == xp.float32 else 1e-5) * float(self.geo.rad.min())) ** 2
        chunk = max(1, budget // (T * Q * Q))
        for p0 in range(0, T, chunk):
            p1 = min(T, p0 + chunk)
            P = p1 - p0
            Xc = X[p0:p1]
            ut = Xc[:, None, :, :] - c[None, p0:, None, :]  # (P, Tq, Qo, 3) test points about source centroids
            R2 = xp.sum(ut * ut, axis=-1)[..., None] + uy2[None, p0:, None, :] - 2 * xp.matmul(ut, uyT[None, p0:])
            zero = R2 <= tol2
            R = xp.sqrt(xp.where(zero, xp.ones_like(R2), R2))
            e = xp.exp(ik * R) / (FOUR_PI * R)
            G = xp.where(zero, xp.full_like(e, ik / FOUR_PI), e)
            F = xp.where(zero, xp.zeros_like(e), (ik * R - 1) * e / (R * R))
            G0 = xp.matmul(G, wc[None, p0:])[..., 0]  # (P, Tq, Qo)
            G1 = xp.matmul(G, wuy[None, p0:])  # (P, Tq, Qo, 3)
            Gd = ut * xp.matmul(F, wc[None, p0:]) - xp.matmul(F, wuy[None, p0:])
            u = (Xc - c[p0:p1, None, :])[:, None, :, :]
            delta = c[p0:p1, None, :] - c[None, p0:, :]
            RR, M0, KK = pair_blocks(G0, G1, Gd, u, w[p0:p1, None, :], delta, a[p0:p1, None], a[None, p0:], xp)
            pl = np.arange(P)[:, None]
            ql = np.arange(T - p0)[None, :]
            half = dev.asarray(np.where(ql > pl, 1.0, np.where(ql == pl, 0.5, 0.0)), dtype=self.rdt)
            cc = coef[p0:p1, None, :, None] * coef[None, p0:, None, :] * half[..., None, None]
            pad = xp.zeros((P, p0, 3, 3), dtype=self.cdt, device=X.device) if dev.framework != "cupy"                 else xp.zeros((P, p0, 3, 3), dtype=self.cdt)
            Lh = xp.concat([pad, cc * (ik * RR - (4 * ik / complex(k) ** 2) * M0[..., None, None])], axis=1)
            Kh = xp.concat([pad, xp.astype(cc * KK, self.cdt)], axis=1)
            for H, Z in ((Lh, L), (Kh, K)):
                H = xp.reshape(xp.permute_dims(H, (0, 2, 1, 3)), (3 * P, 3 * T))
                Hc = xp.take(H, d["col_plus"], axis=1) + xp.take(H, d["col_minus"], axis=1)
                for s in (1.0, -1.0):
                    sel = d["sgn"][p0:p1] == s
                    rows = d["lt"][p0:p1][sel]
                    hidx = (d["h"][p0:p1] - 3 * p0)[sel]
                    if len(rows):
                        ri = dev.asarray(rows, dtype=xp.int64)
                        Z[ri, :] = Z[ri, :] + xp.take(Hc, dev.asarray(hidx, dtype=xp.int64), axis=0)
        L = L + xp.permute_dims(L, (1, 0))
        K = K + xp.permute_dims(K, (1, 0))
        ri, ci = d["corr_r"], d["corr_c"]
        A, B, C = d["corr"]
        L[ri, ci] = L[ri, ci] + (ik * A - (ik / complex(k) ** 2) * B)
        K[ri, ci] = K[ri, ci] + C
        return L, K


def region_wavenumbers(mesh: Mesh, lam: float, n0: float) -> list[complex]:
    out = [complex(n0)]
    for mat in mesh.region_materials[1:]:
        n = complex(np.sqrt(complex(mat.eps(np.array([lam]))[0])))
        out.append(-n if n.imag < 0 else n)
    return [2 * math.pi * n / lam for n in out]


def assemble(dg: DeviceGeometry, lam: float, n0: float, budget: int):
    """The 2N x 2N PMCHWT matrix, unknowns ordered [J' (all edges), M (all edges)]."""
    xp, geo = dg.xp, dg.geo
    N = geo.mesh.n_edges
    k0 = 2 * math.pi * n0 / lam
    Z = None
    for r, k in enumerate(region_wavenumbers(geo.mesh, lam, n0)):
        reg = geo.regions[r]
        if not len(reg.edges):
            continue
        L, K = dg.region_operators(r, k, budget)
        s = dg.device.asarray(reg.sign, dtype=dg.rdt)
        S = s[:, None] * s[None, :]
        ratio = complex(k0 / k)  # eta_r / eta_0 = n_0 / n_r
        L, K = L * S, K * S
        if Z is None:
            Z = xp.zeros((2 * N, 2 * N), dtype=dg.cdt, device=L.device) if dg.device.framework != "cupy" \
                else xp.zeros((2 * N, 2 * N), dtype=dg.cdt)
        for g0, g1, l0 in reg.runs:
            for h0, h1, m0 in reg.runs:
                lr, lc = slice(l0, l0 + g1 - g0), slice(m0, m0 + h1 - h0)
                Lb, Kb = L[lr, lc], K[lr, lc]
                Z[g0:g1, h0:h1] += ratio * Lb
                Z[g0:g1, N + h0:N + h1] -= Kb
                Z[N + g0:N + g1, h0:h1] += Kb
                Z[N + g0:N + g1, N + h0:N + h1] += Lb / ratio
        del L, K
    return Z


def incident_rhs(geo: Geometry, lam: float, n0: float, incidences) -> np.ndarray:
    """-[<f, E_inc>; <f, eta_0 H_inc>] on surfaces facing the host, for unit plane waves; shape (2N, n_inc)."""
    k = 2 * math.pi * n0 / lam
    mesh = geo.mesh
    N = mesh.n_edges
    out = np.zeros((2 * N, len(incidences)), dtype=complex)
    u = geo.X - geo.c[:, None, :]
    for col, (khat, ehat) in enumerate(incidences):
        hhat = np.cross(khat, ehat)
        ph = geo.w * np.exp(1j * k * geo.X @ khat)  # (T, Q)
        for block, pol in ((0, ehat), (1, hhat)):
            # <f_(t,i), pol e^{ik.r}> = coef [sum w ph (u.pol) - (a_i.pol) sum w ph]
            val = geo.coef * (np.sum(ph * (u @ pol), axis=1)[:, None] - (geo.a @ pol) * np.sum(ph, axis=1)[:, None])
            out[block * N:(block + 1) * N, col] = -np.bincount(mesh.tri_edge.reshape(-1), weights=val.real.reshape(-1),
                                                               minlength=N) \
                - 1j * np.bincount(mesh.tri_edge.reshape(-1), weights=val.imag.reshape(-1), minlength=N)
    # the incident field lives in the host only: surfaces not facing it are not driven
    inner = ~np.isin(mesh.edge_surface, [s for s, surf in enumerate(mesh.surfaces) if surf.outer == 0])
    out[:N][inner] = 0
    out[N:][inner] = 0
    return out


def gmres(matvec, b, xp, tol: float, restart: int, max_iter: int, precond=None):
    """Restarted GMRES with right preconditioning; returns (x, iterations, relative residual)."""
    M = precond or (lambda v: v)
    norm = lambda v: float(xp.sqrt(xp.sum(xp.real(v * xp.conj(v)))))  # noqa: E731
    bnorm = norm(b) or 1.0
    x = xp.zeros_like(b)
    r = b
    it = 0
    res = norm(r) / bnorm
    while it < max_iter and res > tol:
        beta = norm(r)
        Vs = [r / beta]
        H = np.zeros((restart + 1, restart), dtype=complex)
        g = np.zeros(restart + 1, dtype=complex)
        g[0] = beta
        cs, sn = np.zeros(restart, dtype=complex), np.zeros(restart, dtype=complex)
        j = 0
        for j in range(restart):
            wv = matvec(M(Vs[j]))
            for i in range(j + 1):
                hij = complex(xp.sum(xp.conj(Vs[i]) * wv))
                H[i, j] = hij
                wv = wv - hij * Vs[i]
            H[j + 1, j] = norm(wv)
            if H[j + 1, j] != 0:
                Vs.append(wv / H[j + 1, j])
            for i in range(j):  # previous Givens rotations [[c*, s*], [-s, c]]
                t = np.conj(cs[i]) * H[i, j] + np.conj(sn[i]) * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            den = np.hypot(abs(H[j, j]), abs(H[j + 1, j]))
            cs[j], sn[j] = (H[j, j] / den, H[j + 1, j] / den) if den else (1.0, 0.0)
            H[j, j] = den
            H[j + 1, j] = 0
            g[j + 1] = -sn[j] * g[j]
            g[j] = np.conj(cs[j]) * g[j]
            it += 1
            res = abs(g[j + 1]) / bnorm
            if res <= tol or it >= max_iter or len(Vs) <= j + 1:
                break
        m = j + 1
        y = np.linalg.solve(np.triu(H[:m, :m]), g[:m])
        upd = Vs[0] * complex(y[0])
        for i in range(1, m):
            upd = upd + Vs[i] * complex(y[i])
        x = x + M(upd)
        r = b - matvec(x)
        res = norm(r) / bnorm
    return x, it, res
