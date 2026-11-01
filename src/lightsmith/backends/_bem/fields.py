"""Cross sections and fields from the solved equivalent currents.

With J' = eta_0 J and M on the RWG functions (unit incident amplitude, host
wavenumber k):

    C_ext = -Re(b^H x)                       optical theorem: Re[e.J'^(k) + h.M^(k)]
    C_abs = Re sum J'^* . (n x M) dS          net Poynting flux into the particles
    F(r) = ik/(4 pi) [J'^_perp(r) - r x M^(r)],   X^(r) = int X e^{-ik r.r'} dS'
    dC_sca/dOmega = |F|^2,   C_sca = int |F|^2 dOmega

C_ext is the Galerkin (variational) functional of the solution, so its error is
quadratic in the current error; C_sca and C_abs are evaluated independently and
the energy balance C_ext = C_sca + C_abs is reported as a self-check.
"""

from __future__ import annotations

import math

import numpy as np

from ...device import Device
from .operators import FOUR_PI, Geometry, cross, region_wavenumbers


def triangle_moments(geo: Geometry, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per triangle, the current X(r) = alpha (r - c) - beta: alpha (T, ...), beta (T, ..., 3).

    `x` holds edge coefficients with shape (N, ...).
    """
    xc = x[geo.mesh.tri_edge] * geo.coef.reshape(geo.coef.shape + (1,) * (x.ndim - 1))  # (T, 3, ...)
    alpha = xc.sum(axis=1)
    beta = np.einsum("ti...,tix->t...x", xc, geo.a)
    return alpha, beta


def outer_triangles(geo: Geometry) -> np.ndarray:
    outer = [s for s, surf in enumerate(geo.mesh.surfaces) if surf.outer == 0]
    return np.nonzero(np.isin(geo.mesh.tri_surface, outer))[0]


def absorption(geo: Geometry, x: np.ndarray) -> np.ndarray:
    """C_abs per right-hand side from the flux of the surface fields; x is (2N, n_rhs)."""
    N = geo.mesh.n_edges
    t = outer_triangles(geo)
    cj = x[:N][geo.mesh.tri_edge[t]] * geo.coef[t][:, :, None]  # (T, 3, n)
    cm = x[N:][geo.mesh.tri_edge[t]] * geo.coef[t][:, :, None]
    a = geo.a[t]
    # int_T f_i . (n x f_j) = c_i c_j A n.(a_j x a_i)
    g = geo.area[t][:, None, None] * np.einsum("tx,tijx->tij", geo.normal[t], np.cross(a[:, None, :, :], a[:, :, None, :]))
    return np.real(np.einsum("tin,tij,tjn->n", np.conj(cj), g, cm))


def far_field(geo: Geometry, x: np.ndarray, k: float, dirs: np.ndarray, device: Device, single: bool,
              chunk: int = 512) -> np.ndarray:
    """Scattering amplitude F (n_dir, 3, n_rhs) in the host medium."""
    xp = device.namespace()
    rdt, cdt = (xp.float32, xp.complex64) if single else (xp.float64, xp.complex128)
    N = geo.mesh.n_edges
    t = outer_triangles(geo)
    centre = geo.c[t].mean(axis=0)  # phase reference; |F| does not depend on it
    out = []
    Xq = geo.X[t] - centre
    u = geo.X[t] - geo.c[t][:, None, :]
    wq = geo.w[t]
    srcs = []
    for blk in (x[:N], x[N:]):
        al, be = triangle_moments(geo, blk)
        cur = al[t][:, None, None, :] * u[..., None] - np.moveaxis(be[t], -1, 1)[:, None]  # (T, Q, 3, n)
        srcs.append(device.asarray((wq[..., None, None] * cur).reshape(-1, 3 * x.shape[1]), dtype=cdt))
    Xd = device.asarray(Xq.reshape(-1, 3), dtype=rdt)
    for s in range(0, len(dirs), chunk):
        dd = dirs[s:s + chunk]
        ph = xp.exp(-1j * k * xp.matmul(device.asarray(dd, dtype=rdt), xp.permute_dims(Xd, (1, 0))))
        ph = xp.astype(ph, cdt) if ph.dtype != cdt else ph
        jh, mh = (Device.to_numpy(xp.matmul(ph, sv)).reshape(len(dd), 3, -1) for sv in srcs)
        dv = dd[:, :, None]
        jperp = jh - dv * np.sum(dv * jh, axis=1, keepdims=True)
        rxm = np.cross(dd[:, None, :], np.moveaxis(mh, 1, 2), axis=-1)  # (d, n, 3)
        out.append(1j * k / FOUR_PI * (jperp - np.moveaxis(rxm, 2, 1)))
    return np.concatenate(out)


def scattering(geo: Geometry, x: np.ndarray, k: float, device: Device, single: bool) -> np.ndarray:
    """C_sca per right-hand side by Gauss-Legendre x trapezoid quadrature of |F|^2."""
    t = outer_triangles(geo)
    c = geo.c[t].mean(axis=0)
    rmax = float(np.linalg.norm(geo.mesh.verts - c, axis=1).max())
    n_t = int(max(16, math.ceil(1.2 * k * rmax) + 12))
    n_p = 2 * n_t
    mu, wmu = np.polynomial.legendre.leggauss(n_t)
    phi = 2 * np.pi * np.arange(n_p) / n_p
    st = np.sqrt(1 - mu**2)
    dirs = np.stack([np.repeat(st, n_p) * np.tile(np.cos(phi), n_t), np.repeat(st, n_p) * np.tile(np.sin(phi), n_t),
                     np.repeat(mu, n_p)], axis=1)
    F = far_field(geo, x, k, dirs, device, single)
    w = np.repeat(wmu, n_p) * (2 * np.pi / n_p)
    return np.einsum("d,dn->n", w, np.sum(np.abs(F) ** 2, axis=1))


def near_field(geo: Geometry, sim, x: np.ndarray, lam: float, pts: np.ndarray, khat, ehat, device: Device,
               single: bool, budget: int) -> np.ndarray:
    """Total E at points (M, 3) for one solution x (2N,); NaN on points the mesh cannot place."""
    mesh = geo.mesh
    n0 = sim.medium_index
    ks = region_wavenumbers(mesh, lam, n0)
    region = np.zeros(len(pts), dtype=int)
    first = 1
    for p in sim.particles:
        r = p.region(pts - np.asarray(p.position_nm))
        region[r >= 0] = first + r[r >= 0]
        first += len(p.region_materials())
    N = mesh.n_edges
    aj, bj = triangle_moments(geo, x[:N])
    am, bm = triangle_moments(geo, x[N:])
    E = np.zeros((len(pts), 3), dtype=complex)
    xp = device.namespace()
    rdt, cdt = (xp.float32, xp.complex64) if single else (xp.float64, xp.complex128)
    for r, k in enumerate(ks):
        sel = np.nonzero(region == r)[0]
        if not len(sel):
            continue
        signs = dict(mesh.region_surfaces(r))
        t = np.nonzero(np.isin(mesh.tri_surface, list(signs)))[0]
        s_t = np.array([signs[s] for s in mesh.tri_surface[t]])
        ratio = complex(ks[0] / k)
        ik = 1j * complex(k)
        coefs = dict(aj=aj[t] * s_t * ratio, bj=bj[t] * (s_t * ratio)[:, None], am=am[t] * s_t, bm=bm[t] * s_t[:, None])

        def combine(G0, G1, Gd, rc, co, mod):
            # E = sum_t s [ratio (ik(aJ G1 - bJ G0) + 2 (ik/k^2) aJ Gd) - Gd x (aM (r - c_t) - bM)]
            out = (ik * (co["aj"][..., None] * G1 - co["bj"] * G0[..., None])
                   + (2 * ik / complex(k) ** 2) * co["aj"][..., None] * Gd)
            out = out - cross(Gd, co["am"][..., None] * rc - co["bm"], mod)
            return mod.sum(out, axis=1)

        X = device.asarray(geo.X[t], dtype=rdt)
        uy = X - device.asarray(geo.c[t], dtype=rdt)[:, None, :]
        w = device.asarray(geo.w[t], dtype=rdt)
        cdev = device.asarray(geo.c[t], dtype=rdt)
        co_dev = {kk: device.asarray(v, dtype=cdt) for kk, v in coefs.items()}
        chunk = max(1, budget // (len(t) * X.shape[1]))
        for s0 in range(0, len(sel), chunk):
            idx = sel[s0:s0 + chunk]
            P = device.asarray(pts[idx], dtype=rdt)
            D = P[:, None, None, :] - X[None]
            R = xp.sqrt(xp.sum(D * D, axis=-1))
            e = xp.exp(ik * R) / (FOUR_PI * R)
            F = (ik * R - 1) * e / (R * R)
            G0 = xp.sum(w * e, axis=-1)
            G1 = xp.sum((w * e)[..., None] * uy[None], axis=-2)
            Gd = xp.sum((w * F)[..., None] * D, axis=-2)
            E[idx] += Device.to_numpy(combine(G0, G1, Gd, P[:, None, :] - cdev[None], co_dev, xp))
        # static singular corrections for points close to a triangle (k-independent part of G)
        from scipy.spatial import cKDTree

        tree = cKDTree(geo.c[t])
        near = tree.query_ball_point(pts[sel], geo.near_factor * 2 * geo.rad[t].max())
        pi = np.concatenate([np.full(len(n), i) for i, n in enumerate(near)]).astype(int) if len(near) else np.zeros(0, int)
        ti = np.concatenate([np.asarray(n, dtype=int) for n in near]) if len(near) else np.zeros(0, int)
        if len(pi):
            g0, g1, gd = geo.static_correction(pts[sel][pi][:, None, :], t[ti])
            co = {kk: v[ti][:, None] for kk, v in coefs.items()}
            rc = (pts[sel][pi] - geo.c[t][ti])[:, None, :]
            np.add.at(E, sel[pi], combine(g0, g1, gd, rc, co, np))
        if r == 0:
            E[sel] += ehat[None, :] * np.exp(1j * k * pts[sel] @ khat)[:, None]
    return E
