"""Eigen-solvers for the plane-wave expansion (array API: NumPy, PyTorch or CuPy).

Units: lengths in a, wavevectors in 1/a (with the 2 pi), eigenvalues
lambda = (omega a / c)^2, normalised frequency a / lambda_0 = sqrt(lambda) / (2 pi).

Dense path (Ho, Chan & Soukoulis, PRL 65, 3152 (1990) inverse rule):
    E_GG' = eps_hat(G - G'),  eta = E^-1 (one inversion, k-independent)
    2D TE (H_z):  M = (k+G).(k+G') eta_GG'
    2D TM (E_z):  M = |k+G| eta_GG' |k+G'|  (exactly the generalised problem |k+G|^2 e = lambda E e)
    3D (H, transverse basis e_1, e_2 per G):  M = eta_GG' [(k+G) x e_a] . [(k+G') x e_b]
All k-points of a chunk are diagonalised in one batched ``eigh``.

Iterative path (matrix-free, MPB-style; Johnson & Joannopoulos, Opt. Express 8, 173 (2001)):
FFT operators on a real-space grid with the subpixel-smoothed inverse-permittivity
tensor (Farjadpour et al., Opt. Lett. 31, 2972 (2006)) and a block LOBPCG
eigensolver (Knyazev, SIAM J. Sci. Comput. 23, 517 (2001)), warm-started along
the k-path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# --- plane-wave basis -------------------------------------------------------------


def plane_wave_set(reciprocal: np.ndarray, n_target: int) -> np.ndarray:
    """Integer indices (N, d) of the ~n_target shortest G, completed to a full shell (keeps symmetry)."""
    d = reciprocal.shape[0]
    meas = abs(np.linalg.det(reciprocal))
    if d == 2:
        g_cut = math.sqrt(n_target * meas / math.pi)
    else:
        g_cut = (3 * n_target * meas / (4 * math.pi)) ** (1 / 3)
    a = 2 * np.pi * np.linalg.inv(reciprocal).T  # direct vectors
    m = [math.ceil(1.5 * g_cut * np.linalg.norm(a[i]) / (2 * np.pi)) + 1 for i in range(d)]
    idx = np.stack(np.meshgrid(*[np.arange(-k, k + 1) for k in m], indexing="ij"), axis=-1).reshape(-1, d)
    g = np.linalg.norm(idx @ reciprocal, axis=1)
    order = np.argsort(g, kind="stable")
    n = min(n_target, len(order))
    edge = g[order[n - 1]] * (1 + 1e-9) + 1e-12
    return idx[g <= edge]


def grid_indices(grid: tuple[int, ...]) -> np.ndarray:
    """Integer G indices of every FFT grid point, (prod(grid), d), in FFT order."""
    axes = [np.rint(np.fft.fftfreq(n) * n).astype(int) for n in grid]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(grid))


def transverse_basis(q: np.ndarray) -> np.ndarray:
    """u_a = q x e_a for an orthonormal pair (e_1, e_2) perpendicular to q. q: (..., 3) -> (..., 2, 3)."""
    qn = np.linalg.norm(q, axis=-1, keepdims=True)
    qh = np.where(qn > 0, q / np.where(qn > 0, qn, 1.0), np.array([0.0, 0.0, 1.0]))  # any frame at q = 0
    ref = np.where(np.abs(qh[..., 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    e1 = np.cross(qh, ref)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(qh, e1)
    # q x e1 = |q| e2, q x e2 = -|q| e1 (zero at q = 0)
    return np.stack([qn * e2, -qn * e1], axis=-2)


def transverse_frame(q: np.ndarray) -> np.ndarray:
    """(e_1, e_2) for transverse_basis, shape (..., 2, 3); used to carry vectors between k-points."""
    qn = np.linalg.norm(q, axis=-1, keepdims=True)
    qh = np.where(qn > 0, q / np.where(qn > 0, qn, 1.0), np.array([0.0, 0.0, 1.0]))  # any frame at q = 0
    ref = np.where(np.abs(qh[..., 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    e1 = np.cross(qh, ref)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    return np.stack([e1, np.cross(qh, e1)], axis=-2)


# --- dense HCS solver -----------------------------------------------------------------


def toeplitz_eps(table, half: tuple[int, ...], idx: np.ndarray, xp):
    """E_GG' = eps_hat(G - G') gathered from an (xp) table; differentiable through `table`."""
    diff = idx[:, None, :] - idx[None, :, :] + np.asarray(half)
    shape = [2 * h + 1 for h in half]
    flat = np.ravel_multi_index(tuple(np.moveaxis(diff, -1, 0)), shape)
    tab = xp.reshape(table, (-1,))
    dev = getattr(tab, "device", None)
    ind = xp.asarray(flat.reshape(-1), device=dev) if dev is not None else xp.asarray(flat.reshape(-1))
    return xp.reshape(xp.take(tab, ind), (len(idx), len(idx)))


def dense_bands(eta, idx: np.ndarray, reciprocal: np.ndarray, k_cart: np.ndarray, pol: str, num_bands: int,
                xp, device, cdtype, chunk_bytes: int = 256 * 2**20):
    """Lowest `num_bands` eigenvalues for each k (rows of k_cart); returns an xp array (nk, num_bands).

    `eta` is either one (N, N) matrix (isotropic rule) or a (d, d, N, N) tensor of
    matrices; for TE the tensor must already be the rotated (grad H_z) form.
    """
    g = idx @ reciprocal
    n = len(idx)
    d = g.shape[1]
    tensor = eta.ndim == 4
    size = 2 * n if pol == "3d" else n
    itemsize = 8 if cdtype == xp.complex64 else 16
    chunk = max(1, int(chunk_bytes // (4 * size * size * itemsize)))
    rdt = xp.float32 if cdtype == xp.complex64 else xp.float64
    out = []
    for s in range(0, len(k_cart), chunk):
        q = k_cart[s:s + chunk, None, :] + g[None]  # (c, N, d)
        if pol == "te" and tensor:
            m = None
            for i in range(d):
                for j in range(d):
                    t = device.asarray(q[:, :, None, i] * q[:, None, :, j], dtype=rdt) * eta[i, j][None]
                    m = t if m is None else m + t
        elif pol == "te":
            w = np.einsum("kgi,khi->kgh", q, q)
            m = device.asarray(w, dtype=rdt) * eta[None]
        elif pol == "tm":
            qn = np.linalg.norm(q, axis=-1)
            m = device.asarray(qn[:, :, None] * qn[:, None, :], dtype=rdt) * eta[None]
        else:
            u = transverse_basis(q)  # (c, N, 2, 3)
            if tensor:
                m = None
                for i in range(3):
                    for j in range(3):
                        w = u[:, :, :, None, None, i] * u[:, None, None, :, :, j]
                        t = device.asarray(w, dtype=rdt) * eta[i, j][None, :, None, :, None]
                        m = t if m is None else m + t
            else:
                w = np.einsum("kgac,khbc->kgahb", u, u)
                m = device.asarray(w, dtype=rdt) * eta[None, :, None, :, None]
            m = xp.reshape(m, (q.shape[0], size, size))
        m = (m + xp.conj(xp.permute_dims(m, (0, 2, 1)))) / 2
        lam = xp.linalg.eigvalsh(m)[:, :num_bands]
        out.append(lam)
    return xp.concat(out, axis=0)


def eigen_to_freq(lam, xp):
    """a / lambda_0 from eigenvalues (omega a / c)^2; tiny negatives (round-off) clamp to 0."""
    return xp.sqrt(xp.clip(xp.real(lam), 0.0, None)) / (2 * math.pi)


# --- LOBPCG ---------------------------------------------------------------------------


def Device_to_numpy(a) -> np.ndarray:  # noqa: N802 - tiny local helper, mirrors Device.to_numpy
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    if hasattr(a, "get"):
        return a.get()
    return np.asarray(a)


def _to_dev(a: np.ndarray, like, xp):
    """Small host matrix -> the device and dtype of `like`."""
    dev = getattr(like, "device", None)
    if dev is not None and not isinstance(like, np.ndarray):
        return xp.asarray(a, dtype=like.dtype, device=dev)
    return xp.asarray(a, dtype=like.dtype)


def _host_herm(m) -> np.ndarray:
    """Hermitian part of a small device matrix, on the host in complex128."""
    h = Device_to_numpy(m).astype(np.complex128)
    return (h + h.conj().T) / 2


def _orth(y, xp, rtol):
    """Orthonormal basis of the columns of y via the Gram matrix (SVQB), dropping tiny directions.

    The small eigenproblem runs on the host in double precision. Returns (q, t) with q = y @ t.
    """
    ev, v = np.linalg.eigh(_host_herm(xp.conj(y.T) @ y))
    keep = ev > rtol * max(ev.max(), 1e-300)
    t = v[:, keep] / np.sqrt(ev[keep])[None, :]
    td = _to_dev(t, y, xp)
    return y @ td, td


@dataclass
class LobpcgInfo:
    iterations: int
    max_residual: float
    converged: bool


def lobpcg(apply_a: Callable, x0, precond: Callable, n_want: int, xp, tol: float, max_iter: int,
           refresh: int = 10):
    """Smallest eigenpairs of a Hermitian operator. x0: (n, m) initial block, m >= n_want.

    Soft locking: only unconverged columns get new search directions. A X, A W, A P
    are tracked by linear combination; A X is recomputed every `refresh` iterations.
    All small (block-sized) eigenproblems run on the host in float64, which keeps
    GPU runs free of tiny-kernel latency and robust in single precision.
    """
    rtol = 1e-12 if x0.dtype == xp.complex128 else 1e-6
    x, _ = _orth(x0, xp, rtol)
    ax = apply_a(x)
    p = ap = None
    th = np.zeros(x.shape[1])
    res_max = float("inf")
    for it in range(1, max_iter + 1):
        th, c = np.linalg.eigh(_host_herm(xp.conj(x.T) @ ax))
        cd = _to_dev(c, x, xp)
        x, ax = x @ cd, ax @ cd
        r = ax - x * _to_dev(th, x, xp)[None, :]
        rn = Device_to_numpy(xp.sqrt(xp.sum(xp.real(r * xp.conj(r)), axis=0))).astype(float)
        scale = np.maximum(np.abs(th), max(abs(th[min(n_want, len(th)) - 1]), 1.0) * 1e-2)
        rel = rn / scale
        res_max = float(rel[:n_want].max())
        if res_max < tol:
            return th, x, LobpcgInfo(it, res_max, True)
        active = np.nonzero(rel >= tol)[0]
        dev = getattr(x, "device", None)
        ind = xp.asarray(active, device=dev) if dev is not None else xp.asarray(active)
        w = precond(xp.take(r, ind, axis=1))
        aw = apply_a(w)
        # Search space [X, W, P]: orthogonalise W and P against X (twice); A y follows the same
        # linear combination because A X, A W and A P are known.
        y = w if p is None else xp.concat([w, p], axis=1)
        ay = aw if p is None else xp.concat([aw, ap], axis=1)
        for _ in range(2):
            coef = xp.conj(x.T) @ y
            y, ay = y - x @ coef, ay - ax @ coef
        qy, t = _orth(y, xp, rtol)
        qy, t2 = _orth(qy, xp, rtol)  # second pass for numerical orthogonality
        aqy = ay @ (t @ t2)
        s = xp.concat([x, qy], axis=1)
        as_ = xp.concat([ax, aqy], axis=1)
        # Rayleigh-Ritz on span(S); S is orthonormal up to round-off, whitened by its Gram matrix.
        gev, gv = np.linalg.eigh(_host_herm(xp.conj(s.T) @ s))
        wh = gv / np.sqrt(np.clip(gev, 1e-300, None))[None, :]
        mu, cc = np.linalg.eigh(wh.conj().T @ _host_herm(xp.conj(s.T) @ as_) @ wh)
        m = x.shape[1]
        cc = _to_dev(wh @ cc[:, :m], x, xp)
        x, ax = s @ cc, as_ @ cc
        p, ap = qy @ cc[m:, :], aqy @ cc[m:, :]
        if it % refresh == 0:
            x, _ = _orth(x, xp, rtol)
            ax = apply_a(x)
            p = ap = None
    return th, x, LobpcgInfo(max_iter, res_max, False)


# --- matrix-free FFT operators -----------------------------------------------------


class FFTOperator:
    """Maxwell operator on the full FFT-grid plane-wave basis with a smoothed eps.

    Vectors are (n_basis, m) blocks of plane-wave coefficients; n_basis = prod(grid)
    for 2D and 2 prod(grid) (transverse components) for 3D.
    """

    def __init__(self, pol: str, grid: tuple[int, ...], reciprocal: np.ndarray, eta: np.ndarray,
                 eps_mean: np.ndarray, device, cdtype):
        self.pol, self.grid, self.device, self.cdtype = pol, tuple(grid), device, cdtype
        self.xp = device.namespace()
        self.d = len(grid)
        self.axes = tuple(range(1, self.d + 1))
        self.g = grid_indices(self.grid) @ reciprocal  # (n, d)
        self.npts = int(np.prod(grid))
        rdt = self.xp.float32 if cdtype == self.xp.complex64 else self.xp.float64
        self.rdt = rdt
        if pol == "tm":
            self.s = device.asarray(eps_mean ** -0.5, dtype=rdt)
            self.s_inv = device.asarray(eps_mean ** 0.5, dtype=rdt)
            self.eta_mean = float(np.mean(1 / eps_mean))
        else:
            if pol == "te":
                # H_z form -div(eta' grad H_z): grad H_z is curl H rotated by 90 degrees, so the tensor
                # acting on E (normal <1/eps>, tangential 1/<eps>) is rotated too: eta' = R^T eta R.
                eta = np.stack([np.stack([eta[1, 1], -eta[0, 1]]), np.stack([-eta[1, 0], eta[0, 0]])])
            self.eta = device.asarray(eta, dtype=rdt)
            self.eta_mean = float(np.mean(np.trace(eta.reshape(self.d, self.d, -1))) / self.d)
            # Pointwise eps tensor for the approximate-inverse preconditioner.
            eps_t = np.linalg.inv(np.moveaxis(eta, (0, 1), (-2, -1)))
            self.eps_t = device.asarray(np.moveaxis(eps_t, (-2, -1), (0, 1)), dtype=rdt)
        self.k = None

    @property
    def size(self) -> int:
        return self.npts * (2 if self.pol == "3d" else 1)

    def set_k(self, k: np.ndarray):
        self.k = k
        q = k[None, :] + self.g
        dev = self.device
        self.q2 = np.sum(q**2, axis=1)
        if self.pol == "te":
            self.qc = [dev.asarray(q[:, i].reshape(self.grid), dtype=self.rdt) for i in range(2)]
        elif self.pol == "tm":
            self.q2d = dev.asarray(self.q2.reshape(self.grid), dtype=self.rdt)
        else:
            u = transverse_basis(q)  # (n, 2, 3)
            self.u = dev.asarray(np.moveaxis(u, 0, -1).reshape((2, 3) + self.grid), dtype=self.rdt)
            self.frame = transverse_frame(q)
        # 1 / |k+G|^2, regularised at k+G = 0 (the omega = 0 modes at Gamma).
        inv_q2 = 1.0 / (self.q2 + (2 * np.pi) ** 2 * 1e-2)
        self.inv_q2 = dev.asarray(inv_q2.reshape(self.grid), dtype=self.rdt)

    def _grid(self, h, comps: int = 1):
        m = h.shape[1]
        return self.xp.reshape(self.xp.permute_dims(h, (1, 0)), (m,) + ((comps,) if comps > 1 else ()) + self.grid)

    def _flat(self, v):
        m = v.shape[0]
        return self.xp.permute_dims(self.xp.reshape(v, (m, -1)), (1, 0))

    def apply(self, h):
        xp, fft = self.xp, self.xp.fft
        if self.pol == "tm":
            v = fft.ifftn(self._grid(h), axes=self.axes) * self.s
            v = fft.fftn(v, axes=self.axes) * self.q2d
            v = fft.fftn(fft.ifftn(v, axes=self.axes) * self.s, axes=self.axes)
            return self._flat(v)
        if self.pol == "te":
            hg = self._grid(h)
            f = [fft.ifftn(hg * self.qc[i], axes=self.axes) for i in range(2)]
            out = None
            for i in range(2):
                vi = self.eta[i, 0] * f[0] + self.eta[i, 1] * f[1]
                term = fft.fftn(vi, axes=self.axes) * self.qc[i]
                out = term if out is None else out + term
            return self._flat(out)
        hg = self._grid(h, 2)  # (m, 2, *grid)
        ax = tuple(a + 1 for a in self.axes)
        w = hg[:, 0:1] * self.u[0][None] + hg[:, 1:2] * self.u[1][None]  # (m, 3, *grid)
        f = fft.ifftn(w, axes=ax)
        v = xp.stack([sum(self.eta[c, j] * f[:, j] for j in range(3)) for c in range(3)], axis=1)
        v = fft.fftn(v, axes=ax)
        out = xp.stack([xp.sum(v * self.u[a][None], axis=1) for a in range(2)], axis=1)  # (m, 2, *grid)
        return self._flat(out)

    def precondition(self, r):
        """Approximate inverse (MPB-style): invert each factor of A = C^H eta C with C^+ = C^H / |k+G|^2."""
        xp, fft = self.xp, self.xp.fft
        if self.pol == "tm":  # exact inverse of s Q^2 s, up to the regularisation at k+G = 0
            v = fft.fftn(fft.ifftn(self._grid(r), axes=self.axes) * self.s_inv, axes=self.axes) * self.inv_q2
            return self._flat(fft.fftn(fft.ifftn(v, axes=self.axes) * self.s_inv, axes=self.axes))
        if self.pol == "te":
            rg = self._grid(r) * self.inv_q2
            f = [fft.ifftn(rg * self.qc[i], axes=self.axes) for i in range(2)]
            out = None
            for i in range(2):
                vi = self.eps_t[i, 0] * f[0] + self.eps_t[i, 1] * f[1]
                term = fft.fftn(vi, axes=self.axes) * self.qc[i]
                out = term if out is None else out + term
            return self._flat(out * self.inv_q2)
        rg = self._grid(r, 2) * self.inv_q2[None, None]
        ax = tuple(a + 1 for a in self.axes)
        w = rg[:, 0:1] * self.u[0][None] + rg[:, 1:2] * self.u[1][None]
        f = fft.ifftn(w, axes=ax)
        v = xp.stack([sum(self.eps_t[c, j] * f[:, j] for j in range(3)) for c in range(3)], axis=1)
        v = fft.fftn(v, axes=ax)
        out = xp.stack([xp.sum(v * self.u[a][None], axis=1) for a in range(2)], axis=1) * self.inv_q2[None, None]
        return self._flat(out)

    def initial(self, m: int, rng: np.random.Generator):
        """Plane waves of lowest |k+G| plus a little noise."""
        n = self.size
        order = np.argsort(np.concatenate([self.q2, self.q2]) if self.pol == "3d" else self.q2, kind="stable")
        x = np.zeros((n, m), dtype=complex)
        x[order[:m], np.arange(m)] = 1.0
        x += 1e-3 * (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m)))
        return self.device.asarray(x, dtype=self.cdtype)

    def noise(self, m: int, rng: np.random.Generator, scale: float):
        n = self.size
        x = (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))) * (scale / math.sqrt(2 * n))
        return self.device.asarray(x, dtype=self.cdtype)

    def carry(self, x_prev, frame_prev: np.ndarray | None):
        """Re-express a 3D transverse block from the previous k's frame in the current one."""
        if self.pol != "3d" or frame_prev is None:
            return x_prev
        # T[a, b, G] = e_a(new) . e_b(old)
        t = np.einsum("gac,gbc->abg", self.frame, frame_prev)
        tt = self.device.asarray(t, dtype=self.rdt)
        xp, n = self.xp, self.npts
        xa, xb = x_prev[:n], x_prev[n:]
        new0 = xp.astype(tt[0, 0], x_prev.dtype)[:, None] * xa + xp.astype(tt[0, 1], x_prev.dtype)[:, None] * xb
        new1 = xp.astype(tt[1, 0], x_prev.dtype)[:, None] * xa + xp.astype(tt[1, 1], x_prev.dtype)[:, None] * xb
        return xp.concat([new0, new1], axis=0)
