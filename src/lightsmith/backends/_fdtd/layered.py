"""Exact discrete background field of a layered medium at oblique incidence (fixed angle, broadband).

A plane wave at a fixed angle has the same transverse slowness s = n sin(theta) (cos phi, sin phi) at every
frequency, so its background field (incident, reflected and transmitted waves) is F(z, t - s . r_par). Per
frequency, the 3D Yee equations for fields proportional to exp(i omega s . r_par) reduce to a 1D discrete
problem along z: transverse differences become i K_x, i K_y with K = 2 sin(omega s dx / 2) / dx, the leapfrog
time step becomes Omega = 2 sin(omega dt / 2) / dt, and every node has the discrete response of its D-form ADE
(the same layer meshing as ``substrate.layered_background``). ``solve`` integrates that problem with
radiation conditions above and below (a QR-stabilised sweep of the outgoing subspace, so total internal
reflection is fine); an inverse FFT gives F at any time. The fields satisfy the main grid's own equations,
so a TFSF box fed by them does not leak, apart from the time interpolation of the tables.

Field arrays are indexed by node k: E_x, E_y, H_z at z_k; E_z, H_x, H_y at z_k + dz / 2.
"""

from __future__ import annotations

import numpy as np

from .engine import pole_coefficients
from .media import Medium


def discrete_eps(m: Medium, omega: np.ndarray, dt: float) -> np.ndarray:
    """eps(omega) of the D-form ADE update: eps_inf + sum_p c3 / (Z - c1 - c2 / Z), Z = exp(-i omega dt)."""
    Z = np.exp(-1j * np.asarray(omega, dtype=float) * dt)
    eps = m.eps_inf + 0j * Z
    for p in m.poles:
        c1, c2, c3 = pole_coefficients(p, dt)
        eps = eps + c3 / (Z - c1 - c2 / Z)
    return eps


def layer_eta(z: np.ndarray, normal: bool, omega: np.ndarray, dt: float, dz: float, z_int: float, host: Medium,
              sub: Medium) -> np.ndarray:
    """1/eps (n_omega, n_z) seen by E nodes at heights z under the layer meshing of layered_background:
    substrate fraction f of the node's pixel, <1/eps> for the normal component, 1/<eps> for tangential ones."""
    f = np.clip((z_int - (np.asarray(z, dtype=float) - dz / 2)) / dz, 0.0, 1.0)[None, :]
    e_h = discrete_eps(host, omega, dt)[:, None]
    e_s = discrete_eps(sub, omega, dt)[:, None]
    if normal:
        return f / e_s + (1 - f) / e_h
    return 1 / (f * e_s + (1 - f) * e_h)


def _transfer(eta_t, eta_z, Kx, Ky, Om, dz):
    """M (n, 4, 4): u_{k+1} = M u_k, u_k = (E_x, E_y at k; H_x, H_y at k - 1/2); eta_t at node k, eta_z at k + 1/2."""
    n = len(Om)
    MH = np.tile(np.eye(4, dtype=complex), (n, 1, 1))
    # H_x' = H_x + dz (-i Om E_y / eta_t + i Kx H_z), H_y' = H_y + dz (i Ky H_z + i Om E_x / eta_t), H_z = (Kx E_y - Ky E_x) / Om
    MH[:, 2, 0] = dz * (-1j * Kx * Ky / Om)
    MH[:, 2, 1] = dz * (-1j * Om / eta_t + 1j * Kx * Kx / Om)
    MH[:, 3, 0] = dz * (1j * Om / eta_t - 1j * Ky * Ky / Om)
    MH[:, 3, 1] = dz * (1j * Ky * Kx / Om)
    ME = np.tile(np.eye(4, dtype=complex), (n, 1, 1))
    # E_z = -eta_z (Kx H_y - Ky H_x) / Om; E_y' = E_y + dz (-i Om H_x + i Ky E_z); E_x' = E_x + dz (i Om H_y + i Kx E_z)
    ez_hx, ez_hy = eta_z * Ky / Om, -eta_z * Kx / Om
    ME[:, 1, 2] = dz * (-1j * Om + 1j * Ky * ez_hx)
    ME[:, 1, 3] = dz * (1j * Ky * ez_hy)
    ME[:, 0, 2] = dz * (1j * Kx * ez_hx)
    ME[:, 0, 3] = dz * (1j * Om + 1j * Kx * ez_hy)
    return ME @ MH


def _modes(M, dz):
    """Eigenvectors of a uniform cell's transfer matrix split into (down, up) bases, each (n, 4, 2).

    lambda = exp(i k_z dz); down: decaying towards -z (Im k_z < 0) or propagating with Re k_z < 0."""
    lam, V = np.linalg.eig(M)
    kz = -1j * np.log(lam) / dz
    tol = 1e-9 / dz
    key = np.where(np.abs(kz.imag) > tol, kz.imag, 1e-12 * kz.real)
    order = np.argsort(key, axis=1)
    V = np.take_along_axis(V, order[:, None, :], axis=2)
    return V[:, :, :2], V[:, :, 2:]


def solve(eta_t, eta_z, Kx, Ky, Om, dz, e_t, from_top: bool):
    """Background fields for an incident wave whose tangential E at the incidence-side end node is e_t (2,).

    eta_t, eta_z: (n_omega, n_k), uniform at both ends (the media continue beyond). Returns a dict of
    (n_omega, n_k) arrays Ex, Ey, Ez, Hx, Hy, Hz (positions in the module docstring) and the incident wave
    alone at the end node, ``inc`` (n_omega, 3): E_x, E_y (node) and E_z (node + 1/2)."""
    nw, nk = eta_t.shape
    Ms = np.stack([_transfer(eta_t[:, k], eta_z[:, k], Kx, Ky, Om, dz) for k in range(nk)], axis=1)  # (nw, nk, 4, 4)
    dn_b, up_b = _modes(Ms[:, 0], dz)
    dn_t, up_t = _modes(Ms[:, -1], dz)
    B = np.zeros((nw, nk, 4, 2), dtype=complex)
    R = np.zeros((nw, nk, 2, 2), dtype=complex)
    if from_top:
        B[:, 0] = np.linalg.qr(dn_b)[0]
        for k in range(nk - 1):
            Q, R[:, k] = np.linalg.qr(Ms[:, k] @ B[:, k])
            B[:, k + 1] = Q
        inc, out, end = dn_t, up_t, nk - 1
    else:
        B[:, -1] = np.linalg.qr(up_t)[0]
        Minv = np.linalg.inv(Ms)
        for k in range(nk - 2, -1, -1):
            Q, R[:, k] = np.linalg.qr(Minv[:, k] @ B[:, k + 1])
            B[:, k] = Q
        inc, out, end = up_b, dn_b, 0
    a = np.linalg.solve(inc[:, :2, :], np.broadcast_to(np.asarray(e_t, dtype=complex), (nw, 2))[..., None])  # (nw, 2, 1)
    u_inc = inc @ a  # incident wave at the end node
    A = np.concatenate([B[:, end], -out], axis=2)  # (nw, 4, 4): B c - out r = inc a
    cr = np.linalg.solve(A, u_inc)
    c = np.zeros((nw, nk, 2, 1), dtype=complex)
    c[:, end] = cr[:, :2]
    if from_top:
        for k in range(nk - 2, -1, -1):
            c[:, k] = np.linalg.solve(R[:, k], c[:, k + 1])
    else:
        for k in range(1, nk):
            c[:, k] = np.linalg.solve(R[:, k - 1], c[:, k - 1])
    u = (B @ c)[..., 0]  # (nw, nk, 4)
    Ex, Ey = u[..., 0], u[..., 1]
    Kx_, Ky_, Om_ = Kx[:, None], Ky[:, None], Om[:, None]
    Hz = (Kx_ * Ey - Ky_ * Ex) / Om_
    # H at k + 1/2 from the H half of the transfer step
    Hx = u[..., 2] + dz * (-1j * Om_ * Ey / eta_t + 1j * Kx_ * Hz)
    Hy = u[..., 3] + dz * (1j * Ky_ * Hz + 1j * Om_ * Ex / eta_t)
    Ez = -eta_z * (Kx_ * Hy - Ky_ * Hx) / Om_
    ui = u_inc[..., 0]
    hz_i = (Kx * ui[:, 1] - Ky * ui[:, 0]) / Om
    et, ez = eta_t[:, end], eta_z[:, end]
    hx_i = ui[:, 2] + dz * (-1j * Om * ui[:, 1] / et + 1j * Kx * hz_i)
    hy_i = ui[:, 3] + dz * (1j * Ky * hz_i + 1j * Om * ui[:, 0] / et)
    ez_i = -ez * (Kx * hy_i - Ky * hx_i) / Om
    return {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz,
            "inc": np.stack([ui[:, 0], ui[:, 1], ez_i], axis=1)}


_STENCIL = np.arange(-2, 4)  # quintic: relative error ~5e-3 (omega h)^6 for samples h apart (cubic: 2e-2 (omega h)^4)
_DENOM = [float(np.prod([xj - xk for xk in _STENCIL if xk != xj])) for xj in _STENCIL]
OMEGA_H = 0.2  # table sample spacing h: omega h at omega_c + 3 omega_half (interpolation error 3e-7 there, 4e-8 at the band edge)


def read_tables(xp, T, n, m: int, cmax, base, i0, u0):
    """Quintic interpolation of tables sampled every m steps: sum_j w_j(u) T[base + c + j] with
    (n + i0 + u0) / m = c + u, c clamped to cmax (zeros past the end). The fine position is split into the
    integers n + i0 and the fraction u0 so that u keeps its precision in float32."""
    q = n + i0
    r = q % m
    c = xp.minimum((q - r) // m, cmax)
    u = (xp.astype(r, u0.dtype) + u0) / m
    d = [u - float(x) for x in _STENCIL]
    idx = base + c
    out = None
    for j in range(len(_STENCIL)):
        w = None
        for k in range(len(_STENCIL)):
            if k != j:
                w = d[k] if w is None else w * d[k]
        term = (w / _DENOM[j]) * T[idx + j]
        out = term if out is None else out + term
    return out


class _View:
    """E_flat / H_flat for DFTMonitor(source=...): the incident wave alone (E_x, E_y, E_z) at the reference point."""

    E_flat = H_flat = None


_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


class ObliqueLayeredPlaneWave:
    """TFSF box [lo, hi] (node-index units) for a plane wave at a fixed oblique angle in a layered background.

    The background fields come from tables F_c(z, tau), tau = t - s . (r_par - r_c), built by `solve` at the
    frequencies of the pulse and an inverse FFT; every correction point reads its component at its own z
    and delay with quintic interpolation. r_c is the box corner the wave reaches first, so all delays are
    >= 0. The reference point for the incident wave is
    the incidence-side end node of the table range at r_c. The incidence medium must be non-dispersive (a
    fixed angle then has a fixed s). Tables are stored every m steps, the largest m with omega h <= OMEGA_H
    (m ~ 3 at 100 cells per wavelength), so that many of them (a Gaussian beam) stay small; the interpolation
    error only leaks, and C_sca of a null particle stays below 1e-16 nm^2.
    With ``focus`` (grid coordinates, nm) the pulse is delayed so that it reaches the focus t_focus after the
    unshifted pulse reaches the reference point: beam components given one t_focus arrive there in phase.

    Without ``sub`` the background is the uniform ``host`` (a leak-free TFSF plane wave at any angle). The
    1D problem then runs along the axis with the largest |k| component: the grid is relabelled by the cyclic
    axis permutation that takes that axis to z, which keeps the Yee staggering, and "z" below is that axis.
    ``origin``, ``spacing``, ``r_c``, ``s`` and the tables are in the relabelled frame; ``shape``, the flat
    indices given to `background`, the monitor view and `incident_reference` are in the grid's own frame.
    ``mirror``: the grid's mirror planes (engine.e_region); only stepped components are corrected, and the box's
    lo is the image of hi there, so delays and tables are those of the whole box."""

    def __init__(self, *, shape, origin, spacing, dt, lo, hi, khat, ehat, inv_eps, host: Medium,
                 z_int: float = -np.inf, sub: Medium | None = None, omega_c, omega_half, max_steps: int,
                 omega_min: float | None = None, focus=None, t_focus: float = 0.0, mirror=(0, 0, 0)):
        from .tfsf import tfsf_slabs

        khat, ehat = np.asarray(khat, dtype=float), np.asarray(ehat, dtype=float)
        a = 2 if sub is not None else int(np.argmax(np.abs(khat)))
        self.perm = perm = np.array([(j + a + 1) % 3 for j in range(3)])  # relabelled axis j = grid axis perm[j]
        self.inv = inv = np.argsort(perm)
        self.shape, self.dt = tuple(shape), float(dt)
        sp_grid = np.broadcast_to(np.asarray(spacing, dtype=float), (3,)).copy()
        sp = sp_grid[perm]
        self.origin, self.spacing = np.asarray(origin, dtype=float)[perm], sp
        self.z_int, self.host, self.sub = float(z_int), host, host if sub is None else sub
        sub = self.sub
        lo_grid, hi_grid = np.asarray(lo), np.asarray(hi)
        lo, hi = lo_grid[perm], hi_grid[perm]
        khat, ehat = khat[perm], ehat[perm]
        self.down = bool(khat[2] < 0)
        inc_m = host if self.down else sub
        if inc_m.dispersive:
            raise ValueError("oblique incidence needs a non-dispersive incidence medium")
        self.n_inc = float(np.sqrt(inc_m.eps_inf))
        self.khat, self.ehat = khat, ehat
        self.s = s = self.n_inc * khat[:2]  # transverse slowness (c = 1)
        dz = sp[2]
        self.k0 = int(lo[2]) - 2
        self.nk = nk = int(hi[2]) + 2 - self.k0 + 1
        self.z_nodes = self.origin[2] + (self.k0 + np.arange(nk)) * dz
        corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])], dtype=float)
        corners = self.origin[:2] + corners * sp[:2]
        self.r_c = corners[np.argmin(corners @ s)]
        r_ref = np.array([*self.r_c, self.z_nodes[-1] if self.down else self.z_nodes[0]])
        self.r_ref = r_ref[inv]  # the reference point in grid coordinates
        self.shift = 0.0
        if focus is not None:
            self.shift = float(t_focus - self.n_inc * (np.asarray(khat)[inv] @ (np.asarray(focus) - self.r_ref)))
            if self.shift < 0:
                raise ValueError("t_focus is too small for the focus to be reached after the reference point")
        self.m = m = max(1, int(OMEGA_H / ((omega_c + 3 * omega_half) * dt)))
        raw = []
        for kind, c, region, g, coef, q in tfsf_slabs(self.shape, lo_grid, hi_grid, dt, sp_grid, mirror=mirror):
            name = ("H" if kind == "E" else "E") + "xyz"[inv[g]]
            q = q[..., perm]
            row = np.floor(q[..., 2] + 1e-9).astype(np.int64) - self.k0
            delay = ((self.origin[:2] + q[..., :2] * sp[:2]) - self.r_c) @ s
            shape_ = tuple(r.stop - r.start for r in region)
            arr = coef * (inv_eps[c][region] if kind == "E" else np.ones(shape_))
            raw.append((kind, c, region, name, row.reshape(-1), delay.reshape(-1), arr, shape_))
        self.tau_lead = max(float(r[5].max()) for r in raw) + 6 * m * dt
        self.tau_p = 2.0 / omega_half
        # the pulse starts at e^-25 (band-limited tables cannot follow a switch-on edge); the high-pass below
        # the band (near-static evanescent content under total internal reflection decays very slowly) spreads
        # it by ~1/omega_lo, so it starts later still
        self.omega_lo = omega_min / 3 if omega_min else None
        self.t0 = max(5.0 * self.tau_p, 10.0 / self.omega_lo if self.omega_lo else 0.0)
        n_max = max(np.sqrt(host.eps_inf), np.sqrt(max(sub.eps_inf, 1.0)))
        T = self.tau_lead + self.shift + 2 * self.t0 + 4 * nk * dz * n_max + 4 * self.tau_p
        N = 1 << int(np.ceil(np.log2(T / dt)))
        while True:
            tau = np.arange(N) * dt - self.tau_lead
            t_c = tau - self.t0 - self.shift
            g = np.exp(-((t_c / self.tau_p) ** 2)) * np.sin(omega_c * t_c)
            X = np.fft.rfft(g)
            om = 2 * np.pi * np.arange(len(X)) / (N * dt)
            if self.omega_lo:
                X = X * (1 - np.exp(-((om / self.omega_lo) ** 2)))  # Gaussian kernel in time: no long tails
            keep = (om > 0) & (om < omega_c + 6 * omega_half)  # the pulse band (beyond it the grid response is meaningless)
            f = self._solve(om[keep])
            spec = np.zeros((6, nk, len(X)), dtype=complex)
            for i, nm in enumerate(_NAMES):
                spec[i][:, keep] = np.conj(f[nm].T) * X[keep][None, :]  # numpy's e^{+i w t}: the solver's response conjugated
            table = np.fft.irfft(spec, n=N, axis=-1)
            inc_spec = np.zeros((3, len(X)), dtype=complex)
            inc_spec[:, keep] = np.conj(f["inc"].T) * X[keep][None, :]
            inc_table = np.fft.irfft(inc_spec, n=N, axis=-1)[inv]  # the grid's E_x, E_y, E_z
            if np.abs(table[..., -N // 16:]).max() < 1e-6 * np.abs(table).max():
                break
            if N >= 1 << 20:
                raise ValueError("oblique layered source: the background field does not decay within the table")
            N *= 2
        self.N = N
        # when the incident wave has passed every point (the latest reads tau = t - delay_max): t_passed, for
        # the run, once the tables fall below 1e-11 of their peak (a truncated tail leaves its share of the box
        # flux in C_abs); t_off, for the ring-down analysis, below 1e-6. The power-of-two table runs on longer.
        env = np.abs(table).reshape(-1, N).max(axis=0)
        i_off, i_end = (int(np.nonzero(env > tol * env.max())[0][-1]) for tol in (1e-6, 1e-11))
        table, inc_table = table[..., ::m], inc_table[..., ::m]
        n_c = table.shape[-1]
        pad = len(_STENCIL) + 2  # zeros after each row: reads from cmax = n_c + 2 on give 0
        L = n_c + pad
        self.cmax = n_c + 2
        self.tables = np.concatenate([table, np.zeros(table.shape[:2] + (pad,))], axis=-1)
        self.inc_table = np.concatenate([inc_table, np.zeros((3, pad))], axis=-1)
        delay_max = max(float(r[5].max()) for r in raw)
        self.t_passed = i_end * dt - self.tau_lead + delay_max
        self.t_off = i_off * dt - self.tau_lead + delay_max
        # per point: flat index of the first stencil sample (before the step's own c), and its fine position
        # at n = 0 split into an integer and a fraction
        self.slabs = []
        for kind, c, region, nm, row, delay, arr, shape_ in raw:
            f0 = (self.tau_lead - delay) / dt + (0.5 if kind == "E" else 0.0)  # E corrections read H at n + 1/2
            i0 = np.floor(f0).astype(np.int64)
            base = (_NAMES.index(nm) * nk + row) * L + _STENCIL[0]
            self.slabs.append((kind, c, region, base, i0, f0 - i0, arr, shape_))
        f0 = self.tau_lead / dt + 1  # the view holds E at the end of the step
        i0 = int(np.floor(f0))
        self.inc = (np.arange(3) * L + _STENCIL[0], np.full(3, i0), np.full(3, f0 - i0))
        self.view = _View()

    def _solve(self, w):
        sp, dz = self.spacing, self.spacing[2]
        Kx = 2 * np.sin(w * self.s[0] * sp[0] / 2) / sp[0]
        Ky = 2 * np.sin(w * self.s[1] * sp[1] / 2) / sp[1]
        Om = 2 * np.sin(w * self.dt / 2) / self.dt
        eta_t = layer_eta(self.z_nodes, False, w, self.dt, dz, self.z_int, self.host, self.sub)
        eta_z = layer_eta(self.z_nodes + dz / 2, True, w, self.dt, dz, self.z_int, self.host, self.sub)
        return solve(eta_t, eta_z, Kx, Ky, Om, dz, self.ehat[:2], self.down)

    def background(self, omega, flat: np.ndarray, kind: str = "E") -> np.ndarray:
        """Background E (kind "E") or H at flat grid indices (n_omega, n), relative to the incident wave whose
        tangential E at the reference point is ehat_t; NaN outside the table's z range."""
        n_cells = int(np.prod(self.shape))
        comp, rest = np.divmod(np.asarray(flat, dtype=np.int64), n_cells)
        idx3 = np.stack(np.unravel_index(rest, self.shape), axis=1)
        off = np.zeros((len(comp), 3))
        if kind == "E":
            off[np.arange(len(comp)), comp] = 0.5
        else:
            off[:] = 0.5
            off[np.arange(len(comp)), comp] = 0.0
        comp, idx3, off = self.inv[comp], idx3[:, self.perm], off[:, self.perm]
        w = np.asarray(omega, dtype=float)
        f = self._solve(w)
        rows = idx3[:, 2] - self.k0
        ok = (rows >= 0) & (rows < self.nk)
        r_par = self.origin[:2] + (idx3[:, :2] + off[:, :2]) * self.spacing[:2]
        phase = np.exp(1j * w[:, None] * ((r_par - self.r_c) @ self.s)[None, :])
        vals = np.zeros((len(w), len(comp)), dtype=complex)
        for c in range(3):
            sel = comp == c
            vals[:, sel] = f[kind + "xyz"[c]][:, np.clip(rows[sel], 0, self.nk - 1)]
        return np.where(ok[None, :], vals * phase, np.nan)

    def incident_reference(self, omega) -> np.ndarray:
        """The incident wave alone at the reference point, (n_omega, 3) in grid components, in the units of
        `background`."""
        return self._solve(np.asarray(omega, dtype=float))["inc"][:, self.inv]

    def bind(self, eng):
        self._T = eng.asarray(self.tables.reshape(-1))
        self._Tinc = eng.asarray(self.inc_table.reshape(-1))
        del self.tables, self.inc_table
        self._slabs = [(kind, c, region, eng.index(base), eng.index(i0), eng.asarray(u0), eng.asarray(arr), shape_)
                       for kind, c, region, base, i0, u0, arr, shape_ in self.slabs]
        self._inc = tuple(f(a) for f, a in zip((eng.index, eng.index, eng.asarray), self.inc))
        self._cmax = eng.index(np.array([self.cmax]))
        self.view.E_flat = eng.zeros((3,))
        self.view.H_flat = self.view.E_flat

    def _apply(self, eng, kind: str):
        target = eng.E if kind == "E" else eng.H
        for k, c, region, base, i0, u0, arr, shape_ in self._slabs:
            if k == kind:
                vals = read_tables(eng.xp, self._T, eng.n, self.m, self._cmax, base, i0, u0)
                target[c][region] += arr * eng.xp.reshape(vals, shape_)

    def correct_H(self, eng):
        self._apply(eng, "H")

    def correct_E(self, eng):
        self._apply(eng, "E")

    def advance_H(self, eng):
        pass

    def advance_E(self, eng):
        self.view.E_flat[...] = read_tables(eng.xp, self._Tinc, eng.n, self.m, self._cmax, *self._inc)
