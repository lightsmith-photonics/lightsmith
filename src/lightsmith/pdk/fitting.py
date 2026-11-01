"""Compact models of sampled S-parameters: pole-residue (vector fitting) or interpolated table.

Both work on the normalised frequency x = (nu - nu_c) / (nu_half_span), nu = 1 / lambda,
after removing a per-entry linear phase exp(i tau x) (the propagation delay), so a
long waveguide does not need many poles:

    S_ij(x) = exp(i tau_ij x) * (d_ij + sum_k r_ijk / (x - a_k))

The poles a_k are shared by all entries (vector fitting: Gustavsen & Semlyen, IEEE
Trans. Power Delivery 14, 1052 (1999), with the fast QR formulation of Deschrijver
et al., IEEE Microw. Wireless Compon. Lett. 18, 383 (2008)). With exp(-i omega t)
a causal response is analytic in the upper half frequency plane, so poles are kept
at Im(a_k) < 0. The pole count is the smallest that reproduces held-out samples
(every other wavelength) to `tol`; if none does, the model falls back to a table
interpolated in magnitude and unwrapped phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SFit:
    """A fitted model of S[w, out, in] (all arrays NumPy)."""

    kind: str  # "pole_residue" | "table"
    nu_c: float
    nu_h: float
    tau: np.ndarray  # (P, P)
    poles: np.ndarray = field(default_factory=lambda: np.zeros(0, complex))  # (N,)
    residues: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), complex))  # (P, P, N)
    const: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), complex))  # (P, P)
    x_table: np.ndarray = field(default_factory=lambda: np.zeros(0))
    s_table: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), complex))  # delay-removed samples
    fit_error: float = 0.0
    validation_error: float | None = None

    def x(self, wavelengths_nm) -> np.ndarray:
        return (1.0 / np.asarray(wavelengths_nm, dtype=float) - self.nu_c) / self.nu_h

    def evaluate(self, wavelengths_nm) -> np.ndarray:
        x = self.x(wavelengths_nm)
        if self.kind == "pole_residue":
            core = self.const[None] + np.einsum("ijk,wk->wij", self.residues,
                                                1.0 / (x[:, None] - self.poles[None, :]))
        else:
            core = _interp_polar(self.x_table, self.s_table, x)
        return core * np.exp(1j * self.tau[None] * x[:, None, None])

    def to_dict(self) -> dict:
        c = lambda a: {"re": np.real(a).tolist(), "im": np.imag(a).tolist()}  # noqa: E731
        return {"kind": self.kind, "nu_c": self.nu_c, "nu_h": self.nu_h, "tau": self.tau.tolist(),
                "poles": c(self.poles), "residues": c(self.residues), "const": c(self.const),
                "x_table": self.x_table.tolist(), "s_table": c(self.s_table), "fit_error": self.fit_error,
                "validation_error": self.validation_error}

    @classmethod
    def from_dict(cls, d: dict) -> "SFit":
        c = lambda v: np.asarray(v["re"], float) + 1j * np.asarray(v["im"], float)  # noqa: E731
        return cls(kind=d["kind"], nu_c=d["nu_c"], nu_h=d["nu_h"], tau=np.asarray(d["tau"], float),
                   poles=c(d["poles"]).reshape(-1), residues=c(d["residues"]), const=c(d["const"]),
                   x_table=np.asarray(d["x_table"], float), s_table=c(d["s_table"]),
                   fit_error=d["fit_error"], validation_error=d["validation_error"])


def _interp_polar(x: np.ndarray, s: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Interpolate s[x, i, j] in magnitude and unwrapped phase (x ascending or descending)."""
    order = np.argsort(x)
    x, s = x[order], s[order]
    if len(x) == 1:
        return np.repeat(s, len(xq), axis=0)
    flat = s.reshape(len(x), -1)
    out = np.empty((len(xq), flat.shape[1]), complex)
    for k in range(flat.shape[1]):
        mag = np.interp(xq, x, np.abs(flat[:, k]))
        ph = np.interp(xq, x, np.unwrap(np.angle(flat[:, k])))
        out[:, k] = mag * np.exp(1j * ph)
    return out.reshape((len(xq),) + s.shape[1:])


def _delays(x: np.ndarray, s: np.ndarray, rel_floor: float = 1e-2) -> np.ndarray:
    """Per-entry linear phase slope (least squares on the unwrapped phase); 0 for negligible entries."""
    P = s.shape[1]
    tau = np.zeros((P, P))
    if len(x) < 2:
        return tau
    order = np.argsort(x)
    big = np.abs(s).max()
    for i in range(P):
        for j in range(P):
            v = s[order, i, j]
            if np.abs(v).min() < rel_floor * big:
                continue
            ph = np.unwrap(np.angle(v))
            tau[i, j] = np.polyfit(x[order], ph, 1)[0]
    return tau


def _basis(x: np.ndarray, poles: np.ndarray) -> np.ndarray:
    return 1.0 / (x[:, None] - poles[None, :])


def _stabilise(poles: np.ndarray, floor: float) -> np.ndarray:
    p = np.where(poles.imag > 0, poles.conj(), poles)
    return np.where(p.imag > -floor, p.real - 1j * floor, p)


def vector_fit(x: np.ndarray, f: np.ndarray, n_poles: int, iterations: int = 20
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit f[k, m] (K samples of M responses) with shared poles: f_m = d_m + sum_n r_mn / (x - a_n).

    Returns (poles (N,), residues (M, N), const (M,))."""
    K, M = f.shape
    if n_poles == 0:
        return np.zeros(0, complex), np.zeros((M, 0), complex), f.mean(axis=0)
    span = max(np.ptp(x), 1e-12)
    floor = 1e-3 * span
    poles = np.linspace(x.min(), x.max(), n_poles) - 1j * 0.1 * span / n_poles
    for _ in range(iterations):
        phi = _basis(x, poles)
        rows, rhs = [], []
        for m in range(M):
            A = np.concatenate([phi, np.ones((K, 1)), -f[:, m:m + 1] * phi], axis=1)
            q, r = np.linalg.qr(A)
            r22 = r[n_poles + 1:, n_poles + 1:]
            rows.append(r22)
            rhs.append((q.conj().T @ f[:, m])[n_poles + 1:])
        c = np.linalg.lstsq(np.concatenate(rows), np.concatenate(rhs), rcond=None)[0]
        new = np.linalg.eigvals(np.diag(poles) - np.outer(np.ones(n_poles), c))
        new = _stabilise(new, floor)
        if np.allclose(np.sort_complex(new), np.sort_complex(poles), rtol=1e-10, atol=1e-12):
            poles = new
            break
        poles = new
    A = np.concatenate([_basis(x, poles), np.ones((K, 1))], axis=1)
    coef = np.linalg.lstsq(A, f, rcond=None)[0]  # (N + 1, M)
    return poles, coef[:n_poles].T, coef[n_poles]


def fit_sparameters(wavelengths_nm, s: np.ndarray, kind: str = "auto", tol: float = 5e-3,
                    max_poles: int = 12) -> SFit:
    """Fit S[w, out, in] sampled at `wavelengths_nm`. kind: auto | pole_residue | table."""
    wl = np.asarray(wavelengths_nm, dtype=float)
    s = np.asarray(s, dtype=complex)
    nu = 1.0 / wl
    nu_c = float((nu.max() + nu.min()) / 2)
    nu_h = float(max((nu.max() - nu.min()) / 2, 1e-12))
    x = (nu - nu_c) / nu_h
    tau = _delays(x, s)
    core = s * np.exp(-1j * tau[None] * x[:, None, None])
    P = s.shape[1]
    table = SFit("table", nu_c, nu_h, tau, x_table=x, s_table=core)
    if kind == "table" or len(wl) < 3:
        table.validation_error = _holdout_table(x, core) if len(wl) >= 3 else None
        return table
    flat = core.reshape(len(x), -1)
    even, odd = np.arange(0, len(x), 2), np.arange(1, len(x), 2)

    def holdout(n: int) -> float:
        p_e, r_e, d_e = vector_fit(x[even], flat[even], n)
        return float(np.abs(d_e[None] + _basis(x[odd], p_e) @ r_e.T - flat[odd]).max())

    chosen, best_val = None, np.inf
    for n in range(0, max_poles + 1):
        if 2 * n + 1 > len(even):
            break
        val = holdout(n)
        if val < best_val:
            chosen, best_val = n, val
        if val <= tol:
            break
    if chosen is None or (best_val > tol and kind == "auto"):
        table.validation_error = _holdout_table(x, core)
        return table
    # Refit on all samples: free poles, or the held-out fit's poles with residues from all samples.
    candidates = [vector_fit(x, flat, chosen)[0], vector_fit(x[even], flat[even], chosen)[0]]
    fits = []
    for poles in candidates:
        A = np.concatenate([_basis(x, poles), np.ones((len(x), 1))], axis=1)
        coef = np.linalg.lstsq(A, flat, rcond=None)[0]
        f = SFit("pole_residue", nu_c, nu_h, tau, poles=poles, residues=coef[:chosen].T.reshape(P, P, chosen),
                 const=coef[chosen].reshape(P, P), validation_error=best_val)
        f.fit_error = float(np.abs(f.evaluate(wl) - s).max())
        fits.append(f)
    fit = min(fits, key=lambda f: f.fit_error)
    if fit.fit_error > tol and kind == "auto":
        table.validation_error = _holdout_table(x, core)
        return table
    return fit


def _holdout_table(x: np.ndarray, core: np.ndarray) -> float | None:
    even, odd = np.arange(0, len(x), 2), np.arange(1, len(x), 2)
    if len(odd) == 0 or len(even) < 2:
        return None
    inner = odd[(x[odd] > min(x[even].min(), x[even].max())) & (x[odd] < max(x[even].min(), x[even].max()))]
    if len(inner) == 0:
        return None
    return float(np.abs(_interp_polar(x[even], core[even], x[inner]) - core[inner]).max())
