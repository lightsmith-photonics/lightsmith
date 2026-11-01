"""Fit measured permittivity with a passive Drude-Lorentz model.

FDTD needs a rational permittivity it can integrate with auxiliary
differential equations; measured (tabulated) data are not usable directly.
:func:`fit_drude_lorentz` turns any material into a :class:`DrudeLorentz`::

    from lightsmith.materials_fit import fit_drude_lorentz, fit_quality

    wl = np.linspace(450, 900, 91)
    dl = fit_drude_lorentz({"kind": "library", "name": "Au"}, wl)
    fit_quality(dl, "Au", wl)["rms_rel_error"]    # ~0.01 (JC data)

Every parameter is constrained to be physical: eps_inf >= 1, and plasma
frequency, Drude damping, oscillator strengths, resonances and pole dampings
are all strictly positive, so Im(eps) > 0 at every positive frequency
(passive, hence stable in FDTD). The fit is a bounded nonlinear least-squares
fit (scipy ``least_squares``) of eps in eV, started from non-negative linear
fits and continued pole by pole, so it does not depend on a user guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import TypeAdapter
from scipy.optimize import least_squares, nnls

from .materials import HC_EV_NM, DrudeLorentz, LibraryMaterial, LorentzPole, Material

DEFAULT_TOL = 0.02  # target for both rms_rel_error and rms_imag_error when n_poles is None
DEFAULT_MAX_POLES = 5
_MIN_SAMPLES = 64  # the band is densified to at least this many points (uniform in energy)
# Im(eps) errors are measured relative to |Im eps| + IMAG_FLOOR * |eps|. Without this,
# |eps|-relative errors ignore the loss of good metals (Ag: Im eps ~ 0.3 next to Re eps ~ -30),
# which sets plasmon linewidths; the floor keeps nearly lossless dielectrics well posed.
IMAG_FLOOR = 0.05

# Bounds (eV). Lower bounds are small but strictly positive. Without the damping floors the
# fit trades Drude damping for a broad in-band pole and leaves gamma at ~0 (degenerate).
_WP = (1e-3, 50.0)
_GAMMA = (5e-3, 20.0)  # Drude damping; room-temperature metals have >~ 0.015 eV
_DELTA = (1e-8, 1e4)
_W0 = (1e-2, 100.0)
_G0 = (1e-4, 50.0)  # pole damping; far (Sellmeier-like) poles of transparent media need ~0 loss


@dataclass
class DrudeLorentzFit:
    """What :func:`fit_drude_lorentz` did, returned with ``return_info=True``."""

    model: DrudeLorentz
    n_poles: int
    drude: bool
    rms_rel_error: float  # sqrt(mean(|eps_fit - eps|^2 / |eps|^2)) over the fitted samples
    max_rel_error: float
    rms_imag_error: float  # RMS of |Im eps_fit - Im eps| / (|Im eps| + IMAG_FLOOR |eps|)
    max_imag_error: float
    rms_abs_error: float
    max_abs_error: float
    band_nm: tuple[float, float]
    wavelengths_nm: np.ndarray = field(repr=False)
    eps_data: np.ndarray = field(repr=False)
    eps_fit: np.ndarray = field(repr=False)
    tried: dict[int, float] = field(default_factory=dict)  # n_poles -> max(rms_rel_error, rms_imag_error)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (without the sample arrays)."""
        return {
            "model": self.model.model_dump(), "n_poles": self.n_poles, "drude": self.drude,
            "rms_rel_error": self.rms_rel_error, "max_rel_error": self.max_rel_error,
            "rms_imag_error": self.rms_imag_error, "max_imag_error": self.max_imag_error,
            "rms_abs_error": self.rms_abs_error, "max_abs_error": self.max_abs_error,
            "band_nm": list(self.band_nm), "n_samples": int(len(self.wavelengths_nm)),
            "tried": {str(k): v for k, v in self.tried.items()}, "message": self.message,
        }


def _as_material(material):
    """Accept a material object, a material dict / JSON spec, or a library name ("Au")."""
    if hasattr(material, "eps"):
        return material
    if isinstance(material, str):
        return LibraryMaterial(name=material)
    return TypeAdapter(Material).validate_python(material)


def _errors(eps_fit: np.ndarray, eps_data: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    diff = np.abs(eps_fit - eps_data)
    rel = diff / np.maximum(np.abs(eps_data), 1e-30)
    im = np.abs(eps_fit.imag - eps_data.imag) / _imag_scale(eps_data)
    w = np.ones_like(rel) if weights is None else weights / weights.sum() * len(rel)
    return {
        "rms_rel_error": float(np.sqrt(np.mean(w * rel**2))), "max_rel_error": float(rel.max()),
        "rms_imag_error": float(np.sqrt(np.mean(w * im**2))), "max_imag_error": float(im.max()),
        "rms_abs_error": float(np.sqrt(np.mean(w * diff**2))), "max_abs_error": float(diff.max()),
    }


def _imag_scale(eps_data: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(eps_data.imag) + IMAG_FLOOR * np.abs(eps_data), 1e-30)


def fit_quality(model, material, wavelengths_nm) -> dict[str, float]:
    """Error of ``model`` (e.g. a fitted :class:`DrudeLorentz`) against ``material`` at ``wavelengths_nm``.

    Returns ``rms_rel_error`` and ``max_rel_error`` (of |eps_model - eps| / |eps|),
    ``rms_imag_error`` and ``max_imag_error`` (of |Im eps_model - Im eps| / (|Im eps| + 0.05 |eps|),
    the loss that sets resonance widths), ``rms_abs_error`` and ``max_abs_error``
    (of |eps_model - eps|), plus ``band_nm``."""
    wl = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    out: dict[str, Any] = _errors(np.asarray(_as_material(model).eps(wl)), np.asarray(_as_material(material).eps(wl)))
    out["band_nm"] = (float(wl.min()), float(wl.max()))
    return out


# --- the model in a flat parameter vector -------------------------------------
# x = [eps_inf, (wp, gamma if drude), (delta_j, omega0_j, Gamma_j) * n]


def _unpack(x: np.ndarray, drude: bool):
    i = 3 if drude else 1
    return x[0], (x[1], x[2]) if drude else None, x[i:].reshape(-1, 3)


def _model_and_jac(x: np.ndarray, w: np.ndarray, drude: bool):
    eps_inf, dr, poles = _unpack(x, drude)
    e = np.full(w.shape, complex(eps_inf))
    cols = [np.ones_like(e)]
    if dr is not None:
        wp, g = dr
        den = w**2 + 1j * g * w
        e = e - wp**2 / den
        cols += [-2 * wp / den, wp**2 * 1j * w / den**2]
    for d, w0, gg in poles:
        den = w0**2 - w**2 - 1j * gg * w
        e = e + d * w0**2 / den
        cols += [w0**2 / den, 2 * d * w0 * (den - w0**2) / den**2, d * w0**2 * 1j * w / den**2]
    return e, np.stack(cols, axis=1)


def _to_model(x: np.ndarray, drude: bool) -> DrudeLorentz:
    eps_inf, dr, poles = _unpack(x, drude)
    return DrudeLorentz(
        eps_inf=float(eps_inf), wp_ev=float(dr[0]) if dr else 0.0, gamma_ev=float(dr[1]) if dr else 0.0,
        poles=[LorentzPole(delta_eps=float(d), omega0_ev=float(w0), gamma_ev=float(g))
               for d, w0, g in sorted(poles.tolist(), key=lambda p: p[1])])


def _bounds(n: int, drude: bool, eps_inf_min: float, w0_min: float):
    lo, hi = [eps_inf_min], [np.inf]
    if drude:
        lo += [_WP[0], _GAMMA[0]]
        hi += [_WP[1], _GAMMA[1]]
    lo += [_DELTA[0], w0_min, _G0[0]] * n
    hi += [_DELTA[1], _W0[1], _G0[1]] * n
    return np.array(lo), np.array(hi)


class _Problem:
    """Weighted residuals of eps(omega): Re(eps) relative to |eps|, Im(eps) relative to
    |Im eps| + IMAG_FLOOR |eps|, i.e. the rms_rel_error and rms_imag_error the user sees."""

    def __init__(self, w, data, sample_w, drude, eps_inf_min):
        self.w, self.data, self.drude, self.eps_inf_min = w, data, drude, eps_inf_min
        # Resonances far below the band are degenerate with Drude damping (the fit would
        # trade gamma for a broad low-energy pole), so keep them above half the lowest energy.
        self.w0_min = max(_W0[0], 0.5 * float(w.min()))
        self.scale = np.sqrt(sample_w) / np.maximum(np.abs(data), 1e-12)  # applied to the real part
        self.scale_im = np.sqrt(sample_w) / _imag_scale(data)

    def residual(self, x):
        e, _ = _model_and_jac(x, self.w, self.drude)
        r = e - self.data
        return np.concatenate([r.real * self.scale, r.imag * self.scale_im])

    def jac(self, x):
        _, j = _model_and_jac(x, self.w, self.drude)
        return np.concatenate([j.real * self.scale[:, None], j.imag * self.scale_im[:, None]])

    def linear_start(self, gamma, poles) -> np.ndarray:
        """Best non-negative eps_inf - eps_inf_min, wp^2 and delta_j for fixed gamma and (omega0_j, Gamma_j)."""
        w = self.w
        cols = [np.ones_like(w, dtype=complex)]
        if self.drude:
            cols.append(-1 / (w**2 + 1j * gamma * w))
        cols += [w0**2 / (w0**2 - w**2 - 1j * g * w) for w0, g in poles]
        a = np.stack(cols, axis=1)
        b = self.data - self.eps_inf_min
        coef, _ = nnls(np.concatenate([a.real * self.scale[:, None], a.imag * self.scale_im[:, None]]),
                       np.concatenate([b.real * self.scale, b.imag * self.scale_im]))
        x = [self.eps_inf_min + coef[0]]
        k = 1
        if self.drude:
            x += [np.sqrt(max(coef[1], _WP[0] ** 2)), gamma]
            k = 2
        for (w0, g), d in zip(poles, coef[k:]):
            x += [max(d, _DELTA[0] * 10), w0, g]
        return np.array(x, dtype=float)

    def refine(self, x0: np.ndarray):
        n = (len(x0) - (3 if self.drude else 1)) // 3
        lo, hi = _bounds(n, self.drude, self.eps_inf_min, self.w0_min)
        x0 = np.clip(x0, lo + 1e-12 * np.maximum(1, np.abs(lo)), hi)
        try:
            sol = least_squares(self.residual, x0, jac=self.jac, bounds=(lo, hi), x_scale="jac",
                                method="trf", max_nfev=50 * (len(x0) + 1), ftol=1e-10, xtol=1e-10, gtol=1e-10)
        except (ValueError, np.linalg.LinAlgError):
            return x0, float(np.sum(self.residual(x0) ** 2))
        return sol.x, float(np.sum(sol.fun**2))


def _fit_n(prob: _Problem, n: int, prev: np.ndarray | None, wmin: float, wmax: float, n_refine: int = 3):
    """Best fit with n Lorentz poles. Candidate starts (the (n-1)-pole fit plus one new
    pole below, inside or above the band; poles spread over and above the band) get
    linear non-negative amplitudes, are ranked by cost, and the best few are refined."""
    starts = []
    gammas = [0.03, 0.3] if prob.drude else [0.0]
    if prev is not None and n > 0:
        _, dr, poles = _unpack(prev, prob.drude)
        old = [(pw, pg) for _, pw, pg in poles]
        # interband transitions of metals usually sit at or above the band's upper edge
        for w0 in np.geomspace(1.01 * prob.w0_min, min(3.0 * wmax, _W0[1] / 2), 8):
            for frac in (0.1, 0.4):
                starts.append(prob.linear_start(dr[1] if dr else 0.0, old + [(w0, frac * w0)]))
    if n > 0:
        spread = np.geomspace(0.8 * wmin, 3.0 * wmax, n + 2)[1:-1] if n > 1 else [1.2 * wmax]
        starts += [prob.linear_start(g, [(w0, 0.3 * w0) for w0 in spread]) for g in gammas]
    else:
        starts += [prob.linear_start(g, []) for g in gammas]
    costs = [float(np.sum(prob.residual(x) ** 2)) for x in starts]
    best = (None, np.inf)
    for i in np.argsort(costs)[:n_refine]:
        x, cost = prob.refine(starts[i])
        if cost < best[1]:
            best = (x, cost)
    return best[0]


def fit_drude_lorentz(material, wavelengths_nm, n_poles: int | None = None, *, drude: bool | None = None,
                      weights=None, max_poles: int = DEFAULT_MAX_POLES, tol: float = DEFAULT_TOL,
                      eps_inf_min: float = 1.0, return_info: bool = False):
    """Fit a passive :class:`DrudeLorentz` to ``material`` over the band spanned by ``wavelengths_nm``.

    ``material`` is any material object, its JSON dict, or a library name ("Au").
    The fit uses ``wavelengths_nm`` (vacuum, nm) as samples, densified to at least
    64 points uniform in photon energy across [min, max] when fewer are given;
    outside the band the model is only constrained to stay physical.
    ``weights`` (same length as ``wavelengths_nm``) up-weights parts of the band.

    The objective weights Re(eps) by 1/|eps| and Im(eps) by 1/(|Im eps| + 0.05 |eps|),
    so the loss of good metals (which sets plasmon linewidths) is fitted too.

    ``n_poles`` is the number of Lorentz poles. With ``None`` it is chosen
    automatically: the smallest number (up to ``max_poles``) for which both
    ``rms_rel_error`` and ``rms_imag_error`` are below ``tol``, or else the best
    found (stopping once a pole improves them by < 10 %). ``drude`` adds the
    free-electron term; ``None`` adds it when Re(eps) < 0 somewhere in the band (metals).

    Returns the :class:`DrudeLorentz`; with ``return_info=True`` returns
    ``(model, DrudeLorentzFit)`` carrying the RMS / max errors. See also :func:`fit_quality`."""
    mat = _as_material(material)
    wl = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float)).ravel()
    if wl.size == 0 or np.any(~np.isfinite(wl)) or np.any(wl <= 0):
        raise ValueError("wavelengths_nm must be a non-empty array of positive wavelengths in nm")
    if n_poles is not None and n_poles < 0:
        raise ValueError("n_poles must be >= 0 (or None to choose automatically)")
    sw = np.ones_like(wl) if weights is None else np.broadcast_to(np.asarray(weights, dtype=float), wl.shape).copy()
    if np.any(sw < 0) or not np.any(sw > 0):
        raise ValueError("weights must be non-negative and not all zero")
    lo, hi = float(wl.min()), float(wl.max())
    if np.unique(wl).size < _MIN_SAMPLES and hi > lo:
        extra = HC_EV_NM / np.linspace(HC_EV_NM / hi, HC_EV_NM / lo, _MIN_SAMPLES)
        # extra samples carry the mean user weight, so the user's points keep their emphasis
        sw = np.concatenate([sw, np.full(extra.shape, sw.mean())])
        wl = np.concatenate([wl, extra])
    order = np.argsort(wl)
    wl, sw = wl[order], sw[order]
    data = np.asarray(mat.eps(wl), dtype=complex)  # raises MaterialRangeError outside the data
    if not np.all(np.isfinite(data)):
        raise ValueError("material permittivity is not finite in the requested band")
    w = HC_EV_NM / wl
    if drude is None:
        drude = bool(np.any(data.real < 0))
    prob = _Problem(w, data, sw / sw.mean(), drude, eps_inf_min)

    fits: dict[int, np.ndarray] = {}
    errs: dict[int, float] = {}
    counts = [n_poles] if n_poles is not None else range(0, max_poles + 1)
    prev = None
    for n in counts:
        if n_poles is not None and n > 0:  # build up pole by pole for a good start
            for m in range(n):
                prev = _fit_n(prob, m, prev, w.min(), w.max())
        x = _fit_n(prob, n, prev, w.min(), w.max())
        fits[n], prev = x, x
        e, _ = _model_and_jac(x, w, drude)
        st = _errors(e, data, sw)
        errs[n] = max(st["rms_rel_error"], st["rms_imag_error"])
        if n_poles is None and (errs[n] <= tol or (n >= 1 and errs[n] > 0.9 * errs[n - 1])):
            break  # good enough, or the last pole improved the error by less than 10%
    if n_poles is None:
        reached = [n for n in errs if errs[n] <= tol]
        # fewest poles reaching tol, else fewest within 10% of the best error found
        chosen = min(reached) if reached else min(n for n in errs if errs[n] <= 1.1 * min(errs.values()))
    else:
        chosen = n_poles
    model = _to_model(fits[chosen], drude)
    if not return_info:
        return model
    eps_fit = np.asarray(model.eps(wl))
    stats = _errors(eps_fit, data, sw)
    msg = (f"{chosen} Lorentz pole(s){' + Drude' if drude else ''} over {lo:g}-{hi:g} nm; RMS relative "
           f"error {stats['rms_rel_error']:.3g}, of Im(eps) {stats['rms_imag_error']:.3g}")
    if n_poles is None and errs[chosen] > tol:
        msg += f" (target {tol:g} not reached with up to {max(errs)} poles)"
    info = DrudeLorentzFit(model=model, n_poles=chosen, drude=drude, band_nm=(lo, hi), wavelengths_nm=wl,
                           eps_data=data, eps_fit=eps_fit, tried=errs, message=msg, **stats)
    return model, info
