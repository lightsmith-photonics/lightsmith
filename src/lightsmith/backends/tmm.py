"""Native transfer-matrix method (TMM) for planar multilayers, CPU or GPU, differentiable.

Coherent layers use the stable Airy recursion of the amplitude reflection
coefficient from the substrate upwards (equivalent to multiplying 2x2 transfer
matrices, but every phase factor is exp(i kz d) with Im(kz) >= 0, so thick
absorbing or evanescent layers never overflow). Thick incoherent layers
(``Layer.coherent=False``) are cascaded in intensity: each coherent group between
two incoherent media is reduced to its forward/backward power coefficients and
the groups are combined with the intensity-matrix method of Katsidis & Siapkas,
Appl. Opt. 41, 3978 (2002), in its equivalent stable recursive form.

Conventions (as everywhere in lightsmith): exp(-i omega t), Im(eps) >= 0 is
loss, lengths in nm, vacuum wavelengths. ``kz`` is chosen with Im(kz) > 0 (or
Re(kz) > 0 when real), which handles absorbing media and total internal
reflection. Amplitudes follow the convention of Byrnes' ``tmm`` package:

- s: ratio of the tangential E field (E along y for incidence in the xz plane);
- p: ``r_p`` is the ratio of the tangential H field, so at normal incidence
  ``r_p = -r_s``; ``t_p`` is the ratio of the full E-field amplitudes,
  ``t_p = (n_in / n_out) * t_H``.

Everything is written once against the array API (``device.namespace()``), so
it runs on NumPy, or on PyTorch where gradients flow to layer thicknesses and
permittivities. :func:`tmm_tensors` is the differentiable entry point and
:func:`optimize_thicknesses` a small gradient-descent helper (e.g. AR coatings).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..device import Device, resolve_device
from ..problems import LayerStackProblem, options_for, wavelength_array
from ..results import SpectrumResult
from .base import Backend, register


class TMMOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precision: Literal["auto", "single", "double"] = Field(
        "auto", description="auto = double (TMM is cheap; interference needs the digits).")


def _branch(xp, q, tol):
    """Pick the root with Im(q) > 0 (decaying along +z), or Re(q) > 0 when q is (numerically) real."""
    return xp.where(xp.imag(q) + tol * xp.abs(q) < 0, -q, q)


def _kz(xp, eps, kx2, tol):
    """Normalised normal wavevector kz/k0 = sqrt(eps - (kx/k0)^2)."""
    return _branch(xp, xp.sqrt(eps - kx2), tol)


def _safe_div(xp, a, b):
    small = xp.abs(b) == 0
    return xp.where(small, xp.zeros_like(a), a / xp.where(small, xp.ones_like(b), b))


def coherent_rt(xp, eps: Sequence, thick_k0: Sequence, kx2, pol: str, tol: float = 1e-10):
    """Amplitude r, t (and admittances) of a coherent stack.

    eps: permittivities of [incident, layer_1, ..., layer_N, exit] media, each
    broadcastable to the batch shape. thick_k0: the N layer thicknesses times k0
    (dimensionless, broadcastable). kx2: (kx/k0)^2. pol: "s" or "p".
    Returns (r, t, eta_in, eta_out) where t is the tangential E (s) or H (p) ratio
    and eta = kz (s) or kz/eps (p) is the normalised admittance.
    """
    q = [_kz(xp, e, kx2, tol) for e in eps]
    eta = q if pol == "s" else [qi / ei for qi, ei in zip(q, eps)]
    n = len(eps)
    # Start at the last interface and climb towards the incident medium.
    r = (eta[n - 2] - eta[n - 1]) / (eta[n - 2] + eta[n - 1])
    t = 2 * eta[n - 2] / (eta[n - 2] + eta[n - 1])
    for j in range(n - 3, -1, -1):
        rj = (eta[j] - eta[j + 1]) / (eta[j] + eta[j + 1])
        tj = 2 * eta[j] / (eta[j] + eta[j + 1])
        ph = xp.exp(1j * q[j + 1] * thick_k0[j])  # one pass through layer j+1
        den = 1 + rj * r * ph * ph
        r, t = (rj + r * ph * ph) / den, tj * t * ph / den
    return r, t, eta[0], eta[-1]


def _power(xp, r, t, eta_in, eta_out):
    return xp.real(r * xp.conj(r)), _safe_div(xp, xp.real(eta_out), xp.real(eta_in)) * xp.real(t * xp.conj(t))


def tmm_tensors(problem: LayerStackProblem, device: Device | str = "cpu", *, thicknesses=None,
                eps: dict[int, Any] | None = None, precision: str = "double") -> dict[str, dict[str, Any]]:
    """Differentiable core: R, T, A (and amplitudes r, t) as arrays of shape (n_angles, n_wavelengths).

    `thicknesses` (sequence or 1D array of length n_layers, nm) and `eps` ({layer index:
    permittivity array of shape (n_wavelengths,) or scalar}) override the problem's values;
    pass torch tensors with ``requires_grad=True`` to differentiate (use device="torch-cpu"
    or "cuda"). Returns {"R": {pol: arr}, "T": ..., "A": ..., "r": ..., "t": ...}; "r"/"t"
    are omitted when the stack has incoherent layers.
    """
    dev = resolve_device(device) if isinstance(device, str) else device
    xp = dev.namespace()
    cdt = xp.complex64 if precision == "single" else xp.complex128
    rdt = xp.float32 if precision == "single" else xp.float64
    tol = 1e-5 if precision == "single" else 1e-10
    wl = wavelength_array(problem.wavelengths)
    nl = len(problem.layers)

    def cplx(a):
        if _is_tensor(a):
            return a.to(cdt)
        return dev.asarray(np.asarray(a, dtype=complex), dtype=cdt)

    def eps_of(i, mat):
        if eps is not None and i in eps:
            e = cplx(eps[i])
            return e if e.ndim > 0 else e * xp.ones((len(wl),), dtype=cdt, device=dev.native_device())
        return cplx(mat.eps(wl))

    def thick(i):
        if thicknesses is None:
            return problem.layers[i].thickness_nm
        d = thicknesses[i]
        return d if _is_tensor(d) else float(d)

    k0 = dev.asarray(2 * np.pi / wl, dtype=rdt)[None, :]  # (1, W)
    e_sup = dev.asarray(np.asarray(problem.superstrate.eps(wl), dtype=complex), dtype=cdt)
    n_sup = xp.sqrt(e_sup)[None, :]
    sin_t = dev.asarray(np.sin(np.radians(problem.angles_deg)), dtype=cdt)[:, None]
    kx2 = (n_sup * sin_t) ** 2  # (A, W), conserved through the stack
    e_layers = [eps_of(i, l.material)[None, :] for i, l in enumerate(problem.layers)]
    e_sub = dev.asarray(np.asarray(problem.substrate.eps(wl), dtype=complex), dtype=cdt)[None, :]
    media = [e_sup[None, :]] + e_layers + [e_sub]
    d_k0 = [thick(i) * k0 for i in range(nl)]
    coherent = [l.coherent for l in problem.layers]

    out: dict[str, dict[str, Any]] = {"R": {}, "T": {}, "A": {}}
    if all(coherent):
        out["r"], out["t"] = {}, {}
    for pol in problem.polarizations:
        if all(coherent):
            r, t, ei, eo = coherent_rt(xp, media, d_k0, kx2, pol, tol)
            R, T = _power(xp, r, t, ei, eo)
            out["r"][pol] = r
            # p: report the E-field transmission ratio t_E = t_H * n_in / n_out (Byrnes' convention).
            out["t"][pol] = t if pol == "s" else t * xp.sqrt(media[0]) / xp.sqrt(media[-1])
        else:
            R, T = _incoherent(xp, media, d_k0, coherent, kx2, pol, tol)
        out["R"][pol], out["T"][pol], out["A"][pol] = R, T, 1 - R - T
    return out


def _is_tensor(a) -> bool:
    return hasattr(a, "detach") or (hasattr(a, "shape") and not isinstance(a, (np.ndarray, np.generic)))


def _incoherent(xp, media, d_k0, coherent, kx2, pol, tol):
    """Intensity cascade over coherent groups separated by incoherent layers."""
    inc = [0] + [i + 1 for i, c in enumerate(coherent) if not c] + [len(media) - 1]  # incoherent media indices
    groups = []  # (Rf, Tf, Rb, Tb) for each coherent group between inc[k] and inc[k+1]
    for a, b in zip(inc[:-1], inc[1:]):
        eps = media[a:b + 1]
        dk = d_k0[a:b - 1]  # thicknesses of the coherent layers a+1 .. b-1 (layer i <-> media i+1)
        rf, tf, ei, eo = coherent_rt(xp, eps, dk, kx2, pol, tol)
        rb, tb, ei2, eo2 = coherent_rt(xp, eps[::-1], dk[::-1], kx2, pol, tol)
        groups.append((*_power(xp, rf, tf, ei, eo), *_power(xp, rb, tb, ei2, eo2)))
    R, T = groups[-1][0], groups[-1][1]
    for k in range(len(groups) - 2, -1, -1):
        m = inc[k + 1]  # the incoherent layer between groups k and k+1 (media index)
        q = _kz(xp, media[m], kx2, tol)
        P = xp.exp(-2 * xp.imag(q) * d_k0[m - 1])  # single-pass power attenuation
        Rf, Tf, Rb, Tb = groups[k]
        den = 1 - Rb * P * P * R
        R, T = Rf + Tf * Tb * P * P * R / den, Tf * P * T / den
    return R, T


def _to_list(a) -> list:
    return Device.to_numpy(a).astype(float).tolist()


@register
class NativeTMM(Backend):
    name = "lightsmith-tmm"
    method = "tmm"
    options_model = TMMOptions
    license = "MIT"
    citation = ("Born & Wolf, Principles of Optics, Sec. 1.6; Katsidis & Siapkas, Appl. Opt. 41, 3978 (2002); "
                "Byrnes, arXiv:1603.02720")
    gpu = True
    differentiable = True
    problem_kinds = ("layer_stack",)

    @classmethod
    def supports(cls, problem):
        if not isinstance(problem, LayerStackProblem):
            return False, "TMM solves planar layer stacks only"
        return True, ""

    @classmethod
    def _options(cls, problem) -> TMMOptions:
        return TMMOptions.model_validate(options_for(problem, cls.name, cls.method))

    @classmethod
    def _precision(cls, problem) -> str:
        p = cls._options(problem).precision
        return "double" if p == "auto" else p

    @classmethod
    def estimate(cls, problem, device):
        n = len(wavelength_array(problem.wavelengths)) * len(problem.angles_deg) * len(problem.polarizations)
        return {"points": n, "layers": len(problem.layers), "precision": cls._precision(problem),
                "incoherent_layers": sum(not l.coherent for l in problem.layers),
                "memory_bytes": int(16 * 8 * n * (len(problem.layers) + 2))}

    @classmethod
    def solve(cls, problem: LayerStackProblem, device: Device) -> SpectrumResult:
        t0 = time.perf_counter()
        prec = cls._precision(problem)
        wl = wavelength_array(problem.wavelengths)
        out = tmm_tensors(problem, device, precision=prec)
        warnings = []
        if np.any(np.imag(problem.superstrate.eps(wl)) > 1e-12):
            warnings.append("absorbing superstrate: R = |r|^2 and T are normalised to the incident "
                            "Poynting flux at the first interface")
        amp = None
        if "r" in out:
            amp = {k: {p: np.stack([Device.to_numpy(v).real, Device.to_numpy(v).imag], axis=-1).tolist()
                       for p, v in out[k].items()} for k in ("r", "t")}
        res = SpectrumResult(
            method=cls.method, backend=cls.name, wavelengths_nm=wl.tolist(), angles_deg=list(problem.angles_deg),
            polarizations=list(problem.polarizations),
            r={p: _to_list(v) for p, v in out["R"].items()}, t={p: _to_list(v) for p, v in out["T"].items()},
            a={p: _to_list(v) for p, v in out["A"].items()},
            r_amplitude=amp["r"] if amp else None, t_amplitude=amp["t"] if amp else None,
            provenance={**device.describe(), "precision": prec, "layers": len(problem.layers),
                        "incoherent_layers": sum(not l.coherent for l in problem.layers),
                        "amplitude_convention": "s: E_y ratio; p: r = H_y ratio, t = E ratio (Byrnes tmm)",
                        "runtime_s": time.perf_counter() - t0},
            warnings=warnings)
        return res


# --- optimisation helper -------------------------------------------------------


def optimize_thicknesses(problem: LayerStackProblem, objective: str | Callable[[dict], Any] = "min_reflectance",
                         *, layers: Sequence[int] | None = None, steps: int = 200, lr: float = 2.0,
                         bounds_nm: tuple[float, float] = (0.0, 1e5), device: str = "torch-cpu",
                         ) -> tuple[LayerStackProblem, dict[str, Any]]:
    """Optimise layer thicknesses by gradient descent (Adam) through the differentiable TMM.

    objective: "min_reflectance" | "max_transmittance" | "min_absorptance" | "max_absorptance"
    (mean over wavelengths, angles and polarizations), or a callable taking the dict
    returned by :func:`tmm_tensors` and returning a scalar torch tensor to minimise.
    layers: indices of the layers to optimise (default: all coherent layers).
    lr: Adam step size in nm. Returns (problem with optimised thicknesses, history).
    Requires PyTorch.
    """
    import torch

    dev = resolve_device(device)
    if dev.framework != "torch":
        raise ValueError("optimize_thicknesses needs a torch device ('torch-cpu' or 'cuda')")
    layers = list(range(len(problem.layers))) if layers is None else list(layers)
    tdev = dev.native_device()
    d = torch.tensor([l.thickness_nm for l in problem.layers], dtype=torch.float64, device=tdev)
    free = torch.tensor([i in layers for i in range(len(problem.layers))], device=tdev)
    p = d.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)

    if callable(objective):
        loss_fn = objective
    else:
        key, sign = {"min_reflectance": ("R", 1), "max_transmittance": ("T", -1),
                     "min_absorptance": ("A", 1), "max_absorptance": ("A", -1)}[objective]

        def loss_fn(o):
            return sign * torch.stack([v.mean() for v in o[key].values()]).mean()

    history = []
    for _ in range(steps):
        opt.zero_grad()
        th = torch.where(free, p, d)
        loss = loss_fn(tmm_tensors(problem, dev, thicknesses=th))
        loss.backward()
        opt.step()
        with torch.no_grad():
            p.clamp_(*bounds_nm)
        history.append(float(loss.detach()))
    final = torch.where(free, p, d).detach().cpu().numpy()
    new_layers = [l.model_copy(update={"thickness_nm": float(t)}) for l, t in zip(problem.layers, final)]
    with torch.no_grad():
        final_loss = float(loss_fn(tmm_tensors(problem, dev, thicknesses=torch.as_tensor(final, device=tdev))))
    return problem.model_copy(update={"layers": new_layers}), {
        "loss": history, "final_loss": final_loss, "thicknesses_nm": final.tolist()}


__all__ = ["NativeTMM", "TMMOptions", "coherent_rt", "optimize_thicknesses", "tmm_tensors"]
