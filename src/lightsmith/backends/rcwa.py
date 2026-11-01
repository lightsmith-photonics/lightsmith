"""Native rigorous coupled-wave analysis (RCWA / Fourier modal method), CPU or GPU, differentiable.

Solves 1D and 2D periodic stacks of patterned layers (``PeriodicProblem``,
including non-rectangular lattices and conical incidence) and unpatterned
stacks (``LayerStackProblem``, a single harmonic). Ingredients:

- Fourier coefficients of eps and 1/eps from exact segment transforms (1D),
  closed-form shape transforms or supersampled rasters (2D), see ``_rcwa.fourier``.
- Correct Fourier factorization: Li's inverse rule for the field component normal
  to 1D grating lines (Li, JOSA A 13, 1870 (1996); Lalanne & Morris and Granet &
  Guizal, JOSA A 13 (1996)), and its 2D generalisation with a normal-vector field
  (Schuster et al., JOSA A 24, 2880 (2007)). ``factorization="laurent"`` gives the
  naive rule for comparison.
- Eigenmodes per layer, stable Redheffer-star S-matrix cascade (``_rcwa.core``).
- Total R/T/A and per-order efficiencies from the Poynting flux of each harmonic,
  and zeroth-order complex amplitudes with the same conventions as the TMM
  backend (s: E ratio; p: r = tangential-H ratio, t = E ratio).

Orders: harmonic (m, n) has in-plane wavevector k_inc + m b1 + n b2 with
a_i . b_j = 2 pi delta_ij; for 1D gratings n = 0.

With a torch device, :func:`rcwa_tensors` is differentiable with respect to layer
thicknesses and pixelated permittivity maps (``eps_grids``), for metasurface
inverse design.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..device import Device, resolve_device
from ..problems import LayerStackProblem, PeriodicProblem, options_for, wavelength_array
from ..results import DiffractionOrder, SpectrumResult
from ._rcwa import (Core, Lattice, PatternedMatrices, form_factors, harmonic_set, is_tensor,
                    normal_field_coefficients, pixel_coefficients, toeplitz_index)
from .base import Backend, register


class RCWAOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    harmonics: int | tuple[int, int] | None = Field(
        None, description="Highest diffraction order kept along each lattice direction (|m| <= M, |n| <= N); "
        "an int applies to both. Default 25 for 1D gratings, 7 per direction for 2D lattices.")
    factorization: Literal["auto", "li", "normal_vector", "laurent"] = Field(
        "auto", description="auto/li/normal_vector: Li's inverse rule for 1D gratings and the normal-vector-field "
        "method for 2D lattices (needed for metals); laurent: naive Laurent rule everywhere.")
    fourier: Literal["auto", "analytic", "raster"] = Field(
        "auto", description="2D eps Fourier coefficients: closed-form shape transforms (auto, when shapes do not "
        "overlap) or a supersampled raster.")
    raster_points: int = Field(256, ge=16, description="Raster grid points per lattice direction.")
    normal_blur: float = Field(0.1, gt=0, description="Gaussian smoothing of the 2D normal-vector field, as a "
                               "fraction of the shortest lattice vector.")
    precision: Literal["auto", "single", "double"] = Field("auto", description="auto = double precision.")
    max_batch_bytes: float = Field(2e9, gt=0, description="Wavelengths are solved in batches using about this "
                                   "much memory.")


def _orders(problem, opts: RCWAOptions) -> tuple[int, int]:
    if isinstance(problem, LayerStackProblem):
        return 0, 0
    h = opts.harmonics
    if h is None:
        h = 25 if problem.lattice_a2_nm is None else 7
    m, n = (h, h) if isinstance(h, int) else h
    return int(m), (0 if problem.lattice_a2_nm is None else int(n))


def _same_material(a, b) -> bool:
    return a.model_dump_json() == b.model_dump_json()


@dataclass
class _LayerGeom:
    thickness_nm: float
    uniform_eps: np.ndarray | None = None  # (W,) for unpatterned layers
    eps_bg: np.ndarray | None = None
    eps_shapes: np.ndarray | None = None  # (S, W)
    phi: np.ndarray | None = None  # (S, 2K1+1, 2K2+1) Toeplitz-range form factors
    normals: np.ndarray | None = None  # (3, 2K1+1, 2K2+1) 2D normal-tensor coefficients
    fourier: str = "none"


@dataclass
class _Prepared:
    lattice: Lattice | None
    harm: np.ndarray  # (N, 2)
    k1: int
    k2: int
    tidx: np.ndarray  # (N, N)
    wl: np.ndarray
    eps_sup: np.ndarray
    eps_sub: np.ndarray
    layers: list[_LayerGeom]
    theta: np.ndarray  # (A,) radians
    phi: float
    pols: list[str]
    factorization: str
    blur: float
    info: dict[str, Any] = field(default_factory=dict)


def _prepare(problem, opts: RCWAOptions) -> _Prepared:
    wl = wavelength_array(problem.wavelengths)
    eps_sup = np.asarray(problem.superstrate.eps(wl), dtype=complex)
    eps_sub = np.asarray(problem.substrate.eps(wl), dtype=complex)
    if isinstance(problem, LayerStackProblem):
        layers = [_LayerGeom(l.thickness_nm, uniform_eps=np.asarray(l.material.eps(wl), dtype=complex))
                  for l in problem.layers]
        harm = np.zeros((1, 2), dtype=int)
        return _Prepared(None, harm, 0, 0, toeplitz_index(harm, 0, 0), wl, eps_sup, eps_sub, layers,
                         np.radians(np.asarray(problem.angles_deg, float)), 0.0, list(problem.polarizations),
                         "laurent", opts.normal_blur)

    lat = Lattice(problem.lattice_a1_nm, problem.lattice_a2_nm)
    m_max, n_max = _orders(problem, opts)
    harm = harmonic_set(lat.dim, m_max, n_max)
    k1, k2 = 2 * m_max, 2 * n_max
    fact = opts.factorization
    if fact in ("auto", "normal_vector"):
        fact = "li"
    # 2D normal fields need the label field on a finer range than the Toeplitz matrices.
    l1, l2 = (k1, k2) if (lat.dim == 1 or fact == "laurent") else (max(k1 + 1, 32), max(k2 + 1, 32))
    layers = []
    for layer in problem.layers:
        shapes = [s for s in layer.shapes if not _same_material(s.material, layer.background)] \
            if False else list(layer.shapes)
        eps_bg = np.asarray(layer.background.eps(wl), dtype=complex)
        if not shapes:
            layers.append(_LayerGeom(layer.thickness_nm, uniform_eps=eps_bg))
            continue
        phi_big, method = form_factors([s.profile for s in shapes], lat, l1, l2, opts.fourier,
                                       opts.raster_points)
        c1, c2 = (phi_big.shape[1] - 1) // 2, (phi_big.shape[2] - 1) // 2
        phi = phi_big[:, c1 - k1:c1 + k1 + 1, c2 - k2:c2 + k2 + 1]
        normals = None
        if lat.dim == 2 and fact == "li":
            labels, uniq = [], []
            for s in shapes:
                if _same_material(s.material, layer.background):
                    labels.append(0.0)
                    continue
                key = s.material.model_dump_json()
                if key not in uniq:
                    uniq.append(key)
                labels.append(float(uniq.index(key) + 1))
            label = np.tensordot(np.asarray(labels), phi_big, axes=1)
            normals = normal_field_coefficients(label, lat, k1, k2, opts.normal_blur)
        layers.append(_LayerGeom(layer.thickness_nm, eps_bg=eps_bg,
                                 eps_shapes=np.stack([np.asarray(s.material.eps(wl), dtype=complex) for s in shapes]),
                                 phi=phi, normals=normals, fourier=method))
    inc = problem.incidence
    return _Prepared(lat, harm, k1, k2, toeplitz_index(harm, k1, k2), wl, eps_sup, eps_sub, layers,
                     np.radians(np.asarray([inc.theta_deg])), float(np.radians(inc.phi_deg)),
                     list(inc.polarizations), fact, opts.normal_blur,
                     info={"fourier": sorted({l.fourier for l in layers if l.phi is not None})})


def _batch_size(n_harm: int, n_points: int, single: bool, max_bytes: float) -> int:
    per = 48 * (2 * n_harm) ** 2 * (8 if single else 16)
    return int(max(1, min(n_points, max_bytes // per)))


def _layer_matrices(xp, prep: _Prepared, coef_eps, coef_inv, normals, unit, dev):
    """Factorised Toeplitz matrices from eps and 1/eps coefficient arrays (B, 2K1+1, 2K2+1)."""
    b = coef_eps.shape[0]
    n = prep.harm.shape[0]
    idx = xp.asarray(prep.tidx.ravel(), device=dev)
    el = xp.reshape(xp.take(xp.reshape(coef_eps, (b, -1)), idx, axis=-1), (b, n, n))
    ezz_inv = xp.linalg.inv(el)
    if prep.factorization == "laurent":
        zero = xp.zeros_like(el)
        return PatternedMatrices(el, zero, zero, el, ezz_inv)
    einv = xp.linalg.inv(xp.reshape(xp.take(xp.reshape(coef_inv, (b, -1)), idx, axis=-1), (b, n, n)))
    delta = einv - el
    if normals is None:  # 1D: constant unit vector along the grating vector -> Li's rules exactly
        ux, uy = unit
        return PatternedMatrices(el + delta * (ux * ux), delta * (ux * uy), delta * (ux * uy),
                                 el + delta * (uy * uy), ezz_inv)
    nxx, nxy, nyy = normals
    dxy = delta @ nxy
    return PatternedMatrices(el + delta @ nxx, dxy, dxy, el + delta @ nyy, ezz_inv)


def rcwa_tensors(problem: PeriodicProblem | LayerStackProblem, device: Device | str = "cpu", *,
                 thicknesses=None, eps_grids: dict[int, Any] | None = None,
                 grid_normals: dict[int, Any] | None = None,
                 options: dict[str, Any] | RCWAOptions | None = None) -> dict[str, Any]:
    """Differentiable core of the RCWA backend.

    thicknesses: per-layer thicknesses (nm) overriding the problem (floats or 0-d tensors).
    eps_grids: {layer index: permittivity map} replacing that layer's shapes. A 2D map has
    shape (Nu, Nv) or (n_wavelengths, Nu, Nv); pixel (i, j) is centred at
    r = u a1 + v a2 with u = (i + 1/2)/Nu - 1/2, v = (j + 1/2)/Nv - 1/2. 1D maps are (Nu,) or
    (n_wavelengths, Nu) along a1. Pixels are transformed exactly (piecewise constant).
    grid_normals: {layer index: map} from which the 2D normal-vector field of a gridded layer
    is built (default: the eps map itself, detached). The field is a factorization aid and is
    held fixed under differentiation; pass a fixed reference (e.g. the binarised design) to
    make gradients consistent with finite differences.
    options: RCWA options (default: the problem's ``options["rcwa"]``).

    Returns arrays of shape (n_angles, n_wavelengths) (n_angles = 1 for periodic problems):
    {"R"|"T"|"A"|"r"|"t": {pol: array}, "R_orders"|"T_orders": {pol: (n_angles, n_wl, N)},
    "harmonics": (N, 2) int array, "info": {...}}.
    """
    dev = resolve_device(device) if isinstance(device, str) else device
    if options is None:
        opts = RCWAOptions.model_validate(options_for(problem, NativeRCWA.name, NativeRCWA.method))
    else:
        opts = options if isinstance(options, RCWAOptions) else RCWAOptions.model_validate(options)
    prep = _prepare(problem, opts)
    xp = dev.namespace()
    nd = dev.native_device()
    single = opts.precision == "single"
    cdt = xp.complex64 if single else xp.complex128
    rdt = xp.float32 if single else xp.float64
    tol = 1e-5 if single else 1e-10
    lat, harm, wl = prep.lattice, prep.harm, prep.wl
    n_h, n_a, n_w = harm.shape[0], len(prep.theta), len(wl)
    z0 = int(np.nonzero((harm[:, 0] == 0) & (harm[:, 1] == 0))[0][0])

    # Flattened (angle, wavelength) points.
    th = np.repeat(prep.theta, n_w)
    lam = np.tile(wl, n_a)
    widx = np.tile(np.arange(n_w), n_a)
    k0 = 2 * np.pi / lam
    n_sup = np.sqrt(prep.eps_sup[widx])
    if lat is None:
        gx = gy = np.zeros(1)
    else:
        gx, gy = lat.g(harm[:, 0], harm[:, 1])
    kx = (n_sup * np.sin(th) * np.cos(prep.phi))[:, None] + gx[None, :] / k0[:, None]
    ky = (n_sup * np.sin(th) * np.sin(prep.phi))[:, None] + gy[None, :] / k0[:, None]
    s_hat = np.array([-np.sin(prep.phi), np.cos(prep.phi)])
    p_tr = np.stack([np.cos(th) * np.cos(prep.phi), np.cos(th) * np.sin(prep.phi)], axis=-1)  # (P, 2)

    def thick(i):
        if thicknesses is None:
            return prep.layers[i].thickness_nm
        d = thicknesses[i]
        return d if is_tensor(d) else float(d)

    unit = (lat.a1 / np.linalg.norm(lat.a1)) if lat is not None else (1.0, 0.0)
    normals_dev = {}
    grids = {}
    if eps_grids and lat is None:
        raise ValueError("eps_grids need a periodic problem")
    for i, g in (eps_grids or {}).items():
        grid = xp.astype(g, cdt) if is_tensor(g) else dev.asarray(np.asarray(g, dtype=complex), dtype=cdt)
        if lat.dim == 1:
            grid = grid[..., None]
        if grid.ndim == 2:
            grid = grid[None]
        grids[i] = grid
        if lat.dim == 2 and prep.factorization == "li":
            ref = (grid_normals or {}).get(i)
            lab = Device.to_numpy(grid[0]) if ref is None else np.asarray(Device.to_numpy(ref))
            lab = lab.reshape(grid.shape[-2:]).astype(complex)
            l1, l2 = max(prep.k1 + 1, 32), max(prep.k2 + 1, 32)
            normals_dev[i] = normal_field_coefficients(pixel_coefficients(np, lab, l1, l2), lat, prep.k1, prep.k2,
                                                       prep.blur)
    tidx = prep.tidx

    def toe_normals(coef):
        idx = xp.asarray(tidx.ravel(), device=nd)
        c = dev.asarray(coef, dtype=cdt)
        return [xp.reshape(xp.take(xp.reshape(c[j], (-1,)), idx, axis=0), (n_h, n_h)) for j in range(3)]

    normals_mats = {i: toe_normals(l.normals) for i, l in enumerate(prep.layers) if l.normals is not None}
    normals_mats.update({i: toe_normals(c) for i, c in normals_dev.items()})

    bs = _batch_size(n_h, len(lam), single, opts.max_batch_bytes)
    outs = []
    for s0 in range(0, len(lam), bs):
        sl = slice(s0, s0 + bs)
        b = len(lam[sl])
        w_sl = widx[sl]
        kx_d, ky_d = dev.asarray(kx[sl], dtype=cdt), dev.asarray(ky[sl], dtype=cdt)
        k0_d = dev.asarray(k0[sl], dtype=rdt)[:, None]
        core = Core(xp, kx_d, ky_d, cdt, tol)
        S = core.s_ref(dev.asarray(prep.eps_sup[w_sl], dtype=cdt))
        for i, lg in enumerate(prep.layers):
            d_k0 = thick(i) * k0_d
            if i in grids:
                g = grids[i]
                g = g if g.shape[0] == 1 else g[xp.asarray(w_sl, device=nd)]
                ce = pixel_coefficients(xp, g, prep.k1, prep.k2, device=nd)
                ci = pixel_coefficients(xp, 1 / g, prep.k1, prep.k2, device=nd)
                if ce.shape[0] == 1 and b > 1:
                    ce, ci = xp.broadcast_to(ce, (b,) + ce.shape[1:]), xp.broadcast_to(ci, (b,) + ci.shape[1:])
                Si = core.s_patterned(_layer_matrices(xp, prep, ce, ci, normals_mats.get(i), unit, nd), d_k0)
            elif lg.uniform_eps is not None:
                Si = core.s_uniform(dev.asarray(lg.uniform_eps[w_sl], dtype=cdt), d_k0)
            else:
                e_bg, e_s = lg.eps_bg[w_sl], lg.eps_shapes[:, w_sl]
                kk = (lg.phi.shape[1] // 2, lg.phi.shape[2] // 2)
                delta0 = np.zeros(lg.phi.shape[1:], dtype=complex)
                delta0[kk] = 1.0
                ce = e_bg[:, None, None] * delta0 + np.einsum("sw,smn->wmn", e_s - e_bg, lg.phi)
                ci = (1 / e_bg)[:, None, None] * delta0 + np.einsum("sw,smn->wmn", 1 / e_s - 1 / e_bg[None], lg.phi)
                mats = _layer_matrices(xp, prep, dev.asarray(ce, dtype=cdt), dev.asarray(ci, dtype=cdt),
                                       normals_mats.get(i), unit, nd)
                Si = core.s_patterned(mats, d_k0)
            S = core.star(S, Si)
        S = core.star(S, core.s_trn(dev.asarray(prep.eps_sub[w_sl], dtype=cdt)))

        # Incident transverse E on the zeroth harmonic, one column per polarization.
        src = np.zeros((b, 2 * n_h, len(prep.pols)), dtype=complex)
        for j, pol in enumerate(prep.pols):
            vec = np.broadcast_to(s_hat, (b, 2)) if pol == "s" else p_tr[sl]
            src[:, z0, j], src[:, n_h + z0, j] = vec[:, 0], vec[:, 1]
        src = dev.asarray(src, dtype=cdt)
        v_ref, _ = core.v_homogeneous(dev.asarray(prep.eps_sup[w_sl], dtype=cdt))
        v_trn, _ = core.v_homogeneous(dev.asarray(prep.eps_sub[w_sl], dtype=cdt))
        c_ref, c_trn = S[0] @ src, S[2] @ src
        h_ref, h_trn = -(v_ref @ c_ref), v_trn @ c_trn
        p_inc = xp.sum(core.flux(src, v_ref @ src), axis=-2)  # (B, P)
        r_ord = -core.flux(c_ref, h_ref) / p_inc[:, None, :]
        t_ord = core.flux(c_trn, h_trn) / p_inc[:, None, :]
        s_d = dev.asarray(s_hat, dtype=cdt)
        n_sup_d = xp.sqrt(dev.asarray(prep.eps_sup[w_sl], dtype=cdt))
        n_sub_d = xp.sqrt(dev.asarray(prep.eps_sub[w_sl], dtype=cdt))

        def proj(f):  # (B, P): tangential component along s_hat of the zeroth harmonic
            return f[:, z0, :] * s_d[0] + f[:, n_h + z0, :] * s_d[1]

        amp_r, amp_t = [], []
        for j, pol in enumerate(prep.pols):
            if pol == "s":
                amp_r.append(proj(c_ref)[:, j])
                amp_t.append(proj(c_trn)[:, j])
            else:
                amp_r.append(proj(h_ref)[:, j] / n_sup_d)
                amp_t.append(proj(h_trn)[:, j] / n_sub_d)
        outs.append((r_ord, t_ord, xp.stack(amp_r, axis=-1), xp.stack(amp_t, axis=-1)))

    r_ord = xp.concat([o[0] for o in outs], axis=0)
    t_ord = xp.concat([o[1] for o in outs], axis=0)
    amp_r = xp.concat([o[2] for o in outs], axis=0)
    amp_t = xp.concat([o[3] for o in outs], axis=0)
    res: dict[str, Any] = {k: {} for k in ("R", "T", "A", "r", "t", "R_orders", "T_orders")}
    for j, pol in enumerate(prep.pols):
        ro = xp.reshape(r_ord[:, :, j], (n_a, n_w, n_h))
        to = xp.reshape(t_ord[:, :, j], (n_a, n_w, n_h))
        res["R_orders"][pol], res["T_orders"][pol] = ro, to
        res["R"][pol], res["T"][pol] = xp.sum(ro, axis=-1), xp.sum(to, axis=-1)
        res["A"][pol] = 1 - res["R"][pol] - res["T"][pol]
        res["r"][pol] = xp.reshape(amp_r[:, j], (n_a, n_w))
        res["t"][pol] = xp.reshape(amp_t[:, j], (n_a, n_w))
    res["harmonics"] = harm
    kpar2 = np.abs(kx) ** 2 + np.abs(ky) ** 2  # (A*W, N)
    prop = (kpar2 < np.real(prep.eps_sup[widx])[:, None]) | (kpar2 < np.real(prep.eps_sub[widx])[:, None])
    res["propagating"] = prop.reshape(n_a, n_w, n_h).any(axis=(0, 1))
    res["info"] = {"harmonics": n_h, "orders_max": [int(harm[:, 0].max()), int(harm[:, 1].max())],
                   "matrix_size": 2 * n_h, "factorization": prep.factorization, "batch_size": bs,
                   "precision": "single" if single else "double", **prep.info}
    return res


def _tolist(a) -> list:
    return Device.to_numpy(a).astype(float).tolist()


def _amp(a) -> list:
    v = Device.to_numpy(a)
    return np.stack([v.real, v.imag], axis=-1).tolist()


@register
class NativeRCWA(Backend):
    name = "lightsmith-rcwa"
    method = "rcwa"
    license = "MIT"
    citation = ("Moharam et al., JOSA A 12, 1068 (1995); Li, JOSA A 13, 1870 (1996) and JOSA A 14, 2758 (1997); "
                "Schuster et al., JOSA A 24, 2880 (2007); Rumpf, PIER B 35, 241 (2011)")
    gpu = True
    differentiable = True
    problem_kinds = ("periodic", "layer_stack")

    @classmethod
    def supports(cls, problem):
        if isinstance(problem, LayerStackProblem):
            if any(not l.coherent for l in problem.layers):
                return False, "RCWA treats every layer coherently; use TMM for incoherent (thick) layers"
            return True, ""
        if isinstance(problem, PeriodicProblem):
            return True, ""
        return False, "RCWA solves periodic and layer_stack problems"

    @classmethod
    def _options(cls, problem) -> RCWAOptions:
        return RCWAOptions.model_validate(options_for(problem, cls.name, cls.method))

    @classmethod
    def estimate(cls, problem, device):
        opts = cls._options(problem)
        m, n = _orders(problem, opts)
        is_1d = isinstance(problem, LayerStackProblem) or problem.lattice_a2_nm is None
        n_h = (2 * m + 1) * (1 if is_1d else 2 * n + 1)
        single = opts.precision == "single"
        n_w = len(wavelength_array(problem.wavelengths))
        n_pts = n_w * (len(problem.angles_deg) if isinstance(problem, LayerStackProblem) else 1)
        patterned = 0 if isinstance(problem, LayerStackProblem) else sum(bool(l.shapes) for l in problem.layers)
        bs = _batch_size(n_h, n_pts, single, opts.max_batch_bytes)
        return {"harmonics": n_h, "orders_max": [m, n], "matrix_size": 2 * n_h,
                "precision": "single" if single else "double", "batch_size": bs,
                "eigensolves": patterned * n_pts,
                "memory_bytes": int(48 * (2 * n_h) ** 2 * (8 if single else 16) * bs),
                "flops": float(30 * (2 * n_h) ** 3 * n_pts * (len(problem.layers) + 2))}

    @classmethod
    def solve(cls, problem, device: Device) -> SpectrumResult:
        t0 = time.perf_counter()
        opts = cls._options(problem)
        out = rcwa_tensors(problem, device, options=opts)
        wl = wavelength_array(problem.wavelengths)
        periodic = isinstance(problem, PeriodicProblem)
        angles = [problem.incidence.theta_deg] if periodic else list(problem.angles_deg)
        pols = list(problem.incidence.polarizations) if periodic else list(problem.polarizations)
        warnings = []
        if np.any(np.imag(problem.superstrate.eps(wl)) > 1e-12):
            warnings.append("absorbing superstrate: efficiencies are normalised to the incident Poynting flux")
        orders = None
        if periodic:
            harm = out["harmonics"]
            orders = [DiffractionOrder(order=(int(harm[h, 0]), int(harm[h, 1])),
                                       r={p: _tolist(out["R_orders"][p][..., h]) for p in pols},
                                       t={p: _tolist(out["T_orders"][p][..., h]) for p in pols})
                      for h in np.nonzero(out["propagating"])[0]]
            a_min = min(float(Device.to_numpy(out["A"][p]).min()) for p in pols)
            lossless = all(np.all(np.abs(np.imag(m.eps(wl))) < 1e-12) for m in _materials(problem))
            if lossless and abs(a_min) > 1e-3:
                warnings.append(f"energy balance off by {abs(a_min):.1e} for a lossless structure; "
                                "increase harmonics or use precision='double'")
        return SpectrumResult(
            method=cls.method, backend=cls.name, wavelengths_nm=wl.tolist(), angles_deg=angles, polarizations=pols,
            r={p: _tolist(out["R"][p]) for p in pols}, t={p: _tolist(out["T"][p]) for p in pols},
            a={p: _tolist(out["A"][p]) for p in pols}, orders=orders,
            r_amplitude={p: _amp(out["r"][p]) for p in pols}, t_amplitude={p: _amp(out["t"][p]) for p in pols},
            provenance={**device.describe(), **out["info"], "runtime_s": time.perf_counter() - t0,
                        "amplitude_convention": "zeroth order; s: E ratio; p: r = tangential H ratio, t = E ratio"},
            warnings=warnings)


def _materials(problem: PeriodicProblem):
    yield problem.superstrate
    yield problem.substrate
    for l in problem.layers:
        yield l.background
        for s in l.shapes:
            yield s.material


__all__ = ["NativeRCWA", "RCWAOptions", "rcwa_tensors"]
