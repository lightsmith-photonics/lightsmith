"""Native Lorenz-Mie theory for homogeneous and multilayer spheres, CPU or GPU.

Written once against the Python array API, so it runs on NumPy (CPU) or
PyTorch (CPU/CUDA) and, with PyTorch, is differentiable end to end: see
:func:`mie_cross_sections` for gradients of cross sections with respect to
radii or permittivities.

Algorithm (Yang, Appl. Opt. 42, 1710 (2003); Pena & Pal, Comput. Phys. Commun.
180, 2348 (2009)), vectorised over wavelengths:

- logarithmic derivatives D1_n = psi_n'/psi_n by downward recursion, started
  at max(n_max, |z|) + 10 |z|^(1/3) + 16 (Wiscombe's +16 alone, Appl. Opt. 19,
  1505 (1980), leaves 1e-3 errors at |z| ~ 250);
- D3_n = xi_n'/xi_n and the ratios psi_n/xi_n of the inner layers by Yang's
  upward recursions, so no Riccati-Bessel function is ever formed and nothing
  overflows, even in single precision;
- layer-to-layer recursion of the effective log derivatives Ha, Hb, then, at
  the (real) outer size parameter x, with h = Ha_n/m for a_n and m Hb_n for b_n,

      a_n = u / (u - i),  u = (psi_n/chi_n)(x) (D1_n(x) - h) / (Dchi_n(x) - h).

  This equals Yang's (psi/xi)(D1 - h)/(D3 - h) but keeps Re(a_n) accurate when
  it is tiny (small, weakly absorbing spheres) instead of losing it to
  cancellation; psi/chi and Dchi come from real upward recursions.

Bohren & Huffman conventions throughout: exp(-i omega t), xi_n = psi_n - i chi_n
(outgoing), Im(m) >= 0 for loss. Wiscombe's n_max = x + 4.05 x^(1/3) + 2 for a
homogeneous sphere; Pena & Pal's max(that, |m_l x_l|, |m_l x_(l-1)|) + 15 for
layered spheres.

Near fields: the field in every region is expanded in psi_n and xi_n of that
region's wavenumber; the expansion coefficients are found by matching tangential
E and H at each interface, marching inward from the known scattered field. That
step forms Riccati-Bessel functions explicitly, so it always runs in double
precision.
"""

from __future__ import annotations

import math
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..device import Device
from ..results import FarFieldData, NearFieldData, Result
from ..spec import LayeredSphere, Simulation, Sphere
from .base import Backend, register, unsupported_outputs

_BYTES_BUDGET = 1 << 30  # per wavelength chunk, for the (wavelength x order) work arrays
_N_WORK = 8  # (W, n_max) complex arrays alive at once in _coefficients


class MieOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precision: Literal["single", "double"] = Field(
        "double", description="Floating-point precision of the series. 'single' is typically ~1e-7 relative but "
        "degrades to ~1e-3 near zeros of psi_n(x); the series is memory-bound, so on GPUs it is no faster than "
        "double. Near fields always use double.")
    n_max: int | None = Field(None, ge=1, description="Number of multipole orders. Default: Wiscombe's rule "
                              "(homogeneous) or Pena & Pal's (layered), per wavelength chunk.")
    wavelength_chunk: int | None = Field(None, ge=1, description="Wavelengths per vectorised batch. "
                                         "Default: as many as fit in ~1 GiB of work arrays.")


# --- the series ---------------------------------------------------------------


def n_max_wiscombe(x: float) -> int:
    return int(math.ceil(x + 4.05 * x ** (1 / 3) + 2))


def _n_max(x_np: list[np.ndarray], m_np: list[np.ndarray]) -> int:
    """Orders needed for size parameters x_np (per layer) and relative indices m_np."""
    n = n_max_wiscombe(float(np.max(x_np[-1])))
    if len(x_np) == 1:
        return n
    zs = [np.abs(m_np[0] * x_np[0])]
    for l in range(1, len(x_np)):
        zs += [np.abs(m_np[l] * x_np[l]), np.abs(m_np[l] * x_np[l - 1])]
    return int(math.ceil(max(n, max(float(np.max(z)) for z in zs)))) + 15


def _n_start(z_abs_max: float, n_max: int) -> int:
    # Wiscombe's max(n_max, |z|) + 16 is not enough for large real |z| (D1 errors ~1e-3 at |z| = 250);
    # the transition region n ~ |z| is ~|z|^(1/3) wide, so add a margin that grows with it.
    return int(math.ceil(max(n_max, z_abs_max) + 10 * z_abs_max ** (1 / 3))) + 16


def _d1_down(z, n_max: int, n_start: int, xp) -> list:
    """D1_n(z) = psi_n'(z)/psi_n(z) for n = 0..n_max by downward recursion."""
    d = xp.zeros_like(z)
    out = [None] * (n_max + 1)
    for n in range(n_start, 0, -1):
        if n <= n_max:
            out[n] = d
        nz = n / z
        d = nz - 1 / (d + nz)
    out[0] = d
    return out


def _d3_up(z, d1: list, xp) -> list:
    """D3_n(z) = xi_n'(z)/xi_n(z) by Yang's upward recursion (uses psi_n xi_n)."""
    psixi = 0.5 * (1 - xp.exp(2j * z))
    d3 = [xp.zeros_like(z) + 1j]
    for n in range(1, len(d1)):
        nz = n / z
        psixi = psixi * (nz - d1[n - 1]) * (nz - d3[n - 1])
        d3.append(d1[n] + 1j / psixi)
    return d3


def _coefficients(x: list, m: list, n_max: int, n_start: int, xp):
    """Scattering coefficients a_n, b_n, n = 1..n_max, shape (..., n_max).

    x: size parameters k r_l of each layer's outer radius (core first), complex arrays.
    m: relative refractive index of each layer, Im(m) >= 0.
    """
    # core: Ha = Hb = D1(m_1 x_1)
    ha = hb = _d1_down(m[0] * x[0], n_max, n_start, xp)
    for l in range(1, len(x)):
        z1, z2 = m[l] * x[l - 1], m[l] * x[l]
        d1a, d1b = _d1_down(z1, n_max, n_start, xp), _d1_down(z2, n_max, n_start, xp)
        d3a, d3b = _d3_up(z1, d1a, xp), _d3_up(z2, d1b, xp)
        # Q_n = (psi_n/xi_n)(z1) / (psi_n/xi_n)(z2), in a form that is stable for Im(z) >= 0
        q = xp.exp(2j * (z2 - z1)) * (xp.exp(2j * z1) - 1) / (xp.exp(2j * z2) - 1)
        rel = m[l] / m[l - 1]
        new_a, new_b = [ha[0]], [hb[0]]
        for n in range(1, n_max + 1):
            q = q * ((d3a[n] + n / z1) / (d1a[n] + n / z1)) / ((d3b[n] + n / z2) / (d1b[n] + n / z2))
            for h, new in ((rel * ha[n], new_a), (hb[n] / rel, new_b)):
                g1, g2 = d1a[n] - h, d3a[n] - h
                qg1 = q * g1
                new.append((g2 * d1b[n] - qg1 * d3b[n]) / (g2 - qg1))
        ha, hb = new_a, new_b
    # Outside (real x): a = P / (P - iQ) with P = psi' - h psi, Q = chi' - h chi, written as
    # a = u / (u - i), u = P/Q = (psi/chi)(D1 - h)/(Dchi - h). Unlike (psi/xi)(...)/(...), this keeps
    # Re(a_n) accurate when it is tiny (small, weakly absorbing spheres), and psi/chi underflows harmlessly.
    xo, mo = x[-1], m[-1]
    d1x = _d1_down(xo, n_max, n_start, xp)
    t = xp.sin(xo) / xp.cos(xo)  # psi_0/chi_0, chi_0 = cos x
    dchi = -t  # chi_0'/chi_0
    a, b = [], []
    for n in range(1, n_max + 1):
        nx = n / xo
        t = t / ((d1x[n] + nx) * (nx - dchi))
        dchi = 1 / (nx - dchi) - nx
        for h, out in ((ha[n] / mo, a), (mo * hb[n], b)):
            u = t * (d1x[n] - h) / (dchi - h)
            out.append(u / (u - 1j))
    return xp.stack(a, axis=-1), xp.stack(b, axis=-1)


def _orders(n_max: int, xp, dtype, device):
    return xp.asarray(np.arange(1, n_max + 1, dtype=float), dtype=dtype, device=device)


def _cross_sections(a, b, k, xp):
    """C_ext, C_sca from coefficients (..., N) and medium wavenumber k (...)."""
    n = _orders(a.shape[-1], xp, xp.real(a).dtype, _dev(a))
    w = 2 * n + 1
    pref = 2 * math.pi / k**2
    c_ext = pref * xp.sum(w * xp.real(a + b), axis=-1)
    c_sca = pref * xp.sum(w * (xp.real(a * xp.conj(a)) + xp.real(b * xp.conj(b))), axis=-1)
    return c_ext, c_sca


def _dev(a):
    from array_api_compat import device

    return device(a)


def _angular(mu: np.ndarray, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """pi_n(mu), tau_n(mu) for n = 1..n_max, shape mu.shape + (n_max,)."""
    pi = np.zeros(mu.shape + (n_max + 1,))
    tau = np.zeros_like(pi)
    if n_max >= 1:
        pi[..., 1] = 1.0
        tau[..., 1] = mu
    for n in range(2, n_max + 1):
        pi[..., n] = (2 * n - 1) / (n - 1) * mu * pi[..., n - 1] - n / (n - 1) * pi[..., n - 2]
        tau[..., n] = n * mu * pi[..., n] - (n + 1) * pi[..., n - 1]
    return pi[..., 1:], tau[..., 1:]


# --- differentiable functional API ---------------------------------------------


def _namespace(*vals):
    import array_api_compat

    arrs = [v for v in vals if not isinstance(v, (int, float, complex, list, tuple))]
    if any(not isinstance(v, np.ndarray) for v in arrs):  # NumPy inputs mix freely with e.g. torch tensors
        arrs = [v for v in arrs if not isinstance(v, np.ndarray)]
    if not arrs:
        import array_api_compat.numpy as xp

        return xp, "cpu", False
    xp = array_api_compat.array_namespace(*arrs)
    single = all(str(a.dtype).split(".")[-1] in ("float32", "complex64") for a in arrs)
    return xp, array_api_compat.device(arrs[0]), single


def mie_cross_sections(radii_nm, eps, wavelength_nm, medium_index: float = 1.0, *, n_max: int | None = None):
    """Differentiable C_ext, C_sca, C_abs (nm^2) of a homogeneous or multilayer sphere.

    radii_nm: outer radius of each region, core first (a scalar for a plain sphere).
    eps: permittivity of each region (scalar or per-wavelength), exp(-i omega t) so Im >= 0 for loss.
    wavelength_nm: vacuum wavelengths, shape (W,).

    Inputs may be NumPy arrays, Python scalars or PyTorch tensors (CPU or CUDA);
    with tensors that require grad, the outputs are differentiable, e.g.::

        r = torch.tensor(40.0, dtype=torch.float64, requires_grad=True)
        c_ext, _, _ = mie_cross_sections(r, -8 + 1j, torch.tensor([550.0], dtype=torch.float64))
        c_ext.sum().backward()   # r.grad = d C_ext / d r
    """
    if not isinstance(radii_nm, (list, tuple)):
        radii_nm = [radii_nm]
    if not isinstance(eps, (list, tuple)):
        eps = [eps]
    if len(radii_nm) != len(eps):
        raise ValueError("need one permittivity per radius")
    import array_api_compat

    xp, dev, single = _namespace(*radii_nm, *eps, wavelength_nm)
    rdt, cdt = (xp.float32, xp.complex64) if single else (xp.float64, xp.complex128)

    def arr(v, dt):
        try:
            native = array_api_compat.array_namespace(v) is xp
        except TypeError:
            native = False
        return xp.astype(v if native else xp.asarray(np.asarray(v), device=dev), dt)

    lam = arr(wavelength_nm, rdt)
    k = 2 * math.pi * medium_index / lam
    x = [xp.astype(k * arr(r, rdt), cdt) for r in radii_nm]
    m = [xp.sqrt(arr(e, cdt)) / medium_index for e in eps]
    m = [xp.where(xp.imag(mm) < 0, -mm, mm) for mm in m]
    x_np = [np.atleast_1d(np.real(Device.to_numpy(v))) for v in x]
    m_np = [np.atleast_1d(Device.to_numpy(v)) for v in m]
    nm = n_max or _n_max(x_np, m_np)
    zmax = max(float(np.max(np.abs(mn * xn))) for mn, xn in zip(m_np, x_np))
    a, b = _coefficients(x, m, nm, _n_start(max(zmax, float(np.max(x_np[-1]))), nm), xp)
    c_ext, c_sca = _cross_sections(a, b, k, xp)
    return c_ext, c_sca, c_ext - c_sca


# --- near field ------------------------------------------------------------------


def _riccati(z, n_max: int, xp, regular_only: bool = False):
    """psi_n, psi_n', xi_n, xi_n' (n = 1..n_max) at complex z, each shape z.shape + (n_max,).

    With regular_only, xi is skipped (returned as None): it is singular at z = 0."""
    zmax = float(np.max(np.abs(Device.to_numpy(z)))) if z.shape and math.prod(z.shape) else 0.0
    d1 = _d1_down(z, n_max, _n_start(zmax, n_max), xp)
    d3 = None if regular_only else _d3_up(z, d1, xp)
    psi, xi = xp.sin(z), -1j * xp.exp(1j * z)
    ps, dps, xs, dxs = [], [], [], []
    for n in range(1, n_max + 1):
        nz = n / z
        psi = psi / (d1[n] + nz)
        ps.append(psi)
        dps.append(d1[n] * psi)
        if d3 is not None:
            xi = xi / (d3[n] + nz)
            xs.append(xi)
            dxs.append(d3[n] * xi)
    stack = lambda v: xp.stack(v, axis=-1) if v else None  # noqa: E731
    return tuple(stack(v) for v in (ps, dps, xs, dxs))


def _region_coefficients(a, b, rho_r, m_regions, xp):
    """Expansion coefficients (TE: A, B; TM: C, D) of psi and xi in every region.

    a, b: (N,) scattering coefficients; rho_r: k * outer radius per layer;
    m_regions: relative index per layer. Returns a list (core first, then
    shells, then outside) of (A, B, C, D), each (N,).
    """
    one = xp.ones_like(a)
    out = [(one, -b, one, -a)]  # outside: incident psi + scattered xi
    m_out = 1.0 + 0j
    for j in range(len(rho_r) - 1, -1, -1):
        m_in = m_regions[j]
        A, B, C, D = out[0]
        po, dpo, xo, dxo = _riccati(xp.asarray(m_out * rho_r[j], dtype=xp.complex128, device=_dev(a))[None],
                                    a.shape[-1], xp)
        pi_, dpi, xi_, dxi = _riccati(xp.asarray(m_in * rho_r[j], dtype=xp.complex128, device=_dev(a))[None],
                                      a.shape[-1], xp)
        te_r, te_d = A * po[0] + B * xo[0], A * dpo[0] + B * dxo[0]
        tm_r, tm_d = C * po[0] + D * xo[0], C * dpo[0] + D * dxo[0]
        rel = m_in / m_out
        # solve R(zi) = F, R'(zi) = G for R = alpha psi + beta xi (Wronskian psi xi' - psi' xi = i)
        solve = lambda f, g: ((f * dxi[0] - g * xi_[0]) / 1j, (g * pi_[0] - f * dpi[0]) / 1j)  # noqa: E731
        A2, B2 = solve(rel * te_r, te_d)
        C2, D2 = solve(tm_r, rel * tm_d)
        if j == 0:  # the core holds no irregular (xi) part
            B2, D2 = xp.zeros_like(B2), xp.zeros_like(D2)
        out.insert(0, (A2, B2, C2, D2))
        m_out = m_in
    return out


def _vsh_field(coef, z, pi, tau, cos_t, sin_t, cos_p, sin_p, xp, regular_only=False, irregular_only=False):
    """Local Cartesian E (3, P) of sum_n E_n [M_o1n - i N_e1n] built from R^M = A psi + B xi, R^N = C psi + D xi.

    regular_only: B = D = 0 (the core); irregular_only: A = C = 0 (the scattered field)."""
    A, B, C, D = coef
    n_max = A.shape[-1]
    psi, dpsi, xi, dxi = _riccati(z, n_max, xp, regular_only)
    n = np.arange(1, n_max + 1)
    en = xp.asarray((1j ** n) * (2 * n + 1) / (n * (n + 1)), dtype=xp.complex128, device=_dev(A))
    nn1 = xp.asarray((n * (n + 1)).astype(float), dtype=xp.complex128, device=_dev(A))
    if regular_only:
        rm, rn, drn = A * psi, C * psi, C * dpsi
    elif irregular_only:
        rm, rn, drn = B * xi, D * xi, D * dxi
    else:
        rm, rn, drn = A * psi + B * xi, C * psi + D * xi, C * dpsi + D * dxi
    e_r = xp.sum(en * (-1j) * nn1 * pi * rn, axis=-1) * sin_t / z**2 * cos_p
    e_t = xp.sum(en * (pi * rm - 1j * tau * drn), axis=-1) / z * cos_p
    e_p = xp.sum(en * (-tau * rm + 1j * pi * drn), axis=-1) / z * sin_p
    ex = e_r * sin_t * cos_p + e_t * cos_t * cos_p - e_p * sin_p
    ey = e_r * sin_t * sin_p + e_t * cos_t * sin_p + e_p * cos_p
    ez = e_r * cos_t - e_t * sin_t
    return xp.stack([ex, ey, ez], axis=0)


def _near_field(sim: Simulation, req, device: Device, opts: MieOptions) -> tuple[list[NearFieldData], int]:
    xp = device.namespace()
    cdt = xp.complex128
    p = sim.particles[0]
    radii = _radii(p)
    mats = p.region_materials()
    khat, ehat = sim.illumination.unit_vectors()
    hhat = np.cross(khat, ehat)
    axis_of = {"x": 0, "y": 1, "z": 2}
    a1, a2 = (axis_of[c] for c in req.plane)
    normal = 3 - a1 - a2
    u = np.linspace(-req.half_width_nm, req.half_width_nm, req.points)
    g1, g2 = np.meshgrid(u, u, indexing="ij")
    pts = np.zeros(g1.shape + (3,))
    pts[..., a1], pts[..., a2], pts[..., normal] = g1, g2, req.offset_nm
    flat = pts.reshape(-1, 3)
    rel = flat - np.asarray(p.position_nm)
    region = p.region(rel)
    loc = np.stack([rel @ ehat, rel @ hhat, rel @ khat], axis=-1)
    r = np.maximum(np.linalg.norm(loc, axis=-1), 1e-9 * radii[-1])
    cos_t = np.clip(loc[:, 2] / r, -1.0, 1.0)
    sin_t = np.sqrt(1 - cos_t**2)
    phi = np.arctan2(loc[:, 1], loc[:, 0])
    out, n_used = [], 0
    for lam in req.wavelengths_nm:
        k = 2 * np.pi * sim.medium_index / lam
        m = [complex(np.sqrt(complex(mat.eps(np.array([lam]))[0]))) / sim.medium_index for mat in mats]
        m = [-v if v.imag < 0 else v for v in m]
        x_np = [np.array([k * rr]) for rr in radii]
        m_np = [np.array([v]) for v in m]
        nm = opts.n_max or (_n_max(x_np, m_np) + (15 if len(radii) == 1 else 0))
        n_used = max(n_used, nm)
        x = [device.asarray(v.astype(complex), dtype=cdt) for v in x_np]
        md = [device.asarray(v, dtype=cdt) for v in m_np]
        zmax = max(abs(mm * xx[0]) for mm, xx in zip(m, x_np))
        a, b = _coefficients(x, md, nm, _n_start(max(zmax, x_np[-1][0]), nm), xp)
        coefs = _region_coefficients(a[0], b[0], [k * rr for rr in radii], m, xp)
        pi_np, tau_np = _angular(cos_t, nm)
        e_loc = np.zeros((3, len(flat)), dtype=complex)
        regions = list(range(len(radii))) + [-1]
        for j, reg in enumerate(regions):
            sel = region == reg
            if not sel.any():
                continue
            mj = m[reg] if reg >= 0 else 1.0
            z = device.asarray((mj * k * r[sel]).astype(complex), dtype=cdt)
            asd = lambda v: device.asarray(v, dtype=cdt)  # noqa: E731
            # outside: scattered series only (the incident wave is added in closed form below)
            e = _vsh_field(coefs[j], z, asd(pi_np[sel]), asd(tau_np[sel]), asd(cos_t[sel]), asd(sin_t[sel]),
                           asd(np.cos(phi[sel])), asd(np.sin(phi[sel])), xp,
                           regular_only=reg == 0, irregular_only=reg < 0)
            e = Device.to_numpy(e)
            if reg < 0:
                e[0] += np.exp(1j * k * loc[sel, 2])  # incident plane wave, polarised along local x
            e_loc[:, sel] = e
        e_glob = (e_loc[0][:, None] * ehat + e_loc[1][:, None] * hhat + e_loc[2][:, None] * khat)
        e_glob *= np.exp(1j * k * (np.asarray(p.position_nm) @ khat))
        enh = np.sum(np.abs(e_glob) ** 2, axis=-1)
        shape = g1.shape
        out.append(NearFieldData(
            wavelength_nm=lam, plane=req.plane, offset_nm=req.offset_nm, axis1_nm=u.tolist(), axis2_nm=u.tolist(),
            enhancement=enh.reshape(shape).tolist(),
            e_real=np.moveaxis(e_glob.real.reshape(shape + (3,)), -1, 0).tolist(),
            e_imag=np.moveaxis(e_glob.imag.reshape(shape + (3,)), -1, 0).tolist()))
    return out, n_used


# --- backend -----------------------------------------------------------------------


def _radii(p) -> list[float]:
    return list(p.shape.radii_nm) if isinstance(p.shape, LayeredSphere) else [p.shape.radius_nm]


@register
class NativeMie(Backend):
    name = "omniscatter-mie"
    method = "mie"
    license = "MIT"
    citation = ("Bohren & Huffman (1983); Yang, Appl. Opt. 42, 1710 (2003); "
                "Pena & Pal, Comput. Phys. Commun. 180, 2348 (2009)")
    gpu = True

    @classmethod
    def supports(cls, sim: Simulation):
        if len(sim.particles) != 1:
            return False, "Mie handles exactly one particle; use 'tmatrix' or 'dda' for clusters"
        if not isinstance(sim.particles[0].shape, (Sphere, LayeredSphere)):
            return False, (f"Mie handles spheres only (plain or layered), got {sim.particles[0].shape.kind!r}; "
                           "use 'dda' or 'tmatrix'")
        why = unsupported_outputs(sim, near_field=True, far_field=True, orientation_average=True)
        return (not why), why

    @classmethod
    def _options(cls, sim) -> MieOptions:
        return MieOptions.model_validate(sim.options_for(cls.name, cls.method))

    @classmethod
    def _setup(cls, sim: Simulation, wl: np.ndarray):
        p = sim.particles[0]
        k = 2 * np.pi * sim.medium_index / wl
        x = [k * r for r in _radii(p)]
        m = [np.sqrt(np.asarray(mat.eps(wl), dtype=complex)) / sim.medium_index for mat in p.region_materials()]
        m = [np.where(v.imag < 0, -v, v) for v in m]
        return k, x, m

    @classmethod
    def _chunk(cls, opts: MieOptions, n_max: int, n_wl: int, single: bool) -> int:
        if opts.wavelength_chunk:
            return opts.wavelength_chunk
        per_wl = (8 if single else 16) * _N_WORK * (n_max + 1)
        return max(1, min(n_wl, _BYTES_BUDGET // per_wl))

    @classmethod
    def estimate(cls, sim, device):
        opts = cls._options(sim)
        wl = sim.wavelength_array()
        _, x, m = cls._setup(sim, wl)
        n_max = opts.n_max or _n_max(x, m)
        single = opts.precision == "single"
        chunk = cls._chunk(opts, n_max, len(wl), single)
        mem = (8 if single else 16) * _N_WORK * (n_max + 1) * min(chunk, len(wl))
        return {"n_max": n_max, "layers": len(x), "wavelengths": len(wl), "wavelength_chunk": chunk,
                "precision": opts.precision, "memory_bytes": int(mem), "cost": "negligible"}

    @classmethod
    def _coefficients_np(cls, sim, wl, device, opts, single):
        """Yields (slice, k, a, b) per wavelength chunk; a, b live on `device`."""
        xp = device.namespace()
        cdt = xp.complex64 if single else xp.complex128
        k, x, m = cls._setup(sim, wl)
        n_glob = opts.n_max or _n_max(x, m)
        chunk = cls._chunk(opts, n_glob, len(wl), single)
        for s in range(0, len(wl), chunk):
            sl = slice(s, s + chunk)
            xs, ms = [v[sl] for v in x], [v[sl] for v in m]
            nm = opts.n_max or _n_max(xs, ms)
            zmax = max(float(np.max(np.abs(mm * xx))) for mm, xx in zip(ms, xs))
            xd = [device.asarray(v.astype(complex), dtype=cdt) for v in xs]
            md = [device.asarray(v, dtype=cdt) for v in ms]
            a, b = _coefficients(xd, md, nm, _n_start(max(zmax, float(np.max(xs[-1]))), nm), xp)
            yield sl, k[sl], a, b

    @classmethod
    def solve(cls, sim: Simulation, device: Device) -> Result:
        t0 = time.perf_counter()
        opts = cls._options(sim)
        single = opts.precision == "single"
        xp = device.namespace()
        rdt = xp.float32 if single else xp.float64
        wl = sim.wavelength_array()
        c_ext, c_sca = np.empty(len(wl)), np.empty(len(wl))
        n_used, chunks = 0, 0
        for sl, k, a, b in cls._coefficients_np(sim, wl, device, opts, single):
            e, s = _cross_sections(a, b, device.asarray(k, dtype=rdt), xp)
            c_ext[sl], c_sca[sl] = Device.to_numpy(e), Device.to_numpy(s)
            n_used, chunks = max(n_used, a.shape[-1]), chunks + 1

        warnings: list[str] = []
        far = near = None
        if sim.outputs.far_field:
            far = cls._far_field(sim, sim.outputs.far_field, device, opts, single)
        n_near = None
        if sim.outputs.near_field:
            near, n_near = _near_field(sim, sim.outputs.near_field, device, opts)
        if not np.all(np.isfinite(c_ext)) or not np.all(np.isfinite(c_sca)):
            warnings.append("non-finite cross sections; try precision='double' or a larger n_max")
        provenance = {**device.describe(), "precision": opts.precision, "n_max": n_used, "layers": len(_radii(sim.particles[0])),
                      "wavelength_chunks": chunks, "algorithm": "Yang (2003) log-derivative recursion",
                      "runtime_s": time.perf_counter() - t0}
        if n_near is not None:
            provenance["near_field"] = {"precision": "double", "n_max": n_near}
        res = Result.from_arrays(
            method=cls.method, backend=cls.name, wavelengths_nm=wl,
            c_ext=c_ext, c_sca=c_sca, c_abs=c_ext - c_sca, r_eq=sim.equivalent_radius_nm(),
            provenance=provenance, warnings=warnings,
        )
        res.near_field, res.far_field = near, far
        return res

    @classmethod
    def _far_field(cls, sim, req, device, opts, single) -> list[FarFieldData]:
        xp = device.namespace()
        cdt = xp.complex64 if single else xp.complex128
        wl = np.asarray(req.wavelengths_nm, dtype=float)
        mu = np.cos(np.radians(np.asarray(req.theta_deg, dtype=float)))
        ph = np.radians(np.asarray(req.phi_deg, dtype=float))
        c2, s2 = np.cos(ph) ** 2, np.sin(ph) ** 2
        out = []
        for sl, k, a, b in cls._coefficients_np(sim, wl, device, opts, single):
            nm = a.shape[-1]
            n = np.arange(1, nm + 1)
            pi_np, tau_np = _angular(mu, nm)
            wn = device.asarray((2 * n + 1) / (n * (n + 1)), dtype=cdt)
            pi_t, tau_t = device.asarray(pi_np.T.copy(), dtype=cdt), device.asarray(tau_np.T.copy(), dtype=cdt)
            aw, bw = a * wn, b * wn  # (W, N)
            s1 = xp.matmul(aw, pi_t) + xp.matmul(bw, tau_t)  # (W, T)
            s2_ = xp.matmul(aw, tau_t) + xp.matmul(bw, pi_t)
            i1 = Device.to_numpy(xp.real(s1 * xp.conj(s1))).astype(float)
            i2 = Device.to_numpy(xp.real(s2_ * xp.conj(s2_))).astype(float)
            dcs = (i2[:, :, None] * c2 + i1[:, :, None] * s2) / (k**2)[:, None, None]
            for w, lam in enumerate(wl[sl]):
                out.append(FarFieldData(wavelength_nm=float(lam), theta_deg=list(req.theta_deg),
                                        phi_deg=list(req.phi_deg), dcsca_domega_nm2=dcs[w].tolist()))
        return out
