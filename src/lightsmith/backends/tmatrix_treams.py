"""T-matrix method for spheres, layered spheres and their clusters, via treams (MIT).

Each particle gets its Mie T-matrix (``treams.TMatrix.sphere``, multilayer
capable). Several particles are coupled by solving the multiple-scattering
problem in a local basis (``TMatrix.cluster(...).interaction.solve()``).
Orientation averages are analytic, from the trace formulas of the cluster's
global T-matrix (Mishchenko, JOSA A 8, 871 (1991)).

Conventions: treams uses exp(-i omega t) and Im(eps) > 0 for loss, like
lightsmith, so permittivities pass through unchanged. Lengths are in nm, so
treams' wave numbers are in 1/nm and its cross sections in nm^2. Fields use
the parity (TE/TM, i.e. M/N) basis.

Far field: with h_l(x) -> (-i)^(l+1) e^{ix}/x, the singular waves become
M_lm -> (-i)^(l+1) X_lm e^{ikr}/(kr) and N_lm -> (-i)^l Y_lm e^{ikr}/(kr), so
the scattered field is E_sca -> F(n) e^{ikr}/(kr) with

    F(n) = sum_j e^{-ik n.r_j} sum_lm [p^M_jlm (-i)^(l+1) X_lm(n) + p^N_jlm (-i)^l Y_lm(n)]

and dC_sca/dOmega = |F|^2 / k^2 for a unit-amplitude incident wave.
"""

from __future__ import annotations

import math
import sys
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..device import Device
from ..results import FarFieldData, NearFieldData, Result
from ..spec import LayeredSphere, Simulation, Sphere
from .base import Backend, register, unsupported_outputs


class TreamsOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lmax: int | None = Field(
        None, ge=1, description="Multipole order of every particle's T-matrix. Default: Wiscombe's criterion "
        "at the shortest wavelength, plus extra orders for particles with close neighbours.")
    extra_orders: int = Field(0, ge=0, description="Orders added to the automatic per-particle lmax.")
    lmax_global: int | None = Field(
        None, ge=1, description="Order of the cluster's global T-matrix, used for orientation_average. "
        "Default: Wiscombe's criterion on the circumscribing sphere, plus 2.")
    convergence_check: bool = Field(
        True, description="For clusters, re-solve the wavelength of peak extinction with lmax + 2 and warn "
        "if the cross sections move by more than convergence_tol.")
    convergence_tol: float = Field(1e-3, gt=0, description="Relative tolerance of the convergence check.")
    max_modes: int = Field(8000, ge=1, description="Refuse clusters whose T-matrix has more modes than this "
                           "(dense, so memory grows as modes^2).")


# --- treams import and a Windows / NumPy 2 work-around -----------------------


class _IntcUfunc:
    """Calls a ufunc after casting integer arrays to C int.

    treams 0.4.7's compiled ufuncs only have loops for C ``long`` ('l'). On
    Windows that is 32-bit, while NumPy >= 2 makes ``int`` 64-bit, so every
    call with treams' own int64 index arrays fails with "ufunc ... not
    supported for the input types". Indices (l, m, pol) are tiny, so the cast
    is lossless.
    """

    def __init__(self, ufunc):
        self._ufunc = ufunc
        self.__name__ = getattr(ufunc, "__name__", "ufunc")
        self.__doc__ = ufunc.__doc__

    def __call__(self, *args, **kwargs):
        return self._ufunc(*(_as_intc(a) for a in args), **kwargs)

    def __getattr__(self, name):
        return getattr(self._ufunc, name)


def _as_intc(a):
    if isinstance(a, np.ndarray) and a.dtype.kind in "iu" and a.dtype != np.intc:
        return a.astype(np.intc)
    if isinstance(a, np.integer):
        return int(a)
    return a


_SHIM_INSTALLED: bool | None = None


def _treams():
    """Import treams (lazily) and, where needed, install the integer-width shim once."""
    global _SHIM_INSTALLED
    import treams
    import treams.coeffs
    import treams.cw
    import treams.pw
    import treams.special
    import treams.sw

    if _SHIM_INSTALLED is None:
        _SHIM_INSTALLED = False
        if np.dtype(np.int_) != np.dtype("l"):
            for name, mod in list(sys.modules.items()):
                if mod is None or not (name == "treams" or name.startswith("treams.")):
                    continue
                for attr, val in list(vars(mod).items()):
                    if isinstance(val, np.ufunc):
                        ins = [t.split("->")[0] for t in val.types]
                        if any("l" in t for t in ins) and not any("q" in t for t in ins):
                            setattr(mod, attr, _IntcUfunc(val))
                            _SHIM_INSTALLED = True
    return treams


def _treams_version() -> str:
    try:
        return version("treams")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


# --- geometry and truncation -------------------------------------------------


def _wiscombe(x: float) -> int:
    """Wiscombe's (1980) number of Mie terms for size parameter x."""
    if x <= 8:
        n = x + 4 * x ** (1 / 3) + 1
    elif x < 4200:
        n = x + 4.05 * x ** (1 / 3) + 2
    else:
        n = x + 4 * x ** (1 / 3) + 2
    return max(1, math.ceil(n))


def _layers(p) -> tuple[list[float], list]:
    """Outer radii of the core and each shell, and their materials, inside out."""
    if isinstance(p.shape, LayeredSphere):
        return list(p.shape.radii_nm), p.region_materials()
    return [p.shape.radius_nm], [p.material]


class _Plan:
    """Truncation orders and centred positions for a simulation (no treams needed)."""

    def __init__(self, sim: Simulation, opts: TreamsOptions):
        self.radii = np.array([p.shape.radius_nm for p in sim.particles])
        pos = np.array([p.position_nm for p in sim.particles], dtype=float)
        lo, hi = (pos - self.radii[:, None]).min(axis=0), (pos + self.radii[:, None]).max(axis=0)
        self.centre = (lo + hi) / 2  # treams works about the bounding-box centre
        self.positions = pos - self.centre
        k_max = 2 * np.pi * sim.medium_index / sim.wavelength_array().min()
        self.extra_close = self._close_orders(pos)
        if opts.lmax is not None:
            self.lmax = [opts.lmax] * len(pos)
        else:
            self.lmax = [_wiscombe(k_max * r) + e + opts.extra_orders for r, e in zip(self.radii, self.extra_close)]
        r_circ = float(np.max(np.linalg.norm(self.positions, axis=1) + self.radii))
        self.r_circumscribing = r_circ
        self.lmax_global = opts.lmax_global or max(_wiscombe(k_max * r_circ) + 2, max(self.lmax))
        self.modes = sum(2 * l * (l + 2) for l in self.lmax)
        self.modes_global = 2 * self.lmax_global * (self.lmax_global + 2)

    def _close_orders(self, pos: np.ndarray) -> list[int]:
        """Extra orders for particles whose surface gap g to a neighbour is below their radius R.

        Near-field coupling converges geometrically in l, more slowly as the gap closes.
        Calibrated on a resonant gold-like Drude dimer (R = 20 nm), where lmax ~ 4.5 (g/R)^-0.63
        gives ~1e-4 in C_ext: lmax 7, 12, 19 at g/R = 0.5, 0.2, 0.1. Dielectrics need fewer.
        Capped at 25 extra orders; the cluster convergence check reports what remains.
        """
        out = []
        for i, r in enumerate(self.radii):
            gaps = [np.linalg.norm(pos[i] - pos[j]) - r - self.radii[j] for j in range(len(pos)) if j != i]
            s = min(gaps, default=np.inf) / r
            out.append(0 if s >= 1 else min(25, math.ceil(4.5 * max(s, 1e-6) ** -0.63 - 4.5)))
        return out


def _overlap(sim: Simulation) -> tuple[int, int] | None:
    ps = sim.particles
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            d = np.linalg.norm(np.subtract(ps[i].position_nm, ps[j].position_nm))
            if d < ps[i].shape.radius_nm + ps[j].shape.radius_nm:
                return i, j
    return None


# --- the solver ----------------------------------------------------------------


class _Solver:
    def __init__(self, sim: Simulation, plan: _Plan):
        self.treams = _treams()
        self.sim, self.plan = sim, plan
        self.khat, self.ehat = sim.illumination.unit_vectors()
        self._cache = None  # last ((wavelength, extra), tmatrix(...) result)

    def k(self, lam: float) -> float:
        return 2 * np.pi * self.sim.medium_index / lam

    def _coupling(self, basis, k: float, lmaxs: list[int]) -> np.ndarray:
        """Singular-to-regular translation matrix between different particles (zero diagonal blocks).

        Same as treams' ``Expand(basis, "singular").inv`` used by ``TMatrix.interaction``,
        but cheaper, which matters because treams' translation coefficients cost ~10 us each:
        it skips the diagonal blocks; it translates along z, where only m = m' couples, and
        rotates, C(r) = R(phi, theta) C(|r| z) R^H (O(L^4) coefficients instead of O(L^5));
        and for equal orders it gets block (j, i) from block (i, j) via
        C(-r) = (-1)^(l+l') (+-1)_(pol == pol') C(r) in the parity basis.
        """
        import warnings

        tr = self.treams
        pidx = np.asarray(basis.pidx)
        l, pol = np.asarray(basis.l), np.asarray(basis.pol)
        idx = [np.flatnonzero(pidx == i) for i in range(len(lmaxs))]
        sub = {lm: tr.SphericalWaveBasis.default(lm) for lm in set(lmaxs)}  # == each particle's block
        out = np.zeros((len(pidx), len(pidx)), dtype=complex)
        pos = np.asarray(basis.positions)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = idx[i], idx[j]
                for to, frm, li, lj, r in ((a, b, lmaxs[i], lmaxs[j], pos[i] - pos[j]),
                                           (b, a, lmaxs[j], lmaxs[i], pos[j] - pos[i])):
                    if to is b and lmaxs[i] == lmaxs[j]:
                        sign = (-1.0) ** (l[b][:, None] + l[a][None, :]) * np.where(
                            pol[b][:, None] == pol[a][None, :], 1.0, -1.0)
                        out[np.ix_(b, a)] = sign * out[np.ix_(a, b)]
                        continue
                    bt, bf = sub[li], sub[lj]
                    lt, mt, pt = (np.asarray(v)[:, None] for v in bt.lms)
                    lf, mf, pf = (np.asarray(v) for v in bf.lms)
                    rs = tr.special.car2sph(r)
                    mask = mt == mf
                    # treams (and we) call ufuncs with 'where' but no 'out'; masked entries are zeroed.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", "'where' used without 'out'", UserWarning)
                        cz = np.asarray(tr.sw.translate(lt, mt, pt, lf, mf, pf, k * rs[0], 0.0, 0.0,
                                                        poltype="parity", singular=True, where=mask))
                        rot_t = np.asarray(tr.rotate(rs[2], rs[1], 0.0, basis=bt))
                        rot_f = rot_t if li == lj else np.asarray(tr.rotate(rs[2], rs[1], 0.0, basis=bf))
                    cz[~mask] = 0
                    out[np.ix_(to, frm)] = rot_t @ cz @ rot_f.conj().T
        return out

    def tmatrix(self, lam: float, extra: int = 0):
        """(cluster T-matrix, diagonal of the uncoupled T-matrices, coupling matrix or None) at `lam`.

        The T-matrix is global for one particle and local (one origin per particle) otherwise.
        """
        if self._cache is not None and self._cache[0] == (lam, extra):
            return self._cache[1]
        tr = self.treams
        k0 = 2 * np.pi / lam
        embedding = tr.Material(self.sim.medium_index**2)
        lmaxs = [lmax + extra for lmax in self.plan.lmax]
        cache, tms = {}, []
        for p, lmax in zip(self.sim.particles, lmaxs):
            radii, mats = _layers(p)
            eps = tuple(complex(m.eps(np.array([lam]))[0]) for m in mats)
            key = (tuple(radii), eps, lmax)
            if key not in cache:
                cache[key] = tr.TMatrix.sphere(lmax, k0, radii, [*map(tr.Material, eps), embedding],
                                               poltype="parity")
            tms.append(cache[key])
        # Spheres are diagonal in the parity basis.
        t_diag = np.concatenate([np.diag(np.asarray(t)) for t in tms])
        if len(tms) == 1:
            out = (tms[0], t_diag, None)
        else:
            basis = tr.TMatrix.cluster(tms, self.plan.positions).basis
            coup = self._coupling(basis, self.k(lam), lmaxs)
            # Multiple scattering: p = T_i (a_i + sum_j C_ij p_j)  =>  T = (1 - T0 C)^-1 T0.
            # T0 ~ (ka)^(2l+1) and C ~ (kd)^-(l+l'+1) span hundreds of orders of magnitude at
            # high l, which wrecks the plain solve (treams' interaction.solve() fails from
            # l ~ 16 for a close gold dimer). With D = sqrt|T0|, u = T0/|T0| and T = D Y D,
            # (1 - u DCD) Y = u has entries of order (a/d)^(l+l'+1) <= 1.
            d = np.sqrt(np.abs(t_diag))
            u = np.divide(t_diag, d**2, out=np.zeros_like(t_diag), where=d > 0)
            y = np.linalg.solve(np.eye(len(t_diag)) - u[:, None] * (d[:, None] * coup * d[None, :]), np.diag(u))
            tmat = d[:, None] * y * d[None, :]
            out = (tr.TMatrix(tmat, k0=k0, basis=basis, material=embedding, poltype="parity"), t_diag, coup)
        self._cache = ((lam, extra), out)
        return out

    def incident(self, tm):
        # Lists of floats: treams' plane_wave fails on an ndarray polarization (`pol in (0, -1)`)
        # and on integer wave vectors (no integer ufunc loop).
        return self.treams.plane_wave([float(v) for v in self.khat], [float(v) for v in self.ehat], k0=tm.k0,
                                      material=tm.material, poltype="parity")

    def cross_sections(self, lam: float, orientation_average: bool, extra: int = 0) -> tuple[float, float]:
        """(C_ext, C_sca) in nm^2."""
        tm, t_diag, coup = self.tmatrix(lam, extra)
        k = self.k(lam)
        if orientation_average:
            if not tm.basis.isglobal:
                basis = self.treams.SphericalWaveBasis.default(self.plan.lmax_global + extra)
                tm = tm.expand(basis)
            return float(np.real(tm.xs_ext_avg)), float(np.real(tm.xs_sca_avg))
        # C_ext = -Re(a^H p)/k^2 (as treams' TMatrix.xs). C_abs is summed over the spheres from
        # their exciting fields e = a + C p: C_abs = -sum (Re t + |t|^2) |e|^2 / k^2. This avoids
        # the cluster-wide regular translation matrix TMatrix.xs builds for C_sca.
        a = np.asarray(self.incident(tm).expand(tm.basis))
        p = np.asarray(tm) @ a
        exc = a if coup is None else a + coup @ p
        ext = -np.real(np.vdot(a, p)) / k**2
        absr = -np.sum((t_diag.real + np.abs(t_diag) ** 2) * np.abs(exc) ** 2) / k**2
        return float(ext), float(ext - absr)

    def scattered(self, lam: float):
        tm = self.tmatrix(lam)[0]
        inc = self.incident(tm)
        return inc, tm @ inc.expand(tm.basis)

    def far_field(self, req) -> list[FarFieldData]:
        sc = self.treams.special
        khat, ehat = self.khat, self.ehat
        hhat = np.cross(khat, ehat)
        th = np.radians(np.asarray(req.theta_deg, dtype=float))[:, None]
        ph = np.radians(np.asarray(req.phi_deg, dtype=float))[None, :]
        nhat = (np.cos(th)[..., None] * khat + np.sin(th)[..., None]
                * (np.cos(ph)[..., None] * ehat + np.sin(ph)[..., None] * hhat))  # (T, P, 3)
        theta_n = np.arccos(np.clip(nhat[..., 2], -1, 1))[..., None]
        phi_n = np.arctan2(nhat[..., 1], nhat[..., 0])[..., None]
        out = []
        for lam in req.wavelengths_nm:
            k = self.k(lam)
            _, sca = self.scattered(lam)
            b = sca.basis
            l, m, pol = (np.asarray(a) for a in (b.l, b.m, b.pol))
            coef = np.asarray(sca)
            x = np.asarray(sc.vsh_X(l, m, theta_n, phi_n))  # (T, P, modes, 3), spherical components
            y = np.asarray(sc.vsh_Y(l, m, theta_n, phi_n))
            amp = np.where((pol == 1)[:, None], ((-1j) ** l)[:, None] * y, ((-1j) ** (l + 1))[:, None] * x)
            phase = np.exp(-1j * k * nhat @ np.asarray(b.positions).T)[..., np.asarray(b.pidx)]  # (T, P, modes)
            f = np.einsum("tpn,tpnc->tpc", coef * phase, amp)
            out.append(FarFieldData(wavelength_nm=lam, theta_deg=list(req.theta_deg), phi_deg=list(req.phi_deg),
                                    dcsca_domega_nm2=(np.sum(np.abs(f) ** 2, axis=-1) / k**2).tolist()))
        return out

    def near_field(self, req, chunk: int = 2048) -> list[NearFieldData]:
        axis_of = {"x": 0, "y": 1, "z": 2}
        a1, a2 = (axis_of[c] for c in req.plane)
        normal = 3 - a1 - a2
        u = np.linspace(-req.half_width_nm, req.half_width_nm, req.points)
        g1, g2 = np.meshgrid(u, u, indexing="ij")
        pts = np.zeros(g1.shape + (3,))
        pts[..., a1], pts[..., a2], pts[..., normal] = g1, g2, req.offset_nm
        flat = pts.reshape(-1, 3) - self.plan.centre  # treams frame
        inside = np.zeros(len(flat), dtype=bool)
        for r, c in zip(self.plan.radii, self.plan.positions):
            inside |= np.sum((flat - c) ** 2, axis=-1) <= r**2
        outside = np.flatnonzero(~inside)
        out = []
        for lam in req.wavelengths_nm:
            inc, sca = self.scattered(lam)
            # Undo the centring so the incident wave has zero phase at the lab origin.
            phase0 = np.exp(1j * self.k(lam) * float(self.khat @ self.plan.centre))
            e = np.full((len(flat), 3), np.nan, dtype=complex)
            for s in range(0, len(outside), chunk):
                idx = outside[s:s + chunk]
                e[idx] = (np.asarray(inc.efield(flat[idx])) + np.asarray(sca.efield(flat[idx]))) * phase0
            enh = np.sum(np.abs(e) ** 2, axis=-1)
            shape = g1.shape
            out.append(NearFieldData(
                wavelength_nm=lam, plane=req.plane, offset_nm=req.offset_nm, axis1_nm=u.tolist(),
                axis2_nm=u.tolist(), enhancement=enh.reshape(shape).tolist(),
                e_real=np.moveaxis(e.real.reshape(shape + (3,)), -1, 0).tolist(),
                e_imag=np.moveaxis(e.imag.reshape(shape + (3,)), -1, 0).tolist()))
        return out


@register
class TreamsTMatrix(Backend):
    name = "treams"
    method = "tmatrix"
    license = "MIT"
    citation = "Beutel, Fernandez-Corbaton & Rockstuhl, Comput. Phys. Commun. 297, 109076 (2024); treams, " \
               "https://github.com/tfp-photonics/treams"

    @classmethod
    def available(cls):
        try:
            import treams  # noqa: F401
        except ImportError:
            return False, "treams is not installed (pip install treams)"
        return True, ""

    @classmethod
    def supports(cls, sim: Simulation):
        for i, p in enumerate(sim.particles):
            if not isinstance(p.shape, (Sphere, LayeredSphere)):
                return False, (f"the treams T-matrix backend handles spheres and layered spheres only, got "
                               f"{p.shape.kind!r} (particle {i}); treams has no supported T-matrix generator "
                               "for non-spherical particles, use 'bem' or 'dda'")
        pair = _overlap(sim)
        if pair is not None:
            return False, (f"particles {pair[0]} and {pair[1]} overlap; the T-matrix method needs "
                           "non-overlapping spheres, use 'dda'")
        why = unsupported_outputs(sim, near_field=True, far_field=True, orientation_average=True)
        if not why and sim.orientation_average and (sim.outputs.near_field or sim.outputs.far_field):
            why = "field outputs are not available with orientation_average"
        return (not why), why

    @classmethod
    def _options(cls, sim: Simulation) -> TreamsOptions:
        return TreamsOptions.model_validate(sim.options_for(cls.name, cls.method))

    @classmethod
    def estimate(cls, sim, device):
        opts = cls._options(sim)
        plan = _Plan(sim, opts)
        dims = [plan.modes] + ([plan.modes_global] if sim.orientation_average and len(sim.particles) > 1 else [])
        # T-matrix, interaction matrix and its LU factors, plus translation work arrays.
        mem = 16 * 4 * max(dims) ** 2
        return {"lmax": plan.lmax, "modes": plan.modes,
                "lmax_global": plan.lmax_global if sim.orientation_average else None,
                "memory_bytes": int(mem), "solves": len(sim.wavelength_array()),
                "fits": plan.modes <= opts.max_modes}

    @classmethod
    def solve(cls, sim: Simulation, device: Device) -> Result:
        t0 = time.perf_counter()
        opts = cls._options(sim)
        plan = _Plan(sim, opts)
        if plan.modes > opts.max_modes:
            raise ValueError(f"cluster T-matrix has {plan.modes} modes, more than max_modes={opts.max_modes}; "
                             "lower lmax or raise max_modes")
        solver = _Solver(sim, plan)
        wl = sim.wavelength_array()
        oa = sim.orientation_average
        cs = np.array([solver.cross_sections(lam, oa) for lam in wl])
        c_ext, c_sca = cs[:, 0], cs[:, 1]

        warnings = [] if device.name == "cpu" else ["treams always runs on the CPU; device ignored"]
        provenance = {
            "treams_version": _treams_version(), "device": "cpu", "lmax": plan.lmax,
            "extra_orders_close": plan.extra_close, "modes": plan.modes, "poltype": "parity",
            "time_convention": "exp(-i omega t)", "integer_shim": bool(_SHIM_INSTALLED),
        }
        if oa and len(sim.particles) > 1:
            provenance["lmax_global"] = plan.lmax_global
        if len(sim.particles) > 1 and opts.convergence_check:
            i = int(np.argmax(c_ext))
            ext2, sca2 = solver.cross_sections(float(wl[i]), oa, extra=2)
            rel = {"c_ext": abs(ext2 - c_ext[i]) / abs(ext2), "c_sca": abs(sca2 - c_sca[i]) / abs(sca2)}
            provenance["convergence"] = {"wavelength_nm": float(wl[i]), "extra_orders": 2,
                                         **{f"rel_change_{q}": float(v) for q, v in rel.items()}}
            if max(rel.values()) > opts.convergence_tol:
                warnings.append(f"not converged in multipole order at {wl[i]:.1f} nm: lmax + 2 changes the cross "
                                f"sections by {max(rel.values()):.1e} (> {opts.convergence_tol:g}); raise "
                                "options['tmatrix']['extra_orders'] or 'lmax'")

        near = far = None
        if sim.outputs.far_field:
            far = solver.far_field(sim.outputs.far_field)
        if sim.outputs.near_field:
            near = solver.near_field(sim.outputs.near_field)
        provenance["runtime_s"] = time.perf_counter() - t0

        res = Result.from_arrays(
            method=cls.method, backend=cls.name, wavelengths_nm=wl,
            c_ext=c_ext, c_sca=c_sca, c_abs=c_ext - c_sca, r_eq=sim.equivalent_radius_nm(),
            provenance=provenance, warnings=warnings,
        )
        res.near_field, res.far_field = near, far
        return res
