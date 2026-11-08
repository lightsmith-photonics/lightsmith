"""Native boundary element method (surface integral equations), CPU or GPU.

Homogeneous particles (and concentric layered spheres) in a homogeneous host are
described by equivalent electric and magnetic surface currents on their
boundaries, expanded in RWG functions on flat-triangle meshes and solved with
the PMCHWT formulation (see ``_bem/operators.py``). Only surfaces are
discretised, so curved metal boundaries are captured far better than by a
staircase volume grid: for a gold sphere the BEM error is ~0.01 % at ~10
triangles per diameter, where DDA and FDTD are off by percent to tens of percent
(see ``benchmarks/bem_convergence.py``).

Cross sections: extinction from the optical theorem (the Galerkin functional
-Re(b^H x), whose error is quadratic in the current error), scattering from the
power flux of the far field over a Gauss-Legendre sphere, and absorption as
their difference (as in MNPBEM). The Poynting flux through the particle surface,
Re sum J'^* . (n x M), is reported in provenance (`c_abs_flux_nm2`) as an
independent check only: with RWG currents it converges slowly (errors of 0.5-2 %
for gold at typical meshes, tens of percent for silver near 350 nm), whereas the
difference is within ~1e-4 of C_ext. For a nearly lossless particle that
difference is a small number obtained by cancellation; its absolute error stays
~1e-4 C_ext.

Kern & Martin, JOSA A 26, 732 (2009); Rao, Wilton & Glisson, IEEE TAP 30, 409
(1982); singular integrals after Graglia, IEEE TAP 41, 1448 (1993).
"""

from __future__ import annotations

import math
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..device import Device
from ..progress import report
from ..results import FarFieldData, NearFieldData, Result
from ..spec import Simulation
from ._bem import fields
from ._bem.mesh import build_mesh, check_overlaps, default_mesh_size, min_gaps, surface_growth
from ._bem.operators import Geometry, DeviceGeometry, assemble, gmres, incident_rhs
from .base import Backend, register, unsupported_outputs


class BEMOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mesh_size_nm: float | None = Field(
        None, gt=0, description="Target mean triangle edge length. Default: the smaller of "
        "size/elements_per_size and lambda_min/(|n|_max elements_per_wavelength).")
    elements_per_size: float = Field(10.0, gt=0, description="Edges across the smallest particle dimension.")
    elements_per_wavelength: float = Field(10.0, gt=0, description="Edges per wavelength in the densest material.")
    volume_correction: bool = Field(
        True, description="Rescale each faceted surface to the true volume (radially for cylinders). "
        "Removes the leading O(h^2) error of the inscribed polyhedron.")
    quadrature: Literal[3, 7, 12] = Field(7, description="Points of the triangle quadrature rule.")
    near_factor: float = Field(2.0, gt=0, description="Triangle pairs closer than near_factor x (sum of their "
                               "circumradii) get analytic singularity extraction.")
    solver: Literal["auto", "direct", "gmres"] = Field(
        "auto", description="direct: dense LU. gmres: restarted GMRES (Jacobi preconditioned) on the dense matrix, "
        "saving the O(N^3) factorisation. auto: direct up to direct_max_unknowns.")
    direct_max_unknowns: int = Field(20000, ge=1)
    tol: float = Field(1e-8, gt=0, description="GMRES relative residual target.")
    restart: int = Field(100, ge=1)
    max_iter: int = Field(3000, ge=1)
    precision: Literal["double", "single"] = Field(
        "double", description="single halves memory and is faster on consumer GPUs; measured within ~2e-4 of "
        "double for gold spheres, but conditioning worsens on very fine meshes.")
    max_unknowns: int = Field(60000, ge=1, description="Refuse problems with more unknowns (2 x edges).")
    orientations: int = Field(32, ge=1, description="Incidence directions (x2 polarizations) for orientation_average.")
    assembly_budget: int | None = Field(
        None, ge=1000, description="Kernel evaluations per assembly chunk. Default 2e6 on CPU, 8e6 on GPU.")


def _orientation_set(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from .dda import _orientation_set as fib

    return fib(n)


@register
class NativeBEM(Backend):
    name = "lightsmith-bem"
    method = "bem"
    options_model = BEMOptions
    license = "MIT"
    citation = ("Kern & Martin, JOSA A 26, 732 (2009); Rao, Wilton & Glisson, IEEE TAP 30, 409 (1982); "
                "Graglia, IEEE TAP 41, 1448 (1993)")
    gpu = True

    @classmethod
    def supports(cls, sim: Simulation):
        if sim.substrate is not None:
            return False, ("BEM here uses the homogeneous-space Green's function and does not support a substrate; "
                           "use 'fdtd'")
        why = unsupported_outputs(sim, near_field=True, far_field=True, orientation_average=True)
        if not why and sim.orientation_average and (sim.outputs.near_field or sim.outputs.far_field):
            why = "field outputs are not available with orientation_average"
        if not why:
            why = cls._overlap(sim)
        return (not why), why

    @classmethod
    def _overlap(cls, sim: Simulation) -> str:
        if len(sim.particles) < 2:
            return ""
        size = min(2 * p.shape.bounding_half_extent().min() for p in sim.particles)
        why = check_overlaps(sim, build_mesh(sim, size / 6, volume_correction=False))
        if why:
            return why
        try:
            opts = cls._options(sim)
        except ValueError:
            return ""  # supports() reports invalid options separately
        if not opts.volume_correction:
            return ""
        # The solve rescales each faceted surface to the true volume, which pushes it slightly outside the
        # exact shape: check those meshes too, or nearly touching particles end up interpenetrating.
        h = cls._mesh_size(sim, opts)
        exact, corrected = build_mesh(sim, h, False), build_mesh(sim, h, True)
        why = check_overlaps(sim, corrected, surface_growth(sim, exact, corrected))
        if why:
            return (why.replace("overlap or touch", "are so close that their volume-corrected meshes overlap")
                    + "; set volume_correction=False, or a smaller mesh_size_nm")
        return ""

    @staticmethod
    def _gap_warnings(sim: Simulation, mesh, h: float) -> list[str]:
        return [f"particles {i} and {j} are about {d:.3g} nm apart, less than the mesh size {h:.3g} nm: the gap "
                f"field is not resolved; set mesh_size_nm below the gap" for i, j, d in min_gaps(sim, mesh) if d < h]

    @classmethod
    def _options(cls, sim) -> BEMOptions:
        return BEMOptions.model_validate(sim.options_for(cls.name, cls.method))

    @classmethod
    def _mesh_size(cls, sim, opts) -> float:
        return opts.mesh_size_nm or default_mesh_size(sim, opts.elements_per_wavelength, opts.elements_per_size)

    @classmethod
    def estimate(cls, sim, device):
        opts = cls._options(sim)
        h = cls._mesh_size(sim, opts)
        mesh = build_mesh(sim, h, opts.volume_correction)
        N = mesh.n_edges
        n = 2 * N
        csize = 8 if opts.precision == "single" else 16
        budget = opts.assembly_budget or (8_000_000 if device.is_gpu else 2_000_000)
        n_reg = max(len(np.nonzero(np.isin(mesh.edge_surface, [s for s, _ in mesh.region_surfaces(r)]))[0])
                    for r in range(mesh.n_regions))
        # system matrix + one region's L and K + ~8 complex work arrays of one assembly chunk
        mem = csize * (n * n + 2 * n_reg**2) + 8 * csize * budget
        solver = cls._solver(opts, n)
        n_inc = 2 * opts.orientations if sim.orientation_average else 1
        return {"unknowns": n, "edges": N, "triangles": int(len(mesh.tris)), "mesh_size_nm": h,
                "mean_edge_nm": mesh.mean_edge_length(), "surfaces": len(mesh.surfaces),
                "precision": opts.precision, "solver": solver, "memory_bytes": int(mem),
                "solves": len(sim.wavelength_array()) * (1 if solver == "direct" else n_inc),
                "fits": n <= opts.max_unknowns}

    @staticmethod
    def _solver(opts: BEMOptions, n: int) -> str:
        if opts.solver != "auto":
            return opts.solver
        return "direct" if n <= opts.direct_max_unknowns else "gmres"

    @classmethod
    def solve(cls, sim: Simulation, device: Device) -> Result:
        t0 = time.perf_counter()
        opts = cls._options(sim)
        why = cls._overlap(sim)
        if why:
            raise ValueError(why)
        single = opts.precision == "single"
        xp = device.namespace()
        wl = sim.wavelength_array()
        n0 = sim.medium_index
        h = cls._mesh_size(sim, opts)
        mesh = build_mesh(sim, h, opts.volume_correction)
        n = 2 * mesh.n_edges
        if n > opts.max_unknowns:
            raise ValueError(f"{n} unknowns exceeds max_unknowns={opts.max_unknowns}; raise mesh_size_nm "
                             f"(now {h:.3g} nm) or max_unknowns")
        geo = Geometry(mesh, opts.quadrature, opts.near_factor)
        dg = DeviceGeometry(geo, device, single)
        budget = opts.assembly_budget or (8_000_000 if device.is_gpu else 2_000_000)
        solver = cls._solver(opts, n)
        khat, ehat = sim.illumination.unit_vectors()
        incidences = _orientation_set(opts.orientations) if sim.orientation_average else [(khat, ehat)]
        t_setup = time.perf_counter() - t0

        # wavelengths needed for fields reuse the cross-section solves when they coincide
        field_wl = set()
        if sim.outputs.far_field:
            field_wl |= set(sim.outputs.far_field.wavelengths_nm)
        if sim.outputs.near_field:
            field_wl |= set(sim.outputs.near_field.wavelengths_nm)
        all_wl = list(wl) + [w for w in sorted(field_wl) if not np.any(np.isclose(wl, w))]

        c_ext, c_sca, c_abs, c_flux = (np.zeros(len(wl)) for _ in range(4))
        solutions: dict[float, np.ndarray] = {}
        iterations, residuals, warnings = [], [], cls._gap_warnings(sim, mesh, h)
        t_asm = t_sol = 0.0
        for i, lam in enumerate(all_wl):
            report("bem_solve", 0.95 * i / len(all_wl), wavelength_nm=float(lam))
            ta = time.perf_counter()
            Z = assemble(dg, lam, n0, budget)
            b = incident_rhs(geo, lam, n0, incidences if i < len(wl) else [(khat, ehat)])
            bd = device.asarray(b, dtype=dg.cdt)
            _ = float(xp.real(Z[0, 0]))  # synchronise for timing
            tb = time.perf_counter()
            t_asm += tb - ta
            if solver == "direct":
                x = Device.to_numpy(xp.linalg.solve(Z, bd))
            else:
                diag = 1 / xp.linalg.diagonal(Z) if hasattr(xp.linalg, "diagonal") else 1 / xp.diagonal(Z)
                cols = []
                for j in range(b.shape[1]):
                    xj, it, res = gmres(lambda v, Z=Z: xp.matmul(Z, v), bd[:, j], xp, opts.tol, opts.restart,
                                        opts.max_iter, precond=lambda v: diag * v)
                    cols.append(Device.to_numpy(xj))
                    iterations.append(it)
                    residuals.append(res)
                    if res > opts.tol:
                        warnings.append(f"GMRES did not converge at {lam:.1f} nm (residual {res:.2e}, {it} it)")
                x = np.stack(cols, axis=1)
            del Z
            t_sol += time.perf_counter() - tb
            x = x.astype(complex)
            if not np.all(np.isfinite(x)):
                warnings.append(f"non-finite currents at {lam:.1f} nm; try precision='double' or a finer mesh")
            if any(np.isclose(lam, w) for w in field_wl):
                solutions[float(lam)] = x[:, 0]
            if i >= len(wl):
                continue
            k = 2 * math.pi * n0 / lam
            ext = -np.real(np.sum(np.conj(b) * x, axis=0))
            sca = fields.scattering(geo, x, k, device, single)
            flux = fields.absorption(geo, x)
            c_ext[i], c_sca[i], c_flux[i] = ext.mean(), sca.mean(), flux.mean()
            c_abs[i] = c_ext[i] - c_sca[i]

        def sol(lam):
            return solutions[min(solutions, key=lambda w: abs(w - lam))]

        far = near = None
        if sim.outputs.far_field:
            far = cls._far_field(geo, sim.outputs.far_field, sol, n0, khat, ehat, device, single)
        if sim.outputs.near_field:
            near = cls._near_field(geo, sim, sim.outputs.near_field, sol, khat, ehat, device, single, budget)

        provenance = {
            **device.describe(), "formulation": "PMCHWT, RWG", "precision": opts.precision, "solver": solver,
            "unknowns": n, "edges": mesh.n_edges, "triangles": int(len(mesh.tris)), "surfaces": len(mesh.surfaces),
            "mesh_size_nm": h, "mean_edge_nm": mesh.mean_edge_length(), "volume_correction": opts.volume_correction,
            "quadrature": opts.quadrature, "near_pairs": geo.n_near_pairs, "incidences": len(incidences),
            "cross_sections": "C_ext: optical theorem; C_sca: far-field flux; C_abs = C_ext - C_sca",
            "c_abs_flux_nm2": c_flux.tolist(),
            "setup_s": t_setup, "assembly_s": t_asm, "solve_s": t_sol, "runtime_s": time.perf_counter() - t0,
        }
        if iterations:
            provenance["iterations"], provenance["residuals"] = iterations, residuals
        res = Result.from_arrays(method=cls.method, backend=cls.name, wavelengths_nm=wl, c_ext=c_ext, c_sca=c_sca,
                                 c_abs=c_abs, r_eq=sim.equivalent_radius_nm(), provenance=provenance, warnings=warnings)
        res.near_field, res.far_field = near, far
        return res

    @staticmethod
    def _far_field(geo, req, sol, n0, khat, ehat, device, single) -> list[FarFieldData]:
        hhat = np.cross(khat, ehat)
        th = np.radians(np.asarray(req.theta_deg))[:, None]
        ph = np.radians(np.asarray(req.phi_deg))[None, :]
        nhat = (np.cos(th)[..., None] * khat + np.sin(th)[..., None]
                * (np.cos(ph)[..., None] * ehat + np.sin(ph)[..., None] * hhat))  # (T, P, 3)
        out = []
        for lam in req.wavelengths_nm:
            k = 2 * math.pi * n0 / lam
            F = fields.far_field(geo, sol(lam)[:, None], k, nhat.reshape(-1, 3), device, single)[..., 0]
            dcs = np.sum(np.abs(F) ** 2, axis=-1).reshape(nhat.shape[:2])
            out.append(FarFieldData(wavelength_nm=lam, theta_deg=list(req.theta_deg), phi_deg=list(req.phi_deg),
                                    dcsca_domega_nm2=dcs.tolist()))
        return out

    @staticmethod
    def _near_field(geo, sim, req, sol, khat, ehat, device, single, budget) -> list[NearFieldData]:
        axis_of = {"x": 0, "y": 1, "z": 2}
        a1, a2 = (axis_of[c] for c in req.plane)
        normal = 3 - a1 - a2
        u = np.linspace(-req.half_width_nm, req.half_width_nm, req.points)
        g1, g2 = np.meshgrid(u, u, indexing="ij")
        pts = np.zeros(g1.shape + (3,))
        pts[..., a1], pts[..., a2], pts[..., normal] = g1, g2, req.offset_nm
        flat = pts.reshape(-1, 3)
        out = []
        for lam in req.wavelengths_nm:
            e = fields.near_field(geo, sim, sol(lam), lam, flat, khat, ehat, device, single, budget // 8)
            shape = g1.shape
            out.append(NearFieldData(
                wavelength_nm=lam, plane=req.plane, offset_nm=req.offset_nm, axis1_nm=u.tolist(), axis2_nm=u.tolist(),
                enhancement=np.sum(np.abs(e) ** 2, axis=-1).reshape(shape).tolist(),
                e_real=np.moveaxis(e.real.reshape(shape + (3,)), -1, 0).tolist(),
                e_imag=np.moveaxis(e.imag.reshape(shape + (3,)), -1, 0).tolist()))
        return out
