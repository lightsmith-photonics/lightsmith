"""The topology-optimisation loop: parametrise -> differentiable solve -> objective -> optimiser.

Use :func:`optimize` for the JSON path (``OptimizationProblem`` in, ``OptimizationResult``
out) or :class:`TopologyOptimizer` to drive the pieces yourself.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..backends import get_backend
from ..device import Device, resolve_device
from ..problems import DeviceProblem, Structure, wavelength_array
from ..results import SParameterResult
from . import objectives as obj
from .export import marching_squares, to_structures
from .optimizers import MMA, Adam
from .parametrization import RegionMap, compose_eps, grayness, lengthscale_constraints, region_map
from .spec import FieldIntensityTerm, OptimizationProblem


class DesignOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    x_nm: list[float]
    y_nm: list[float]
    density: list[list[float]] = Field(description="Projected density at the final beta, [x][y].")
    raw: list[list[float]] = Field(description="Optimisation variables, [x][y].")


class OptimizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = "optimization"
    method: str
    backend: str
    history: list[dict[str, Any]] = Field(description="Per iteration: objective (maximised), terms, beta, "
                                          "binarization, penalty, evaluations.")
    objective_initial: float
    objective_final: float
    objective_binary: float = Field(description="Objective of the hard-thresholded (binary) final design.")
    designs: list[DesignOutput]
    structures: list[Structure] = Field(description="Exported binary design (polygons via marching squares).")
    final_device: DeviceProblem = Field(description="The input device plus the exported structures.")
    sparameters: SParameterResult | None = Field(None, description="S-parameters of final_device (re-solved).")
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class IterationState:
    iteration: int
    objective: float  # value being maximised (without penalties)
    terms: dict[str, float]
    beta: float
    binarization: float
    penalty: float
    evaluations: int
    x: np.ndarray = field(repr=False)
    elapsed_s: float = 0.0

    def record(self) -> dict[str, Any]:
        return {"iteration": self.iteration, "objective": self.objective, "terms": self.terms, "beta": self.beta,
                "binarization": self.binarization, "penalty": self.penalty, "evaluations": self.evaluations,
                "elapsed_s": round(self.elapsed_s, 3)}


Callback = Callable[[IterationState], bool | None]


def check_backend(backend) -> None:
    """Raise unless `backend` implements the differentiable protocol of :mod:`lightsmith.inverse`."""
    missing = [m for m in ("design_grid", "base_permittivity", "differentiable_solve") if not hasattr(backend, m)]
    if not getattr(backend, "differentiable", False) or missing:
        raise ValueError(f"backend {getattr(backend, 'name', backend)!r} is not usable for inverse design: needs "
                         f"differentiable=True and {', '.join(missing) or 'the functional API'}")


class TopologyOptimizer:
    """Density-based topology optimisation of an :class:`OptimizationProblem`."""

    def __init__(self, problem: OptimizationProblem | dict | str, device: str | Device = "cpu"):
        import torch

        if isinstance(problem, str):
            problem = OptimizationProblem.model_validate_json(problem)
        elif isinstance(problem, dict):
            problem = OptimizationProblem.model_validate(problem)
        self.problem = problem
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.tdev = torch.device(self.device.name) if self.device.framework == "torch" else torch.device("cpu")
        self.backend = get_backend(problem.backend, kind="device")
        check_backend(self.backend)
        dev_problem = problem.device
        if problem.options:  # optimisation-level solver options override the device's, key by key per method
            dev_opts = dev_problem.options
            merged = {m: {**dev_opts.get(m, {}), **problem.options.get(m, {})} for m in {*dev_opts, *problem.options}}
            dev_problem = dev_problem.model_copy(update={"options": merged})
        self.device_problem = dev_problem
        self.ctx = self.backend.prepare(dev_problem) if hasattr(self.backend, "prepare") else {}
        self.grid = self.backend.design_grid(dev_problem)
        self.wavelengths = wavelength_array(dev_problem.wavelengths)
        self.base = torch.as_tensor(self.backend.base_permittivity(dev_problem), device=self.tdev)
        self.regions: list[RegionMap] = [
            region_map(r.name, r.center_nm, r.size_nm, self.grid, r.material_low.eps(self.wavelengths),
                       r.material_high.eps(self.wavelengths), filter_radius_nm=r.filter_radius_nm, eta=r.eta,
                       min_feature_nm=r.min_feature_nm, symmetry=tuple(r.symmetry))
            for r in problem.design_regions]
        self._check_ports()
        labels = self._labels()
        self.sources = obj.required_sources(problem.objective, labels)
        self.need_fields = any(isinstance(t, FieldIntensityTerm) for t in problem.objective)
        self.evaluations = 0
        self.warnings: list[str] = []

    # --- setup helpers ---------------------------------------------------------------

    def _labels(self) -> list[str]:
        out = []
        for p in self.device_problem.ports:
            out += [p.name] if p.num_modes == 1 else [f"{p.name}@{m}" for m in range(p.num_modes)]
        return out

    def _check_ports(self) -> None:
        eng = self.ctx.get("engine")
        if eng is None:
            return
        for reg in self.regions:
            for pg in eng.ports:
                rng = (reg.ix, reg.iy)[pg.axis]
                lines = {pg.i0 + k * pg.d for k in (-2, -1, 0, 1, 2, 3)}
                trans = (reg.iy, reg.ix)[pg.axis]
                if any(rng.start <= i < rng.stop for i in lines) and trans.start < pg.hi and pg.lo < trans.stop:
                    raise ValueError(f"design region {reg.name!r} overlaps port {pg.name!r} (keep >= 3 cells "
                                     "between a port plane and the design region)")

    def initial_x(self) -> np.ndarray:
        parts = []
        for r, reg in zip(self.problem.design_regions, self.regions):
            if r.init == "random":
                parts.append(np.random.default_rng(r.seed).uniform(0, 1, reg.size))
            else:
                parts.append(np.full(reg.size, float(r.init)))
        return np.concatenate(parts)

    def split(self, x):
        out, k = [], 0
        for reg in self.regions:
            out.append(x[k:k + reg.size].reshape(reg.shape))
            k += reg.size
        return out

    # --- evaluation ----------------------------------------------------------------------

    def permittivity(self, x, beta: float):
        """(eps grid (n_wl, Nx, Ny), projected densities, filtered densities); differentiable in x."""
        rhos = self.split(x)
        filt = [reg.filtered(r) for reg, r in zip(self.regions, rhos)]
        from .parametrization import tanh_projection

        proj = [tanh_projection(f, beta, reg.eta) for reg, f in zip(self.regions, filt)]
        return compose_eps(self.base, self.regions, proj), proj, filt

    def solve(self, eps, with_fields: bool | None = None):
        wf = self.need_fields if with_fields is None else with_fields
        return self.backend.differentiable_solve(self.device_problem, eps, self.device, sources=self.sources,
                                                 with_fields=wf, **self.ctx)

    def evaluate(self, x: np.ndarray, beta: float, grad: bool = True) -> tuple[float, np.ndarray | None, dict]:
        """Loss (= -objective + penalty, minimised), its gradient, and details."""
        import torch

        self.evaluations += 1
        xt = torch.as_tensor(np.asarray(x, float), device=self.tdev).requires_grad_(grad)
        with torch.set_grad_enabled(grad):
            eps, proj, filt = self.permittivity(xt, beta)
            sol = self.solve(eps)
            F, parts = obj.evaluate(self.problem.objective, sol)
            pen = torch.zeros((), dtype=torch.float64, device=self.tdev)
            st = self.problem.optimizer
            if st.lengthscale_weight > 0 and beta >= st.lengthscale_from_beta:
                for reg, rp, rf in zip(self.regions, proj, filt):
                    if reg.min_feature_nm:
                        gs, gv = lengthscale_constraints(rf, rp, reg.dl_nm, reg.filter_radius_nm, reg.min_feature_nm)
                        pen = pen + st.lengthscale_weight * (gs + gv)
            loss = -F + pen
        g = None
        if grad:
            loss.backward()
            g = xt.grad.detach().cpu().numpy()
        info = {"objective": float(F.detach()), "terms": {k: float(v.detach()) for k, v in parts.items()},
                "penalty": float(pen.detach()),
                "binarization": 1 - float(np.mean([grayness(p.detach().cpu().numpy()) for p in proj]))}
        return float(loss.detach()), g, info

    # --- the loop --------------------------------------------------------------------

    def run(self, callback: Callback | None = None, checkpoint: str | None = None, checkpoint_every: int = 5,
            resume: bool = True, x0: np.ndarray | None = None) -> dict[str, Any]:
        st = self.problem.optimizer
        t0 = time.perf_counter()
        x = self.initial_x() if x0 is None else np.asarray(x0, float).copy()
        beta, it, history = st.beta_start, 0, []
        opt = MMA(move=st.move_limit) if st.method == "mma" else Adam(lr=st.learning_rate)
        if checkpoint and resume and os.path.exists(checkpoint):
            x, beta, it, history, opt_state = load_checkpoint(checkpoint)
            opt.load(opt_state) if st.method != "lbfgsb" else None

        cache: dict[str, Any] = {}

        def fg(xv, b=None):
            b = beta if b is None else b
            f, g, info = self.evaluate(xv, b)
            cache.update(x=xv.copy(), f=f, g=g, info=info)
            return f, g

        def emit(xv, info) -> bool:
            nonlocal it
            state = IterationState(iteration=it, objective=info["objective"], terms=info["terms"], beta=beta,
                                   binarization=info["binarization"], penalty=info["penalty"],
                                   evaluations=self.evaluations, x=xv, elapsed_s=time.perf_counter() - t0)
            history.append(state.record())
            stop = bool(callback(state)) if callback else False
            if checkpoint and (it % checkpoint_every == 0):
                save_checkpoint(checkpoint, xv, beta, it + 1, history, opt.state() if st.method != "lbfgsb" else {})
            it += 1
            if st.ftol > 0 and len(history) > 10:
                old, new = history[-11]["objective"], history[-1]["objective"]
                if abs(new - old) <= st.ftol * max(abs(new), 1e-12) and history[-11]["beta"] == beta:
                    stop = True
            return stop

        if st.method == "lbfgsb":
            from scipy.optimize import minimize

            stop = False
            while it < st.max_iter and not stop:
                fg(x)
                stop = emit(x, cache["info"])
                n = min(st.beta_every, st.max_iter - it)
                if n <= 0 or stop:
                    break

                def cb(xk):
                    if not np.array_equal(xk, cache["x"]):
                        fg(xk)
                    if emit(xk, cache["info"]):
                        raise StopIteration

                try:
                    res = minimize(fg, x, jac=True, method="L-BFGS-B", bounds=[(0, 1)] * len(x), callback=cb,
                                   options={"maxiter": n - 1 if n > 1 else 1, "ftol": 0, "gtol": 1e-12})
                    x = res.x
                except StopIteration:
                    x = cache["x"]
                    stop = True
                if beta < st.beta_max:
                    beta = min(beta * st.beta_growth, st.beta_max)
        else:
            f, g = fg(x)
            while it < st.max_iter:
                if emit(x, cache["info"]) or it >= st.max_iter:
                    break
                if it % st.beta_every == 0 and beta < st.beta_max and it > 0:
                    beta = min(beta * st.beta_growth, st.beta_max)
                    opt.reset()
                    f, g = fg(x)
                x, cached = opt.step(x, f, g, fg)
                if cached is not None:
                    f, g = cached
                    if not np.array_equal(cache.get("x"), x):
                        f, g = fg(x)
                else:
                    f, g = fg(x)
        if checkpoint:
            save_checkpoint(checkpoint, x, beta, it, history, opt.state() if st.method != "lbfgsb" else {})
        return {"x": x, "beta": beta, "history": history, "runtime_s": time.perf_counter() - t0}

    # --- outputs ------------------------------------------------------------------------

    def export(self, x: np.ndarray, beta: float) -> tuple[list[Structure], list]:
        """Structures (region box + polygons) of the binary design: the filtered density contoured at eta.

        This is the beta -> infinity limit of the projection, i.e. the design scored by
        ``objective_binary``.
        """
        import torch

        with torch.no_grad():
            _, _, filt = self.permittivity(torch.as_tensor(x, device=self.tdev), beta)
        structs, designs = [], []
        for r, reg, rf in zip(self.problem.design_regions, self.regions, filt):
            contours = marching_squares(rf.cpu().numpy(), reg.x_nm, reg.y_nm, level=reg.eta)
            zc, zs = r.center_nm[2], r.size_nm[2]
            zmin, zmax = (zc - zs / 2, zc + zs / 2) if zs > 0 else (zc - 110.0, zc + 110.0)
            if self.problem.device.dimension == 2:
                zmin, zmax = min(zmin, self.grid.z0 - 1), max(zmax, self.grid.z0 + 1)
            # The region box spans whole cells around the nodes inside the region.
            cx = (reg.x_nm[0] + reg.x_nm[-1]) / 2
            cy = (reg.y_nm[0] + reg.y_nm[-1]) / 2
            size = (len(reg.x_nm) * reg.dl_nm, len(reg.y_nm) * reg.dl_nm)
            structs += to_structures(contours, r.material_high, r.material_low, zmin, zmax, (cx, cy), size, r.name)
            designs.append(contours)
        return structs, designs


def save_checkpoint(path: str, x, beta, iteration, history, opt_state: dict) -> None:
    arrays = {f"opt_{k}": np.asarray(v) for k, v in opt_state.items() if v is not None}
    tmp = path + ".tmp.npz"
    np.savez(tmp, x=x, beta=beta, iteration=iteration, history=json.dumps(history), **arrays)
    os.replace(tmp, path)


def load_checkpoint(path: str):
    with np.load(path, allow_pickle=False) as d:
        opt = {k[4:]: (d[k] if d[k].ndim else d[k].item()) for k in d.files if k.startswith("opt_")}
        return d["x"].copy(), float(d["beta"]), int(d["iteration"]), json.loads(str(d["history"])), opt


def optimize(problem: OptimizationProblem | dict | str, device: str = "cpu", callback: Callback | None = None,
             checkpoint: str | None = None, checkpoint_every: int = 5, resume: bool = True,
             validate: bool = True) -> OptimizationResult:
    """Run topology optimisation and export the final design.

    device: 'cpu', 'gpu', 'auto', ... (FDFD uses SciPy on CPU, the torch block solver on GPU).
    callback(state) -> True stops early. checkpoint: path of an .npz written every
    `checkpoint_every` iterations; with resume=True an existing checkpoint is continued.
    validate: re-solve the exported binary geometry and attach its S-parameters.
    """
    topt = TopologyOptimizer(problem, device)
    out = topt.run(callback=callback, checkpoint=checkpoint, checkpoint_every=checkpoint_every, resume=resume)
    x, beta, history = out["x"], out["beta"], out["history"]
    p = topt.problem
    f_bin = topt.evaluate(x, math.inf, grad=False)[2]["objective"]
    structs, _ = topt.export(x, beta)
    final_device = p.device.model_copy(update={"structures": list(p.device.structures) + structs})
    import torch

    with torch.no_grad():
        _, proj, _ = topt.permittivity(torch.as_tensor(x, device=topt.tdev), beta)
    designs = [DesignOutput(name=reg.name, x_nm=reg.x_nm.tolist(), y_nm=reg.y_nm.tolist(),
                            density=rp.cpu().numpy().tolist(), raw=xr.tolist())
               for reg, rp, xr in zip(topt.regions, proj, topt.split(x))]
    warnings = list(topt.warnings)
    if history and history[-1]["binarization"] < 0.9:
        warnings.append(f"final design is only {history[-1]['binarization']:.0%} binary; raise beta_max or "
                        "max_iter (objective_binary is the thresholded performance)")
    sp = None
    if validate:
        b = topt.backend
        dev = topt.device if b.gpu else resolve_device("cpu")
        sp = b.solve(final_device, dev)
        warnings += [f"validation: {w}" for w in sp.warnings]
    return OptimizationResult(
        method=p.optimizer.method, backend=topt.backend.name, history=history,
        objective_initial=history[0]["objective"] if history else float("nan"),
        objective_final=history[-1]["objective"] if history else float("nan"),
        objective_binary=f_bin, designs=designs, structures=structs, final_device=final_device, sparameters=sp,
        provenance={**topt.device.describe(), "evaluations": topt.evaluations, "runtime_s": out["runtime_s"],
                    "final_beta": beta, "grid": list(topt.grid.shape), "dl_nm": float(topt.grid.dl)},
        warnings=warnings)
