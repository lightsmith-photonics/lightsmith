"""Bound-constrained gradient optimisers on NumPy vectors in [0, 1] (minimisation).

- :class:`MMA`: Svanberg's method of moving asymptotes for a box-constrained
  objective, with the globally convergent (GCMMA) conservativeness loop, so every
  accepted iterate decreases the objective. Without constraints the MMA
  subproblem is separable and solved in closed form. Svanberg, SIAM J. Optim.
  12, 555 (2002). As a practical relaxation, a trial point that already lowers the
  objective is accepted even if the approximation was not conservative there.
- :class:`Adam`: Kingma & Ba (2015), projected onto the box.
- L-BFGS-B is run through SciPy directly in :mod:`omniscatter.inverse.optimize`.

Each optimiser's ``step(x, f, g, evaluate)`` returns ``(x_new, cached)``, where
``cached`` is ``(f, g)`` at ``x_new`` when the step already evaluated it.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

Evaluate = Callable[[np.ndarray], tuple[float, np.ndarray]]


class Adam:
    def __init__(self, lr: float = 0.02, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, x, f, g, evaluate: Evaluate | None = None):
        if self.m is None:
            self.m, self.v = np.zeros_like(x), np.zeros_like(x)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return np.clip(x - self.lr * mh / (np.sqrt(vh) + self.eps), 0, 1), None

    def state(self) -> dict[str, Any]:
        return {"m": self.m, "v": self.v, "t": self.t}

    def load(self, st: dict[str, Any]) -> None:
        self.m, self.v, self.t = st.get("m"), st.get("v"), int(st.get("t", 0))

    def reset(self) -> None:
        """Called when the objective changes (beta continuation)."""
        self.m = self.v = None
        self.t = 0


class MMA:
    def __init__(self, move: float = 0.2, asyinit: float = 0.5, asyincr: float = 1.2, asydecr: float = 0.7,
                 max_inner: int = 8):
        self.move, self.asyinit, self.asyincr, self.asydecr = move, asyinit, asyincr, asydecr
        self.max_inner = max_inner
        self.reset()

    def reset(self) -> None:
        self.x1 = self.x2 = None  # previous iterates
        self.low = self.upp = None
        self.k = 0
        self.rho = None  # conservativeness, carried over between iterations (GCMMA)

    def state(self) -> dict[str, Any]:
        return {"x1": self.x1, "x2": self.x2, "low": self.low, "upp": self.upp, "k": self.k, "rho": self.rho}

    def load(self, st: dict[str, Any]) -> None:
        self.x1, self.x2, self.low, self.upp = st.get("x1"), st.get("x2"), st.get("low"), st.get("upp")
        self.k = int(st.get("k", 0))
        self.rho = None if st.get("rho") is None else float(st.get("rho"))

    def _asymptotes(self, x):
        rng = 1.0
        if self.k < 2 or self.x1 is None or self.x2 is None or self.low is None:
            low, upp = x - self.asyinit * rng, x + self.asyinit * rng
        else:
            sgn = (x - self.x1) * (self.x1 - self.x2)
            gam = np.where(sgn > 0, self.asyincr, np.where(sgn < 0, self.asydecr, 1.0))
            low = x - gam * (self.x1 - self.low)
            upp = x + gam * (self.upp - self.x1)
            low = np.clip(low, x - 10 * rng, x - 0.01 * rng)
            upp = np.clip(upp, x + 0.01 * rng, x + 10 * rng)
        return low, upp

    def step(self, x, f, g, evaluate: Evaluate):
        low, upp = self._asymptotes(x)
        alpha = np.maximum.reduce([np.zeros_like(x), low + 0.1 * (x - low), x - self.move])
        beta = np.minimum.reduce([np.ones_like(x), upp - 0.1 * (upp - x), x + self.move])
        gp, gm = np.maximum(g, 0), np.maximum(-g, 0)
        rho_min = 1e-5 * max(np.abs(g).mean(), 1e-12)
        rho = max(0.1 * self.rho, rho_min) if self.rho is not None else max(0.1 * np.abs(g).mean(), rho_min)
        cached = None
        for _ in range(self.max_inner):
            p = (upp - x) ** 2 * (1.001 * gp + 0.001 * gm + rho)
            q = (x - low) ** 2 * (0.001 * gp + 1.001 * gm + rho)
            sp, sq = np.sqrt(p), np.sqrt(q)
            y = np.clip((low * sp + upp * sq) / (sp + sq), alpha, beta)
            approx = f + np.sum(p / (upp - y) + q / (y - low)) - np.sum(p / (upp - x) + q / (x - low))
            fy, gy = evaluate(y)
            cached = (y, fy, gy)
            # Accept a conservative approximation (GCMMA), or any decrease of the objective;
            # both keep the iterates monotone.
            if fy <= approx + 1e-12 * abs(f) or fy < f:
                break
            dk = np.sum((upp - low) * (y - x) ** 2 / ((upp - y) * (y - low)))
            rho = min(10 * rho, 1.1 * (rho + (fy - approx) / max(dk, 1e-30)))
        y, fy, gy = cached
        if fy > f:  # never accept an increase (can only happen if the inner loop ran out)
            return x, (f, g)
        self.x2, self.x1 = self.x1, x.copy()
        self.low, self.upp = low, upp
        self.k += 1
        self.rho = rho
        return y, (fy, gy)
