"""Beam propagation method (BPM) engine: FDE-consistent transverse operator, Padé/Crank-Nicolson
steps, and block-tridiagonal cyclic-reduction solves in the array API (NumPy CPU or torch CUDA)."""

from .engine import RunStats, Settings, port_fields, propagate
from .operator import Layout, Operator
from .pade import cn_factors, pade_rational, symbol
from .tridiag import BlockTridiag, block_matvec

__all__ = ["BlockTridiag", "Layout", "Operator", "RunStats", "Settings", "block_matvec", "cn_factors",
           "pade_rational", "port_fields", "propagate", "symbol"]
