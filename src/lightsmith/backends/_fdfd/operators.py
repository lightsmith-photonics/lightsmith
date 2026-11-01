"""The 2D frequency-domain Helmholtz operator as five diagonals.

Written against the array API, so the same code builds the operator from NumPy
arrays (plain solves) or torch tensors (autograd tracks eps -> operator).

With all lengths scaled by the cell size dl (so the operator is O(1)):

TM (Ez):  A = -(Dxb Sx Dxf + Dyb Sy Dyf) - (k0 dl)^2 eps
TE (Hz):  A = -(Dxb Sx eps_y^-1 Dxf + Dyb Sy eps_x^-1 Dyf) - (k0 dl)^2

where eps_y (eps_x) is the permittivity at the x-edges (y-edges), the arithmetic
mean of the two adjacent nodes (the Yee E-field positions), and S are the
SC-PML stretch factors. Row (i, j) of ``A x`` is

    c0 x[i,j] + cxm x[i-1,j] + cxp x[i+1,j] + cym x[i,j-1] + cyp x[i,j+1]

with x = 0 outside the grid. The flat index is ``i * Ny + j`` (C order).
"""

from __future__ import annotations

from typing import NamedTuple


class Diagonals(NamedTuple):
    c0: object
    cxm: object
    cxp: object
    cym: object
    cyp: object


def edge_coefficients(xp, eps, pol: str, s_edge, axis: int):
    """Coefficient on each edge along `axis`: (1/s) for TM, (1/(s eps_edge)) for TE. Shape (N+1, M) or (N, M+1)."""
    if axis == 0:
        inv_s = 1 / s_edge[:, None]
    else:
        inv_s = 1 / s_edge[None, :]
    if pol == "tm":
        return inv_s
    if axis == 0:
        padded = xp.concat([eps[:1, :], eps, eps[-1:, :]], axis=0)
        eps_edge = (padded[:-1, :] + padded[1:, :]) / 2
    else:
        padded = xp.concat([eps[:, :1], eps, eps[:, -1:]], axis=1)
        eps_edge = (padded[:, :-1] + padded[:, 1:]) / 2
    return inv_s / eps_edge


def helmholtz(xp, eps, pol: str, k0dl: float, sx: tuple, sy: tuple, masks: tuple) -> Diagonals:
    """Assemble the five diagonals (each (Nx, Ny)) of the dl^2-scaled operator.

    sx, sy: (s_node, s_edge) stretch factors as arrays of the namespace `xp`.
    masks: (mx_lo, mx_hi, my_lo, my_hi) 0/1 arrays zeroing couplings across the grid boundary.
    """
    sxn, sxe = sx
    syn, sye = sy
    mx_lo, mx_hi, my_lo, my_hi = masks
    ax = edge_coefficients(xp, eps, pol, sxe, 0)  # (Nx+1, Ny) or (Nx+1, 1)
    ay = edge_coefficients(xp, eps, pol, sye, 1)  # (Nx, Ny+1) or (1, Ny+1)
    inv_sxn = 1 / sxn[:, None]
    inv_syn = 1 / syn[None, :]
    cxp = -ax[1:, :] * inv_sxn * mx_hi
    cxm = -ax[:-1, :] * inv_sxn * mx_lo
    cyp = -ay[:, 1:] * inv_syn * my_hi
    cym = -ay[:, :-1] * inv_syn * my_lo
    c0 = (ax[1:, :] + ax[:-1, :]) * inv_sxn + (ay[:, 1:] + ay[:, :-1]) * inv_syn
    if pol == "tm":
        c0 = c0 - k0dl**2 * eps
    else:
        c0 = c0 - k0dl**2
    shape = tuple(eps.shape)
    return Diagonals(*(xp.broadcast_to(c, shape) if tuple(c.shape) != shape else c
                       for c in (c0, cxm, cxp, cym, cyp)))


def boundary_masks(xp, shape, like=None, device=None):
    nx, ny = shape
    import numpy as np

    mx_lo = np.ones((nx, 1)); mx_lo[0] = 0  # noqa: E702
    mx_hi = np.ones((nx, 1)); mx_hi[-1] = 0  # noqa: E702
    my_lo = np.ones((1, ny)); my_lo[0, 0] = 0  # noqa: E702
    my_hi = np.ones((1, ny)); my_hi[0, -1] = 0  # noqa: E702
    if device is None:
        return tuple(xp.asarray(m) for m in (mx_lo, mx_hi, my_lo, my_hi))
    return tuple(xp.asarray(m, device=device) for m in (mx_lo, mx_hi, my_lo, my_hi))


def matvec(xp, d: Diagonals, x):
    """A @ x for x of shape (Nx, Ny) or (Nx, Ny, k)."""
    extra = tuple(x.shape[2:])
    c = [v.reshape(tuple(v.shape) + (1,) * len(extra)) for v in d]
    zx = xp.zeros_like(x[:1])
    zy = xp.zeros_like(x[:, :1])
    x_im = xp.concat([zx, x[:-1]], axis=0)
    x_ip = xp.concat([x[1:], zx], axis=0)
    x_jm = xp.concat([zy, x[:, :-1]], axis=1)
    x_jp = xp.concat([x[:, 1:], zy], axis=1)
    return c[0] * x + c[1] * x_im + c[2] * x_ip + c[3] * x_jm + c[4] * x_jp
