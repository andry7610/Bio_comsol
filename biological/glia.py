
"""Глия как объёмный сток: неявный шаг (I + dt*D*L + dt*k*P) c_new = c_old.

Сток линеен по c -> Ньютон вырождается в один линейный шаг.
Маски - 3D (shape сетки), вектор c - плоский в порядке графа (C-order).
"""

import numpy as np
import scipy.sparse as sp

from grid3d import GridGraph3D


class GliaRegion:
    def __init__(self, mask3d, rate):
        self.mask = np.asarray(mask3d, dtype=bool)
        self.rate = float(rate)
        self.nodes = np.flatnonzero(self.mask.ravel())


def box_mask(g, lo, hi):
    """Полузамкнутый бокс [lo, hi) по каждой оси, в индексах сетки."""
    nx, ny, nz = g.shape
    ix = np.arange(nx)
    iy = np.arange(ny)
    iz = np.arange(nz)
    mx = (ix >= lo[0]) & (ix < hi[0])
    my = (iy >= lo[1]) & (iy < hi[1])
    mz = (iz >= lo[2]) & (iz < hi[2])
    return mx[:, None, None] & my[None, :, None] & mz[None, None, :]


def sphere_mask(g, center, radius):
    nx, ny, nz = g.shape
    ix = np.arange(nx)[:, None, None]
    iy = np.arange(ny)[None, :, None]
    iz = np.arange(nz)[None, None, :]
    d2 = (ix - center[0]) ** 2 + (iy - center[1]) ** 2 + (iz - center[2]) ** 2
    return d2 <= radius ** 2


def glia_step(L, c, dt, D, region):
    """Один неявный шаг с глиальным стоком. c - плоский вектор."""
    L = sp.csr_matrix(L)
    n = L.shape[0]
    c = np.asarray(c, dtype=float)
    k = region.rate
    Sdiag = np.zeros(n)
    Sdiag[region.nodes] = k
    A = sp.eye(n, format="csr") + dt * D * L + sp.diags(dt * Sdiag)
    rhs = c.copy()
    import scipy.sparse.linalg as spla
    if n <= SSOR_LIMIT_:
        x, _ = _solve(A, rhs, "ssor")
    else:
        x, _ = _solve(A, rhs, "jacobi")
    return x


SSOR_LIMIT_ = 40000


def _solve(A, b, pre):
    from iterative_solver import solve_iterative
    return solve_iterative(A, b, method="cg", preconditioner=pre, tol=1e-10)
