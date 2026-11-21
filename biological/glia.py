"""Глия: объёмный сток ионов. Маски 3D, вектор c — плоский (C-order графа)."""
import numpy as np
import scipy.sparse as sp
try:
    from .iterative_solver import (solve_iterative, cg_solve, ssor_preconditioner,
                                   jacobi_preconditioner, SSOR_NODE_LIMIT)
except ImportError:
    from iterative_solver import (solve_iterative, cg_solve, ssor_preconditioner,
                                  jacobi_preconditioner, SSOR_NODE_LIMIT)


def box_mask(shape, box):
    """box = ((x0,x1),(y0,y1),(z0,z1)) — включительно."""
    m = np.zeros(tuple(int(s) for s in shape), dtype=bool)
    (x0, x1), (y0, y1), (z0, z1) = box
    m[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True
    return m


def sphere_mask(shape, center, radius):
    """Евклидова сфера: d <= r (включительно)."""
    nx, ny, nz = (int(s) for s in shape)
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    d2 = (ii - center[0]) ** 2 + (jj - center[1]) ** 2 + (kk - center[2]) ** 2
    return d2 <= radius ** 2 + 1e-9


class GliaRegion:
    """Маска зоны глии + скорость потребления k. diagonal — индикатор маски;
    rate входит в систему один раз (в glia_step), двойного счёта нет."""

    def __init__(self, mask, rate=1.0):
        self.mask = np.asarray(mask, dtype=bool)
        self.rate = float(rate)

    @property
    def diagonal(self):
        return self.mask.astype(float).ravel()

    @property
    def nnodes(self):
        return int(self.mask.sum())


def _system_matrix(L, dt, glia):
    N = L.shape[0]
    return (sp.identity(N, format="csr") + dt * L + dt * glia.rate * sp.diags(glia.diagonal)).tocsr()


def glia_step(L, c, dt, glia, tol=1e-12, method="auto", preconditioner="auto"):
    """Один неявный шаг: (I + dt*L + dt*k*P) c_new = c_old."""
    c = np.asarray(c, dtype=float)
    A = _system_matrix(L, dt, glia)
    x, _st = solve_iterative(A, c, method=method, preconditioner=preconditioner, tol=tol)
    return x


def glia_series(L, c0, dt, glia, steps, tol=1e-10, return_trace=False):
    """Серия шагов с кэшем матрицы A и SSOR-предобуславливателя (факторизация один раз)."""
    c = np.asarray(c0, dtype=float).copy()
    A = _system_matrix(L, dt, glia)
    N = A.shape[0]
    if N <= SSOR_NODE_LIMIT:
        M = ssor_preconditioner(A)
    else:
        M = jacobi_preconditioner(A)
    trace = [c.copy()] if return_trace else None
    for _ in range(steps):
        if M is not None:
            c, _n, _r = cg_solve(A, c, M, tol=tol)
        else:
            c, _st = solve_iterative(A, c, method="cg", preconditioner="jacobi", tol=tol)
        if return_trace:
            trace.append(c.copy())
    return (c, trace) if return_trace else c
