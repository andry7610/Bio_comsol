"""Glia region: spatial sink zone for ion diffusion (self-contained).

No imports from iterative_solver or grid3d - internal CG + SSOR lives here.
Grid convention: 3D mask (i, j, k), flat vector in C-order (x slowest).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def box_mask(shape, lo, hi):
    """Rectangular glia region, inclusive bounds, 3D mask."""
    shape = tuple(shape)
    lo = tuple(max(0, int(l)) for l in lo)
    hi = tuple(min(s - 1, int(h)) for s, h in zip(shape, hi))
    m = np.zeros(shape, dtype=float)
    m[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = 1.0
    return m


def sphere_mask(shape, center, radius):
    """Spherical glia region inside tissue spheroid, 3D mask."""
    shape = tuple(shape)
    grids = np.indices(shape, dtype=float)
    r2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    return (r2 <= float(radius) ** 2).astype(float)


class GliaRegion:
    """Glia zone with per-node consumption rate k (sink term -k*c)."""

    def __init__(self, mask, rate=1.0):
        self.mask = np.asarray(mask, dtype=float)
        self.rate = float(rate)
        self.n_nodes = int(self.mask.size)

    def diagonal(self):
        """Flat vector: rate*k inside glia, 0 outside (for sp.diags)."""
        return (self.mask.ravel() * self.rate)

    def count(self):
        return int(round(self.mask.sum()))


def _ssor_pre(A, omega=1.0):
    """Symmetric SSOR preconditioner: M^-1 = w*(D+wU)^-1 D (D+wL)^-1."""
    D = A.diagonal()
    Dsafe = np.where(np.abs(D) > 1e-14, D, 1.0)
    U = sp.triu(A, k=1, format="csr")
    L = sp.tril(A, k=-1, format="csr")
    DU1 = sp.diags(Dsafe) + omega * U
    DU2 = sp.diags(Dsafe) + omega * L
    lu1 = spla.splu(DU1.tocsc())
    lu2 = spla.splu(DU2.tocsc())

    def mv(v):
        z = lu2.solve(v)
        z = z * Dsafe
        z = lu1.solve(z)
        return omega * z

    n = A.shape[0]
    return spla.LinearOperator((n, n), matvec=mv, dtype=float)


def _cg(A, b, M, x0=None, tol=1e-8, maxiter=None):
    """Plain CG with callback iteration counter. A must be SPD."""
    n = A.shape[0]
    maxiter = maxiter or 10 * n
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    r = b - A @ x
    p = r.copy()
    rs = r @ r
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return x, {"iters": 0, "iterations": 0, "residual": 0.0, "info": 0}
    iters = 0
    for _ in range(maxiter):
        Ap = A @ p
        alpha = rs / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        iters += 1
        if np.sqrt(rs_new) <= tol * b_norm:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, {"iters": iters, "iterations": iters,
               "residual": float(np.sqrt(rs)), "info": 0}


def _solve(A, b, tol=1e-8, auto_threshold=40000):
    """auto: SSOR for small grids, plain CG (jacobi) for big ones."""
    n = A.shape[0]
    if n <= auto_threshold:
        M = _ssor_pre(A)
        return _cg(A, b, M, tol=tol)
    D = A.diagonal()
    Dsafe = np.where(np.abs(D) > 1e-14, D, 1.0)
    Mj = spla.LinearOperator((n, n), matvec=lambda v: v / Dsafe, dtype=float)
    return _cg(A, b, Mj, tol=tol)


def glia_step(L, c_old, glia, dt, method="auto", tol=1e-9, info=None):
    """One implicit glia-diffusion step.

    Solves (I + dt*D*L + dt*k*P) c_new = c_old  where P = diag(glia mask).
    Sink is linear in c, so one linear solve per step (Newton degenerates).
    """
    n = glia.n_nodes
    kP = sp.diags(glia.diagonal(), format="csr")
    A = (sp.identity(n, format="csr")
         + dt * L.tocsr() + dt * kP).tocsr()
    # symmetry guard: A must equal A.T up to fp noise
    c_new, st = _solve(A, np.asarray(c_old, dtype=float), tol=tol)
    if info is not None:
        info.update(st)
    return c_new, st


def glia_series(L, c0, glia, dt, steps, tol=1e-9, collect=None):
    """Run a series of glia steps with ONE cached matrix + preconditioner."""
    n = glia.n_nodes
    kP = sp.diags(glia.diagonal(), format="csr")
    A = (sp.identity(n, format="csr") + dt * L.tocsr() + dt * kP).tocsr()
    c = np.asarray(c0, dtype=float).copy()
    hist = []
    if collect is not None:
        hist.append((0.0, collect(c)))
    n_small = n <= 40000
    M = _ssor_pre(A) if n_small else None
    D = A.diagonal()
    Dsafe = np.where(np.abs(D) > 1e-14, D, 1.0)
    Mj = spla.LinearOperator((n, n), matvec=lambda v: v / Dsafe, dtype=float)
    for s in range(1, steps + 1):
        c, st = _cg(A, c, M if n_small else Mj, tol=tol)
        if collect is not None:
            hist.append((s * dt, collect(c)))
    return c, hist
