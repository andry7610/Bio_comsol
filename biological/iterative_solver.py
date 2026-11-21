"""Iterative solvers: CG (SSOR/Jacobi) and BiCGSTAB (ILU/Jacobi)."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, bicgstab, spsolve, splu, spilu, LinearOperator

SSOR_NODE_LIMIT = 40000


def _split(L):
    A = sp.csr_matrix(L)
    D = sp.diags(A.diagonal(), format="csr")
    Z = (A - D).tocsr()
    Z.eliminate_zeros()
    r, c = Z.nonzero()
    up = c > r
    lo = c < r
    U = sp.csr_matrix((Z.data[up], (r[up], c[up])), shape=Z.shape)
    Lo = sp.csr_matrix((Z.data[lo], (r[lo], c[lo])), shape=Z.shape)
    return D, Lo, U


def _ssor_pre(L, omega=1.0):
    """M = (1/w)(D+wL) D^-1 (D+wU);  M^-1 v = w (D+wU)^-1 D (D+wL)^-1 v."""
    D, Lo, U = _split(L)
    lu_lo = splu((D + omega * Lo).tocsc())
    lu_up = splu((D + omega * U).tocsc())
    ddiag = D.diagonal()

    def mv(v):
        t = lu_lo.solve(np.asarray(v, dtype=np.float64))
        t = ddiag * t
        t = lu_up.solve(t)
        return omega * t

    return LinearOperator(L.shape, matvec=mv, dtype=np.float64)


def _ilu_pre(A, drop_tol=1e-4, fill_factor=10):
    ilu = spilu(A.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    return LinearOperator(A.shape, matvec=ilu.solve, dtype=np.float64)


def _jacobi_pre(A):
    d = A.diagonal()
    d = np.where(np.abs(d) > 0, 1.0 / d, 1.0)

    def mv(v):
        return d * np.asarray(v, dtype=np.float64)

    return LinearOperator(A.shape, matvec=mv, dtype=np.float64)


def _make_pre(A, method, preconditioner):
    n = A.shape[0]
    p = preconditioner
    if p in (None, "none"):
        return None
    p = p.lower()
    if method == "cg":
        allowed = ("ssor", "jacobi", "auto")
    else:
        allowed = ("ilu", "jacobi", "auto")
    if p not in allowed:
        raise ValueError(
            f"preconditioner {preconditioner!r} is not allowed for {method!r}; "
            f"allowed: {allowed}"
        )
    if p == "auto":
        p = "ssor" if (method == "cg" and n <= SSOR_NODE_LIMIT) else (
            "ilu" if (method == "bicgstab" and n <= SSOR_NODE_LIMIT) else "jacobi")
    if p == "ssor":
        return _ssor_pre(A)
    if p == "ilu":
        return _ilu_pre(A)
    return _jacobi_pre(A)


def solve_iterative(A, b, method="cg", preconditioner="auto",
                    tol=1e-10, rtol=None, maxiter=None):
    """Solve A x = b iteratively. tol and rtol are aliases; stats keys
    'iters'/'iterations' are aliases too."""
    if rtol is not None:
        tol = rtol
    method = str(method).lower()
    if method not in ("cg", "bicgstab"):
        raise ValueError(f"unknown method {method!r}; allowed: 'cg', 'bicgstab'")
    A = sp.csr_matrix(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = _make_pre(A, method, preconditioner)
    stats = {"iters": 0, "iterations": 0, "residual": None,
             "info": None, "method": method, "preconditioner": preconditioner}

    def cb(xk):
        stats["iters"] += 1

    solver = cg if method == "cg" else bicgstab
    x, info = solver(A, b, M=M, rtol=tol,
                     maxiter=maxiter if maxiter is not None else 10 * A.shape[0],
                     callback=cb)
    stats["iters"] = int(stats["iters"])
    stats["iterations"] = stats["iters"]
    stats["info"] = int(info)
    res = np.linalg.norm(A @ x - b)
    nb = np.linalg.norm(b)
    stats["residual"] = float(res / nb) if nb > 0 else float(res)
    return x, stats


def solve_direct(A, b):
    return spsolve(sp.csr_matrix(A, dtype=np.float64),
                   np.asarray(b, dtype=np.float64))


def dirichlet_system(L, pins, values, b):
    """Dirichlet elimination that keeps symmetry: zero rows AND columns of
    pins, unit diagonal on pins, rhs adjusted: b_i -= sum_{j in pins} L_ij v_j,
    and b_pin = values."""
    A0 = sp.csr_matrix(L, dtype=np.float64)
    pins = np.asarray(pins, dtype=int)
    values = np.asarray(values, dtype=np.float64)
    coo = A0.tocoo()
    keep = ~np.isin(coo.row, pins) & ~np.isin(coo.col, pins)
    A = sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])),
                      shape=A0.shape)
    e = np.zeros(A0.shape[0])
    e[pins] = 1.0
    A = A + sp.diags(e, format="csr")
    b_pin = np.asarray(b, dtype=np.float64).copy()
    contrib = A0[:, pins] @ values
    interior = ~np.isin(np.arange(A0.shape[0]), pins)
    b_pin[interior] -= contrib[interior]
    b_pin[pins] = values
    return A, b_pin
