import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

SSOR_NODE_LIMIT = 40000  # выше — auto переключается на jacobi (замечание dip: splu на 1003 жрёт память)


def _jacobi_precond(A):
    d = A.diagonal().copy()
    d[d == 0] = 1.0
    return spl.LinearOperator(A.shape, matvec=lambda x: x / d, dtype=float)


def _ssor_precond(A, omega=1.0):
    d = A.diagonal()
    D = sp.diags(d).tocsc()
    L = sp.tril(A, k=-1).tocsc()
    U = sp.triu(A, k=1).tocsc()
    luL = spl.splu((D + omega * L).tocsc())
    luU = spl.splu((D + omega * U).tocsc())
    scale = 1.0 / (2.0 - omega)

    def mv(r):
        return scale * luU.solve(D @ luL.solve(r))

    return spl.LinearOperator(A.shape, matvec=mv, dtype=float)


def _ilu_precond(A):
    ilu = spl.spilu(A.tocsc())
    return spl.LinearOperator(A.shape, matvec=ilu.solve, dtype=float)


def solve_direct(A, b):
    return spl.spsolve(sp.csc_matrix(A), np.asarray(b, float))


def _make_precond(A, preconditioner):
    if preconditioner == "ssor":
        return _ssor_precond(A)
    if preconditioner == "jacobi":
        return _jacobi_precond(A)
    if preconditioner == "ilu":
        return _ilu_precond(A)
    raise ValueError(f"unknown preconditioner: {preconditioner!r}")


def solve_iterative(A, b, method="cg", preconditioner="auto",
                    tol=1e-10, maxiter=20000):
    """CG (method='cg') или BiCGSTAB (method='bicgstab').
    preconditioner: 'auto' (ssor до SSOR_NODE_LIMIT, дальше jacobi),
    'ssor' (только CG — симметричный), 'ilu' (только BiCGSTAB), 'jacobi'.
    Возвращает (x, stats) со счётчиком итераций и относительным остатком.
    """
    A = sp.csr_matrix(A)
    b = np.asarray(b, float)
    if method not in ("cg", "bicgstab"):
        raise ValueError(f"unknown method: {method!r}")
    if preconditioner == "auto":
        preconditioner = "ssor" if A.shape[0] <= SSOR_NODE_LIMIT else "jacobi"
    if method == "cg" and preconditioner == "ilu":
        raise ValueError("CG requires a symmetric preconditioner; "
                         "ilu gives a nonsymmetric LU factor. "
                         "Use 'ssor' or 'jacobi' for CG, 'ilu' for bicgstab.")
    M = _make_precond(A, preconditioner)
    counter = [0]

    def cb(xk):
        counter[0] += 1

    if method == "cg":
        x, info = spl.cg(A, b, M=M, rtol=tol, maxiter=maxiter, callback=cb)
    else:
        x, info = spl.bicgstab(A, b, M=M, rtol=tol, maxiter=maxiter, callback=cb)
    if x is None:
        x = np.zeros_like(b)
    denom = np.linalg.norm(b) + 1e-300
    rel_res = float(np.linalg.norm(A @ x - b) / denom)
    stats = {"iterations": counter[0], "rel_residual": rel_res,
             "converged": rel_res <= 10 * tol, "preconditioner": preconditioner}
    return x, stats


def dirichlet_system(L, pins, values, b):
    """Двусторонняя элиминация Дирихле с сохранением симметрии.
    A = Dm L Dm + Dp (строки/столбцы пинов занулены, диагональ пина = 1),
    b' = b - L[:, pins] @ values, b'[pins] = values.
    Возвращает (A, b'); решение полной системы даёт x[pins] == values точно.
    """
    L = sp.csr_matrix(L, dtype=float)
    n = L.shape[0]
    pins = np.atleast_1d(np.asarray(pins, dtype=int))
    values = np.atleast_1d(np.asarray(values, dtype=float))
    b = np.asarray(b, dtype=float).copy()
    mask = np.ones(n, dtype=bool)
    mask[pins] = False
    Dm = sp.diags(mask.astype(float))
    Dp = sp.diags((~mask).astype(float))
    A = (Dm @ L @ Dm + Dp).tocsr()
    col_contrib = sp.csr_matrix(L)[:, pins] @ values
    b = b - col_contrib
    b[pins] = values
    return A, b


def pin_dirichlet(L, pins, values, b):
    """Обёртка обратной совместимости: то же, что dirichlet_system."""
    return dirichlet_system(L, pins, values, b)
