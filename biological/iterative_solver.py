
"""Итерационный солвер: CG+SSOR (SPD), BiCGSTAB+ILU (несимметричные),
решение Дирихле через двустороннюю элиминацию A_pin = P A P + B.

auto-предобуславливатель: SSOR до SSOR_NODE_LIMIT узлов, дальше Жакоби.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SSOR_NODE_LIMIT = 40000


def _ssor_pre(A, omega=1.0):
    n = A.shape[0]
    D = A.diagonal()
    L = sp.tril(A, k=-1).tocsr()
    U = sp.triu(A, k=1).tocsr()
    LU1 = spla.splu((L + sp.diags(D)).tocsc())
    LU2 = spla.splu((omega * U + sp.diags(D)).tocsc())  # D + omega*U: верхний треугольник!

    def mv(r):
        z1 = LU1.solve(r)
        z2 = LU2.solve(z1)
        return (1.0 - omega) * r + omega * z2

    return spla.LinearOperator((n, n), matvec=mv, dtype=A.dtype)


def _jacobi_pre(A):
    n = A.shape[0]
    d = A.diagonal()
    if np.any(np.abs(d) < 1e-300):
        raise ValueError("zero diagonal: jacobi preconditioner undefined")
    return spla.LinearOperator((n, n), matvec=lambda r: r / d, dtype=A.dtype)


def _ilu_pre(A):
    ilu = spla.spilu(A.tocsc())
    n = A.shape[0]
    return spla.LinearOperator((n, n), matvec=ilu.solve, dtype=A.dtype)


def solve_iterative(A, b, method="cg", preconditioner="none", tol=None,
                    rtol=None, maxiter=10000):
    """method: cg | bicgstab. preconditioner: none | jacobi | ssor | ilu | auto.

    Совместимость: параметр может прийти как tol=... или rtol=..."""
    if rtol is None:
        rtol = tol if tol is not None else 1e-10
    A = sp.csr_matrix(A)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]

    if method not in ("cg", "bicgstab"):
        raise ValueError(f"unknown method: {method!r}")

    if preconditioner == "auto":
        preconditioner = "ssor" if n <= SSOR_NODE_LIMIT else "jacobi"

    if method == "cg":
        if not _is_symmetric(A):
            raise ValueError("CG requires symmetric matrix")
        if preconditioner == "ilu":
            raise ValueError("ILU is unsymmetric: use CG with ssor/jacobi")
        pre = {
            "none": None,
            "jacobi": lambda: _jacobi_pre(A),
            "ssor": lambda: _ssor_pre(A),
        }[preconditioner]()
        counter = {"n": 0}
        x, info = spla.cg(A, b, M=pre, rtol=rtol, maxiter=maxiter,
                          callback=lambda xk: counter.__setitem__("n", counter["n"] + 1))
    else:
        if preconditioner not in ("none", "jacobi", "ilu", "auto"):
            raise ValueError(f"unknown preconditioner: {preconditioner!r}")
        pre = {
            "none": None,
            "jacobi": lambda: _jacobi_pre(A),
            "ilu": lambda: _ilu_pre(A),
        }[preconditioner]()
        counter = {"n": 0}
        x, info = spla.bicgstab(A, b, M=pre, rtol=rtol, maxiter=maxiter,
                                callback=lambda xk: counter.__setitem__("n", counter["n"] + 1))

    if info != 0:
        raise RuntimeError(f"solver did not converge, info={info}")
    res = np.linalg.norm(b - A @ x) / max(np.linalg.norm(b), 1e-300)
    return x, {"iterations": counter["n"], "residual": res,
               "method": method, "preconditioner": preconditioner}


def _is_symmetric(A, tol=1e-12):
    As = abs(A - A.T)
    return As.nnz == 0 or np.abs(As.data).max() < tol


def solve_direct(A, b):
    return spla.spsolve(sp.csr_matrix(A), np.asarray(b, dtype=float))


def pin_dirichlet(L, pins, values):
    """Двусторонняя элиминация: строки И столбцы пинов заземляются,
    диагональ пинов = 1 (симметрия сохраняется).

    Внимание: правую часть корректно собирает dirichlet_system."""
    L = sp.csr_matrix(L)
    n = L.shape[0]
    pins = np.asarray(pins, dtype=int)
    values = np.asarray(values, dtype=float)
    keep = np.ones(n, dtype=bool)
    keep[pins] = False
    P = sp.eye(n, format="csr").multiply(keep[:, None])
    A = (P @ L @ P).tocsr()
    B = sp.eye(n, format="csr") - P
    A = A + B  # на пинах A_ii = 1
    return A


def dirichlet_system(L, pins, values, b=None):
    """Возвращает (A_pin, b_pin) корректной системы после элиминации:
    строки и столбцы пинов заземлены, b_pin = b - L[:, pins] @ values."""
    L = sp.csr_matrix(L)
    n = L.shape[0]
    pins = np.asarray(pins, dtype=int)
    values = np.asarray(values, dtype=float)
    A = pin_dirichlet(L, pins, values)
    if b is None:
        b = np.zeros(n)
    b = np.asarray(b, dtype=float).copy()
    col = sp.csr_matrix(L)[:, pins] @ values
    b_pin = b - col
    b_pin[pins] = values  # строка A_pin i: 1*x_i = value
    return A, b_pin
