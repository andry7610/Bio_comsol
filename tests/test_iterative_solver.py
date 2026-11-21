import os
import sys
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biological.grid3d import GridGraph3D
from biological.iterative_solver import (solve_iterative, solve_direct,
                                         dirichlet_system, SSOR_NODE_LIMIT)


def _poisson2d(n):
    g = GridGraph3D(n, n, 1)
    return g, (g.laplacian + 0.1 * sp.identity(g.N)).tocsr()


def test_cg_matches_direct_and_analytics_2d():
    g, A = _poisson2d(10)
    rng = np.random.default_rng(42)
    b = rng.random(g.N)
    x_d = solve_direct(A, b)
    x_c, st = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-12)
    assert np.abs(x_c - x_d).max() < 1e-8
    assert st["converged"]
    # аналитика: A x = b, остаток прямого решения — машинная точность
    assert np.abs(A @ x_d - b).max() < 1e-9


def test_dirichlet_linear_profile_exact_3d():
    g = GridGraph3D(6, 6, 6)
    lo = g.face_nodes(0, "lo")
    hi = g.face_nodes(0, "hi")
    pins = np.concatenate([lo, hi])
    vals = np.concatenate([np.zeros(len(lo)), -1.8 * np.ones(len(hi))])
    A, b = dirichlet_system(g.laplacian, pins, vals, np.zeros(g.N))
    # проверка симметрии
    asym = abs(A - A.T)
    assert asym.nnz == 0 or np.abs(asym.data).max() < 1e-12
    x, st = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-12)
    xi, yi, zi = g.coords(np.arange(g.N))
    analytic = -1.8 * xi / (g.nx - 1)
    assert np.abs(x - analytic).max() < 1e-7
    assert np.abs(x[pins] - vals).max() < 1e-12


def test_bicgstab_ilu_on_nonsymmetric():
    g, A0 = _poisson2d(10)
    rng = np.random.default_rng(7)
    R = sp.random(g.N, g.N, density=0.05, random_state=7, format="csr")
    S = R - R.T  # кососимметричное возмущение — территория Пикара
    A = (A0 + 0.05 * S).tocsr()
    b = rng.random(g.N)
    x_d = solve_direct(A, b)
    x_i, st = solve_iterative(A, b, method="bicgstab", preconditioner="ilu", tol=1e-10)
    assert np.abs(x_i - x_d).max() < 1e-7
    assert st["converged"]


def test_single_pin_cures_neumann_singularity():
    n = 20
    diag = np.full(n, -2.0)
    diag[0] = diag[-1] = -1.0
    L = sp.diags(diag) + sp.diags(np.ones(n - 1), 1) + sp.diags(np.ones(n - 1), -1)
    A, b = dirichlet_system(L, [0], [7.0], np.zeros(n))
    x, st = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-12)
    assert np.abs(x - 7.0).max() < 1e-9


def test_ssor_reduces_cg_iterations_vs_jacobi():
    g, A = _poisson2d(16)
    rng = np.random.default_rng(3)
    b = rng.random(g.N)
    _, st_j = solve_iterative(A, b, method="cg", preconditioner="jacobi", tol=1e-10)
    _, st_s = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-10)
    assert st_s["iterations"] < st_j["iterations"]


def test_ilu_reduces_bicgstab_iterations_vs_jacobi():
    g, A0 = _poisson2d(16)
    rng = np.random.default_rng(11)
    R = sp.random(g.N, g.N, density=0.05, random_state=11, format="csr")
    A = (A0 + 0.05 * (R - R.T)).tocsr()
    b = rng.random(g.N)
    _, st_j = solve_iterative(A, b, method="bicgstab", preconditioner="jacobi", tol=1e-10)
    _, st_i = solve_iterative(A, b, method="bicgstab", preconditioner="ilu", tol=1e-10)
    assert st_i["iterations"] < st_j["iterations"]


def test_unknown_method_raises_valueerror():
    g, A = _poisson2d(6)
    try:
        solve_iterative(A, np.zeros(g.N), method="sorcery")
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError not raised")


def test_feasibility_3d_cube_20():
    import time
    g = GridGraph3D(20, 20, 20)
    lo = g.face_nodes(0, "lo")
    hi = g.face_nodes(0, "hi")
    pins = np.concatenate([lo, hi])
    vals = np.concatenate([np.zeros(len(lo)), -1.8 * np.ones(len(hi))])
    A, b = dirichlet_system(g.laplacian, pins, vals, np.zeros(g.N))
    t0 = time.perf_counter()
    x, st = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-10)
    dt_s = time.perf_counter() - t0
    xi, yi, zi = g.coords(np.arange(g.N))
    analytic = -1.8 * xi / (g.nx - 1)
    assert st["converged"]
    assert st["iterations"] <= 200
    assert np.abs(x - analytic).max() < 1e-6
    print(f"[feasibility 20^3] nodes={g.N} iters={st['iterations']} "
          f"time={dt_s:.2f}s rel_res={st['rel_residual']:.2e}")
