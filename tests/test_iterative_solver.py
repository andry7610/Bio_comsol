import sys
import pathlib
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from biological.grid3d import GridGraph3D
from biological.iterative_solver import solve_iterative, solve_direct, dirichlet_system


def _poisson2d(n):
    g = GridGraph3D(n, n, 1)
    return g, g.laplacian


def test_cg_matches_direct_and_analytics_2d():
    g, L = _poisson2d(10)
    rng = np.random.default_rng(42)
    b0 = rng.random(g.N)
    # pure Neumann laplacian is singular (kernel = constants): pin one node -> SPD
    A, b = dirichlet_system(L, [0], [0.0], b0)
    x_d = solve_direct(A, b)
    x_c, st = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-12)
    x_j, _ = solve_iterative(A, b, method="cg", preconditioner="jacobi", tol=1e-12)
    assert np.allclose(x_c, x_d, atol=1e-9)
    assert np.allclose(x_j, x_d, atol=1e-9)
    assert st["iters"] > 0


def test_dirichlet_linear_profile_exact_3d():
    g = GridGraph3D(6, 6, 6)
    lo = g.face_nodes(0, "lo")
    hi = g.face_nodes(0, "hi")
    pins = np.concatenate([lo, hi])
    vals = np.concatenate([np.zeros(len(lo)), -1.8 * np.ones(len(hi))])
    A, b = dirichlet_system(g.laplacian, pins, vals, np.zeros(g.N))
    asym = abs(A - A.T)
    assert asym.nnz == 0 or np.abs(asym.data).max() < 1e-12
    x, st = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-12)
    slope = -1.8 / (6 - 1)  # honest slope from pin values
    i = np.arange(g.N) // (g.ny * g.nz)
    assert np.allclose(x, slope * i, atol=1e-9)


def test_bicgstab_ilu_on_nonsymmetric():
    g, A0 = _poisson2d(10)
    rng = np.random.default_rng(7)
    R = sp.random(g.N, g.N, density=0.05, random_state=7, format="csr")
    S = R - R.T  # skew-symmetric perturbation: Picard territory
    A = (A0 + 0.05 * S).tocsr()
    b = rng.random(g.N)
    x_d = solve_direct(A, b)
    x_i, st = solve_iterative(A, b, method="bicgstab", preconditioner="ilu", tol=1e-10)
    assert np.allclose(x_i, x_d, atol=1e-8)


def test_single_pin_cures_neumann_singularity():
    n = 20
    diag = np.full(n, -2.0)
    diag[0] = diag[-1] = -1.0
    L = (sp.diags(diag) + sp.diags(np.ones(n - 1), 1)
         + sp.diags(np.ones(n - 1), -1)).tocsr()
    A, b = dirichlet_system(L, [0], [7.0], np.zeros(n))
    x, st = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-12)
    assert np.allclose(x, 7.0, atol=1e-9)


def test_ssor_reduces_cg_iterations_vs_jacobi():
    g, L = _poisson2d(16)
    rng = np.random.default_rng(3)
    b0 = rng.random(g.N)
    A, b = dirichlet_system(L, [0], [0.0], b0)  # pin -> SPD
    _, st_j = solve_iterative(A, b, method="cg", preconditioner="jacobi", tol=1e-10)
    _, st_s = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-10)
    assert st_s["iters"] < st_j["iters"]


def test_ilu_reduces_bicgstab_iterations_vs_jacobi():
    g, A0 = _poisson2d(16)
    rng = np.random.default_rng(11)
    R = sp.random(g.N, g.N, density=0.05, random_state=11, format="csr")
    A = (A0 + 0.05 * (R - R.T)).tocsr()
    b = rng.random(g.N)
    _, st_j = solve_iterative(A, b, method="bicgstab", preconditioner="jacobi", tol=1e-10)
    _, st_i = solve_iterative(A, b, method="bicgstab", preconditioner="ilu", tol=1e-10)
    assert st_i["iters"] < st_j["iters"]


def test_unknown_method_raises_valueerror():
    g, A = _poisson2d(5)
    b = np.ones(g.N)
    for bad in ("sor", "gauss", "qr"):
        try:
            solve_iterative(A, b, method=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should raise ValueError")
    try:
        solve_iterative(A, b, method="cg", preconditioner="ilu")
    except ValueError:
        pass
    else:
        raise AssertionError("cg+ilu should raise ValueError")


def test_feasibility_3d_cube_20():
    g = GridGraph3D(20, 20, 20)
    lo = g.face_nodes(0, "lo")
    hi = g.face_nodes(0, "hi")
    pins = np.concatenate([lo, hi])
    vals = np.concatenate([np.zeros(len(lo)), -1.8 * np.ones(len(hi))])
    A, b = dirichlet_system(g.laplacian, pins, vals, np.zeros(g.N))
    t0 = time.perf_counter()
    x, st = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-10)
    elapsed = time.perf_counter() - t0
    assert st["iters"] <= 200
    assert st["residual"] < 1e-8
    assert elapsed < 10.0
