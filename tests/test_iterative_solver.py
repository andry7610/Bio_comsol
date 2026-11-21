
import importlib.util
import os
import numpy as np
import scipy.sparse as sp


def _load(name, rel):
    root = os.path.join(os.path.dirname(__file__), "..", "biological")
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, rel))
    return importlib.util.module_from_spec(spec), spec


_spec, _ = _load("_is_mod", "iterative_solver.py")
import sys as _sys
_sys.modules["_is_mod"] = _spec
_spec.loader.exec_module(_spec)
solve_iterative = _spec.solve_iterative
solve_direct = _spec.solve_direct
dirichlet_system = _spec.dirichlet_system
pin_dirichlet = _spec.pin_dirichlet

_spec_g, _ = _load("_g3_mod", "grid3d.py")
_sys.modules["_g3_mod"] = _spec_g
_spec_g.loader.exec_module(_spec_g)
GridGraph3D = _spec_g.GridGraph3D


def _poisson2d(n):
    from scipy.sparse import diags
    main = np.full(n * n, -4.0)
    off = np.ones(n * n - 1)
    off[n - 1::n] = 0.0
    A = diags([off, main, off], [-1, 0, 1]) + diags(np.ones(n * n - n), n) + diags(np.ones(n * n - n), -n)
    return None, A.tocsr()


def test_cg_matches_direct_and_analytics_2d():
    _, A = _poisson2d(10)
    rng = np.random.default_rng(42)
    b = rng.random(A.shape[0])
    x_d = solve_direct(A, b)
    x_c, st = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-12)
    assert np.max(np.abs(x_d - x_c)) < 1e-9


def test_dirichlet_linear_profile_exact_3d():
    g = GridGraph3D(6, 6, 6)
    L = g.laplacian
    lo = g.face_nodes(0, "lo")
    hi = g.face_nodes(0, "hi")
    pins = np.concatenate([lo, hi])
    vals = np.concatenate([np.zeros(len(lo)), -1.8 * np.ones(len(hi))])
    A, b = dirichlet_system(L, pins, vals, np.zeros(g.N))
    asym = abs(A - A.T)
    assert asym.nnz == 0 or np.abs(asym.data).max() < 1e-12
    x, st = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-12)
    assert np.max(np.abs(x[pins] - vals)) < 1e-12
    ix, iy, iz = g.coords(50)
    assert abs(x[g.index(ix, iy, iz)] - (-1.8 / 5 * ix)) < 1e-10


def test_bicgstab_ilu_on_nonsymmetric():
    _, A0 = _poisson2d(10)
    rng = np.random.default_rng(7)
    R = sp.random(A0.shape[0], A0.shape[0], density=0.05, random_state=7, format="csr")
    A = (A0 + 0.05 * (R - R.T)).tocsr()
    b = rng.random(A.shape[0])
    x_d = solve_direct(A, b)
    x_i, st = solve_iterative(A, b, method="bicgstab", preconditioner="ilu", tol=1e-10)
    assert np.max(np.abs(x_d - x_i)) < 1e-8


def test_single_pin_cures_neumann_singularity():
    n = 20
    diag = np.full(n, -2.0)
    diag[0] = diag[-1] = -1.0
    L = sp.diags(diag) + sp.diags(np.ones(n - 1), 1) + sp.diags(np.ones(n - 1), -1)
    A, b = dirichlet_system(L.tocsr(), [0], [7.0], np.zeros(n))
    x, st = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-12)
    assert np.allclose(x, 7.0, atol=1e-10)


def test_ssor_reduces_cg_iterations_vs_jacobi():
    _, A = _poisson2d(16)
    rng = np.random.default_rng(3)
    b = rng.random(A.shape[0])
    _, st_j = solve_iterative(A, b, method="cg", preconditioner="jacobi", tol=1e-10)
    _, st_s = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-10)
    assert st_s["iterations"] < st_j["iterations"]


def test_ilu_reduces_bicgstab_iterations_vs_jacobi():
    _, A0 = _poisson2d(16)
    rng = np.random.default_rng(11)
    R = sp.random(A0.shape[0], A0.shape[0], density=0.05, random_state=11, format="csr")
    A = (A0 + 0.05 * (R - R.T)).tocsr()
    b = rng.random(A.shape[0])
    _, st_j = solve_iterative(A, b, method="bicgstab", preconditioner="jacobi", tol=1e-10)
    _, st_i = solve_iterative(A, b, method="bicgstab", preconditioner="ilu", tol=1e-10)
    assert st_i["iterations"] < st_j["iterations"]


def test_unknown_method_raises_valueerror():
    try:
        solve_iterative(sp.eye(5), np.ones(5), method="sorcery")
        raise AssertionError("must raise")
    except ValueError:
        pass


def test_feasibility_3d_cube_20():
    import time
    g = GridGraph3D(20, 20, 20)
    L = g.laplacian
    lo = g.face_nodes(0, "lo")
    hi = g.face_nodes(0, "hi")
    pins = np.concatenate([lo, hi])
    vals = np.concatenate([np.zeros(len(lo)), -1.8 * np.ones(len(hi))])
    A, b = dirichlet_system(L, pins, vals, np.zeros(g.N))
    t0 = time.perf_counter()
    x, st = solve_iterative(A, b, method="cg", preconditioner="ssor", tol=1e-10)
    dt_ = time.perf_counter() - t0
    assert dt_ < 30.0
    assert np.max(np.abs(x[pins] - vals)) < 1e-10
