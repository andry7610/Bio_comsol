import os
import sys
import time
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biological.grid3d import GridGraph3D
from biological.iterative_solver import solve_iterative

N_STEPS = 20
DT = 0.05


def run_wave(n=13, steps=N_STEPS, dt=DT):
    g = GridGraph3D(n, n, n)
    L = g.laplacian
    c = np.zeros(g.N)
    c[g.index(n // 2, n // 2, n // 2)] = 1.0
    m0 = c.sum()
    masses = [m0]
    variances = []
    iters = []
    I = sp.identity(g.N, format="csr")
    A = (I + dt * L).tocsr()  # неявный шаг: L = D - W (PSD), генератор = -L,
    # поэтому (I + dt L) c_{t+1} = c_t — M-матрица, держит позитивность
    ci, cy, cz = g.coords(np.arange(g.N))
    r2 = (ci - n // 2) ** 2 + (cy - n // 2) ** 2 + (cz - n // 2) ** 2
    for _ in range(steps):
        c, st = solve_iterative(A, c, method="cg", preconditioner="auto",
                                tol=1e-12)
        masses.append(c.sum())
        variances.append((c * r2).sum() / c.sum())
        iters.append(st["iterations"])
    return g, c, np.array(masses), np.array(variances), iters, dt


def test_wave_mass_conserved():
    g, c, masses, var, iters, dt = run_wave()
    assert np.abs(masses - masses[0]).max() < 1e-9


def test_wave_nonnegative():
    g, c, masses, var, iters, dt = run_wave()
    assert c.min() > -1e-12


def test_wave_second_moment_slope_near_2dD():
    g, c, masses, var, iters, dt = run_wave()
    t = np.arange(1, len(var) + 1) * dt
    slope = np.polyfit(t, var, 1)[0]
    # аналитика: dM/dt = 2*d*D = 6 при D=1
    assert abs(slope - 6.0) < 0.6
    print(f"[wave] slope={slope:.4f} (target 6.0)")


def test_wave_axis_symmetry():
    g, c, masses, var, iters, dt = run_wave()
    n = g.nx
    k = 3
    a = c[g.index(n // 2 + k, n // 2, n // 2)]
    b = c[g.index(n // 2, n // 2 + k, n // 2)]
    d = c[g.index(n // 2, n // 2, n // 2 + k)]
    assert abs(a - b) < 1e-9 and abs(a - d) < 1e-9


def test_wave_shell_monotonicity():
    g, c, masses, var, iters, dt = run_wave()
    n = g.nx
    ci, cy, cz = g.coords(np.arange(g.N))
    manh = (np.abs(ci - n // 2) + np.abs(cy - n // 2) + np.abs(cz - n // 2))
    means = [c[manh == k].mean() for k in range(9)]
    for k in range(len(means) - 1):
        assert means[k + 1] < means[k]


def test_wave_cg_iterations_bounded():
    g, c, masses, var, iters, dt = run_wave()
    assert max(iters) <= 20
