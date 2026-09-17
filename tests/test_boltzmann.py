"""
Тест Больцмана: проверка, что стационарное распределение ионов
совпадает с аналитическим c(x) = c0 * exp(-z * phi(x)).

Использует внешний квадратичный потенциал (не Пуассон),
чтобы проверить миграционный член в чистом виде.
"""
import sys
from pathlib import Path

# Фикс путей для запуска из корня репозитория
ROOT = Path(__file__).resolve().parents
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


class ChainGraph:
    """1D-цепочка из N узлов. Плотный лапласиан."""
    def __init__(self, N):
        self.N = N
        W = np.zeros((N, N))
        for i in range(N - 1):
            W[i, i + 1] = 1.0
            W[i + 1, i] = 1.0
        degrees = np.sum(W, axis=1)
        self.L_dense = np.diag(degrees) - W

    @property
    def edges(self):
        return [(i, i + 1, 1.0) for i in range(self.N - 1)]

    @property
    def laplacian(self):
        return self.L_dense


def run_boltzmann_test():
    N = 30
    graph = ChainGraph(N)
    edges = graph.edges
    # L = np.asarray(graph.laplacian) # В этом тесте лапласиан не нужен напрямую

    # Квадратичный внешний потенциал: 0 на левом конце, 1.0 на правом
    x = np.arange(N, dtype=float)
    phi = (x / (N - 1)) ** 2 * 1.0

    ions = [
        {"name": "Na", "D": 0.1,  "z": 1,  "c0": 10.0},
        {"name": "Cl", "D": 0.15, "z": -1, "c0": 10.0},
    ]

    dt = 2.0
    F_RT = 1.0

    # Равномерные начальные концентрации
    c = np.array([np.full(N, ion["c0"]) for ion in ions])

    # Граничные условия Дирихле = больцмановские значения
    dirichlet_nodes = [0, N - 1]
    dirichlet_vals = {}
    for k in range(len(ions)):
        c_left = ions[k]["c0"] * np.exp(-ions[k]["z"] * phi * F_RT)
        c_right = ions[k]["c0"] * np.exp(-ions[k]["z"] * phi[-1] * F_RT)
        dirichlet_vals[k] = {0: c_left, N - 1: c_right}
        c[k, 0] = c_left
        c[k, -1] = c_right

    print("=== Boltzmann Equilibrium Test ===")
    print(f"Graph: {N} nodes chain, quadratic potential 0 -> {phi[-1]:.1f}")
    print(f"dt = {dt}")
    for k in range(len(ions)):
        print(f"  {ions[k]['name']}: c={dirichlet_vals[k]:.4f}, "
              f"c[-1]={dirichlet_vals[k][N-1]:.4f}")

    # Симуляция: backward Euler + миграция в матрице (implicit)
    for step in range(500000):
        c_prev = c.copy()

        for k in range(len(ions)):
            D = ions[k]["D"]
            z = ions[k]["z"]

            A = np.zeros((N, N))
            for (i, j, w) in edges:
                dphi = phi[j] - phi[i]

                # Диффузия
                A[i, i] -= dt * D * w
                A[i, j] += dt * D * w
                A[j, j] -= dt * D * w
                A[j, i] += dt * D * w

                # Миграция (знак исправлен)
                coef = dt * D * z * F_RT * 0.5 * dphi
                A[i, i] += coef
                A[i, j] += coef
                A[j, i] -= coef
                A[j, j] -= coef

            M = np.eye(N) - A
            b = c[k].copy()

            for node in dirichlet_nodes:
                M[node, :] = 0.0
                M[node, node] = 1.0
                b[node] = dirichlet_vals[k][node]

            c[k] = np.linalg.solve(M, b)

        drift = np.max(np.abs(c - c_prev))
        if step % 50000 == 0:
            print(f"  step {step}: drift={drift:.4e}")
        if drift < 1e-10:
            print(f"  Converged at step {step}")
            break

    # Сравнение с аналитическим Больцманом
    phi_0 = phi
    passed = True
    for k in range(len(ions)):
        c_anal = ions[k]["c0"] * np.exp(-ions[k]["z"] * (phi - phi_0) * F_RT)
        interior = slice(1, -1)
        rel_err = np.max(
            np.abs(c[k, interior] - c_anal[interior]) / c_anal[interior]
        )
        print(f"\n{ions[k]['name']} (z={ions[k]['z']:+.0f}):")
        print(f"  Sim:       [{c[k,1]:.4f} ... {c[k,N//2]:.4f} ... {c[k,-2]:.4f}]")
        print(f"  Boltzmann: [{c_anal:.4f} ... {c_anal[N//2]:.4f} ... {c_anal[-2]:.4f}]")
        print(f"  Max Rel Error (interior) = {rel_err:.4%}")
        if rel_err > 0.05:
            passed = False

    print("\n=== VERDICT ===")
    if passed:
        print("✅ TEST PASSED: Both ions within 5% of Boltzmann distribution.")
    else:
        print("❌ TEST FAILED: Distribution deviates beyond 5% tolerance.")


if __name__ == "__main__":
    run_boltzmann_test()
