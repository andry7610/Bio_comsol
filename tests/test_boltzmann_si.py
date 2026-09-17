"""
Тест Больцмана v2 — с физическими единицами СИ.
Проверка: стационарное распределение ионов
совпадает с c(x) = c0 * exp(-z * F/RT * phi(x)).

Реальные параметры: Na+, K+, Cl- при T = 310 K.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def run_boltzmann_test_si():
    N = 30
    graph = ChainGraph(N)
    edges = graph.edges

    # --- Физические константы (СИ) ---
    T = 310.0
    R = 8.314462618
    F = 96485.33212
    F_RT = F / (R * T)  # ≈ 38.94 1/В
    V_T = R * T / F      # ≈ 25.69 мВ

    # Внешний потенциал: 0 В слева, 50 мВ справа
    x = np.arange(N, dtype=float)
    phi_max = 0.050  # 50 мВ
    phi = (x / (N - 1)) ** 2 * phi_max

    # --- Ионы в СИ ---
    ions = [
        {"name": "Na", "D": 1.33e-9, "z": 1,  "c0": 145.0},
        {"name": "K",  "D": 1.96e-9, "z": 1,  "c0": 4.0},
        {"name": "Cl", "D": 2.03e-9, "z": -1, "c0": 110.0},
    ]

    # --- Безразмерная нормализация ---
    # c* = c / c_ref, phi* = phi * F/RT, t* = t * D_ref / L_ref^2
    c_ref = 145.0  # опорная концентрация (Na+)
    phi_star = phi * F_RT  # безразмерный потенциал

    dt_star = 2.0  # безразмерный шаг (аналог v0.1)

    print("=== Boltzmann Equilibrium Test (SI) ===")
    print(f"T = {T} K, F/RT = {F_RT:.2f} 1/V, V_T = {V_T*1000:.2f} mV")
    print(f"Potential: 0 -> {phi_max*1000:.0f} mV")
    print(f"Ions: {[ion['name'] for ion in ions]}")
    print(f"c_ref = {c_ref} mM")
    print()

    # Безразмерные концентрации
    c = np.array([np.full(N, ion["c0"] / c_ref) for ion in ions])

    # Граничные условия Дирихле = больцмановские значения
    dirichlet_nodes = [0, N - 1]
    dirichlet_vals = {}
    for k in range(len(ions)):
        z = ions[k]["z"]
        c_left = (ions[k]["c0"] / c_ref) * np.exp(-z * phi_star[0])
        c_right = (ions[k]["c0"] / c_ref) * np.exp(-z * phi_star[-1])
        dirichlet_vals[k] = {0: c_left, N - 1: c_right}
        c[k, 0] = c_left
        c[k, -1] = c_right

    for k in range(len(ions)):
        print(f"  {ions[k]['name']}: c*[0]={c[k,0]:.6f}, "
              f"c*[-1]={c[k,-1]:.6f}  (c0={ions[k]['c0']} mM)")

    # --- Симуляция: backward Euler (безразмерный) ---
    for step in range(500000):
        c_prev = c.copy()

        for k in range(len(ions)):
            D_star = ions[k]["D"] / ions[0]["D"]  # нормировка D к D_Na
            z = ions[k]["z"]

            A = np.zeros((N, N))
            for (i, j, w) in edges:
                dphi = phi_star[j] - phi_star[i]

                # Диффузия
                A[i, i] -= dt_star * D_star * w
                A[i, j] += dt_star * D_star * w
                A[j, j] -= dt_star * D_star * w
                A[j, i] += dt_star * D_star * w

                # Миграция (знак исправлен в v0.3)
                coef = dt_star * D_star * z * 0.5 * dphi
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

    # --- Сравнение с аналитическим Больцманом ---
    phi_0_star = phi_star[0]
    passed = True
    for k in range(len(ions)):
        c_anal_star = (ions[k]["c0"] / c_ref) * \
            np.exp(-ions[k]["z"] * (phi_star - phi_0_star))
        interior = slice(1, -1)
        rel_err = np.max(
            np.abs(c[k, interior] - c_anal_star[interior]) / c_anal_star[interior]
        )
        # Обратный перевод в мМ для наглядности
        c_sim_mM = c[k] * c_ref
        c_anal_mM = c_anal_star * c_ref
        print(f"\n{ions[k]['name']} (z={ions[k]['z']:+.0f}, D={ions[k]['D']:.2e} m²/s):")
        print(f"  Sim (mM):       [{c_sim_mM[1]:.3f} ... "
              f"{c_sim_mM[N//2]:.3f} ... {c_sim_mM[-2]:.3f}]")
        print(f"  Boltzmann (mM): [{c_anal_mM[1]:.3f} ... "
              f"{c_anal_mM[N//2]:.3f} ... {c_anal_mM[-2]:.3f}]")
        print(f"  Max Rel Error (interior) = {rel_err:.4%}")
        if rel_err > 0.05:
            passed = False

    print("\n=== VERDICT ===")
    if passed:
        print("✅ TEST PASSED: All ions within 5% of Boltzmann distribution (SI).")
    else:
        print("❌ TEST FAILED: Distribution deviates beyond 5% tolerance.")


if __name__ == "__main__":
    run_boltzmann_test_si()
