"""
tests/test_validation.py

Валидация Bio_comsol против аналитических решений.
Три теста:
  1. Равновесие Нернста — Больцмановский профиль
  2. 1D-диффузия (erf-решение + проверка порядка сходимости)
  3. Больцмановское распределение (R² + наклон)

Все тесты используют безразмерные параметры: D=1.0, F_RT=1.0.
Это проверяет численную схему, а не биологию.
"""

import numpy as np
from scipy.special import erf
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from network.graph import Graph
from biological.diffusion import BiologicalDiffusion


# ============================================================
# Вспомогательные функции
# ============================================================

def make_chain(N, L=1.0, x0=0.0, cross_section=1.0, conductivity=1.0):
    """Создать цепочку из N узлов длиной L с заданным сечением."""
    x = np.linspace(x0, x0 + L, N)
    pos = np.column_stack((x, np.zeros_like(x)))
    return Graph(
        N, topology="chain", pos=pos,
        conductivity=conductivity, cross_section=cross_section,
    )


def compute_D_eff(D, graph):
    """Эффективный коэффициент диффузии.

    D_eff = D * conductivity * cross_section * dx,
    где dx = L / (N-1) — шаг сетки.
    """
    dx = graph.pos[1, 0] - graph.pos[0, 0]
    return D * graph.conductivity * graph.cross_section * dx


def run_to_steady(bio, tol=1e-12, max_steps=50000):
    """Запустить до стационара. Возвращает число шагов."""
    for i in range(max_steps):
        c_before = bio.c.copy()
        bio.step()
        dc = np.max(np.abs(bio.c - c_before))
        if dc < tol:
            return i + 1
    return max_steps


def make_graph_with_D_eff(N, L, D, D_eff_target=1.0, conductivity=1.0):
    """Создать граф с заданным D_eff."""
    dx = L / (N - 1)
    cross_section = D_eff_target / (D * conductivity * dx)
    graph = make_chain(N, L=L, cross_section=cross_section, conductivity=conductivity)
    D_eff = compute_D_eff(D, graph)
    assert abs(D_eff - D_eff_target) < 0.01, f"D_eff={D_eff}, expected {D_eff_target}"
    return graph, D_eff


# ============================================================
# Тест 1: Равновесие Нернста
# ============================================================

def test_nernst_equilibrium():
    """
    При мембранном потенциале, равном Нернстовскому, система
    выходит на стационар с Больцмановским профилем концентрации.

    Boltzmann: c(x) = c_left * exp(-z * F_RT * (phi(x) - phi(0)))
    => membrane_potential = -(1/(z*F_RT)) * ln(c_right / c_left)

    Допуск 1e-3 — ошибка центрированной дискретизации O(dx^2).
    """
    F_RT = 1.0
    z = 1
    c_left = 1.0
    c_right = 10.0

    # Нернстовский потенциал
    membrane_potential = -(1.0 / (z * F_RT)) * np.log(c_right / c_left)

    N = 50
    L = 1.0
    D = 1.0
    graph, D_eff = make_graph_with_D_eff(N, L, D, D_eff_target=1.0)

    ions = [{
        "name": "K",
        "D": D,
        "z": z,
        "c0": c_left,
        "c_left": c_left,
        "c_right": c_right,
    }]

    dt = 0.01
    bio = BiologicalDiffusion(
        graph, ions, dt=dt, F_RT=F_RT,
        membrane_potential=membrane_potential,
        self_consistent=False,
        use_newton=True,
    )

    # Прогон до стационара
    n_steps = run_to_steady(bio, tol=1e-12, max_steps=50000)

    # Проверка 1: стационар достигнут
    c_before = bio.c.copy()
    bio.step()
    dc = np.max(np.abs(bio.c - c_before))
    assert dc < 1e-10, (
        f"Нернст: не достигнут стационар, dc={dc:.2e} за {n_steps} шагов"
    )

    # Проверка 2: профиль Больцмановский
    phi = bio.phi
    c_theory = c_left * np.exp(-z * F_RT * (phi - phi[0]))
    rel_err = np.max(np.abs(bio.c[0] - c_theory)) / c_left
    assert rel_err < 1e-3, (
        f"Нернст: профиль не Больцмановский, err={rel_err:.2e}"
    )

    # Проверка 3: формула Нернста
    phi_actual = bio.phi[-1] - bio.phi[0]
    E_nernst = (1.0 / (z * F_RT)) * np.log(c_right / c_left)
    nernst_err = abs(phi_actual - (-E_nernst)) / abs(E_nernst)
    assert nernst_err < 1e-3, (
        f"Нернст: phi={phi_actual:.4f}, expected={-E_nernst:.4f}, err={nernst_err:.2e}"
    )


# ============================================================
# Тест 2: 1D-диффузия (erf-решение)
# ============================================================

def _run_diffusion_erf(dt, D=1.0, t_final=1.0, c0=1.0, N=100):
    """
    Запустить 1D-диффузию с начальным ступенчатым профилем.
    F_RT=0 → чистая диффузия, без электромиграции.
    Возвращает (c_numeric, x, D_eff).
    """
    L = 10.0  # большой домен — границы не влияют на внутреннюю область
    graph, D_eff = make_graph_with_D_eff(N, L, D, D_eff_target=1.0)

    x = graph.pos[:, 0]
    x_center = L / 2

    ions = [{
        "name": "K",
        "D": D,
        "z": 1,
        "c0": c0 / 2,
        "c_left": 0.0,
        "c_right": c0,
    }]

    bio = BiologicalDiffusion(
        graph, ions, dt=dt, F_RT=0.0,
        membrane_potential=0.0,
        self_consistent=False,
        use_newton=True,
    )

    # Начальный профиль: tanh-ступенька в центре
    width = 2.0 * (L / (N - 1))  # ~2 ячейки
    bio.c[0, :] = (c0 / 2.0) * (1.0 + np.tanh((x - x_center) / width))
    bio.c[0, 0] = 0.0
    bio.c[0, -1] = c0

    n_steps = int(round(t_final / dt))
    for _ in range(n_steps):
        bio.step()

    return bio.c[0].copy(), x, D_eff


def test_1d_diffusion():
    """
    Профиль диффузии должен совпасть с erf-решением:
        c(x, t) = (c0/2) * [1 + erf((x - x_center) / (2*sqrt(D_eff*t)))]

    Допуск 1e-2 — backward Euler первого порядка по времени.
    """
    D = 1.0
    t_final = 1.0
    c0 = 1.0
    dt = 0.001
    N = 100

    c_num, x, D_eff = _run_diffusion_erf(dt=dt, D=D, t_final=t_final, c0=c0, N=N)

    # Аналитика
    x_center = 10.0 / 2
    c_analytic = (c0 / 2.0) * (
        1.0 + erf((x - x_center) / (2.0 * np.sqrt(D_eff * t_final)))
    )

    # Сравнение (исключая граничные узлы)
    interior = slice(5, -5)
    err = np.max(np.abs(c_num[interior] - c_analytic[interior])) / c0
    assert err < 1e-2, f"1D диффузия: max err = {err:.2e} > 1e-2"


# ============================================================
# Тест 2b: Порядок сходимости (синус-метод)
# ============================================================

def _run_diffusion_sin(dt, D=1.0, t_final=1.0, N=100):
    """
    Запустить диффузию sin(pi*x/L) с нулевыми границами.
    Аналитика: c(x,t) = sin(pi*x/L) * exp(-alpha*t),
    alpha = D_eff * pi^2 / L^2.

    Преимущество: один мод Фурье → пространственная ошибка пренебрежимо мала,
    временная ошибка доминирует → чистая проверка порядка по dt.
    """
    L = 10.0
    graph, D_eff = make_graph_with_D_eff(N, L, D, D_eff_target=1.0)

    x = graph.pos[:, 0]
    c_init = np.sin(np.pi * x / L)
    c_init[0] = 0.0
    c_init[-1] = 0.0

    ions = [{
        "name": "K",
        "D": D,
        "z": 1,
        "c0": 0.0,
        "c_left": 0.0,
        "c_right": 0.0,
    }]

    bio = BiologicalDiffusion(
        graph, ions, dt=dt, F_RT=0.0,
        membrane_potential=0.0,
        self_consistent=False,
        use_newton=True,
    )

    bio.c[0, :] = c_init

    n_steps = int(round(t_final / dt))
    for _ in range(n_steps):
        bio.step()

    return bio.c[0].copy(), x, D_eff


def test_1d_diffusion_order():
    """
    Проверка порядка сходимости backward Euler:
    при dt -> dt/2 ошибка должна упасть в ~2 раза (первый порядок).

    Используем sin(pi*x/L) — один мод Фурье, пространственная ошибка мала.
    """
    D = 1.0
    t_final = 1.0
    N = 100
    L = 10.0

    dt1 = 0.05
    dt2 = 0.025

    c1, x, D_eff = _run_diffusion_sin(dt=dt1, D=D, t_final=t_final, N=N)
    c2, _, _ = _run_diffusion_sin(dt=dt2, D=D, t_final=t_final, N=N)

    # Аналитика
    alpha = D_eff * np.pi**2 / L**2
    c_analytic = np.sin(np.pi * x / L) * np.exp(-alpha * t_final)

    interior = slice(5, -5)
    err1 = np.max(np.abs(c1[interior] - c_analytic[interior]))
    err2 = np.max(np.abs(c2[interior] - c_analytic[interior]))

    ratio = err1 / err2
    assert 1.7 < ratio < 2.3, (
        f"Порядок не первый: ratio = {ratio:.2f} "
        f"(err1={err1:.2e}, err2={err2:.2e})"
    )


# ============================================================
# Тест 3: Больцмановское распределение
# ============================================================

def test_boltzmann_distribution():
    """
    При фиксированном линейном потенциале phi(x) равновесная
    концентрация:
        c(x) = c_left * exp(-z * F_RT * (phi(x) - phi(0)))

    Проверяем:
      1. R^2 > 0.99 для регрессии c vs c_theory
      2. Наклон ln(c) от phi = -z * F_RT (в пределах 5%)
    """
    F_RT = 1.0
    z = 1
    c0 = 1.0
    phi_max = 2.0  # z*F_RT*phi_max = 2.0 < 5 (умеренный)

    # c_right, согласованный с Больцманом
    c_right = c0 * np.exp(-z * F_RT * phi_max)

    N = 50
    L = 1.0
    D = 1.0
    graph, D_eff = make_graph_with_D_eff(N, L, D, D_eff_target=1.0)

    ions = [{
        "name": "K",
        "D": D,
        "z": z,
        "c0": c0,
        "c_left": c0,
        "c_right": c_right,
    }]

    dt = 0.01
    bio = BiologicalDiffusion(
        graph, ions, dt=dt, F_RT=F_RT,
        membrane_potential=phi_max,
        self_consistent=False,  # фиксированный линейный потенциал
        use_newton=True,
    )

    # Прогон до стационара
    run_to_steady(bio, tol=1e-12, max_steps=50000)

    # Аналитика
    phi = bio.phi
    c_theory = c0 * np.exp(-z * F_RT * (phi - phi[0]))

    # Проверка 1: R^2
    c_num = bio.c[0]
    ss_res = np.sum((c_num - c_theory) ** 2)
    ss_tot = np.sum((c_num - np.mean(c_num)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    assert r2 > 0.99, f"Больцман: R^2 = {r2:.6f} < 0.99"

    # Проверка 2: наклон
    # ln(c) = ln(c0) - z*F_RT*(phi - phi0)
    # slope of ln(c) vs phi = -z*F_RT
    slope, intercept = np.polyfit(phi, np.log(c_num), 1)
    slope_theory = -z * F_RT
    slope_err = abs(slope - slope_theory) / abs(slope_theory)
    assert slope_err < 0.05, (
        f"Больцман: наклон = {slope:.4f}, ожидаем {slope_theory:.4f}, "
        f"ошибка {slope_err:.2%}"
    )
