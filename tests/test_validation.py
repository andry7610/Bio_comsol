"""
tests/test_validation.py

Валидация Bio_comsol против аналитических решений.

Три теста:
  1. Равновесие Нернста — стационарный потенциал и профиль концентрации
  2. 1D-диффузия — сравнение с erf-решением + проверка порядка сходимости
  3. Больцмановское распределение — экспоненциальный профиль c(φ)

Допуски:
  Тест 1: rtol = 1e-3 (прямое сравнение с формулой Нернста)
  Тест 2: rtol = 1e-2 (backward Euler — первый порядок по времени)
  Тест 2b: порядок сходимости ratio ≈ 2 при dt → dt/2
  Тест 3: R² > 0.99, наклон в пределах 5%
"""

import numpy as np
from scipy.special import erf
import pytest

from network.graph import Graph
from biological.diffusion import BiologicalDiffusion


# ============================================================
# Вспомогательные функции
# ============================================================

def make_chain(N, L=1.0, x0=-0.5, conductivity=1.0, cross_section=1.0):
    """Создать цепочку из N узлов на отрезке [x0, x0+L]."""
    x = np.linspace(x0, x0 + L, N)
    pos = np.column_stack((x, np.zeros_like(x)))
    return Graph(N=N, topology="chain", pos=pos,
                 conductivity=conductivity, cross_section=cross_section)


def run_to_steady(bio, tol=1e-12, max_steps=50000):
    """Прогнать модель до стационара. Возвращает число шагов."""
    for i in range(max_steps):
        c_prev = bio.c.copy()
        bio.step()
        if np.max(np.abs(bio.c - c_prev)) < tol:
            return i + 1
    return max_steps


# ============================================================
# Тест 1: Равновесие Нернста
# ============================================================

def test_nernst_equilibrium():
    """
    При мембранном потенциале, равном Нернстовскому для данного иона,
    система выходит на стационар с нулевым потоком.

    Boltzmann: c(x) = c_left * exp(-z * F_RT * phi(x))
    => membrane_potential = -(1/(z*F_RT)) * ln(c_right / c_left)
    => E_Nernst = (1/(z*F_RT)) * ln(c_right / c_left) = -membrane_potential

    Проверяем:
      1. Стационар достигнут (dc/dt → 0)
      2. Профиль концентрации — Больцмановский
      3. Формула Нернста выполнена
    """
    F_RT = 38.9       # 1/В при 310 K
    z = 1
    c_left = 10e-6    # моль/м³
    c_right = 100e-6

    # --- Нернстовский потенциал (в конвенции модели) ---
    membrane_potential = -(1.0 / (z * F_RT)) * np.log(c_right / c_left)
    E_nernst = (1.0 / (z * F_RT)) * np.log(c_right / c_left)

    # --- Граф ---
    N = 50
    graph = make_chain(N, L=1.0, x0=0.0)

    # --- Модель: один ион, фиксированный потенциал ---
    ions = [{
        "name": "K",
        "D": 1e-5,
        "z": z,
        "c0": c_left,
        "c_left": c_left,
        "c_right": c_right,
    }]

    bio = BiologicalDiffusion(
        graph, ions, dt=1e-3, F_RT=F_RT,
        membrane_potential=membrane_potential,
        self_consistent=False,
        use_newton=True,
    )

    # --- Прогон до стационара ---
    n_steps = run_to_steady(bio, tol=1e-12, max_steps=50000)

    # --- Проверка 1: стационар достигнут ---
    c_before = bio.c.copy()
    bio.step()
    dc = np.max(np.abs(bio.c - c_before))
    assert dc < 1e-10, f"Нернст: не достигнут стационар, dc={dc:.2e} за {n_steps} шагов"

    # --- Проверка 2: профиль Больцмановский ---
    phi = bio.phi
    c_theory = c_left * np.exp(-z * F_RT * phi)
    c_theory *= c_left / c_theory[0]   # нормировка на левую границу
    rel_err = np.max(np.abs(bio.c[0] - c_theory)) / c_left
    assert rel_err < 1e-3, f"Нернст: профиль не Больцмановский, err={rel_err:.2e}"

    # --- Проверка 3: формула Нернста ---
    phi_diff = phi[-1] - phi[0]
    E_numerical = -phi_diff  # в конвенции модели
    nernst_err = abs(E_numerical - E_nernst) / abs(E_nernst)
    assert nernst_err < 1e-3, f"Нернст: E_num={E_numerical:.6f}, E_theory={E_nernst:.6f}, err={nernst_err:.2e}"


# ============================================================
# Тест 2: 1D-диффузия (с проверкой порядка сходимости)
# ============================================================

def _run_diffusion(dt, N=200, D=1.0, t_final=1.0):
    """
    Запустить 1D-диффузию (чистую, без электромиграции).
    Начальный профиль: сглаженная ступенька (tanh) в центре.
    Возвращает (c_numeric, x, D_eff).
    """
    # --- Граф: отрезок [-0.5, 0.5] ---
    graph = make_chain(N, L=1.0, x0=-0.5,
                       conductivity=1.0, cross_section=1.0)

    # --- Эффективный коэффициент диффузии ---
    dx = graph.pos[1, 0] - graph.pos[0, 0]
    w = graph.edges[0][2]
    D_eff = D * w * dx ** 2

    # --- Ион: F_RT=0 → нет электромиграции, phi=0 ---
    c0 = 1.0
    ions = [{
        "name": "K",
        "D": D,
        "z": 1,
        "c0": c0 * 0.5,
        "c_left": 0.0,       # левая граница = 0
        "c_right": c0,       # правая граница = c0
    }]

    bio = BiologicalDiffusion(
        graph, ions, dt=dt, F_RT=0.0,
        membrane_potential=0.0,
        self_consistent=False,
        use_newton=True,
    )

    # --- Начальный профиль: tanh-ступенька в центре ---
    x = graph.pos[:, 0]
    bio.c[0] = (c0 / 2.0) * (1.0 + np.tanh(x / dx))

    # --- Прогон ---
    n_steps = int(round(t_final / dt))
    for _ in range(n_steps):
        bio.step()

    return bio.c[0], x, D_eff


def test_1d_diffusion():
    """
    Профиль диффузии должен совпасть с erf-решением:
        c(x, t) = (c0/2) * [1 + erf(x / (2*sqrt(D_eff*t)))]

    Допуск 1e-2 — backward Euler первого порядка по времени.
    """
    D = 1.0
    t_final = 1.0
    c0 = 1.0
    dt = 1e-3

    c_num, x, D_eff = _run_diffusion(dt=dt, D=D, t_final=t_final)

    # --- Аналитика ---
    c_analytic = (c0 / 2.0) * (1.0 + erf(x / (2.0 * np.sqrt(D_eff * t_final))))

    # --- Сравнение (исключая граничные узлы) ---
    interior = slice(5, -5)
    err = np.max(np.abs(c_num[interior] - c_analytic[interior])) / c0
    assert err < 1e-2, f"1D диффузия: max err = {err:.2e} > 1e-2"


def test_1d_diffusion_order():
    """
    Проверка порядка сходимости backward Euler:
    при dt → dt/2 ошибка должна упасть в ~2 раза (первый порядок).

    Если ratio не в [1.7, 2.3] — либо баг, либо не первый порядок.
    """
    D = 1.0
    t_final = 1.0
    c0 = 1.0

    c1, x, D_eff = _run_diffusion(dt=1e-3, D=D, t_final=t_final)
    c2, _, _ = _run_diffusion(dt=5e-4, D=D, t_final=t_final)

    c_analytic = (c0 / 2.0) * (1.0 + erf(x / (2.0 * np.sqrt(D_eff * t_final))))

    interior = slice(5, -5)
    err1 = np.max(np.abs(c1[interior] - c_analytic[interior])) / c0
    err2 = np.max(np.abs(c2[interior] - c_analytic[interior])) / c0

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
    При фиксированном внешнем потенциале φ(x) равновесная концентрация:
        c(x) = c_left * exp(-z * F_RT * (φ(x) - φ(0)))

    Проверяем:
      1. R² > 0.99 для линейной регрессии ln(c) от φ
      2. Наклон = -z * F_RT (в пределах 5%)
    """
    F_RT = 38.9
    z = 1
    c0 = 100e-6
    phi_max = 0.03   # 30 мВ → z*F_RT*phi = 1.17 < 5

    # --- c_right, согласованный с Больцманом ---
    c_right = c0 * np.exp(-z * F_RT * phi_max)

    N = 50
    graph = make_chain(N, L=1.0, x0=0.0)

    ions = [{
        "name": "K",
        "D": 1e-5,
        "z": z,
        "c0": c0,
        "c_left": c0,
        "c_right": c_right,
    }]

    bio = BiologicalDiffusion(
        graph, ions, dt=1e-3, F_RT=F_RT,
        membrane_potential=phi_max,
        self_consistent=False,   # фиксированный линейный потенциал
        use_newton=True,
    )

    # --- Прогон до стационара ---
    run_to_steady(bio, tol=1e-12, max_steps=50000)

    # --- Аналитика ---
    phi = bio.phi
    c_theory = c0 * np.exp(-z * F_RT * (phi - phi[0]))

    # --- Проверка 1: R² ---
    c_num = bio.c[0]
    ss_res = np.sum((c_num - c_theory) ** 2)
    ss_tot = np.sum((c_num - np.mean(c_num)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    assert r2 > 0.99, f"Больцман: R² = {r2:.6f} < 0.99"

    # --- Проверка 2: наклон ln(c) vs φ ---
    slope, intercept = np.polyfit(phi, np.log(c_num), 1)
    slope_theory = -z * F_RT
    slope_err = abs(slope - slope_theory) / abs(slope_theory)
    assert slope_err < 0.05, (
        f"Больцман: наклон {slope:.2f}, теория {slope_theory:.2f}, "
        f"err={slope_err:.2e}"
    )
