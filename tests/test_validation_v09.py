# -*- coding: utf-8 -*-
"""tests/test_validation_v09.py - Валидатор паспорта надёжности v0.9.

Закрывает три подозреваемых из аудита (2026-11):
  1. Знак Пуассона и дебаевская экранировка (после фикса v0.7.2)
  2. Позитивность концентраций и порог числа Пекле (Pe < 2)
  3. Эквивалентность Пикара и Ньютона + сходимость self_consistent

Известные ограничения, зафиксированные xfail-тестами (не баги, а границы
применимости v0.9):
  - Pe > 2 на ребро: центральные разности теряют М-матрицу -> осцилляции
  - dt ~ 1.0 при self_consistent: квази-Ньютон (Гуммель) без перекрёстного
    блока якобиана расходится; лечится уменьшением dt (до v1.0 / upwind)
"""

import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

# гарантируем импорт из корня репозитория
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biological.diffusion import BiologicalDiffusion


# ---------------------------------------------------------------- хелперы

class ChainGraph:
    """Цепочка 1D: N узлов, рёбра весом 1, лапласиан (D - A)."""

    def __init__(self, N):
        self.N = N
        self.edges = [(i, i + 1, 1.0) for i in range(N - 1)]
        A = sp.lil_matrix((N, N))
        for i, j, w in self.edges:
            A[i, j] = -w
            A[j, i] = -w
        for i in range(N):
            A[i, i] = 1.0 if (i == 0 or i == N - 1) else 2.0
        self.laplacian = A.tocsr()


def ions_neutral(c0=1.0):
    """Квазинейтральная пара Na/Cl в безразмерных единицах."""
    return [{"name": "Na", "D": 1.0, "z": 1, "c0": c0},
            {"name": "Cl", "D": 1.0, "z": -1, "c0": c0}]


def make_model(N=41, dt=0.02, F_RT=5.0, membrane=0.0, lam=0.5,
               self_consistent=True, use_newton=True, **kw):
    g = ChainGraph(N)
    return BiologicalDiffusion(
        g, ions_neutral(), dt=dt, F_RT=F_RT, membrane_potential=membrane,
        self_consistent=self_consistent, poisson_lambda=lam,
        use_newton=use_newton, newton_tol=1e-10, newton_max_iter=10, **kw)


def depletion_front_model(F_RT=10.0, membrane=2.5, dt=0.5, N=11):
    """Истощающий фронт: пустая ванна слева, катионы утекают влево."""
    g = ChainGraph(N)
    m = BiologicalDiffusion(g, ions_neutral(), dt=dt, F_RT=F_RT,
                            membrane_potential=membrane, self_consistent=False,
                            use_newton=True, newton_tol=1e-10, newton_max_iter=10)
    m.c_boundary[:, 0] = 0.0
    m.c[:, 0] = 0.0
    return m


# ------------------------------------------------- 1. знак Пуассона (v0.7.2)

def test_poisson_sign_positive_bump():
    """Положительный бамп заряда -> положительный потенциал (знак v0.7.2)."""
    m = make_model(N=41)
    m.c[0, 20] += 0.1
    m._solve_poisson()
    phi = m.phi
    assert np.all(phi >= -1e-14), "отрицательный потенциал от положительного заряда"
    assert abs(phi[20] - np.max(phi)) < 1e-12, "максимум phi не на бампе"
    assert phi[20] > 0.0


def test_poisson_sign_negative_bump():
    """Отрицательный бамп -> отрицательный потенциал (симметрия знака)."""
    m = make_model(N=41)
    m.c[1, 20] += 0.1  # бамп анионов
    m._solve_poisson()
    phi = m.phi
    assert np.all(phi <= 1e-14), "положительный потенциал от отрицательного заряда"
    assert abs(phi[20] - np.min(phi)) < 1e-12
    assert phi[20] < 0.0


# ------------------------------------- 2. дебаевская экранировка (ГЧ-линия)

def test_debye_screening_suppression():
    """Самосогласованная релаксация: локальный избыток заряда экранируется.

    Избыточный заряд (+0.05) в центре цепочки должен быть подавлен
    подвижными ионами: квазинейтральность восстанавливается.
    Замер на эталоне v0.7.2: подавление x640 к шагу 50, машинный ноль к 300.
    """
    m = make_model(N=41, dt=0.02)
    m.c[0, 20] += 0.05
    m._solve_poisson()
    rho_start = m.quasineutrality_error()
    assert rho_start > 1e-2, "бамп не создал начального заряда"

    while m.step_count < 50:
        m.step()
    rho_50 = m.quasineutrality_error()
    assert rho_50 < rho_start / 10.0, (
        f"экранировка слабая: подавление только x{rho_start / rho_50:.1f} (ожидалось >= 10)")

    while m.step_count < 300:
        m.step()
    rho_end = m.quasineutrality_error()
    assert rho_end < 1e-8, f"заряд не сведён: {rho_end:.2e}"
    assert np.max(np.abs(m.phi)) < 1e-6, "потенциал не затух вместе с зарядом"


# --------------------------------------- 3. позитивность и порог Пекле

def test_positivity_pe_below_2():
    """Pe < 2 на ребро: фронт истощения без отрицательных концентраций."""
    m = depletion_front_model(F_RT=10.0, membrane=1.5, dt=0.5)  # Pe = 1.5
    for _ in range(300):
        m.step()
    assert np.all(m.c >= -1e-12), f"отрицательные концентрации при Pe<2: min={m.c.min():.3e}"


@pytest.mark.xfail(reason="известное ограничение v0.9: центральные разности, "
                          "Pe > 2 теряет М-матрицу -> осцилляции (upwind - в v1.0)",
                   strict=False)
def test_positivity_pe_above_2():
    """Pe = 2.5 > 2: позитивность НЕ гарантируется (документированный сбой)."""
    m = depletion_front_model(F_RT=10.0, membrane=2.5, dt=0.5)  # Pe = 2.5
    for _ in range(300):
        m.step()
    assert np.all(m.c >= -1e-12), f"min(c) = {m.c.min():.3e} при Pe>2"


# ---------------------------------- 4. эквивалентность Пикара и Ньютона

def test_picard_newton_equivalence():
    """Оба солвера сходятся к одному и тому же решению (tol 1e-6)."""
    N, c0 = 15, 1.0
    kw = dict(dt=0.05, F_RT=5.0, membrane_potential=0.0,
              self_consistent=True, poisson_lambda=0.5)
    m_pic = BiologicalDiffusion(ChainGraph(N), ions_neutral(c0),
                                picard_tol=1e-9, picard_max_iter=50,
                                use_newton=False, **kw)
    m_new = BiologicalDiffusion(ChainGraph(N), ions_neutral(c0),
                                use_newton=True, newton_tol=1e-10,
                                newton_max_iter=10, **kw)
    for mm in (m_pic, m_new):
        mm.c[0, 7] += 0.05
        mm._solve_poisson()
    for _ in range(200):
        m_pic.step()
        m_new.step()
    assert np.max(np.abs(m_pic.c - m_new.c)) < 1e-6
    assert np.max(np.abs(m_pic.phi - m_new.phi)) < 1e-6
    assert abs(m_pic.quasineutrality_error() - m_new.quasineutrality_error()) < 1e-6


# -------------------------- 5. сходимость self_consistent (пункт 2 аудита)

def test_newton_selfconsistent_converges():
    """Квази-Ньютон при умеренном dt: невязка снимается в лимит итераций."""
    m = make_model(N=21, dt=0.05)
    m.c[0, 10] += 0.1
    m._solve_poisson()
    iters = [m.step() for _ in range(50)]
    assert max(iters) <= m.newton_max_iter, "Ньютон не уложился в лимит итераций"
    assert np.all(np.isfinite(m.c)), "NaN/inf в решении"
    assert m.quasineutrality_error() < 1e-8


@pytest.mark.xfail(reason="известное ограничение v0.9: квази-Ньютон без "
                          "перекрёстного блока якобиана расходится при dt ~ 1.0",
                   strict=False)
def test_selfconsistent_large_dt_remains_finite():
    """dt = 1.0 при self_consistent: решение НЕ обязано быть конечным (xfail)."""
    with np.errstate(all="ignore"):
        m = make_model(N=15, dt=1.0)
        m.c[0, 7] += 0.05
        m._solve_poisson()
        for _ in range(60):
            m.step()
        assert np.all(np.isfinite(m.c)), "расходимость при большом dt (документирована)"
