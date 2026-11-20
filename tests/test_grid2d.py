# -*- coding: utf-8 -*-
"""tests/test_grid2d.py — Замки Ступени 3, часть 2: 2D-ванна.

Тот же солвер BiologicalDiffusion БЕЗ единой правки: границы читаются
из атрибутов графа GridGraph2D (node_type, phi_boundary, c_boundary).
Ступень 3 закрывает пункт "границы в атрибуты графа" паспорта v1.0.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biological.diffusion import BiologicalDiffusion
from biological.grid2d import GridGraph2D


def ions_neutral(c0=1.0):
    """Квазинейтральная пара Na/Cl."""
    return [{"name": "Na", "D": 1.0, "z": 1, "c0": c0},
            {"name": "Cl", "D": 1.0, "z": -1, "c0": c0}]


def bath_model(nx=21, ny=21, phi_bottom=0.0, phi_top=0.0,
               dt=0.02, lam=0.5, self_consistent=True, use_newton=True, **kw):
    g = GridGraph2D(nx, ny)
    g.mark_perimeter()
    g.set_phi_bath(phi_bottom=phi_bottom, phi_top=phi_top)
    m = BiologicalDiffusion(
        g, ions_neutral(), dt=dt, self_consistent=self_consistent,
        poisson_lambda=lam, use_newton=use_newton,
        newton_tol=1e-10, newton_max_iter=10, **kw)
    return g, m


# ------------------------------------------- 1. симметрия Пуассона на 2D

def test_grid2d_poisson_bump_symmetry():
    """Бамп в центре 21x21, ванна нулевая -> phi зеркально симметрична."""
    g, m = bath_model(nx=21, ny=21)
    center = (g.nx // 2) + (g.ny // 2) * g.nx
    m.c[0, center] += 0.1          # бамп катионов
    m._solve_poisson()
    phi = m.phi.reshape(g.ny, g.nx)
    assert np.allclose(phi, phi[:, ::-1], atol=1e-12), "нет зеркала по x"
    assert np.allclose(phi, phi[::-1, :], atol=1e-12), "нет зеркала по y"
    assert np.allclose(phi, np.flipud(np.fliplr(phi)), atol=1e-12), "нет зеркала по диагонали"
    assert phi[g.ny // 2, g.nx // 2] == phi.max(), "максимум не в центре"


# -------------------------------- 2. линейное поле между стенками (Лаплас)

def test_grid2d_linear_field_recovery():
    """rho = 0, ванна: низ 0, верх 0.6, бока линейно -> решение строго линейно.

    Линейная функция гармонична и совпадает с данными на всей границе,
    значит это ТОЧНОЕ решение дискретной задачи (до машинной точности).
    """
    V, n = 0.6, 21
    g, m = bath_model(nx=n, ny=n, phi_bottom=0.0, phi_top=V)
    assert abs(m.charge_per_node().max()) < 1e-12, "старт не квазинейтрален"
    m._solve_poisson()
    phi = m.phi.reshape(g.ny, g.nx)
    exact = V * np.arange(g.ny)[:, None] / (g.ny - 1)  # broadcast по x
    assert np.max(np.abs(phi - exact)) < 1e-8, (
        f"поле не линейно: max отклонение {np.max(np.abs(phi - exact)):.2e}")


# ------------------------------------- 3. SG-фронт на 2D: позитивность

def test_grid2d_sg_front_positivity():
    """Фронт истощения в 2D, Pe = 2.5 на вертикальном ребре.

    Перенос 1D-замка (Pe=2.5): ванна с концентрацией 0 сверху,
    поле гонит катионы вверх (к низкому phi). SG-схема обязана держать
    позитивность БЕЗ осцилляций — то же, что дало Pe > 2 в 1D.
    """
    nx, ny = 15, 11
    g = GridGraph2D(nx, ny)
    g.mark_perimeter()
    g.set_phi_bath(phi_bottom=2.5, phi_top=0.0)   # поле вверх
    g.set_c_bath(["Na", "Cl"], c_top=0.0)          # ванна-истощение сверху

    m = BiologicalDiffusion(
        g, ions_neutral(), dt=0.5, F_RT=10.0,
        self_consistent=False, use_newton=True,
        newton_tol=1e-10, newton_max_iter=10)

    # стартовое поле: точная линейная ванна (в 1D то же самое даёт linspace)
    m.phi = 2.5 * (1.0 - g.node_y / (g.ny - 1))
    assert np.allclose(m.phi, m.phi.reshape(ny, nx)[:, ::-1].ravel())

    for _ in range(120):
        m.step()
    assert np.all(np.isfinite(m.c)), "NaN/inf во фронте"
    assert m.c.min() >= -1e-12, f"отрицательные концентрации: min={m.c.min():.3e}"
    # фронт обязан доехать: у верхней стенки катионы истощены
    top_row = m.c[0].reshape(ny, nx)[-1, :]
    assert top_row.max() < 0.5, "фронт не дошёл до ванны"
