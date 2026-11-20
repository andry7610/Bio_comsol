# -*- coding: utf-8 -*-
"""tests/test_boundary_graph.py - Ступень 3: границы в атрибутах графа.

Замки:
  1. Граф с node_type/phi_boundary/c_boundary даёт в 1D ТОТ ЖЕ результат,
     что и старый фолбэк по индексам 0 и N-1 (обратная совместимость).
  2. Потенциал на графовой границе берётся из phi_boundary.
  3. Дирихле-границу можно поставить в произвольный узел (не только 0/N-1),
     и решение остаётся кусочно-монотонным между границами.
"""

import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from biological.diffusion import BiologicalDiffusion


class ChainGraph:
    """Цепочка 1D без атрибутов границ (фолбэк: узлы 0 и N-1)."""

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


class AttrChainGraph(ChainGraph):
    """Та же цепочка, но границы заданы атрибутами графа (Ступень 3)."""

    def __init__(self, N, node_type, phi_boundary=None, c_boundary=None):
        super().__init__(N)
        self.node_type = node_type
        self.phi_boundary = phi_boundary or {}
        self.c_boundary = c_boundary or {}


def ions_neutral(c0=1.0):
    return [{"name": "Na", "D": 1.0, "z": 1, "c0": c0},
            {"name": "Cl", "D": 1.0, "z": -1, "c0": c0}]


def run(model, steps):
    for _ in range(steps):
        model.step()
    return model


def test_graph_attributes_match_index_fallback():
    """1D из атрибутов графа == 1D из индексов (замок обратной совместимости)."""
    N, V = 15, 0.4
    kw = dict(dt=0.02, F_RT=5.0, membrane_potential=V,
              self_consistent=True, poisson_lambda=0.5,
              use_newton=True, newton_tol=1e-10, newton_max_iter=10)

    m_fallback = BiologicalDiffusion(ChainGraph(N), ions_neutral(), **kw)
    m_attrs = BiologicalDiffusion(
        AttrChainGraph(
            N, node_type={0: "boundary", N - 1: "boundary"},
            phi_boundary={0: 0.0, N - 1: V},
            c_boundary={0: {"Na": 1.0, "Cl": 1.0},
                        N - 1: {"Na": 1.0, "Cl": 1.0}},
        ), ions_neutral(), **kw)

    m_fallback.c[0, N // 2] += 0.1
    m_attrs.c[0, N // 2] += 0.1
    m_fallback._solve_poisson()
    m_attrs._solve_poisson()
    assert np.max(np.abs(m_fallback.phi - m_attrs.phi)) < 1e-12

    run(m_fallback, 40)
    run(m_attrs, 40)
    assert np.max(np.abs(m_fallback.c - m_attrs.c)) < 1e-10
    assert np.max(np.abs(m_fallback.phi - m_attrs.phi)) < 1e-10


def test_boundary_phi_from_graph():
    """Потенциал на графовой границе = значение из phi_boundary."""
    N, V = 11, 0.7
    g = AttrChainGraph(N, node_type={0: "boundary", N - 1: "boundary"},
                       phi_boundary={0: 0.0, N - 1: V})
    m = BiologicalDiffusion(g, ions_neutral(), dt=0.02, F_RT=5.0,
                            self_consistent=True, poisson_lambda=0.5,
                            use_newton=True, newton_tol=1e-10,
                            newton_max_iter=10)
    m._solve_poisson()
    assert abs(m.phi[N - 1] - V) < 1e-12
    assert abs(m.phi[0]) < 1e-12


def test_interior_dirichlet_node():
    """Дирихле-граница в произвольном узле: решение кусочно-монотонно."""
    N, V_mid = 21, 0.5
    g = AttrChainGraph(
        N, node_type={0: "boundary", 10: "boundary", N - 1: "boundary"},
        phi_boundary={0: 0.0, 10: V_mid, N - 1: 0.0})
    m = BiologicalDiffusion(g, ions_neutral(), dt=0.02, F_RT=5.0,
                            self_consistent=True, poisson_lambda=0.5,
                            use_newton=True, newton_tol=1e-10,
                            newton_max_iter=10)
    m._solve_poisson()
    assert abs(m.phi[10] - V_mid) < 1e-12
    left = m.phi[1:10]
    right = m.phi[11:]
    assert np.all(np.diff(left) > 0), "слева от внутренней границы нет роста"
    assert np.all(np.diff(right) < 0), "справа от внутренней границы нет спада"
    # несколько шагов времени: граница держится, решение конечно
    run(m, 30)
    assert np.all(np.isfinite(m.c))
    assert abs(m.phi[10] - V_mid) < 1e-10
