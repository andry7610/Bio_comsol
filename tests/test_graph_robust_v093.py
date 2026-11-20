"""Замки регрессии v0.9.3: защитное чтение граф-атрибутов.

Причина появления: CI (0c32677) упал с 36 failed / 39 errors —
ValueError "truth value of an array" в конструкторе BiologicalDiffusion.
Корень: словарь node_type у реального графа проекта содержит НЕстроковые
значения (массивы признаков узлов). Сравнение `t == "boundary"` на массиве
даёт массив, а не bool -> ValueError ещё в __init__.

Замки:
1. node_type с массивами-значениями -> конструктор не падает,
   включается фолбэк-семантика (узлы 0 и N-1, phi 0/phi_membrane).
2. node_type со строками без "boundary" -> фолбэк (раньше: пустое
   множество границ и вырожденная матрица Пуассона, phi = nan).
3. c_boundary со значениями-массивами (spec не dict) -> без падения,
   Дирихле = c0.
"""

import numpy as np
import scipy.sparse as sp

from biological.diffusion import BiologicalDiffusion


IONS = [
    {"name": "Na", "D": 1.0, "z":  1, "c0": 0.14},
    {"name": "K",  "D": 1.0, "z":  1, "c0": 0.005},
    {"name": "Cl", "D": 1.0, "z": -1, "c0": 0.145},
    {"name": "Ca", "D": 1.0, "z":  2, "c0": 0.0001},
]


class _MockGraph:
    """Минимальный граф: N, edges, laplacian + произвольные атрибуты."""

    def __init__(self, N=10, **attrs):
        self.N = N
        self.edges = [(i, i + 1, 1.0) for i in range(N - 1)]
        L = np.zeros((N, N))
        for i in range(N - 1):
            L[i, i] += 1.0
            L[i + 1, i + 1] += 1.0
            L[i, i + 1] -= 1.0
            L[i + 1, i] -= 1.0
        self.laplacian = sp.csr_matrix(L)
        for key, val in attrs.items():
            setattr(self, key, val)


def test_node_type_with_array_values_falls_back():
    """Словарь node_type с массивами вместо строк: не падать, фолбэк."""
    g = _MockGraph(10, node_type={n: np.array([1.0, 0.0]) for n in range(10)})
    m = BiologicalDiffusion(g, IONS, self_consistent=True, use_newton=True)
    for _ in range(3):
        m.step()
    assert m.boundary_nodes == {0, 9}
    assert np.isfinite(m.phi).all()
    assert abs(m.phi[0]) < 1e-10
    assert abs(m.phi[-1] - (-1.8)) < 1e-10


def test_node_type_without_boundary_labels_falls_back():
    """node_type есть, меток "boundary" нет: фолбэк, а не вырожденный Пуассон."""
    g = _MockGraph(10, node_type={n: "interior" for n in range(10)})
    m = BiologicalDiffusion(g, IONS, self_consistent=True, use_newton=True)
    for _ in range(3):
        m.step()
    assert m.boundary_nodes == {0, 9}
    assert np.isfinite(m.phi).all()
    assert abs(m.phi[0]) < 1e-10
    assert abs(m.phi[-1] + 1.8) < 1e-10


def test_c_boundary_with_array_specs_is_ignored():
    """c_boundary с не-dict значениями (массивы): не падать, Дирихле = c0."""
    g = _MockGraph(
        10,
        node_type={n: ("boundary" if n in (0, 9) else "interior")
                   for n in range(10)},
        c_boundary={0: np.array([0.1, 0.1, 0.1, 0.1]),
                    9: np.array([0.1, 0.1, 0.1, 0.1])},
    )
    m = BiologicalDiffusion(g, IONS, self_consistent=True)
    for _ in range(3):
        m.step()
    assert np.isfinite(m.c).all()
    assert m.c.min() >= -1e-12
