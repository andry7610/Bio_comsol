"""Минимальный генератор графа для Bio-COMSOL.

Создаёт 1D-цепочку с рёбрами к ближайшим соседям.
Лапласиан — разреженная матрица.
"""

import numpy as np
import scipy.sparse as sp


class Graph:
    def __init__(self, N, p_edge=1.0, seed=42):
        """
        Args:
            N: число узлов
            p_edge: вероятность ребра между соседями (1.0 = полная цепочка)
            seed: зерно генератора
        """
        self.N = N
        rng = np.random.default_rng(seed)

        edges = []
        for i in range(N - 1):
            if rng.random() < p_edge:
                edges.append((i, i + 1, 1.0))

        self.edges = edges

        # Лапласиан: L = D - A
        row, col, data = [], [], []
        for (i, j, w) in edges:
            row.extend([i, j, i, j])
            col.extend([j, i, i, j])
            data.extend([-w, -w, w, w])

        self.laplacian = sp.csr_matrix(
            (data, (row, col)), shape=(N, N)
        )
