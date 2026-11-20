# -*- coding: utf-8 -*-
"""biological/grid2d.py — Квадратная сетка как граф (Ступень 3, часть 2).

GridGraph2D(nx, ny): узлы n = ix + iy*nx, рёбра по окрестности
фон Неймана (вправо и вверх), вес каждого ребра w = 1.0.

Контракт тот же, что у 1D-цепочки: N, edges, laplacian. Плюс атрибуты
границ для солвера (Ступень 3):

    node_type     {узел: "boundary"}
    phi_boundary  {узел: phi}
    c_boundary    {узел: {имя_иона: значение}}

Солвер BiologicalDiffusion читает границы из этих атрибутов, поэтому
2D-ванна подключается БЕЗ единой правки в diffusion.py.

Вспомогательные методы:
    mark_perimeter()  — все узлы периметра -> boundary
    set_phi_bath()    — Дирихле-ванна для потенциала (низ/верх, бока линейно)
    set_c_bath()      — Дирихле-ванна для концентраций (низ/верх)
"""

import numpy as np
import scipy.sparse as sp


class GridGraph2D:
    """Квадратная сетка nx x ny как граф с лапласианом (D - A)."""

    def __init__(self, nx, ny, w=1.0):
        if nx < 2 or ny < 2:
            raise ValueError("нужна сетка минимум 2x2")
        self.nx = int(nx)
        self.ny = int(ny)
        self.w = float(w)
        self.N = self.nx * self.ny

        # --- рёбра: вправо и вверх (фон Нейман) ---
        edges = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                n = ix + iy * self.nx
                if ix + 1 < self.nx:
                    edges.append((n, n + 1, self.w))      # сосед справа
                if iy + 1 < self.ny:
                    edges.append((n, n + self.nx, self.w))  # сосед сверху
        self.edges = edges

        # --- лапласиан: диагональ = степень, сосед -w ---
        A = sp.lil_matrix((self.N, self.N))
        deg = np.zeros(self.N)
        for i, j, ww in edges:
            A[i, j] = -ww
            A[j, i] = -ww
            deg[i] += ww
            deg[j] += ww
        for n in range(self.N):
            A[n, n] = deg[n]
        self.laplacian = A.tocsr()

        # --- координаты узлов ---
        self.node_x = np.array([n % self.nx for n in range(self.N)])
        self.node_y = np.array([n // self.nx for n in range(self.N)])

        # --- атрибуты границ (контракт Ступени 3) ---
        self.node_type = {}
        self.phi_boundary = {}
        self.c_boundary = {}

    # ------------------------------------------------------------------
    def coord(self, n):
        return int(self.node_x[n]), int(self.node_y[n])

    def perimeter(self):
        """Узлы периметра: x = 0, x = nx-1, y = 0 или y = ny-1."""
        return sorted(n for n in range(self.N)
                      if self.node_x[n] in (0, self.nx - 1)
                      or self.node_y[n] in (0, self.ny - 1))

    def mark_perimeter(self):
        """Все узлы периметра -> node_type="boundary". До конструирования модели."""
        self.node_type = {n: "boundary" for n in self.perimeter()}

    def set_phi_bath(self, phi_bottom=0.0, phi_top=None, sides="linear"):
        """Дирихле-ванна для потенциала.

        Низ (iy=0) = phi_bottom, верх (iy=ny-1) = phi_top,
        боковые: sides="linear" — линейная интерполяция между низом и верхом
        (тогда чистое поле строго вертикальное), sides="flat" — phi_bottom.
        Требует mark_perimeter() (или делает его сама).
        """
        if phi_top is None:
            phi_top = phi_bottom
        if not self.node_type:
            self.mark_perimeter()
        H = self.ny - 1
        for n in self.perimeter():
            iy = int(self.node_y[n])
            if iy == 0:
                self.phi_boundary[n] = float(phi_bottom)
            elif iy == self.ny - 1:
                self.phi_boundary[n] = float(phi_top)
            elif sides == "linear" and H > 0:
                self.phi_boundary[n] = float(
                    phi_bottom + (phi_top - phi_bottom) * iy / H)
            else:
                self.phi_boundary[n] = float(phi_bottom)

    def set_c_bath(self, ion_names, c_bottom=None, c_top=None):
        """Дирихле-ванна для концентраций: {узел: {ион: значение}}.

        c_bottom — низ сетки, c_top — верх. Скаляр -> всем ионам,
        dict {ион: значение} — поимённо. Боковые стороны остаются свободными.
        None для стороны — эта сторона свободна (концентрация живая, с c0).
        """
        def norm(v):
            if isinstance(v, dict):
                return {name: float(v.get(name, 0.0)) for name in ion_names}
            return {name: float(v) for name in ion_names}

        bath = {}
        for n in self.perimeter():
            iy = int(self.node_y[n])
            if iy == 0 and c_bottom is not None:
                bath[n] = norm(c_bottom)
            elif iy == self.ny - 1 and c_top is not None:
                bath[n] = norm(c_top)
        self.c_boundary = bath
