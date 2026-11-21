"""biological/grid3d.py — регулярная 3D-сетка (Ступень 6, часть 1).

Контракт совпадает с grid2d: .N, .edges, .laplacian, .adjacency,
плюс сервис face_nodes(axis, side) и interior_nodes().
7-точечный шаблон, единичные веса рёбер. Солвер слеп к геометрии:
BiologicalDiffusion работает с GridGraph3D без правок ядра.

Индексация: index(x, y, z) = (x * ny + y) * nz + z (z — самый быстрый).
"""

import numpy as np
import scipy.sparse as sp


class GridGraph3D:
    """Регулярная сетка nx * ny * nz, 7-точечный шаблон."""

    def __init__(self, nx, ny, nz):
        for name, v in (("nx", nx), ("ny", ny), ("nz", nz)):
            if not isinstance(v, (int, np.integer)) or isinstance(v, bool):
                raise TypeError("%s must be int" % name)
            if v < 1:
                raise ValueError("%s must be >= 1" % name)
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.shape = (self.nx, self.ny, self.nz)
        self.N = self.nx * self.ny * self.nz
        self._build()
        self._build_matrices()

    def index(self, x, y, z):
        return (x * self.ny + y) * self.nz + z

    def coords(self, idx):
        z = idx % self.nz
        rest = idx // self.nz
        y = rest % self.ny
        x = rest // self.ny
        return (x, y, z)

    def _build(self):
        # Рёбра вдоль каждой оси: сдвиг индекса по оси x = +ny*nz,
        # по y = +nz, по z = +1. Векторизовано ради сборки 50^3 < 5 c.
        arr = np.arange(self.N).reshape(self.nx, self.ny, self.nz)
        pairs = []
        pairs.append((arr[:-1, :, :].ravel(), arr[1:, :, :].ravel()))
        pairs.append((arr[:, :-1, :].ravel(), arr[:, 1:, :].ravel()))
        pairs.append((arr[:, :, :-1].ravel(), arr[:, :, 1:].ravel()))
        i_all = np.concatenate([p[0] for p in pairs])
        j_all = np.concatenate([p[1] for p in pairs])
        w = np.ones(i_all.size)
        self.edges = list(zip(i_all.tolist(), j_all.tolist(), w.tolist()))

    def _build_matrices(self):
        i_arr = np.array([e[0] for e in self.edges], dtype=int)
        j_arr = np.array([e[1] for e in self.edges], dtype=int)
        w = np.array([e[2] for e in self.edges], dtype=float)
        deg = np.bincount(np.concatenate([i_arr, j_arr]),
                          minlength=self.N).astype(float)
        self.degrees = deg
        rows = np.concatenate([i_arr, j_arr, np.arange(self.N)])
        cols = np.concatenate([j_arr, i_arr, np.arange(self.N)])
        data = np.concatenate([-w, -w, deg])
        self.laplacian = sp.csr_matrix((data, (rows, cols)),
                                       shape=(self.N, self.N))
        arows = np.concatenate([i_arr, j_arr])
        acols = np.concatenate([j_arr, i_arr])
        adata = np.concatenate([w, w])
        self.adjacency = sp.csr_matrix((adata, (arows, acols)),
                                       shape=(self.N, self.N))

    def face_nodes(self, axis, side):
        """Узлы грани. axis: 0/1/2, side: 0|'lo' или 1|'hi'."""
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        if side in (0, "lo"):
            s = 0
        elif side in (1, "hi"):
            s = -1
        else:
            raise ValueError("side must be 0/'lo' or 1/'hi'")
        arr = np.arange(self.N).reshape(self.nx, self.ny, self.nz)
        sl = [slice(None)] * 3
        sl[axis] = s
        return arr[tuple(sl)].ravel().tolist()

    def interior_nodes(self):
        """Узлы, у которых все три координаты строго внутри сетки."""
        if self.nx < 3 or self.ny < 3 or self.nz < 3:
            return []
        arr = np.arange(self.N).reshape(self.nx, self.ny, self.nz)
        return arr[1:-1, 1:-1, 1:-1].ravel().tolist()
