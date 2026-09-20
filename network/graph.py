"""
network/graph.py — Графовая топология для GAK-WaveCAD.
v0.3.1: Координаты узлов, метрические веса, метод neighbors().
"""
import numpy as np
from scipy import sparse


class Graph:
    """Граф с пространственной метрикой.

    Вес ребра: w_ij = sigma * S / |r_ij|,
    где sigma — проводимость [С/м], S — площадь сечения [м²],
    |r_ij| — евклидово расстояние [м].
    """

    def __init__(self, N, topology="chain", p_edge=None, seed=42,
                 pos=None, conductivity=1.0, cross_section=1e-8):
        """
        Args:
            N: Число узлов.
            topology: "chain", "ring" или "random".
            p_edge: Вероятность ребра (для topology="random").
            seed: Случайное зерно.
            pos: ndarray (N, 2) координат [м]. None = цепочка 1 мкм.
            conductivity: σ [С/м] — удельная проводимость среды.
            cross_section: S [м²] — эффективное сечение канала.
        """
        self.N = N
        self.conductivity = conductivity
        self.cross_section = cross_section
        self.node_type = np.zeros(N, dtype=int)

        # --- Координаты узлов ---
        if pos is None:
            L_total = 1e-6
            x = np.linspace(0, L_total, N)
            self.pos = np.column_stack((x, np.zeros_like(x)))
        else:
            assert len(pos) == N
            self.pos = np.asarray(pos, dtype=float)

        # --- Матрица смежности ---
        if topology == "chain":
            W = np.zeros((N, N))
            for i in range(N - 1):
                W[i, i + 1] = 1.0
                W[i + 1, i] = 1.0
        elif topology == "ring":
            W = np.zeros((N, N))
            for i in range(N):
                j = (i + 1) % N
                W[i, j] = 1.0
                W[j, i] = 1.0
        elif topology == "random":
            assert p_edge is not None
            rng = np.random.default_rng(seed)
            W = rng.uniform(size=(N, N)) < p_edge
            W = W | W.T
            np.fill_diagonal(W, False)
            W = W.astype(float)
        else:
            raise ValueError(f"Неизвестная топология: {topology}")

        # --- Защита от изолированных узлов ---
        degrees = W.sum(axis=1)
        isolated = np.where(degrees == 0)[0]
        for idx in isolated:
            dists = np.linalg.norm(self.pos - self.pos[idx], axis=1)
            dists[idx] = np.inf
            nearest = np.argmin(dists)
            W[idx, nearest] = 1.0
            W[nearest, idx] = 1.0

        # --- Метрические веса ---
        distances = np.linalg.norm(
            self.pos[:, None, :] - self.pos[None, :, :], axis=-1
        )
        with np.errstate(divide='ignore'):
            weights = conductivity * cross_section / distances
        weights[np.isinf(weights)] = 0.0
        np.fill_diagonal(weights, 0.0)
        W = W * weights

        # --- Лапласиан ---
        self.W = sparse.csr_matrix(W)
        D = sparse.diags(W.sum(axis=1).ravel())
        self.L_dense = (D - self.W).toarray()
        self.L_csr = sparse.csr_matrix(
            self.L_dense + 1e-8 * sparse.eye(N)
        )

        # --- Список рёбер ---
        self.edges = []
        W_coo = sparse.csr_matrix(W).tocoo()
        for i, j, w in zip(W_coo.row, W_coo.col, W_coo.data):
            if i < j:
                self.edges.append((int(i), int(j), float(w)))

    def neighbors(self, i):
        """Возвращает список индексов соседей узла i."""
        return self.W.getrow(i).indices.tolist()

    @property
    def laplacian(self):
        return self.L_dense
