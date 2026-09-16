# Bio-COMSOL

Ионный транспорт на графах: модель Нернста-Планка-Пуассона для синтетических
ионных систем и биологической ткани.

## Структура

bio_comsol/ ├── synthetic/ # Рабочий модуль (v0.3) │ ├── diffusion.py # Неявная схема + Пикар │ ├── diffusion_v0.1.py # Архив: явный Эйлер │ └── tests/ ├── biological/ # Каркас (v0.4) │ └── diffusion.py # Заглушка ├── common/ │ ├── graph_utils.py # Генератор графа │ └── solvers.py # Заглушка (spsolve внутри модулей) ├── config.yaml ├── README.md ├── LICENSE └── .gitignore

text

## Что готово

- **synthetic v0.3** — пореберная неявная схема backward Euler
  с итерациями Пикара. Безразмерные переменные, `eps* = 1.0`.
  Устойчива при `F_RT = 1.0`, `dt` до `1e-2`.
  Сохранение заряда на уровне машинного эпсилона.
  Граничные условия Дирихле («ванна») и заготовка под мембрану.

- **biological v0.4** — каркас. Квазинейтральное приближение
  и метод Ньютона — в планах.

## Запуск теста

```bash
pip install numpy scipy
python -m synthetic.tests.test_synthetic
Лицензия
MIT — используйте, форкайте, развивайте.

text

### `.gitignore`

pycache/ *.pyc *.pyo *.egg-info/ .venv/ venv/ .DS_Store *.log

text

### `common/graph_utils.py`

```python
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
Тест — synthetic/tests/test_synthetic.py
python
"""Тест synthetic v0.3: граничные условия, устойчивость, сохранение."""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from common.graph_utils import Graph
from synthetic.diffusion import SyntheticBioDiffusion

IONS = [
    {"name": "Na", "D": 0.01, "z": 1, "c0": 10.0},
    {"name": "K", "D": 0.02, "z": 1, "c0": 100.0},
    {"name": "Cl", "D": 0.015, "z": -1, "c0": 110.0},
]


def main():
    graph = Graph(N=50, p_edge=1.0, seed=42)
    bio = SyntheticBioDiffusion(graph, IONS, eps=1.0, F_RT=1.0, dt=1e-3)

    Q0 = bio.total_charge()
    print(f"Bio-COMSOL Synthetic v0.3 — Test")
    print(f"Graph: {bio.N} nodes, {len(bio.edges)} edges")
    print(f"Ions: {bio.names}")
    print(f"Initial charge: {Q0:.6e}")
    print()

    steps = 500
    for step in range(steps):
        bio.step()
        if step % 100 == 0 or step == steps - 1:
            Q = bio.total_charge()
            neg = (bio.c < 0).sum()
            print(f"Step {step:4d}: Q = {Q:.4e}, "
                  f"range = [{bio.c.min():.3f}..{bio.c.max():.3f}], "
                  f"neg = {neg}")

    print()
    print("=== RESULTS ===")
    print(f"Final charge:   {bio.total_charge():.6e}")
    print(f"Dirichlet [0]:   Na={bio.c[0,0]:.4f}, K={bio.c[1,0]:.4f}, "
          f"Cl={bio.c[2,0]:.4f}")
    print(f"Conc. range:     {bio.c.min():.4f} .. {bio.c.max():.4f}")
    print(f"Negative conc.:  {(bio.c < 0).sum()}")


if __name__ == "__main__":
    main()
# Запуск тестов (из корня репозитория):
#   python tests/test_bio_diffusion_implicit.py
