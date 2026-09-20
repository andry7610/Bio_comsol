"""biological/diffusion.py — Квазинейтральная модель Нернста-Планка.

Версия: v0.5

Отличия от v0.4:
  - Интеграция ИИ-бэкендов (network/) — автоанализ симуляции
  - Метод analyze_state() — вызов нейросети в любой момент
  - Метод validate_params() — проверка параметров перед запуском
  - Периодический анализ в step() через analyze_every

Версия: v0.4
  - Квазинейтральное приближение: Σ z_k · c_k ≈ 0 в каждом узле
  - Мембранный потенциал: градиент φ на границах (по умолчанию -70 мВ)
  - 4 иона: Na, K, Cl, Ca (конфигурируется)
  - Итерации Пикара для электродиффузионного члена
  - Потенциал φ — внешний (фиксированный), не самостоясогласованный

План на v0.6:
  - Самостоясогласованный потенциал из условия квазинейтральности
  - Метод Ньютона
  - Активные ионные каналы (Hodgkin-Huxley)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from network import get_backend, AnalysisResult


class BiologicalDiffusion:
    """Квазинейтральная модель ионного транспорта на графе.

    Уравнение Нернста-Планка для иона k:

        ∂c_k/∂t = D_k · ∇²c_k + D_k · z_k · (F/RT) · ∇(c_k · ∇φ)

    Квазинейтральность (диагностическая, v0.4):

        Σ_k z_k · c_k ≈ 0  в каждом узле

    Мембранный потенциал:

        φ(0) = 0           — внешняя граница (межклетник)
        φ(N-1) = φ_mem     — внутренняя граница (цитоплазма)
        φ(внутри) — линейная интерполяция

    Граничные условия:
        Дирихле («ванна») на граничных узлах графа

    ИИ-интеграция (v0.5):
        Периодический автоанализ через network.get_backend().
        Анализируются: стабильность, квазинейтральность, физичность.
    """

    def __init__(self, graph, ions, dt=1e-3, F_RT=1.0,
                 membrane_potential=-1.8,
                 picard_tol=1e-6, picard_max_iter=20,
                 ai_backend=None, analyze_every=0):

        self.graph = graph
        self.N = graph.N
        self.L = graph.laplacian
        self.edges = graph.edges
        self.dt = dt
        self.F_RT = F_RT
        self.phi_membrane = membrane_potential
        self.picard_tol = picard_tol
        self.picard_max_iter = picard_max_iter

        # --- Ионы ---
        self.n_ions = len(ions)
        self.names = [ion["name"] for ion in ions]
        self.D = np.array([ion["D"] for ion in ions])
        self.z = np.array([ion["z"] for ion in ions])
        self.c0_vals = np.array([ion["c0"] for ion in ions])

        # Концентрации (n_ions, N)
        self.c = np.tile(self.c0_vals.reshape(-1, 1), (1, self.N))

        # Граничные концентрации (Дирихле «ванна»)
        # c_boundary[k] = [c_left, c_right]
        self.c_boundary = np.zeros((self.n_ions, 2))
        for k, ion in enumerate(ions):
            self.c_boundary[k, 0] = ion.get("c_left", ion["c0"])
            self.c_boundary[k, 1] = ion.get("c_right", ion["c0"])

        # --- Потенциал: градиент от 0 (внешний) до φ_mem (внутренний) ---
        self.phi = np.linspace(0.0, self.phi_membrane, self.N)

        # --- Граничные узлы ---
        self.boundary_nodes = {0, self.N - 1} if self.N > 1 else set()

        # --- Матрицы системы (предрассчитанные) ---
        self.I = sp.eye(self.N, format='csr')
        self.A_mats = []
        for k in range(self.n_ions):
            A = (self.I - self.dt * self.D[k] * self.L).tolil()
            for node in self.boundary_nodes:
                A[node, :] = 0
                A[node, node] = 1.0
            self.A_mats.append(A.tocsr())

        # --- Матрица инцидентности для адвективного члена ---
        self._build_incidence()

        self.step_count = 0
        self.last_picard_iters = 0

        # --- ИИ-бэкенд ---
        self.ai_backend = ai_backend or get_backend()
        self.analyze_every = analyze_every
        self.last_analysis: AnalysisResult | None = None

    def _build_incidence(self):
        """Матрица инцидентности B (N × E) для потоков на рёбрах."""
        E = len(self.edges)
        if E == 0:
            self.B = sp.csr_matrix((self.N, 0))
            return
        row, col, data = [], [], []
        for e_idx, (i, j, w) in enumerate(self.edges):
            row.extend([i, j])
            col.extend([e_idx, e_idx])
            data.extend([-w, w])
        self.B = sp.csr_matrix((data, (row, col)), shape=(self.N, E))

    def _edge_mid_conc(self, c_k):
        """Средняя концентрация иона k на рёбрах."""
        if len(self.edges) == 0:
            return np.zeros(0)
        mid = np.zeros(len(self.edges))
        for e_idx, (i, j, _) in enumerate(self.edges):
            mid[e_idx] = 0.5 * (c_k[i] + c_k[j])
        return mid

    def _advective_term(self, c_k, z_k, D_k):
        """Электрический поток для иона k.

        adv = D_k · z_k · (F/RT) · B · diag(c_mid) · B^T · φ

        Здесь c_mid и φ берутся с текущей итерации Пикара —
        это и есть суть метода Пикара (линеаризация).
        """
        if len(self.edges) == 0:
            return np.zeros(self.N)
        c_mid = self._edge_mid_conc(c_k)
        grad_phi = self.B.T @ self.phi              # (E,)
        flux = D_k * z_k * self.F_RT * c_mid * grad_phi
        return self.B @ flux                        # (N,)

    # --- Диагностика ---

    def total_charge(self):
        """Суммарный заряд системы."""
        return float(np.sum(self.z[:, None] * self.c))

    def charge_per_node(self):
        """Заряд в каждом узле: Σ_k z_k · c_k[node]."""
        return np.sum(self.z[:, None] * self.c, axis=0)

    def quasineutrality_error(self):
        """Максимальная невязка квазинейтральности по узлам."""
        return float(np.max(np.abs(self.charge_per_node())))

    # --- ИИ-анализ ---

    def analyze_state(self) -> AnalysisResult:
        """Анализ текущего состояния симуляции через ИИ-бэкенд.

        Возвращает AnalysisResult с summary, warnings, suggestions.
        Работает с любым бэкендом: manual (всегда), local (Ollama), api.
        """
        context = {
            "ion_names": self.names,
            "phi": self.phi,
            "membrane_potential": self.phi_membrane,
            "dt": self.dt,
        }
        self.last_analysis = self.ai_backend.analyze(
            concentrations=self.c,
            charges=self.z,
            time_step=self.step_count,
            context=context,
        )
        return self.last_analysis

    def validate_params(self) -> AnalysisResult:
        """Проверка параметров симуляции через ИИ-бэкенд.

        Вызывается до запуска симуляции для раннего обнаружения проблем.
        """
        params = {
            "dt": self.dt,
            "F_RT": self.F_RT,
            "membrane_potential": self.phi_membrane,
            "ions": [
                {
                    "name": self.names[k],
                    "D": float(self.D[k]),
                    "z": int(self.z[k]),
                    "c0": float(self.c0_vals[k]),
                    "c_left": float(self.c_boundary[k, 0]),
                    "c_right": float(self.c_boundary[k, 1]),
                }
                for k in range(self.n_ions)
            ],
        }
        return self.ai_backend.validate_params(params)

    # --- Шаг по времени ---

    def step(self):
        """Один шаг по времени. backward Euler + итерации Пикара.

        Схема:
            (I - dt · D_k · L) · c_k^{n+1} = c_k^n + dt · adv_k(c^{iter}, φ)

        Итерации Пикара: adv_k пересчитывается на каждой итерации
        с обновлёнными концентрациями.

        Если analyze_every > 0, каждые analyze_every шагов
        запускается ИИ-анализ состояния.
        """
        c_old = self.c.copy()
        c_iter = c_old.copy()

        for iteration in range(self.picard_max_iter):
            c_prev = c_iter.copy()

            for k in range(self.n_ions):
                # Адвективный член (электрический поток)
                adv = self._advective_term(c_iter[k], self.z[k], self.D[k])

                # RHS: c_old + dt * adv
                rhs = c_old[k] + self.dt * adv

                # Дирихле на границах
                rhs[0] = self.c_boundary[k, 0]
                if self.N > 1:
                    rhs[self.N - 1] = self.c_boundary[k, 1]

                # Решаем линейную систему
                c_iter[k] = spla.spsolve(self.A_mats[k], rhs)

            # Проверка сходимости Пикара
            diff = np.max(np.abs(c_iter - c_prev))
            if diff < self.picard_tol:
                break

        self.c = c_iter
        self.step_count += 1
        self.last_picard_iters = iteration + 1

        # Периодический ИИ-анализ
        if self.analyze_every > 0 and self.step_count % self.analyze_every == 0:
            self.analyze_state()

        return self.last_picard_iters
