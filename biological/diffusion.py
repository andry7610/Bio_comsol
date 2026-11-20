"""biological/diffusion.py — Квазинейтральная модель Нернста-Планка.

Версия: v0.9.3 (фикс CI: защита чтения граф-атрибутов)
  - node_type с НЕстроковыми значениями (массивы признаков узлов) больше
    не рушит конструктор: меткой границы считается только строка "boundary".
  - Если в node_type нет ни одной метки "boundary" — честный фолбэк на
    легаси-узлы 0 и N-1 (раньше получали пустое множество границ и
    вырожденную матрицу Пуассона: phi = nan).
  - c_boundary_graph: значения не-dict (массивы) => c0, без truth-value.

Версия: v0.9.2 (Ступень 4, sparse-Ньютон)
  - Сборка якобиана Ньютона в CSR без tolil-преобразований в горячем цикле:
      J_k = A_mats[k] + dt * (S_mask @ adv_jac),
    где S_mask — предрассчитанная маска внутренних узлов (нули на границах),
    а строки Дирихле уже зашиты в A_mats (предрассчитаны для Пикара).
    Граничная строка J_k получается сложением, а не перезаписью через lil.
  - Исправлено: финальный solve Пуассона после цикла Ньютона —
    phi теперь согласован с принятыми c (раньше отставал на итерацию).
  - Замки: sparse-Ньютон == legacy-Ньютон (tolil) до 1e-12,
    sparse-Ньютон == Пикар (сходимость к одному аттрактору),
    строки Дирихле якобиана == e_node.

Версия: v0.9.0 (Ступень 3, границы из атрибутов графа)
  - node_type / phi_boundary / c_boundary читаются из графа;
    фолбэк — узлы 0 и N-1 со значениями c_left / c_right и phi = 0 / phi_membrane.
  - Дирихле-границу можно поставить в произвольный узел графа.

Версия: v0.8.0
  - SG-схема Шарфеттера-Гуммеля: позитичность при любом dt (Pe > 2 закрыт).

Версия: v0.7.2
  - Знак Пуассона: L*phi = +lambda*rho (экранировка).

Версия: v0.7
  - Метод Ньютона с аналитическим якобианом.

Версия: v0.4
  - Квазинейтральность, Пикар.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from network import get_backend, AnalysisResult
from sg_scheme import bernoulli


class BiologicalDiffusion:
    """Квазинейтральная модель ионного транспорта на графе."""

    def __init__(self, graph, ions, dt=1e-3, F_RT=1.0,
                 membrane_potential=-1.8,
                 picard_tol=1e-6, picard_max_iter=20,
                 ai_backend=None, analyze_every=0,
                 self_consistent=False, poisson_lambda=1.0,
                 use_newton=False, newton_tol=1e-8, newton_max_iter=5,
                 use_sg=True):

        self.graph = graph
        self.N = graph.N
        self.L = graph.laplacian
        self.edges = graph.edges
        self.dt = dt
        self.F_RT = F_RT
        self.phi_membrane = membrane_potential
        self.picard_tol = picard_tol
        self.picard_max_iter = picard_max_iter
        self.self_consistent = self_consistent
        self.poisson_lambda = poisson_lambda
        self.use_newton = use_newton
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.use_sg = use_sg

        # --- Ионы ---
        self.n_ions = len(ions)
        self.names = [ion["name"] for ion in ions]
        self.D = np.array([ion["D"] for ion in ions])
        self.z = np.array([ion["z"] for ion in ions])
        self.c0_vals = np.array([ion["c0"] for ion in ions])

        # Концентрации (n_ions, N)
        self.c = np.tile(self.c0_vals.reshape(-1, 1), (1, self.N))

        # Граничные концентрации (Дирихле "ванна", фолбэк)
        self.c_boundary = np.zeros((self.n_ions, 2))
        for k, ion in enumerate(ions):
            self.c_boundary[k, 0] = ion.get("c_left", ion["c0"])
            self.c_boundary[k, 1] = ion.get("c_right", ion["c0"])

        # --- Потенциал ---
        self.phi = np.linspace(0.0, self.phi_membrane, self.N)

        # --- Граничные узлы: атрибуты графа (Ступень 3) или фолбэк ---
        node_types = getattr(graph, "node_type", None)
        phi_b = getattr(graph, "phi_boundary", None)
        c_b = getattr(graph, "c_boundary", None)
        self._has_graph_bounds = isinstance(node_types, dict) and len(node_types) > 0
        if self._has_graph_bounds:
            # v0.9.3: защита от нестроковых значений в node_type.
            # Словарь node_type может нести признаки узлов (массивы) —
            # сравнение массива со строкой запрещено numpy. Меткой границы
            # считается только строка "boundary" (np.str_ тоже строка).
            self.boundary_nodes = set()
            for n, t in node_types.items():
                if isinstance(t, str) and t == "boundary":
                    self.boundary_nodes.add(n)
            self.phi_boundary = dict(phi_b) if isinstance(phi_b, dict) else {}
            self.c_boundary_graph = dict(c_b) if isinstance(c_b, dict) else {}
            if not self.boundary_nodes:
                # Словарь node_type есть, но меток "boundary" в нём нет —
                # это не разметка границ. Честный фолбэк на легаси-узлы
                # 0 и N-1 (иначе матрица Пуассона вырождена).
                self._has_graph_bounds = False
        if not self._has_graph_bounds:
            self.boundary_nodes = {0, self.N - 1} if self.N > 1 else set()
            self.phi_boundary = {0: 0.0}
            if self.N > 1:
                self.phi_boundary[self.N - 1] = self.phi_membrane
            self.c_boundary_graph = {}

        # --- Матрицы системы (предрассчитанные для Пикара; строки Дирихле
        #     переиспользуются Ньютоном в v0.9.2) ---
        self.I = sp.eye(self.N, format='csr')
        self.A_mats = []
        for k in range(self.n_ions):
            A = (self.I + self.dt * self.D[k] * self.L).tolil()
            for node in self.boundary_nodes:
                A[node, :] = 0
                A[node, node] = 1.0
            self.A_mats.append(A.tocsr())

        # --- Матрица Пуассона ---
        if self_consistent:
            A_pois = self.L.copy().tolil()
            for node in self.boundary_nodes:
                A_pois[node, :] = 0
                A_pois[node, node] = 1.0
            self.A_poisson = A_pois.tocsr()

        # --- Матрица инцидентности B, средние E_mid, селекторы SG ---
        self._build_incidence()

        # --- Маска внутренних узлов для sparse-сборки якобиана (v0.9.2) ---
        mask = np.ones(self.N)
        for node in self.boundary_nodes:
            mask[node] = 0.0
        self._interior_mask = sp.diags(mask, format='csr')

        self.step_count = 0
        self.last_picard_iters = 0
        self.last_newton_iters = 0
        self.last_iters = 0

        # --- ИИ-бэкенд ---
        self.ai_backend = ai_backend or get_backend()
        self.analyze_every = analyze_every
        self.last_analysis: AnalysisResult | None = None

    def _dirichlet_c_value(self, k, node):
        """Значение Дирихле для иона k в узле node (Ступень 3).

        Графовые атрибуты > строковые ссылки на ванну > c0.
        """
        if self._has_graph_bounds:
            spec = self.c_boundary_graph.get(node)
            # v0.9.3: только dict-спека без truth-value на массивах;
            # значения-массивы (признаки узлов) игнорируются => c0.
            if not isinstance(spec, dict) or len(spec) == 0:
                return self.c0_vals[k]
            val = spec.get(self.names[k])
            if val is None:
                return self.c0_vals[k]
            if isinstance(val, str):
                if val == "left":
                    return float(self.c_boundary[k, 0])
                if val == "right":
                    return float(self.c_boundary[k, 1])
                raise ValueError("unknown c_boundary spec: %r" % (val,))
            return float(val)
        if node in self.boundary_nodes:
            if node == min(self.boundary_nodes):
                return float(self.c_boundary[k, 0])
            return float(self.c_boundary[k, 1])
        return self.c0_vals[k]

    def _build_incidence(self):
        """Матрица инцидентности B, средние E_mid, селекторы SG."""
        E = len(self.edges)
        if E == 0:
            self.B = sp.csr_matrix((self.N, 0))
            self.B_unweighted = sp.csr_matrix((self.N, 0))
            self.E_mid = sp.csr_matrix((0, self.N))
            self.E_head = sp.csr_matrix((0, self.N))
            self.E_tail = sp.csr_matrix((0, self.N))
            self._edge_i = np.zeros(0, dtype=int)
            self._edge_j = np.zeros(0, dtype=int)
            return

        row_b, col_b, data_b = [], [], []
        row_bu, col_bu, data_bu = [], [], []
        row_e, col_e, data_e = [], [], []

        self._edge_i = np.array([e[0] for e in self.edges], dtype=int)
        self._edge_j = np.array([e[1] for e in self.edges], dtype=int)

        for e_idx, (i, j, w) in enumerate(self.edges):
            row_b.extend([i, j])
            col_b.extend([e_idx, e_idx])
            data_b.extend([-w, w])

            row_bu.extend([i, j])
            col_bu.extend([e_idx, e_idx])
            data_bu.extend([-1.0, 1.0])

            row_e.extend([e_idx, e_idx])
            col_e.extend([i, j])
            data_e.extend([0.5, 0.5])

        self.B = sp.csr_matrix((data_b, (row_b, col_b)), shape=(self.N, E))
        self.B_unweighted = sp.csr_matrix((data_bu, (row_bu, col_bu)),
                                          shape=(self.N, E))
        self.E_mid = sp.csr_matrix((data_e, (row_e, col_e)), shape=(E, self.N))

        rows = np.arange(E)
        self.E_head = sp.csr_matrix((np.ones(E), (rows, self._edge_i)),
                                    shape=(E, self.N))
        self.E_tail = sp.csr_matrix((np.ones(E), (rows, self._edge_j)),
                                    shape=(E, self.N))

    def _edge_mid_conc(self, c_k):
        if len(self.edges) == 0:
            return np.zeros(0)
        return self.E_mid @ c_k

    def _advective_term(self, c_k, z_k, D_k):
        if len(self.edges) == 0:
            return np.zeros(self.N)
        grad_phi = self.B_unweighted.T @ self.phi
        if not self.use_sg:
            c_mid = self._edge_mid_conc(c_k)
            flux = D_k * z_k * self.F_RT * c_mid * grad_phi
            return self.B @ flux
        psi = z_k * self.F_RT * grad_phi
        b_p = bernoulli(psi)
        b_m = bernoulli(-psi)
        flux = D_k * ((1.0 - b_p) * c_k[self._edge_i]
                      + (b_m - 1.0) * c_k[self._edge_j])
        return self.B @ flux

    def _solve_poisson(self):
        """Самосогласованный потенциал: L * phi = +lambda * rho."""
        rho = np.sum(self.z[:, None] * self.c, axis=0)
        rhs = self.poisson_lambda * rho
        for node in self.boundary_nodes:
            rhs[node] = self.phi_boundary.get(node, 0.0)
        self.phi = spla.spsolve(self.A_poisson, rhs)

    # --- Диагностика ---

    def total_charge(self):
        return float(np.sum(self.z[:, None] * self.c))

    def charge_per_node(self):
        return np.sum(self.z[:, None] * self.c, axis=0)

    def quasineutrality_error(self):
        return float(np.max(np.abs(self.charge_per_node())))

    # --- ИИ-анализ ---

    def analyze_state(self) -> AnalysisResult:
        context = {
            "ion_names": self.names,
            "phi": self.phi,
            "membrane_potential": self.phi_membrane,
            "dt": self.dt,
            "self_consistent": self.self_consistent,
            "use_newton": self.use_newton,
        }
        self.last_analysis = self.ai_backend.analyze(
            concentrations=self.c,
            charges=self.z,
            time_step=self.step_count,
            context=context,
        )
        return self.last_analysis

    def validate_params(self) -> AnalysisResult:
        params = {
            "dt": self.dt,
            "F_RT": self.F_RT,
            "membrane_potential": self.phi_membrane,
            "self_consistent": self.self_consistent,
            "poisson_lambda": self.poisson_lambda,
            "use_newton": self.use_newton,
            "newton_tol": self.newton_tol,
            "newton_max_iter": self.newton_max_iter,
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
        if self.use_newton:
            iters = self._step_newton()
        else:
            iters = self._step_picard()

        if self.analyze_every > 0 and self.step_count % self.analyze_every == 0:
            self.analyze_state()

        return iters

    def _step_picard(self):
        """Шаг Пикара: backward Euler + итерации фиксированной точки."""
        c_old = self.c.copy()
        c_iter = c_old.copy()

        for iteration in range(self.picard_max_iter):
            c_prev = c_iter.copy()

            for k in range(self.n_ions):
                adv = self._advective_term(c_iter[k], self.z[k], self.D[k])
                rhs = c_old[k] - self.dt * adv
                for node in self.boundary_nodes:
                    rhs[node] = self._dirichlet_c_value(k, node)
                c_iter[k] = spla.spsolve(self.A_mats[k], rhs)

            diff = np.max(np.abs(c_iter - c_prev))
            if diff < self.picard_tol:
                break

        self.c = c_iter
        self.step_count += 1
        self.last_picard_iters = iteration + 1
        self.last_iters = self.last_picard_iters

        if self.self_consistent:
            self._solve_poisson()

        return self.last_picard_iters

    def _newton_jacobian(self, k, grad_phi):
        """Якобиан иона k (v0.9.2: чисто CSR, без tolil в цикле).

        J_k = A_mats[k] + dt * (S_mask @ adv_jac).
        Строки Дирихле уже сидят в A_mats[k] (идентичность на границах),
        S_mask обнуляет граничные строки adv_jac — сумма даёт ровно то,
        что раньше делалось через tolil-перезапись.
        """
        if len(self.edges) > 0:
            if self.use_sg:
                psi_e = self.z[k] * self.F_RT * grad_phi
                b_p = bernoulli(psi_e)
                b_m = bernoulli(-psi_e)
                adv_jac = self.D[k] * (self.B @ (
                    sp.diags(1.0 - b_p) @ self.E_head
                    + sp.diags(b_m - 1.0) @ self.E_tail))
            else:
                adv_jac = (self.D[k] * self.z[k] * self.F_RT
                           * (self.B @ sp.diags(grad_phi) @ self.E_mid))
            return self.A_mats[k] + self.dt * (self._interior_mask @ adv_jac)
        return self.A_mats[k]

    def _step_newton(self):
        """Шаг Ньютона: backward Euler + аналитический якобиан (v0.9.2)."""
        c_old = self.c.copy()
        c_iter = c_old.copy()

        for iteration in range(self.newton_max_iter):
            c_prev = c_iter.copy()

            grad_phi = self.B_unweighted.T @ self.phi  # (E,)

            for k in range(self.n_ions):
                J_k = self._newton_jacobian(k, grad_phi)

                rhs = c_old[k].copy()
                for node in self.boundary_nodes:
                    rhs[node] = self._dirichlet_c_value(k, node)

                c_iter[k] = spla.spsolve(J_k, rhs)

            if self.self_consistent:
                self._solve_poisson()

            diff = np.max(np.abs(c_iter - c_prev))
            if diff < self.newton_tol:
                break

        self.c = c_iter
        self.step_count += 1
        self.last_newton_iters = iteration + 1
        self.last_iters = self.last_newton_iters

        # Финальное согласование phi с принятыми концентрациями (v0.9.2):
        # внутри цикла solve Пуассона использует ещё не присвоенный self.c,
        # поэтому без этого solve phi отстаёт на одну итерацию.
        if self.self_consistent:
            self._solve_poisson()

        return self.last_newton_iters
