"""biological/diffusion.py — Квазинейтральная модель Нернста-Планка.

Версия: v0.9.0
  - НОВОЕ (Ступень 3, v1.0): границы вынесены в атрибуты графа
  - Граф может нести: node_type = {узел: "boundary"},
    phi_boundary = {узел: значение phi}, c_boundary = {узел: {ион: значение}}
  - Модель читает их в __init__; Дирихле ставится в ЛЮБОМ узле, не только 0/N-1
  - Фолбэк: атрибутов нет -> старое поведение v0.9 (узлы 0 и N-1,
    c_left/c_right, phi: 0 и phi_membrane). Правка m.c_boundary[:, 0] после
    конструирования по-прежнему работает (замок обратной совместимости).
  - Замок равенства: 1D из атрибутов == 1D из индексов (tests/test_boundary_graph.py)

Версия: v0.8.0
  - НОВОЕ (Ступень 2, v1.0): SG-схема Шарфеттера-Гуммеля в адвективном члене
  - Центральные разности (E_mid) заменены Бернулли-весами (sg_scheme.bernoulli)
  - Осцилляции при Pe > 2 (документированная граница паспорта v0.9) устранены:
    комбинированная матрица (I + dt*D*L + dt*adv_jac) == точная SG-матрица
    (M-матрица) => позитичность при любом dt. Проверено в песочнице: фронт
    Pe=10 держится в [0, 1] и Ньютоном, и Пикаром; старая схема даёт выбросы.
  - use_sg=True по умолчанию; use_sg=False возвращает старую схему (сравнение)
  - Якобиан Ньютона согласован с SG-потоком

Версия: v0.7.2
  - ИСПРАВЛЕНО: знак в уравнении Пуассона (+lambda*rho вместо -lambda*rho)
    Было: L*phi = -lambda*rho  =>  nabla^2 phi = +rho  (антиэкранировка, взрыв)
    Стало: L*phi = +lambda*rho  =>  nabla^2 phi = -rho  (физика, экранировка)

Версия: v0.7.1
  - ИСПРАВЛЕНО: знак диффузии (I + dt*D*L вместо I - dt*D*L)
  - ИСПРАВЛЕНО: масштаб адвекции (B_unweighted для grad_phi, w вместо w²)
  - ИСПРАВЛЕНО: знак адвекции в Picard и Newton (- dt * adv)

Версия: v0.7
  - Метод Ньютона с аналитическим якобианом (use_newton=True)
  - Квадратичная сходимость: 1-3 итерации вместо 20 у Пикара
  - Матрица E_mid для точного якобиана адвективного члена
  - last_iters — общее поле, last_newton_iters — для Ньютона

Версия: v0.6.1
  - Самосогласованный потенциал из уравнения Пуассона

Версия: v0.5
  - ИИ-бэкенды, автоанализ

Версия: v0.4
  - Квазинейтральность, 4 иона, Пикар
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from network import get_backend, AnalysisResult
from sg_scheme import bernoulli


class BiologicalDiffusion:
    """Квазинейтральная модель ионного транспорта на графе.

    Уравнение Нернста-Планка для иона k:

        ∂c_k/∂t = D_k · ∇²c_k + D_k · z_k · (F/RT) · ∇(c_k · ∇φ)

    В графе: L·c ≈ -∇²c, B@flux ≈ -∇·flux.
    Уравнение: ∂c/∂t = -D·L·c - adv_code(c)

    SG-схема (v0.8.0): дрейфовый поток на ребре e=(i,j) с
    psi = z·(F/RT)·(phi_j - phi_i) берётся Бернулли-весами:

        flux_e = D·[ (1 - B(psi))·c_i + (B(-psi) - 1)·c_j ]

    При psi -> 0 это точно совпадает со старой центральной схемой,
    при |psi| >> 1 берёт концентрацию с наветренной стороны.
    Сумма с неявной диффузией даёт точный SG-поток на ребре:

        Gamma_e = w·D·[ B(psi)·c_i - B(-psi)·c_j ]

    Комбинированная матрица системы — M-матрица: позитичность при любом dt.

    Метод Ньютона (v0.7):
        Построение якобиана J_k = I + dt·D_k·L + dt·adv_jacobian_k
        При фиксированном φ система линейна → Ньютон сходится за 1 шаг.
        При self_consistent=True — за 2-3 шага.

    Самосогласованный потенциал (v0.7.2, исправлен знак):
        L · φ = +λ · ρ,  ρ = Σ_k z_k · c_k

    Метод Пикара (v0.4):
        Линеаризация: adv_k пересчитывается на каждой итерации.
        Линейная сходимость, ~20 итераций.
    """

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

        # Граничные концентрации (Дирихле «ванна»)
        self.c_boundary = np.zeros((self.n_ions, 2))
        for k, ion in enumerate(ions):
            self.c_boundary[k, 0] = ion.get("c_left", ion["c0"])
            self.c_boundary[k, 1] = ion.get("c_right", ion["c0"])

        # --- Потенциал ---
        self.phi = np.linspace(0.0, self.phi_membrane, self.N)

        # --- Граничные узлы (v0.9.0, Ступень 3: из атрибутов графа) ---
        node_type = getattr(graph, "node_type", None)
        if node_type:
            self.boundary_nodes = {int(n) for n, t in node_type.items()
                                   if t == "boundary"}
            phi_b = {int(n): float(v) for n, v in
                     (getattr(graph, "phi_boundary", None) or {}).items()}
            self.phi_boundary = {n: phi_b.get(n, self.phi_membrane)
                                 for n in self.boundary_nodes}
            c_b_graph = getattr(graph, "c_boundary", None) or {}
            self._c_boundary_nodes = {}
            for n in self.boundary_nodes:
                src = c_b_graph.get(n, {})
                self._c_boundary_nodes[n] = np.array(
                    [src.get(self.names[k], self.c0_vals[k])
                     for k in range(self.n_ions)], dtype=float)
        else:
            # Фолбэк v0.9: границы в индексах 0 и N-1 (старое поведение)
            self.boundary_nodes = {0, self.N - 1} if self.N > 1 else set()
            self.phi_boundary = {n: (0.0 if n == 0 else self.phi_membrane)
                                 for n in self.boundary_nodes}
            self._c_boundary_nodes = {n: ("left" if n == 0 else "right")
                                      for n in self.boundary_nodes}

        # --- Матрицы системы (предрассчитанные для Пикара) ---
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

        self.step_count = 0
        self.last_picard_iters = 0
        self.last_newton_iters = 0
        self.last_iters = 0

        # --- ИИ-бэкенд ---
        self.ai_backend = ai_backend or get_backend()
        self.analyze_every = analyze_every
        self.last_analysis: AnalysisResult | None = None

    def _build_incidence(self):
        """Матрица инцидентности B (N × E), средние E_mid (E × N)
        и селекторы концов рёбер E_head/E_tail для SG-схемы (v0.8.0).

        B[node, edge] — поток из узла в ребро (с весом).
        B_unweighted[node, edge] — то же, но веса = 1 (для grad_phi).
        E_mid[edge, node] — 0.5 если узел — конец ребра, 0 иначе.
        E_head[edge, node] — 1 если узел — начало ребра (i).
        E_tail[edge, node] — 1 если узел — конец ребра (j).
        """
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
            # Взвешенная матрица B (для дивергенции)
            row_b.extend([i, j])
            col_b.extend([e_idx, e_idx])
            data_b.extend([-w, w])

            # Невзвешенная матрица B_unweighted (для градиента phi)
            row_bu.extend([i, j])
            col_bu.extend([e_idx, e_idx])
            data_bu.extend([-1.0, 1.0])

            row_e.extend([e_idx, e_idx])
            col_e.extend([i, j])
            data_e.extend([0.5, 0.5])

        self.B = sp.csr_matrix((data_b, (row_b, col_b)),
                              shape=(self.N, E))
        self.B_unweighted = sp.csr_matrix((data_bu, (row_bu, col_bu)),
                                          shape=(self.N, E))
        self.E_mid = sp.csr_matrix((data_e, (row_e, col_e)),
                                   shape=(E, self.N))

        rows = np.arange(E)
        self.E_head = sp.csr_matrix((np.ones(E), (rows, self._edge_i)),
                                    shape=(E, self.N))
        self.E_tail = sp.csr_matrix((np.ones(E), (rows, self._edge_j)),
                                    shape=(E, self.N))

    def _edge_mid_conc(self, c_k):
        """Средняя концентрация иона k на рёбрах."""
        if len(self.edges) == 0:
            return np.zeros(0)
        return self.E_mid @ c_k

    def _advective_term(self, c_k, z_k, D_k):
        """Электрический поток для иона k.

        SG (v0.8.0): psi = z·F_RT·(phi_j - phi_i), Бернулли-веса:
            flux_e = D·[ (1 - B(psi))·c_i + (B(-psi) - 1)·c_j ]
        При psi -> 0 точно совпадает со старой центральной схемой.
        При |psi| >> 1 берёт концентрацию с наветренной стороны,
        что снимает осцилляции при Pe > 2.

        B@flux ≈ -∇·flux, поэтому возвращаем B@flux (знак учитывается
        при подстановке в уравнение: ∂c/∂t = -D·L·c - adv_code).
        """
        if len(self.edges) == 0:
            return np.zeros(self.N)
        grad_phi = self.B_unweighted.T @ self.phi
        if not self.use_sg:
            # Старая схема (центральные разности) — только для сравнения
            c_mid = self._edge_mid_conc(c_k)
            flux = D_k * z_k * self.F_RT * c_mid * grad_phi
            return self.B @ flux
        psi = z_k * self.F_RT * grad_phi
        b_p = bernoulli(psi)
        b_m = bernoulli(-psi)
        flux = D_k * ((1.0 - b_p) * c_k[self._edge_i]
                      + (b_m - 1.0) * c_k[self._edge_j])
        return self.B @ flux

    def _dirichlet_c_value(self, k, node):
        """Значение Дирихле для иона k в граничном узле (Ступень 3).

        Фолбэк-узлы ("left"/"right") читают self.c_boundary — как в v0.9:
        правка массива после конструирования продолжает работать.
        Узлы из атрибутов графа несут собственные значения; без данных — c0.
        """
        src = self._c_boundary_nodes.get(node)
        if isinstance(src, str):
            return self.c_boundary[k, 0 if src == "left" else 1]
        if src is not None:
            return float(src[k])
        return float(self.c0_vals[k])

    def _solve_poisson(self):
        """Самосогласованный потенциал: L · φ = +λ · ρ (v0.7.2: исправлен знак).

        Ступень 3: Дирихле на phi берётся из self.phi_boundary
        ({узел: значение}); в фолбэке это {0: 0.0, N-1: phi_membrane}.
        """
        rho = np.sum(self.z[:, None] * self.c, axis=0)
        rhs = self.poisson_lambda * rho
        for node, value in self.phi_boundary.items():
            rhs[node] = value
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
        """Один шаг по времени.

        use_newton=True → метод Ньютона (квадратичная сходимость).
        use_newton=False → итерации Пикара (линейная сходимость).
        """
        if self.use_newton:
            iters = self._step_newton()
        else:
            iters = self._step_picard()

        if self.analyze_every > 0 and self.step_count % self.analyze_every == 0:
            self.analyze_state()

        return iters

    def _step_picard(self):
        """Шаг Пикара: backward Euler + итерации фиксированной точки.

        Уравнение: (I + dt·D·L)·c_new = c_old - dt·adv(c_iter)
        """
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

    def _step_newton(self):
        """Шаг Ньютона: backward Euler + аналитический якобиан.

        Уравнение: (I + dt·D·L)·c_new = c_old - dt·adv(c_new)
        Невязка: F(c) = (I + dt·D·L)·c - c_old + dt·adv(c) = 0
        Якобиан: J = I + dt·D·L + dt·adv_jac

        SG (v0.8.0): adv_jac = D_k·(B @ (diag(1-B(psi))·E_head
                                        + diag(B(-psi)-1)·E_tail))
        При фиксированном φ система линейна → 1 итерация.
        При self_consistent=True φ пересчитывается → 2-3 итерации.
        """
        c_old = self.c.copy()
        c_iter = c_old.copy()

        for iteration in range(self.newton_max_iter):
            c_prev = c_iter.copy()

            grad_phi = self.B_unweighted.T @ self.phi  # (E,)

            for k in range(self.n_ions):
                # Якобиан адвективного члена
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
                    J_k = (self.I
                           + self.dt * self.D[k] * self.L
                           + self.dt * adv_jac)
                else:
                    J_k = self.I + self.dt * self.D[k] * self.L

                # Граничные условия Дирихле
                J_k = J_k.tolil()
                for node in self.boundary_nodes:
                    J_k[node, :] = 0
                    J_k[node, node] = 1.0
                J_k = J_k.tocsr()

                # RHS
                rhs = c_old[k].copy()
                for node in self.boundary_nodes:
                    rhs[node] = self._dirichlet_c_value(k, node)

                c_iter[k] = spla.spsolve(J_k, rhs)

            # Самосогласованный потенциал (внутри цикла!)
            if self.self_consistent:
                self._solve_poisson()

            # Проверка сходимости Ньютона
            diff = np.max(np.abs(c_iter - c_prev))
            if diff < self.newton_tol:
                break

        self.c = c_iter
        self.step_count += 1
        self.last_newton_iters = iteration + 1
        self.last_iters = self.last_newton_iters

        return self.last_newton_iters
