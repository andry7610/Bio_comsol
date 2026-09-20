"""biological/neuron.py — Модель Ходжкина-Хаксли на графе.

Версия: v0.6.2
  - np.clip(V, -100, 100) в rate-функциях — защита от overflow exp
  - np.errstate в rate-функциях — подавление RuntimeWarning

Версия: v0.6
  - Динамический Нернст: E_Na, E_K из концентраций
  - use_nernst флаг

Версия: v0.5
  - ИИ-интеграция (network/) — автоанализ состояния нейрона
"""

import numpy as np
from network import get_backend, AnalysisResult


class NeuronLayer:
    """Слой нейронов Ходжкина-Хаксли на графе.

    Parameters
    ----------
    graph : Graph
        Граф из network.graph.
    neuron_nodes : list
        Индексы узлов графа, где расположены нейроны.
    dt : float
        Шаг по времени (с).
    use_nernst : bool
        Если True — равновесные потенциалы E_Na, E_K считаются
        динамически из концентраций (уравнение Нернста).
    """

    def __init__(self, graph, neuron_nodes=None, dt=0.01,
                 use_nernst=False,
                 g_Na=120.0, g_K=36.0, g_L=0.3,
                 E_Na=50.0, E_K=-75.0, E_L=-54.3,
                 C_m=1.0, V_rest=-65.0,
                 V_clip_min=-90.0, V_clip_max=60.0,
                 ai_backend=None, analyze_every=0):

        self.graph = graph
        self.N = graph.N
        self.neuron_indices = (np.array(neuron_nodes, dtype=int)
                               if neuron_nodes else np.array([], dtype=int))
        self.n_neurons = len(self.neuron_indices)
        self.dt = dt
        self.use_nernst = use_nernst

        # Проводимости ионных каналов (mS/cm²)
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L

        # Равновесные потенциалы (мВ)
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L

        # Динамические равновесные потенциалы (None, пока не вычислены)
        self.E_Na_dynamic = None
        self.E_K_dynamic = None

        # Ёмкость мембраны (мкФ/cm²)
        self.C_m = C_m

        # Потенциал покоя (мВ)
        self.V_rest = V_rest

        # Ограничения потенциала
        self.V_clip_min = V_clip_min
        self.V_clip_max = V_clip_max

        # Ворота Ходжкина-Хаксли
        self.m = np.zeros(self.n_neurons)
        self.h = np.zeros(self.n_neurons)
        self.n = np.zeros(self.n_neurons)

        # Инициализация ворот в стационарном состоянии при V_rest
        if self.n_neurons > 0:
            V_init = np.full(self.n_neurons, V_rest)
            a_m = self._alpha_m(V_init)
            b_m = self._beta_m(V_init)
            a_h = self._alpha_h(V_init)
            b_h = self._beta_h(V_init)
            a_n = self._alpha_n(V_init)
            b_n = self._beta_n(V_init)

            self.m = a_m / (a_m + b_m)
            self.h = a_h / (a_h + b_h)
            self.n = a_n / (a_n + b_n)

        # ИИ-бэкенд
        self.ai_backend = ai_backend or get_backend()
        self.analyze_every = analyze_every
        self.last_analysis = None
        self.step_count = 0

    # --- Rate-функции Ходжкина-Хаксли ---

    def _alpha_m(self, V):
        V = np.clip(V, -100.0, 100.0)
        x = V + 40.0
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            result = 0.1 * x / (1.0 - np.exp(-0.1 * x))
        result = np.where(np.abs(x) < 1e-6, 1.0, result)
        return result

    def _beta_m(self, V):
        V = np.clip(V, -100.0, 100.0)
        return 4.0 * np.exp(-V / 18.0)

    def _alpha_h(self, V):
        V = np.clip(V, -100.0, 100.0)
        return 0.07 * np.exp(-V / 20.0)

    def _beta_h(self, V):
        V = np.clip(V, -100.0, 100.0)
        return 1.0 / (1.0 + np.exp(-0.1 * (V + 35.0)))

    def _alpha_n(self, V):
        V = np.clip(V, -100.0, 100.0)
        x = V + 55.0
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            result = 0.01 * x / (1.0 - np.exp(-0.1 * x))
        result = np.where(np.abs(x) < 1e-6, 0.1, result)
        return result

    def _beta_n(self, V):
        V = np.clip(V, -100.0, 100.0)
        return 0.125 * np.exp(-V / 80.0)

    # --- Обновление ворот ---

    def update_gates(self, V, dt=None):
        """Обновляет ворота m, h, n по схеме forward Euler."""
        if self.n_neurons == 0:
            return

        dt = dt if dt is not None else self.dt
        V = np.clip(V, -100.0, 100.0)

        a_m = self._alpha_m(V)
        b_m = self._beta_m(V)
        a_h = self._alpha_h(V)
        b_h = self._beta_h(V)
        a_n = self._alpha_n(V)
        b_n = self._beta_n(V)

        dm = a_m * (1.0 - self.m) - b_m * self.m
        dh = a_h * (1.0 - self.h) - b_h * self.h
        dn = a_n * (1.0 - self.n) - b_n * self.n

        self.m = np.clip(self.m + dt * dm, 0.0, 1.0)
        self.h = np.clip(self.h + dt * dh, 0.0, 1.0)
        self.n = np.clip(self.n + dt * dn, 0.0, 1.0)

    # --- Расчёт тока ---

    def compute_current(self, V, c_ions=None, z_ions=None):
        """Вычисляет ионный ток через мембрану.

        I = g_Na * m³ * h * (V - E_Na) +
            g_K  * n⁴ *     (V - E_K)  +
            g_L  *           (V - E_L)

        Если use_nernst=True и переданы c_ions, то E_Na и E_K
        пересчитываются динамически через compute_nernst.
        """
        if self.n_neurons == 0:
            return np.array([])

        # Динамический Нернст
        if self.use_nernst and c_ions is not None:
            if z_ions is None:
                z_ions = np.array([1, 1, -1, 2])  # Na, K, Cl, Ca
            self.compute_nernst(c_ions, z_ions)

        # Выбор E_Na, E_K
        if self.use_nernst and self.E_Na_dynamic is not None:
            E_Na = self.E_Na_dynamic
            E_K = self.E_K_dynamic
        else:
            E_Na = self.E_Na
            E_K = self.E_K

        I = (self.g_Na * self.m**3 * self.h * (V - E_Na) +
             self.g_K * self.n**4 * (V - E_K) +
             self.g_L * (V - self.E_L))

        return I

    # --- Динамический Нернст ---

    def compute_nernst(self, c_global, z_ions):
        """Вычисляет равновесные потенциалы из уравнения Нернста.

        E = (RT / (z * F)) * ln(c_out / c_in)

        Returns (E_Na, E_K) для узлов-нейронов.
        При NaN/inf (нулевые концентрации) — fallback на статические E.
        """
        if self.n_neurons == 0:
            self.E_Na_dynamic = np.array([])
            self.E_K_dynamic = np.array([])
            return np.array([]), np.array([])

        RT_F = 26.7  # мВ, RT/F при T=310 K

        # Na (индекс 0)
        c_out_Na = np.mean(c_global[0])
        c_in_Na = c_global[0][self.neuron_indices]

        # K (индекс 1)
        c_out_K = np.mean(c_global[1])
        c_in_K = c_global[1][self.neuron_indices]

        with np.errstate(divide='ignore', invalid='ignore'):
            E_Na = (RT_F / z_ions[0]) * np.log(c_out_Na / c_in_Na)
            E_K = (RT_F / z_ions[1]) * np.log(c_out_K / c_in_K)

        # Защита от NaN / inf
        E_Na = np.where(np.isnan(E_Na) | np.isinf(E_Na), self.E_Na, E_Na)
        E_K = np.where(np.isnan(E_K) | np.isinf(E_K), self.E_K, E_K)

        self.E_Na_dynamic = E_Na
        self.E_K_dynamic = E_K

        return E_Na, E_K

    # --- Применение к графу ---

    def apply_to_graph(self, phi, c=None):
        """Обновляет потенциал на узлах-нейронах.

        Parameters
        ----------
        phi : np.ndarray
            Потенциал на графе (N,).
        c : np.ndarray, optional
            Концентрации ионов (n_ions, N) — для динамического Нернста.

        Returns
        -------
        np.ndarray
            Обновлённый потенциал (N,).
        """
        if self.n_neurons == 0:
            return phi.copy()

        phi = phi.copy()
        V = phi[self.neuron_indices]

        self.update_gates(V, self.dt)

        if c is not None:
            I = self.compute_current(V, c_ions=c)
        else:
            I = self.compute_current(V)

        # C_m * dV/dt = -I  →  V_new = V - dt * I / C_m
        V_new = V - self.dt * I / self.C_m
        V_new = np.clip(V_new, self.V_clip_min, self.V_clip_max)
        phi[self.neuron_indices] = V_new

        self.step_count += 1

        # Периодический ИИ-анализ
        if self.analyze_every > 0 and self.step_count % self.analyze_every == 0:
            self.analyze_state()

        return phi

    # --- ИИ-анализ ---

    def analyze_state(self):
        """Анализ текущего состояния нейрона через ИИ-бэкенд."""
        context = {
            "neuron_indices": self.neuron_indices,
            "V_rest": self.V_rest,
            "g_Na": self.g_Na,
            "g_K": self.g_K,
            "g_L": self.g_L,
            "E_Na": self.E_Na,
            "E_K": self.E_K,
            "E_L": self.E_L,
            "C_m": self.C_m,
            "dt": self.dt,
            "use_nernst": self.use_nernst,
        }
        self.last_analysis = self.ai_backend.analyze(
            concentrations=np.stack([self.m, self.h, self.n]) if self.n_neurons > 0 else np.zeros((3, 0)),
            charges=np.array([1, 1, 1]),
            time_step=self.step_count,
            context=context,
        )
        return self.last_analysis

    def validate_params(self):
        """Проверка параметров нейрона через ИИ-бэкенд."""
        params = {
            "dt": self.dt,
            "g_Na": self.g_Na,
            "g_K": self.g_K,
            "g_L": self.g_L,
            "E_Na": self.E_Na,
            "E_K": self.E_K,
            "E_L": self.E_L,
            "C_m": self.C_m,
            "V_rest": self.V_rest,
            "use_nernst": self.use_nernst,
        }
        return self.ai_backend.validate_params(params)
# === END ===
