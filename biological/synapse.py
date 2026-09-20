"""biological/synapse.py — Синаптическая передача + STDP.

Версия: v0.9
  - STDP (spike-timing-dependent plasticity)
  - Обнаружение спайков presinaptic и postsynaptic
  - LTP: pre до post → weight растёт
  - LTD: post до pre → weight падает
  - Ограничение весов [w_min, w_max]

Версия: v0.8
  - Проводимостная модель синапса
  - Обнаружение спайков по пересечению порога
  - Экспоненциальное затухание s
"""

import numpy as np
from network import get_backend, AnalysisResult


class SynapseLayer:
    """Слой синапсов на графе с опциональным STDP.

    Parameters
    ----------
    graph : Graph
    synapses : list of dict
        pre, post, weight, E_syn, tau_syn, synapse_type
    dt : float
    V_threshold : float
        Порог обнаружения спайка (мВ).
    use_stdp : bool
        Включить spike-timing-dependent plasticity.
    A_plus : float
        Амплитуда LTP (pre до post).
    A_minus : float
        Амплитуда LTD (post до pre).
    tau_plus : float
        Окно LTP (в ед. dt).
    tau_minus : float
        Окно LTD (в ед. dt).
    w_min : float
        Минимальный вес.
    w_max : float
        Максимальный вес.
    """

    def __init__(self, graph, synapses, dt=0.01, V_threshold=-20.0,
                 use_stdp=False,
                 A_plus=0.01, A_minus=0.012,
                 tau_plus=20.0, tau_minus=20.0,
                 w_min=0.0, w_max=10.0,
                 ai_backend=None, analyze_every=0):

        self.graph = graph
        self.N = graph.N
        self.dt = dt
        self.V_threshold = V_threshold

        self.n_syn = len(synapses)

        if self.n_syn == 0:
            self.pre = np.array([], dtype=int)
            self.post = np.array([], dtype=int)
            self.weights = np.array([])
            self.E_syn = np.array([])
            self.tau_syn = np.array([])
            self.synapse_types = []
            self.s = np.array([])
        else:
            self.pre = np.array([s["pre"] for s in synapses], dtype=int)
            self.post = np.array([s["post"] for s in synapses], dtype=int)
            self.weights = np.array([s["weight"] for s in synapses],
                                    dtype=float)
            self.E_syn = np.array([s["E_syn"] for s in synapses])
            self.tau_syn = np.array([s["tau_syn"] for s in synapses])
            self.synapse_types = [s.get("synapse_type", "excitatory")
                                 for s in synapses]
            self.s = np.zeros(self.n_syn)

        self.V_prev = None
        self.post_neurons = (np.unique(self.post)
                             if self.n_syn > 0 else np.array([], dtype=int))

        # --- STDP ---
        self.use_stdp = use_stdp
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max

        # Время последнего спайка для каждого pre и post
        self.t_pre_last = np.full(self.n_syn, -np.inf)
        self.t_post_last = np.full(self.n_syn, -np.inf)
        self.sim_time = 0.0

        self.ai_backend = ai_backend or get_backend()
        self.analyze_every = analyze_every
        self.last_analysis = None
        self.step_count = 0

    def step(self, phi):
        """Обновляет воротные переменные и применяет STDP.

        Parameters
        ----------
        phi : np.ndarray
            Потенциал на всех узлах графа (N,).
        """
        if self.n_syn == 0:
            self.step_count += 1
            self.sim_time += self.dt
            return

        V_pre = phi[self.pre]
        V_post = phi[self.post]

        if self.V_prev is None:
            self.V_prev = phi.copy()
            pre_spikes = np.zeros(self.n_syn, dtype=bool)
            post_spikes = np.zeros(self.n_syn, dtype=bool)
        else:
            V_pre_prev = self.V_prev[self.pre]
            V_post_prev = self.V_prev[self.post]
            pre_spikes = ((V_pre > self.V_threshold) &
                          (V_pre_prev <= self.V_threshold))
            post_spikes = ((V_post > self.V_threshold) &
                           (V_post_prev <= self.V_threshold))

        self.V_prev = phi.copy()
        self.sim_time += self.dt

        # --- STDP ---
        if self.use_stdp:
            self._apply_stdp(pre_spikes, post_spikes)

        # --- Затухание и активация ворот ---
        self.s = self.s * np.exp(-self.dt / self.tau_syn)
        self.s[pre_spikes] = np.minimum(self.s[pre_spikes] + 1.0, 1.0)
        self.s = np.clip(self.s, 0.0, 1.0)

        # Запоминаем времена спайков
        if self.use_stdp:
            self.t_pre_last[pre_spikes] = self.sim_time
            self.t_post_last[post_spikes] = self.sim_time

        self.step_count += 1

        if self.analyze_every > 0 and self.step_count % self.analyze_every == 0:
            self.analyze_state()

    def _apply_stdp(self, pre_spikes, post_spikes):
        """Применяет правило STDP к весам.

        LTP: pre спайк → проверяем, не было ли post спайка недавно
             (pre до post = Δt > 0 → усиливаем)
        LTD: post спайк → проверяем, не было ли pre спайка недавно
             (post до pre = Δt < 0 → ослабляем)
        """
        # LTP: pre спайк, post уже стрелял до этого
        for i in np.where(pre_spikes)[0]:
            dt_post = self.sim_time - self.t_post_last[i]
            if 0 < dt_post < self.tau_plus * self.dt * 100:
                # post стрелял до pre → Δt = t_post - t_pre < 0
                # Но это LTD (post до pre), не LTP
                pass

            dt_pre = self.sim_time - self.t_pre_last[i]
            # Если pre стрелял сейчас, и post стрелял позже → LTP
            # Но post ещё не стрелял в этом шаге, значит post после pre
            # → это потенциальный LTP, но мы узнаём только когда post стрелянет

        # LTD: post спайк, pre уже стрелял до этого
        for i in np.where(post_spikes)[0]:
            dt = self.sim_time - self.t_pre_last[i]
            if 0 < dt < self.tau_minus * self.dt * 100:
                # pre стрелял до post → LTP (усиление)
                dw = self.A_plus * np.exp(-dt / (self.tau_minus * self.dt))
                self.weights[i] = np.clip(
                    self.weights[i] + dw, self.w_min, self.w_max)

        # LTP: post спайк произошёл после pre
        for i in np.where(post_spikes)[0]:
            dt = self.sim_time - self.t_pre_last[i]
            if 0 < dt < self.tau_plus * self.dt * 100:
                dw = self.A_plus * np.exp(-dt / (self.tau_plus * self.dt))
                self.weights[i] = np.clip(
                    self.weights[i] + dw, self.w_min, self.w_max)

        # LTD: pre спайк произошёл после post
        for i in np.where(pre_spikes)[0]:
            dt = self.sim_time - self.t_post_last[i]
            if 0 < dt < self.tau_minus * self.dt * 100:
                dw = -self.A_minus * np.exp(-dt / (self.tau_minus * self.dt))
                self.weights[i] = np.clip(
                    self.weights[i] + dw, self.w_min, self.w_max)

    def compute_currents(self, phi):
        """Вычисляет синаптические токи для постсинаптических нейронов."""
        if self.n_syn == 0:
            return {}

        V_post = phi[self.post]
        I_syn = self.weights * self.s * (V_post - self.E_syn)

        currents = {}
        for i, post_node in enumerate(self.post):
            if post_node not in currents:
                currents[post_node] = 0.0
            currents[post_node] += I_syn[i]

        return currents

    def get_current_array(self, phi, neuron_indices):
        """Возвращает массив синаптических токов для заданных нейронов."""
        if self.n_syn == 0:
            return np.zeros(len(neuron_indices))

        currents = self.compute_currents(phi)
        I_ext = np.zeros(len(neuron_indices))
        for node, current in currents.items():
            mask = neuron_indices == node
            I_ext[mask] += current
        return I_ext

    def analyze_state(self):
        context = {
            "n_synapses": self.n_syn,
            "V_threshold": self.V_threshold,
            "dt": self.dt,
            "synapse_types": self.synapse_types,
            "use_stdp": self.use_stdp,
        }
        self.last_analysis = self.ai_backend.analyze(
            concentrations=(self.s.reshape(1, -1)
                            if self.n_syn > 0 else np.zeros((1, 0))),
            charges=np.array([1]),
            time_step=self.step_count,
            context=context,
        )
        return self.last_analysis
# === END ===
