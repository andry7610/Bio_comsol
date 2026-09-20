"""biological/synapse.py — Синаптическая передача на графе.

Версия: v0.8
  - Проводимостная модель синапса (conductance-based)
  - Обнаружение спайков по пересечению порога снизу вверх
  - Экспоненциальное затухание воротной переменной s
  - Интеграция с NeuronLayer через I_ext
"""

import numpy as np
from network import get_backend, AnalysisResult


class SynapseLayer:
    """Слой синапсов на графе.

    Parameters
    ----------
    graph : Graph
        Граф из network.graph.
    synapses : list of dict
        Каждый словарь:
            pre: int — индекс узла графа пресинаптического нейрона
            post: int — индекс узла графа постсинаптического нейрона
            weight: float — проводимость синапса (mS/cm²)
            E_syn: float — потенциал реверсии (мВ)
            tau_syn: float — постоянная времени затухания (в ед. dt)
            synapse_type: str — "excitatory" или "inhibitory"
    dt : float
        Шаг по времени.
    V_threshold : float
        Порог обнаружения спайка (мВ).
    """

    def __init__(self, graph, synapses, dt=0.01, V_threshold=-20.0,
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
            self.weights = np.array([s["weight"] for s in synapses])
            self.E_syn = np.array([s["E_syn"] for s in synapses])
            self.tau_syn = np.array([s["tau_syn"] for s in synapses])
            self.synapse_types = [s.get("synapse_type", "excitatory")
                                 for s in synapses]
            self.s = np.zeros(self.n_syn)

        self.V_prev = None
        self.post_neurons = (np.unique(self.post)
                             if self.n_syn > 0 else np.array([], dtype=int))

        self.ai_backend = ai_backend or get_backend()
        self.analyze_every = analyze_every
        self.last_analysis = None
        self.step_count = 0

    def step(self, phi):
        """Обновляет воротные переменные синапсов.

        Parameters
        ----------
        phi : np.ndarray
            Потенциал на всех узлах графа (N,).
        """
        if self.n_syn == 0:
            self.step_count += 1
            return

        V_pre = phi[self.pre]

        if self.V_prev is None:
            self.V_prev = V_pre.copy()
            spikes = np.zeros(self.n_syn, dtype=bool)
        else:
            spikes = ((V_pre > self.V_threshold) &
                      (self.V_prev <= self.V_threshold))

        self.V_prev = V_pre.copy()

        self.s = self.s * np.exp(-self.dt / self.tau_syn)
        self.s[spikes] = np.minimum(self.s[spikes] + 1.0, 1.0)
        self.s = np.clip(self.s, 0.0, 1.0)

        self.step_count += 1

        if self.analyze_every > 0 and self.step_count % self.analyze_every == 0:
            self.analyze_state()

    def compute_currents(self, phi):
        """Вычисляет синаптические токи для постсинаптических нейронов.

        Returns
        -------
        dict
            {node_index: total_synaptic_current}
        """
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
        """Возвращает массив синаптических токов для заданных нейронов.

        Parameters
        ----------
        phi : np.ndarray
            Потенциал на графе (N,).
        neuron_indices : np.ndarray
            Индексы нейронов в графе.

        Returns
        -------
        np.ndarray
            Токи для каждого нейрона (n_neurons,).
        """
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
