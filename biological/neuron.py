"""biological/neuron.py — Активный нейро-слой на базе квазинейтральной модели.

Реализует упрощённую модель Ходжкина-Хаксли (Na/K каналы) в выделенных узлах.
Поддерживает динамический расчёт равновесных потенциалов по формуле Нернста.
"""

import numpy as np

R = 8.314
T = 310.0
F = 96485.0
RT_OVER_F = R * T / F * 1000.0


class NeuronLayer:
    """Нейро-слой, накладываемый на граф Bio-COMSOL."""

    def __init__(self, graph, neuron_nodes, dt=1e-3,
                 use_nernst=False, na_index=0, k_index=1):
        self.N = graph.N
        self.neuron_indices = np.array(neuron_nodes)
        self.dt = dt
        self.use_nernst = use_nernst
        self.na_index = na_index
        self.k_index = k_index

        self.g_Na = 120.0
        self.E_Na = 50.0
        self.g_K = 36.0
        self.E_K = -75.0
        self.g_L = 0.3
        self.E_L = -54.3

        self.m = np.full(len(neuron_nodes), 0.05)
        self.h = np.full(len(neuron_nodes), 0.6)
        self.n = np.full(len(neuron_nodes), 0.32)

        self.last_current = np.zeros(len(neuron_nodes))
        self.E_Na_dynamic = None
        self.E_K_dynamic = None

    @staticmethod
    def _alpha_m(V):
        if V == -40:
            return 0.1
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-0.1 * (V + 40.0)))

    @staticmethod
    def _beta_m(V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def _alpha_h(V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def _beta_h(V):
        return 1.0 / (1.0 + np.exp(-0.1 * (V + 35.0)))

    @staticmethod
    def _alpha_n(V):
        if V == -55:
            return 0.01
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-0.1 * (V + 55.0)))

    @staticmethod
    def _beta_n(V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def update_gates(self, V_neurons):
        """Обновляет переменные ворот (m, h, n) для всех нейронов."""
        for idx in range(len(V_neurons)):
            V = V_neurons[idx]
            am = self._alpha_m(V)
            bm = self._beta_m(V)
            ah = self._alpha_h(V)
            bh = self._beta_h(V)
            an = self._alpha_n(V)
            bn = self._beta_n(V)

            tau_m = 1.0 / (am + bm)
            tau_h = 1.0 / (ah + bh)
            tau_n = 1.0 / (an + bn)

            m_inf = am / (am + bm)
            h_inf = ah / (ah + bh)
            n_inf = an / (an + bn)

            self.m[idx] = m_inf + (self.m[idx] - m_inf) * np.exp(-self.dt / tau_m)
            self.h[idx] = h_inf + (self.h[idx] - h_inf) * np.exp(-self.dt / tau_h)
            self.n[idx] = n_inf + (self.n[idx] - n_inf) * np.exp(-self.dt / tau_n)

    def compute_nernst(self, c_global, z_ions):
        """Считает E_Na и E_K по формуле Нернста из концентраций."""
        n_neurons = len(self.neuron_indices)
        if n_neurons == 0:
            return np.array([]), np.array([])

        nodes = self.neuron_indices

        z_na = z_ions[self.na_index]
        c_na_all = c_global[self.na_index]
        c_na_out = np.mean(c_na_all)
        c_na_in = c_na_all[nodes]

        with np.errstate(divide='ignore', invalid='ignore'):
            E_Na = (RT_OVER_F / z_na) * np.log(c_na_out / c_na_in)
        E_Na = np.nan_to_num(E_Na, nan=self.E_Na, posinf=self.E_Na, neginf=self.E_Na)

        z_k = z_ions[self.k_index]
        c_k_all = c_global[self.k_index]
        c_k_out = np.mean(c_k_all)
        c_k_in = c_k_all[nodes]

        with np.errstate(divide='ignore', invalid='ignore'):
            E_K = (RT_OVER_F / z_k) * np.log(c_k_out / c_k_in)
        E_K = np.nan_to_num(E_K, nan=self.E_K, posinf=self.E_K, neginf=self.E_K)

        self.E_Na_dynamic = E_Na
        self.E_K_dynamic = E_K

        return E_Na, E_K

    def compute_current(self, V_neurons, c_ions=None, z_ions=None):
        """Считает мембранный ток I_ion для нейронов."""
        if self.use_nernst and c_ions is not None and z_ions is not None:
            E_Na, E_K = self.compute_nernst(c_ions, z_ions)
        else:
            E_Na = self.E_Na
            E_K = self.E_K

        I_Na = self.g_Na * (self.m ** 3) * self.h * (V_neurons - E_Na)
        I_K = self.g_K * (self.n ** 4) * (V_neurons - E_K)
        I_L = self.g_L * (V_neurons - self.E_L)

        I_total = I_Na + I_K + I_L
        self.last_current = I_total
        return I_total

    def apply_to_graph(self, phi_global, c_global=None):
        """Обновляет глобальный потенциал phi на основе нейро-активности."""
        if len(self.neuron_indices) == 0:
            return phi_global

        V_local = phi_global[self.neuron_indices]
        self.update_gates(V_local)

        if self.use_nernst and c_global is not None:
            z_ions = np.array([1, 1, -1, 2])
            I = self.compute_current(V_local, c_global, z_ions)
        else:
            I = self.compute_current(V_local)

        C_m = 1.0
        dV = -I * self.dt / C_m
        V_new = V_local + dV
        V_new = np.clip(V_new, -90.0, 60.0)

        phi_updated = phi_global.copy()
        phi_updated[self.neuron_indices] = V_new

        return phi_updated
