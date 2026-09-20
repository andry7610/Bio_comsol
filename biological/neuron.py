"""biological/neuron.py — модель нейрона (Hodgkin–Huxley)."""

import numpy as np

class NeuronLayer:
    def __init__(self, n_neurons, dt=1e-3, g_Na=120.0, g_K=36.0,
                 E_Na=50.0, E_K=-75.0, E_L=-54.4, g_L=0.3):
        self.n_neurons = n_neurons
        self.dt = dt
        self.g_Na = g_Na
        self.g_K = g_K
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.g_L = g_L

        self.m = np.zeros(n_neurons)
        self.h = np.zeros(n_neurons)
        self.n = np.zeros(n_neurons)

        self.V = np.full(n_neurons, -65.0)  # mV
        self.last_current = np.zeros(n_neurons)
        self.neuron_indices = np.arange(n_neurons)

    def _alpha_m(self, V):
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-0.1 * (V + 40.0)))

    def _beta_m(self, V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def _alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def _beta_h(self, V):
        return 1.0 / (1.0 + np.exp(-0.1 * (V + 35.0)))

    def _alpha_n(self, V):
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-0.1 * (V + 55.0)))

    def _beta_n(self, V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def update_gates(self, V_neurons):
        for idx, V in enumerate(V_neurons):
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

    def compute_currents(self):
        I_Na = self.g_Na * (self.m ** 3) * self.h * (self.V - self.E_Na)
        I_K = self.g_K * (self.n ** 4) * (self.V - self.E_K)
        I_L = self.g_L * (self.V - self.E_L)
        total = I_Na + I_K + I_L
        self.last_current = total
        return total

    def step(self, external_current=None):
        if external_current is None:
            external_current = np.zeros(self.n_neurons)
        self.update_gates(self.V)
        I = self.compute_currents()
        dV = (external_current - I) * self.dt
        self.V += dV
