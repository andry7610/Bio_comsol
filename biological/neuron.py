"""biological/neuron.py — Активный нейро-слой на базе квазинейтральной модели.

Реализует упрощённую модель Ходжкина-Хаксли (Na/K каналы) в выделенных узлах.
Поддерживает динамический расчёт равновесных потенциалов по формуле Нернста.
"""

import numpy as np

# Физические константы для формулы Нернста
R = 8.314        # Дж/(моль·К)
T = 310.0        # К (37 °C)
F = 96485.0      # Кл/моль
RT_OVER_F = R * T / F * 1000.0  # мВ (~26.7 мВ при 310 К)


class NeuronLayer:
    """Нейро-слой, накладываемый на граф Bio-COMSOL."""

    def __init__(self, graph, neuron_nodes, dt=1e-3,
                 use_nernst=False, na_index=0, k_index=1):
        """
        Args:
            graph: объект Graph из network/graph.py
            neuron_nodes: список индексов узлов с активными нейронами
            dt: шаг по времени
            use_nernst: если True — E_Na/E_K считаются из концентраций
            na_index: индекс Na в массиве ионов солвера
            k_index: индекс K в массиве ионов солвера
        """
        self.N = graph.N
        self.neuron_indices = np.array(neuron_nodes)
        self.dt = dt
        self.use_nernst = use_nernst
        self.na_index = na_index
        self.k_index = k_index

        # Параметры каналов (статический fallback)
        self.g_Na = 120.0
        self.E_Na = 50.0
        self.g_K = 36.0
        self.E_K = -75.0
        self.g_L = 0.3
        self.E_L = -54.3

        # Ворота в покое
        self.m = np.full(len(neuron_nodes), 0.05)
        self.h = np.full(len(neuron_nodes), 0.6)
        self.n = np.full(len(neuron_nodes), 0.32)

        self.last_current = np.zeros(len(neuron_nodes))

        # Последние динамические потенциалы Нернста
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

