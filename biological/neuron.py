"""
biological/neuron.py — Активный нейро-слой на базе квазинейтральной модели.
Версия: v0.1 (совместима с biological v0.5)

Реализует:
- Упрощённую модель Ходжкина-Хаксли (Na/K каналы) в выделенных узлах.
- Расчёт мембранного тока и потенциала действия.
- Обратную связь: обновлённый потенциал (phi) для основного солвера.
"""

import numpy as np
from scipy.integrate import odeint

class NeuronLayer:
    """
    Нейро-слой, который накладывается на граф Bio-COMSOL.
    
    Логика:
    1. Основной солвер (BiologicalDiffusion) считает пассивный транспорт.
    2. NeuronLayer берёт концентрации и потенциал, активирует каналы в узлах-нейронах.
    3. Считает локальный потенциал действия.
    4. Возвращает обновлённый phi для следующего шага солвера.
    """

    def __init__(self, graph, neuron_nodes, dt=1e-3):
        """
        Args:
            graph: объект Graph из common/graph_utils.py
            neuron_nodes: список индексов узлов, где есть активные нейроны (например, [10, 20, 30])
            dt: шаг по времени (должен совпадать с основным солвером)
        """
        self.N = graph.N
        self.neuron_indices = np.array(neuron_nodes)
        self.dt = dt
        
        # Параметры каналов (в безразмерных единицах или масштабированных под F_RT=1.0)
        # g_Na, g_K — проводимости; E_Na, E_K — равновесные потенциалы
        self.g_Na = 120.0
        self.E_Na = 50.0   # мВ (или в твоих единицах потенциала)
        self.g_K = 36.0
        self.E_K = -75.0
        self.g_L = 0.3
        self.E_L = -54.3

        # Переменные состояния для каждого нейрона (m, h, n)
        # Инициализируем в покое
        self.m = np.full(len(neuron_nodes), 0.05)  # активация Na
        self.h = np.full(len(neuron_nodes), 0.6)   # инактивация Na
        self.n = np.full(len(neuron_nodes), 0.32)  # активация K

        # Для хранения последнего рассчитанного тока (для логов)
        self.last_current = np.zeros(len(neuron_nodes))

    @staticmethod
    def _alpha_m(V):
        # V в мВ
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
        m_new, h_new, n_new = [], [], []
        for V in V_neurons:
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

            # Эйлер для ворот (можно заменить на odeint для большей точности)
            m = m_inf + (self.m[0] - m_inf) * np.exp(-self.dt / tau_m) if len(self.m) > 0 else m_inf
            h = h_inf + (self.h[0] - h_inf) * np.exp(-self.dt / tau_h) if len(self.h) > 0 else h_inf
            n = n_inf + (self.n[0] - n_inf) * np.exp(-self.dt / tau_n) if len(self.n) > 0 else n_inf

            m_new.append(m)
            h_new.append(h)
            n_new.append(n)

        self.m = np.array(m_new)
        self.h = np.array(h_new)
        self.n = np.array(n_new)

    def compute_current(self, V_neurons, c_ions, z_ions):
        """
        Считает мембранный ток I_ion для нейронов.
        V_neurons: потенциал в узлах нейронов (из основного солвера)
        c_ions: концентрации (для расчёта равновесных потенциалов по Нернсту, если нужно)
        z_ions: заряды
        
        Для простоты пока используем фиксированные E_Na, E_K.
        В будущем можно заменить на динамические E_k из Нернста.
        """
        I_Na = self.g_Na * (self.m ** 3) * self.h * (V_neurons - self.E_Na)
        I_K = self.g_K * (self.n ** 4) * (V_neurons - self.E_K)
        I_L = self.g_L * (V_neurons - self.E_L)
        
        I_total = I_Na + I_K + I_L
        self.last_current = I_total
        return I_total

    def apply_to_graph(self, phi_global, c_global):
        """
        Основной метод: обновляет глобальный потенциал phi на основе нейро-активности.
        
        Логика обратной связи:
        1. Берём phi в узлах neuron_indices.
        2. Считаем токи I.
        3. Обновляем V_new = V_old + I * R_eff (или просто добавляем вклад в потенциал).
        4. Вставляем обновлённые значения обратно в phi_global.
        
        Это создаёт волну потенциала действия, которая затем влияет на пассивный транспорт ионов.
        """
        if len(self.neuron_indices) == 0:
            return phi_global

        # 1. Получаем локальные потенциалы
        V_local = phi_global[self.neuron_indices]

        # 2. Обновляем ворота
        self.update_gates(V_local)

        # 3. Считаем ток
        # Примечание: c_global и z_ions нужны, если захочешь сделать E_k динамическим
        I = self.compute_current(V_local, c_global, np.array([1, 1, -1, 2])) # Na, K, Cl, Ca

        # 4. Обновляем потенциал (простая RC-модель)
        # dV/dt = -I / C_m. Здесь dt уже известен, C_m можно подобрать.
        C_m = 1.0  # ёмкость мембраны (подбирается под масштаб времени)
        dV = -I * self.dt / C_m
        
        V_new = V_local + dV

        # Ограничиваем потенциал, чтобы не улетал в космос (физические границы)
        V_new = np.clip(V_new, -90.0, 60.0)

        # 5. Вставляем обратно в глобальный phi
        phi_updated = phi_global.copy()
        phi_updated[self.neuron_indices] = V_new

        return phi_updated
