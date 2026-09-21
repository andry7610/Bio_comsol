"""bio_bridge.py — Мост между Bio_comsol и GAK-WaveCAD.

Связь: нейрон (ХХ) -> спайк -> акустический сдвиг -> вращение плазмы
       плазма -> температура -> обратный ток -> нейроны

Версия: v0.1
"""

import numpy as np


# --- Константы связи ---
V_THRESHOLD = -20.0       # порог спайка, мВ (как в SynapseLayer)
ACOUSTIC_COUPLING = 0.1   # кэВ·с/м³ на спайк/с (коэффициент преобразования)
ACOUSTIC_DECAY = 0.1      # затухание акустического сдвига, с
FEEDBACK_GAIN = 5e-6      # коэффициент обратной связи, А/м²
TEMP_REF = 1e6            # опорная температура для обратной связи, К


class BioBridge:
    """Мост между нейронной сетью (Bio_comsol) и плазмой (GAK-WaveCAD).

    Parameters
    ----------
    n_neurons : int
        Число нейронов в сети.
    V_threshold : float
        Порог обнаружения спайка (мВ).
    k_acoustic : float
        Коэффициент: спайк/с -> Гц (акустический сдвиг).
    tau_acoustic : float
        Время затухания акустического сдвига (с).
    k_feedback : float
        Коэффициент: T_plasma -> I_feedback (А/м²).
    T_ref : float
        Опорная температура для нормировки обратной связи (К).
    """

    def __init__(self, n_neurons=100,
                 V_threshold=V_THRESHOLD,
                 k_acoustic=ACOUSTIC_COUPLING,
                 tau_acoustic=ACOUSTIC_DECAY,
                 k_feedback=FEEDBACK_GAIN,
                 T_ref=TEMP_REF):

        self.n_neurons = n_neurons
        self.V_threshold = V_threshold
        self.k_acoustic = k_acoustic
        self.tau_acoustic = tau_acoustic
        self.k_feedback = k_feedback
        self.T_ref = T_ref

        # Внутреннее состояние
        self.acoustic_shift = 0.0
        self.spike_count_total = 0
        self.feedback_current = np.zeros(n_neurons)
        self.last_spikes = np.zeros(n_neurons, dtype=bool)
        self.spike_rate = 0.0

    def detect_spikes(self, V_curr, V_prev):
        """Детекция спайков: пересечение порога снизу вверх.

        Parameters
        ----------
        V_curr : np.ndarray
            Текущий потенциал нейронов (мВ).
        V_prev : np.ndarray
            Предыдущий потенциал (мВ).

        Returns
        -------
        np.ndarray (bool)
            Маска спайков.
        """
        V_curr = np.asarray(V_curr)
        V_prev = np.asarray(V_prev)

        spikes = (V_curr > self.V_threshold) & (V_prev <= self.V_threshold)
        return spikes

    def spikes_to_acoustic(self, n_spikes, dt):
        """Преобразование числа спайков в акустический сдвиг частоты.

        acoustic_shift = k_acoustic * spike_rate * exp(-dt / tau)

        Parameters
        ----------
        n_spikes : int
            Число спайков за шаг dt.
        dt : float
            Шаг по времени (с).

        Returns
        -------
        float
            Сдвиг частоты (Гц).
        """
        if dt <= 0:
            return 0.0

        spike_rate = n_spikes / dt
        self.spike_rate = spike_rate

        # Затухание предыдущего сдвига + новый вклад
        self.acoustic_shift *= np.exp(-dt / self.tau_acoustic)
        self.acoustic_shift += self.k_acoustic * spike_rate

        return self.acoustic_shift

    def plasma_to_feedback(self, T_plasma, phase="STABLE"):
        """Обратная связь: температура плазмы -> ток на нейроны.

        I = -k_feedback * tanh(T / T_ref - 1)

        Горячая плазма (T > T_ref) -> тормозящий ток (I < 0).
        Холодная плазма (T < T_ref) -> возбуждающий ток (I > 0).

        Parameters
        ----------
        T_plasma : float
            Температура плазмы (К).
        phase : str
            Фаза плазмы (STABLE / WARNING / CONTACT / BREAKDOWN).

        Returns
        -------
        np.ndarray
            Ток обратной связи на каждый нейрон (А/м²).
        """
        if self.T_ref <= 0:
            return np.zeros(self.n_neurons)

        # Нормировка
        x = T_plasma / self.T_ref - 1.0

        # Коэффициент фазы
        if phase == "BREAKDOWN":
            phase_factor = 3.0
        elif phase == "CONTACT":
            phase_factor = 2.0
        elif phase == "WARNING":
            phase_factor = 1.5
        else:
            phase_factor = 1.0

        I_fb = -self.k_feedback * np.tanh(x) * phase_factor

        self.feedback_current = np.full(self.n_neurons, I_fb)
        return self.feedback_current

    def step(self, V_curr, V_prev, T_plasma, dt, phase="STABLE"):
        """Полный цикл моста за один шаг.

        1. Детекция спайков
        2. Спайки -> акустический сдвиг
        3. Плазма -> обратный ток

        Parameters
        ----------
        V_curr : np.ndarray
            Текущий потенциал нейронов (мВ).
        V_prev : np.ndarray
            Предыдущий потенциал (мВ).
        T_plasma : float
            Температура плазмы (К).
        dt : float
            Шаг по времени (с).
        phase : str
            Фаза плазмы.

        Returns
        -------
        dict
            acoustic_shift, n_spikes, feedback_current, spike_rate.
        """
        # 1. Спайки
        spikes = self.detect_spikes(V_curr, V_prev)
        n_spikes = int(np.sum(spikes))
        self.last_spikes = spikes
        self.spike_count_total += n_spikes

        # 2. Акустический сдвиг
        acoustic_shift = self.spikes_to_acoustic(n_spikes, dt)

        # 3. Обратный ток
        feedback = self.plasma_to_feedback(T_plasma, phase)

        return {
            "acoustic_shift": float(acoustic_shift),
            "n_spikes": n_spikes,
            "spike_rate": float(self.spike_rate),
            "feedback_current": feedback,
            "spikes": spikes,
        }

    def get_acoustic_shift(self):
        """Текущий акустический сдвиг для PlasmaMonitor.run()."""
        return float(self.acoustic_shift)

    def get_feedback_current(self):
        """Текущий ток обратной связи для NeuronLayer.apply_to_graph()."""
        return self.feedback_current.copy()

    def reset(self):
        """Сброс состояния моста."""
        self.acoustic_shift = 0.0
        self.spike_count_total = 0
        self.feedback_current = np.zeros(self.n_neurons)
        self.last_spikes = np.zeros(self.n_neurons, dtype=bool)
        self.spike_rate = 0.0
