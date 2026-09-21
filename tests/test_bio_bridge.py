"""test_bio_bridge.py — 13 тестов моста Bio_comsol <-> GAK-WaveCAD."""

import numpy as np
import pytest

from bio_bridge import BioBridge


class TestInit:
    """Проверка инициализации."""

    def test_default_params(self):
        b = BioBridge()
        assert b.n_neurons == 100
        assert b.acoustic_shift == 0.0
        assert b.spike_count_total == 0
        assert len(b.feedback_current) == 100

    def test_custom_params(self):
        b = BioBridge(n_neurons=50, k_acoustic=0.2, T_ref=2e6)
        assert b.n_neurons == 50
        assert b.k_acoustic == 0.2
        assert b.T_ref == 2e6


class TestSpikeDetection:
    """Детекция спайков."""

    def test_spike_detected(self):
        """Спайк: пересечение порога снизу вверх."""
        b = BioBridge(n_neurons=5)
        V_prev = np.array([-30.0, -25.0, -20.0, -19.0, -10.0])
        V_curr = np.array([-25.0, -20.0, -15.0, -18.0, -5.0])
        spikes = b.detect_spikes(V_curr, V_prev)
        # Нейрон 0: -30 -> -25, оба ниже порога -> нет
        # Нейрон 1: -25 -> -20, точно на пороге -> нет (нужно >)
        # Нейрон 2: -20 -> -15, переход через порог -> да
        # Нейрон 3: -19 -> -18, оба выше -> нет
        # Нейрон 4: -10 -> -5, оба выше -> нет
        assert spikes[2] == True
        assert spikes[0] == False
        assert spikes[3] == False

    def test_no_spike_below_threshold(self):
        """Нет спайка, если оба значения ниже порога."""
        b = BioBridge(n_neurons=3)
        V_prev = np.array([-50.0, -40.0, -30.0])
        V_curr = np.array([-45.0, -35.0, -25.0])
        spikes = b.detect_spikes(V_curr, V_prev)
        assert not np.any(spikes)

    def test_no_spike_both_above(self):
        """Нет спайка, если оба значения выше порога."""
        b = BioBridge(n_neurons=3)
        V_prev = np.array([-10.0, 0.0, 10.0])
        V_curr = np.array([-5.0, 5.0, 15.0])
        spikes = b.detect_spikes(V_curr, V_prev)
        assert not np.any(spikes)


class TestAcousticShift:
    """Акустический сдвиг."""

    def test_spikes_produce_shift(self):
        """Спайки создают положительный акустический сдвиг."""
        b = BioBridge(n_neurons=10, k_acoustic=1.0)
        shift = b.spikes_to_acoustic(n_spikes=5, dt=0.01)
        # spike_rate = 500, shift = 500
        assert shift > 0
        assert shift == pytest.approx(500.0, rel=0.01)

    def test_decay_between_steps(self):
        """Между шагами акустический сдвиг затухает."""
        b = BioBridge(n_neurons=10, k_acoustic=1.0, tau_acoustic=0.1)
        b.spikes_to_acoustic(n_spikes=10, dt=0.01)
        shift1 = b.get_acoustic_shift()

        # Шаг без спайков
        shift2 = b.spikes_to_acoustic(n_spikes=0, dt=0.01)
        assert shift2 < shift1

    def test_no_spikes_zero_shift_after_decay(self):
        """После долгого времени без спайков сдвиг -> 0."""
        b = BioBridge(n_neurons=10, k_acoustic=1.0, tau_acoustic=0.1)
        b.spikes_to_acoustic(n_spikes=100, dt=0.01)
        # 100 шагов без спайков
        for _ in range(200):
            b.spikes_to_acoustic(n_spikes=0, dt=0.01)
        shift = b.get_acoustic_shift()
        assert abs(shift) < 1e-3


class TestFeedback:
    """Обратная связь: плазма -> нейроны."""

    def test_hot_plasma_inhibits(self):
        """Горячая плазма (T > T_ref) -> тормозящий ток (I < 0)."""
        b = BioBridge(n_neurons=5, k_feedback=1e-5, T_ref=1e6)
        I = b.plasma_to_feedback(T_plasma=5e6, phase="STABLE")
        assert np.all(I < 0), f"Горячая плазма должна тормозить, got I={I}"

    def test_cold_plasma_excites(self):
        """Холодная плазма (T < T_ref) -> возбуждающий ток (I > 0)."""
        b = BioBridge(n_neurons=5, k_feedback=1e-5, T_ref=1e6)
        I = b.plasma_to_feedback(T_plasma=5e5, phase="STABLE")
        assert np.all(I > 0), f"Холодная плазма должна возбуждать, got I={I}"

    def test_equilibrium_zero_current(self):
        """T = T_ref -> нулевой ток (равновесие)."""
        b = BioBridge(n_neurons=5, k_feedback=1e-5, T_ref=1e6)
        I = b.plasma_to_feedback(T_plasma=1e6, phase="STABLE")
        assert np.allclose(I, 0.0, atol=1e-12)

    def test_breakdown_amplifies_feedback(self):
        """Фаза BREAKDOWN усиливает обратную связь."""
        b = BioBridge(n_neurons=5, k_feedback=1e-5, T_ref=1e6)
        I_stable = b.plasma_to_feedback(5e6, phase="STABLE")
        I_breakdown = b.plasma_to_feedback(5e6, phase="BREAKDOWN")
        assert np.all(np.abs(I_breakdown) > np.abs(I_stable))


class TestFullStep:
    """Полный цикл step()."""

    def test_step_returns_all_keys(self):
        """step() возвращает все ключи."""
        b = BioBridge(n_neurons=10)
        V_prev = np.full(10, -65.0)
        V_curr = np.array([-65, -65, -65, -15, -65,
                           -65, -65, -65, -65, -65], dtype=float)
        result = b.step(V_curr, V_prev, T_plasma=1e6, dt=0.01, phase="STABLE")
        assert "acoustic_shift" in result
        assert "n_spikes" in result
        assert "spike_rate" in result
        assert "feedback_current" in result
        assert "spikes" in result

    def test_step_spike_count_accumulates(self):
        """Счётчик спайков накапливается между шагами."""
        b = BioBridge(n_neurons=5)
        V_prev = np.full(5, -65.0)
        V_curr = np.array([-10, -65, -65, -65, -65], dtype=float)
        b.step(V_curr, V_prev, T_plasma=1e6, dt=0.01)
        assert b.spike_count_total == 1

        V_prev2 = V_curr.copy()
        V_curr2 = np.array([-10, -10, -65, -65, -65], dtype=float)
        b.step(V_curr2, V_prev2, T_plasma=1e6, dt=0.01)
        assert b.spike_count_total == 2  # только новый спайк (нейрон 1)
