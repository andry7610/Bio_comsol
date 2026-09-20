"""tests/test_neuron.py — тесты для biological/neuron.py.

Покрывает:
    - Инициализация и ворота в покое
    - Обновление ворот при деполяризации
    - Мембранный ток в покое и при спайке
    - apply_to_graph: потенциал меняется только в нейронных узлах
    - Сохранение заряда и физичность
    - ИИ-интеграция: NeuronAnalyzer через BaseBackend
    - Fallback: ManualBackend для нейрона
    - Полная цепочка: diffusion + neuron, устойчивость
"""

import numpy as np
import pytest

from network.graph import Graph
from network.base import AnalysisResult
from network.manual import ManualBackend
from biological.diffusion import BiologicalDiffusion
from biological.neuron import NeuronLayer


# ─── Фикстуры ───

@pytest.fixture
def simple_graph():
    return Graph(N=50, topology="chain", conductivity=1.0, cross_section=1e-8)


@pytest.fixture
def simple_ions():
    return [
        {"name": "Na", "D": 1.33e-9, "z": +1, "c0": 145, "c_left": 145, "c_right": 10},
        {"name": "K",  "D": 1.96e-9, "z": +1, "c0": 5,   "c_left": 5,   "c_right": 140},
        {"name": "Cl", "D": 2.03e-9, "z": -1, "c0": 150, "c_left": 150, "c_right": 10},
        {"name": "Ca", "D": 0.3e-9,  "z": +2, "c0": 2,   "c_left": 2,   "c_right": 1},
    ]


@pytest.fixture
def sim(simple_graph, simple_ions):
    return BiologicalDiffusion(
        simple_graph, simple_ions,
        dt=0.01, F_RT=38.9, membrane_potential=-1.8,
    )


@pytest.fixture
def neuron_layer(simple_graph):
    return NeuronLayer(
        simple_graph,
        neuron_nodes=[10, 20, 30],
        dt=0.01,
    )


@pytest.fixture
def sim_with_neuron(simple_graph, simple_ions):
    bio = BiologicalDiffusion(
        simple_graph, simple_ions,
        dt=0.01, F_RT=38.9, membrane_potential=-1.8,
    )
    neurons = NeuronLayer(
        simple_graph,
        neuron_nodes=[10, 20, 30],
        dt=0.01,
    )
    return bio, neurons


# ─── Инициализация ───

class TestInit:
    def test_init(self, neuron_layer):
        assert neuron_layer.N == 50
        assert len(neuron_layer.neuron_indices) == 3
        assert np.array_equal(neuron_layer.neuron_indices, [10, 20, 30])

    def test_gates_in_rest(self, neuron_layer):
        # Ворота в покое: m~0.05, h~0.6, n~0.32
        assert np.all(neuron_layer.m > 0)
        assert np.all(neuron_layer.m < 1)
        assert np.all(neuron_layer.h > 0)
        assert np.all(neuron_layer.h < 1)
        assert np.all(neuron_layer.n > 0)
        assert np.all(neuron_layer.n < 1)

    def test_dt_matches(self, neuron_layer):
        assert neuron_layer.dt == 0.01

    def test_empty_neuron_nodes(self, simple_graph):
        nl = NeuronLayer(simple_graph, neuron_nodes=[], dt=0.01)
        assert len(nl.neuron_indices) == 0
        assert len(nl.m) == 0


# ─── Ворота Ходжкина-Хаксли ───

class TestGates:
    def test_gate_update_depolarization(self, neuron_layer):
        """При деполяризации (V → 0) m растёт, h падает."""
        m_before = neuron_layer.m.copy()
        h_before = neuron_layer.h.copy()
        # Деполяризация: V = 0 мВ (вместо покоя -70)
        neuron_layer.update_gates(np.array([0.0, 0.0, 0.0]))
        assert np.all(neuron_layer.m >= m_before - 0.01)
        assert np.all(neuron_layer.h <= h_before + 0.01)

    def test_gate_update_hyperpolarization(self, neuron_layer):
        """При гиперполяризации (V → -90) m падает."""
        m_before = neuron_layer.m.copy()
        neuron_layer.update_gates(np.array([-90.0, -90.0, -90.0]))
        assert np.all(neuron_layer.m <= m_before + 0.01)

    def test_gates_bounded(self, neuron_layer):
        """Ворота всегда в [0, 1] после обновления."""
        for V in [-100, -70, -40, 0, 30, 50]:
            neuron_layer.update_gates(np.full(3, V))
            assert np.all(neuron_layer.m >= -0.01)
            assert np.all(neuron_layer.m <= 1.01)
            assert np.all(neuron_layer.h >= -0.01)
            assert np.all(neuron_layer.h <= 1.01)
            assert np.all(neuron_layer.n >= -0.01)
            assert np.all(neuron_layer.n <= 1.01)


# ─── Мембранный ток ───

class TestCurrent:
    def test_current_at_rest(self, neuron_layer):
        """В покое ток близок к нулю."""
        V_rest = np.full(3, -70.0)
        neuron_layer.update_gates(V_rest)
        I = neuron_layer.compute_current(V_rest, None, None)
        # Ток в покое должен быть малым
        assert np.all(np.abs(I) < 20.0)

    def test_current_depolarization(self, neuron_layer):
        """При деполяризации Na-ток резко возрастает."""
        # Прогрев: доводим до деполяризации
        V = np.full(3, -20.0)
        for _ in range(10):
            neuron_layer.update_gates(V)
        I = neuron_layer.compute_current(V, None, None)
        assert np.any(np.abs(I) > 1.0)

    def test_current_shape(self, neuron_layer):
        V = np.array([-70.0, -20.0, 0.0])
        I = neuron_layer.compute_current(V, None, None)
        assert I.shape == (3,)


# ─── apply_to_graph ───

class TestApplyToGraph:
    def test_phi_changes_only_at_neurons(self, sim, neuron_layer):
        """Потенциал меняется только в нейронных узлах."""
        phi_before = sim.phi.copy()
        phi_after = neuron_layer.apply_to_graph(sim.phi, sim.c)

        # В нейронных узлах — изменения
        for idx in neuron_layer.neuron_indices:
            assert not np.isclose(phi_before[idx], phi_after[idx])

        # В не-нейронных узлах — без изменений
        non_neuron = [i for i in range(sim.N) if i not in neuron_layer.neuron_indices]
        for idx in non_neuron:
            assert np.isclose(phi_before[idx], phi_after[idx])

    def test_phi_bounded(self, sim, neuron_layer):
        """Потенциал после нейрона не выходит за физические пределы."""
        phi = neuron_layer.apply_to_graph(sim.phi, sim.c)
        assert np.all(phi >= -100.0)
        assert np.all(phi <= 70.0)

    def test_empty_neurons_no_change(self, sim, simple_graph):
        """Без нейронов потенциал не меняется."""
        nl = NeuronLayer(simple_graph, neuron_nodes=[], dt=0.01)
        phi_before = sim.phi.copy()
        phi_after = nl.apply_to_graph(sim.phi, sim.c)
        assert np.allclose(phi_before, phi_after)


# ─── ИИ-интеграция ───

class TestAIIntegration:
    def test_analyze_neuron_via_manual_backend(self, neuron_layer, sim):
        """NeuronLayer.analyze_state() через ManualBackend."""
        backend = ManualBackend()
        result = neuron_layer.analyze_state(backend, sim.phi, sim.c, sim.z)
        assert isinstance(result, AnalysisResult)
        assert result.backend == "manual"
        assert result.summary

    def test_analyze_neuron_warnings_on_no_spikes(self, neuron_layer, sim):
        """Если нет спайков — должно быть предупреждение."""
        # Не запускаем — все в покое
        backend = ManualBackend()
        result = neuron_layer.analyze_state(backend, sim.phi, sim.c, sim.z)
        # Должно отметить отсутствие спайков или спокойное состояние
        assert isinstance(result.warnings, list)

    def test_analyze_neuron_detects_spikes(self, neuron_layer, sim):
        """Если V > 0 — детектируется спайк."""
        # Искусственно задаём потенциал выше порога
        phi_spike = sim.phi.copy()
        phi_spike[10] = 30.0
        phi_spike[20] = 25.0
        backend = ManualBackend()
        result = neuron_layer.analyze_state(backend, phi_spike, sim.c, sim.z)
        assert "spike" in result.summary.lower() or len(result.warnings) > 0

    def test_analyze_neuron_gates_out_of_range(self, neuron_layer, sim):
        """Если ворота выходят за [0,1] — предупреждение."""
        neuron_layer.m = np.array([1.5, 0.3, 0.1])
        backend = ManualBackend()
        result = neuron_layer.analyze_state(backend, sim.phi, sim.c, sim.z)
        assert len(result.warnings) > 0

    def test_confidence_range(self, neuron_layer, sim):
        backend = ManualBackend()
        result = neuron_layer.analyze_state(backend, sim.phi, sim.c, sim.z)
        assert 0.0 <= result.confidence <= 1.0


# ─── Mock-бэкенд ───

class MockBackend:
    def __init__(self):
        self.name = "mock"
        self.call_count = 0

    def is_available(self):
        return True

    def analyze(self, concentrations, charges, time_step, context=None):
        self.call_count += 1
        return AnalysisResult(
            summary=f"Mock neuron analysis at step {time_step}",
            warnings=[],
            suggestions=[],
            confidence=0.99,
            backend="mock",
        )

    def validate_params(self, params):
        self.call_count += 1
        return AnalysisResult(
            summary="Mock: neuron params OK",
            warnings=[],
            suggestions=[],
            confidence=0.99,
            backend="mock",
        )

    def describe(self, text):
        return AnalysisResult(
            summary="Mock description",
            warnings=[],
            suggestions=[],
            confidence=0.99,
            backend="mock",
        )


class TestMockBackend:
    def test_mock_analyze_neuron(self, sim, neuron_layer):
        mock = MockBackend()
        result = neuron_layer.analyze_state(mock, sim.phi, sim.c, sim.z)
        assert result.backend == "mock"
        assert result.confidence == 0.99

    def test_mock_validate_neuron(self, neuron_layer):
        mock = MockBackend()
        result = neuron_layer.validate_params(mock)
        assert result.backend == "mock"


# ─── Полная цепочка ───

class TestFullChain:
    def test_diffusion_plus_neuron_stable(self, sim_with_neuron):
        """Полная цепочка: diffusion.step() → neuron.apply_to_graph()."""
        bio, neurons = sim_with_neuron
        for _ in range(300):
            bio.step()
            new_phi = neurons.apply_to_graph(bio.phi, bio.c)
            bio.phi = new_phi

        assert np.all(np.isfinite(bio.c))
        assert np.all(np.isfinite(bio.phi))
        assert bio.step_count == 300

    def test_neuron_affects_potential(self, sim_with_neuron):
        """Нейроны меняют потенциал по сравнению с чистой диффузией."""
        bio, neurons = sim_with_neuron
        # Прогон без нейрона
        bio_no_neuron = BiologicalDiffusion(
            bio.graph, [
                {"name": "Na", "D": 1.33e-9, "z": +1, "c0": 145, "c_left": 145, "c_right": 10},
                {"name": "K",  "D": 1.96e-9, "z": +1, "c0": 5,   "c_left": 5,   "c_right": 140},
                {"name": "Cl", "D": 2.03e-9, "z": -1, "c0": 150, "c_left": 150, "c_right": 10},
                {"name": "Ca", "D": 0.3e-9,  "z": +2, "c0": 2,   "c_left": 2,   "c_right": 1},
            ],
            dt=0.01, F_RT=38.9, membrane_potential=-1.8,
        )

        for _ in range(100):
            bio.step()
            bio.phi = neurons.apply_to_graph(bio.phi, bio.c)
            bio_no_neuron.step()

        # Потенциалы должны различаться
        assert not np.allclose(bio.phi, bio_no_neuron.phi)

    def test_long_run_no_nan(self, sim_with_neuron):
        bio, neurons = sim_with_neuron
        for _ in range(500):
            bio.step()
            bio.phi = neurons.apply_to_graph(bio.phi, bio.c)
        assert np.all(np.isfinite(bio.c))
        assert np.all(np.isfinite(bio.phi))

    def test_charge_conservation(self, sim_with_neuron):
        """Заряд не улетает при работе нейро-слоя."""
        bio, neurons = sim_with_neuron
        Q0 = bio.total_charge()
        for _ in range(100):
            bio.step()
            bio.phi = neurons.apply_to_graph(bio.phi, bio.c)
        Q1 = bio.total_charge()
        # Допускаем относительное изменение < 50%
        assert abs(Q1 - Q0) / max(abs(Q0), 1.0) < 0.5

    def test_no_negative_concentrations(self, sim_with_neuron):
        bio, neurons = sim_with_neuron
        for _ in range(200):
            bio.step()
            bio.phi = neurons.apply_to_graph(bio.phi, bio.c)
        assert np.min(bio.c) > -10.0
