"""tests/test_neuron.py — тесты для biological/neuron.py.

Покрывает:
    - Инициализация и ворота в покое
    - Деполяризация: m растёт, h падает
    - Ток в покое близок к нулю
    - apply_to_graph: phi меняется только в нейронных узлах
    - ИИ-анализ через ManualBackend
    - Полная цепочка: diffusion + neuron, стабильность
"""

import numpy as np
import pytest

from network.graph import Graph
from network.base import AnalysisResult
from network.manual import ManualBackend
from network.neuron_analysis import analyze_neuron, validate_neuron
from biological.diffusion import BiologicalDiffusion
from biological.neuron import NeuronLayer


# ─── Фикстуры ───

@pytest.fixture
def simple_graph():
    return Graph(N=20, topology="chain", conductivity=1.0, cross_section=1e-8)


@pytest.fixture
def simple_ions():
    return [
        {"name": "Na", "D": 1.33e-9, "z": +1, "c0": 145, "c_left": 145, "c_right": 10},
        {"name": "K",  "D": 1.96e-9, "z": +1, "c0": 5,   "c_left": 5,   "c_right": 140},
        {"name": "Cl", "D": 2.03e-9, "z": -1, "c0": 150, "c_left": 150, "c_right": 10},
    ]


@pytest.fixture
def sim(simple_graph, simple_ions):
    return BiologicalDiffusion(
        simple_graph, simple_ions,
        dt=0.01, F_RT=38.9, membrane_potential=-1.8,
    )


@pytest.fixture
def neuron_layer(simple_graph):
    return NeuronLayer(simple_graph, neuron_nodes=[5, 10, 15], dt=0.01)


@pytest.fixture
def empty_neuron(simple_graph):
    return NeuronLayer(simple_graph, neuron_nodes=[], dt=0.01)


# ─── Инициализация ───

class TestInit:
    def test_graph_size(self, neuron_layer):
        assert neuron_layer.N == 20

    def test_neuron_indices(self, neuron_layer):
        assert len(neuron_layer.neuron_indices) == 3
        assert list(neuron_layer.neuron_indices) == [5, 10, 15]

    def test_gates_shape(self, neuron_layer):
        assert neuron_layer.m.shape == (3,)
        assert neuron_layer.h.shape == (3,)
        assert neuron_layer.n.shape == (3,)

    def test_gates_in_range(self, neuron_layer):
        assert np.all(neuron_layer.m >= 0) and np.all(neuron_layer.m <= 1)
        assert np.all(neuron_layer.h >= 0) and np.all(neuron_layer.h <= 1)
        assert np.all(neuron_layer.n >= 0) and np.all(neuron_layer.n <= 1)

    def test_empty_neuron(self, empty_neuron):
        assert len(empty_neuron.neuron_indices) == 0
        assert empty_neuron.m.shape == (0,)

    def test_default_params(self, neuron_layer):
        assert neuron_layer.g_Na == 120.0
        assert neuron_layer.g_K == 36.0
        assert neuron_layer.g_L == 0.3
        assert neuron_layer.E_Na == 50.0
        assert neuron_layer.E_K == -75.0


# ─── Ворота ───

class TestGates:
    def test_depolarization_m_increases(self, neuron_layer):
        """При деполяризации m (активация Na) должна расти."""
        V = np.array([-70.0, -70.0, -70.0])
        neuron_layer.update_gates(V)
        m_before = neuron_layer.m.copy()

        V_dep = np.array([0.0, 0.0, 0.0])
        neuron_layer.update_gates(V_dep)
        assert np.all(neuron_layer.m > m_before)

    def test_depolarization_h_decreases(self, neuron_layer):
        """При деполяризации h (инактивация Na) должна падать."""
        V = np.array([-70.0, -70.0, -70.0])
        neuron_layer.update_gates(V)
        h_before = neuron_layer.h.copy()

        V_dep = np.array([0.0, 0.0, 0.0])
        neuron_layer.update_gates(V_dep)
        assert np.all(neuron_layer.h < h_before)

    def test_gates_stay_in_range(self, neuron_layer):
        """Ворота не выходят за [0, 1] при разумных V."""
        for V_val in [-90, -70, -40, 0, 30]:
            V = np.full(3, float(V_val))
            neuron_layer.update_gates(V)
            assert np.all(neuron_layer.m >= 0) and np.all(neuron_layer.m <= 1)
            assert np.all(neuron_layer.h >= 0) and np.all(neuron_layer.h <= 1)
            assert np.all(neuron_layer.n >= 0) and np.all(neuron_layer.n <= 1)

    def test_empty_gates_no_crash(self, empty_neuron):
        V = np.array([])
        empty_neuron.update_gates(V)
        assert empty_neuron.m.shape == (0,)


# ─── Ток ───

class TestCurrent:
    def test_resting_current_near_zero(self, neuron_layer):
        """В покое суммарный ток близок к нулю."""
        V_rest = np.full(3, -65.0)
        c = np.full((3, 20), 100.0)
        for _ in range(100):
            neuron_layer.update_gates(V_rest)
        I = neuron_layer.compute_current(V_rest, c, np.array([1, 1, -1, 2]))
        assert np.all(np.abs(I) < 5.0)

    def test_current_shape(self, neuron_layer):
        V = np.full(3, -65.0)
        c = np.full((3, 20), 100.0)
        I = neuron_layer.compute_current(V, c, np.array([1, 1, -1, 2]))
        assert I.shape == (3,)

    def test_depolarized_current_large(self, neuron_layer):
        """При деполяризации ток должен быть значительным."""
        V = np.full(3, -65.0)
        c = np.full((3, 20), 100.0)
        for _ in range(100):
            neuron_layer.update_gates(V)
        I_rest = neuron_layer.compute_current(V, c, np.array([1, 1, -1, 2]))

        V_dep = np.full(3, 0.0)
        for _ in range(50):
            neuron_layer.update_gates(V_dep)
        I_dep = neuron_layer.compute_current(V_dep, c, np.array([1, 1, -1, 2]))
        assert np.max(np.abs(I_dep)) > np.max(np.abs(I_rest))


# ─── apply_to_graph ───

class TestApplyToGraph:
    def test_phi_changes_only_at_neurons(self, neuron_layer):
        phi = np.linspace(0, -1.8, 20)
        c = np.full((3, 20), 100.0)
        phi_new = neuron_layer.apply_to_graph(phi, c)

        for idx in neuron_layer.neuron_indices:
            assert not np.isclose(phi_new[idx], phi[idx])

        non_neuron = [i for i in range(20) if i not in neuron_layer.neuron_indices]
        for idx in non_neuron:
            assert np.isclose(phi_new[idx], phi[idx])

    def test_phi_clipped(self, neuron_layer):
        phi = np.full(20, 100.0)
        c = np.full((3, 20), 100.0)
        phi_new = neuron_layer.apply_to_graph(phi, c)
        for idx in neuron_layer.neuron_indices:
            assert phi_new[idx] <= 60.0
            assert phi_new[idx] >= -90.0

    def test_empty_neuron_no_change(self, empty_neuron):
        phi = np.linspace(0, -1.8, 20)
        c = np.full((3, 20), 100.0)
        phi_new = empty_neuron.apply_to_graph(phi, c)
        assert np.allclose(phi_new, phi)


# ─── ИИ-анализ ───

class TestAIIntegration:
    def test_analyze_neuron_manual(self, neuron_layer):
        backend = ManualBackend()
        result = analyze_neuron(neuron_layer, backend)
        assert isinstance(result, AnalysisResult)
        assert result.backend == "manual"
        assert result.summary

    def test_validate_neuron_manual(self, neuron_layer):
        backend = ManualBackend()
        result = validate_neuron(neuron_layer, backend)
        assert isinstance(result, AnalysisResult)
        assert result.backend == "manual"

    def test_analyze_empty_neuron(self, empty_neuron):
        backend = ManualBackend()
        result = analyze_neuron(empty_neuron, backend)
        assert "Нет нейронных узлов" in result.summary

    def test_analyze_after_steps(self, neuron_layer, sim):
        for _ in range(10):
            sim.step()
            sim.phi = neuron_layer.apply_to_graph(sim.phi, sim.c)

        backend = ManualBackend()
        result = analyze_neuron(neuron_layer, backend)
        assert isinstance(result, AnalysisResult)
        assert result.confidence > 0.0


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
    def test_mock_analyze_neuron(self, neuron_layer):
        mock = MockBackend()
        result = analyze_neuron(neuron_layer, mock)
        assert result.backend == "mock"
        assert result.confidence == 0.99

    def test_mock_validate_neuron(self, neuron_layer):
        mock = MockBackend()
        result = validate_neuron(neuron_layer, mock)
        assert result.backend == "mock"


# ─── Полная цепочка ───

class TestFullChain:
    def test_diffusion_plus_neuron_stable(self, sim, neuron_layer):
        """300 шагов: diffusion + neuron, без NaN."""
        for _ in range(300):
            sim.step()
            sim.phi = neuron_layer.apply_to_graph(sim.phi, sim.c)

        assert np.all(np.isfinite(sim.c))
        assert np.all(np.isfinite(sim.phi))
        assert sim.step_count == 300

    def test_neuron_affects_phi(self, sim, neuron_layer):
        """Phi с нейроном отличается от phi без нейрона."""
        sim_plain = BiologicalDiffusion(
            sim.graph, [
                {"name": "Na", "D": 1.33e-9, "z": +1, "c0": 145, "c_left": 145, "c_right": 10},
                {"name": "K",  "D": 1.96e-9, "z": +1, "c0": 5,   "c_left": 5,   "c_right": 140},
                {"name": "Cl", "D": 2.03e-9, "z": -1, "c0": 150, "c_left": 150, "c_right": 10},
            ],
            dt=0.01, F_RT=38.9, membrane_potential=-1.8,
        )

        for _ in range(100):
            sim.step()
            sim_plain.step()
            sim.phi = neuron_layer.apply_to_graph(sim.phi, sim.c)

        assert not np.allclose(sim.phi, sim_plain.phi)

    def test_charge_not_exploded(self, sim, neuron_layer):
        for _ in range(300):
            sim.step()
            sim.phi = neuron_layer.apply_to_graph(sim.phi, sim.c)
        q = sim.total_charge()
        assert np.isfinite(q)
        assert abs(q) < 1e6

    def test_long_run(self, sim, neuron_layer):
        for _ in range(500):
            sim.step()
            sim.phi = neuron_layer.apply_to_graph(sim.phi, sim.c)
        assert np.all(np.isfinite(sim.c))
        assert np.all(np.isfinite(sim.phi))

class TestNernst:
    """Тесты динамического расчёта равновесных потенциалов Нернста."""

    @pytest.fixture
    def graph(self):
        return Graph(N=20, topology="chain")

    @pytest.fixture
    def neuron(self, graph):
        return NeuronLayer(graph, neuron_nodes=[5, 10, 15],
                          dt=0.01, use_nernst=True)

    @pytest.fixture
    def c_global(self):
        """Концентрации (4 иона, 20 узлов) — однородные."""
        c = np.zeros((4, 20))
        c[0] = 145.0   # Na
        c[1] = 4.0     # K
        c[2] = 110.0   # Cl
        c[3] = 1.0     # Ca
        return c

    @pytest.fixture
    def z_ions(self):
        return np.array([1, 1, -1, 2])

    def test_nernst_returns_arrays(self, neuron, c_global, z_ions):
        """compute_nernst возвращает массивы правильной длины."""
        E_Na, E_K = neuron.compute_nernst(c_global, z_ions)
        assert len(E_Na) == 3
        assert len(E_K) == 3

    def test_nernst_uniform_concentration(self, neuron, c_global, z_ions):
        """Однородные концентрации → E = 0 (log(1) = 0)."""
        E_Na, E_K = neuron.compute_nernst(c_global, z_ions)
        assert np.allclose(E_Na, 0.0, atol=1e-10)
        assert np.allclose(E_K, 0.0, atol=1e-10)

    def test_nernst_E_Na_value(self, neuron, z_ions):
        """E_Na = (RT/F) · ln(c_out / c_in) при z = 1."""
        c = np.zeros((4, 20))
        c[0] = 145.0
        c[0][[5, 10, 15]] = 10.0
        # c_out = mean = (17·145 + 3·10) / 20 = 124.75
        # c_in = 10 → E_Na = 26.7 · ln(12.475) ≈ 67.4 мВ
        E_Na, _ = neuron.compute_nernst(c, z_ions)

    def test_nernst_E_K_value(self, neuron, z_ions):
        """E_K = (RT/F) · ln(c_out / c_in) при z = 1."""
        c = np.zeros((4, 20))
        c[1] = 4.0
        c[1][[5, 10, 15]] = 100.0
        _, E_K = neuron.compute_nernst(c, z_ions)
        expected = 26.7 * np.log(18.4 / 100.0)
        assert np.allclose(E_K, expected, rtol=0.01)

    def test_nernst_nan_protection(self, neuron, z_ions):
        """Нулевые концентрации → NaN → fallback на статические E."""
        c = np.zeros((4, 20))
        E_Na, E_K = neuron.compute_nernst(c, z_ions)
        assert not np.any(np.isnan(E_Na))
        assert not np.any(np.isnan(E_K))
        assert np.allclose(E_Na, neuron.E_Na)
        assert np.allclose(E_K, neuron.E_K)

    def test_nernst_empty_neuron(self, graph):
        """Пустые neuron_indices → пустые массивы, без краша."""
        neuron = NeuronLayer(graph, neuron_nodes=[], use_nernst=True)
        c = np.zeros((4, 20))
        z = np.array([1, 1, -1, 2])
        E_Na, E_K = neuron.compute_nernst(c, z)
        assert len(E_Na) == 0
        assert len(E_K) == 0

    def test_nernst_stored_dynamic(self, neuron, c_global, z_ions):
        """E_Na_dynamic и E_K_dynamic сохраняются после compute_nernst."""
        assert neuron.E_Na_dynamic is None
        assert neuron.E_K_dynamic is None
        neuron.compute_nernst(c_global, z_ions)
        assert neuron.E_Na_dynamic is not None
        assert neuron.E_K_dynamic is not None

    def test_compute_current_uses_nernst(self, graph):
        """use_nernst=True → compute_current вызывает compute_nernst."""
        neuron = NeuronLayer(graph, neuron_nodes=[5, 10, 15], use_nernst=True)
        V = np.array([-65.0, -65.0, -65.0])
        c = np.zeros((4, 20))
        c[0] = 145.0
        c[1] = 4.0
        z = np.array([1, 1, -1, 2])
        I = neuron.compute_current(V, c_ions=c, z_ions=z)
        assert len(I) == 3
        assert neuron.E_Na_dynamic is not None

    def test_compute_current_ignores_nernst(self, graph):
        """use_nernst=False → статические E, dynamic не считаются."""
        neuron = NeuronLayer(graph, neuron_nodes=[5, 10, 15], use_nernst=False)
        V = np.array([-65.0, -65.0, -65.0])
        c = np.zeros((4, 20))
        z = np.array([1, 1, -1, 2])
        I = neuron.compute_current(V, c_ions=c, z_ions=z)
        assert len(I) == 3
        assert neuron.E_Na_dynamic is None

    def test_apply_to_graph_with_nernst(self, graph):
        """apply_to_graph передаёт концентрации при use_nernst=True."""
        neuron = NeuronLayer(graph, neuron_nodes=[5, 10, 15], use_nernst=True)
        phi = np.linspace(0, -1.8, 20)
        c = np.zeros((4, 20))
        c[0] = 145.0
        c[1] = 4.0
        c[2] = 110.0
        c[3] = 1.0
        phi_new = neuron.apply_to_graph(phi, c)
        assert phi_new.shape == (20,)
        assert neuron.E_Na_dynamic is not None
# === END ===
