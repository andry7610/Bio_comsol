"""tests/test_synapse.py — Тесты SynapseLayer (v0.9)."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from network.graph import Graph
from biological.synapse import SynapseLayer
from biological.neuron import NeuronLayer
from biological.diffusion import BiologicalDiffusion
from network import AnalysisResult


# --- Инициализация ---

class TestInit:

    def test_init_empty(self):
        graph = Graph(N=20, topology="chain")
        syn = SynapseLayer(graph, [], dt=0.01)
        assert syn.n_syn == 0
        assert len(syn.s) == 0

    def test_init_with_synapses(self):
        graph = Graph(N=20, topology="chain")
        synapses = [
            {"pre": 5, "post": 10, "weight": 0.5, "E_syn": 0.0, "tau_syn": 3.0},
            {"pre": 10, "post": 15, "weight": 0.3, "E_syn": -80.0, "tau_syn": 5.0},
        ]
        syn = SynapseLayer(graph, synapses, dt=0.01)
        assert syn.n_syn == 2
        assert np.allclose(syn.weights, [0.5, 0.3])
        assert np.allclose(syn.E_syn, [0.0, -80.0])
        assert np.allclose(syn.s, [0.0, 0.0])

    def test_pre_post_indices(self):
        graph = Graph(N=20, topology="chain")
        synapses = [{"pre": 5, "post": 10, "weight": 0.5, "E_syn": 0.0, "tau_syn": 3.0}]
        syn = SynapseLayer(graph, synapses, dt=0.01)
        assert syn.pre[0] == 5
        assert syn.post[0] == 10

    def test_gates_initialized_zero(self):
        graph = Graph(N=20, topology="chain")
        synapses = [{"pre": 5, "post": 10, "weight": 0.5, "E_syn": 0.0, "tau_syn": 3.0}]
        syn = SynapseLayer(graph, synapses, dt=0.01)
        assert np.all(syn.s == 0.0)


# --- Обнаружение спайков ---

class TestSpikeDetection:

    @pytest.fixture
    def syn(self):
        graph = Graph(N=20, topology="chain")
        synapses = [{"pre": 5, "post": 10, "weight": 0.5, "E_syn": 0.0, "tau_syn": 3.0}]
        return SynapseLayer(graph, synapses, dt=0.01, V_threshold=-20.0)

    def test_no_spike_below_threshold(self, syn):
        phi = np.full(20, -50.0)
        syn.step(phi)
        syn.step(phi)
        assert syn.s[0] == 0.0

    def test_spike_detected_on_rising_edge(self, syn):
        phi1 = np.full(20, -50.0)
        syn.step(phi1)
        phi2 = phi1.copy()
        phi2[5] = 0.0
        syn.step(phi2)
        assert syn.s[0] > 0.0

    def test_no_spike_staying_above(self, syn):
        phi = np.full(20, 0.0)
        syn.step(phi)
        syn.step(phi)
        assert syn.s[0] == 0.0

    def test_multiple_spikes(self):
        graph = Graph(N=20, topology="chain")
        synapses = [
            {"pre": 5, "post": 10, "weight": 0.5, "E_syn": 0.0, "tau_syn": 3.0},
            {"pre": 10, "post": 15, "weight": 0.3, "E_syn": 0.0, "tau_syn": 5.0},
        ]
        syn = SynapseLayer(graph, synapses, dt=0.01, V_threshold=-20.0)
        phi = np.full(20, -50.0)
        syn.step(phi)
        phi[5] = 0.0
        phi[10] = 0.0
        syn.step(phi)
        assert syn.s[0] > 0.0
        assert syn.s[1] > 0.0


# --- Затухание ---

class TestDecay:

    def test_exponential_decay(self):
        graph = Graph(N=20, topology="chain")
        synapses = [{"pre": 5, "post": 10, "weight": 0.5, "E_syn": 0.0, "tau_syn": 3.0}]
        syn = SynapseLayer(graph, synapses, dt=0.01, V_threshold=-20.0)
        syn.s[0] = 1.0
        phi = np.full(20, -50.0)
        syn.step(phi)
        s_before = syn.s[0]
        syn.step(phi)
        expected = s_before * np.exp(-0.01 / 3.0)
        assert abs(syn.s[0] - expected) < 1e-10

    def test_decay_to_zero(self):
        graph = Graph(N=20, topology="chain")
        synapses = [{"pre": 5, "post": 10, "weight": 0.5, "E_syn": 0.0, "tau_syn": 1.0}]
        syn = SynapseLayer(graph, synapses, dt=0.1, V_threshold=-20.0)
        syn.s[0] = 1.0
        phi = np.full(20, -50.0)
        syn.step(phi)
        for _ in range(300):
            syn.step(phi)
        assert syn.s[0] < 1e-10


# --- Синаптический ток ---

class TestSynapticCurrent:

    def test_no_current_when_gate_zero(self):
        graph = Graph(N=20, topology="chain")
        synapses = [{"pre": 5, "post": 10, "weight": 1.0, "E_syn": 0.0, "tau_syn": 3.0}]
        syn = SynapseLayer(graph, synapses, dt=0.01)
        phi = np.zeros(20)
        currents = syn.compute_currents(phi)
        assert all(abs(v) < 1e-10 for v in currents.values())

    def test_excitatory_current(self):
        graph = Graph(N=20, topology="chain")
        synapses = [{"pre": 5, "post": 10, "weight": 1.0, "E_syn": 0.0, "tau_syn": 3.0}]
        syn = SynapseLayer(graph, synapses, dt=0.01)
        syn.s[0] = 1.0
        phi = np.full(20, -65.0)
        currents = syn.compute_currents(phi)
        assert abs(currents[10] - (-65.0)) < 1e-10

    def test_inhibitory_current(self):
        graph = Graph(N=20, topology="chain")
        synapses = [{"pre": 5, "post": 10, "weight": 1.0, "E_syn": -80.0, "tau_syn": 3.0}]
        syn = SynapseLayer(graph, synapses, dt=0.01)
        syn.s[0] = 1.0
        phi = np.full(20, -65.0)
        currents = syn.compute_currents(phi)
        assert abs(currents[10] - 15.0) < 1e-10

    def test_summation(self):
        graph = Graph(N=20, topology="chain")
        synapses = [
            {"pre": 3, "post": 10, "weight": 1.0, "E_syn": 0.0, "tau_syn": 3.0},
            {"pre": 5, "post": 10, "weight": 2.0, "E_syn": 0.0, "tau_syn": 3.0},
        ]
        syn = SynapseLayer(graph, synapses, dt=0.01)
        syn.s[0] = 1.0
        syn.s[1] = 1.0
        phi = np.full(20, -65.0)
        currents = syn.compute_currents(phi)
        assert abs(currents[10] - (-195.0)) < 1e-10

    def test_get_current_array(self):
        graph = Graph(N=20, topology="chain")
        synapses = [{"pre": 5, "post": 10, "weight": 1.0, "E_syn": 0.0, "tau_syn": 3.0}]
        syn = SynapseLayer(graph, synapses, dt=0.01)
        syn.s[0] = 1.0
        phi = np.full(20, -65.0)
        neuron_indices = np.array([5, 10, 15])
        I_ext = syn.get_current_array(phi, neuron_indices)
        assert I_ext.shape == (3,)
        assert abs(I_ext[0]) < 1e-10
        assert abs(I_ext[1] - (-65.0)) < 1e-10
        assert abs(I_ext[2]) < 1e-10


# --- Интеграция с NeuronLayer ---

class TestIntegration:

    def test_neuron_accepts_iext(self):
        graph = Graph(N=20, topology="chain")
        neuron = NeuronLayer(graph, neuron_nodes=[5, 10, 15], dt=0.01)
        phi = np.full(20, -65.0)
        I_ext = np.array([0.0, 10.0, 0.0])
        phi_with = neuron.apply_to_graph(phi.copy(), I_ext=I_ext)
        neuron2 = NeuronLayer(graph, neuron_nodes=[5, 10, 15], dt=0.01)
        phi_without = neuron2.apply_to_graph(phi.copy())
        assert not np.isclose(phi_with[10], phi_without[10])

    def test_excitatory_depolarizes(self):
        graph = Graph(N=20, topology="chain")
        synapses = [{"pre": 5, "post": 10, "weight": 5.0, "E_syn": 0.0, "tau_syn": 3.0}]
        syn = SynapseLayer(graph, synapses, dt=0.01, V_threshold=-20.0)
        syn.s[0] = 1.0
        phi = np.full(20, -65.0)
        neuron = NeuronLayer(graph, neuron_nodes=[5, 10, 15], dt=0.01)
        I_ext = syn.get_current_array(phi, neuron.neuron_indices)
        phi_with = neuron.apply_to_graph(phi.copy(), I_ext=I_ext)
        neuron2 = NeuronLayer(graph, neuron_nodes=[5, 10, 15], dt=0.01)
        phi_without = neuron2.apply_to_graph(phi.copy())
        assert phi_with[10] > phi_without[10]

    def test_inhibitory_hyperpolarizes(self):
        graph = Graph(N=20, topology="chain")
        synapses = [{"pre": 5, "post": 10, "weight": 5.0, "E_syn": -80.0, "tau_syn": 3.0}]
        syn = SynapseLayer(graph, synapses, dt=0.01, V_threshold=-20.0)
        syn.s[0] = 1.0
        phi = np.full(20, -65.0)
        neuron = NeuronLayer(graph, neuron_nodes=[5, 10, 15], dt=0.01)
        I_ext = syn.get_current_array(phi, neuron.neuron_indices)
        phi_with = neuron.apply_to_graph(phi.copy(), I_ext=I_ext)
        neuron2 = NeuronLayer(graph, neuron_nodes=[5, 10, 15], dt=0.01)
        phi_without = neuron2.apply_to_graph(phi.copy())
        assert phi_with[10] < phi_without[10]

    def test_spike_propagation(self):
        graph = Graph(N=20, topology="chain")
        neuron = NeuronLayer(graph, neuron_nodes=[5, 10, 15], dt=0.01)
        synapses = [{"pre": 5, "post": 10, "weight": 10.0, "E_syn": 0.0, "tau_syn": 5.0}]
        syn = SynapseLayer(graph, synapses, dt=0.01, V_threshold=-20.0)
        phi = np.full(20, -65.0)
        syn.step(phi)
        phi[5] = 0.0
        syn.step(phi)
        assert syn.s[0] > 0.0
        I_ext = syn.get_current_array(phi, neuron.neuron_indices)
        V_before = phi[10]
        phi = neuron.apply_to_graph(phi, I_ext=I_ext)
        assert phi[10] > V_before

    def test_full_loop_stable(self):
        graph = Graph(N=20, topology="chain")
        neuron = NeuronLayer(graph, neuron_nodes=[5, 10, 15], dt=0.01)
        synapses = [
            {"pre": 5, "post": 10, "weight": 0.5, "E_syn": 0.0, "tau_syn": 3.0},
            {"pre": 10, "post": 15, "weight": 0.3, "E_syn": 0.0, "tau_syn": 3.0},
        ]
        syn = SynapseLayer(graph, synapses, dt=0.01, V_threshold=-20.0)
        phi = np.full(20, -65.0)
        for _ in range(100):
            syn.step(phi)
            I_ext = syn.get_current_array(phi, neuron.neuron_indices)
            phi = neuron.apply_to_graph(phi, I_ext=I_ext)
        assert not np.any(np.isnan(phi))
        assert np.all(np.abs(phi) < 1e6)

    def test_diffusion_neuron_synapse_stable(self):
        graph = Graph(N=20, topology="chain")
        ions = [
            {"name": "Na", "D": 1.33e-9, "z": 1, "c0": 145.0},
            {"name": "K", "D": 1.96e-9, "z": 1, "c0": 4.0},
            {"name": "Cl", "D": 2.03e-9, "z": -1, "c0": 110.0},
            {"name": "Ca", "D": 0.79e-9, "z": 2, "c0": 1.0},
        ]
        solver = BiologicalDiffusion(graph, ions, dt=0.01)
        neuron = NeuronLayer(graph, neuron_nodes=[5, 10, 15], dt=0.01)
        synapses = [
            {"pre": 5, "post": 10, "weight": 0.5, "E_syn": 0.0, "tau_syn": 3.0},
            {"pre": 10, "post": 15, "weight": 0.3, "E_syn": 0.0, "tau_syn": 3.0},
        ]
        syn = SynapseLayer(graph, synapses, dt=0.01, V_threshold=-20.0)
        for _ in range(50):
            solver.step()
            syn.step(solver.phi)
            I_ext = syn.get_current_array(solver.phi, neuron.neuron_indices)
            solver.phi = neuron.apply_to_graph(solver.phi, solver.c, I_ext=I_ext)
        assert not np.any(np.isnan(solver.c))
        assert not np.any(np.isnan(solver.phi))


# --- Конфиг ---

class TestConfig:

    def test_config_passthrough(self):
        from main import build_synapses, build_graph
        cfg = {
            'graph': {'N': 20, 'topology': 'chain'},
            'synapse': {
                'enabled': True,
                'V_threshold': -30.0,
                'synapses': [
                    {'pre': 5, 'post': 10, 'weight': 0.5,
                     'E_syn': 0.0, 'tau_syn': 3.0},
                ],
            },
        }
        g = build_graph(cfg)
        syn = build_synapses(cfg, g)
        assert syn is not None
        assert syn.n_syn == 1
        assert syn.V_threshold == -30.0

    def test_config_disabled(self):
        from main import build_synapses, build_graph
        cfg = {
            'graph': {'N': 20, 'topology': 'chain'},
            'synapse': {'enabled': False},
        }
        g = build_graph(cfg)
        syn = build_synapses(cfg, g)
        assert syn is None

    def test_config_missing_section(self):
        from main import build_synapses, build_graph
        cfg = {
            'graph': {'N': 20, 'topology': 'chain'},
        }
        g = build_graph(cfg)
        syn = build_synapses(cfg, g)
        assert syn is None

    def test_config_stdp_passthrough(self):
        from main import build_synapses, build_graph
        cfg = {
            'graph': {'N': 20, 'topology': 'chain'},
            'synapse': {
                'enabled': True,
                'V_threshold': -30.0,
                'use_stdp': True,
                'A_plus': 0.02,
                'A_minus': 0.025,
                'tau_plus': 15.0,
                'tau_minus': 25.0,
                'w_min': 0.1,
                'w_max': 8.0,
                'synapses': [
                    {'pre': 5, 'post': 10, 'weight': 0.5,
                     'E_syn': 0.0, 'tau_syn': 3.0},
                ],
            },
        }
        g = build_graph(cfg)
        syn = build_synapses(cfg, g)
        assert syn.use_stdp is True
        assert syn.A_plus == 0.02
        assert syn.A_minus == 0.025
        assert syn.tau_plus == 15.0
        assert syn.tau_minus == 25.0
        assert syn.w_min == 0.1
        assert syn.w_max == 8.0


# --- STDP (v0.9) ---

class TestSTDP:
    """Тесты spike-timing-dependent plasticity."""

    @pytest.fixture
    def graph(self):
        return Graph(N=20, topology="chain")

    @pytest.fixture
    def synapses(self):
        return [{"pre": 5, "post": 10, "weight": 1.0,
                 "E_syn": 0.0, "tau_syn": 3.0}]

    def test_stdp_disabled_by_default(self, graph, synapses):
        syn = SynapseLayer(graph, synapses, dt=0.01)
        assert syn.use_stdp is False

    def test_stdp_flag_set(self, graph, synapses):
        syn = SynapseLayer(graph, synapses, dt=0.01, use_stdp=True)
        assert syn.use_stdp is True

    def test_stdp_params_set(self, graph, synapses):
        syn = SynapseLayer(graph, synapses, dt=0.01, use_stdp=True,
                           A_plus=0.05, A_minus=0.06,
                           tau_plus=10.0, tau_minus=15.0,
                           w_min=0.0, w_max=5.0)
        assert syn.A_plus == 0.05
        assert syn.A_minus == 0.06
        assert syn.tau_plus == 10.0
        assert syn.tau_minus == 15.0
        assert syn.w_min == 0.0
        assert syn.w_max == 5.0

    def test_no_weight_change_without_stdp(self, graph, synapses):
        syn = SynapseLayer(graph, synapses, dt=0.01, use_stdp=False)
        w_before = syn.weights[0]
        phi = np.full(20, -50.0)
        syn.step(phi)
        phi[5] = 0.0
        syn.step(phi)
        phi[5] = -50.0
        phi[10] = 0.0
        syn.step(phi)
        assert syn.weights[0] == w_before

    def test_ltp_pre_before_post(self, graph, synapses):
        """Pre спайк до post → LTP (weight растёт)."""
        syn = SynapseLayer(graph, synapses, dt=0.01, use_stdp=True,
                           A_plus=0.1, A_minus=0.1,
                           tau_plus=20.0, tau_minus=20.0)
        w_before = syn.weights[0]

        # Шаг 1: нет спайков
        phi = np.full(20, -50.0)
        syn.step(phi)

        # Шаг 2: pre спайк
        phi[5] = 0.0
        syn.step(phi)

        # Шаг 3: post спайк (через 1 шаг)
        phi[5] = -50.0
        phi[10] = 0.0
        syn.step(phi)

        assert syn.weights[0] > w_before

    def test_ltd_post_before_pre(self, graph, synapses):
        """Post спайк до pre → LTD (weight падает)."""
        syn = SynapseLayer(graph, synapses, dt=0.01, use_stdp=True,
                           A_plus=0.1, A_minus=0.1,
                           tau_plus=20.0, tau_minus=20.0)
        w_before = syn.weights[0]

        # Шаг 1: нет спайков
        phi = np.full(20, -50.0)
        syn.step(phi)

        # Шаг 2: post спайк
        phi[10] = 0.0
        syn.step(phi)

        # Шаг 3: pre спайк (через 1 шаг)
        phi[10] = -50.0
        phi[5] = 0.0
        syn.step(phi)

        assert syn.weights[0] < w_before

    def test_weight_clipped_to_max(self, graph, synapses):
        """Вес не может превысить w_max."""
        syn = SynapseLayer(graph, synapses, dt=0.01, use_stdp=True,
                           A_plus=100.0, A_minus=0.1,
                           tau_plus=20.0, tau_minus=20.0,
                           w_max=2.0)
        phi = np.full(20, -50.0)
        syn.step(phi)
        phi[5] = 0.0
        syn.step(phi)
        phi[5] = -50.0
        phi[10] = 0.0
        syn.step(phi)
        assert syn.weights[0] <= 2.0

    def test_weight_clipped_to_min(self, graph, synapses):
        """Вес не может упасть ниже w_min."""
        syn = SynapseLayer(graph, synapses, dt=0.01, use_stdp=True,
                           A_plus=0.1, A_minus=100.0,
                           tau_plus=20.0, tau_minus=20.0,
                           w_min=0.5)
        phi = np.full(20, -50.0)
        syn.step(phi)
        phi[10] = 0.0
        syn.step(phi)
        phi[10] = -50.0
        phi[5] = 0.0
        syn.step(phi)
        assert syn.weights[0] >= 0.5

    def test_no_change_when_no_correlation(self, graph, synapses):
        """Нет спайков → вес не меняется."""
        syn = SynapseLayer(graph, synapses, dt=0.01, use_stdp=True)
        w_before = syn.weights[0]
        phi = np.full(20, -50.0)
        for _ in range(100):
            syn.step(phi)
        assert abs(syn.weights[0] - w_before) < 1e-15

    def test_stdp_decay_with_timing(self, graph, synapses):
        """Чем больше задержка между спайками, тем меньше изменение веса."""
        # Быстрая пара
        syn_fast = SynapseLayer(graph, synapses, dt=0.01, use_stdp=True,
                                A_plus=0.1, tau_plus=20.0,
                                tau_minus=20.0, A_minus=0.1)
        w_fast = syn_fast.weights[0]

        phi = np.full(20, -50.0)
        syn_fast.step(phi)
        phi[5] = 0.0
        syn_fast.step(phi)
        phi[5] = -50.0
        phi[10] = 0.0
        syn_fast.step(phi)
        dw_fast = syn_fast.weights[0] - w_fast

        # Медленная пара (большая задержка)
        syn_slow = SynapseLayer(graph, synapses, dt=0.01, use_stdp=True,
                                A_plus=0.1, tau_plus=20.0,
                                tau_minus=20.0, A_minus=0.1)
        w_slow = syn_slow.weights[0]

        phi = np.full(20, -50.0)
        syn_slow.step(phi)
        phi[5] = 0.0
        syn_slow.step(phi)
        phi[5] = -50.0
        for _ in range(50):
            syn_slow.step(phi)
        phi[10] = 0.0
        syn_slow.step(phi)
        dw_slow = syn_slow.weights[0] - w_slow

        assert dw_fast > dw_slow
        assert dw_slow > 0  # всё ещё LTP, но слабее

    def test_stdp_stable_long_run(self, graph):
        """STDP в длинном цикле — без NaN и взрывов."""
        synapses = [
            {"pre": 5, "post": 10, "weight": 1.0,
             "E_syn": 0.0, "tau_syn": 3.0},
            {"pre": 10, "post": 15, "weight": 0.5,
             "E_syn": 0.0, "tau_syn": 3.0},
        ]
        syn = SynapseLayer(graph, synapses, dt=0.01, use_stdp=True)
        neuron = NeuronLayer(graph, neuron_nodes=[5, 10, 15], dt=0.01)
        phi = np.full(20, -65.0)
        for _ in range(200):
            syn.step(phi)
            I_ext = syn.get_current_array(phi, neuron.neuron_indices)
            phi = neuron.apply_to_graph(phi, I_ext=I_ext)
        assert not np.any(np.isnan(phi))
        assert np.all(syn.weights >= syn.w_min - 1e-10)
        assert np.all(syn.weights <= syn.w_max + 1e-10)

    def test_stdp_with_diffusion_stable(self, graph):
        """Полный цикл: диффузия + нейрон + синапс + STDP."""
        ions = [
            {"name": "Na", "D": 1.33e-9, "z": 1, "c0": 145.0},
            {"name": "K", "D": 1.96e-9, "z": 1, "c0": 4.0},
            {"name": "Cl", "D": 2.03e-9, "z": -1, "c0": 110.0},
            {"name": "Ca", "D": 0.79e-9, "z": 2, "c0": 1.0},
        ]
        solver = BiologicalDiffusion(graph, ions, dt=0.01)
        neuron = NeuronLayer(graph, neuron_nodes=[5, 10, 15], dt=0.01)
        synapses = [
            {"pre": 5, "post": 10, "weight": 1.0,
             "E_syn": 0.0, "tau_syn": 3.0},
            {"pre": 10, "post": 15, "weight": 0.5,
             "E_syn": 0.0, "tau_syn": 3.0},
        ]
        syn = SynapseLayer(graph, synapses, dt=0.01, use_stdp=True)
        for _ in range(50):
            solver.step()
            syn.step(solver.phi)
            I_ext = syn.get_current_array(solver.phi, neuron.neuron_indices)
            solver.phi = neuron.apply_to_graph(
                solver.phi, solver.c, I_ext=I_ext)
        assert not np.any(np.isnan(solver.c))
        assert not np.any(np.isnan(solver.phi))
        assert np.all(syn.weights >= -1e-10)
        assert np.all(syn.weights < 1e6)
# === END ===
