import numpy as np
import pytest

from network.graph import Graph
from biological.synapse import SynapseLayer
from biological.neuron import NeuronLayer
from biological.diffusion import BiologicalDiffusion


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
# === END ===
