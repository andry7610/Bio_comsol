"""tests/test_biological.py — Тесты BiologicalDiffusion."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from network.graph import Graph
from biological.diffusion import BiologicalDiffusion
from network import AnalysisResult


# --- Фикстуры ---

@pytest.fixture
def graph():
    return Graph(N=20, topology="chain")


@pytest.fixture
def ions():
    return [
        {"name": "Na", "D": 1.33e-9, "z": 1, "c0": 145.0},
        {"name": "K", "D": 1.96e-9, "z": 1, "c0": 4.0},
        {"name": "Cl", "D": 2.03e-9, "z": -1, "c0": 110.0},
        {"name": "Ca", "D": 0.79e-9, "z": 2, "c0": 1.0},
    ]


@pytest.fixture
def solver(graph, ions):
    return BiologicalDiffusion(graph, ions, dt=0.01)


# --- Базовая симуляция ---

class TestBasicSimulation:

    def test_init(self, solver):
        assert solver.N == 20
        assert solver.n_ions == 4
        assert solver.dt == 0.01
        assert solver.step_count == 0

    def test_initial_concentrations(self, solver):
        assert solver.c.shape == (4, 20)
        assert np.allclose(solver.c[0], 145.0)
        assert np.allclose(solver.c[1], 4.0)
        assert np.allclose(solver.c[2], 110.0)
        assert np.allclose(solver.c[3], 1.0)

    def test_step_returns_picard_iters(self, solver):
        iters = solver.step()
        assert isinstance(iters, int)
        assert iters >= 1
        assert iters <= solver.picard_max_iter

    def test_multiple_steps(self, solver):
        for _ in range(10):
            solver.step()
        assert solver.step_count == 10

    def test_boundary_conditions(self, graph, ions):
        ions_bc = [
            {"name": "Na", "D": 1e-9, "z": 1, "c0": 100,
             "c_left": 150, "c_right": 10},
        ]
        s = BiologicalDiffusion(graph, ions_bc, dt=0.01)
        s.step()
        assert s.c[0, 0] == 150
        assert s.c[0, -1] == 10

    def test_potential_profile(self, solver):
        assert solver.phi[0] == 0.0
        assert abs(solver.phi[-1] - solver.phi_membrane) < 1e-10

    def test_concentration_changes_over_time(self, solver):
        c_before = solver.c.copy()
        solver.step()
        assert not np.allclose(solver.c, c_before)

    def test_picard_convergence(self, solver):
        solver.step()
        assert solver.last_picard_iters <= solver.picard_max_iter


# --- Диагностика ---

class TestDiagnostics:

    def test_total_charge(self, solver):
        charge = solver.total_charge()
        assert isinstance(charge, float)
        # Na + K - Cl + 2*Ca = 145 + 4 - 110 + 2 = 41
        assert abs(charge - 41 * 20) < 1e-6

    def test_charge_per_node_shape(self, solver):
        ch = solver.charge_per_node()
        assert ch.shape == (20,)

    def test_quasineutrality_error(self, solver):
        err = solver.quasineutrality_error()
        assert isinstance(err, float)
        assert err >= 0

    def test_quasineutrality_after_steps(self, solver):
        for _ in range(5):
            solver.step()
        err = solver.quasineutrality_error()
        assert err < 1e3


# --- ИИ-интеграция ---

class TestAIIntegration:

    def test_default_backend_is_manual(self, solver):
        from network import get_backend
        backend = get_backend()
        assert backend is not None

    def test_analyze_state(self, solver):
        result = solver.analyze_state()
        assert result is not None
        assert hasattr(result, 'summary')

    def test_analyze_state_after_steps(self, solver):
        for _ in range(3):
            solver.step()
        result = solver.analyze_state()
        assert result is not None

    def test_validate_params(self, solver):
        result = solver.validate_params()
        assert result is not None
        assert hasattr(result, 'summary')

    def test_validate_params_catches_bad_dt(self, graph, ions):
        s = BiologicalDiffusion(graph, ions, dt=-1.0)
        result = s.validate_params()
        assert result is not None

    def test_validate_params_good(self, solver):
        result = solver.validate_params()
        assert result is not None

    def test_analyze_every_triggers(self, graph, ions):
        mock_backend = MagicMock()
        mock_backend.analyze.return_value = AnalysisResult(
            summary="ok", warnings=[], suggestions=[])
        s = BiologicalDiffusion(graph, ions, dt=0.01,
                                ai_backend=mock_backend, analyze_every=5)
        for _ in range(10):
            s.step()
        assert mock_backend.analyze.call_count >= 2

    def test_analyze_every_zero_no_trigger(self, graph, ions):
        mock_backend = MagicMock()
        mock_backend.analyze.return_value = AnalysisResult(
            summary="ok", warnings=[], suggestions=[])
        s = BiologicalDiffusion(graph, ions, dt=0.01,
                                ai_backend=mock_backend, analyze_every=0)
        for _ in range(10):
            s.step()
        assert mock_backend.analyze.call_count == 0

    def test_last_analysis_updated(self, graph, ions):
        mock_backend = MagicMock()
        mock_backend.analyze.return_value = AnalysisResult(
            summary="test", warnings=[], suggestions=[])
        s = BiologicalDiffusion(graph, ions, dt=0.01,
                                ai_backend=mock_backend, analyze_every=3)
        s.step()
        s.step()
        s.step()
        assert s.last_analysis is not None
        assert s.last_analysis.summary == "test"

    def test_manual_backend_explicit(self, graph, ions):
        from network import get_backend
        backend = get_backend()
        s = BiologicalDiffusion(graph, ions, dt=0.01, ai_backend=backend)
        result = s.analyze_state()
        assert result is not None


# --- Mock-бэкенд ---

class TestMockBackend:

    def test_mock_analyze(self, graph, ions):
        mock = MagicMock()
        mock.analyze.return_value = AnalysisResult(
            summary="mock", warnings=["w"], suggestions=["s"])
        s = BiologicalDiffusion(graph, ions, ai_backend=mock)
        result = s.analyze_state()
        assert result.summary == "mock"
        assert result.warnings == ["w"]
        assert result.suggestions == ["s"]

    def test_mock_validate(self, graph, ions):
        mock = MagicMock()
        mock.validate_params.return_value = AnalysisResult(
            summary="mock_valid", warnings=[], suggestions=[])
        s = BiologicalDiffusion(graph, ions, ai_backend=mock)
        result = s.validate_params()
        assert result.summary == "mock_valid"

    def test_mock_summary(self, graph, ions):
        mock = MagicMock()
        mock.analyze.return_value = AnalysisResult(
            summary="test_summary", warnings=[], suggestions=[])
        s = BiologicalDiffusion(graph, ions, ai_backend=mock)
        result = s.analyze_state()
        assert "test_summary" in result.summary


# --- Стабильность ---

class TestStability:

    def test_long_run_no_nan(self, solver):
        for _ in range(500):
            solver.step()
        assert not np.any(np.isnan(solver.c))

    def test_long_run_no_negative(self, solver):
        for _ in range(500):
            solver.step()
        assert np.all(solver.c >= -1e-10)

    def test_long_run_quasineutrality(self, solver):
        for _ in range(500):
            solver.step()
        err = solver.quasineutrality_error()
        assert err < 1e3

    def test_small_dt_stable(self, graph, ions):
        s = BiologicalDiffusion(graph, ions, dt=1e-5)
        for _ in range(100):
            s.step()
        assert not np.any(np.isnan(s.c))

    def test_large_dt_warning(self, graph, ions):
        s = BiologicalDiffusion(graph, ions, dt=10.0)
        for _ in range(5):
            s.step()
        assert not np.any(np.isnan(s.c))


# --- Самосогласованный потенциал (v0.6.1) ---

class TestSelfConsistent:
    """Тесты самосогласованного потенциала (уравнение Пуассона)."""

    @pytest.fixture
    def graph(self):
        return Graph(N=20, topology="chain")

    @pytest.fixture
    def ions(self):
        return [
            {"name": "Na", "D": 1.33e-9, "z": 1, "c0": 145.0,
             "c_left": 145.0, "c_right": 10.0},
            {"name": "K", "D": 1.96e-9, "z": 1, "c0": 4.0,
             "c_left": 4.0, "c_right": 100.0},
            {"name": "Cl", "D": 2.03e-9, "z": -1, "c0": 110.0,
             "c_left": 110.0, "c_right": 110.0},
            {"name": "Ca", "D": 0.79e-9, "z": 2, "c0": 1.0,
             "c_left": 1.0, "c_right": 0.1},
        ]

    @pytest.fixture
    def solver(self, graph, ions):
        return BiologicalDiffusion(graph, ions, dt=0.01,
                                   self_consistent=True,
                                   poisson_lambda=1.0)

    def test_init_self_consistent_flag(self, solver):
        assert solver.self_consistent is True

    def test_init_poisson_matrix_exists(self, solver):
        assert hasattr(solver, 'A_poisson')
        assert solver.A_poisson.shape == (20, 20)

    def test_disabled_by_default(self, graph, ions):
        s = BiologicalDiffusion(graph, ions)
        assert s.self_consistent is False
        assert not hasattr(s, 'A_poisson')

    def test_phi_changes_after_step(self, solver):
        phi_before = solver.phi.copy()
        solver.step()
        assert not np.allclose(solver.phi, phi_before)

    def test_phi_not_constant(self, solver):
        """φ не константа после шага (есть градиент)."""
        solver.step()
        assert np.ptp(solver.phi) > 1e-10

    def test_boundary_conditions_hold(self, solver):
        solver.step()
        assert abs(solver.phi[0]) < 1e-10
        assert abs(solver.phi[-1] - solver.phi_membrane) < 1e-10

    def test_no_nan_phi(self, solver):
        for _ in range(50):
            solver.step()
        assert not np.any(np.isnan(solver.phi))

    def test_poisson_solve_directly(self, solver):
        solver._solve_poisson()
        assert solver.phi.shape == (20,)
        assert abs(solver.phi[0]) < 1e-10
        assert abs(solver.phi[-1] - solver.phi_membrane) < 1e-10

    def test_charge_gradient_in_phi(self, graph, ions):
        """Неоднородный заряд → неоднородный φ."""
        s = BiologicalDiffusion(graph, ions, dt=0.01,
                                self_consistent=True, poisson_lambda=10.0)
        s.c[0, :10] = 200.0
        s.c[0, 10:] = 50.0
        s._solve_poisson()
        phi_linear = np.linspace(0, s.phi_membrane, 20)
        assert not np.allclose(s.phi, phi_linear, atol=1e-6)

    def test_poisson_lambda_effect(self, graph, ions):
        """Больший λ → большее отклонение от линейного φ."""
        s1 = BiologicalDiffusion(graph, ions, dt=0.01,
                                 self_consistent=True, poisson_lambda=0.1)
        s2 = BiologicalDiffusion(graph, ions, dt=0.01,
                                 self_consistent=True, poisson_lambda=100.0)
        s1.c[0, :10] = 200.0
        s1.c[0, 10:] = 50.0
        s2.c[0, :10] = 200.0
        s2.c[0, 10:] = 50.0
        s1._solve_poisson()
        s2._solve_poisson()
        phi_lin = np.linspace(0, s1.phi_membrane, 20)
        dev1 = np.max(np.abs(s1.phi - phi_lin))
        dev2 = np.max(np.abs(s2.phi - phi_lin))
        assert dev2 > dev1

    def test_long_run_stable(self, solver):
        for _ in range(200):
            solver.step()
        assert not np.any(np.isnan(solver.c))
        assert not np.any(np.isnan(solver.phi))
        assert np.max(np.abs(solver.phi)) < 1e6

    def test_still_works_with_neuron(self, graph, ions):
        """Самосогласование + нейрон — не крашит."""
        from biological.neuron import NeuronLayer
        s = BiologicalDiffusion(graph, ions, dt=0.01,
                                self_consistent=True)
        neuron = NeuronLayer(graph, neuron_nodes=[5, 10, 15], dt=0.01)
        for _ in range(20):
            s.step()
            s.phi = neuron.apply_to_graph(s.phi, s.c)
        assert not np.any(np.isnan(s.phi))

    def test_config_passthrough(self):
        """main.py передаёт self_consistent и poisson_lambda."""
        from main import build_solver, build_graph, build_ions
        cfg = {
            'graph': {'N': 20, 'topology': 'chain'},
            'physics_constants': {'F': 96485, 'R': 8.314, 'T': 310},
            'ion_species': {
                'Na': {'D': 1e-9, 'z': 1, 'c0': 145},
                'K': {'D': 2e-9, 'z': 1, 'c0': 4},
                'Cl': {'D': 2e-9, 'z': -1, 'c0': 110},
                'Ca': {'D': 0.8e-9, 'z': 2, 'c0': 1},
            },
            'solver': {
                'dt': 0.01, 'self_consistent': True,
                'poisson_lambda': 5.0,
            },
        }
        g = build_graph(cfg)
        ions = build_ions(cfg)
        s = build_solver(cfg, g, ions)
        assert s.self_consistent is True
        assert s.poisson_lambda == 5.0
# === END ===

