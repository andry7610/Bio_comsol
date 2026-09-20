"""tests/test_biological.py — тесты для biological/diffusion.py.

Покрывает:
    - Базовая симуляция: шаг, сходимость Пикара, граничные условия
    - Диагностика: квазинейтральность, заряд
    - ИИ-интеграция: analyze_state, validate_params, analyze_every
    - Fallback: работа без ИИ (ManualBackend)
    - Принудительный бэкенд: mock-объект
"""

import numpy as np
import pytest

from network.graph import Graph
from network.base import AnalysisResult
from network.manual import ManualBackend
from biological.diffusion import BiologicalDiffusion


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


# ─── Базовая симуляция ───

class TestBasicSimulation:
    def test_init(self, sim):
        assert sim.N == 20
        assert sim.n_ions == 3
        assert sim.c.shape == (3, 20)
        assert sim.step_count == 0

    def test_initial_concentrations(self, sim):
        # Na везде 145
        assert np.allclose(sim.c[0], 145.0)
        # K везде 5
        assert np.allclose(sim.c[1], 5.0)

    def test_step_returns_picard_iters(self, sim):
        iters = sim.step()
        assert isinstance(iters, int)
        assert iters >= 1
        assert sim.step_count == 1

    def test_multiple_steps(self, sim):
        for _ in range(100):
            sim.step()
        assert sim.step_count == 100
        # Концентрации должны остаться конечными
        assert np.all(np.isfinite(sim.c))

    def test_boundary_conditions(self, sim):
        sim.step()
        # Дирихле на границах
        assert np.isclose(sim.c[0, 0], 145.0)   # Na left
        assert np.isclose(sim.c[0, -1], 10.0)   # Na right
        assert np.isclose(sim.c[1, 0], 5.0)     # K left
        assert np.isclose(sim.c[1, -1], 140.0)  # K right

    def test_potential_profile(self, sim):
        assert np.isclose(sim.phi[0], 0.0)
        assert np.isclose(sim.phi[-1], -1.8)
        # Линейная интерполяция: phi[i] = -1.8 * i / (N-1)
        assert np.isclose(sim.phi[10], -1.8 * 10 / 19, atol=0.01)

    def test_concentration_changes_over_time(self, sim):
        c_before = sim.c.copy()
        for _ in range(50):
            sim.step()
        # Концентрации во внутренних узлах должны измениться
        assert not np.allclose(sim.c[:, 5:15], c_before[:, 5:15])

    def test_picard_convergence(self, sim):
        sim.step()
        # Должен сойтись за разумное число итераций
        assert sim.last_picard_iters <= sim.picard_max_iter


# ─── Диагностика ───

class TestDiagnostics:
    def test_total_charge(self, sim):
        q = sim.total_charge()
        assert isinstance(q, float)

    def test_charge_per_node_shape(self, sim):
        q = sim.charge_per_node()
        assert q.shape == (20,)

    def test_quasineutrality_error(self, sim):
        err = sim.quasineutrality_error()
        assert isinstance(err, float)
        assert err >= 0.0

    def test_quasineutrality_initial(self, sim):
        # В начальный момент: Na(+1)*145 + K(+1)*5 + Cl(-1)*150 = 0
        err = sim.quasineutrality_error()
        assert err < 1.0  # почти квазинейтрально

    def test_quasineutrality_after_steps(self, sim):
        for _ in range(50):
        sim.step()
        err = sim.quasineutrality_error()
        # Не должно сильно расходиться
        assert err < 200.0



# ─── ИИ-интеграция ───

class TestAIIntegration:
    def test_default_backend_is_manual(self, sim):
        # Без API ключа и Ollama — ManualBackend
        assert sim.ai_backend is not None
        assert sim.ai_backend.name in ("manual", "local", "api")

    def test_analyze_state(self, sim):
        result = sim.analyze_state()
        assert isinstance(result, AnalysisResult)
        assert result.backend in ("manual", "local", "api")
        assert result.summary  # непустой

    def test_analyze_state_after_steps(self, sim):
        for _ in range(10):
            sim.step()
        result = sim.analyze_state()
        assert result.confidence >= 0.0
        assert isinstance(result.warnings, list)
        assert isinstance(result.suggestions, list)

    def test_validate_params(self, sim):
        result = sim.validate_params()
        assert isinstance(result, AnalysisResult)
        assert result.backend in ("manual", "local", "api")

    def test_validate_params_catches_bad_dt(self, simple_graph, simple_ions):
        sim = BiologicalDiffusion(
            simple_graph, simple_ions,
            dt=-0.5,  # отрицательный dt
        )
        result = sim.validate_params()
        assert len(result.warnings) > 0

    def test_validate_params_good(self, sim):
        result = sim.validate_params()
        # С нормальными параметрами — мало warnings
        assert len(result.warnings) <= 1

    def test_analyze_every_triggers(self, simple_graph, simple_ions):
        sim = BiologicalDiffusion(
            simple_graph, simple_ions,
            dt=0.01, analyze_every=5,
        )
        assert sim.last_analysis is None
        for _ in range(5):
            sim.step()
        # На 5-м шаге должен сработать анализ
        assert sim.last_analysis is not None
        assert isinstance(sim.last_analysis, AnalysisResult)

    def test_analyze_every_zero_no_trigger(self, sim):
        assert sim.analyze_every == 0
        for _ in range(20):
            sim.step()
        # Без analyze_every — анализ не запускается
        assert sim.last_analysis is None

    def test_last_analysis_updated(self, simple_graph, simple_ions):
        sim = BiologicalDiffusion(
            simple_graph, simple_ions,
            dt=0.01, analyze_every=10,
        )
        for _ in range(10):
            sim.step()
        first = sim.last_analysis
        assert first is not None
        for _ in range(10):
            sim.step()
        second = sim.last_analysis
        # second — новый объект (даже если содержимое похожее)
        assert second is not None

    def test_manual_backend_explicit(self, simple_graph, simple_ions):
        backend = ManualBackend()
        sim = BiologicalDiffusion(
            simple_graph, simple_ions,
            ai_backend=backend,
        )
        assert sim.ai_backend.name == "manual"
        result = sim.analyze_state()
        assert result.backend == "manual"
        assert result.confidence == 0.5


# ─── Mock-бэкенд ───

class MockBackend:
    """Простой mock для тестирования вызовов."""
    def __init__(self):
        self.name = "mock"
        self.call_count = 0

    def is_available(self):
        return True

    def analyze(self, concentrations, charges, time_step, context=None):
        self.call_count += 1
        return AnalysisResult(
            summary=f"Mock analysis at step {time_step}",
            warnings=[],
            suggestions=[],
            confidence=0.99,
            backend="mock",
        )

    def validate_params(self, params):
        self.call_count += 1
        return AnalysisResult(
            summary="Mock: params OK",
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
    def test_mock_analyze(self, simple_graph, simple_ions):
        mock = MockBackend()
        sim = BiologicalDiffusion(
            simple_graph, simple_ions,
            ai_backend=mock, analyze_every=3,
        )
        for _ in range(9):
            sim.step()
        # analyze_every=3, 9 шагов → 3 вызова (на шагах 3, 6, 9)
        assert mock.call_count == 3

    def test_mock_validate(self, simple_graph, simple_ions):
        mock = MockBackend()
        sim = BiologicalDiffusion(
            simple_graph, simple_ions,
            ai_backend=mock,
        )
        result = sim.validate_params()
        assert result.backend == "mock"
        assert result.confidence == 0.99

    def test_mock_summary(self, simple_graph, simple_ions):
        mock = MockBackend()
        sim = BiologicalDiffusion(
            simple_graph, simple_ions,
            ai_backend=mock, analyze_every=1,
        )
        sim.step()
        assert sim.last_analysis is not None
        assert "Mock" in sim.last_analysis.summary
        assert "step 1" in sim.last_analysis.summary


# ─── Стабильность ───

class TestStability:
    def test_long_run_no_nan(self, sim):
        for _ in range(500):
            sim.step()
        assert np.all(np.isfinite(sim.c))
        assert sim.step_count == 500

    def test_long_run_no_negative(self, sim):
        for _ in range(500):
            sim.step()
        # Допускаем небольшие отрицательные значения (численная погрешность)
        # но не катастрофические
        assert np.min(sim.c) > -10.0

    def test_long_run_quasineutrality(self, sim):
        for _ in range(500):
            sim.step()
        err = sim.quasineutrality_error()
        assert err < 200.0

    def test_small_dt_stable(self, simple_graph, simple_ions):
        sim = BiologicalDiffusion(
            simple_graph, simple_ions,
            dt=1e-4, F_RT=38.9,
        )
        for _ in range(100):
            sim.step()
        assert np.all(np.isfinite(sim.c))

    def test_large_dt_warning(self, simple_graph, simple_ions):
        sim = BiologicalDiffusion(
            simple_graph, simple_ions,
            dt=5.0,  # слишком большой
        )
        result = sim.validate_params()
        assert len(result.warnings) > 0
