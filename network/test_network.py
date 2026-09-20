"""tests/test_network.py — тесты для network/.

Покрывает:
    - AnalysisResult: формат, строковое представление
    - ManualBackend: анализ, валидация, описание
    - LocalBackend: недоступность без Ollama
    - APIBackend: недоступность без ключа
    - config: автовыбор бэкенда
    - Graph: топологии, веса, соседи, лапласиан
    - prompts: формат и содержание
"""

import os
import numpy as np
import pytest

from network.base import BaseBackend, AnalysisResult
from network.manual import ManualBackend
from network.local import LocalBackend
from network.api import APIBackend
from network.config import get_backend, get_backend_info
from network.graph import Graph
from network.prompts import (
    build_analysis_prompt,
    build_validation_prompt,
    build_describe_prompt,
)


# ─── AnalysisResult ───

class TestAnalysisResult:
    def test_creation(self):
        r = AnalysisResult(
            summary="Test",
            warnings=["w1"],
            suggestions=["s1"],
            confidence=0.8,
            backend="manual",
        )
        assert r.summary == "Test"
        assert r.warnings == ["w1"]
        assert r.suggestions == ["s1"]
        assert r.confidence == 0.8
        assert r.backend == "manual"

    def test_str_format(self):
        r = AnalysisResult("OK", [], [], 0.5, "manual")
        s = str(r)
        assert "[manual]" in s
        assert "OK" in s
        assert "50%" in s

    def test_str_with_warnings(self):
        r = AnalysisResult("OK", ["unstable"], [], 0.5, "manual")
        s = str(r)
        assert "⚠" in s
        assert "unstable" in s

    def test_str_with_suggestions(self):
        r = AnalysisResult("OK", [], ["reduce dt"], 0.5, "manual")
        s = str(r)
        assert "→" in s
        assert "reduce dt" in s


# ─── ManualBackend ───

class TestManualBackend:
    def setup_method(self):
        self.backend = ManualBackend()

    def test_name(self):
        assert self.backend.name == "manual"

    def test_always_available(self):
        assert self.backend.is_available() is True

    def test_analyze_stable(self):
        c = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        z = np.array([+1, -1])
        r = self.backend.analyze(c, z, time_step=10)
        assert r.backend == "manual"
        assert r.confidence == 0.5
        assert "stable" in r.summary.lower()

    def test_analyze_negative_concentrations(self):
        c = np.array([[1.0, -0.5, 1.0], [1.0, 1.0, 1.0]])
        z = np.array([+1, -1])
        r = self.backend.analyze(c, z, time_step=5)
        assert any("negative" in w.lower() for w in r.warnings)

    def test_analyze_nan(self):
        c = np.array([[1.0, np.nan, 1.0], [1.0, 1.0, 1.0]])
        z = np.array([+1, -1])
        r = self.backend.analyze(c, z, time_step=5)
        assert any("nan" in w.lower() or "inf" in w.lower() for w in r.warnings)

    def test_analyze_high_concentration(self):
        c = np.array([[1e5, 1e5, 1e5], [1.0, 1.0, 1.0]])
        z = np.array([+1, -1])
        r = self.backend.analyze(c, z, time_step=3)
        assert any("high concentration" in w.lower() for w in r.warnings)

    def test_analyze_quasineutrality(self):
        c = np.array([[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]])
        z = np.array([+1, -1])
        r = self.backend.analyze(c, z, time_step=1)
        assert any("quasineutrality" in w.lower() for w in r.warnings)

    def test_analyze_with_context(self):
        c = np.array([[1.0, 1.0], [1.0, 1.0]])
        z = np.array([+1, -1])
        ctx = {"ion_names": ["Na", "Cl"], "membrane_potential": -0.07}
        r = self.backend.analyze(c, z, time_step=1, context=ctx)
        assert r.backend == "manual"
        assert r.confidence == 0.5

    def test_validate_params_ok(self):
        params = {
            "dt": 0.01,
            "F_RT": 38.9,
            "membrane_potential": -0.07,
            "ions": [
                {"name": "Na", "D": 1.33e-9, "z": +1, "c0": 145, "c_left": 145, "c_right": 10},
                {"name": "K", "D": 1.96e-9, "z": +1, "c0": 5, "c_left": 5, "c_right": 140},
            ],
        }
        r = self.backend.validate_params(params)
        assert r.backend == "manual"
        assert r.confidence == 0.6
        assert len(r.warnings) == 0

    def test_validate_params_bad_dt(self):
        params = {"dt": -0.1, "F_RT": 38.9, "ions": []}
        r = self.backend.validate_params(params)
        assert any("dt" in w.lower() for w in r.warnings)

    def test_validate_params_large_dt(self):
        params = {"dt": 2.0, "F_RT": 38.9, "ions": []}
        r = self.backend.validate_params(params)
        assert any("dt" in w.lower() for w in r.warnings)

    def test_validate_params_bad_F_RT(self):
        params = {"dt": 0.01, "F_RT": -1.0, "ions": []}
        r = self.backend.validate_params(params)
        assert any("F_RT" in w.lower() for w in r.warnings)

    def test_validate_params_negative_c0(self):
        params = {
            "dt": 0.01, "F_RT": 38.9,
            "ions": [{"name": "Na", "z": +1, "c0": -5}],
        }
        r = self.backend.validate_params(params)
        assert any("c0" in w.lower() and "negative" in w.lower() for w in r.warnings)

    def test_validate_params_zero_charge(self):
        params = {
            "dt": 0.01, "F_RT": 38.9,
            "ions": [{"name": "X", "z": 0, "c0": 1}],
        }
        r = self.backend.validate_params(params)
        assert any("z=0" in w.lower() or "charge" in w.lower() for w in r.warnings)

    def test_validate_params_extreme_membrane(self):
        params = {
            "dt": 0.01, "F_RT": 38.9,
            "membrane_potential": -50,
            "ions": [],
        }
        r = self.backend.validate_params(params)
        assert any("membrane" in w.lower() for w in r.warnings)

    def test_describe(self):
        r = self.backend.describe("Concentration dropped to 0.1 mM")
        assert r.backend == "manual"
        assert r.confidence == 0.3
        assert "manual" in r.summary.lower() or "Concentration" in r.summary


# ─── LocalBackend (без Ollama) ───

class TestLocalBackend:
    def setup_method(self):
        # Гарантируем, что Ollama не запущена в тестах
        self.backend = LocalBackend()
        self.backend._available = False

    def test_name(self):
        assert self.backend.name == "local"

    def test_unavailable_without_ollama(self):
        assert self.backend.is_available() is False

    def test_analyze_returns_fallback(self):
        c = np.array([[1.0, 1.0], [1.0, 1.0]])
        z = np.array([+1, -1])
        r = self.backend.analyze(c, z, time_step=1)
        assert r.backend == "local"
        assert r.confidence == 0.0
        assert any("ollama" in s.lower() for s in r.suggestions)

    def test_validate_params_returns_fallback(self):
        r = self.backend.validate_params({"dt": 0.01})
        assert r.confidence == 0.0
        assert r.backend == "local"

    def test_describe_returns_fallback(self):
        r = self.backend.describe("test")
        assert r.confidence == 0.0
        assert r.backend == "local"

    def test_parse_response_basic(self):
        raw = "SUMMARY: All good\nWARNINGS: None\nSUGGESTIONS: Continue"
        r = self.backend._parse_response(raw)
        assert r.backend == "local"
        assert "All good" in r.summary or "All good" in r.summary

    def test_parse_response_empty(self):
        r = self.backend._parse_response("")
        assert r.backend == "local"
        assert r.confidence == 0.75


# ─── APIBackend (без ключа) ───

class TestAPIBackend:
    def setup_method(self):
        # Чистим env, чтобы тесты не зависели от окружения
        os.environ.pop("BIOCOMSOL_API_KEY", None)
        self.backend = APIBackend()
        self.backend._available = False

    def test_name(self):
        assert self.backend.name == "api"

    def test_unavailable_without_key(self):
        assert self.backend.is_available() is False

    def test_analyze_returns_fallback(self):
        c = np.array([[1.0, 1.0], [1.0, 1.0]])
        z = np.array([+1, -1])
        r = self.backend.analyze(c, z, time_step=1)
        assert r.backend == "api"
        assert r.confidence == 0.0
        assert any("API_KEY" in s for s in r.suggestions)

    def test_validate_params_returns_fallback(self):
        r = self.backend.validate_params({"dt": 0.01})
        assert r.confidence == 0.0
        assert r.backend == "api"

    def test_describe_returns_fallback(self):
        r = self.backend.describe("test")
        assert r.confidence == 0.0
        assert r.backend == "api"

    def test_available_with_key(self):
        backend = APIBackend()
        backend._api_key = "fake_key_123"
        backend._available = None
        assert backend.is_available() is True

    def test_parse_response_basic(self):
        raw = "SUMMARY: Stable\nWARNINGS: None\nSUGGESTIONS: Proceed"
        r = self.backend._parse_response(raw)
        assert r.backend == "api"
        assert r.confidence == 0.75

    def test_parse_response_with_warnings(self):
        raw = (
            "SUMMARY: Unstable\n"
            "WARNINGS: dt too large\n"
            "SUGGESTIONS: Reduce dt to 0.001"
        )
        r = self.backend._parse_response(raw)
        assert any("dt" in w.lower() for w in r.warnings)
        assert any("dt" in s.lower() for s in r.suggestions)


# ─── config ───

class TestConfig:
    def setup_method(self):
        os.environ.pop("BIOCOMSOL_API_KEY", None)
        os.environ.pop("BIOCOMSOL_BACKEND", None)

    def test_auto_falls_back_to_manual(self):
        # Без ключа и без Ollama — manual
        backend = get_backend()
        assert backend.name in ("manual", "local")

    def test_forced_manual(self):
        backend = get_backend(forced="manual")
        assert backend.name == "manual"

    def test_forced_api(self):
        backend = get_backend(forced="api")
        assert backend.name == "api"
        assert backend.is_available() is False

    def test_forced_local(self):
        backend = get_backend(forced="local")
        assert backend.name == "local"

    def test_backend_info(self):
        info = get_backend_info()
        assert "api" in info
        assert "local" in info
        assert "manual" in info
        assert info["manual"]["available"] is True

    def test_env_backend_manual(self):
        os.environ["BIOCOMSOL_BACKEND"] = "manual"
        backend = get_backend()
        assert backend.name == "manual"
        os.environ.pop("BIOCOMSOL_BACKEND", None)


# ─── Graph ───

class TestGraph:
    def test_chain_topology(self):
        g = Graph(N=10, topology="chain")
        assert g.N == 10
        assert len(g.edges) == 9
        assert g.neighbors(0) == [1]
        assert g.neighbors(5) == [4, 6]
        assert g.neighbors(9) == [8]

    def test_ring_topology(self):
        g = Graph(N=5, topology="ring")
        assert len(g.edges) == 5
        assert 0 in g.neighbors(4)
        assert 4 in g.neighbors(0)

    def test_random_topology(self):
        g = Graph(N=20, topology="random", p_edge=0.3, seed=42)
        assert g.N == 20
        assert len(g.edges) > 0
        # Все узлы должны иметь хотя бы одного соседа
        for i in range(20):
            assert len(g.neighbors(i)) > 0

    def test_metric_weights(self):
        g = Graph(N=5, topology="chain", conductivity=2.0, cross_section=1e-8)
        # Вес ребра = sigma * S / |r_ij|
        for i, j, w in g.edges:
            assert w > 0
            dist = np.linalg.norm(g.pos[i] - g.pos[j])
            expected = 2.0 * 1e-8 / dist
            assert np.isclose(w, expected, rtol=1e-6)

    def test_custom_positions(self):
        pos = np.array([[0, 0], [1, 0], [2, 0], [0, 1]], dtype=float)
        g = Graph(N=4, topology="chain", pos=pos)
        assert g.pos.shape == (4, 2)
        assert np.allclose(g.pos, pos)

    def test_laplacian_shape(self):
        g = Graph(N=10, topology="chain")
        assert g.laplacian.shape == (10, 10)

    def test_laplacian_row_sum(self):
        g = Graph(N=10, topology="chain")
        row_sums = g.laplacian.sum(axis=1)
        # Лапласиан: сумма строки ~ 0 (с регуляризацией)
        assert np.allclose(row_sums, 1e-8, atol=1e-10)

    def test_laplacian_symmetric(self):
        g = Graph(N=8, topology="ring")
        assert np.allclose(g.laplacian, g.laplacian.T, atol=1e-10)

    def test_sparse_weights(self):
        g = Graph(N=10, topology="chain")
        assert g.W.shape == (10, 10)
        # В цепочке максимум 2 соседа
        for i in range(10):
            assert len(g.neighbors(i)) <= 2

    def test_isolated_nodes_fixed(self):
        # random с малой p_edge может дать изолированные узлы
        g = Graph(N=10, topology="random", p_edge=0.01, seed=1)
        for i in range(10):
            assert len(g.neighbors(i)) > 0

    def test_default_positions(self):
        g = Graph(N=5, topology="chain")
        # По умолчанию — цепочка длиной 1 мкм
        assert g.pos[0, 0] == 0.0
        assert np.isclose(g.pos[-1, 0], 1e-6)
        assert np.allclose(g.pos[:, 1], 0.0)

    def test_unknown_topology(self):
        with pytest.raises(ValueError, match="Неизвестная топология"):
            Graph(N=5, topology="star")


# ─── Prompts ───

class TestPrompts:
    def test_analysis_prompt_contains_ion_data(self):
        c = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        z = np.array([+1, -1])
        ctx = {"ion_names": ["Na", "Cl"]}
        prompt = build_analysis_prompt(c, z, time_step=42, context=ctx)
        assert "Na" in prompt
        assert "Cl" in prompt
        assert "42" in prompt
        assert "SUMMARY" in prompt
        assert "WARNINGS" in prompt
        assert "SUGGESTIONS" in prompt

    def test_analysis_prompt_quasineutrality(self):
        c = np.array([[5.0, 5.0], [1.0, 1.0]])
        z = np.array([+1, -1])
        prompt = build_analysis_prompt(c, z, time_step=1)
        assert "Quasineutrality" in prompt or "quasineutrality" in prompt

    def test_validation_prompt_contains_params(self):
        params = {
            "dt": 0.01,
            "F_RT": 38.9,
            "membrane_potential": -0.07,
            "ions": [
                {"name": "Na", "D": 1.33e-9, "z": +1, "c0": 145},
            ],
        }
        prompt = build_validation_prompt(params)
        assert "0.01" in prompt
        assert "38.9" in prompt
        assert "Na" in prompt
        assert "SUMMARY" in prompt

    def test_describe_prompt_contains_text(self):
        prompt = build_describe_prompt("Na concentration: 145 mM")
        assert "Na concentration" in prompt
        assert "SUMMARY" in prompt

    def test_format_array_short(self):
        from network.prompts import _format_array
        arr = np.array([1.0, 2.0, 3.0])
        result = _format_array(arr)
        assert "1.0" in result or "1.0" in result
        assert "..." not in result

    def test_format_array_long(self):
        from network.prompts import _format_array
        arr = np.linspace(0, 100, 100)
        result = _format_array(arr, max_vals=10)
        assert "..." in result
        assert "min=" in result
        assert "max=" in result
