"""network/neuron_analysis.py — ИИ-анализ нейро-слоя."""

import numpy as np
from network.base import AnalysisResult


def manual_neuron_analysis(neuron):
    """Эвристический анализ нейро-слоя для ManualBackend."""
    warnings = []
    suggestions = []
    n_neurons = len(neuron.neuron_indices)

    spikes = np.sum(np.abs(neuron.last_current) > 50.0)
    max_current = float(np.max(np.abs(neuron.last_current))) if n_neurons > 0 else 0.0

    gate_issues = []
    if n_neurons > 0:
        if np.any(neuron.m < 0) or np.any(neuron.m > 1):
            gate_issues.append("m")
        if np.any(neuron.h < 0) or np.any(neuron.h > 1):
            gate_issues.append("h")
        if np.any(neuron.n < 0) or np.any(neuron.n > 1):
            gate_issues.append("n")

    for g in gate_issues:
        warnings.append(f"Ворота {g} вышли за пределы [0, 1]")

    if n_neurons > 0 and max_current > 500.0:
        warnings.append(f"Аномально высокий ток: {max_current:.1f}")
        suggestions.append("Уменьшите g_Na или dt")

    if n_neurons > 0 and spikes == 0:
        suggestions.append("Нет спайков — возможно, g_Na слишком мала или потенциал покоя слишком низкий")
    elif spikes > 0:
        suggestions.append(f"Обнаружено {spikes} активных нейронов — система возбудима")

    summary = f"Neurons: {n_neurons}, spikes: {spikes}, max |I|: {max_current:.2f}"
    confidence = 0.7 if n_neurons > 0 else 0.5

    return summary, warnings, suggestions, confidence


def manual_neuron_validation(params: dict):
    """Проверка параметров нейрона перед запуском."""
    warnings = []
    suggestions = []

    dt = params.get("dt", 1e-3)
    if dt <= 0:
        warnings.append("dt <= 0 — недопустимо")
    if dt > 0.1:
        warnings.append(f"dt={dt} слишком велик для HH-динамики")
        suggestions.append("Используйте dt < 0.01 для устойчивости ворот")

    g_Na = params.get("g_Na", 120.0)
    if g_Na <= 0:
        warnings.append("g_Na <= 0 — не будет спайков")

    g_K = params.get("g_K", 36.0)
    if g_K <= 0:
        warnings.append("g_K <= 0 — не будет реполяризации")

    E_Na = params.get("E_Na", 50.0)
    E_K = params.get("E_K", -75.0)
    if E_Na <= E_K:
        warnings.append("E_Na <= E_K — некорректные равновесные потенциалы")

    neuron_nodes = params.get("neuron_nodes", [])
    if len(neuron_nodes) == 0:
        warnings.append("Список нейронных узлов пуст")
    if any(n < 0 for n in neuron_nodes):
        warnings.append("Отрицательные индексы узлов")

    summary = "OK" if not warnings else f"{len(warnings)} предупреждений"
    confidence = 0.6

    return summary, warnings, suggestions, confidence


def analyze_neuron(neuron, backend):
    """Анализ нейрона через любой бэкенд."""
    n_neurons = len(neuron.neuron_indices)

    if n_neurons == 0:
        return AnalysisResult(
            summary="Нет нейронных узлов",
            warnings=[],
            suggestions=[],
            confidence=0.5,
            backend=backend.name,
        )

    context = {
        "neuron": True,
        "neuron_indices": neuron.neuron_indices.tolist(),
        "m": neuron.m.tolist(),
        "h": neuron.h.tolist(),
        "n": neuron.n.tolist(),
        "last_current": neuron.last_current.tolist(),
        "dt": neuron.dt,
        "g_Na": neuron.g_Na,
        "g_K": neuron.g_K,
        "E_Na": neuron.E_Na,
        "E_K": neuron.E_K,
    }

    if backend.name == "manual":
        summary, warnings, suggestions, confidence = manual_neuron_analysis(neuron)
        return AnalysisResult(summary, warnings, suggestions, confidence, "manual")

    dummy_conc = np.array([neuron.m, neuron.h, neuron.n])
    dummy_charges = np.array([1, 1, 1])
    return backend.analyze(dummy_conc, dummy_charges, 0, context)


def validate_neuron(neuron, backend):
    """Валидация параметров нейрона."""
    params = {
        "dt": neuron.dt,
        "g_Na": neuron.g_Na,
        "g_K": neuron.g_K,
        "E_Na": neuron.E_Na,
        "E_K": neuron.E_K,
        "neuron_nodes": neuron.neuron_indices.tolist(),
    }

    if backend.name == "manual":
        summary, warnings, suggestions, confidence = manual_neuron_validation(params)
        return AnalysisResult(summary, warnings, suggestions, confidence, "manual")

    return backend.validate_params(params)
