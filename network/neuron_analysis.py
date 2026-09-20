"""network/neuron_analysis.py — ИИ-анализ нейро-слоя.

Расширяет существующий network/ для анализа NeuronLayer.
Использует тот же интерфейс AnalysisResult из base.py.
"""

import numpy as np
from network.base import AnalysisResult


def manual_neuron_analysis(neuron) -> tuple:
    """Эвристический анализ нейро-слоя для ManualBackend.

    Returns:
        (summary, warnings, suggestions, confidence)
    """
    warnings = []
    suggestions = []
    n_neurons = len(neuron.neuron_indices)

    # --- Спайки ---
    spikes = 0
    max_current = 0.0
    if n_neurons > 0:
        spikes = int(np.sum(np.abs(neuron.last_current) > 50.0))
        max_current = float(np.max(np.abs(neuron.last_current)))

    # --- Ворота ---
    gate_issues = []
    if n_neurons > 0:
        if np.any(neuron.m < 0) or np.any(neuron.m > 1):
            gate_issues.append("m")
        if np.any(neuron.h < 0) or np.any(neuron.h > 1):
            gate_issues.append("h")
        if np.any(neuron.n < 0) or np.any(neuron.n > 1):
            gate_issues.append("n")

    for g in gate_issues:
        warnings.append("Gates {} out of [0, 1]".format(g))

    # --- Токи ---
    if n_neurons > 0 and max_current > 500.0:
        warnings.append("High current: {:.1f}".format(max_current))
        suggestions.append("Reduce g_Na or dt")

    # --- Спайки ---
    if n_neurons > 0 and spikes == 0:
        suggestions.append("No spikes detected — g_Na too small or resting V too low")
    elif spikes > 0:
        suggestions.append("{} active neurons detected — system is excitable".format(spikes))

    summary = "Neurons: {}, spikes: {}, max |I|: {:.2f}".format(n_neurons, spikes, max_current)
    confidence = 0.7 if n_neurons > 0 else 0.5

    return summary, warnings, suggestions, confidence


def manual_neuron_validation(params) -> tuple:
    """Проверка параметров нейрона перед запуском."""
    warnings = []
    suggestions = []

    dt = params.get("dt", 1e-3)
    if dt <= 0:
        warnings.append("dt <= 0")
    if dt > 0.1:
        warnings.append("dt={} too large for HH dynamics".format(dt))
        suggestions.append("Use dt < 0.01 for gate stability")

    g_Na = params.get("g_Na", 120.0)
    if g_Na <= 0:
        warnings.append("g_Na <= 0 — no spikes")

    g_K = params.get("g_K", 36.0)
    if g_K <= 0:
        warnings.append("g_K <= 0 — no repolarization")

    E_Na = params.get("E_Na", 50.0)
    E_K = params.get("E_K", -75.0)
    if E_Na <= E_K:
        warnings.append("E_Na <= E_K — invalid equilibrium potentials")

    neuron_nodes = params.get("neuron_nodes", [])
    if len(neuron_nodes) == 0:
        warnings.append("Neuron nodes list is empty")
    if any(n < 0 for n in neuron_nodes):
        warnings.append("Negative node indices")

    summary = "OK" if not warnings else "{} warnings".format(len(warnings))
    confidence = 0.6

    return summary, warnings, suggestions, confidence


def analyze_neuron(neuron, backend) -> AnalysisResult:
    """Анализ нейрона через любой бэкенд (manual/local/api)."""
    n_neurons = len(neuron.neuron_indices)

    if n_neurons == 0:
        return AnalysisResult(
            summary="No neuron nodes",
            warnings=[],
            suggestions=[],
            confidence=0.5,
            backend=backend.name,
        )

    if backend.name == "manual":
        summary, warnings, suggestions, confidence = manual_neuron_analysis(neuron)
        return AnalysisResult(summary, warnings, suggestions, confidence, "manual")

    # Для local/api — передаём через стандартный analyze
    dummy_conc = np.array([neuron.m, neuron.h, neuron.n])
    dummy_charges = np.array([1, 1, 1])
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
    return backend.analyze(dummy_conc, dummy_charges, 0, context)


def validate_neuron(neuron, backend) -> AnalysisResult:
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
