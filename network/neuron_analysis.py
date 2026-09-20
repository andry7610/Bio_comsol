"""network/neuron_analysis.py — Анализ нейро-слоя через BaseBackend.

Совместим с network/base.py: использует тот же AnalysisResult.
Не наследует BaseBackend — это вспомогательный класс, который
вызывает существующий бэкенд с расширенным context.

Логика:
    1. NeuronLayer вызывает analyze_state(backend, phi, c, z)
    2. Здесь данные нейрона упаковываются в context
    3. Backend.analyze() вызывается с обычными концентрациями + context
    4. ManualBackend / LocalBackend / ApiBackend обрабатывают context["neuron"]
"""

import numpy as np
from network.base import AnalysisResult


class NeuronAnalyzer:
    """Анализатор состояния нейро-слоя.

    Работает поверх любого BaseBackend: упаковывает нейронные данные
    (ворота m/h/n, токи, потенциалы в нейронных узлах) в context
    и передаёт в backend.analyze().
    """

    @staticmethod
    def analyze(backend, neuron_layer, phi, c, z):
        """Полный анализ нейро-слоя через бэкенд.

        Args:
            backend: объект BaseBackend (manual, local, api)
            neuron_layer: NeuronLayer
            phi: потенциал во всех узлах (N,)
            c: концентрации (n_ions, N)
            z: заряды (n_ions,)

        Returns:
            AnalysisResult
        """
        # Данные нейронов для context
        n_idx = neuron_layer.neuron_indices
        neuron_data = {
            "type": "neuron_analysis",
            "n_neurons": len(n_idx),
            "neuron_indices": n_idx.tolist(),
            "phi_neuron": phi[n_idx].tolist() if len(n_idx) > 0 else [],
            "gates": {
                "m": neuron_layer.m.tolist() if len(n_idx) > 0 else [],
                "h": neuron_layer.h.tolist() if len(n_idx) > 0 else [],
                "n": neuron_layer.n.tolist() if len(n_idx) > 0 else [],
            },
            "last_current": neuron_layer.last_current.tolist() if len(n_idx) > 0 else [],
            "dt": neuron_layer.dt,
        }

        # Общий context
        context = {
            "neuron": neuron_data,
            "ion_names": ["Na", "K", "Cl", "Ca"][: c.shape[0]],
            "phi_global": phi.tolist(),
            "membrane_potential": float(phi[-1]) if len(phi) > 0 else 0.0,
        }

        # Вызываем бэкенд (передаём концентрации и заряды как обычно)
        result = backend.analyze(
            concentrations=c,
            charges=z,
            time_step=neuron_layer.step_count if hasattr(neuron_layer, "step_count") else 0,
            context=context,
        )

        return result

    @staticmethod
    def validate(backend, params):
        """Проверка параметров нейрона через бэкенд.

        Args:
            backend: объект BaseBackend
            params: dict с параметрами нейрона
                (dt, g_Na, g_K, E_Na, E_K, neuron_nodes)

        Returns:
            AnalysisResult
        """
        context = {
            "neuron": {
                "type": "neuron_validation",
                "params": params,
            }
        }

        return backend.validate_params(params)


# ─── Локальные хелперы для ManualBackend ───

def manual_neuron_analysis(neuron_data):
    """Эвристический анализ нейро-слоя для ManualBackend.

    Вызывается из manual.py когда context["neuron"] присутствует.
    Возвращает (summary, warnings, suggestions, confidence).
    """
    warnings = []
    suggestions = []
    n_neurons = neuron_data.get("n_neurons", 0)

    if n_neurons == 0:
        return (
            "No active neurons in the graph.",
            [],
            ["Add neuron nodes to see action potentials."],
            0.5,
        )

    phi_neuron = np.array(neuron_data.get("phi_neuron", []))
    gates = neuron_data.get("gates", {})
    m = np.array(gates.get("m", []))
    h = np.array(gates.get("h", []))
    n_g = np.array(gates.get("n", []))
    currents = np.array(neuron_data.get("last_current", []))

    # --- Спайки ---
    n_spikes = int(np.sum(phi_neuron > 0.0)) if len(phi_neuron) > 0 else 0
    if n_spikes > 0:
        summary = f"Neuron activity detected: {n_spikes}/{n_neurons} neurons spiking."
    else:
        summary = f"Neurons at rest: 0/{n_neurons} spiking. System calm."
        if n_neurons > 0:
            suggestions.append("Increase g_Na or apply external stimulus to trigger action potentials.")

    # --- Ворота ---
    if len(m) > 0:
        if np.any(m > 1.0) or np.any(m < 0.0):
            warnings.append(f"Gate m out of [0,1]: min={m.min():.3f}, max={m.max():.3f}")
        if np.any(h > 1.0) or np.any(h < 0.0):
            warnings.append(f"Gate h out of [0,1]: min={h.min():.3f}, max={h.max():.3f}")
        if np.any(n_g > 1.0) or np.any(n_g < 0.0):
            warnings.append(f"Gate n out of [0,1]: min={n_g.min():.3f}, max={n_g.max():.3f}")

    # --- Токи ---
    if len(currents) > 0:
        max_I = np.max(np.abs(currents))
        if max_I > 1000.0:
            warnings.append(f"Very high membrane current: {max_I:.1f} — possible numerical instability.")
        elif max_I < 0.01:
            suggestions.append("Membrane current near zero — check if channels are active.")

    # --- Потенциал ---
    if len(phi_neuron) > 0:
        v_min, v_max = float(phi_neuron.min()), float(phi_neuron.max())
        if v_max > 60.0:
            warnings.append(f"Potential very high: {v_max:.1f} mV — check clipping.")
        if v_min < -100.0:
            warnings.append(f"Potential very low: {v_min:.1f} mV — hyperpolarization too strong.")

    confidence = 0.5  # Manual backend — фиксированная уверенность

    return summary, warnings, suggestions, confidence


def manual_neuron_validation(params):
    """Эвристическая валидация параметров нейрона для ManualBackend."""
    warnings = []
    suggestions = []

    dt = params.get("dt", 0.01)
    if dt <= 0:
        warnings.append("dt must be positive.")
    elif dt > 1.0:
        warnings.append(f"dt={dt} is very large for neuron dynamics — may miss spikes.")

    g_Na = params.get("g_Na", 120.0)
    g_K = params.get("g_K", 36.0)
    if g_Na <= 0:
        warnings.append("g_Na must be positive for action potentials.")
    if g_K <= 0:
        warnings.append("g_K must be positive for repolarization.")

    E_Na = params.get("E_Na", 50.0)
    E_K = params.get("E_K", -75.0)
    if E_Na < E_K:
        warnings.append("E_Na < E_K — check sign convention (E_Na should be positive).")

    nodes = params.get("neuron_nodes", [])
    if len(nodes) == 0:
        suggestions.append("No neuron nodes specified — the layer will be passive.")

    summary = "Neuron parameters validated."
    if not warnings:
        summary += " No issues found."
    confidence = 0.5

    return summary, warnings, suggestions, confidence
