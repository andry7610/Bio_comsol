"""Шаблоны промптов для нейросетевых бэкендов.

Превращают сырые данные симуляции (массивы, числа) в текстовые промпты,
понятные языковой модели. Используются и local.py, и api.py.
"""

import numpy as np
from typing import Optional


def _format_array(arr: np.ndarray, max_vals: int = 10) -> str:
    """Сжимает массив в строку: первые и последние значения, min/max."""
    if arr.size <= max_vals:
        return str(np.round(arr, 4).tolist())
    head = np.round(arr[:max_vals // 2], 4).tolist()
    tail = np.round(arr[-max_vals // 2:], 4).tolist()
    return f"{head} ... {tail} (min={arr.min():.4f}, max={arr.max():.4f})"


def build_analysis_prompt(
    concentrations: np.ndarray,
    charges: np.ndarray,
    time_step: int,
    context: Optional[dict] = None,
) -> str:
    """Промпт для анализа состояния симуляции."""
    context = context or {}
    ion_names = context.get("ion_names", [f"ion_{i}" for i in range(len(charges))])
    phi = context.get("phi", None)
    membrane = context.get("membrane_potential", None)
    dt = context.get("dt", None)

    lines = [
        "Analyze the following state of a biological ion diffusion simulation.",
        "The system uses Nernst-Planck-Poisson equations on a graph.",
        "",
        f"Time step: {time_step}",
        f"Number of ions: {len(ion_names)}",
    ]

    if dt is not None:
        lines.append(f"dt: {dt}")

    if membrane is not None:
        lines.append(f"Membrane potential: {membrane}")

    lines.append("")
    lines.append("Ion data (name, charge z, concentration profile):")

    for k in range(len(charges)):
        c = concentrations[k]
        name = ion_names[k] if k < len(ion_names) else f"ion_{k}"
        z = charges[k]
        lines.append(
            f"  {name} (z={z:+.0f}): {_format_array(c)}"
        )

    if phi is not None:
        lines.append("")
        lines.append(f"Potential profile: {_format_array(phi)}")

    # Вычисляемые метрики
    total_charge = np.sum(concentrations * charges[:, None], axis=0)
    qn_err = float(np.max(np.abs(total_charge)))
    c_min = float(concentrations.min())
    c_max = float(concentrations.max())
    neg_count = int((concentrations < 0).sum())
    nan_count = int((~np.isfinite(concentrations)).sum())

    lines.append("")
    lines.append("Computed metrics:")
    lines.append(f"  Quasineutrality error: {qn_err:.4e}")
    lines.append(f"  Concentration range: [{c_min:.4f}, {c_max:.4f}]")
    lines.append(f"  Negative values: {neg_count}")
    lines.append(f"  NaN/Inf values: {nan_count}")

    lines.append("")
    lines.append(
        "Respond in this format:\n"
        "SUMMARY: <brief description of the simulation state>\n"
        "WARNINGS: <list any issues: instability, non-physical values, etc.>\n"
        "SUGGESTIONS: <recommendations: adjust dt, boundary conditions, etc.>"
    )

    return "\n".join(lines)


def build_validation_prompt(params: dict) -> str:
    """Промпт для проверки параметров симуляции перед запуском."""
    lines = [
        "Validate the following parameters for a Nernst-Planck-Poisson",
        "biological diffusion simulation. Check for physical correctness,",
        "numerical stability risks, and common mistakes.",
        "",
        "Parameters:",
    ]

    dt = params.get("dt")
    F_RT = params.get("F_RT")
    membrane = params.get("membrane_potential")
    ions = params.get("ions", [])

    if dt is not None:
        lines.append(f"  dt: {dt}")

    if F_RT is not None:
        lines.append(f"  F/RT: {F_RT}")

    if membrane is not None:
        lines.append(f"  Membrane potential: {membrane}")

    lines.append("")
    lines.append("Ions:")
    for ion in ions:
        lines.append(
            f"  {ion.get('name', '?')}: "
            f"D={ion.get('D', '?')}, "
            f"z={ion.get('z', '?')}, "
            f"c0={ion.get('c0', '?')}, "
            f"c_left={ion.get('c_left', '?')}, "
            f"c_right={ion.get('c_right', '?')}"
        )

    lines.append("")
    lines.append(
        "Respond in this format:\n"
        "SUMMARY: <verdict: are parameters valid?>\n"
        "WARNINGS: <any problems: non-physical values, stability risks>\n"
        "SUGGESTIONS: <fixes or improvements>"
    )

    return "\n".join(lines)


def build_describe_prompt(text: str) -> str:
    """Промпт для генерации человекочитаемого описания результатов."""
    return (
        "Write a concise scientific description of the following "
        "simulation results. Use clear language suitable for a research report. "
        "Focus on physical interpretation, not raw numbers.\n\n"
        f"Data:\n{text}\n\n"
        "Format:\n"
        "SUMMARY: <description of what happened in the simulation>\n"
        "WARNINGS: <any concerns>\n"
        "SUGGESTIONS: <next steps or experiments to try>"
    )
