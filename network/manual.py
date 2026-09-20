"""Ручной бэкенд — без нейросети.

Работает всегда: шаблонные ответы на основе анализа данных.
Используется по умолчанию, если нет ни API, ни локальной модели.
"""

from typing import Optional
import numpy as np

from network.base import BaseBackend, AnalysisResult


class ManualBackend(BaseBackend):
    """Бэкенд без ИИ — простые эвристики по данным симуляции."""

    @property
    def name(self) -> str:
        return "manual"

    def is_available(self) -> bool:
        return True

    def analyze(
        self,
        concentrations: np.ndarray,
        charges: np.ndarray,
        time_step: int,
        context: Optional[dict] = None,
    ) -> AnalysisResult:
        context = context or {}
        ion_names = context.get("ion_names", [f"ion_{i}" for i in range(len(charges))])

        warnings = []
        suggestions = []

        # Проверка на отрицательные концентрации
        neg_count = int((concentrations < 0).sum())
        if neg_count > 0:
            warnings.append(f"Negative concentrations detected: {neg_count} cells")

        # Проверка на NaN / Inf
        if not np.isfinite(concentrations).all():
            warnings.append("NaN or Inf detected in concentrations — instability likely")

        # Проверка квазинейтральности
        total_charge = np.sum(concentrations * charges[:, None], axis=0)
        qn_error = float(np.max(np.abs(total_charge)))
        if qn_error > 1.0:
            warnings.append(f"Quasineutrality error high: {qn_error:.4e}")
            suggestions.append("Consider smaller dt or more Picard iterations")

        # Диапазон концентраций
        c_min = float(concentrations.min())
        c_max = float(concentrations.max())
        if c_max > 1e4:
            warnings.append(f"Very high concentration: {c_max:.2e}")
            suggestions.append("Check boundary conditions — values may be unrealistic")

        # Сводка
        summary = (
            f"Step {time_step}: {len(ion_names)} ions, "
            f"concentration range [{c_min:.4f} .. {c_max:.4f}], "
            f"QN error = {qn_error:.4e}"
        )

        if not warnings:
            summary += " — stable"
        if not suggestions:
            suggestions.append("Parameters look reasonable — proceed with simulation")

        return AnalysisResult(
            summary=summary,
            warnings=warnings,
            suggestions=suggestions,
            confidence=0.5,
            backend="manual",
        )

    def validate_params(self, params: dict) -> AnalysisResult:
        warnings = []
        suggestions = []

        dt = params.get("dt", 0.0)
        if dt <= 0:
            warnings.append("dt must be positive")
        elif dt > 1.0:
            warnings.append(f"dt={dt} is large — may cause instability")
            suggestions.append("Consider dt < 0.1 for stiff systems")

        F_RT = params.get("F_RT", 0.0)
        if F_RT <= 0:
            warnings.append("F_RT must be positive")

        ions = params.get("ions", [])
        for ion in ions:
            if ion.get("c0", 0) < 0:
                warnings.append(f"{ion.get('name', '?')}: c0 is negative")
            if ion.get("z", 0) == 0:
                warnings.append(f"{ion.get('name', '?')}: charge z=0 — no migration")

        membrane = params.get("membrane_potential", 0.0)
        if abs(membrane) > 10:
            warnings.append(f"Membrane potential {membrane} is extreme")

        summary = "Parameters checked (heuristic rules)"
        if not warnings:
            summary += " — no issues found"
            suggestions.append("Parameters look valid — ready to run")

        return AnalysisResult(
            summary=summary,
            warnings=warnings,
            suggestions=suggestions,
            confidence=0.6,
            backend="manual",
        )

    def describe(self, text: str) -> AnalysisResult:
        return AnalysisResult(
            summary=f"Manual description: {text[:200]}",
            warnings=[],
            suggestions=["Use a local or API backend for detailed analysis"],
            confidence=0.3,
            backend="manual",
        )
