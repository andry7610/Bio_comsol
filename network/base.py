"""Абстрактный интерфейс для нейросетевых бэкендов bio_comsol.

Все бэкенды (manual, local, api) реализуют один интерфейс.
Ядро biological/ не знает, какой бэкенд активен — оно просто зовёт методы.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnalysisResult:
    """Результат анализа симуляции нейросетью."""
    summary: str           # Краткое описание результата
    warnings: list         # Список предупреждений (строки)
    suggestions: list       # Список рекомендаций (строки)
    confidence: float      # 0.0 — 1.0, уверенность модели
    backend: str           # Какой бэкенд ответил: "manual", "local", "api"

    def __str__(self):
        lines = [f"[{self.backend}] {self.summary}"]
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  ⚠ {w}" for w in self.warnings)
        if self.suggestions:
            lines.append("Suggestions:")
            lines.extend(f"  → {s}" for s in self.suggestions)
        lines.append(f"Confidence: {self.confidence:.0%}")
        return "\n".join(lines)


class BaseBackend(ABC):
    """Базовый класс для всех ИИ-бэкендов."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя бэкенда: 'manual', 'local', 'api'."""
        ...

    @abstractmethod
    def analyze(
        self,
        concentrations: "np.ndarray",
        charges: "np.ndarray",
        time_step: int,
        context: Optional[dict] = None,
    ) -> AnalysisResult:
        """Анализ состояния симуляции.

        Args:
            concentrations: массив концентраций ионов [n_ions, N]
            charges: массив зарядов [n_ions]
            time_step: текущий шаг симуляции
            context: доп. параметры (имена ионов, потенциал и т.д.)

        Returns:
            AnalysisResult с описанием, предупреждениями, рекомендациями
        """
        ...

    @abstractmethod
    def validate_params(self, params: dict) -> AnalysisResult:
        """Проверка параметров симуляции перед запуском.

        Args:
            params: словарь параметров (dt, F_RT, ионы, мембранный потенциал)

        Returns:
            AnalysisResult с вердиктом и рекомендациями
        """
        ...

    @abstractmethod
    def describe(self, text: str) -> AnalysisResult:
        """Генерация текстового описания результатов.

        Args:
            text: сырые данные/числа для описания

        Returns:
            AnalysisResult с описанием
        """
        ...

    def is_available(self) -> bool:
        """Доступен ли бэкенд прямо сейчас."""
        return True
