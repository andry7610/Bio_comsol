"""network — нейросетевые бэкенды и графовая топология для bio_comsol.

Экспортирует:
    get_backend  — автоматический выбор ИИ-бэкенда
    get_backend_info — статус всех бэкендов
    BaseBackend  — абстрактный интерфейс
    AnalysisResult — формат ответа
    Graph        — метрический граф с пространственной топологией
"""

from network.base import BaseBackend, AnalysisResult
from network.config import get_backend, get_backend_info
from network.graph import Graph

__all__ = [
    "get_backend",
    "get_backend_info",
    "BaseBackend",
    "AnalysisResult",
    "Graph",
]

__version__ = "0.3.1"

