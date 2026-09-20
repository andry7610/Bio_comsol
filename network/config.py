"""Автоматический выбор ИИ-бэкенда.

Приоритет:
    1. API  — если установлен BIOCOMSOL_API_KEY
    2. Local — если запущена Ollama и есть модель
    3. Manual — всегда работает, без нейросети

Переменные окружения (читаются из .env или системы):
    BIOCOMSOL_API_KEY   — ключ API (DeepSeek, OpenAI, Yandex GPT)
    BIOCOMSOL_API_URL    — базовый URL API (по умолчанию DeepSeek)
    BIOCOMSOL_API_MODEL  — имя модели API
    BIOCOMSOL_LOCAL_MODEL — имя модели в Ollama (по умолчанию deepseek-r1:7b)
    BIOCOMSOL_OLLAMA_URL  — URL Ollama (по умолчанию http://localhost:11434)
    BIOCOMSOL_BACKEND    — принудительный выбор: "api", "local", "manual"
"""

import os
from typing import Optional

from network.base import BaseBackend
from network.manual import ManualBackend
from network.local import LocalBackend
from network.api import APIBackend


def _load_env_file() -> None:
    """Читает .env из корня проекта, если он есть."""
    try:
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".env",
        )
        if os.path.isfile(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
    except (OSError, IOError):
        pass


_load_env_file()


def get_backend(forced: Optional[str] = None) -> BaseBackend:
    """Возвращает лучший доступный бэкенд.

    Args:
        forced: принудительный выбор — "api", "local" или "manual".
                Если None — выбирается автоматически.

    Returns:
        Экземпляр BaseBackend.
    """
    choice = forced or os.environ.get("BIOCOMSOL_BACKEND", "")

    if choice == "api":
        return APIBackend()
    if choice == "local":
        return LocalBackend()
    if choice == "manual":
        return ManualBackend()

    # Автоматический выбор по приоритету
    api = APIBackend()
    if api.is_available():
        return api

    local = LocalBackend()
    if local.is_available():
        return local

    return ManualBackend()


def get_backend_info() -> dict:
    """Возвращает статус всех бэкендов — для диагностики и вывода."""
    api = APIBackend()
    local = LocalBackend()

    return {
        "api": {
            "available": api.is_available(),
            "model": api._model,
            "url": api._base_url,
        },
        "local": {
            "available": local.is_available(),
            "model": local._model,
            "url": local._url,
        },
        "manual": {
            "available": True,
            "model": None,
            "url": None,
        },
        "active": os.environ.get("BIOCOMSOL_BACKEND", "auto"),
    }


__all__ = ["get_backend", "get_backend_info", "BaseBackend"]
