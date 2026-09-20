"""Локальный бэкенд — Ollama + DeepSeek (или любая другая локальная модель).

Требует установленный Ollama и скачанную модель.
По умолчанию: deepseek-r1:7b
Настройка через переменные окружения:
    BIOCOMSOL_LOCAL_MODEL — имя модели в Ollama
    BIOCOMSOL_OLLAMA_URL — URL Ollama (по умолчанию http://localhost:11434)
"""

import os
import json
import urllib.request
import urllib.error
from typing import Optional

import numpy as np

from network.base import BaseBackend, AnalysisResult
from network.prompts import build_analysis_prompt, build_validation_prompt, build_describe_prompt


class LocalBackend(BaseBackend):
    """Локальная нейросеть через Ollama API."""

    def __init__(self):
        self._model = os.environ.get("BIOCOMSOL_LOCAL_MODEL", "deepseek-r1:7b")
        self._url = os.environ.get("BIOCOMSOL_OLLAMA_URL", "http://localhost:11434")
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "local"

    def is_available(self) -> bool:
        """Проверяет, запущена ли Ollama и есть ли модель."""
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(
                f"{self._url}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                models = [m["name"] for m in data.get("models", [])]
                self._available = any(self._model in m for m in models)
                if not self._available:
                    self._available = len(models) > 0
                    if self._available:
                        self._model = models[0]
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            self._available = False
        return self._available

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Отправляет промпт в Ollama, возвращает текст ответа."""
        try:
            payload = json.dumps({
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self._url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
                return data.get("response", "").strip()
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as e:
            return None

    def analyze(
        self,
        concentrations: np.ndarray,
        charges: np.ndarray,
        time_step: int,
        context: Optional[dict] = None,
    ) -> AnalysisResult:
        context = context or {}

        if not self.is_available():
            return AnalysisResult(
                summary="Local backend unavailable — Ollama not running",
                warnings=["Install Ollama and pull a model to enable local AI"],
                suggestions=["Run: ollama pull deepseek-r1:7b"],
                confidence=0.0,
                backend="local",
            )

        prompt = build_analysis_prompt(concentrations, charges, time_step, context)
        raw = self._call_ollama(prompt)

        if raw is None:
            return AnalysisResult(
                summary="Local model call failed",
                warnings=["Ollama request error — check logs"],
                suggestions=["Verify Ollama is running and model is pulled"],
                confidence=0.0,
                backend="local",
            )

        return self._parse_response(raw)

    def validate_params(self, params: dict) -> AnalysisResult:
        if not self.is_available():
            return AnalysisResult(
                summary="Local backend unavailable — Ollama not running",
                warnings=["Install Ollama and pull a model to enable local AI"],
                suggestions=["Run: ollama pull deepseek-r1:7b"],
                confidence=0.0,
                backend="local",
            )

        prompt = build_validation_prompt(params)
        raw = self._call_ollama(prompt)

        if raw is None:
            return AnalysisResult(
                summary="Local model call failed",
                warnings=["Ollama request error"],
                suggestions=["Verify Ollama is running"],
                confidence=0.0,
                backend="local",
            )

        return self._parse_response(raw)

    def describe(self, text: str) -> AnalysisResult:
        if not self.is_available():
            return AnalysisResult(
                summary="Local backend unavailable",
                warnings=["Ollama not running"],
                suggestions=["Start Ollama to enable AI descriptions"],
                confidence=0.0,
                backend="local",
            )

        prompt = build_describe_prompt(text)
        raw = self._call_ollama(prompt)

        if raw is None:
            return AnalysisResult(
                summary="Local model call failed",
                warnings=["Ollama request error"],
                suggestions=[],
                confidence=0.0,
                backend="local",
            )

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> AnalysisResult:
        """Парсит ответ модели в AnalysisResult.

        Модель может вернуть текст в свободной форме —
        стараемся извлечь ключевые части, но не ломаемся, если формат другой.
        """
        warnings = []
        suggestions = []
        summary = raw[:500] if len(raw) > 500 else raw

        # Пытаемся найти маркеры в ответе
        lines = raw.split("\n")
        summary_lines = []
        in_warnings = False
        in_suggestions = False

        for line in lines:
            line = line.strip()
            low = line.lower()
            if "warning" in low or "warning" in low or "warning:" in low:
                in_warnings = True
                in_suggestions = False
                continue
            if "suggest" in low or "recommend" in low:
                in_warnings = False
                in_suggestions = True
                continue
            if low.startswith("summary") or low.startswith("анализ") or low.startswith("результат"):
                in_warnings = False
                in_suggestions = False
                continue

            if not line:
                continue

            if in_warnings:
                warnings.append(line)
            elif in_suggestions:
                suggestions.append(line)
            else:
                summary_lines.append(line)

        if summary_lines:
            summary = " ".join(summary_lines[:3])

        return AnalysisResult(
            summary=summary,
            warnings=warnings,
            suggestions=suggestions,
            confidence=0.75,
            backend="local",
        )
