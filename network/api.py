"""API-бэкенд — внешняя нейросеть через OpenAI-совместимый API.

Требует API-ключ в переменной окружения BIOCOMSOL_API_KEY.
Базовый URL настраивается через BIOCOMSOL_API_URL.
Модель — через BIOCOMSOL_API_MODEL.

Поддерживает любой OpenAI-совместимый провайдер:
    DeepSeek:  https://api.deepseek.com/v1       model: deepseek-chat
    OpenAI:    https://api.openai.com/v1        model: gpt-4o-mini
    Yandex:    https://llm.api.cloud.yandex.net  model: yandexgpt
"""

import os
import json
import urllib.request
import urllib.error
from typing import Optional

import numpy as np

from network.base import BaseBackend, AnalysisResult
from network.prompts import build_analysis_prompt, build_validation_prompt, build_describe_prompt


class APIBackend(BaseBackend):
    """Внешняя нейросеть через OpenAI-совместимый API."""

    def __init__(self):
        self._api_key = os.environ.get("BIOCOMSOL_API_KEY", "")
        self._base_url = os.environ.get(
            "BIOCOMSOL_API_URL",
            "https://api.deepseek.com/v1",
        )
        self._model = os.environ.get("BIOCOMSOL_API_MODEL", "deepseek-chat")
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "api"

    def is_available(self) -> bool:
        """Доступен ли API — проверяет наличие ключа."""
        if self._available is not None:
            return self._available
        self._available = bool(self._api_key)
        return self._available

    def _call_api(self, prompt: str) -> Optional[str]:
        """Отправляет запрос к API, возвращает текст ответа."""
        if not self._api_key:
            return None

        try:
            payload = json.dumps({
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a scientific assistant for a biological "
                            "diffusion simulator (Nernst-Planck-Poisson). "
                            "Analyze simulation data and respond concisely. "
                            "Format: SUMMARY, WARNINGS (if any), SUGGESTIONS (if any)."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 1024,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self._base_url}/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                choices = data.get("choices", [])
                if choices:
                    return choices[0]["message"]["content"].strip()
                return None
        except urllib.error.HTTPError as e:
            return None
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
            return None

    def analyze(
        self,
        concentrations: np.ndarray,
        charges: np.ndarray,
        time_step: int,
        context: Optional[dict] = None,
    ) -> AnalysisResult:
        if not self.is_available():
            return AnalysisResult(
                summary="API backend unavailable — no API key set",
                warnings=["Set BIOCOMSOL_API_KEY environment variable to enable API AI"],
                suggestions=[
                    "Create .env file with: BIOCOMSOL_API_KEY=your_key_here",
                    "Or use local backend with Ollama",
                ],
                confidence=0.0,
                backend="api",
            )

        prompt = build_analysis_prompt(concentrations, charges, time_step, context)
        raw = self._call_api(prompt)

        if raw is None:
            return AnalysisResult(
                summary="API call failed",
                warnings=["Request error — check API key, URL, and network"],
                suggestions=["Verify BIOCOMSOL_API_URL and BIOCOMSOL_API_MODEL"],
                confidence=0.0,
                backend="api",
            )

        return self._parse_response(raw)

    def validate_params(self, params: dict) -> AnalysisResult:
        if not self.is_available():
            return AnalysisResult(
                summary="API backend unavailable — no API key set",
                warnings=["Set BIOCOMSOL_API_KEY to enable parameter validation"],
                suggestions=["Use local or manual backend as fallback"],
                confidence=0.0,
                backend="api",
            )

        prompt = build_validation_prompt(params)
        raw = self._call_api(prompt)

        if raw is None:
            return AnalysisResult(
                summary="API call failed",
                warnings=["Request error during validation"],
                suggestions=["Check API connectivity"],
                confidence=0.0,
                backend="api",
            )

        return self._parse_response(raw)

    def describe(self, text: str) -> AnalysisResult:
        if not self.is_available():
            return AnalysisResult(
                summary="API backend unavailable",
                warnings=["No API key configured"],
                suggestions=["Set BIOCOMSOL_API_KEY or use another backend"],
                confidence=0.0,
                backend="api",
            )

        prompt = build_describe_prompt(text)
        raw = self._call_api(prompt)

        if raw is None:
            return AnalysisResult(
                summary="API call failed",
                warnings=["Request error during description generation"],
                suggestions=[],
                confidence=0.0,
                backend="api",
            )

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> AnalysisResult:
        """Парсит ответ API в AnalysisResult."""
        warnings = []
        suggestions = []
        summary_lines = []

        in_warnings = False
        in_suggestions = False

        for line in raw.split("\n"):
            line = line.strip()
            low = line.lower()

            if "warning" in low or low.startswith("warn"):
                in_warnings = True
                in_suggestions = False
                # Текст после маркера — тоже предупреждение
                rest = line.split(":", 1)
                if len(rest) > 1 and rest[1].strip():
                    warnings.append(rest[1].strip())
                continue

            if "suggest" in low or "recommend" in low:
                in_warnings = False
                in_suggestions = True
                rest = line.split(":", 1)
                if len(rest) > 1 and rest[1].strip():
                    suggestions.append(rest[1].strip())
                continue

            if low.startswith("summary") or low.startswith("анализ") or low.startswith("результат"):
                in_warnings = False
                in_suggestions = False
                rest = line.split(":", 1)
                if len(rest) > 1 and rest[1].strip():
                    summary_lines.append(rest[1].strip())
                continue

            if not line:
                continue

            if in_warnings:
                warnings.append(line)
            elif in_suggestions:
                suggestions.append(line)
            else:
                summary_lines.append(line)

        summary = " ".join(summary_lines[:4]) if summary_lines else raw[:500]

        return AnalysisResult(
            summary=summary,
            warnings=warnings,
            suggestions=suggestions,
            confidence=0.85,
            backend="api",
        )
