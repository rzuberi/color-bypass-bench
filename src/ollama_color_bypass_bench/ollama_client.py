"""Minimal Ollama /api/chat client with retries and timeout handling."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence

import requests

from .config import DecodeParams, OllamaConfig


@dataclass(frozen=True)
class ChatMessage:
    """Chat message payload for Ollama-compatible role/content messages."""

    role: str
    content: str


class OllamaClientError(RuntimeError):
    """Raised when Ollama chat requests fail permanently."""


class OllamaChatClient:
    """Thin wrapper around Ollama's ``/api/chat`` endpoint."""

    def __init__(self, config: OllamaConfig) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._timeout_seconds = config.timeout_seconds
        self._max_retries = max(0, config.max_retries)
        self._retry_backoff_seconds = max(0.0, config.retry_backoff_seconds)
        self._session = requests.Session()

    def _normalize_messages(self, messages: Sequence[Mapping[str, str] | ChatMessage]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, ChatMessage):
                role = message.role
                content = message.content
            else:
                role = str(message["role"])
                content = str(message["content"])
            normalized.append({"role": role, "content": content})
        return normalized

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, str] | ChatMessage],
        decode: DecodeParams,
        *,
        seed: int | None = None,
    ) -> str:
        """Call Ollama chat API and return assistant text content."""

        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": self._normalize_messages(messages),
            "options": {
                "temperature": decode.temperature,
                "top_p": decode.top_p,
                "num_predict": decode.num_predict,
            },
        }
        if seed is not None:
            payload["options"]["seed"] = int(seed)

        url = f"{self._base_url}/api/chat"
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._session.post(url, json=payload, timeout=self._timeout_seconds)
                response.raise_for_status()
                data = response.json()
                message = data.get("message", {})
                content = message.get("content")
                if not isinstance(content, str):
                    raise OllamaClientError("Ollama response missing message.content string")
                return content.strip()
            except (requests.RequestException, ValueError, OllamaClientError) as exc:
                last_error = exc
                if attempt == self._max_retries:
                    break
                if self._retry_backoff_seconds > 0:
                    time.sleep(self._retry_backoff_seconds * (2**attempt))

        raise OllamaClientError(f"Ollama chat failed after retries for model={model}: {last_error}") from last_error
