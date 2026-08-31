"""Ollama vision adapter for locally hosted models such as Qwen2.5-VL."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from ....ports.vlm_port import VLMProviderError
from .base import BaseVLMAdapter, encode_base64

LOGGER = logging.getLogger(__name__)


class OllamaVLMAdapter(BaseVLMAdapter):
    """Implements VLMProviderPort against a local Ollama server.

    This adapter is the on-premise escape hatch: identity documents often cannot
    leave the corporate network, and running Qwen2.5-VL locally keeps the exact
    same graph running without a single line of domain code changing.
    """

    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        *,
        host: str = "http://localhost:11434",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_seconds: float = 180.0,
        keep_alive: str = "5m",
        classifier_model: Optional[str] = None,
    ) -> None:
        super().__init__(
            model=model,
            provider_name="ollama",
            max_tokens=max_tokens,
            temperature=temperature,
            # Local inference on CPU or a modest GPU is far slower than a hosted
            # API, so the default timeout is deliberately generous.
            timeout_seconds=timeout_seconds,
            classifier_model=classifier_model,
        )
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazily instantiate and cache the HTTP client."""
        if self._client is None:
            try:
                import httpx
            except ImportError as error:  # pragma: no cover - environment dependent
                raise VLMProviderError(
                    self.provider_name,
                    "The httpx package is not installed. Run: pip install httpx",
                ) from error
            self._client = httpx.Client(base_url=self.host, timeout=self.timeout_seconds)
        return self._client

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[bytes],
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Send one multimodal chat request to the Ollama /api/chat endpoint."""
        payload = {
            "model": model or self.model,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    # Ollama expects bare base64 strings, without a data URI prefix.
                    "images": [encode_base64(image_bytes) for image_bytes in images],
                },
            ],
        }

        try:
            response = self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
        except Exception as error:  # noqa: BLE001 - HTTP exceptions are wrapped by design
            raise VLMProviderError(self.provider_name, str(error)) from error

        content = (body.get("message") or {}).get("content")
        if not content:
            raise VLMProviderError(
                self.provider_name,
                f"The Ollama response contained no message content: {body!r}",
            )
        return str(content)

    def close(self) -> None:
        """Close the underlying HTTP client if one was created."""
        if self._client is not None:
            self._client.close()
        self._client = None
