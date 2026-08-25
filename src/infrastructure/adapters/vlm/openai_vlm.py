"""OpenAI vision adapter (GPT-4o family)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ....ports.vlm_port import VLMProviderError
from .base import BaseVLMAdapter, build_data_uri

LOGGER = logging.getLogger(__name__)


class OpenAIVLMAdapter(BaseVLMAdapter):
    """Implements VLMProviderPort on top of the OpenAI Chat Completions API.

    The SDK is imported lazily so that a deployment which only uses Anthropic or
    a local Ollama model never needs the openai package installed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        *,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        image_detail: str = "high",
    ) -> None:
        super().__init__(
            model=model,
            provider_name="openai",
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        if not api_key:
            raise VLMProviderError("openai", "OPENAI_API_KEY is required.")
        self._api_key = api_key
        self._base_url = base_url
        self._organization = organization
        # "high" forces the tiled high resolution path, which is what makes a
        # magnified ROI crop worth sending in the first place.
        self.image_detail = image_detail
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazily instantiate and cache the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as error:  # pragma: no cover - environment dependent
                raise VLMProviderError(
                    self.provider_name,
                    "The openai package is not installed. Run: pip install openai",
                ) from error
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                organization=self._organization,
                timeout=self.timeout_seconds,
            )
        return self._client

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send one multimodal chat completion request."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": build_data_uri(image_bytes),
                                    "detail": self.image_detail,
                                },
                            },
                        ],
                    },
                ],
            )
        except Exception as error:  # noqa: BLE001 - SDK exceptions are wrapped by design
            raise VLMProviderError(self.provider_name, str(error)) from error

        choices = getattr(response, "choices", None)
        if not choices:
            raise VLMProviderError(self.provider_name, "The response contained no choices.")
        content = choices[0].message.content
        if content is None:
            raise VLMProviderError(self.provider_name, "The response message had no content.")
        return str(content)

    def close(self) -> None:
        """Close the underlying HTTP client if one was created."""
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
        self._client = None
