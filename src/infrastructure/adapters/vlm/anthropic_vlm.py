"""Anthropic vision adapter (Claude family)."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from ....ports.vlm_port import VLMProviderError
from .base import BaseVLMAdapter, detect_media_type, encode_base64

LOGGER = logging.getLogger(__name__)


class AnthropicVLMAdapter(BaseVLMAdapter):
    """Implements VLMProviderPort on top of the Anthropic Messages API.

    Anthropic takes the system prompt as a top level parameter rather than as a
    message, and expects raw base64 image blocks instead of data URIs, which is
    the only real difference from the OpenAI adapter.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        *,
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        classifier_model: Optional[str] = None,
    ) -> None:
        super().__init__(
            model=model,
            provider_name="anthropic",
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            classifier_model=classifier_model,
        )
        if not api_key:
            raise VLMProviderError("anthropic", "ANTHROPIC_API_KEY is required.")
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazily instantiate and cache the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as error:  # pragma: no cover - environment dependent
                raise VLMProviderError(
                    self.provider_name,
                    "The anthropic package is not installed. Run: pip install anthropic",
                ) from error
            kwargs: dict[str, Any] = {"api_key": self._api_key, "timeout": self.timeout_seconds}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[bytes],
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Send one multimodal Messages API request carrying every page."""
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": detect_media_type(image_bytes),
                    "data": encode_base64(image_bytes),
                },
            }
            for image_bytes in images
        ]
        # The text block goes last: Claude reads images better when the question
        # follows the material it is about.
        content.append({"type": "text", "text": user_prompt})

        try:
            response = self.client.messages.create(
                model=model or self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as error:  # noqa: BLE001 - SDK exceptions are wrapped by design
            raise VLMProviderError(self.provider_name, str(error)) from error

        # The Messages API returns a list of content blocks; only the text ones
        # are relevant for a transcription task.
        text_blocks = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        if not text_blocks:
            raise VLMProviderError(self.provider_name, "The response contained no text block.")
        return "\n".join(text_blocks)

    def close(self) -> None:
        """Close the underlying HTTP client if one was created."""
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
        self._client = None
