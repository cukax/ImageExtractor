"""Port describing any Vision Language Model provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

#: Sentinel returned by the ROI retry when the crop is genuinely illegible.
#: Nodes rely on this exact token to stop retrying a hopeless field.
UNREADABLE_TOKEN = "UNREADABLE"


class VLMProviderError(RuntimeError):
    """Raised when a VLM provider call fails in a non recoverable way."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.message = message


class VLMProviderPort(ABC):
    """Driven port for vision capable large language models.

    Two capabilities are exposed, matching the two prompt templates the system
    relies on:

    * :meth:`analyze_roi_crop` performs the high precision single field re-read
      used by the retry node.
    * :meth:`analyze_full_document` performs the lightweight whole document
      extraction used by the native VLM extractor adapter.
    """

    #: Stable identifier used in logs and in provider metadata.
    provider_name: str = "unknown"

    @abstractmethod
    def analyze_roi_crop(self, crop_bytes: bytes, field_name: str, error_context: str) -> str:
        """Re-read a single field from a magnified crop of the document.

        Args:
            crop_bytes: PNG or JPEG bytes of the padded region of interest.
            field_name: Canonical name of the field being repaired.
            error_context: The validation error that triggered the retry, used
                to tell the model what "correct" looks like.

        Returns:
            The raw value as a plain string, or ``UNREADABLE`` when the crop
            cannot be deciphered.

        Raises:
            VLMProviderError: When the provider call fails.
        """

    @abstractmethod
    def analyze_full_document(
        self,
        image_bytes: bytes,
        doc_type: str,
        json_schema: dict[str, Any],
        layout_hints: Optional[str] = None,
    ) -> str:
        """Extract every requested field of a document in a single call.

        Args:
            image_bytes: Preprocessed full document image.
            doc_type: Business document type, injected into the prompt.
            json_schema: JSON Schema the model must fill in.
            layout_hints: Optional orientation or layout guidance.

        Returns:
            The raw model response, expected to contain a JSON object.

        Raises:
            VLMProviderError: When the provider call fails.
        """

    def close(self) -> None:
        """Release any client held by the adapter. Optional for most SDKs."""
        return None
