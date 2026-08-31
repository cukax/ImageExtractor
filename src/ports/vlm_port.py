"""Port describing any Vision Language Model provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

#: Sentinel returned by the ROI retry when the crop is genuinely illegible.
#: Nodes rely on this exact token to stop retrying a hopeless field.
UNREADABLE_TOKEN = "UNREADABLE"

#: Label the page classifier falls back to when it recognizes nothing.
UNKNOWN_PAGE_TYPE = "UNKNOWN"


class VLMProviderError(RuntimeError):
    """Raised when a VLM provider call fails in a non recoverable way."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.message = message


class VLMProviderPort(ABC):
    """Driven port for vision capable large language models.

    Four capabilities are exposed, matching the prompt templates the system
    relies on:

    * :meth:`analyze_roi_crop` performs the high precision single field re-read
      used by the retry nodes.
    * :meth:`analyze_full_document` performs the lightweight whole document
      extraction used by the native VLM extractor adapter.
    * :meth:`analyze_document_pages` extends the previous one to a multi page
      logical document, read in a single pass.
    * :meth:`classify_page` assigns a page type token during dossier clustering.

    Only the first two are abstract. The last two ship with a default so that an
    adapter written against an earlier version of this port keeps working.
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
        request_bounding_boxes: bool = False,
    ) -> str:
        """Extract every requested field of a document in a single call.

        Args:
            image_bytes: Preprocessed full document image.
            doc_type: Business document type, injected into the prompt.
            json_schema: JSON Schema the model must fill in.
            layout_hints: Optional orientation or layout guidance.
            request_bounding_boxes: Whether to also ask the model to ground each
                field. It is a per-call argument rather than provider state
                because one process legitimately runs both a grounded and an
                ungrounded extractor over a single shared provider.

        Returns:
            The raw model response, expected to contain a JSON object.

        Raises:
            VLMProviderError: When the provider call fails.
        """

    def analyze_document_pages(
        self,
        images_bytes: Sequence[bytes],
        doc_type: str,
        json_schema: dict[str, Any],
        layout_hints: Optional[str] = None,
        request_bounding_boxes: bool = False,
    ) -> str:
        """Extract every field of a multi page document in a single call.

        A logical document such as an INE is split across a front and a back
        page, and a field may only be legible on one of them. Sending both in one
        request is what lets the model reconcile them; sending them separately
        would produce two half filled documents nothing can merge reliably.

        The default implementation reads the first page only, which keeps any
        adapter that predates this method functional, just less capable.

        Args:
            images_bytes: One preprocessed image per page, in document order.
            doc_type: Business document type, injected into the prompt.
            json_schema: JSON Schema the model must fill in.
            layout_hints: Optional orientation or layout guidance.
            request_bounding_boxes: Whether to also ask the model to ground each
                field.

        Returns:
            The raw model response, expected to contain a JSON object.

        Raises:
            VLMProviderError: When the provider call fails, or when no page was
                supplied at all.
        """
        if not images_bytes:
            raise VLMProviderError(self.provider_name, "No page was supplied for analysis.")
        return self.analyze_full_document(
            image_bytes=images_bytes[0],
            doc_type=doc_type,
            json_schema=json_schema,
            layout_hints=layout_hints,
            request_bounding_boxes=request_bounding_boxes,
        )

    def classify_page(self, image_bytes: bytes, allowed_types: Sequence[str]) -> str:
        """Assign a page type token to one page of a dossier.

        Args:
            image_bytes: Low resolution copy of the page. Classification is a
                layout question, so the full resolution raster is not needed.
            allowed_types: The closed vocabulary the model must choose from.

        Returns:
            One of ``allowed_types``, or ``UNKNOWN_PAGE_TYPE`` when nothing fits.

        Raises:
            VLMProviderError: When the provider does not support classification,
                or when the call fails.
        """
        raise VLMProviderError(
            self.provider_name,
            "This provider does not implement page classification.",
        )

    def close(self) -> None:
        """Release any client held by the adapter. Optional for most SDKs."""
        return None
