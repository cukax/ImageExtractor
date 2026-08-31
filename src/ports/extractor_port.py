"""Port describing any structured document extraction service."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..domain.models import ExtractedField, ExtractionResult

LOGGER = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    """Raised when an extraction provider fails in a non recoverable way.

    Adapters wrap their SDK specific exceptions in this type so the graph never
    has to import boto3, azure-core or google-api-core to handle an error.
    """

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.message = message


class DocumentExtractorPort(ABC):
    """Driven port for OCR / document understanding services.

    Every implementation must return the same provider agnostic
    :class:`ExtractionResult`, including normalized bounding boxes whenever the
    service exposes geometry. The workflow depends on this abstraction only,
    which is what makes Azure, AWS, Google and a local VLM interchangeable.
    """

    #: Stable identifier written into ExtractionResult.provider and into logs.
    provider_name: str = "unknown"

    @abstractmethod
    def extract_document(self, image_bytes: bytes, doc_type: str) -> ExtractionResult:
        """Extract the structured fields of a document.

        Args:
            image_bytes: Preprocessed image payload, JPEG or PNG encoded.
            doc_type: Business document type, for example "INE" or "Invoice".

        Returns:
            The provider agnostic extraction result.

        Raises:
            ExtractionError: When the provider call fails or returns nothing
                the adapter can map onto the domain contract.
        """

    def extract_pages(
        self,
        images_bytes: Sequence[bytes],
        doc_type: str,
    ) -> ExtractionResult:
        """Extract one logical document spread across several page images.

        The default implementation analyzes each page independently and merges
        the results: the first non empty value for a field wins, and every
        bounding box is tagged with the page it was found on so a later ROI crop
        knows which raster to open.

        First-wins is the right merge rule for an identity document: the front
        carries the authoritative name and photo, and the back repeats a subset
        of it, often more faintly.

        Args:
            images_bytes: One preprocessed image per page, in document order.
            doc_type: Business document type.

        Returns:
            A single merged extraction result.

        Raises:
            ExtractionError: When no page was supplied, or when the provider
                fails on every page.
        """
        if not images_bytes:
            raise ExtractionError(self.provider_name, "No page was supplied for extraction.")
        if len(images_bytes) == 1:
            return self.extract_document(images_bytes[0], doc_type)

        merged_fields: dict[str, ExtractedField] = {}
        merged_extras: dict[str, Any] = {}
        warnings: list[str] = []
        failures: list[str] = []

        for page_index, image_bytes in enumerate(images_bytes):
            try:
                result = self.extract_document(image_bytes, doc_type)
            except ExtractionError as error:
                # One unreadable page must not sink a two page document: the
                # other page may well carry every field that matters.
                LOGGER.warning("Page %d failed extraction: %s", page_index, error)
                failures.append(f"page {page_index}: {error.message}")
                continue

            warnings.extend(result.warnings)
            for field_name, field in result.fields.items():
                located = field
                if field.bounding_box is not None:
                    located = field.model_copy(
                        update={"bounding_box": field.bounding_box.model_copy(
                            update={"page": page_index}
                        )}
                    )
                existing = merged_fields.get(field_name)
                if existing is None or existing.value in (None, ""):
                    merged_fields[field_name] = located
            for extra_name, extra_value in result.structured_extras.items():
                merged_extras.setdefault(extra_name, extra_value)

        if not merged_fields and failures:
            raise ExtractionError(
                self.provider_name,
                f"Every page failed extraction: {'; '.join(failures)}",
            )
        warnings.extend(failures)

        return ExtractionResult(
            doc_type=doc_type,
            provider=self.provider_name,
            fields=merged_fields,
            structured_extras=merged_extras,
            page_count=len(images_bytes),
            warnings=warnings,
        )

    def supports(self, doc_type: str) -> bool:
        """Report whether the adapter has a dedicated model for a document type.

        The default implementation accepts everything; adapters backed by
        prebuilt models override it to advertise their real coverage.
        """
        return True

    def close(self) -> None:
        """Release any client held by the adapter. Optional for most SDKs."""
        return None
