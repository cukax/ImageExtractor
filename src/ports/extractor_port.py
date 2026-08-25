"""Port describing any structured document extraction service."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..domain.models import ExtractionResult


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

    def supports(self, doc_type: str) -> bool:
        """Report whether the adapter has a dedicated model for a document type.

        The default implementation accepts everything; adapters backed by
        prebuilt models override it to advertise their real coverage.
        """
        return True

    def close(self) -> None:
        """Release any client held by the adapter. Optional for most SDKs."""
        return None
