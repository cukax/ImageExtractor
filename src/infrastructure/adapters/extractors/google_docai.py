"""Google Cloud Document AI adapter."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ....domain.models import BoundingBox, ExtractedField, ExtractionResult, normalize_key
from ....ports.extractor_port import DocumentExtractorPort, ExtractionError

LOGGER = logging.getLogger(__name__)


class GoogleDocAIAdapter(DocumentExtractorPort):
    """Implements DocumentExtractorPort on top of Google Cloud Document AI.

    Document AI binds one processor to one document family, so the adapter takes
    a document type to processor id map. Everything else, including the region
    specific API endpoint, is derived from configuration.
    """

    provider_name = "google_document_ai"

    def __init__(
        self,
        project_id: str,
        location: str,
        processor_by_doc_type: dict[str, str],
        *,
        default_processor_id: Optional[str] = None,
        processor_version: Optional[str] = None,
        credentials_path: Optional[str] = None,
        mime_type: str = "image/jpeg",
    ) -> None:
        if not project_id or not location:
            raise ExtractionError(
                self.provider_name,
                "GOOGLE_PROJECT_ID and GOOGLE_LOCATION are both required.",
            )
        if not processor_by_doc_type and not default_processor_id:
            raise ExtractionError(
                self.provider_name,
                "At least one Document AI processor id must be configured.",
            )
        self._project_id = project_id
        self._location = location
        self._processor_by_doc_type = {
            normalize_key(key): value for key, value in (processor_by_doc_type or {}).items()
        }
        self._default_processor_id = default_processor_id
        self._processor_version = processor_version
        self._credentials_path = credentials_path
        self._mime_type = mime_type
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazily instantiate and cache the Document AI client."""
        if self._client is None:
            try:
                from google.api_core.client_options import ClientOptions
                from google.cloud import documentai
            except ImportError as error:  # pragma: no cover - environment dependent
                raise ExtractionError(
                    self.provider_name,
                    "The google-cloud-documentai package is not installed. "
                    "Run: pip install google-cloud-documentai",
                ) from error

            client_options = ClientOptions(
                api_endpoint=f"{self._location}-documentai.googleapis.com"
            )
            if self._credentials_path:
                self._client = documentai.DocumentProcessorServiceClient.from_service_account_file(
                    self._credentials_path,
                    client_options=client_options,
                )
            else:
                # Falls back to Application Default Credentials, which is the
                # right behaviour on GKE, Cloud Run and Compute Engine.
                self._client = documentai.DocumentProcessorServiceClient(
                    client_options=client_options
                )
        return self._client

    def resolve_processor_id(self, doc_type: str) -> str:
        """Resolve the processor bound to a document type."""
        processor_id = self._processor_by_doc_type.get(
            normalize_key(doc_type), self._default_processor_id
        )
        if not processor_id:
            raise ExtractionError(
                self.provider_name,
                f"No Document AI processor is configured for document type {doc_type!r}.",
            )
        return processor_id

    def supports(self, doc_type: str) -> bool:
        """True when a processor is registered for the document type."""
        return (
            normalize_key(doc_type) in self._processor_by_doc_type
            or self._default_processor_id is not None
        )

    def _resource_name(self, processor_id: str) -> str:
        """Build the fully qualified processor (or processor version) resource name."""
        if self._processor_version:
            return self.client.processor_version_path(
                self._project_id, self._location, processor_id, self._processor_version
            )
        return self.client.processor_path(self._project_id, self._location, processor_id)

    # -- Port implementation ------------------------------------------------ #
    def extract_document(self, image_bytes: bytes, doc_type: str) -> ExtractionResult:
        """Process a document and map the Document AI entities onto the contract."""
        processor_id = self.resolve_processor_id(doc_type)
        try:
            from google.cloud import documentai

            request = documentai.ProcessRequest(
                name=self._resource_name(processor_id),
                raw_document=documentai.RawDocument(
                    content=image_bytes,
                    mime_type=self._mime_type,
                ),
            )
            response = self.client.process_document(request=request)
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ExtractionError(
                self.provider_name, "The google-cloud-documentai package is not installed."
            ) from error
        except Exception as error:  # noqa: BLE001 - SDK exceptions are wrapped by design
            raise ExtractionError(self.provider_name, str(error)) from error

        document = response.document
        fields: dict[str, ExtractedField] = {}
        warnings: list[str] = []

        for entity in document.entities or []:
            extracted = self._map_entity(entity, warnings)
            if extracted is not None:
                fields.setdefault(extracted.name, extracted)
            # Nested properties carry the sub fields of composite entities such
            # as an address block or a line item.
            for child in getattr(entity, "properties", []) or []:
                child_field = self._map_entity(child, warnings)
                if child_field is not None:
                    fields.setdefault(child_field.name, child_field)

        if not fields:
            warnings.append(f"Processor {processor_id} returned no entity for this document.")

        return ExtractionResult(
            doc_type=doc_type,
            provider=self.provider_name,
            fields=fields,
            page_count=len(document.pages or []) or 1,
            warnings=warnings,
            raw_payload={"processor_id": processor_id},
        )

    def _map_entity(self, entity: Any, warnings: list[str]) -> Optional[ExtractedField]:
        """Convert one Document AI entity into an ExtractedField."""
        name = getattr(entity, "type_", None) or getattr(entity, "type", None)
        if not name:
            return None

        # mention_text is the literal text on the page; normalized_value would
        # reformat dates and money and break the regex validators.
        value = getattr(entity, "mention_text", None) or None
        confidence = getattr(entity, "confidence", None)

        bounding_box: Optional[BoundingBox] = None
        page_anchor = getattr(entity, "page_anchor", None)
        page_refs = getattr(page_anchor, "page_refs", None) if page_anchor else None
        if page_refs:
            page_ref = page_refs[0]
            vertices = getattr(getattr(page_ref, "bounding_poly", None), "normalized_vertices", None)
            if vertices:
                try:
                    bounding_box = BoundingBox.from_normalized_vertices(
                        vertices=vertices,
                        page=int(getattr(page_ref, "page", 0) or 0),
                    )
                except (ValueError, TypeError) as error:
                    warnings.append(f"Could not map the bounding poly of entity {name}: {error}")

        return ExtractedField(
            name=str(name),
            value=str(value) if value is not None else None,
            confidence=float(confidence) if confidence is not None else None,
            bounding_box=bounding_box,
        )

    def close(self) -> None:
        """Release the Document AI client reference."""
        self._client = None
