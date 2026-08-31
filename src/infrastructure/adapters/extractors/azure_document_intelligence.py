"""Azure AI Document Intelligence adapter."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from ....domain.models import BoundingBox, ExtractedField, ExtractionResult, normalize_key
from ....ports.extractor_port import DocumentExtractorPort, ExtractionError

LOGGER = logging.getLogger(__name__)

#: Token audience for Azure AI services, used when authenticating with Entra ID.
DEFAULT_CREDENTIAL_SCOPE = "https://cognitiveservices.azure.com/.default"

#: Document type to prebuilt model mapping. Custom trained models are plugged in
#: through configuration without touching this adapter.
DEFAULT_MODEL_BY_DOC_TYPE: dict[str, str] = {
    "ine": "prebuilt-idDocument",
    "ife": "prebuilt-idDocument",
    "id": "prebuilt-idDocument",
    "idcard": "prebuilt-idDocument",
    "iddocument": "prebuilt-idDocument",
    "passport": "prebuilt-idDocument",
    "driverlicense": "prebuilt-idDocument",
    "invoice": "prebuilt-invoice",
    "factura": "prebuilt-invoice",
    "receipt": "prebuilt-receipt",
    "form": "prebuilt-document",
}

DEFAULT_FALLBACK_MODEL = "prebuilt-document"


def _read(source: Any, key: str, default: Any = None) -> Any:
    """Read a key from an Azure model that may behave as an object or a mapping.

    The Document Intelligence SDK has changed the shape of DocumentField between
    minor versions; going through both access paths keeps the adapter stable
    across upgrades.
    """
    if source is None:
        return default
    value = getattr(source, key, None)
    if value is not None:
        return value
    if isinstance(source, dict):
        return source.get(key, default)
    try:
        return source[key]
    except (TypeError, KeyError, IndexError):
        return default


def _normalize_polygon(
    polygon: Sequence[float],
    page_width: float,
    page_height: float,
    page: int = 0,
) -> BoundingBox:
    """Translate an Azure 8-point polygon into a normalized bounding box.

    Azure reports geometry as a flat ``[x1, y1, x2, y2, x3, y3, x4, y4]`` list of
    the four corners, expressed in the same unit as the page dimensions (inches
    for a PDF, pixels for an image). The domain works exclusively in normalized
    [0.0, 1.0] coordinates so a box survives the resizing done during
    preprocessing, which is what lets one box crop either the optimized image or
    the full resolution original.

    Coordinates outside the page are clamped rather than rejected: a rotated scan
    routinely produces a corner a hair beyond the edge, and a clamped box still
    crops correctly while a rejected one would lose the retry entirely.

    Raises:
        ValueError: When the polygon is malformed or the page has no area.
    """
    return BoundingBox.from_polygon(
        polygon=[float(coordinate) for coordinate in polygon],
        page_width=page_width,
        page_height=page_height,
        page=page,
    )


class AzureDocumentIntelligenceAdapter(DocumentExtractorPort):
    """Implements DocumentExtractorPort on top of Azure AI Document Intelligence."""

    provider_name = "azure_document_intelligence"

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        *,
        model_by_doc_type: Optional[dict[str, str]] = None,
        fallback_model: str = DEFAULT_FALLBACK_MODEL,
        credential_scope: str = DEFAULT_CREDENTIAL_SCOPE,
    ) -> None:
        """Configure the adapter.

        Args:
            endpoint: Resource endpoint, always required.
            api_key: Optional. Leave it empty to authenticate secretlessly with
                Microsoft Entra ID through ``DefaultAzureCredential``, which
                resolves a Managed Identity on a deployed host and the
                developer's ``az login`` session locally.
            model_by_doc_type: Overrides of the prebuilt model mapping.
            fallback_model: Model used for an unregistered document type.
            credential_scope: Token audience used for Entra ID authentication.
        """
        if not endpoint:
            raise ExtractionError(
                self.provider_name,
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is required.",
            )
        self._endpoint = endpoint
        self._api_key = api_key or None
        self._model_by_doc_type = {**DEFAULT_MODEL_BY_DOC_TYPE, **(model_by_doc_type or {})}
        self._fallback_model = fallback_model
        self._credential_scope = credential_scope
        self._client: Any = None
        self._credential: Any = None

    @property
    def uses_managed_identity(self) -> bool:
        """True when the adapter authenticates through Entra ID rather than a key."""
        return not self._api_key

    # -- Client management -------------------------------------------------- #
    @property
    def client(self) -> Any:
        """Lazily instantiate and cache the Document Intelligence client."""
        if self._client is None:
            try:
                from azure.ai.documentintelligence import DocumentIntelligenceClient
            except ImportError as error:  # pragma: no cover - environment dependent
                raise ExtractionError(
                    self.provider_name,
                    "The azure-ai-documentintelligence package is not installed. "
                    "Run: pip install azure-ai-documentintelligence",
                ) from error
            self._client = DocumentIntelligenceClient(
                endpoint=self._endpoint,
                credential=self._build_credential(),
            )
        return self._client

    def _build_credential(self) -> Any:
        """Resolve the credential: an explicit key, or Entra ID by default."""
        if self._api_key:
            from azure.core.credentials import AzureKeyCredential

            LOGGER.warning(
                "Authenticating to Document Intelligence with an API key; prefer a "
                "Managed Identity by leaving AZURE_DOCUMENT_INTELLIGENCE_KEY empty."
            )
            return AzureKeyCredential(self._api_key)

        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ExtractionError(
                self.provider_name,
                "No API key was configured and the azure-identity package is not "
                "installed. Run: pip install azure-identity",
            ) from error

        LOGGER.info(
            "Document Intelligence authentication resolved through "
            "DefaultAzureCredential (scope %s)",
            self._credential_scope,
        )
        self._credential = DefaultAzureCredential()
        return self._credential

    def resolve_model_id(self, doc_type: str) -> str:
        """Resolve the Azure model id bound to a business document type."""
        return self._model_by_doc_type.get(normalize_key(doc_type), self._fallback_model)

    def supports(self, doc_type: str) -> bool:
        """True when a dedicated prebuilt model is registered for the type."""
        return normalize_key(doc_type) in self._model_by_doc_type

    # -- Port implementation ------------------------------------------------ #
    def extract_document(self, image_bytes: bytes, doc_type: str) -> ExtractionResult:
        """Analyze a document and map the Azure payload onto the domain contract."""
        model_id = self.resolve_model_id(doc_type)
        try:
            from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

            poller = self.client.begin_analyze_document(
                model_id,
                AnalyzeDocumentRequest(bytes_source=image_bytes),
            )
            analyze_result = poller.result()
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ExtractionError(
                self.provider_name,
                "The azure-ai-documentintelligence package is not installed.",
            ) from error
        except Exception as error:  # noqa: BLE001 - SDK exceptions are wrapped by design
            raise ExtractionError(self.provider_name, str(error)) from error

        page_sizes = self._collect_page_sizes(analyze_result)
        fields: dict[str, ExtractedField] = {}
        warnings: list[str] = []

        documents = _read(analyze_result, "documents") or []
        for document in documents:
            for field_name, field_payload in (_read(document, "fields") or {}).items():
                extracted = self._map_field(field_name, field_payload, page_sizes, warnings)
                if extracted is not None:
                    fields.setdefault(field_name, extracted)

        if not documents:
            warnings.append(
                f"Model {model_id} returned no structured document; only raw layout is available."
            )

        return ExtractionResult(
            doc_type=doc_type,
            provider=self.provider_name,
            fields=fields,
            page_count=len(page_sizes) or 1,
            warnings=warnings,
            raw_payload={"model_id": model_id},
        )

    # -- Mapping helpers ---------------------------------------------------- #
    def _collect_page_sizes(self, analyze_result: Any) -> dict[int, tuple[float, float]]:
        """Index the width and height of every page by its 1 based page number."""
        page_sizes: dict[int, tuple[float, float]] = {}
        for index, page in enumerate(_read(analyze_result, "pages") or [], start=1):
            width = _read(page, "width")
            height = _read(page, "height")
            page_number = _read(page, "page_number") or _read(page, "pageNumber") or index
            if width and height:
                page_sizes[int(page_number)] = (float(width), float(height))
        return page_sizes

    def _map_field(
        self,
        field_name: str,
        field_payload: Any,
        page_sizes: dict[int, tuple[float, float]],
        warnings: list[str],
    ) -> Optional[ExtractedField]:
        """Convert a single Azure DocumentField into an ExtractedField."""
        if field_payload is None:
            return None

        # "content" holds the literal text as printed, which is what the regex
        # validators and the ROI retry compare against. The typed value would
        # silently reformat dates and amounts and defeat that comparison.
        value = _read(field_payload, "content")
        if value is None:
            value = _read(field_payload, "value_string") or _read(field_payload, "valueString")
        if value is None:
            value = _read(field_payload, "value")

        bounding_box: Optional[BoundingBox] = None
        regions = _read(field_payload, "bounding_regions") or _read(field_payload, "boundingRegions")
        if regions:
            region = regions[0]
            polygon = _read(region, "polygon")
            page_number = int(_read(region, "page_number") or _read(region, "pageNumber") or 1)
            page_size = page_sizes.get(page_number)
            if polygon and page_size:
                try:
                    bounding_box = _normalize_polygon(
                        polygon=polygon,
                        page_width=page_size[0],
                        page_height=page_size[1],
                        page=page_number - 1,
                    )
                except (ValueError, TypeError) as error:
                    warnings.append(f"Could not map the polygon of field {field_name}: {error}")

        confidence = _read(field_payload, "confidence")
        return ExtractedField(
            name=field_name,
            value=None if value is None else str(value),
            confidence=float(confidence) if confidence is not None else None,
            bounding_box=bounding_box,
        )

    def close(self) -> None:
        """Close the underlying Azure client and any credential it borrowed."""
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
        self._client = None
        if self._credential is not None and hasattr(self._credential, "close"):
            self._credential.close()
        self._credential = None
