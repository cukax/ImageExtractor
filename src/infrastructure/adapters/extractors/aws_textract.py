"""AWS Textract adapter."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ....domain.models import BoundingBox, ExtractedField, ExtractionResult, normalize_key
from ....ports.extractor_port import DocumentExtractorPort, ExtractionError

LOGGER = logging.getLogger(__name__)

#: Textract exposes three different APIs; the document type selects which one to
#: call, because AnalyzeID, AnalyzeExpense and AnalyzeDocument return completely
#: different payload shapes.
_IDENTITY_DOC_TYPES = frozenset({"ine", "ife", "id", "idcard", "iddocument", "passport", "driverlicense"})
_EXPENSE_DOC_TYPES = frozenset({"invoice", "factura", "receipt"})


def _to_bounding_box(geometry: Optional[dict[str, Any]]) -> Optional[BoundingBox]:
    """Convert a Textract Geometry block into a normalized BoundingBox.

    Textract already reports normalized coordinates, so this is a straight
    rename rather than a projection.
    """
    if not geometry:
        return None
    box = geometry.get("BoundingBox") or {}
    if not box:
        return None
    try:
        return BoundingBox(
            x=max(0.0, min(1.0, float(box.get("Left", 0.0)))),
            y=max(0.0, min(1.0, float(box.get("Top", 0.0)))),
            width=max(0.0, min(1.0, float(box.get("Width", 0.0)))),
            height=max(0.0, min(1.0, float(box.get("Height", 0.0)))),
        )
    except (TypeError, ValueError):
        return None


def _confidence(raw_confidence: Any) -> Optional[float]:
    """Convert a Textract percentage confidence into the 0.0 - 1.0 range."""
    if raw_confidence is None:
        return None
    try:
        return max(0.0, min(1.0, float(raw_confidence) / 100.0))
    except (TypeError, ValueError):
        return None


class AWSTextractAdapter(DocumentExtractorPort):
    """Implements DocumentExtractorPort on top of Amazon Textract."""

    provider_name = "aws_textract"

    def __init__(
        self,
        region_name: str,
        *,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> None:
        if not region_name:
            raise ExtractionError(self.provider_name, "AWS_REGION is required.")
        self._region_name = region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        self._endpoint_url = endpoint_url
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazily instantiate and cache the boto3 Textract client.

        When no explicit key is configured, boto3 falls back to its standard
        credential chain (IAM role, profile, environment), which is what a
        production deployment on ECS or EKS should rely on.
        """
        if self._client is None:
            try:
                import boto3
            except ImportError as error:  # pragma: no cover - environment dependent
                raise ExtractionError(
                    self.provider_name,
                    "The boto3 package is not installed. Run: pip install boto3",
                ) from error
            self._client = boto3.client(
                "textract",
                region_name=self._region_name,
                aws_access_key_id=self._aws_access_key_id,
                aws_secret_access_key=self._aws_secret_access_key,
                aws_session_token=self._aws_session_token,
                endpoint_url=self._endpoint_url,
            )
        return self._client

    def supports(self, doc_type: str) -> bool:
        """True for identity documents and expense documents."""
        token = normalize_key(doc_type)
        return token in _IDENTITY_DOC_TYPES or token in _EXPENSE_DOC_TYPES

    # -- Port implementation ------------------------------------------------ #
    def extract_document(self, image_bytes: bytes, doc_type: str) -> ExtractionResult:
        """Route to the Textract API that matches the document type."""
        token = normalize_key(doc_type)
        try:
            if token in _IDENTITY_DOC_TYPES:
                return self._extract_identity(image_bytes, doc_type)
            if token in _EXPENSE_DOC_TYPES:
                return self._extract_expense(image_bytes, doc_type)
            return self._extract_forms(image_bytes, doc_type)
        except ExtractionError:
            raise
        except Exception as error:  # noqa: BLE001 - SDK exceptions are wrapped by design
            raise ExtractionError(self.provider_name, str(error)) from error

    # -- AnalyzeID ---------------------------------------------------------- #
    def _extract_identity(self, image_bytes: bytes, doc_type: str) -> ExtractionResult:
        """Call AnalyzeID and map IdentityDocumentFields onto the domain contract."""
        response = self.client.analyze_id(DocumentPages=[{"Bytes": image_bytes}])
        fields: dict[str, ExtractedField] = {}
        warnings: list[str] = []

        identity_documents = response.get("IdentityDocuments") or []
        for document in identity_documents:
            for entry in document.get("IdentityDocumentFields") or []:
                name = (entry.get("Type") or {}).get("Text")
                if not name:
                    continue
                detection = entry.get("ValueDetection") or {}
                fields.setdefault(
                    name,
                    ExtractedField(
                        name=name,
                        value=detection.get("Text") or None,
                        confidence=_confidence(detection.get("Confidence")),
                        bounding_box=_to_bounding_box(detection.get("Geometry")),
                    ),
                )

        if not identity_documents:
            warnings.append("AnalyzeID did not recognize any identity document in the image.")

        return ExtractionResult(
            doc_type=doc_type,
            provider=self.provider_name,
            fields=fields,
            page_count=max(1, len(identity_documents)),
            warnings=warnings,
            raw_payload={"api": "analyze_id"},
        )

    # -- AnalyzeExpense ----------------------------------------------------- #
    def _extract_expense(self, image_bytes: bytes, doc_type: str) -> ExtractionResult:
        """Call AnalyzeExpense and map summary fields plus line items."""
        response = self.client.analyze_expense(Document={"Bytes": image_bytes})
        fields: dict[str, ExtractedField] = {}
        warnings: list[str] = []
        line_items: list[dict[str, Any]] = []

        expense_documents = response.get("ExpenseDocuments") or []
        for document in expense_documents:
            for entry in document.get("SummaryFields") or []:
                name = (entry.get("Type") or {}).get("Text")
                if not name or name == "OTHER":
                    continue
                detection = entry.get("ValueDetection") or {}
                fields.setdefault(
                    name,
                    ExtractedField(
                        name=name,
                        value=detection.get("Text") or None,
                        confidence=_confidence(detection.get("Confidence")),
                        bounding_box=_to_bounding_box(detection.get("Geometry")),
                    ),
                )

            for group in document.get("LineItemGroups") or []:
                for item in group.get("LineItems") or []:
                    line_items.append(self._map_line_item(item))

        if not expense_documents:
            warnings.append("AnalyzeExpense did not recognize any expense document in the image.")

        return ExtractionResult(
            doc_type=doc_type,
            provider=self.provider_name,
            fields=fields,
            # Line items are nested records, so they travel through the
            # structured channel rather than as a flat ExtractedField.
            structured_extras={"line_items": line_items} if line_items else {},
            page_count=max(1, len(expense_documents)),
            warnings=warnings,
            raw_payload={"api": "analyze_expense"},
        )

    def _map_line_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Flatten one Textract LineItem into a plain description / amount record."""
        mapping = {
            "ITEM": "description",
            "QUANTITY": "quantity",
            "UNIT_PRICE": "unit_price",
            "PRICE": "amount",
        }
        record: dict[str, Any] = {}
        for entry in item.get("LineItemExpenseFields") or []:
            name = (entry.get("Type") or {}).get("Text")
            target = mapping.get(name or "")
            if target is None:
                continue
            record[target] = (entry.get("ValueDetection") or {}).get("Text")
        return record

    # -- AnalyzeDocument (FORMS) -------------------------------------------- #
    def _extract_forms(self, image_bytes: bytes, doc_type: str) -> ExtractionResult:
        """Call AnalyzeDocument with the FORMS feature and rebuild key / value pairs.

        Textract returns a flat block graph, so keys, values and the words they
        contain have to be re-assembled through the Relationships edges.
        """
        response = self.client.analyze_document(
            Document={"Bytes": image_bytes},
            FeatureTypes=["FORMS"],
        )
        blocks_by_id = {block["Id"]: block for block in response.get("Blocks") or []}
        fields: dict[str, ExtractedField] = {}

        for block in blocks_by_id.values():
            if block.get("BlockType") != "KEY_VALUE_SET":
                continue
            if "KEY" not in (block.get("EntityTypes") or []):
                continue

            key_text = self._collect_text(block, blocks_by_id)
            value_block = self._find_value_block(block, blocks_by_id)
            if not key_text or value_block is None:
                continue

            fields.setdefault(
                key_text,
                ExtractedField(
                    name=key_text,
                    value=self._collect_text(value_block, blocks_by_id) or None,
                    confidence=_confidence(value_block.get("Confidence")),
                    bounding_box=_to_bounding_box(value_block.get("Geometry")),
                ),
            )

        page_count = sum(
            1 for block in blocks_by_id.values() if block.get("BlockType") == "PAGE"
        )
        return ExtractionResult(
            doc_type=doc_type,
            provider=self.provider_name,
            fields=fields,
            page_count=max(1, page_count),
            warnings=[] if fields else ["AnalyzeDocument found no key / value pair."],
            raw_payload={"api": "analyze_document"},
        )

    def _find_value_block(
        self,
        key_block: dict[str, Any],
        blocks_by_id: dict[str, dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Follow the VALUE relationship of a KEY block."""
        for relationship in key_block.get("Relationships") or []:
            if relationship.get("Type") != "VALUE":
                continue
            for block_id in relationship.get("Ids") or []:
                value_block = blocks_by_id.get(block_id)
                if value_block is not None:
                    return value_block
        return None

    def _collect_text(
        self,
        block: dict[str, Any],
        blocks_by_id: dict[str, dict[str, Any]],
    ) -> str:
        """Concatenate the WORD and SELECTION_ELEMENT children of a block."""
        words: list[str] = []
        for relationship in block.get("Relationships") or []:
            if relationship.get("Type") != "CHILD":
                continue
            for block_id in relationship.get("Ids") or []:
                child = blocks_by_id.get(block_id) or {}
                if child.get("BlockType") == "WORD":
                    words.append(child.get("Text", ""))
                elif child.get("BlockType") == "SELECTION_ELEMENT":
                    if child.get("SelectionStatus") == "SELECTED":
                        words.append("SELECTED")
        return " ".join(word for word in words if word).strip()

    def close(self) -> None:
        """Release the boto3 client reference."""
        self._client = None
