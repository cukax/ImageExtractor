"""Extraction adapter backed directly by a Vision Language Model.

This adapter is the cheap, dependency free path: it needs no OCR vendor at all
and reuses whichever VLMProviderPort is configured. It is also the reference
implementation of prompt template A, the primary extraction prompt.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from types import UnionType
from typing import Any, Optional, Union, get_args, get_origin

from ....domain.models import (
    BoundingBox,
    DocumentSchema,
    ExtractedField,
    ExtractionResult,
    GenericFormSchema,
    get_schema_for,
    normalize_key,
)
from ....ports.extractor_port import DocumentExtractorPort, ExtractionError
from ....ports.vlm_port import VLMProviderError, VLMProviderPort
from ..vlm.base import extract_json_object

LOGGER = logging.getLogger(__name__)

#: Key the model is asked to use when grounding is enabled.
BOUNDING_BOX_KEY = "_bounding_boxes"


def _json_type_for(annotation: Any) -> str:
    """Map a Python annotation onto the JSON type advertised in the prompt.

    Scalars are all advertised as strings on purpose: the model must transcribe
    what is printed, and letting it emit a number invites silent reformatting of
    amounts and dates that would defeat the regex validators.
    """
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        inner = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _json_type_for(inner[0]) if inner else "string"
    if origin in (list, tuple, set):
        return "array"
    if annotation in (dict,) or origin is dict:
        return "object"
    return "string"


def build_prompt_schema(schema_cls: type[DocumentSchema]) -> dict[str, Any]:
    """Build a compact JSON Schema for the prompt.

    Pydantic model_json_schema() is technically correct but verbose: it nests
    $defs and anyOf wrappers that waste tokens and confuse smaller local models.
    A flat, description rich schema extracts far more reliably.
    """
    if schema_cls is GenericFormSchema:
        return {
            "type": "object",
            "description": (
                "Return one property per key / value pair printed on the document. "
                "Use the printed label as the property name, in snake_case."
            ),
            "additionalProperties": {"type": ["string", "null"]},
        }

    properties: dict[str, Any] = {}
    for field_name, field_info in schema_cls.model_fields.items():
        json_type = _json_type_for(field_info.annotation)
        entry: dict[str, Any] = {"type": [json_type, "null"]}
        if field_info.description:
            entry["description"] = field_info.description
        if json_type == "array":
            entry["items"] = {
                "type": "object",
                "properties": {
                    "description": {"type": ["string", "null"]},
                    "quantity": {"type": ["string", "null"]},
                    "unit_price": {"type": ["string", "null"]},
                    "amount": {"type": ["string", "null"]},
                },
            }
        properties[field_name] = entry

    return {"type": "object", "properties": properties, "additionalProperties": False}


class NativeVLMExtractorAdapter(DocumentExtractorPort):
    """Implements DocumentExtractorPort by prompting a VLM for the whole document.

    Composition over inheritance: the adapter holds a VLMProviderPort, so the
    exact same class runs on GPT-4o, Claude or a local Qwen2.5-VL depending only
    on configuration.
    """

    provider_name = "native_vlm"

    def __init__(
        self,
        vlm_provider: VLMProviderPort,
        *,
        request_bounding_boxes: bool = False,
        layout_hints_by_doc_type: Optional[dict[str, str]] = None,
    ) -> None:
        self._vlm_provider = vlm_provider
        # Grounding quality varies enormously between models. When it is off,
        # the retry node falls back to re-reading the whole page.
        self._request_bounding_boxes = request_bounding_boxes
        self._layout_hints = {
            normalize_key(key): value for key, value in (layout_hints_by_doc_type or {}).items()
        }
        # The flag is read by BaseVLMAdapter when it renders prompt template A.
        setattr(self._vlm_provider, "request_bounding_boxes", request_bounding_boxes)

    @property
    def vlm_provider(self) -> VLMProviderPort:
        """Expose the injected VLM provider, mainly for tests and diagnostics."""
        return self._vlm_provider

    def layout_hints_for(self, doc_type: str) -> Optional[str]:
        """Return the configured orientation or layout hint for a document type."""
        return self._layout_hints.get(normalize_key(doc_type))

    # -- Port implementation ------------------------------------------------ #
    def extract_document(self, image_bytes: bytes, doc_type: str) -> ExtractionResult:
        """Prompt the VLM for every field of the document in a single call."""
        schema_cls = get_schema_for(doc_type)
        json_schema = build_prompt_schema(schema_cls)

        try:
            raw_response = self._vlm_provider.analyze_full_document(
                image_bytes=image_bytes,
                doc_type=doc_type,
                json_schema=json_schema,
                layout_hints=self.layout_hints_for(doc_type),
            )
            payload = extract_json_object(raw_response, self.provider_name)
        except VLMProviderError as error:
            raise ExtractionError(self.provider_name, error.message) from error
        except Exception as error:  # noqa: BLE001 - defensive, keeps the graph alive
            raise ExtractionError(self.provider_name, str(error)) from error

        return self._build_result(payload, doc_type)

    # -- Mapping helpers ---------------------------------------------------- #
    def _build_result(self, payload: dict[str, Any], doc_type: str) -> ExtractionResult:
        """Map the parsed JSON response onto the domain contract."""
        # Some models wrap the answer in a single container key despite the
        # instructions; unwrapping it costs nothing and saves a full retry.
        if len(payload) == 1:
            only_key, only_value = next(iter(payload.items()))
            if only_key.lower() in {"fields", "data", "result", "document"} and isinstance(
                only_value, dict
            ):
                payload = only_value

        raw_boxes = payload.pop(BOUNDING_BOX_KEY, None) or {}
        fields: dict[str, ExtractedField] = {}
        structured_extras: dict[str, Any] = {}
        warnings: list[str] = []

        for field_name, raw_value in payload.items():
            if isinstance(raw_value, (list, dict)):
                structured_extras[field_name] = raw_value
                continue
            fields[field_name] = ExtractedField(
                name=field_name,
                value=self._to_text(raw_value),
                # A VLM does not report a calibrated confidence, and inventing
                # one would mislead every downstream consumer.
                confidence=None,
                bounding_box=self._to_bounding_box(raw_boxes.get(field_name), field_name, warnings),
            )

        if self._request_bounding_boxes and not raw_boxes:
            warnings.append(
                "Grounding was requested but the model returned no bounding box; "
                "the ROI retry will fall back to the full page."
            )

        return ExtractionResult(
            doc_type=doc_type,
            provider=self.provider_name,
            fields=fields,
            structured_extras=structured_extras,
            page_count=1,
            warnings=warnings,
            raw_payload={"vlm_provider": self._vlm_provider.provider_name},
        )

    def _to_text(self, raw_value: Any) -> Optional[str]:
        """Normalize a JSON scalar into the string contract of ExtractedField."""
        if raw_value is None:
            return None
        if isinstance(raw_value, bool):
            return "true" if raw_value else "false"
        if isinstance(raw_value, (int, float, Decimal)):
            return str(raw_value)
        text = str(raw_value).strip()
        # Models sometimes spell out an empty field instead of emitting null.
        if not text or text.lower() in {"null", "none", "n/a", "na", "unreadable"}:
            return None
        return text

    def _to_bounding_box(
        self,
        raw_box: Any,
        field_name: str,
        warnings: list[str],
    ) -> Optional[BoundingBox]:
        """Convert a model reported [x, y, w, h] list into a BoundingBox."""
        if not raw_box:
            return None
        try:
            if isinstance(raw_box, dict):
                raw_box = [
                    raw_box.get("x", 0.0),
                    raw_box.get("y", 0.0),
                    raw_box.get("width", raw_box.get("w", 0.0)),
                    raw_box.get("height", raw_box.get("h", 0.0)),
                ]
            box = BoundingBox.from_tuple([float(value) for value in raw_box][:4])
        except (TypeError, ValueError, IndexError) as error:
            warnings.append(f"Ignored a malformed bounding box for {field_name}: {error}")
            return None
        return None if box.is_degenerate else box

    def close(self) -> None:
        """Close the injected VLM provider."""
        self._vlm_provider.close()
