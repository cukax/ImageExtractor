"""Pydantic data contracts and the document type registry."""

from __future__ import annotations

from .base import (
    BBoxTuple,
    BoundingBox,
    DocumentSchema,
    ExtractedField,
    ExtractionResult,
    normalize_key,
    parse_decimal,
)
from .generic_form import GenericFormSchema
from .id_card import IDCardSchema
from .invoice import InvoiceLineItem, InvoiceSchema

# Canonical document type tokens understood by the workflow.
DOC_TYPE_ID_CARD = "INE"
DOC_TYPE_INVOICE = "Invoice"
DOC_TYPE_FORM = "Form"

# Maps a normalized document type onto the schema that parses it. Adding a new
# document type is a domain-only change: no adapter and no graph node is touched.
SCHEMA_REGISTRY: dict[str, type[DocumentSchema]] = {
    "ine": IDCardSchema,
    "ife": IDCardSchema,
    "id": IDCardSchema,
    "idcard": IDCardSchema,
    "iddocument": IDCardSchema,
    "passport": IDCardSchema,
    "driverlicense": IDCardSchema,
    "invoice": InvoiceSchema,
    "factura": InvoiceSchema,
    "receipt": InvoiceSchema,
    "form": GenericFormSchema,
    "generic": GenericFormSchema,
}


def get_schema_for(doc_type: str) -> type[DocumentSchema]:
    """Resolve the Pydantic schema bound to a document type.

    Unknown document types fall back to GenericFormSchema instead of raising, so
    an unexpected label degrades to best effort extraction rather than an
    operational outage.
    """
    return SCHEMA_REGISTRY.get(normalize_key(doc_type), GenericFormSchema)


def is_known_doc_type(doc_type: str) -> bool:
    """Report whether the document type has a dedicated typed schema."""
    return normalize_key(doc_type) in SCHEMA_REGISTRY


__all__ = [
    "BBoxTuple",
    "BoundingBox",
    "DOC_TYPE_FORM",
    "DOC_TYPE_ID_CARD",
    "DOC_TYPE_INVOICE",
    "DocumentSchema",
    "ExtractedField",
    "ExtractionResult",
    "GenericFormSchema",
    "IDCardSchema",
    "InvoiceLineItem",
    "InvoiceSchema",
    "SCHEMA_REGISTRY",
    "get_schema_for",
    "is_known_doc_type",
    "normalize_key",
    "parse_decimal",
]
