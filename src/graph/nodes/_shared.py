"""Helpers shared by the single document workflow and the dossier worker.

Both pipelines repair a failing field the same way: they compose the error
context handed to the vision model, and they push the repaired raw text back
through the Pydantic schema. Keeping one implementation here is what stops the
two graphs from drifting apart.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from pydantic import ValidationError

from ...domain.models import (
    BoundingBox,
    DocumentSchema,
    ExtractionResult,
    get_schema_for,
    normalize_key,
)
from ...domain.validation_tools import expected_format_for
from ...domain.worker_state import DocumentWorkerState, WorkerStatus
from ...ports.extractor_port import DocumentExtractorPort, ExtractionError

LOGGER = logging.getLogger(__name__)


def build_error_context(
    doc_type: str,
    field_name: str,
    validation_errors: dict[str, str],
) -> str:
    """Compose the error context handed to the VLM.

    The documented expected format is appended to the raw validation message:
    telling the model what a correct value looks like measurably reduces the
    number of second failures.
    """
    message = validation_errors.get(field_name, "The extracted value failed validation.")
    expected_format = expected_format_for(doc_type, field_name)
    if expected_format and expected_format not in message:
        return f"{message} The expected format is {expected_format}."
    return message


def renormalize(doc_type: str, extracted_data: dict[str, Any]) -> dict[str, Any]:
    """Re-run the schema validators over the patched data.

    A value returned by the VLM is raw text, so pushing it back through the
    Pydantic schema restores the invariants the extraction node established
    (uppercase codes, parsed amounts). A failure here is not fatal: the raw
    values simply reach the validation node unchanged.
    """
    schema_cls = get_schema_for(doc_type)
    try:
        return schema_cls.from_flat_dict(extracted_data).to_flat_dict()
    except ValidationError as error:
        LOGGER.warning("Could not renormalize the repaired data: %s", error)
        return extracted_data


def page_by_field(
    result: ExtractionResult,
    schema_cls: type[DocumentSchema],
) -> dict[str, int]:
    """Index which page each located field was found on.

    The bounding box tuple carried in the worker state is deliberately flat
    (x, y, w, h), so the page index travels beside it. Without it the ROI retry
    would crop the front of an identity card looking for a field printed on the
    back.
    """
    lookup = schema_cls.alias_lookup()
    pages: dict[str, int] = {}
    for provider_name, field in result.fields.items():
        box: Optional[BoundingBox] = field.bounding_box
        if box is None or box.is_degenerate:
            continue
        # A schema-less GenericFormSchema has no alias table, so the provider
        # name is already the canonical key.
        canonical_field = lookup.get(normalize_key(provider_name), provider_name)
        pages.setdefault(canonical_field, box.page)
    return pages


def build_worker_extraction_node(
    extractor: DocumentExtractorPort,
    branch_label: str,
) -> Callable[[DocumentWorkerState], dict[str, Any]]:
    """Build a worker extraction node around any DocumentExtractorPort.

    The template based branch and the visual grounding branch are the same
    algorithm over two different adapters: call the port across every page, map
    the payload onto the typed schema, keep the geometry. Sharing one
    implementation is what guarantees the two branches produce state a single
    validation node can consume without branching on how it was produced.

    Args:
        extractor: The port implementing this branch.
        branch_label: Node name, used in logs and in the provider metadata so a
            reviewer can tell which engine produced a disputed value.
    """

    def extraction_node(state: DocumentWorkerState) -> dict[str, Any]:
        doc_type = state.get("doc_type") or ""
        images_bytes = state.get("images_bytes") or []

        try:
            result = extractor.extract_pages(images_bytes, doc_type)
        except ExtractionError as error:
            LOGGER.error("[%s] Extraction failed: %s", branch_label, error)
            return {
                "status": WorkerStatus.FAILED.value,
                "errors": [*state.get("errors", []), str(error)],
            }

        schema_cls = get_schema_for(doc_type)
        try:
            document, bounding_boxes, unmapped = schema_cls.from_extraction(result)
        except ValidationError as error:
            LOGGER.error(
                "[%s] The provider payload does not fit %s: %s",
                branch_label,
                schema_cls.__name__,
                error,
            )
            return {
                "status": WorkerStatus.FAILED.value,
                "errors": [
                    *state.get("errors", []),
                    f"Could not map the {result.provider} payload onto "
                    f"{schema_cls.__name__}: {error}",
                ],
            }

        if unmapped:
            LOGGER.info(
                "[%s] Provider fields ignored by %s: %s",
                branch_label,
                schema_cls.__name__,
                unmapped,
            )

        located = sorted(bounding_boxes)
        if not located:
            LOGGER.info(
                "[%s] No field of %s was located; the ROI retry will re-read whole pages",
                branch_label,
                state.get("doc_id"),
            )

        return {
            "extracted_data": document.to_flat_dict(),
            "bounding_boxes": bounding_boxes,
            "bounding_box_pages": page_by_field(result, schema_cls),
            "status": WorkerStatus.PROCESSING.value,
            "provider_metadata": {
                "branch": branch_label,
                "provider": result.provider,
                "schema": schema_cls.__name__,
                "page_count": result.page_count,
                "warnings": result.warnings,
                "unmapped_fields": unmapped,
                "located_fields": located,
            },
        }

    return extraction_node
