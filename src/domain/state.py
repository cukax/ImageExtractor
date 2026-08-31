"""LangGraph state contract shared by every node of the OCR workflow."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Optional, TypedDict

from .models import BBoxTuple


class DocumentStatus(StrEnum):
    """Lifecycle of a document inside the workflow.

    Declared as a StrEnum so the values compare equal to the plain strings used
    in the state dictionary and stay readable in any checkpointer backend.
    """

    PROCESSING = "PROCESSING"
    VALIDATED = "VALIDATED"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    REJECTED_BLUR = "REJECTED_BLUR"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"


class OCRState(TypedDict, total=False):
    """Mutable state carried across the OCR graph.

    total=False lets each node return only the keys it actually owns, which is
    the idiomatic LangGraph partial update pattern.
    """

    # --- Input --------------------------------------------------------------
    image_bytes: bytes
    """Working image: EXIF corrected, enhanced, resized and JPEG compressed."""

    doc_type: str
    """Document type token, for example 'INE', 'Invoice' or 'Form'."""

    # --- Extraction output --------------------------------------------------
    extracted_data: dict[str, Any]
    """Flat {field_name: value} mapping produced by the extraction node."""

    bounding_boxes: dict[str, BBoxTuple]
    """Normalized (x, y, w, h) coordinates per field, used by the ROI retry."""

    # --- Validation output --------------------------------------------------
    validation_errors: dict[str, str]
    """Human readable error message per failing field."""

    failed_fields: list[str]
    """Fields that did not pass validation on the latest validation pass."""

    retry_count: int
    """Number of completed validation failures, capped by settings.max_retries."""

    # --- Preprocessing output ----------------------------------------------
    is_readable: bool
    """False when the fail fast blur filter rejected the image."""

    status: str
    """One of the DocumentStatus values."""

    # --- Operational extensions --------------------------------------------
    # The keys below are not part of the functional contract but are required to
    # run the workflow safely in production. They are grouped separately so the
    # core contract above stays easy to read.
    original_image_bytes: bytes
    """EXIF corrected full resolution original, cropped by the ROI retry node."""

    blur_variance: float
    """Laplacian variance measured during preprocessing, kept for telemetry."""

    flagged_fields: list[str]
    """Fields the VLM declared UNREADABLE, excluded from further retries."""

    preprocessing_report: dict[str, Any]
    """Steps applied by the preprocessing node, kept for auditing."""

    provider_metadata: dict[str, Any]
    """Extractor identity, page count and provider warnings."""

    human_review_payload: dict[str, Any]
    """Payload surfaced to the operator when the graph interrupts."""

    human_review_result: dict[str, Any]
    """Corrections and decision returned by the operator on resume."""

    errors: list[str]
    """Non recoverable infrastructure errors collected during the run."""


def initial_state(
    image_bytes: bytes,
    doc_type: str,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> OCRState:
    """Build a fully initialized state, so no node has to guess a default.

    Explicit initialization matters: LangGraph merges partial updates, and a
    missing key would force every node into defensive .get() calls with silent
    fallbacks that hide bugs.
    """
    state: OCRState = {
        "image_bytes": image_bytes,
        "original_image_bytes": image_bytes,
        "doc_type": doc_type,
        "extracted_data": {},
        "bounding_boxes": {},
        "validation_errors": {},
        "failed_fields": [],
        "flagged_fields": [],
        "retry_count": 0,
        "is_readable": True,
        "blur_variance": 0.0,
        "status": DocumentStatus.PROCESSING.value,
        "preprocessing_report": {},
        "provider_metadata": {},
        "human_review_payload": {},
        "human_review_result": {},
        "errors": [],
    }
    if extra:
        state.update(extra)  # type: ignore[typeddict-item]
    return state
