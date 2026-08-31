"""State contract of the per-document worker subgraph.

One worker instance handles one *logical* document, which may span several pages:
an INE front and its back, or a two page invoice. That is the single difference
from :mod:`src.domain.state`, whose workflow is scoped to a single image.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Optional, TypedDict

from .models import BBoxTuple
from .orchestrator_state import LogicalDocType, UnresolvedFieldDetail


class ExtractionStrategy(StrEnum):
    """Which extraction branch the worker router selected."""

    DOC_INTELLIGENCE = "DOC_INTELLIGENCE"
    """Template based extraction through a DocumentExtractorPort."""

    VLM_GROUNDING = "VLM_GROUNDING"
    """Single pass visual grounding through a VLMProviderPort."""


class WorkerStatus(StrEnum):
    """Lifecycle of a document inside the worker subgraph."""

    PROCESSING = "PROCESSING"
    VALIDATED = "VALIDATED"
    REQUIRES_MANUAL_ENTRY = "REQUIRES_MANUAL_ENTRY"
    FAILED = "FAILED"


class DocumentWorkerState(TypedDict, total=False):
    """Mutable state carried across the worker subgraph.

    total=False lets each node return only the keys it owns, which is the
    idiomatic LangGraph partial update pattern.
    """

    # --- Input --------------------------------------------------------------
    doc_id: str
    """Stable identifier of this document inside its dossier."""

    images_bytes: list[bytes]
    """Working images, one per page. Supports 1+ pages, for example front + back."""

    doc_type: Optional[str]
    """Logical document type assigned by the clustering step."""

    # --- Routing ------------------------------------------------------------
    extraction_strategy: Optional[str]
    """One of the :class:`ExtractionStrategy` values, set by the router node."""

    # --- Extraction output --------------------------------------------------
    extracted_data: dict[str, Any]
    """Flat {field_name: value} mapping produced by the extraction branch."""

    bounding_boxes: dict[str, Optional[BBoxTuple]]
    """Normalized (x, y, w, h) coordinates per field, used by the ROI retry."""

    # --- Validation output --------------------------------------------------
    failed_fields: list[str]
    """Fields that did not pass validation on the latest validation pass."""

    validation_errors: dict[str, str]
    """Human readable error message per failing field."""

    retry_count: int
    """Number of completed validation failures, capped by the policy retry budget."""

    status: str
    """One of the :class:`WorkerStatus` values."""

    # --- Operational extensions --------------------------------------------
    # The keys below are not part of the functional contract but are required to
    # run the worker safely in production. They are grouped separately so the
    # core contract above stays easy to read.
    original_images_bytes: list[bytes]
    """Full resolution pages, cropped by the ROI retry node."""

    bounding_box_pages: dict[str, int]
    """Which page each bounding box belongs to, so the retry crops the right raster."""

    flagged_fields: list[str]
    """Fields the VLM declared UNREADABLE, excluded from further retries."""

    unresolved_fields: list[UnresolvedFieldDetail]
    """Populated by the headless unresolved handler at the end of the run."""

    page_indexes: list[int]
    """Source page indexes this document was assembled from, kept for traceability."""

    provider_metadata: dict[str, Any]
    """Extractor identity, page count and provider warnings."""

    errors: list[str]
    """Non recoverable infrastructure errors collected during the run."""


def initial_worker_state(
    doc_id: str,
    doc_type: str,
    images_bytes: list[bytes],
    *,
    original_images_bytes: Optional[list[bytes]] = None,
    page_indexes: Optional[list[int]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> DocumentWorkerState:
    """Build a fully initialized worker state, so no node has to guess a default."""
    state: DocumentWorkerState = {
        "doc_id": doc_id,
        "doc_type": doc_type or LogicalDocType.UNKNOWN.value,
        "images_bytes": list(images_bytes),
        "original_images_bytes": list(original_images_bytes or images_bytes),
        "extraction_strategy": None,
        "extracted_data": {},
        "bounding_boxes": {},
        "bounding_box_pages": {},
        "failed_fields": [],
        "flagged_fields": [],
        "validation_errors": {},
        "unresolved_fields": [],
        "page_indexes": list(page_indexes or []),
        "retry_count": 0,
        "status": WorkerStatus.PROCESSING.value,
        "provider_metadata": {},
        "errors": [],
    }
    if extra:
        state.update(extra)  # type: ignore[typeddict-item]
    return state


__all__ = [
    "DocumentWorkerState",
    "ExtractionStrategy",
    "WorkerStatus",
    "initial_worker_state",
]
