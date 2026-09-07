"""Global state and output contract of the multi-document dossier orchestrator.

This module owns the Map-Reduce contract: the orchestrator fans a dossier out into
one worker per logical document, and every worker fans back in by appending a
single :class:`ProcessedDocumentResult` to ``processed_documents``.

It also owns :class:`BoundingBox2D`, the geometry shape of the *published* JSON
contract. It deliberately differs from :class:`~src.domain.models.base.BoundingBox`,
which is the internal working representation: the published contract names its
corner ``x_min`` / ``y_min`` and carries no page index, so consumers of the report
never have to know how many rasters the engine juggled internally.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Any, Optional, Sequence, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from .models import BBoxTuple


class PageType(StrEnum):
    """Label the page classifier assigns to a single rasterized page.

    ``INE_COMBINED`` covers the single-page case: a photocopy that already shows
    both the front and the back of the same card side by side or stacked, which
    is common in KYC scans. It is a page type in its own right rather than a
    pairing outcome, and it sits alongside :attr:`INE_FRONT` / :attr:`INE_BACK`,
    which remain for a card split across two separate pages.
    """

    INE_COMBINED = "INE_COMBINED"
    INE_FRONT = "INE_FRONT"
    INE_BACK = "INE_BACK"
    PASSPORT = "PASSPORT"
    MEXICAN_CEDULA = "MEXICAN_CEDULA"
    PROOF_OF_ADDRESS = "PROOF_OF_ADDRESS"
    INVOICE = "INVOICE"
    UNKNOWN = "UNKNOWN"


class LogicalDocType(StrEnum):
    """Type of a logical document, after the semantic clustering step.

    ``INE_COMBINED`` is reached two ways: the classifier assigns it directly to a
    single page that already shows both sides, or the clustering step pairs a
    front page with its back page found elsewhere in the dossier.
    """

    INE_COMBINED = "INE_COMBINED"
    INE_FRONT = "INE_FRONT"
    INE_BACK = "INE_BACK"
    PASSPORT = "PASSPORT"
    MEXICAN_CEDULA = "MEXICAN_CEDULA"
    PROOF_OF_ADDRESS = "PROOF_OF_ADDRESS"
    INVOICE = "INVOICE"
    UNKNOWN = "UNKNOWN"


class DocumentResultStatus(StrEnum):
    """Terminal status of a single document inside the dossier."""

    SUCCESS = "SUCCESS"
    REQUIRES_MANUAL_ENTRY = "REQUIRES_MANUAL_ENTRY"
    FAILED = "FAILED"


class DossierStatus(StrEnum):
    """Terminal status of the dossier as a whole."""

    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    FAILED = "FAILED"


class BoundingBox2D(BaseModel):
    """Normalized region of a page, in the shape of the published JSON contract."""

    model_config = ConfigDict(frozen=True)

    x_min: float = Field(description="Top-left X coordinate normalized [0.0, 1.0]")
    y_min: float = Field(description="Top-left Y coordinate normalized [0.0, 1.0]")
    width: float = Field(description="Bounding box width normalized [0.0, 1.0]")
    height: float = Field(description="Bounding box height normalized [0.0, 1.0]")

    @classmethod
    def from_bbox_tuple(cls, values: Optional[Sequence[float]]) -> Optional["BoundingBox2D"]:
        """Convert the internal (x, y, w, h) tuple into the published shape.

        Returns None for a missing or malformed tuple rather than raising: a
        report must still be produced for a field the engine could not locate.
        """
        if not values:
            return None
        try:
            x_min, y_min, width, height = (float(value) for value in tuple(values)[:4])
        except (TypeError, ValueError):
            return None
        return cls(x_min=x_min, y_min=y_min, width=width, height=height)


class UnresolvedFieldDetail(BaseModel):
    """A field that never passed validation, with everything an operator needs.

    The bounding box is the payload that matters: it lets a review UI jump
    straight to the region of the page instead of making a human hunt for it.
    """

    field_name: str
    error_reason: str
    bounding_box: Optional[BoundingBox2D] = Field(
        default=None,
        description="Normalized coordinates of the field, or None when it was never located.",
    )


class ProcessedDocumentResult(BaseModel):
    """Outcome of one logical document, appended to the orchestrator state."""

    doc_id: str
    doc_type: str
    status: str = Field(description="SUCCESS | REQUIRES_MANUAL_ENTRY | FAILED")
    extracted_data: dict[str, Any] = Field(default_factory=dict)
    successful_fields: list[str] = Field(default_factory=list)
    unresolved_fields: list[UnresolvedFieldDetail] = Field(default_factory=list)
    page_indexes: list[int] = Field(
        default_factory=list,
        description="Source pages this document was assembled from, kept for traceability.",
    )
    extraction_strategy: Optional[str] = Field(
        default=None,
        description="Which branch of the worker router produced the values.",
    )


class DossierOrchestratorState(TypedDict, total=False):
    """Mutable state carried across the orchestrator graph.

    ``processed_documents`` is the reduce channel: workers run in parallel and
    each appends exactly one result, which is why it carries an ``operator.add``
    reducer instead of being overwritten.
    """

    dossier_id: str
    """Caller supplied identifier, used as the prefix of every ``doc_id``."""

    pdf_bytes: bytes
    """Source file: a PDF, or a single image that is treated as a one page dossier."""

    raw_pages: list[dict[str, Any]]
    """One entry per rasterized page: index, size and the classified ``page_type``."""

    grouped_documents: list[dict[str, Any]]
    """Logical documents produced by the semantic clustering step."""

    processed_documents: Annotated[list[ProcessedDocumentResult], operator.add]
    """Reduce channel, appended to by every parallel worker."""

    dossier_status: str
    """One of the :class:`DossierStatus` values."""

    errors: list[str]
    """Non recoverable infrastructure errors collected during the run."""


class DossierReport(BaseModel):
    """The master output contract: the whole dossier, without any image binary."""

    dossier_id: str
    dossier_status: str
    total_documents_processed: int
    documents: list[ProcessedDocumentResult] = Field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the JSON safe projection published to callers."""
        return self.model_dump(mode="json")


def initial_dossier_state(
    pdf_bytes: bytes,
    dossier_id: str,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> DossierOrchestratorState:
    """Build a fully initialized orchestrator state.

    Explicit initialization matters for the same reason it does in the single
    document workflow: LangGraph merges partial updates, and a missing key forces
    every node into defensive ``.get()`` calls with silent fallbacks.
    """
    state: DossierOrchestratorState = {
        "dossier_id": dossier_id,
        "pdf_bytes": pdf_bytes,
        "raw_pages": [],
        "grouped_documents": [],
        "processed_documents": [],
        "dossier_status": DossierStatus.PROCESSING.value,
        "errors": [],
    }
    if extra:
        state.update(extra)  # type: ignore[typeddict-item]
    return state


def resolve_dossier_status(
    processed_documents: Sequence[ProcessedDocumentResult],
    *,
    had_documents: bool = True,
) -> DossierStatus:
    """Derive the dossier status from the outcome of its documents.

    A single document needing manual entry downgrades the whole dossier to
    ``REQUIRES_REVIEW``: a dossier is only usable when every piece of it is.
    """
    if not had_documents:
        return DossierStatus.FAILED
    if not processed_documents:
        return DossierStatus.FAILED
    if all(
        document.status == DocumentResultStatus.SUCCESS for document in processed_documents
    ):
        return DossierStatus.COMPLETED
    if all(document.status == DocumentResultStatus.FAILED for document in processed_documents):
        return DossierStatus.FAILED
    return DossierStatus.REQUIRES_REVIEW


def build_dossier_report(state: DossierOrchestratorState) -> DossierReport:
    """Project the final orchestrator state onto the published JSON contract.

    Image payloads are dropped here rather than in the caller, so no consumer of
    the report can accidentally serialize a few megabytes of PNG.
    """
    documents = list(state.get("processed_documents", []))
    return DossierReport(
        dossier_id=state.get("dossier_id", ""),
        dossier_status=state.get("dossier_status", DossierStatus.PROCESSING.value),
        total_documents_processed=len(documents),
        documents=documents,
    )


def build_document_id(dossier_id: str, doc_type: str, ordinal: int) -> str:
    """Compose the stable identifier of a document inside a dossier.

    The ordinal, rather than the page number, keeps the id stable when a logical
    document spans several non contiguous pages.
    """
    return f"{dossier_id}_{doc_type}_{ordinal}"


__all__ = [
    "BoundingBox2D",
    "BBoxTuple",
    "DocumentResultStatus",
    "DossierOrchestratorState",
    "DossierReport",
    "DossierStatus",
    "LogicalDocType",
    "PageType",
    "ProcessedDocumentResult",
    "UnresolvedFieldDetail",
    "build_document_id",
    "build_dossier_report",
    "initial_dossier_state",
    "resolve_dossier_status",
]
