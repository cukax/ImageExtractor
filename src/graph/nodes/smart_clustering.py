"""Phase 1: splitter and semantic clustering.

Three steps, in order:

1. Rasterize every page of the dossier at 300 DPI.
2. Classify each page independently, from a low resolution copy.
3. Group the pages into *logical* documents, pairing a front with its back even
   when they are not adjacent in the file.

Step 3 is the reason this node exists at all. A naive splitter assumes a document
ends where the next one begins, which is wrong for the single most common dossier
shape in practice: an identity card photocopied front on page 1 and back on page 3,
with a utility bill in between.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

from ...domain.models import RenderedPage
from ...domain.orchestrator_state import (
    DossierOrchestratorState,
    DossierStatus,
    LogicalDocType,
    PageType,
    build_document_id,
)
from ...domain.policy import WorkflowPolicy
from ...domain.preprocessing import downscale_for_classification
from ...ports.pdf_port import PDFProcessingError, PDFProcessorPort
from ...ports.vlm_port import VLMProviderError, VLMProviderPort

LOGGER = logging.getLogger(__name__)

SMART_CLUSTERING_NODE = "smart_clustering"

#: Closed vocabulary handed to the classifier. Keeping it closed is what makes
#: the answer routable: an open ended label would need a second mapping step.
CLASSIFIABLE_PAGE_TYPES: tuple[str, ...] = tuple(page_type.value for page_type in PageType)

#: Page types that pair into a single two sided logical document.
_PAIRED_PAGE_TYPES: dict[str, str] = {
    PageType.INE_FRONT.value: PageType.INE_BACK.value,
}


def _normalize_page_type(raw_label: Optional[str]) -> str:
    """Coerce a model answer into a member of the closed page type vocabulary."""
    if not raw_label:
        return PageType.UNKNOWN.value
    candidate = str(raw_label).strip().upper().replace(" ", "_").replace("-", "_")
    for page_type in CLASSIFIABLE_PAGE_TYPES:
        if candidate == page_type:
            return page_type
    # A chatty model sometimes answers "This page is an INE_FRONT."; recovering
    # the token from the sentence is cheaper than paying for a second call.
    for page_type in CLASSIFIABLE_PAGE_TYPES:
        if page_type in candidate:
            return page_type
    LOGGER.warning("Unrecognized page type %r; falling back to UNKNOWN", raw_label)
    return PageType.UNKNOWN.value


def _classify_pages(
    pages: Sequence[RenderedPage],
    vlm_provider: VLMProviderPort,
    policy: WorkflowPolicy,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Assign a page type to every rasterized page.

    A classification failure is never fatal: the page falls back to ``UNKNOWN``,
    which the router sends down the visual grounding path, so a dossier is still
    processed end to end when the classifier is unavailable.
    """
    classified: list[dict[str, Any]] = []
    for page in pages:
        try:
            thumbnail = downscale_for_classification(
                page.image_bytes,
                policy.classification_max_dimension,
            )
            raw_label = vlm_provider.classify_page(thumbnail, CLASSIFIABLE_PAGE_TYPES)
            page_type = _normalize_page_type(raw_label)
        except (VLMProviderError, ValueError) as error:
            LOGGER.error("Page %d could not be classified: %s", page.page_index, error)
            warnings.append(f"Page {page.page_index} could not be classified: {error}")
            page_type = PageType.UNKNOWN.value

        LOGGER.info("Page %d classified as %s", page.page_index, page_type)
        classified.append(
            {
                "page_index": page.page_index,
                "page_type": page_type,
                "width": page.width,
                "height": page.height,
            }
        )
    return classified


def _group_pages_into_logical_docs(
    classified_pages: Sequence[dict[str, Any]],
    dossier_id: str,
) -> list[dict[str, Any]]:
    """Group classified pages into logical documents.

    A front page claims the *first still unclaimed* matching back page anywhere
    later in the dossier, which is what handles the interleaved layout. Every
    other page becomes a single page document of its own type.
    """
    remaining = list(classified_pages)
    claimed: set[int] = set()
    grouped: list[dict[str, Any]] = []
    ordinal_by_type: dict[str, int] = {}

    def _next_ordinal(doc_type: str) -> int:
        ordinal_by_type[doc_type] = ordinal_by_type.get(doc_type, 0) + 1
        return ordinal_by_type[doc_type]

    for position, page in enumerate(remaining):
        if page["page_index"] in claimed:
            continue

        page_type = page["page_type"]
        back_type = _PAIRED_PAGE_TYPES.get(page_type)
        page_indexes = [page["page_index"]]
        doc_type = page_type

        if back_type is not None:
            partner = next(
                (
                    candidate
                    for candidate in remaining[position + 1 :]
                    if candidate["page_type"] == back_type
                    and candidate["page_index"] not in claimed
                ),
                None,
            )
            if partner is not None:
                claimed.add(partner["page_index"])
                page_indexes.append(partner["page_index"])
                doc_type = LogicalDocType.INE_COMBINED.value
                LOGGER.info(
                    "Paired page %d (%s) with page %d (%s) into %s",
                    page["page_index"],
                    page_type,
                    partner["page_index"],
                    back_type,
                    doc_type,
                )

        claimed.add(page["page_index"])
        grouped.append(
            {
                "doc_id": build_document_id(dossier_id, doc_type, _next_ordinal(doc_type)),
                "doc_type": doc_type,
                "page_indexes": page_indexes,
            }
        )

    return grouped


def build_smart_clustering_node(
    pdf_processor: PDFProcessorPort,
    vlm_provider: VLMProviderPort,
    policy: WorkflowPolicy,
) -> Callable[[DossierOrchestratorState], dict[str, Any]]:
    """Build the splitter and clustering node around the injected ports.

    Args:
        pdf_processor: Rasterizes the source file.
        vlm_provider: Classifies each page.
        policy: Supplies the render DPI and the classification resolution.
    """

    def smart_clustering_node(state: DossierOrchestratorState) -> dict[str, Any]:
        dossier_id = state.get("dossier_id", "")
        document_bytes = state.get("pdf_bytes") or b""
        warnings: list[str] = []

        if not document_bytes:
            return {
                "raw_pages": [],
                "grouped_documents": [],
                "dossier_status": DossierStatus.FAILED.value,
                "errors": [*state.get("errors", []), "The dossier payload is empty."],
            }

        try:
            pages = pdf_processor.render_pages(document_bytes, dpi=policy.pdf_render_dpi)
        except PDFProcessingError as error:
            LOGGER.error("Rasterization failed: %s", error)
            return {
                "raw_pages": [],
                "grouped_documents": [],
                "dossier_status": DossierStatus.FAILED.value,
                "errors": [*state.get("errors", []), str(error)],
            }

        if not pages:
            return {
                "raw_pages": [],
                "grouped_documents": [],
                "dossier_status": DossierStatus.FAILED.value,
                "errors": [*state.get("errors", []), "The dossier contains no page."],
            }

        classified_pages = _classify_pages(pages, vlm_provider, policy, warnings)
        grouped_documents = _group_pages_into_logical_docs(classified_pages, dossier_id)

        # The rasters travel with the grouped documents rather than the page
        # list, so the fan-out payload is self contained and a worker never has
        # to reach back into the orchestrator state.
        pages_by_index = {page.page_index: page for page in pages}
        for document in grouped_documents:
            document["images_bytes"] = [
                pages_by_index[index].image_bytes
                for index in document["page_indexes"]
                if index in pages_by_index
            ]

        LOGGER.info(
            "Dossier %s split into %d page(s) and %d logical document(s)",
            dossier_id,
            len(pages),
            len(grouped_documents),
        )
        return {
            "raw_pages": classified_pages,
            "grouped_documents": grouped_documents,
            "dossier_status": DossierStatus.PROCESSING.value,
            "errors": [*state.get("errors", []), *warnings],
        }

    return smart_clustering_node
