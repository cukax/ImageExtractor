"""Phase 2: dynamic fan-out.

Turns the list of logical documents produced by the clustering node into one
``Send`` instruction per document. LangGraph then instantiates that many worker
tasks and runs them concurrently, which is the map half of the Map-Reduce.

``Send`` is what makes the fan-out *dynamic*: the number of workers is only known
at runtime, once the dossier has actually been split, so it cannot be expressed
as a static edge.
"""

from __future__ import annotations

import logging
from typing import Any, Union

from langgraph.types import Send

from ..domain.orchestrator_state import DossierOrchestratorState, LogicalDocType
from ..domain.worker_state import initial_worker_state
from .nodes import CONSOLIDATION_NODE

LOGGER = logging.getLogger(__name__)

DOCUMENT_WORKER_NODE = "document_worker"


def dispatch_documents(
    state: DossierOrchestratorState,
) -> Union[str, list[Send]]:
    """Emit one worker instruction per logical document.

    Returns the consolidation node directly when the dossier produced no
    document, so an empty or unreadable upload still reaches a terminal state
    instead of dangling.
    """
    grouped_documents = state.get("grouped_documents") or []
    if not grouped_documents:
        LOGGER.warning(
            "Dossier %s produced no logical document; skipping the fan-out",
            state.get("dossier_id"),
        )
        return CONSOLIDATION_NODE

    LOGGER.info(
        "Fanning dossier %s out to %d parallel worker(s)",
        state.get("dossier_id"),
        len(grouped_documents),
    )
    return [
        Send(DOCUMENT_WORKER_NODE, _build_worker_payload(document))
        for document in grouped_documents
    ]


def _build_worker_payload(document: dict[str, Any]) -> dict[str, Any]:
    """Build the self contained state one worker instance starts from.

    The payload carries its own page rasters rather than an index into the
    orchestrator state: a worker must never reach back into the parent, because
    under a fan-out there is no ordering guarantee about when it runs.
    """
    images_bytes = list(document.get("images_bytes") or [])
    return dict(
        initial_worker_state(
            doc_id=document.get("doc_id", ""),
            doc_type=document.get("doc_type") or LogicalDocType.UNKNOWN.value,
            images_bytes=images_bytes,
            page_indexes=list(document.get("page_indexes") or []),
        )
    )
