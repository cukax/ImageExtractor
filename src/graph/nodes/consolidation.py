"""Fan-in node: consolidates every worker result into a dossier verdict.

By the time this node runs, LangGraph has already merged the parallel worker
outputs into ``processed_documents`` through the state's ``operator.add`` reducer.
All that is left is to decide what the dossier as a whole is worth.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ...domain.orchestrator_state import (
    DossierOrchestratorState,
    DossierStatus,
    resolve_dossier_status,
)

LOGGER = logging.getLogger(__name__)

CONSOLIDATION_NODE = "consolidation"


def build_consolidation_node() -> Callable[[DossierOrchestratorState], dict[str, Any]]:
    """Build the consolidation node."""

    def consolidation_node(state: DossierOrchestratorState) -> dict[str, Any]:
        processed_documents = list(state.get("processed_documents", []))
        had_documents = bool(state.get("grouped_documents"))

        if state.get("dossier_status") == DossierStatus.FAILED and not processed_documents:
            # Clustering already failed the dossier; do not overwrite the verdict
            # with a softer one derived from an empty document list.
            return {"dossier_status": DossierStatus.FAILED.value}

        status = resolve_dossier_status(processed_documents, had_documents=had_documents)
        LOGGER.info(
            "Dossier %s consolidated: %d document(s), status %s",
            state.get("dossier_id"),
            len(processed_documents),
            status.value,
        )
        return {"dossier_status": status.value}

    return consolidation_node
