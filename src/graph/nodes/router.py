"""Phase 3: worker router.

Decides which extraction branch a logical document takes. The decision table is
data, not code: it lives on :class:`~src.domain.policy.WorkflowPolicy`, so a
deployment can move a document type from one branch to the other through
configuration without touching the graph.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ...domain.policy import WorkflowPolicy
from ...domain.worker_state import DocumentWorkerState, ExtractionStrategy, WorkerStatus

LOGGER = logging.getLogger(__name__)

ROUTER_NODE = "router"
DOC_INTEL_NODE = "doc_intel"
VLM_GROUNDING_NODE = "vlm_grounding"

#: Maps the strategy recorded in the state onto the node that implements it.
NODE_BY_STRATEGY: dict[str, str] = {
    ExtractionStrategy.DOC_INTELLIGENCE.value: DOC_INTEL_NODE,
    ExtractionStrategy.VLM_GROUNDING.value: VLM_GROUNDING_NODE,
}


def build_router_node(policy: WorkflowPolicy) -> Callable[[DocumentWorkerState], dict[str, Any]]:
    """Build the router node, which only records the decision it made.

    Recording the strategy in the state rather than branching inline keeps the
    decision auditable: it ends up in the document result, so a reviewer can see
    which engine produced a disputed value.
    """

    def router_node(state: DocumentWorkerState) -> dict[str, Any]:
        doc_type = state.get("doc_type") or ""
        strategy = policy.strategy_for(doc_type)
        LOGGER.info(
            "Document %s of type %s routed to %s",
            state.get("doc_id"),
            doc_type or "UNKNOWN",
            strategy,
        )
        return {
            "extraction_strategy": strategy,
            "status": WorkerStatus.PROCESSING.value,
        }

    return router_node


def route_after_router(state: DocumentWorkerState) -> str:
    """Send the document to the node implementing the selected strategy."""
    strategy = state.get("extraction_strategy") or ExtractionStrategy.VLM_GROUNDING.value
    return NODE_BY_STRATEGY.get(strategy, VLM_GROUNDING_NODE)
