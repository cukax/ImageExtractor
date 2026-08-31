"""Worker subgraph: everything that happens to one logical document.

One instance of this graph runs per document in the dossier, in parallel with its
siblings. It is a pure assembly of ports, exactly like ``workflow.py``: no
business logic and no provider SDK import, which is what lets the same subgraph
run against Azure, AWS, Google or a local vision model.

The graph is strictly headless. Where the single document workflow escalates to a
human through ``interrupt()``, this one terminates in the unresolved handler and
reports ``REQUIRES_MANUAL_ENTRY``, so a dossier run never blocks.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from ..domain.policy import WorkflowPolicy
from ..domain.worker_state import DocumentWorkerState, WorkerStatus
from ..ports.extractor_port import DocumentExtractorPort
from ..ports.vlm_port import VLMProviderPort
from .nodes import (
    DOC_INTEL_NODE,
    ROUTER_NODE,
    UNRESOLVED_HANDLER_NODE,
    VLM_GROUNDING_NODE,
    VLM_RETRY_NODE,
    WORKER_VALIDATION_NODE,
    build_doc_intel_node,
    build_route_after_worker_validation,
    build_router_node,
    build_unresolved_handler_node,
    build_vlm_grounding_node,
    build_vlm_retry_node,
    build_worker_validation_node,
    route_after_router,
)

LOGGER = logging.getLogger(__name__)


def build_worker_graph(
    doc_intel_extractor: DocumentExtractorPort,
    grounding_extractor: DocumentExtractorPort,
    vlm_provider: VLMProviderPort,
    policy: Optional[WorkflowPolicy] = None,
) -> StateGraph:
    """Build the uncompiled worker subgraph with every port already injected.

    Args:
        doc_intel_extractor: Serves the template based branch (identity documents).
        grounding_extractor: Serves the visual grounding branch. Passing two
            distinct ports is what lets one dossier mix a prebuilt OCR model with
            a vision deployment; passing the same instance twice is valid too.
        vlm_provider: Performs the targeted ROI re-reads.
        policy: Retry budget, routing table and preprocessing knobs.
    """
    policy = policy or WorkflowPolicy()
    graph = StateGraph(DocumentWorkerState)

    graph.add_node(ROUTER_NODE, build_router_node(policy))
    graph.add_node(DOC_INTEL_NODE, build_doc_intel_node(doc_intel_extractor))
    graph.add_node(VLM_GROUNDING_NODE, build_vlm_grounding_node(grounding_extractor))
    graph.add_node(WORKER_VALIDATION_NODE, build_worker_validation_node(policy))
    graph.add_node(VLM_RETRY_NODE, build_vlm_retry_node(vlm_provider, policy))
    graph.add_node(UNRESOLVED_HANDLER_NODE, build_unresolved_handler_node())

    graph.add_edge(START, ROUTER_NODE)
    graph.add_conditional_edges(
        ROUTER_NODE,
        route_after_router,
        {DOC_INTEL_NODE: DOC_INTEL_NODE, VLM_GROUNDING_NODE: VLM_GROUNDING_NODE},
    )

    # Both branches converge on the same validation node: whichever engine read
    # the document, the same deterministic tools police the result.
    graph.add_conditional_edges(
        DOC_INTEL_NODE,
        _route_after_extraction,
        {WORKER_VALIDATION_NODE: WORKER_VALIDATION_NODE, END: END},
    )
    graph.add_conditional_edges(
        VLM_GROUNDING_NODE,
        _route_after_extraction,
        {WORKER_VALIDATION_NODE: WORKER_VALIDATION_NODE, END: END},
    )

    # The agentic loop: validate, repair the failing fields, validate again.
    graph.add_conditional_edges(
        WORKER_VALIDATION_NODE,
        build_route_after_worker_validation(policy),
        {
            VLM_RETRY_NODE: VLM_RETRY_NODE,
            UNRESOLVED_HANDLER_NODE: UNRESOLVED_HANDLER_NODE,
            END: END,
        },
    )
    graph.add_edge(VLM_RETRY_NODE, WORKER_VALIDATION_NODE)
    graph.add_edge(UNRESOLVED_HANDLER_NODE, END)

    return graph


def _route_after_extraction(state: DocumentWorkerState) -> str:
    """Skip validation when the extraction branch failed outright."""
    if state.get("status") == WorkerStatus.FAILED:
        return END
    return WORKER_VALIDATION_NODE


def compile_worker_graph(
    doc_intel_extractor: DocumentExtractorPort,
    grounding_extractor: DocumentExtractorPort,
    vlm_provider: VLMProviderPort,
    policy: Optional[WorkflowPolicy] = None,
) -> Any:
    """Compile the worker subgraph.

    Deliberately compiled without a checkpointer. A worker is a pure function of
    its input payload and never suspends, so durability belongs one level up, on
    the orchestrator that owns the dossier.
    """
    compiled = build_worker_graph(
        doc_intel_extractor=doc_intel_extractor,
        grounding_extractor=grounding_extractor,
        vlm_provider=vlm_provider,
        policy=policy,
    ).compile()
    LOGGER.info(
        "Worker subgraph compiled with doc_intel=%s grounding=%s vlm=%s",
        doc_intel_extractor.provider_name,
        grounding_extractor.provider_name,
        vlm_provider.provider_name,
    )
    return compiled
