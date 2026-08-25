"""StateGraph construction and compilation.

The workflow is a pure assembly: it receives ports, wires the nodes together and
compiles the graph. It contains no business logic, and no import of any provider
SDK, which is what lets the same graph run against Azure, AWS, Google or a local
VLM without a single edit.

``build_graph`` depends on the domain and the ports only. ``compile_workflow``
is the convenience composition helper, and is the single function here that
reaches into the infrastructure layer to resolve the configured adapters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from langgraph.graph import END, START, StateGraph

from ..domain.policy import WorkflowPolicy
from ..domain.state import OCRState, initial_state
from ..ports.extractor_port import DocumentExtractorPort
from ..ports.vlm_port import VLMProviderPort

if TYPE_CHECKING:  # pragma: no cover - type checking only, no runtime dependency
    from ..infrastructure.config import Dependencies

from .nodes import (
    CROP_AND_VLM_RETRY_NODE,
    EXTRACTION_NODE,
    HITL_FALLBACK_NODE,
    PREPROCESS_NODE,
    VALIDATION_NODE,
    build_crop_and_vlm_retry_node,
    build_extraction_node,
    build_hitl_fallback_node,
    build_preprocess_node,
    build_route_after_validation,
    build_validation_node,
    route_after_extraction,
    route_after_preprocess,
)

LOGGER = logging.getLogger(__name__)


def _default_checkpointer() -> Any:
    """Return an in memory checkpointer, renamed across LangGraph versions.

    A checkpointer is mandatory here: interrupt() has nowhere to persist the
    suspended run without one. Production deployments should pass a durable
    implementation (Postgres, Redis, SQLite) instead.
    """
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except ImportError:  # pragma: no cover - older LangGraph releases
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


def build_graph(
    extractor: DocumentExtractorPort,
    vlm_provider: VLMProviderPort,
    policy: Optional[WorkflowPolicy] = None,
) -> StateGraph:
    """Build the uncompiled StateGraph with every port already injected."""
    policy = policy or WorkflowPolicy()
    graph = StateGraph(OCRState)

    graph.add_node(PREPROCESS_NODE, build_preprocess_node(policy))
    graph.add_node(EXTRACTION_NODE, build_extraction_node(extractor))
    graph.add_node(VALIDATION_NODE, build_validation_node(policy))
    graph.add_node(CROP_AND_VLM_RETRY_NODE, build_crop_and_vlm_retry_node(vlm_provider, policy))
    graph.add_node(HITL_FALLBACK_NODE, build_hitl_fallback_node(policy))

    graph.add_edge(START, PREPROCESS_NODE)

    # Fail fast: an unreadable page ends the run before any paid API call.
    graph.add_conditional_edges(
        PREPROCESS_NODE,
        route_after_preprocess,
        {EXTRACTION_NODE: EXTRACTION_NODE, END: END},
    )
    graph.add_conditional_edges(
        EXTRACTION_NODE,
        route_after_extraction,
        {VALIDATION_NODE: VALIDATION_NODE, END: END},
    )

    # The agentic loop: validate, repair the failing fields, validate again.
    graph.add_conditional_edges(
        VALIDATION_NODE,
        build_route_after_validation(policy),
        {
            CROP_AND_VLM_RETRY_NODE: CROP_AND_VLM_RETRY_NODE,
            HITL_FALLBACK_NODE: HITL_FALLBACK_NODE,
            END: END,
        },
    )
    graph.add_edge(CROP_AND_VLM_RETRY_NODE, VALIDATION_NODE)
    graph.add_edge(HITL_FALLBACK_NODE, END)

    return graph


def compile_workflow(
    dependencies: Optional["Dependencies"] = None,
    *,
    checkpointer: Any = None,
) -> Any:
    """Build and compile the workflow, ready to be invoked.

    Args:
        dependencies: Resolved ports. Defaults to the configured providers.
        checkpointer: LangGraph checkpointer. Defaults to an in memory saver,
            which is enough for a single process but loses suspended runs on
            restart.

    Returns:
        The compiled LangGraph application.
    """
    # Imported here rather than at module scope: this is the composition seam,
    # and keeping it local preserves the inward dependency rule for build_graph.
    from ..infrastructure.config import build_dependencies

    dependencies = dependencies or build_dependencies()
    graph = build_graph(
        extractor=dependencies.extractor,
        vlm_provider=dependencies.vlm_provider,
        policy=dependencies.settings.to_workflow_policy(),
    )
    compiled = graph.compile(checkpointer=checkpointer or _default_checkpointer())
    LOGGER.info(
        "Workflow compiled with extractor=%s vlm=%s max_retries=%d",
        dependencies.extractor.provider_name,
        dependencies.vlm_provider.provider_name,
        dependencies.settings.max_retries,
    )
    return compiled


def run_document(
    compiled_workflow: Any,
    image_bytes: bytes,
    doc_type: str,
    *,
    thread_id: str,
) -> dict[str, Any]:
    """Run one document through a compiled workflow.

    The thread id identifies the run inside the checkpointer; it is the handle
    an operator later uses to resume an interrupted document.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return compiled_workflow.invoke(initial_state(image_bytes, doc_type), config=config)


def resume_document(
    compiled_workflow: Any,
    *,
    thread_id: str,
    approved: bool,
    corrections: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Resume a run suspended by the human in the loop node.

    Args:
        compiled_workflow: The same compiled graph, bound to the same checkpointer.
        thread_id: Identifier of the suspended run.
        approved: Whether the operator accepts the document.
        corrections: Field level corrections keyed by canonical field name.
    """
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}
    return compiled_workflow.invoke(
        Command(resume={"approved": approved, "corrections": corrections or {}}),
        config=config,
    )
