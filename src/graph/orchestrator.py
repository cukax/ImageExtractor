"""Dossier orchestrator: the Map-Reduce graph over a multi-document file.

```
START -> smart_clustering -> [dynamic fan-out] -> document_worker (xN) -> consolidation -> END
```

``build_orchestrator_graph`` depends on the domain and the ports only.
``compile_dossier_workflow`` is the convenience composition helper, and is the
single function here that reaches into the infrastructure layer to resolve the
configured adapters.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Optional

from langgraph.graph import END, START, StateGraph

from ..domain.orchestrator_state import (
    DocumentResultStatus,
    DossierOrchestratorState,
    DossierReport,
    ProcessedDocumentResult,
    UnresolvedFieldDetail,
    build_dossier_report,
    initial_dossier_state,
)
from ..domain.policy import WorkflowPolicy
from ..domain.worker_state import DocumentWorkerState, WorkerStatus
from ..ports.extractor_port import DocumentExtractorPort
from ..ports.pdf_port import PDFProcessorPort
from ..ports.vlm_port import VLMProviderPort

if TYPE_CHECKING:  # pragma: no cover - type checking only, no runtime dependency
    from ..infrastructure.config import Dependencies

from .edges import DOCUMENT_WORKER_NODE, dispatch_documents
from .nodes import (
    CONSOLIDATION_NODE,
    SMART_CLUSTERING_NODE,
    build_consolidation_node,
    build_smart_clustering_node,
)
from .worker_graph import compile_worker_graph

LOGGER = logging.getLogger(__name__)

#: Status of a worker that never produced a usable document.
_FAILED_WORKER_STATUSES = frozenset({WorkerStatus.FAILED.value})


def build_document_worker_node(compiled_worker_graph: Any) -> Any:
    """Wrap the compiled worker subgraph as a node of the orchestrator.

    The wrapper exists to translate: a worker speaks ``DocumentWorkerState``,
    while the orchestrator's reduce channel expects a single
    :class:`ProcessedDocumentResult`. Doing the translation here, rather than
    letting the subgraph write the parent state directly, keeps the two state
    contracts independent.
    """

    def document_worker_node(state: DocumentWorkerState) -> dict[str, Any]:
        doc_id = state.get("doc_id", "")
        try:
            final_state = compiled_worker_graph.invoke(state)
        except Exception as error:  # noqa: BLE001 - one bad document must not sink the dossier
            LOGGER.exception("Worker for document %s crashed", doc_id)
            return {
                "processed_documents": [
                    ProcessedDocumentResult(
                        doc_id=doc_id,
                        doc_type=state.get("doc_type") or "UNKNOWN",
                        status=DocumentResultStatus.FAILED.value,
                        extracted_data={},
                        page_indexes=list(state.get("page_indexes") or []),
                    )
                ],
                "errors": [f"Worker for {doc_id} failed: {error}"],
            }

        return {"processed_documents": [_to_document_result(final_state)]}

    return document_worker_node


def _to_document_result(worker_state: DocumentWorkerState) -> ProcessedDocumentResult:
    """Project a finished worker state onto the published document contract."""
    extracted_data = dict(worker_state.get("extracted_data", {}))
    unresolved_fields = list(worker_state.get("unresolved_fields", []))
    unresolved_names = {detail.field_name for detail in unresolved_fields}
    worker_status = worker_state.get("status", WorkerStatus.PROCESSING.value)

    # A field counts as successful when it carries a value and is not among the
    # unresolved ones, so the two lists together always describe every field the
    # engine actually produced.
    successful_fields = sorted(
        field_name
        for field_name, value in extracted_data.items()
        if field_name not in unresolved_names and value not in (None, "", [], {})
    )

    if worker_status in _FAILED_WORKER_STATUSES:
        status = DocumentResultStatus.FAILED.value
    elif unresolved_fields or worker_status == WorkerStatus.REQUIRES_MANUAL_ENTRY:
        status = DocumentResultStatus.REQUIRES_MANUAL_ENTRY.value
    elif not successful_fields:
        # A document type with no validation rules registered passes validation
        # trivially, even when the extraction returned nothing at all. Reporting
        # that as a success would be a lie the consumer cannot detect.
        status = DocumentResultStatus.REQUIRES_MANUAL_ENTRY.value
        unresolved_fields = [
            UnresolvedFieldDetail(
                field_name="*",
                error_reason=(
                    "The extraction returned no value for any field of this document."
                ),
            )
        ]
    else:
        status = DocumentResultStatus.SUCCESS.value

    return ProcessedDocumentResult(
        doc_id=worker_state.get("doc_id", ""),
        doc_type=worker_state.get("doc_type") or "UNKNOWN",
        status=status,
        extracted_data=extracted_data,
        successful_fields=successful_fields,
        unresolved_fields=unresolved_fields,
        page_indexes=list(worker_state.get("page_indexes") or []),
        extraction_strategy=worker_state.get("extraction_strategy"),
    )


def build_orchestrator_graph(
    pdf_processor: PDFProcessorPort,
    doc_intel_extractor: DocumentExtractorPort,
    grounding_extractor: DocumentExtractorPort,
    vlm_provider: VLMProviderPort,
    policy: Optional[WorkflowPolicy] = None,
) -> StateGraph:
    """Build the uncompiled Map-Reduce graph with every port already injected."""
    policy = policy or WorkflowPolicy()
    worker_graph = compile_worker_graph(
        doc_intel_extractor=doc_intel_extractor,
        grounding_extractor=grounding_extractor,
        vlm_provider=vlm_provider,
        policy=policy,
    )

    graph = StateGraph(DossierOrchestratorState)
    graph.add_node(
        SMART_CLUSTERING_NODE,
        build_smart_clustering_node(pdf_processor, vlm_provider, policy),
    )
    graph.add_node(DOCUMENT_WORKER_NODE, build_document_worker_node(worker_graph))
    graph.add_node(CONSOLIDATION_NODE, build_consolidation_node())

    graph.add_edge(START, SMART_CLUSTERING_NODE)
    # The map half: one Send per logical document, or a direct jump to the
    # consolidation node when the dossier yielded nothing to process.
    graph.add_conditional_edges(
        SMART_CLUSTERING_NODE,
        dispatch_documents,
        [DOCUMENT_WORKER_NODE, CONSOLIDATION_NODE],
    )
    # The reduce half: every worker converges here, and LangGraph waits for all
    # of them before running the consolidation node.
    graph.add_edge(DOCUMENT_WORKER_NODE, CONSOLIDATION_NODE)
    graph.add_edge(CONSOLIDATION_NODE, END)

    return graph


def compile_dossier_workflow(
    dependencies: Optional["Dependencies"] = None,
    *,
    checkpointer: Any = None,
) -> Any:
    """Build and compile the dossier workflow, ready to be invoked.

    Args:
        dependencies: Resolved ports. Defaults to the configured providers.
        checkpointer: Optional LangGraph checkpointer. The dossier engine is
            headless and never suspends, so unlike the single document workflow
            it does not need one; passing a durable saver still buys resumption
            after a crash on a long dossier.

    Returns:
        The compiled LangGraph application.
    """
    # Imported here rather than at module scope: this is the composition seam,
    # and keeping it local preserves the inward dependency rule.
    from ..infrastructure.config import build_dependencies

    dependencies = dependencies or build_dependencies()
    graph = build_orchestrator_graph(
        pdf_processor=dependencies.pdf_processor,
        doc_intel_extractor=dependencies.extractor,
        grounding_extractor=dependencies.grounding_extractor,
        vlm_provider=dependencies.vlm_provider,
        policy=dependencies.settings.to_workflow_policy(),
    )
    compiled = graph.compile(checkpointer=checkpointer)
    LOGGER.info(
        "Dossier workflow compiled with pdf=%s doc_intel=%s grounding=%s vlm=%s",
        dependencies.pdf_processor.provider_name,
        dependencies.extractor.provider_name,
        dependencies.grounding_extractor.provider_name,
        dependencies.vlm_provider.provider_name,
    )
    return compiled


def run_dossier(
    compiled_workflow: Any,
    pdf_bytes: bytes,
    *,
    dossier_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> DossierReport:
    """Run one dossier end to end and return the master report.

    The run is unattended by construction: no node in this graph can suspend, so
    the call always returns a terminal report.
    """
    dossier_id = dossier_id or f"DOSSIER_{uuid.uuid4().hex[:12].upper()}"
    config = {"configurable": {"thread_id": thread_id or dossier_id}}
    final_state = compiled_workflow.invoke(
        initial_dossier_state(pdf_bytes, dossier_id),
        config=config,
    )
    return build_dossier_report(final_state)
