"""LangGraph nodes for both workflows.

Every node is produced by a factory that closes over the ports it needs. That is
the dependency injection seam of the system: the node functions themselves only
ever see a DocumentExtractorPort, a VLMProviderPort or a PDFProcessorPort, never
a concrete SDK.

Two families live here:

* ``single_document``: the original per-image pipeline, human-in-the-loop.
* the dossier nodes: clustering, routing, the two extraction branches, worker
  validation, ROI retry and the headless unresolved handler.
"""

from __future__ import annotations

from .consolidation import CONSOLIDATION_NODE, build_consolidation_node
from .doc_intel import DOC_INTEL_NODE, build_doc_intel_node
from .router import (
    NODE_BY_STRATEGY,
    ROUTER_NODE,
    build_router_node,
    route_after_router,
)
from .single_document import (
    CROP_AND_VLM_RETRY_NODE,
    EXTRACTION_NODE,
    HITL_FALLBACK_NODE,
    PREPROCESS_NODE,
    VALIDATION_NODE,
    NodeCallable,
    build_crop_and_vlm_retry_node,
    build_extraction_node,
    build_hitl_fallback_node,
    build_preprocess_node,
    build_route_after_validation,
    build_validation_node,
    route_after_extraction,
    route_after_preprocess,
)
from .smart_clustering import (
    CLASSIFIABLE_PAGE_TYPES,
    SMART_CLUSTERING_NODE,
    build_smart_clustering_node,
)
from .unresolved_handler import UNRESOLVED_HANDLER_NODE, build_unresolved_handler_node
from .vlm_grounding import VLM_GROUNDING_NODE, build_vlm_grounding_node
from .vlm_retry import VLM_RETRY_NODE, build_vlm_retry_node
from .worker_validation import (
    WORKER_VALIDATION_NODE,
    build_route_after_worker_validation,
    build_worker_validation_node,
)

__all__ = [
    # --- Single document workflow ---
    "CROP_AND_VLM_RETRY_NODE",
    "EXTRACTION_NODE",
    "HITL_FALLBACK_NODE",
    "NodeCallable",
    "PREPROCESS_NODE",
    "VALIDATION_NODE",
    "build_crop_and_vlm_retry_node",
    "build_extraction_node",
    "build_hitl_fallback_node",
    "build_preprocess_node",
    "build_route_after_validation",
    "build_validation_node",
    "route_after_extraction",
    "route_after_preprocess",
    # --- Dossier orchestrator ---
    "CLASSIFIABLE_PAGE_TYPES",
    "CONSOLIDATION_NODE",
    "SMART_CLUSTERING_NODE",
    "build_consolidation_node",
    "build_smart_clustering_node",
    # --- Document worker ---
    "DOC_INTEL_NODE",
    "NODE_BY_STRATEGY",
    "ROUTER_NODE",
    "UNRESOLVED_HANDLER_NODE",
    "VLM_GROUNDING_NODE",
    "VLM_RETRY_NODE",
    "WORKER_VALIDATION_NODE",
    "build_doc_intel_node",
    "build_route_after_worker_validation",
    "build_router_node",
    "build_unresolved_handler_node",
    "build_vlm_grounding_node",
    "build_vlm_retry_node",
    "build_worker_validation_node",
    "route_after_router",
]
