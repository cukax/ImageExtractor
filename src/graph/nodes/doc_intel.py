"""Phase 4A: template based extraction node.

Invokes a :class:`~src.ports.extractor_port.DocumentExtractorPort` backed by a
document intelligence service over every page of the logical document. The
adapter has already normalized the provider geometry (Azure's 8-point polygons,
Google's normalized vertices) into the domain
:class:`~src.domain.models.base.BoundingBox`, so nothing provider specific
reaches the graph.
"""

from __future__ import annotations

from typing import Any, Callable

from ...domain.worker_state import DocumentWorkerState
from ...ports.extractor_port import DocumentExtractorPort
from ._shared import build_worker_extraction_node

DOC_INTEL_NODE = "doc_intel"


def build_doc_intel_node(
    extractor: DocumentExtractorPort,
) -> Callable[[DocumentWorkerState], dict[str, Any]]:
    """Build the document intelligence node around an injected extractor port."""
    return build_worker_extraction_node(extractor, DOC_INTEL_NODE)
