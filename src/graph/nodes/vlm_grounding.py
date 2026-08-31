"""Phase 4B: visual grounding extraction node.

Reads every page of the logical document in a single vision call, asking the model
for both the field values and their normalized bounding boxes. Grounding is what
makes the ROI retry worth doing: without geometry the repair pass degrades to
re-reading a whole page.

The vision model is reached through the very same
:class:`~src.ports.extractor_port.DocumentExtractorPort` the template branch uses,
so the JSON recovery and schema mapping logic exists in one place only. The
composition root injects a grounding capable adapter here, which lets one dossier
mix an Azure prebuilt model on identity documents with a Foundry vision
deployment on everything else.
"""

from __future__ import annotations

from typing import Any, Callable

from ...domain.worker_state import DocumentWorkerState
from ...ports.extractor_port import DocumentExtractorPort
from ._shared import build_worker_extraction_node

VLM_GROUNDING_NODE = "vlm_grounding"


def build_vlm_grounding_node(
    extractor: DocumentExtractorPort,
) -> Callable[[DocumentWorkerState], dict[str, Any]]:
    """Build the visual grounding node around a grounding capable extractor port."""
    return build_worker_extraction_node(extractor, VLM_GROUNDING_NODE)
