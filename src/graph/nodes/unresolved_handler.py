"""Phase 7: headless unresolved handler.

The terminal node of a worker that could not validate every field. It is the
deliberate replacement for the single document workflow's human-in-the-loop
escalation: it never calls ``interrupt()``, so a dossier run always completes
unattended.

Nothing is discarded. The unvalidated raw text stays in ``extracted_data``,
because a value that failed a format check is still the best evidence of what is
printed on the page, and an operator correcting it later would rather edit a near
miss than retype the field from scratch.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ...domain.orchestrator_state import BoundingBox2D, UnresolvedFieldDetail
from ...domain.worker_state import DocumentWorkerState, WorkerStatus

LOGGER = logging.getLogger(__name__)

UNRESOLVED_HANDLER_NODE = "unresolved_handler"


def build_unresolved_handler_node() -> Callable[[DocumentWorkerState], dict[str, Any]]:
    """Build the headless terminal node for documents needing manual entry."""

    def unresolved_handler_node(state: DocumentWorkerState) -> dict[str, Any]:
        validation_errors = state.get("validation_errors", {})
        failed_fields = list(state.get("failed_fields", []))
        flagged_fields = sorted(set(state.get("flagged_fields", [])) | set(failed_fields))
        bounding_boxes = state.get("bounding_boxes", {})

        unresolved_fields = [
            UnresolvedFieldDetail(
                field_name=field_name,
                error_reason=validation_errors.get(
                    field_name,
                    "The field could not be validated automatically.",
                ),
                bounding_box=BoundingBox2D.from_bbox_tuple(bounding_boxes.get(field_name)),
            )
            for field_name in flagged_fields
        ]

        LOGGER.warning(
            "Document %s requires manual entry on %d field(s): %s",
            state.get("doc_id"),
            len(unresolved_fields),
            flagged_fields,
        )
        return {
            # extracted_data is deliberately left untouched: the dirty text is
            # the evidence an operator needs.
            "unresolved_fields": unresolved_fields,
            "flagged_fields": flagged_fields,
            "status": WorkerStatus.REQUIRES_MANUAL_ENTRY.value,
        }

    return unresolved_handler_node
