"""Phase 5: tool validation inside the worker subgraph.

Runs the very same deterministic tools the single document workflow runs — regex
formats, ISO date parsing, arithmetic verification — over the worker state.

Like its single document counterpart, this node owns the retry counter.
Incrementing it here rather than in the retry node keeps the budget correct even
when the retry node repairs nothing because no failing field could be located.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langgraph.graph import END

from ...domain.policy import WorkflowPolicy
from ...domain.validation_tools import validate_document
from ...domain.worker_state import DocumentWorkerState, WorkerStatus
from .unresolved_handler import UNRESOLVED_HANDLER_NODE
from .vlm_retry import VLM_RETRY_NODE

LOGGER = logging.getLogger(__name__)

WORKER_VALIDATION_NODE = "validation"


def build_worker_validation_node(
    policy: WorkflowPolicy,
) -> Callable[[DocumentWorkerState], dict[str, Any]]:
    """Build the worker validation node."""

    def validation_node(state: DocumentWorkerState) -> dict[str, Any]:
        doc_type = state.get("doc_type") or ""
        extracted_data = state.get("extracted_data", {})
        errors = validate_document(doc_type, extracted_data)

        if not errors:
            LOGGER.info(
                "Document %s validated after %d retries",
                state.get("doc_id"),
                state.get("retry_count", 0),
            )
            return {
                "validation_errors": {},
                "failed_fields": [],
                "status": WorkerStatus.VALIDATED.value,
            }

        failed_fields = sorted(errors)
        retry_count = int(state.get("retry_count", 0)) + 1
        LOGGER.warning(
            "Document %s failed validation on %s (attempt %d/%d)",
            state.get("doc_id"),
            failed_fields,
            retry_count,
            policy.max_retries,
        )
        return {
            "validation_errors": errors,
            "failed_fields": failed_fields,
            "retry_count": retry_count,
            "status": WorkerStatus.PROCESSING.value,
        }

    return validation_node


def build_route_after_worker_validation(
    policy: WorkflowPolicy,
) -> Callable[[DocumentWorkerState], str]:
    """Build the router arbitrating between success, retry and manual entry."""

    def route_after_validation(state: DocumentWorkerState) -> str:
        if state.get("status") == WorkerStatus.VALIDATED:
            return END

        flagged_fields = set(state.get("flagged_fields", []))
        retryable = [
            field for field in state.get("failed_fields", []) if field not in flagged_fields
        ]
        if not retryable:
            # Every remaining error is on a field the model already declared
            # unreadable, so another pass cannot change the outcome.
            return UNRESOLVED_HANDLER_NODE

        if int(state.get("retry_count", 0)) <= policy.max_retries:
            return VLM_RETRY_NODE
        return UNRESOLVED_HANDLER_NODE

    return route_after_validation
