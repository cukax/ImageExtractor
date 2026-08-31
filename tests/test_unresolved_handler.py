"""Tests for the headless unresolved handler.

The single guarantee this node exists to provide: a dossier run never blocks on a
human. Everything else it does is about losing as little information as possible
on the way to that guarantee.
"""

from __future__ import annotations

from src.domain.orchestrator_state import BoundingBox2D
from src.domain.worker_state import WorkerStatus, initial_worker_state
from src.graph.nodes.unresolved_handler import build_unresolved_handler_node

TOTAL_BOX = (0.7241, 0.8512, 0.1820, 0.0315)


def _failed_state() -> dict:
    """A worker state that exhausted its retry budget on one field."""
    state = initial_worker_state(
        "DOSSIER_2026_99482_INVOICE_2",
        "INVOICE",
        [b"page-bytes"],
    )
    state.update(
        {
            "extracted_data": {"issuer_rfc": "ABC123456T12", "total_amount": "$1,160.SO"},
            "bounding_boxes": {"total_amount": TOTAL_BOX},
            "validation_errors": {
                "total_amount": "Value contains non-numeric characters for currency validation"
            },
            "failed_fields": ["total_amount"],
            "retry_count": 4,
        }
    )
    return state


def test_the_node_never_interrupts() -> None:
    # The node is called directly, outside any LangGraph runtime. A call to
    # interrupt() here would raise, so simply returning proves the run is
    # unattended by construction.
    update = build_unresolved_handler_node()(_failed_state())

    assert update["status"] == WorkerStatus.REQUIRES_MANUAL_ENTRY


def test_the_dirty_text_is_preserved_rather_than_discarded() -> None:
    state = _failed_state()

    update = build_unresolved_handler_node()(state)

    # extracted_data is deliberately absent from the update, so the unvalidated
    # value survives: an operator would rather fix a near miss than retype the
    # field from scratch.
    assert "extracted_data" not in update
    assert state["extracted_data"]["total_amount"] == "$1,160.SO"


def test_every_unresolved_field_carries_its_reason_and_its_geometry() -> None:
    update = build_unresolved_handler_node()(_failed_state())

    assert len(update["unresolved_fields"]) == 1
    detail = update["unresolved_fields"][0]
    assert detail.field_name == "total_amount"
    assert "non-numeric" in detail.error_reason
    # The box is what lets a review UI jump straight to the region of the page.
    assert detail.bounding_box == BoundingBox2D(
        x_min=0.7241, y_min=0.8512, width=0.1820, height=0.0315
    )


def test_a_field_that_was_never_located_still_reports_its_error() -> None:
    state = _failed_state()
    state["bounding_boxes"] = {}

    update = build_unresolved_handler_node()(state)

    detail = update["unresolved_fields"][0]
    assert detail.bounding_box is None
    assert detail.error_reason


def test_fields_flagged_unreadable_are_reported_alongside_the_failed_ones() -> None:
    state = _failed_state()
    state["flagged_fields"] = ["issuer_rfc"]

    update = build_unresolved_handler_node()(state)

    reported = {detail.field_name for detail in update["unresolved_fields"]}
    assert reported == {"issuer_rfc", "total_amount"}


def test_a_field_without_a_recorded_error_gets_a_readable_default() -> None:
    state = _failed_state()
    state["validation_errors"] = {}

    update = build_unresolved_handler_node()(state)

    assert update["unresolved_fields"][0].error_reason == (
        "The field could not be validated automatically."
    )


def test_the_published_geometry_uses_the_contract_key_names() -> None:
    update = build_unresolved_handler_node()(_failed_state())

    payload = update["unresolved_fields"][0].model_dump(mode="json")

    # The published contract names the corner x_min / y_min and carries no page
    # index, unlike the internal working representation.
    assert set(payload["bounding_box"]) == {"x_min", "y_min", "width", "height"}
