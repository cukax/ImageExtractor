"""End to end tests of the LangGraph workflow, driven entirely by fakes."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.domain.state import DocumentStatus
from src.graph.workflow import build_graph, resume_document, run_document
from src.infrastructure.config import Settings
from src.ports.vlm_port import UNREADABLE_TOKEN

from .conftest import FakeExtractorAdapter, FakeVLMAdapter

# A consistent invoice: 1000 - 0 + 160 = 1160.
CONSISTENT_INVOICE = {
    "invoice_id": ("A-100234", (0.70, 0.05, 0.20, 0.03)),
    "invoice_date": ("31/01/2026", (0.70, 0.10, 0.20, 0.03)),
    "vendor_name": ("Servicios Integrales SA de CV", (0.05, 0.05, 0.40, 0.04)),
    "subtotal": ("1000.00", (0.75, 0.80, 0.15, 0.03)),
    "tax_amount": ("160.00", (0.75, 0.85, 0.15, 0.03)),
    "total_amount": ("1160.00", (0.75, 0.90, 0.15, 0.03)),
}


def _with_broken_total(**overrides: Any) -> dict[str, Any]:
    """Return the invoice with a total that breaks the arithmetic rule."""
    return {**CONSISTENT_INVOICE, "total_amount": ("1600.00", (0.75, 0.90, 0.15, 0.03)), **overrides}


def _compile(extractor: Any, vlm: Any, settings: Settings) -> Any:
    """Compile the graph with the injected fakes and an in memory checkpointer."""
    policy = settings.to_workflow_policy()
    return build_graph(extractor, vlm, policy).compile(checkpointer=InMemorySaver())


def _pending_interrupt(workflow: Any, thread_id: str) -> Any:
    """Return the first pending interrupt of a thread, or None."""
    snapshot = workflow.get_state({"configurable": {"thread_id": thread_id}})
    for task in getattr(snapshot, "tasks", []) or []:
        for task_interrupt in getattr(task, "interrupts", []) or []:
            return task_interrupt
    return None


# --------------------------------------------------------------------------- #
# Fail fast filter
# --------------------------------------------------------------------------- #
def test_a_blurred_document_never_reaches_the_extractor(
    blurred_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(CONSISTENT_INVOICE)
    vlm = FakeVLMAdapter()
    workflow = _compile(extractor, vlm, settings)

    final_state = run_document(workflow, blurred_image_bytes, "Invoice", thread_id="blur-1")

    assert final_state["status"] == DocumentStatus.REJECTED_BLUR
    assert final_state["is_readable"] is False
    # The whole point of the filter: no paid API call was made.
    assert extractor.calls == []
    assert vlm.roi_calls == []


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_a_consistent_document_validates_without_any_retry(
    sharp_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(CONSISTENT_INVOICE)
    vlm = FakeVLMAdapter()
    workflow = _compile(extractor, vlm, settings)

    final_state = run_document(workflow, sharp_image_bytes, "Invoice", thread_id="ok-1")

    assert final_state["status"] == DocumentStatus.VALIDATED
    assert final_state["validation_errors"] == {}
    assert final_state["retry_count"] == 0
    assert vlm.roi_calls == []
    assert final_state["extracted_data"]["total_amount"] == "1160.00"
    assert len(extractor.calls) == 1


def test_the_preprocessed_image_is_what_reaches_the_extractor(
    sharp_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(CONSISTENT_INVOICE)
    workflow = _compile(extractor, FakeVLMAdapter(), settings)

    run_document(workflow, sharp_image_bytes, "Invoice", thread_id="pre-1")

    payload_size, doc_type = extractor.calls[0]
    assert doc_type == "Invoice"
    # Recompressed at quality 85, so it must differ from the original upload.
    assert payload_size != len(sharp_image_bytes)


# --------------------------------------------------------------------------- #
# Targeted ROI retry
# --------------------------------------------------------------------------- #
def test_the_roi_retry_repairs_a_single_failing_field(
    sharp_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(_with_broken_total())
    vlm = FakeVLMAdapter({"total_amount": "1160.00"})
    workflow = _compile(extractor, vlm, settings)

    final_state = run_document(workflow, sharp_image_bytes, "Invoice", thread_id="retry-1")

    assert final_state["status"] == DocumentStatus.VALIDATED
    assert final_state["extracted_data"]["total_amount"] == "1160.00"
    # Exactly one field was re-read, and the extractor was not called again.
    assert [call[0] for call in vlm.roi_calls] == ["total_amount"]
    assert len(extractor.calls) == 1


def test_the_retry_prompt_carries_the_validation_error_and_the_expected_format(
    sharp_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(_with_broken_total())
    vlm = FakeVLMAdapter({"total_amount": "1160.00"})
    workflow = _compile(extractor, vlm, settings)

    run_document(workflow, sharp_image_bytes, "Invoice", thread_id="retry-2")

    _, error_context, crop_size = vlm.roi_calls[0]
    assert "Arithmetic inconsistency" in error_context
    assert "expected format" in error_context.lower()
    assert crop_size > 0


def test_the_crop_is_much_smaller_than_the_full_page(
    sharp_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(_with_broken_total())
    vlm = FakeVLMAdapter({"total_amount": "1160.00"})
    workflow = _compile(extractor, vlm, settings)

    final_state = run_document(workflow, sharp_image_bytes, "Invoice", thread_id="crop-1")

    _, _, crop_size = vlm.roi_calls[0]
    assert crop_size < len(final_state["image_bytes"])


def test_a_field_without_geometry_falls_back_to_a_full_page_reread(
    sharp_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(_with_broken_total(total_amount=("1600.00", None)))
    vlm = FakeVLMAdapter({"total_amount": "1160.00"})
    workflow = _compile(extractor, vlm, settings)

    final_state = run_document(workflow, sharp_image_bytes, "Invoice", thread_id="nobox-1")

    assert final_state["status"] == DocumentStatus.VALIDATED
    _, _, crop_size = vlm.roi_calls[0]
    # No bounding box, so the whole preprocessed page was sent instead.
    assert crop_size == len(final_state["image_bytes"])


# --------------------------------------------------------------------------- #
# Escalation
# --------------------------------------------------------------------------- #
def test_the_retry_budget_is_enforced_before_escalating(
    sharp_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(_with_broken_total())
    # The model keeps returning an inconsistent value, so the loop must stop.
    vlm = FakeVLMAdapter({"total_amount": "9999.00"})
    workflow = _compile(extractor, vlm, settings)

    run_document(workflow, sharp_image_bytes, "Invoice", thread_id="hitl-1")

    assert len(vlm.roi_calls) == settings.max_retries
    pending = _pending_interrupt(workflow, "hitl-1")
    assert pending is not None
    assert "total_amount" in pending.value["flagged_fields"]
    assert pending.value["retry_count"] == settings.max_retries + 1


def test_an_unreadable_field_escalates_immediately(
    sharp_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(_with_broken_total())
    vlm = FakeVLMAdapter({"total_amount": UNREADABLE_TOKEN})
    workflow = _compile(extractor, vlm, settings)

    run_document(workflow, sharp_image_bytes, "Invoice", thread_id="hitl-2")

    # One attempt is enough: re-reading a field the model called unreadable
    # would only burn tokens.
    assert len(vlm.roi_calls) == 1
    assert _pending_interrupt(workflow, "hitl-2") is not None


def test_an_operator_correction_resumes_and_validates_the_document(
    sharp_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(_with_broken_total())
    vlm = FakeVLMAdapter({"total_amount": "9999.00"})
    workflow = _compile(extractor, vlm, settings)
    run_document(workflow, sharp_image_bytes, "Invoice", thread_id="hitl-3")

    final_state = resume_document(
        workflow,
        thread_id="hitl-3",
        approved=True,
        corrections={"total_amount": "1160.00"},
    )

    assert final_state["status"] == DocumentStatus.VALIDATED
    assert final_state["extracted_data"]["total_amount"] == "1160.00"
    assert final_state["human_review_result"]["approved"] is True
    assert final_state["flagged_fields"] == []


def test_an_operator_cannot_approve_a_still_invalid_document(
    sharp_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(_with_broken_total())
    vlm = FakeVLMAdapter({"total_amount": "9999.00"})
    workflow = _compile(extractor, vlm, settings)
    run_document(workflow, sharp_image_bytes, "Invoice", thread_id="hitl-4")

    final_state = resume_document(
        workflow,
        thread_id="hitl-4",
        approved=True,
        corrections={"total_amount": "7777.00"},
    )

    # The same tools that rejected the automated extraction also police the
    # operator, so a bad correction cannot be waved through.
    assert final_state["status"] == DocumentStatus.HUMAN_REVIEW_REQUIRED
    assert "total_amount" in final_state["validation_errors"]


# --------------------------------------------------------------------------- #
# Infrastructure failures
# --------------------------------------------------------------------------- #
def test_a_provider_outage_ends_the_run_without_crashing(
    sharp_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(CONSISTENT_INVOICE, raise_error=True)
    workflow = _compile(extractor, FakeVLMAdapter(), settings)

    final_state = run_document(workflow, sharp_image_bytes, "Invoice", thread_id="fail-1")

    assert final_state["status"] == DocumentStatus.EXTRACTION_FAILED
    assert any("Simulated provider outage" in error for error in final_state["errors"])


def test_an_empty_payload_is_reported_rather_than_raising(settings: Settings) -> None:
    workflow = _compile(FakeExtractorAdapter(CONSISTENT_INVOICE), FakeVLMAdapter(), settings)

    final_state = run_document(workflow, b"", "Invoice", thread_id="empty-1")

    assert final_state["status"] == DocumentStatus.EXTRACTION_FAILED
    assert final_state["is_readable"] is False


# --------------------------------------------------------------------------- #
# Identity documents
# --------------------------------------------------------------------------- #
def test_an_id_card_is_repaired_through_the_same_graph(
    sharp_image_bytes: bytes,
    settings: Settings,
) -> None:
    extractor = FakeExtractorAdapter(
        {
            "Name": ("MARIA GOMEZ CRUZ", (0.35, 0.20, 0.55, 0.05)),
            # A classic OCR confusion: the letter O read instead of the digit 0.
            "CURP": ("GOCM85O315MDFMRR07", (0.35, 0.55, 0.55, 0.04)),
            "VoterKey": ("GOCRMR85031509M400", (0.35, 0.62, 0.55, 0.04)),
            "DateOfBirth": ("15/03/1985", (0.35, 0.35, 0.30, 0.04)),
        }
    )
    vlm = FakeVLMAdapter({"curp": "GOCM850315MDFMRR07"})
    workflow = _compile(extractor, vlm, settings)

    final_state = run_document(workflow, sharp_image_bytes, "INE", thread_id="ine-1")

    assert final_state["status"] == DocumentStatus.VALIDATED
    assert final_state["extracted_data"]["curp"] == "GOCM850315MDFMRR07"
    # The provider vocabulary was translated into the canonical field names.
    assert final_state["extracted_data"]["full_name"] == "MARIA GOMEZ CRUZ"
    assert "curp" in final_state["bounding_boxes"]


@pytest.mark.parametrize("doc_type", ["INE", "Invoice", "Form"])
def test_every_document_type_runs_through_the_same_graph(
    sharp_image_bytes: bytes,
    settings: Settings,
    doc_type: str,
) -> None:
    extractor = FakeExtractorAdapter({"some_field": ("some value", (0.1, 0.1, 0.2, 0.05))})
    workflow = _compile(extractor, FakeVLMAdapter(), settings)

    final_state = run_document(workflow, sharp_image_bytes, doc_type, thread_id=f"any-{doc_type}")

    assert final_state["status"] in {
        DocumentStatus.VALIDATED,
        DocumentStatus.HUMAN_REVIEW_REQUIRED,
        DocumentStatus.PROCESSING,
    }
