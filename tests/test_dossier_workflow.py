"""End to end tests of the dossier Map-Reduce graph, driven entirely by fakes."""

from __future__ import annotations

from typing import Any

from src.domain.orchestrator_state import (
    DocumentResultStatus,
    DossierStatus,
    LogicalDocType,
    PageType,
)
from src.domain.policy import WorkflowPolicy
from src.domain.worker_state import ExtractionStrategy
from src.graph.orchestrator import build_orchestrator_graph, run_dossier
from src.infrastructure.config import Settings
from src.ports.vlm_port import UNREADABLE_TOKEN

from .conftest import FakeExtractorAdapter, FakePDFProcessorAdapter, FakeVLMAdapter

DOSSIER_ID = "DOSSIER_2026_99482"

# A consistent invoice: 1000 - 0 + 160 = 1160.
CONSISTENT_INVOICE = {
    "invoice_id": ("A-100234", (0.70, 0.05, 0.20, 0.03)),
    "invoice_date": ("31/01/2026", (0.70, 0.10, 0.20, 0.03)),
    "vendor_name": ("Servicios Integrales SA de CV", (0.05, 0.05, 0.40, 0.04)),
    "subtotal": ("1000.00", (0.75, 0.80, 0.15, 0.03)),
    "tax_amount": ("160.00", (0.75, 0.85, 0.15, 0.03)),
    "total_amount": ("1160.00", (0.75, 0.90, 0.15, 0.03)),
}

VALID_ID_CARD = {
    "Name": ("MARIA GOMEZ CRUZ", (0.35, 0.20, 0.55, 0.05)),
    "CURP": ("GOCM850315MDFMRR07", (0.35, 0.55, 0.55, 0.04)),
    "VoterKey": ("GOCRMR85031509M400", (0.35, 0.62, 0.55, 0.04)),
    "DateOfBirth": ("15/03/1985", (0.35, 0.35, 0.30, 0.04)),
}


def _with_broken_total(**overrides: Any) -> dict[str, Any]:
    """Return the invoice with a total that breaks the arithmetic rule."""
    return {
        **CONSISTENT_INVOICE,
        "total_amount": ("1600.00", (0.75, 0.90, 0.15, 0.03)),
        **overrides,
    }


def _compile(
    pages: list[bytes],
    page_types: list[str],
    *,
    doc_intel_fields: dict[str, Any],
    grounding_fields: dict[str, Any],
    vlm_answers: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> tuple[Any, FakeVLMAdapter, FakeExtractorAdapter, FakeExtractorAdapter]:
    """Compile the orchestrator with fakes on every port."""
    policy = (settings or Settings(_env_file=None)).to_workflow_policy()
    pdf_processor = FakePDFProcessorAdapter(pages)
    vlm = FakeVLMAdapter(vlm_answers or {}, page_types=list(page_types))
    doc_intel = FakeExtractorAdapter(doc_intel_fields)
    grounding = FakeExtractorAdapter(grounding_fields)

    workflow = build_orchestrator_graph(
        pdf_processor=pdf_processor,
        doc_intel_extractor=doc_intel,
        grounding_extractor=grounding,
        vlm_provider=vlm,
        policy=policy,
    ).compile()
    return workflow, vlm, doc_intel, grounding


# --------------------------------------------------------------------------- #
# Map-Reduce end to end
# --------------------------------------------------------------------------- #
def test_a_dossier_fans_out_to_one_worker_per_logical_document(
    dossier_pages: list[bytes],
) -> None:
    workflow, _, doc_intel, grounding = _compile(
        dossier_pages,
        [PageType.INE_FRONT, PageType.INVOICE, PageType.INE_BACK],
        doc_intel_fields=VALID_ID_CARD,
        grounding_fields=CONSISTENT_INVOICE,
    )

    report = run_dossier(workflow, b"%PDF-fake", dossier_id=DOSSIER_ID)

    # Three pages, two logical documents: the interleaved INE was reassembled.
    assert report.total_documents_processed == 2
    assert report.dossier_status == DossierStatus.COMPLETED
    doc_types = {document.doc_type for document in report.documents}
    assert doc_types == {LogicalDocType.INE_COMBINED, PageType.INVOICE}
    # Each branch was used exactly once, by the document the router sent it.
    assert len(doc_intel.calls) == 2  # front and back, merged by extract_pages
    assert len(grounding.calls) == 1


def test_each_document_is_routed_to_the_branch_its_type_dictates(
    dossier_pages: list[bytes],
) -> None:
    workflow, _, _, _ = _compile(
        dossier_pages[:2],
        [PageType.PASSPORT, PageType.PROOF_OF_ADDRESS],
        doc_intel_fields=VALID_ID_CARD,
        grounding_fields=CONSISTENT_INVOICE,
    )

    report = run_dossier(workflow, b"%PDF-fake", dossier_id=DOSSIER_ID)

    strategies = {
        document.doc_type: document.extraction_strategy for document in report.documents
    }
    assert strategies[PageType.PASSPORT] == ExtractionStrategy.DOC_INTELLIGENCE
    assert strategies[PageType.PROOF_OF_ADDRESS] == ExtractionStrategy.VLM_GROUNDING


def test_a_multi_page_document_reaches_the_extractor_as_one_document(
    dossier_pages: list[bytes],
) -> None:
    workflow, _, doc_intel, _ = _compile(
        dossier_pages[:2],
        [PageType.INE_FRONT, PageType.INE_BACK],
        doc_intel_fields=VALID_ID_CARD,
        grounding_fields={},
    )

    report = run_dossier(workflow, b"%PDF-fake", dossier_id=DOSSIER_ID)

    assert report.total_documents_processed == 1
    assert report.documents[0].page_indexes == [0, 1]
    # Both pages were analyzed and merged into one result, not two documents.
    assert len(doc_intel.calls) == 2


# --------------------------------------------------------------------------- #
# The master output contract
# --------------------------------------------------------------------------- #
def test_the_report_matches_the_published_json_contract(
    dossier_pages: list[bytes],
) -> None:
    workflow, _, _, _ = _compile(
        dossier_pages[:1],
        [PageType.INVOICE],
        doc_intel_fields={},
        grounding_fields=CONSISTENT_INVOICE,
    )

    payload = run_dossier(workflow, b"%PDF-fake", dossier_id=DOSSIER_ID).to_json_dict()

    assert set(payload) == {
        "dossier_id",
        "dossier_status",
        "total_documents_processed",
        "documents",
    }
    assert payload["dossier_id"] == DOSSIER_ID
    document = payload["documents"][0]
    assert document["doc_id"] == f"{DOSSIER_ID}_INVOICE_1"
    assert document["status"] == DocumentResultStatus.SUCCESS
    assert "total_amount" in document["successful_fields"]
    assert document["unresolved_fields"] == []


def test_the_report_carries_no_image_binary(dossier_pages: list[bytes]) -> None:
    workflow, _, _, _ = _compile(
        dossier_pages[:1],
        [PageType.INVOICE],
        doc_intel_fields={},
        grounding_fields=CONSISTENT_INVOICE,
    )

    payload = run_dossier(workflow, b"%PDF-fake", dossier_id=DOSSIER_ID).to_json_dict()

    # Serializing a few megabytes of PNG into a report is the mistake this
    # projection exists to make impossible.
    serialized = str(payload)
    assert "images_bytes" not in serialized
    assert "pdf_bytes" not in serialized


def test_a_generated_dossier_id_is_used_when_none_is_supplied(
    dossier_pages: list[bytes],
) -> None:
    workflow, _, _, _ = _compile(
        dossier_pages[:1],
        [PageType.INVOICE],
        doc_intel_fields={},
        grounding_fields=CONSISTENT_INVOICE,
    )

    report = run_dossier(workflow, b"%PDF-fake")

    assert report.dossier_id.startswith("DOSSIER_")
    assert report.documents[0].doc_id.startswith(report.dossier_id)


# --------------------------------------------------------------------------- #
# The headless repair loop
# --------------------------------------------------------------------------- #
def test_a_failing_field_is_repaired_by_the_targeted_retry(
    dossier_pages: list[bytes],
) -> None:
    workflow, vlm, _, _ = _compile(
        dossier_pages[:1],
        [PageType.INVOICE],
        doc_intel_fields={},
        grounding_fields=_with_broken_total(),
        vlm_answers={"total_amount": "1160.00"},
    )

    report = run_dossier(workflow, b"%PDF-fake", dossier_id=DOSSIER_ID)

    assert report.dossier_status == DossierStatus.COMPLETED
    assert report.documents[0].status == DocumentResultStatus.SUCCESS
    assert report.documents[0].extracted_data["total_amount"] == "1160.00"
    assert [call[0] for call in vlm.roi_calls] == ["total_amount"]


def test_an_unrepairable_field_ends_in_manual_entry_without_blocking(
    dossier_pages: list[bytes],
) -> None:
    workflow, vlm, _, _ = _compile(
        dossier_pages[:1],
        [PageType.INVOICE],
        doc_intel_fields={},
        grounding_fields=_with_broken_total(),
        vlm_answers={"total_amount": "9999.00"},
    )

    report = run_dossier(workflow, b"%PDF-fake", dossier_id=DOSSIER_ID)

    # The run completed on its own: no interrupt, no suspended thread.
    assert report.dossier_status == DossierStatus.REQUIRES_REVIEW
    document = report.documents[0]
    assert document.status == DocumentResultStatus.REQUIRES_MANUAL_ENTRY
    assert [detail.field_name for detail in document.unresolved_fields] == ["total_amount"]
    # The dirty value is retained for the operator, and the box locates it.
    assert document.extracted_data["total_amount"] == "9999.00"
    assert document.unresolved_fields[0].bounding_box is not None
    assert len(vlm.roi_calls) == WorkflowPolicy().max_retries


def test_an_unreadable_field_stops_retrying_immediately(
    dossier_pages: list[bytes],
) -> None:
    workflow, vlm, _, _ = _compile(
        dossier_pages[:1],
        [PageType.INVOICE],
        doc_intel_fields={},
        grounding_fields=_with_broken_total(),
        vlm_answers={"total_amount": UNREADABLE_TOKEN},
    )

    report = run_dossier(workflow, b"%PDF-fake", dossier_id=DOSSIER_ID)

    assert len(vlm.roi_calls) == 1
    assert report.documents[0].status == DocumentResultStatus.REQUIRES_MANUAL_ENTRY


def test_the_retry_crop_is_enhanced_and_much_smaller_than_the_page(
    dossier_pages: list[bytes],
) -> None:
    workflow, vlm, _, _ = _compile(
        dossier_pages[:1],
        [PageType.INVOICE],
        doc_intel_fields={},
        grounding_fields=_with_broken_total(),
        vlm_answers={"total_amount": "1160.00"},
    )

    run_dossier(workflow, b"%PDF-fake", dossier_id=DOSSIER_ID)

    _, error_context, crop_size = vlm.roi_calls[0]
    assert "Arithmetic inconsistency" in error_context
    assert "expected format" in error_context.lower()
    assert 0 < crop_size < len(dossier_pages[0])


# --------------------------------------------------------------------------- #
# Failure isolation
# --------------------------------------------------------------------------- #
def test_one_failing_document_does_not_sink_the_whole_dossier(
    dossier_pages: list[bytes],
) -> None:
    policy = Settings(_env_file=None).to_workflow_policy()
    workflow = build_orchestrator_graph(
        pdf_processor=FakePDFProcessorAdapter(dossier_pages[:2]),
        # The identity branch is down; the invoice branch is healthy.
        doc_intel_extractor=FakeExtractorAdapter(VALID_ID_CARD, raise_error=True),
        grounding_extractor=FakeExtractorAdapter(CONSISTENT_INVOICE),
        vlm_provider=FakeVLMAdapter(page_types=[PageType.PASSPORT, PageType.INVOICE]),
        policy=policy,
    ).compile()

    report = run_dossier(workflow, b"%PDF-fake", dossier_id=DOSSIER_ID)

    by_type = {document.doc_type: document for document in report.documents}
    assert by_type[PageType.PASSPORT].status == DocumentResultStatus.FAILED
    assert by_type[PageType.INVOICE].status == DocumentResultStatus.SUCCESS
    assert report.dossier_status == DossierStatus.REQUIRES_REVIEW


def test_a_document_that_extracted_nothing_is_not_reported_as_a_success(
    dossier_pages: list[bytes],
) -> None:
    workflow, _, _, _ = _compile(
        dossier_pages[:1],
        # MEXICAN_CEDULA has no validation rules registered, so it passes
        # validation trivially even on an empty extraction.
        [PageType.MEXICAN_CEDULA],
        doc_intel_fields={},
        grounding_fields={},
    )

    report = run_dossier(workflow, b"%PDF-fake", dossier_id=DOSSIER_ID)

    # Reporting an empty document as SUCCESS would be a lie the consumer of the
    # report cannot detect.
    document = report.documents[0]
    assert document.status == DocumentResultStatus.REQUIRES_MANUAL_ENTRY
    assert document.successful_fields == []
    assert "no value for any field" in document.unresolved_fields[0].error_reason


def test_a_dossier_that_yields_no_page_still_reaches_a_terminal_report() -> None:
    policy = Settings(_env_file=None).to_workflow_policy()
    workflow = build_orchestrator_graph(
        pdf_processor=FakePDFProcessorAdapter([]),
        doc_intel_extractor=FakeExtractorAdapter({}),
        grounding_extractor=FakeExtractorAdapter({}),
        vlm_provider=FakeVLMAdapter(),
        policy=policy,
    ).compile()

    report = run_dossier(workflow, b"%PDF-empty", dossier_id=DOSSIER_ID)

    assert report.dossier_status == DossierStatus.FAILED
    assert report.total_documents_processed == 0


def test_a_rasterization_failure_reaches_a_terminal_report() -> None:
    policy = Settings(_env_file=None).to_workflow_policy()
    workflow = build_orchestrator_graph(
        pdf_processor=FakePDFProcessorAdapter(raise_error=True),
        doc_intel_extractor=FakeExtractorAdapter({}),
        grounding_extractor=FakeExtractorAdapter({}),
        vlm_provider=FakeVLMAdapter(),
        policy=policy,
    ).compile()

    report = run_dossier(workflow, b"%PDF-broken", dossier_id=DOSSIER_ID)

    assert report.dossier_status == DossierStatus.FAILED


# --------------------------------------------------------------------------- #
# The reduce channel
# --------------------------------------------------------------------------- #
def test_every_parallel_worker_appends_exactly_one_result(
    dossier_pages: list[bytes],
) -> None:
    workflow, _, _, _ = _compile(
        dossier_pages,
        [PageType.INVOICE, PageType.INVOICE, PageType.INVOICE],
        doc_intel_fields={},
        grounding_fields=CONSISTENT_INVOICE,
    )

    report = run_dossier(workflow, b"%PDF-fake", dossier_id=DOSSIER_ID)

    # Three concurrent workers, three results, no loss and no duplication: the
    # operator.add reducer is what guarantees it.
    assert report.total_documents_processed == 3
    assert len({document.doc_id for document in report.documents}) == 3
