"""Tests for the dossier splitter and the semantic clustering step."""

from __future__ import annotations

from typing import Any

import pytest

from src.domain.orchestrator_state import (
    DossierStatus,
    LogicalDocType,
    PageType,
    initial_dossier_state,
)
from src.domain.policy import WorkflowPolicy
from src.graph.nodes.smart_clustering import (
    _group_pages_into_logical_docs,
    _normalize_page_type,
    build_smart_clustering_node,
)

from .conftest import FakePDFProcessorAdapter, FakeVLMAdapter

DOSSIER_ID = "DOSSIER_2026_99482"


def _page(page_index: int, page_type: str) -> dict[str, Any]:
    """Build one classified page entry, as the classifier step produces it."""
    return {"page_index": page_index, "page_type": page_type, "width": 1200, "height": 800}


# --------------------------------------------------------------------------- #
# Semantic grouping
# --------------------------------------------------------------------------- #
def test_an_interleaved_front_and_back_are_paired_into_one_document() -> None:
    pages = [
        _page(0, PageType.INE_FRONT),
        _page(1, PageType.PROOF_OF_ADDRESS),
        _page(2, PageType.INE_BACK),
    ]

    grouped = _group_pages_into_logical_docs(pages, DOSSIER_ID)

    # The whole point of semantic clustering: page 0 and page 2 belong together
    # even though an unrelated page sits between them.
    assert len(grouped) == 2
    combined = grouped[0]
    assert combined["doc_type"] == LogicalDocType.INE_COMBINED
    assert combined["page_indexes"] == [0, 2]
    assert grouped[1]["doc_type"] == PageType.PROOF_OF_ADDRESS
    assert grouped[1]["page_indexes"] == [1]


def test_adjacent_front_and_back_are_paired_too() -> None:
    pages = [_page(0, PageType.INE_FRONT), _page(1, PageType.INE_BACK)]

    grouped = _group_pages_into_logical_docs(pages, DOSSIER_ID)

    assert len(grouped) == 1
    assert grouped[0]["page_indexes"] == [0, 1]


def test_two_identity_cards_do_not_cross_pair() -> None:
    pages = [
        _page(0, PageType.INE_FRONT),
        _page(1, PageType.INE_FRONT),
        _page(2, PageType.INE_BACK),
        _page(3, PageType.INE_BACK),
    ]

    grouped = _group_pages_into_logical_docs(pages, DOSSIER_ID)

    # Each front claims the first still unclaimed back, so no page is used twice
    # and no back is left orphaned.
    assert [document["page_indexes"] for document in grouped] == [[0, 2], [1, 3]]
    assert all(document["doc_type"] == LogicalDocType.INE_COMBINED for document in grouped)


def test_an_unpaired_front_stays_a_single_page_document() -> None:
    pages = [_page(0, PageType.INE_FRONT), _page(1, PageType.INVOICE)]

    grouped = _group_pages_into_logical_docs(pages, DOSSIER_ID)

    assert grouped[0]["doc_type"] == PageType.INE_FRONT
    assert grouped[0]["page_indexes"] == [0]


def test_an_orphan_back_page_becomes_its_own_document() -> None:
    grouped = _group_pages_into_logical_docs([_page(0, PageType.INE_BACK)], DOSSIER_ID)

    assert len(grouped) == 1
    assert grouped[0]["doc_type"] == PageType.INE_BACK


def test_unknown_pages_are_kept_rather_than_dropped() -> None:
    pages = [_page(0, PageType.UNKNOWN), _page(1, PageType.UNKNOWN)]

    grouped = _group_pages_into_logical_docs(pages, DOSSIER_ID)

    # Dropping an unrecognized page would silently lose a document; the router
    # sends UNKNOWN down the visual grounding path instead.
    assert len(grouped) == 2


def test_document_ids_are_unique_and_carry_the_dossier_prefix() -> None:
    pages = [
        _page(0, PageType.INVOICE),
        _page(1, PageType.INVOICE),
        _page(2, PageType.PROOF_OF_ADDRESS),
    ]

    grouped = _group_pages_into_logical_docs(pages, DOSSIER_ID)

    doc_ids = [document["doc_id"] for document in grouped]
    assert doc_ids == [
        f"{DOSSIER_ID}_INVOICE_1",
        f"{DOSSIER_ID}_INVOICE_2",
        f"{DOSSIER_ID}_PROOF_OF_ADDRESS_1",
    ]
    assert len(set(doc_ids)) == len(doc_ids)


def test_an_empty_page_list_produces_no_document() -> None:
    assert _group_pages_into_logical_docs([], DOSSIER_ID) == []


# --------------------------------------------------------------------------- #
# Label normalization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        ("INE_FRONT", PageType.INE_FRONT),
        ("ine_front", PageType.INE_FRONT),
        ("ine front", PageType.INE_FRONT),
        ("INE-FRONT", PageType.INE_FRONT),
        # A chatty model that ignored the "label only" instruction.
        ("This page is an INE_BACK.", PageType.INE_BACK),
        ("something else entirely", PageType.UNKNOWN),
        ("", PageType.UNKNOWN),
        (None, PageType.UNKNOWN),
    ],
)
def test_page_labels_are_coerced_into_the_closed_vocabulary(
    raw_label: str,
    expected: str,
) -> None:
    assert _normalize_page_type(raw_label) == expected


# --------------------------------------------------------------------------- #
# The node end to end
# --------------------------------------------------------------------------- #
def test_the_clustering_node_classifies_and_groups_every_page(
    dossier_pages: list[bytes],
) -> None:
    pdf_processor = FakePDFProcessorAdapter(dossier_pages)
    vlm = FakeVLMAdapter(
        page_types=[PageType.INE_FRONT, PageType.INVOICE, PageType.INE_BACK]
    )
    node = build_smart_clustering_node(pdf_processor, vlm, WorkflowPolicy())

    update = node(initial_dossier_state(b"%PDF-fake", DOSSIER_ID))

    assert len(update["raw_pages"]) == 3
    assert len(update["grouped_documents"]) == 2
    assert update["grouped_documents"][0]["doc_type"] == LogicalDocType.INE_COMBINED
    # The rasters travel with the document, so the fan-out payload is self contained.
    assert len(update["grouped_documents"][0]["images_bytes"]) == 2
    assert pdf_processor.calls[0][1] == WorkflowPolicy().pdf_render_dpi


def test_the_classifier_sees_a_downscaled_copy_not_the_full_raster(
    dossier_pages: list[bytes],
) -> None:
    vlm = FakeVLMAdapter(page_types=[PageType.INVOICE])
    node = build_smart_clustering_node(
        FakePDFProcessorAdapter(dossier_pages[:1]),
        vlm,
        WorkflowPolicy(),
    )

    node(initial_dossier_state(b"%PDF-fake", DOSSIER_ID))

    # Classification is a layout question; sending the 300 DPI raster would
    # multiply the token bill of the most frequent call in the run.
    assert vlm.classify_calls[0] < len(dossier_pages[0])


def test_a_classifier_outage_degrades_to_unknown_instead_of_failing(
    dossier_pages: list[bytes],
) -> None:
    node = build_smart_clustering_node(
        FakePDFProcessorAdapter(dossier_pages),
        FakeVLMAdapter(raise_on_classify=True),
        WorkflowPolicy(),
    )

    update = node(initial_dossier_state(b"%PDF-fake", DOSSIER_ID))

    assert all(page["page_type"] == PageType.UNKNOWN for page in update["raw_pages"])
    assert len(update["grouped_documents"]) == 3
    assert update["dossier_status"] == DossierStatus.PROCESSING
    assert any("could not be classified" in error for error in update["errors"])


def test_a_rasterization_failure_fails_the_dossier() -> None:
    node = build_smart_clustering_node(
        FakePDFProcessorAdapter(raise_error=True),
        FakeVLMAdapter(),
        WorkflowPolicy(),
    )

    update = node(initial_dossier_state(b"%PDF-broken", DOSSIER_ID))

    assert update["dossier_status"] == DossierStatus.FAILED
    assert update["grouped_documents"] == []
    assert any("Simulated rasterization failure" in error for error in update["errors"])


def test_an_empty_payload_fails_before_any_api_call() -> None:
    vlm = FakeVLMAdapter()
    pdf_processor = FakePDFProcessorAdapter([])
    node = build_smart_clustering_node(pdf_processor, vlm, WorkflowPolicy())

    update = node(initial_dossier_state(b"", DOSSIER_ID))

    assert update["dossier_status"] == DossierStatus.FAILED
    assert pdf_processor.calls == []
    assert vlm.classify_calls == []
