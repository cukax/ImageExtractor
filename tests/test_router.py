"""Tests for the worker router and its configurable decision table."""

from __future__ import annotations

import pytest

from src.domain.orchestrator_state import LogicalDocType
from src.domain.policy import WorkflowPolicy
from src.domain.worker_state import ExtractionStrategy, WorkerStatus, initial_worker_state
from src.graph.nodes.router import (
    DOC_INTEL_NODE,
    VLM_GROUNDING_NODE,
    build_router_node,
    route_after_router,
)


def _state(doc_type: str) -> dict:
    """Build a worker state carrying only what the router looks at."""
    return initial_worker_state("doc-1", doc_type, [b"page"])


# --------------------------------------------------------------------------- #
# The decision table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "doc_type",
    [
        LogicalDocType.INE_COMBINED,
        LogicalDocType.INE_FRONT,
        LogicalDocType.INE_BACK,
        LogicalDocType.PASSPORT,
        LogicalDocType.MEXICAN_CEDULA,
    ],
)
def test_identity_documents_route_to_document_intelligence(doc_type: str) -> None:
    update = build_router_node(WorkflowPolicy())(_state(doc_type))

    assert update["extraction_strategy"] == ExtractionStrategy.DOC_INTELLIGENCE
    assert update["status"] == WorkerStatus.PROCESSING


@pytest.mark.parametrize(
    "doc_type",
    [
        LogicalDocType.INVOICE,
        LogicalDocType.PROOF_OF_ADDRESS,
        LogicalDocType.UNKNOWN,
    ],
)
def test_everything_else_routes_to_visual_grounding(doc_type: str) -> None:
    update = build_router_node(WorkflowPolicy())(_state(doc_type))

    assert update["extraction_strategy"] == ExtractionStrategy.VLM_GROUNDING


def test_an_unregistered_document_type_falls_back_to_visual_grounding() -> None:
    update = build_router_node(WorkflowPolicy())(_state("SOME_NEW_FORM"))

    # Defaulting to a vision model rather than a prebuilt one is deliberate: a
    # vision model degrades to best effort on an unexpected layout, while a
    # prebuilt model returns nothing at all.
    assert update["extraction_strategy"] == ExtractionStrategy.VLM_GROUNDING


def test_an_empty_document_type_falls_back_to_visual_grounding() -> None:
    update = build_router_node(WorkflowPolicy())(_state(""))

    assert update["extraction_strategy"] == ExtractionStrategy.VLM_GROUNDING


@pytest.mark.parametrize("spelling", ["INE_COMBINED", "ine_combined", "InECombined"])
def test_the_routing_table_is_matched_on_the_normalized_token(spelling: str) -> None:
    update = build_router_node(WorkflowPolicy())(_state(spelling))

    assert update["extraction_strategy"] == ExtractionStrategy.DOC_INTELLIGENCE


def test_the_decision_table_is_configurable_without_touching_the_graph() -> None:
    # Moving invoices onto the template branch must be a configuration change,
    # not a code change.
    policy = WorkflowPolicy(
        strategy_by_doc_type={
            LogicalDocType.INVOICE.value: ExtractionStrategy.DOC_INTELLIGENCE.value
        }
    )

    update = build_router_node(policy)(_state(LogicalDocType.INVOICE))

    assert update["extraction_strategy"] == ExtractionStrategy.DOC_INTELLIGENCE


# --------------------------------------------------------------------------- #
# The conditional edge
# --------------------------------------------------------------------------- #
def test_the_recorded_strategy_selects_the_node() -> None:
    assert route_after_router(
        {"extraction_strategy": ExtractionStrategy.DOC_INTELLIGENCE.value}
    ) == DOC_INTEL_NODE
    assert route_after_router(
        {"extraction_strategy": ExtractionStrategy.VLM_GROUNDING.value}
    ) == VLM_GROUNDING_NODE


def test_a_missing_strategy_still_reaches_an_extraction_node() -> None:
    # A router that somehow left no decision must not dangle the worker.
    assert route_after_router({}) == VLM_GROUNDING_NODE


def test_the_decision_is_recorded_in_the_state_for_auditing() -> None:
    update = build_router_node(WorkflowPolicy())(_state(LogicalDocType.PASSPORT))

    # Recording the branch is what lets a reviewer see which engine produced a
    # disputed value.
    assert "extraction_strategy" in update
