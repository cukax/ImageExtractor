"""Tests for the adapter layer that need no network access."""

from __future__ import annotations

import json

import pytest

from src.domain.models import InvoiceSchema
from src.infrastructure.adapters.extractors.native_vlm import (
    NativeVLMExtractorAdapter,
    build_prompt_schema,
)
from src.infrastructure.adapters.vlm.base import (
    detect_media_type,
    extract_json_object,
    sanitize_scalar_response,
)
from src.infrastructure.adapters.vlm.prompts import (
    PRIMARY_EXTRACTION_SYSTEM_PROMPT,
    ROI_RETRY_SYSTEM_PROMPT,
    build_primary_extraction_user_prompt,
    build_roi_retry_user_prompt,
)
from src.ports.extractor_port import ExtractionError
from src.ports.vlm_port import UNREADABLE_TOKEN, VLMProviderError

from .conftest import FakeVLMAdapter


# --------------------------------------------------------------------------- #
# Response sanitizing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw_response", "expected"),
    [
        ("GOCM850315MDFMRR07", "GOCM850315MDFMRR07"),
        ("```\nGOCM850315MDFMRR07\n```", "GOCM850315MDFMRR07"),
        ('"1,234.56"', "1,234.56"),
        ("The value is 1160.00", "1160.00"),
        ("1160.00\nThis is the total printed at the bottom.", "1160.00"),
        ("UNREADABLE", UNREADABLE_TOKEN),
        ("   ", UNREADABLE_TOKEN),
        ("", UNREADABLE_TOKEN),
    ],
)
def test_a_chatty_answer_is_reduced_to_the_bare_value(raw_response: str, expected: str) -> None:
    assert sanitize_scalar_response(raw_response) == expected


# --------------------------------------------------------------------------- #
# JSON recovery
# --------------------------------------------------------------------------- #
def test_a_clean_json_object_parses() -> None:
    assert extract_json_object('{"invoice_id": "A-1"}', "test") == {"invoice_id": "A-1"}


def test_a_fenced_json_object_parses() -> None:
    raw_response = '```json\n{"invoice_id": "A-1"}\n```'

    assert extract_json_object(raw_response, "test") == {"invoice_id": "A-1"}


def test_a_json_object_buried_in_prose_is_recovered() -> None:
    raw_response = 'Here is the result:\n{"invoice_id": "A-1", "note": "}"}\nHope this helps.'

    assert extract_json_object(raw_response, "test") == {"invoice_id": "A-1", "note": "}"}


def test_a_response_without_json_is_rejected() -> None:
    with pytest.raises(VLMProviderError):
        extract_json_object("I could not read this document.", "test")


# --------------------------------------------------------------------------- #
# Media type sniffing
# --------------------------------------------------------------------------- #
def test_media_types_are_sniffed_from_the_magic_bytes() -> None:
    assert detect_media_type(b"\x89PNG\r\n\x1a\n rest") == "image/png"
    assert detect_media_type(b"\xff\xd8\xff\xe0 rest") == "image/jpeg"


# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #
def test_the_roi_prompt_contains_the_field_and_the_error() -> None:
    prompt = build_roi_retry_user_prompt("total_amount", "Arithmetic inconsistency.")

    assert "'total_amount'" in prompt
    assert "Arithmetic inconsistency." in prompt
    assert "Return ONLY the raw extracted value as a plain string." in prompt
    assert "UNREADABLE" in prompt
    assert "high-resolution image crop analysis" in ROI_RETRY_SYSTEM_PROMPT


def test_the_primary_prompt_embeds_the_schema_and_the_layout_hints() -> None:
    schema = build_prompt_schema(InvoiceSchema)
    prompt = build_primary_extraction_user_prompt(
        "Invoice", schema, layout_hints="Totals are bottom right."
    )

    assert "Invoice" in prompt
    assert "total_amount" in prompt
    assert "Totals are bottom right." in prompt
    assert "_bounding_boxes" not in prompt
    assert "set its value to null" in PRIMARY_EXTRACTION_SYSTEM_PROMPT


def test_grounding_is_requested_only_when_enabled() -> None:
    schema = build_prompt_schema(InvoiceSchema)

    assert "_bounding_boxes" in build_primary_extraction_user_prompt(
        "Invoice", schema, request_bounding_boxes=True
    )


def test_the_prompt_schema_stays_flat_and_documented() -> None:
    schema = build_prompt_schema(InvoiceSchema)

    assert schema["type"] == "object"
    assert "$defs" not in schema
    # Amounts are advertised as strings so the model transcribes rather than
    # reformats what is printed.
    assert schema["properties"]["total_amount"]["type"] == ["string", "null"]
    assert schema["properties"]["line_items"]["type"] == ["array", "null"]


# --------------------------------------------------------------------------- #
# Native VLM extractor adapter
# --------------------------------------------------------------------------- #
def test_the_native_extractor_maps_a_json_response_onto_the_contract() -> None:
    response = json.dumps(
        {
            "invoice_id": "A-100234",
            "subtotal": "1000.00",
            "tax_amount": "160.00",
            "total_amount": "1160.00",
            "due_date": None,
            "line_items": [{"description": "Consulting", "amount": "1000.00"}],
        }
    )
    adapter = NativeVLMExtractorAdapter(FakeVLMAdapter(full_document_response=response))

    result = adapter.extract_document(b"fake-image", "Invoice")

    assert result.provider == "native_vlm"
    assert result.fields["invoice_id"].value == "A-100234"
    assert result.fields["due_date"].value is None
    assert result.structured_extras["line_items"][0]["amount"] == "1000.00"


def test_the_native_extractor_unwraps_a_container_key() -> None:
    response = json.dumps({"fields": {"invoice_id": "A-1"}})
    adapter = NativeVLMExtractorAdapter(FakeVLMAdapter(full_document_response=response))

    result = adapter.extract_document(b"fake-image", "Invoice")

    assert result.fields["invoice_id"].value == "A-1"


def test_the_native_extractor_treats_spelled_out_blanks_as_null() -> None:
    response = json.dumps({"invoice_id": "N/A", "vendor_name": "null", "due_date": ""})
    adapter = NativeVLMExtractorAdapter(FakeVLMAdapter(full_document_response=response))

    result = adapter.extract_document(b"fake-image", "Invoice")

    assert result.fields["invoice_id"].value is None
    assert result.fields["vendor_name"].value is None
    assert result.fields["due_date"].value is None


def test_the_native_extractor_maps_reported_bounding_boxes() -> None:
    response = json.dumps(
        {
            "invoice_id": "A-1",
            "_bounding_boxes": {"invoice_id": [0.7, 0.05, 0.2, 0.03]},
        }
    )
    adapter = NativeVLMExtractorAdapter(
        FakeVLMAdapter(full_document_response=response),
        request_bounding_boxes=True,
    )

    result = adapter.extract_document(b"fake-image", "Invoice")

    assert result.bounding_boxes()["invoice_id"] == pytest.approx((0.7, 0.05, 0.2, 0.03))


def test_the_native_extractor_ignores_a_malformed_bounding_box() -> None:
    response = json.dumps(
        {"invoice_id": "A-1", "_bounding_boxes": {"invoice_id": ["left", "top"]}}
    )
    adapter = NativeVLMExtractorAdapter(
        FakeVLMAdapter(full_document_response=response),
        request_bounding_boxes=True,
    )

    result = adapter.extract_document(b"fake-image", "Invoice")

    assert result.bounding_boxes() == {}
    assert result.warnings


def test_a_non_json_response_becomes_an_extraction_error() -> None:
    adapter = NativeVLMExtractorAdapter(FakeVLMAdapter(full_document_response="I cannot read it."))

    with pytest.raises(ExtractionError):
        adapter.extract_document(b"fake-image", "Invoice")


def test_layout_hints_are_resolved_per_document_type() -> None:
    adapter = NativeVLMExtractorAdapter(
        FakeVLMAdapter(full_document_response="{}"),
        layout_hints_by_doc_type={"INE": "The photo is on the left."},
    )

    assert adapter.layout_hints_for("ine") == "The photo is on the left."
    assert adapter.layout_hints_for("Invoice") is None
