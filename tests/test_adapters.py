"""Tests for the adapter layer that need no network access."""

from __future__ import annotations

import json

import pytest

from src.domain.models import InvoiceSchema
from src.infrastructure.adapters.extractors.azure_document_intelligence import (
    AzureDocumentIntelligenceAdapter,
    _normalize_polygon,
)
from src.infrastructure.adapters.extractors.native_vlm import (
    NativeVLMExtractorAdapter,
    build_prompt_schema,
)
from src.infrastructure.adapters.processors import PyMuPDFAdapter
from src.infrastructure.adapters.vlm.azure_foundry_vlm import AzureAIFoundryVLMAdapter
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
from src.ports.pdf_port import PDFProcessingError
from src.ports.vlm_port import UNREADABLE_TOKEN, VLMProviderError

from .conftest import FakeExtractorAdapter, FakeVLMAdapter, make_pdf_bytes


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


def test_the_native_extractor_accepts_the_grounding_contract_key_names() -> None:
    response = json.dumps(
        {
            "invoice_id": "A-1",
            # The shape the visual grounding prompt asks for, whose corner is
            # named x_min / y_min rather than x / y.
            "_bounding_boxes": {
                "invoice_id": {"x_min": 0.7, "y_min": 0.05, "width": 0.2, "height": 0.03}
            },
        }
    )
    adapter = NativeVLMExtractorAdapter(
        FakeVLMAdapter(full_document_response=response),
        request_bounding_boxes=True,
    )

    result = adapter.extract_document(b"fake-image", "Invoice")

    assert result.bounding_boxes()["invoice_id"] == pytest.approx((0.7, 0.05, 0.2, 0.03))


def test_the_native_extractor_reads_every_page_in_one_call() -> None:
    vlm = FakeVLMAdapter(full_document_response=json.dumps({"invoice_id": "A-1"}))
    adapter = NativeVLMExtractorAdapter(vlm)

    result = adapter.extract_pages([b"page-one", b"page-two"], "Invoice")

    # One call carrying both pages, not one call per page: splitting them would
    # lose exactly the cross-page information the single call recovers.
    assert vlm.page_calls == [("Invoice", 2)]
    assert result.page_count == 2


def test_the_multi_page_prompt_forbids_one_answer_per_page() -> None:
    schema = build_prompt_schema(InvoiceSchema)

    single = build_primary_extraction_user_prompt("Invoice", schema, page_count=1)
    multiple = build_primary_extraction_user_prompt("Invoice", schema, page_count=2)

    assert "spans 2 images" in multiple
    assert "Merge them into ONE JSON object" in multiple
    assert "spans" not in single


# --------------------------------------------------------------------------- #
# Multi page extraction merge (the port's default implementation)
# --------------------------------------------------------------------------- #
def test_pages_are_merged_with_the_first_non_empty_value_winning() -> None:
    class TwoPageExtractor(FakeExtractorAdapter):
        """Returns a different payload for each page."""

        FRONT = {"Name": ("MARIA GOMEZ", (0.1, 0.1, 0.2, 0.05)), "CURP": (None, None)}
        BACK = {
            "Name": ("M. GOMEZ", None),
            "CURP": ("GOCM850315MDFMRR07", (0.3, 0.4, 0.2, 0.05)),
        }

        def extract_document(self, image_bytes: bytes, doc_type: str):
            self.calls.append((len(image_bytes), doc_type))
            payload = self.FRONT if image_bytes == b"front" else self.BACK
            return FakeExtractorAdapter(payload).extract_document(image_bytes, doc_type)

    result = TwoPageExtractor({}).extract_pages([b"front", b"back"], "INE")

    # The front carries the authoritative name; the back contributes the CURP
    # the front left empty.
    assert result.fields["Name"].value == "MARIA GOMEZ"
    assert result.fields["CURP"].value == "GOCM850315MDFMRR07"
    assert result.page_count == 2


def test_a_merged_box_remembers_which_page_it_came_from() -> None:
    extractor = FakeExtractorAdapter({"CURP": ("GOCM850315MDFMRR07", (0.3, 0.4, 0.2, 0.05))})

    result = extractor.extract_pages([b"front", b"back"], "INE")

    # Without the page index the ROI retry would crop the front of a card
    # looking for a field printed on its back.
    assert result.fields["CURP"].bounding_box.page == 0


def test_one_unreadable_page_does_not_sink_a_two_page_document() -> None:
    class HalfBrokenExtractor(FakeExtractorAdapter):
        def extract_document(self, image_bytes: bytes, doc_type: str):
            self.calls.append((len(image_bytes), doc_type))
            if image_bytes == b"back":
                raise ExtractionError(self.provider_name, "Page unreadable.")
            return FakeExtractorAdapter(
                {"Name": ("MARIA GOMEZ", (0.1, 0.1, 0.2, 0.05))}
            ).extract_document(image_bytes, doc_type)

    result = HalfBrokenExtractor({}).extract_pages([b"front", b"back"], "INE")

    assert result.fields["Name"].value == "MARIA GOMEZ"
    assert any("Page unreadable" in warning for warning in result.warnings)


def test_a_document_whose_every_page_fails_raises() -> None:
    extractor = FakeExtractorAdapter({}, raise_error=True)

    with pytest.raises(ExtractionError):
        extractor.extract_pages([b"front", b"back"], "INE")


def test_extracting_no_page_at_all_raises() -> None:
    with pytest.raises(ExtractionError):
        FakeExtractorAdapter({}).extract_pages([], "INE")


# --------------------------------------------------------------------------- #
# Coordinate normalization (Azure Document Intelligence)
# --------------------------------------------------------------------------- #
def test_an_azure_polygon_becomes_a_normalized_bounding_box() -> None:
    # The 8-point form Azure reports: four corners, in page units.
    polygon = [2.0, 1.0, 6.0, 1.0, 6.0, 3.0, 2.0, 3.0]

    box = _normalize_polygon(polygon, page_width=8.0, page_height=10.0)

    assert box.x == pytest.approx(0.25)
    assert box.y == pytest.approx(0.10)
    assert box.width == pytest.approx(0.50)
    assert box.height == pytest.approx(0.20)


def test_a_rotated_polygon_is_reduced_to_its_enclosing_box() -> None:
    # A scan tilted a few degrees produces corners that no longer align.
    polygon = [2.0, 1.2, 6.0, 1.0, 6.2, 3.0, 2.2, 3.2]

    box = _normalize_polygon(polygon, page_width=10.0, page_height=10.0)

    assert box.x == pytest.approx(0.20)
    assert box.y == pytest.approx(0.10)
    assert box.width == pytest.approx(0.42)
    assert box.height == pytest.approx(0.22)


def test_out_of_bounds_coordinates_are_clamped_rather_than_rejected() -> None:
    # A corner a hair beyond the page edge is routine on a rotated scan; a
    # clamped box still crops correctly, while a rejected one loses the retry.
    polygon = [-1.0, -2.0, 12.0, -2.0, 12.0, 14.0, -1.0, 14.0]

    box = _normalize_polygon(polygon, page_width=10.0, page_height=10.0)

    assert box.x == 0.0
    assert box.y == 0.0
    assert box.width == 1.0
    assert box.height == 1.0
    assert not box.is_degenerate


def test_the_page_index_travels_with_the_box() -> None:
    polygon = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    box = _normalize_polygon(polygon, page_width=10.0, page_height=10.0, page=2)

    assert box.page == 2


@pytest.mark.parametrize("polygon", [[], [1.0, 2.0], [1.0, 2.0, 3.0]])
def test_a_malformed_polygon_is_rejected(polygon: list[float]) -> None:
    with pytest.raises(ValueError):
        _normalize_polygon(polygon, page_width=10.0, page_height=10.0)


def test_a_page_without_dimensions_is_rejected() -> None:
    polygon = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    with pytest.raises(ValueError):
        _normalize_polygon(polygon, page_width=0.0, page_height=10.0)


def test_the_azure_adapter_selects_managed_identity_when_no_key_is_given() -> None:
    adapter = AzureDocumentIntelligenceAdapter(endpoint="https://example.cognitiveservices.azure.com/")

    # An empty key is the preferred enterprise configuration, not an error.
    assert adapter.uses_managed_identity is True


def test_the_azure_adapter_still_accepts_an_explicit_key() -> None:
    adapter = AzureDocumentIntelligenceAdapter(
        endpoint="https://example.cognitiveservices.azure.com/",
        api_key="secret",
    )

    assert adapter.uses_managed_identity is False


def test_the_azure_adapter_requires_an_endpoint() -> None:
    with pytest.raises(ExtractionError):
        AzureDocumentIntelligenceAdapter(endpoint="", api_key="secret")


# --------------------------------------------------------------------------- #
# Microsoft AI Foundry adapter
# --------------------------------------------------------------------------- #
def test_the_foundry_adapter_defaults_to_managed_identity() -> None:
    adapter = AzureAIFoundryVLMAdapter(
        endpoint="https://example.openai.azure.com/",
        deployment="gpt-4o",
    )

    assert adapter.uses_managed_identity is True
    assert adapter.provider_name == "azure_foundry"


def test_the_foundry_adapter_uses_a_cheaper_deployment_for_classification() -> None:
    adapter = AzureAIFoundryVLMAdapter(
        endpoint="https://example.openai.azure.com/",
        deployment="gpt-4o",
        classifier_deployment="gpt-4o-mini",
    )

    # Classification is the most frequent and least demanding call of a dossier
    # run, so it must not pay the full vision model price.
    assert adapter.classifier_model == "gpt-4o-mini"
    assert adapter.model == "gpt-4o"


def test_the_foundry_adapter_falls_back_to_the_main_deployment() -> None:
    adapter = AzureAIFoundryVLMAdapter(
        endpoint="https://example.openai.azure.com/",
        deployment="gpt-4o",
    )

    assert adapter.classifier_model == "gpt-4o"


@pytest.mark.parametrize(
    ("endpoint", "deployment"),
    [("", "gpt-4o"), ("https://example.openai.azure.com/", "")],
)
def test_the_foundry_adapter_requires_an_endpoint_and_a_deployment(
    endpoint: str,
    deployment: str,
) -> None:
    with pytest.raises(VLMProviderError):
        AzureAIFoundryVLMAdapter(endpoint=endpoint, deployment=deployment)


# --------------------------------------------------------------------------- #
# PDF rasterization
# --------------------------------------------------------------------------- #
def test_every_pdf_page_is_rasterized_in_order() -> None:
    pages = PyMuPDFAdapter().render_pages(make_pdf_bytes(page_count=3), dpi=150)

    assert [page.page_index for page in pages] == [0, 1, 2]
    assert all(page.image_bytes.startswith(b"\x89PNG") for page in pages)
    assert all(page.width > 0 and page.height > 0 for page in pages)


def test_a_higher_dpi_yields_a_larger_raster() -> None:
    pdf_bytes = make_pdf_bytes(page_count=1)

    low = PyMuPDFAdapter().render_pages(pdf_bytes, dpi=72)[0]
    high = PyMuPDFAdapter().render_pages(pdf_bytes, dpi=300)[0]

    # 300 DPI is the floor at which small print on an identity card survives
    # rasterization, so the resolution must actually reach the renderer.
    assert high.width > low.width
    assert high.height > low.height


def test_a_bare_image_is_treated_as_a_one_page_dossier(sharp_image_bytes: bytes) -> None:
    pages = PyMuPDFAdapter().render_pages(sharp_image_bytes)

    # A caller must not have to branch on the upload format.
    assert len(pages) == 1
    assert pages[0].image_bytes == sharp_image_bytes
    assert (pages[0].width, pages[0].height) == (1200, 800)


def test_the_page_count_is_read_without_rasterizing() -> None:
    assert PyMuPDFAdapter().page_count(make_pdf_bytes(page_count=4)) == 4


def test_an_empty_payload_is_reported_as_a_processing_error() -> None:
    with pytest.raises(PDFProcessingError):
        PyMuPDFAdapter().render_pages(b"")


def test_an_undecodable_payload_is_reported_as_a_processing_error() -> None:
    with pytest.raises(PDFProcessingError):
        PyMuPDFAdapter().render_pages(b"this is neither a pdf nor an image")
