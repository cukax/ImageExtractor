"""Tests for the regex and business rule validation tools."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from src.domain.models import (
    IDCardSchema,
    InvoiceSchema,
    ProofOfAddressSchema,
    get_schema_for,
    parse_decimal,
)
from src.domain.validation_tools import (
    check_invoice_line_items,
    check_invoice_totals,
    expected_format_for,
    parse_date,
    validate_document,
)

VALID_ID_CARD = {
    "full_name": "MARIA GOMEZ CRUZ",
    "curp": "GOCM850315MDFMRR07",
    "voter_key": "GOCRMR85031509M400",
    "date_of_birth": "15/03/1985",
    "sex": "M",
}

VALID_INVOICE = {
    "invoice_id": "A-100234",
    "invoice_date": "31/01/2026",
    "vendor_name": "Servicios Integrales SA de CV",
    "subtotal": "1000.00",
    "tax_amount": "160.00",
    "total_amount": "1160.00",
}


# --------------------------------------------------------------------------- #
# Regex tools
# --------------------------------------------------------------------------- #
def test_a_valid_id_card_produces_no_error() -> None:
    assert validate_document("INE", VALID_ID_CARD) == {}


def test_a_malformed_curp_is_reported_on_its_own_field() -> None:
    errors = validate_document("INE", {**VALID_ID_CARD, "curp": "GOCM85O315MDFMRR0"})

    assert "curp" in errors
    assert "GOCM85O315MDFMRR0" in errors["curp"]


def test_a_missing_required_field_is_reported() -> None:
    errors = validate_document("INE", {**VALID_ID_CARD, "date_of_birth": None})

    assert "date_of_birth" in errors
    assert "missing or was illegible" in errors["date_of_birth"]


def test_an_optional_field_may_be_absent() -> None:
    errors = validate_document("INE", {**VALID_ID_CARD, "postal_code": None})

    assert "postal_code" not in errors


def test_an_optional_field_is_still_checked_when_present() -> None:
    errors = validate_document("INE", {**VALID_ID_CARD, "postal_code": "1234"})

    assert "postal_code" in errors


@pytest.mark.parametrize(
    "raw_date",
    ["15/03/1985", "1985-03-15", "15-03-1985", "15.03.1985"],
)
def test_supported_date_layouts_parse(raw_date: str) -> None:
    assert parse_date(raw_date) is not None


def test_an_unknown_document_type_has_no_rule_and_never_blocks() -> None:
    assert validate_document("Form", {"anything": "goes"}) == {}


# --------------------------------------------------------------------------- #
# Business rule tools
# --------------------------------------------------------------------------- #
def test_consistent_totals_pass() -> None:
    assert check_invoice_totals(VALID_INVOICE) == {}


def test_inconsistent_totals_are_attributed_to_the_total_field() -> None:
    errors = check_invoice_totals({**VALID_INVOICE, "total_amount": "1600.00"})

    assert set(errors) == {"total_amount"}
    assert "Arithmetic inconsistency" in errors["total_amount"]


def test_a_discount_is_subtracted_before_tax_is_added() -> None:
    invoice = {
        "subtotal": "1000.00",
        "discount_amount": "100.00",
        "tax_amount": "144.00",
        "total_amount": "1044.00",
    }

    assert check_invoice_totals(invoice) == {}


def test_a_rounding_difference_within_tolerance_is_accepted() -> None:
    assert check_invoice_totals({**VALID_INVOICE, "total_amount": "1160.01"}) == {}


def test_a_missing_amount_does_not_produce_a_duplicate_error() -> None:
    # The field level rule already reports the missing subtotal; the business
    # rule must stay silent so the retry targets a single field.
    assert check_invoice_totals({**VALID_INVOICE, "subtotal": None}) == {}


def test_line_items_are_checked_against_the_subtotal() -> None:
    errors = check_invoice_line_items(
        {
            "subtotal": "1000.00",
            "line_items": [{"amount": "400.00"}, {"amount": "550.00"}],
        }
    )

    assert set(errors) == {"subtotal"}


def test_partial_line_detail_cannot_prove_an_inconsistency() -> None:
    errors = check_invoice_line_items(
        {"subtotal": "1000.00", "line_items": [{"amount": "400.00"}, {"amount": None}]}
    )

    assert errors == {}


def test_the_curp_birth_date_cross_check_catches_a_misread_digit() -> None:
    errors = validate_document("INE", {**VALID_ID_CARD, "date_of_birth": "15/03/1986"})

    assert "curp" in errors
    assert "does not match" in errors["curp"]


def test_a_future_birth_date_is_rejected() -> None:
    errors = validate_document(
        "INE",
        {**VALID_ID_CARD, "curp": None, "date_of_birth": "15/03/2099"},
    )

    assert "date_of_birth" in errors


# --------------------------------------------------------------------------- #
# Number parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1,234.56", Decimal("1234.56")),
        ("1.234,56", Decimal("1234.56")),
        ("$ 1 234.56 MXN", Decimal("1234.56")),
        ("(250.00)", Decimal("-250.00")),
        ("", None),
        ("illegible", None),
    ],
)
def test_decimal_parsing_handles_ocr_noise(raw_value: str, expected: Decimal | None) -> None:
    assert parse_decimal(raw_value) == expected


# --------------------------------------------------------------------------- #
# Schema wiring
# --------------------------------------------------------------------------- #
def test_the_registry_resolves_the_typed_schemas() -> None:
    assert get_schema_for("INE") is IDCardSchema
    assert get_schema_for("invoice") is InvoiceSchema


def test_expected_formats_are_available_for_the_vlm_prompt() -> None:
    assert "CURP" in (expected_format_for("INE", "curp") or "")
    assert expected_format_for("INE", "unknown_field") is None


# --------------------------------------------------------------------------- #
# Dossier document types
# --------------------------------------------------------------------------- #
def test_the_registry_resolves_the_dossier_document_types() -> None:
    assert get_schema_for("PROOF_OF_ADDRESS") is ProofOfAddressSchema
    # A front and a back merged into one logical document still parse with the
    # very same identity card schema.
    assert get_schema_for("INE_COMBINED") is IDCardSchema
    assert get_schema_for("MEXICAN_CEDULA") is IDCardSchema


def test_an_id_card_validates_under_its_dossier_token() -> None:
    # The clustering step labels the document INE_COMBINED, so the same rules
    # that police an "INE" must be registered for it too.
    assert validate_document("INE_COMBINED", VALID_ID_CARD) == {}


VALID_PROOF_OF_ADDRESS = {
    "holder_name": "MARIA GOMEZ CRUZ",
    "service_provider": "Comision Federal de Electricidad",
    "address": "Av Insurgentes Sur 1234, Col Del Valle",
    "postal_code": "03100",
    "issue_date": "15/01/2026",
}


def test_a_consistent_proof_of_address_validates() -> None:
    assert validate_document("PROOF_OF_ADDRESS", VALID_PROOF_OF_ADDRESS) == {}


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("postal_code", "3100"),
        ("postal_code", "ABCDE"),
        # A truncated address is the most common silent OCR failure here.
        ("address", "Av Ins"),
        ("issue_date", "not a date"),
        ("holder_name", "X"),
    ],
)
def test_a_malformed_proof_of_address_field_is_reported(field: str, bad_value: str) -> None:
    errors = validate_document(
        "PROOF_OF_ADDRESS", {**VALID_PROOF_OF_ADDRESS, field: bad_value}
    )

    assert field in errors


def test_a_missing_required_proof_of_address_field_is_reported() -> None:
    payload = {**VALID_PROOF_OF_ADDRESS}
    del payload["postal_code"]

    assert "postal_code" in validate_document("PROOF_OF_ADDRESS", payload)


def test_an_optional_proof_of_address_field_may_be_absent() -> None:
    payload = {**VALID_PROOF_OF_ADDRESS}
    del payload["service_provider"]

    assert validate_document("PROOF_OF_ADDRESS", payload) == {}


def test_a_proof_of_address_issued_in_the_future_is_rejected() -> None:
    future_year = date.today().year + 2
    errors = validate_document(
        "PROOF_OF_ADDRESS",
        {**VALID_PROOF_OF_ADDRESS, "issue_date": f"15/01/{future_year}"},
    )

    assert "issue_date" in errors
    assert "future" in errors["issue_date"]


def test_the_proof_of_address_schema_normalizes_its_codes() -> None:
    document = ProofOfAddressSchema.model_validate(
        {"postal_code": " 031 00 ", "account_number": "abc 123"}
    )

    # OCR sprinkles whitespace into codes; normalizing here keeps the regex
    # rules simple.
    assert document.postal_code == "03100"
    assert document.account_number == "ABC123"
