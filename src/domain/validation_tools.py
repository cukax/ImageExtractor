"""Isolated, deterministic validation tools executed by the validation node.

Two families of tools live here:

* Regex tools, which check that a single field respects a strict format.
* Business rule tools, which check numerical or logical consistency across
  several fields at once.

Both families return the same {field_name: error_message} contract, so the ROI
retry node can always tell which field to re-read and why it failed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Mapping, Optional

from .models import normalize_key, parse_decimal

# --------------------------------------------------------------------------- #
# Regular expressions
# --------------------------------------------------------------------------- #
# CURP: 4 letters, birth date, sex, federal entity, 3 internal consonants,
# a homonym differentiator and a check digit.
CURP_PATTERN = re.compile(
    r"^[A-Z][AEIOUX][A-Z]{2}"
    r"\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
    r"[HM]"
    r"(?:AS|BC|BS|CC|CL|CM|CS|CH|DF|DG|GT|GR|HG|JC|MC|MN|MS|NT|NL|OC|PL|QT|QR|SP|SL|SR|TC|TS|TL|VZ|YN|ZS|NE)"
    r"[B-DF-HJ-NP-TV-Z]{3}"
    r"[A-Z\d]\d$"
)

# RFC: 3 letters for companies, 4 for individuals, then the date and the
# three character homoclave.
RFC_PATTERN = re.compile(r"^[A-ZN&]{3,4}\d{6}[A-Z\d]{3}$")

# INE elector key: 6 name consonants, birth date, federal entity, sex, 3 digits.
VOTER_KEY_PATTERN = re.compile(r"^[A-Z]{6}\d{8}[HM]\d{3}$")

MEXICAN_POSTAL_CODE_PATTERN = re.compile(r"^\d{5}$")
YEAR_PATTERN = re.compile(r"^(?:19|20)\d{2}$")
CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")
SEX_PATTERN = re.compile(r"^[HMX]$")
NON_EMPTY_NAME_PATTERN = re.compile(r"^[A-Za-zÀ-ÿ'.\- ]{2,120}$")

SUPPORTED_DATE_FORMATS: tuple[str, ...] = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %B %Y",
)

DEFAULT_AMOUNT_TOLERANCE = Decimal("0.01")


# --------------------------------------------------------------------------- #
# Primitive checks
# --------------------------------------------------------------------------- #
def matches(pattern: re.Pattern[str], value: str) -> bool:
    """Return True when the whole trimmed value matches the pattern."""
    return bool(pattern.fullmatch(value.strip()))


def parse_date(value: Any) -> Optional[date]:
    """Parse a date printed on a document, trying every supported layout."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for date_format in SUPPORTED_DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    return None


def is_valid_date(value: str) -> bool:
    """True when the value parses as a real calendar date."""
    return parse_date(value) is not None


# --------------------------------------------------------------------------- #
# Rule declarations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FieldRule:
    """A single field level validation tool."""

    field: str
    check: Callable[[str], bool]
    expected_format: str
    """Human readable format description, forwarded to the VLM as error context."""

    required: bool = True
    """When False a missing value is accepted, but a present value is checked."""

    def evaluate(self, value: Any) -> Optional[str]:
        """Return an error message, or None when the field is acceptable."""
        if value is None or str(value).strip() == "":
            if self.required:
                return f"The field is missing or was illegible. Expected {self.expected_format}."
            return None
        if not self.check(str(value)):
            return (
                f"The value {str(value)!r} does not match the expected format. "
                f"Expected {self.expected_format}."
            )
        return None


@dataclass(frozen=True, slots=True)
class DocumentRule:
    """A cross field business rule."""

    name: str
    check: Callable[[Mapping[str, Any]], dict[str, str]]
    """Returns {field_name: error_message} for the fields it holds responsible."""

    def evaluate(self, data: Mapping[str, Any]) -> dict[str, str]:
        """Run the rule, never letting a malformed document raise."""
        try:
            return self.check(data)
        except Exception as error:  # pragma: no cover - defensive guard
            return {self.name: f"Business rule {self.name} could not be evaluated: {error}"}


# --------------------------------------------------------------------------- #
# Business rule implementations
# --------------------------------------------------------------------------- #
def check_invoice_totals(
    data: Mapping[str, Any],
    tolerance: Decimal = DEFAULT_AMOUNT_TOLERANCE,
) -> dict[str, str]:
    """Verify that subtotal - discount + tax equals the invoice total.

    The error is attributed to total_amount because that is the single field the
    ROI retry should re-read first; the subtotal and the tax are reported as
    context inside the message.
    """
    subtotal = parse_decimal(data.get("subtotal"))
    tax_amount = parse_decimal(data.get("tax_amount"))
    total_amount = parse_decimal(data.get("total_amount"))
    discount = parse_decimal(data.get("discount_amount")) or Decimal("0")

    if subtotal is None or tax_amount is None or total_amount is None:
        # Missing amounts are already reported by the field level rules; adding a
        # second error for the same root cause would only pollute the retry.
        return {}

    expected_total = subtotal - discount + tax_amount
    difference = abs(expected_total - total_amount)
    if difference <= tolerance:
        return {}

    return {
        "total_amount": (
            f"Arithmetic inconsistency: subtotal ({subtotal}) - discount ({discount}) "
            f"+ tax ({tax_amount}) equals {expected_total}, but the extracted total is "
            f"{total_amount} (difference of {difference})."
        )
    }


def check_invoice_line_items(
    data: Mapping[str, Any],
    tolerance: Decimal = DEFAULT_AMOUNT_TOLERANCE,
) -> dict[str, str]:
    """Verify that the line item amounts add up to the subtotal."""
    subtotal = parse_decimal(data.get("subtotal"))
    line_items = data.get("line_items") or []
    if subtotal is None or not line_items:
        return {}

    amounts = [parse_decimal(item.get("amount")) for item in line_items if isinstance(item, dict)]
    known_amounts = [amount for amount in amounts if amount is not None]
    if len(known_amounts) != len(amounts) or not known_amounts:
        # Partial line detail cannot prove an inconsistency.
        return {}

    line_total = sum(known_amounts, Decimal("0"))
    if abs(line_total - subtotal) <= tolerance:
        return {}
    return {
        "subtotal": (
            f"The {len(known_amounts)} line items add up to {line_total}, "
            f"which does not match the extracted subtotal {subtotal}."
        )
    }


def check_invoice_dates(data: Mapping[str, Any]) -> dict[str, str]:
    """Verify that the due date is not earlier than the invoice date."""
    invoice_date = parse_date(data.get("invoice_date"))
    due_date = parse_date(data.get("due_date"))
    if invoice_date is None or due_date is None:
        return {}
    if due_date < invoice_date:
        return {
            "due_date": (
                f"The due date ({due_date.isoformat()}) precedes the invoice date "
                f"({invoice_date.isoformat()})."
            )
        }
    return {}


def check_birth_date_in_past(data: Mapping[str, Any]) -> dict[str, str]:
    """Verify that the birth date is a plausible past date."""
    birth_date = parse_date(data.get("date_of_birth"))
    if birth_date is None:
        return {}
    today = date.today()
    if birth_date > today:
        return {"date_of_birth": f"The birth date {birth_date.isoformat()} is in the future."}
    if birth_date.year < today.year - 120:
        return {
            "date_of_birth": (
                f"The birth date {birth_date.isoformat()} implies an implausible age."
            )
        }
    return {}


def check_curp_matches_birth_date(data: Mapping[str, Any]) -> dict[str, str]:
    """Cross check the birth date encoded inside the CURP against the printed one.

    The CURP embeds the birth date at positions 5 to 10 (YYMMDD). A mismatch
    almost always means one of the two fields was misread, which is exactly the
    kind of error a targeted re-read can fix.
    """
    curp = data.get("curp")
    birth_date = parse_date(data.get("date_of_birth"))
    if not curp or birth_date is None:
        return {}

    curp_text = str(curp).strip().upper()
    if not matches(CURP_PATTERN, curp_text):
        # The format rule already reports the malformed CURP.
        return {}

    encoded = curp_text[4:10]
    if encoded == birth_date.strftime("%y%m%d"):
        return {}
    return {
        "curp": (
            f"The birth date encoded in the CURP ({encoded}) does not match the printed "
            f"birth date ({birth_date.strftime('%y%m%d')})."
        )
    }


def check_curp_matches_sex(data: Mapping[str, Any]) -> dict[str, str]:
    """Cross check the sex marker encoded in the CURP against the printed one."""
    curp = data.get("curp")
    sex = data.get("sex")
    if not curp or not sex:
        return {}
    curp_text = str(curp).strip().upper()
    if not matches(CURP_PATTERN, curp_text):
        return {}
    if curp_text[10] == str(sex).strip().upper()[:1]:
        return {}
    return {
        "sex": (
            f"The sex encoded in the CURP ({curp_text[10]}) does not match the extracted "
            f"value ({sex})."
        )
    }


# --------------------------------------------------------------------------- #
# Rule registry
# --------------------------------------------------------------------------- #
ID_CARD_FIELD_RULES: tuple[FieldRule, ...] = (
    FieldRule(
        field="full_name",
        check=lambda value: matches(NON_EMPTY_NAME_PATTERN, value),
        expected_format="a name of 2 to 120 alphabetic characters",
    ),
    FieldRule(
        field="curp",
        check=lambda value: matches(CURP_PATTERN, value.upper()),
        expected_format="an 18 character CURP such as GOMC850315HDFNRR09",
    ),
    FieldRule(
        field="voter_key",
        check=lambda value: matches(VOTER_KEY_PATTERN, value.upper()),
        expected_format="an 18 character elector key such as GOMCCR85031509H400",
    ),
    FieldRule(
        field="date_of_birth",
        check=is_valid_date,
        expected_format="a calendar date such as 15/03/1985",
    ),
    FieldRule(
        field="sex",
        check=lambda value: matches(SEX_PATTERN, value.upper()),
        expected_format="a single letter: H, M or X",
        required=False,
    ),
    FieldRule(
        field="expiration_date",
        check=lambda value: is_valid_date(value) or matches(YEAR_PATTERN, value),
        expected_format="a calendar date or a four digit year",
        required=False,
    ),
    FieldRule(
        field="postal_code",
        check=lambda value: matches(MEXICAN_POSTAL_CODE_PATTERN, value),
        expected_format="a five digit postal code",
        required=False,
    ),
)

INVOICE_FIELD_RULES: tuple[FieldRule, ...] = (
    FieldRule(
        field="invoice_id",
        check=lambda value: 1 <= len(value.strip()) <= 60,
        expected_format="a non empty invoice number of at most 60 characters",
    ),
    FieldRule(
        field="invoice_date",
        check=is_valid_date,
        expected_format="a calendar date such as 31/01/2026",
    ),
    FieldRule(
        field="due_date",
        check=is_valid_date,
        expected_format="a calendar date such as 28/02/2026",
        required=False,
    ),
    FieldRule(
        field="vendor_name",
        check=lambda value: len(value.strip()) >= 2,
        expected_format="a non empty vendor name",
    ),
    FieldRule(
        field="vendor_tax_id",
        check=lambda value: matches(RFC_PATTERN, value.upper()),
        expected_format="a 12 or 13 character RFC such as ABC010203XY9",
        required=False,
    ),
    FieldRule(
        field="customer_tax_id",
        check=lambda value: matches(RFC_PATTERN, value.upper()),
        expected_format="a 12 or 13 character RFC such as ABC010203XY9",
        required=False,
    ),
    FieldRule(
        field="currency",
        check=lambda value: matches(CURRENCY_PATTERN, value.upper()),
        expected_format="a three letter ISO 4217 code such as MXN or USD",
        required=False,
    ),
    FieldRule(
        field="subtotal",
        check=lambda value: parse_decimal(value) is not None,
        expected_format="a decimal amount such as 1234.56",
    ),
    FieldRule(
        field="tax_amount",
        check=lambda value: parse_decimal(value) is not None,
        expected_format="a decimal amount such as 197.53",
    ),
    FieldRule(
        field="total_amount",
        check=lambda value: parse_decimal(value) is not None,
        expected_format="a decimal amount such as 1432.09",
    ),
)

PROOF_OF_ADDRESS_FIELD_RULES: tuple[FieldRule, ...] = (
    FieldRule(
        field="holder_name",
        check=lambda value: matches(NON_EMPTY_NAME_PATTERN, value),
        expected_format="a name of 2 to 120 alphabetic characters",
    ),
    FieldRule(
        field="address",
        # An address that fits in fewer than ten characters is a fragment, not a
        # location, and is the single most common silent OCR truncation here.
        check=lambda value: len(value.strip()) >= 10,
        expected_format="a full street address of at least 10 characters",
    ),
    FieldRule(
        field="postal_code",
        check=lambda value: matches(MEXICAN_POSTAL_CODE_PATTERN, value),
        expected_format="a five digit postal code",
    ),
    FieldRule(
        field="issue_date",
        check=is_valid_date,
        expected_format="a calendar date such as 31/01/2026",
    ),
    FieldRule(
        field="service_provider",
        check=lambda value: len(value.strip()) >= 2,
        expected_format="a non empty service provider name",
        required=False,
    ),
    FieldRule(
        field="account_number",
        check=lambda value: 1 <= len(value.strip()) <= 40,
        expected_format="a non empty account number of at most 40 characters",
        required=False,
    ),
)


def check_proof_of_address_dates(data: Mapping[str, Any]) -> dict[str, str]:
    """Verify that the issue date is not in the future.

    Recency is a policy question that belongs to the caller, so no maximum age is
    enforced here; a date printed in the future, on the other hand, is always a
    misread and is exactly what a targeted re-read can fix.
    """
    issue_date = parse_date(data.get("issue_date"))
    if issue_date is None:
        return {}
    if issue_date > date.today():
        return {
            "issue_date": (
                f"The issue date {issue_date.isoformat()} is in the future, which is "
                f"impossible for an already issued document."
            )
        }
    return {}


ID_CARD_DOCUMENT_RULES: tuple[DocumentRule, ...] = (
    DocumentRule(name="birth_date_in_past", check=check_birth_date_in_past),
    DocumentRule(name="curp_birth_date_consistency", check=check_curp_matches_birth_date),
    DocumentRule(name="curp_sex_consistency", check=check_curp_matches_sex),
)

INVOICE_DOCUMENT_RULES: tuple[DocumentRule, ...] = (
    DocumentRule(name="invoice_totals", check=check_invoice_totals),
    DocumentRule(name="invoice_line_items", check=check_invoice_line_items),
    DocumentRule(name="invoice_dates", check=check_invoice_dates),
)

PROOF_OF_ADDRESS_DOCUMENT_RULES: tuple[DocumentRule, ...] = (
    DocumentRule(name="proof_of_address_dates", check=check_proof_of_address_dates),
)

FIELD_RULES_BY_DOC_TYPE: dict[str, tuple[FieldRule, ...]] = {
    "ine": ID_CARD_FIELD_RULES,
    "ife": ID_CARD_FIELD_RULES,
    "id": ID_CARD_FIELD_RULES,
    "idcard": ID_CARD_FIELD_RULES,
    "iddocument": ID_CARD_FIELD_RULES,
    # Dossier level tokens produced by the semantic clustering step.
    "inefront": ID_CARD_FIELD_RULES,
    "ineback": ID_CARD_FIELD_RULES,
    "inecombined": ID_CARD_FIELD_RULES,
    "invoice": INVOICE_FIELD_RULES,
    "factura": INVOICE_FIELD_RULES,
    "proofofaddress": PROOF_OF_ADDRESS_FIELD_RULES,
    "comprobantededomicilio": PROOF_OF_ADDRESS_FIELD_RULES,
    "utilitybill": PROOF_OF_ADDRESS_FIELD_RULES,
}

DOCUMENT_RULES_BY_DOC_TYPE: dict[str, tuple[DocumentRule, ...]] = {
    "ine": ID_CARD_DOCUMENT_RULES,
    "ife": ID_CARD_DOCUMENT_RULES,
    "id": ID_CARD_DOCUMENT_RULES,
    "idcard": ID_CARD_DOCUMENT_RULES,
    "iddocument": ID_CARD_DOCUMENT_RULES,
    "inefront": ID_CARD_DOCUMENT_RULES,
    "ineback": ID_CARD_DOCUMENT_RULES,
    "inecombined": ID_CARD_DOCUMENT_RULES,
    "invoice": INVOICE_DOCUMENT_RULES,
    "factura": INVOICE_DOCUMENT_RULES,
    "proofofaddress": PROOF_OF_ADDRESS_DOCUMENT_RULES,
    "comprobantededomicilio": PROOF_OF_ADDRESS_DOCUMENT_RULES,
    "utilitybill": PROOF_OF_ADDRESS_DOCUMENT_RULES,
}


def field_rules_for(doc_type: str) -> tuple[FieldRule, ...]:
    """Return the field level tools registered for a document type."""
    return FIELD_RULES_BY_DOC_TYPE.get(normalize_key(doc_type), ())


def document_rules_for(doc_type: str) -> tuple[DocumentRule, ...]:
    """Return the business rule tools registered for a document type."""
    return DOCUMENT_RULES_BY_DOC_TYPE.get(normalize_key(doc_type), ())


def expected_format_for(doc_type: str, field_name: str) -> Optional[str]:
    """Look up the documented format of a field, used to enrich the VLM prompt."""
    for rule in field_rules_for(doc_type):
        if rule.field == field_name:
            return rule.expected_format
    return None


def validate_document(doc_type: str, extracted_data: Mapping[str, Any]) -> dict[str, str]:
    """Run every registered tool for the document type.

    Returns a {field_name: error_message} mapping. An empty mapping means the
    document is valid and the graph can terminate successfully.
    """
    errors: dict[str, str] = {}

    for rule in field_rules_for(doc_type):
        message = rule.evaluate(extracted_data.get(rule.field))
        if message:
            errors[rule.field] = message

    for document_rule in document_rules_for(doc_type):
        for field_name, message in document_rule.evaluate(extracted_data).items():
            # Field level errors take precedence: they describe the root cause,
            # while a business rule only observes the symptom.
            errors.setdefault(field_name, message)

    return errors
