"""Typed contract for invoices and other commercial documents."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base import DocumentSchema, parse_decimal


class InvoiceLineItem(BaseModel):
    """A single billed line, used by the optional line level consistency rule."""

    model_config = ConfigDict(str_strip_whitespace=True)

    description: Optional[str] = None
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    amount: Optional[Decimal] = None

    @field_validator("quantity", "unit_price", "amount", mode="before")
    @classmethod
    def _coerce_amount(cls, raw_value: Any) -> Optional[Decimal]:
        """Reuse the shared OCR aware number parser for every numeric column."""
        return parse_decimal(raw_value)


class InvoiceSchema(DocumentSchema):
    """Canonical representation of an invoice.

    Monetary fields are typed as Decimal so that the arithmetic business rules
    (subtotal + tax == total) never suffer from binary floating point drift.
    Unparseable amounts collapse to None and are reported by the validation
    node, which is exactly the signal the targeted VLM retry needs.
    """

    # --- Header -------------------------------------------------------------
    invoice_id: Optional[str] = Field(default=None, description="Invoice or folio number.")
    invoice_date: Optional[str] = Field(default=None, description="Issue date as printed.")
    due_date: Optional[str] = Field(default=None)
    purchase_order: Optional[str] = Field(default=None)
    currency: Optional[str] = Field(default=None, description="ISO 4217 currency code.")

    # --- Parties ------------------------------------------------------------
    vendor_name: Optional[str] = Field(default=None)
    vendor_tax_id: Optional[str] = Field(default=None, description="Vendor RFC / VAT / EIN.")
    vendor_address: Optional[str] = Field(default=None)
    customer_name: Optional[str] = Field(default=None)
    customer_tax_id: Optional[str] = Field(default=None, description="Customer RFC / VAT / EIN.")
    customer_address: Optional[str] = Field(default=None)

    # --- Amounts ------------------------------------------------------------
    subtotal: Optional[Decimal] = Field(default=None, description="Net amount before tax.")
    tax_amount: Optional[Decimal] = Field(default=None, description="Total tax charged.")
    discount_amount: Optional[Decimal] = Field(default=None)
    total_amount: Optional[Decimal] = Field(default=None, description="Gross amount due.")

    # --- Detail -------------------------------------------------------------
    line_items: list[InvoiceLineItem] = Field(default_factory=list)

    ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "invoice_id": ("InvoiceId", "InvoiceNumber", "INVOICE_RECEIPT_ID", "folio", "numero_factura"),
        "invoice_date": ("InvoiceDate", "INVOICE_RECEIPT_DATE", "fecha_factura", "issue_date"),
        "due_date": ("DueDate", "DUE_DATE", "fecha_vencimiento"),
        "purchase_order": ("PurchaseOrder", "PO_NUMBER", "orden_compra"),
        "currency": ("CurrencyCode", "Currency", "moneda"),
        "vendor_name": ("VendorName", "VENDOR_NAME", "SupplierName", "emisor", "proveedor"),
        "vendor_tax_id": (
            "VendorTaxId",
            "VENDOR_VAT_NUMBER",
            "TAX_PAYER_ID",
            "rfc_emisor",
            "vendor_rfc",
        ),
        "vendor_address": ("VendorAddress", "VENDOR_ADDRESS", "domicilio_emisor"),
        "customer_name": ("CustomerName", "RECEIVER_NAME", "receptor", "cliente"),
        "customer_tax_id": (
            "CustomerTaxId",
            "RECEIVER_VAT_NUMBER",
            "rfc_receptor",
            "customer_rfc",
        ),
        "customer_address": ("CustomerAddress", "RECEIVER_ADDRESS", "domicilio_receptor"),
        "subtotal": ("SubTotal", "Subtotal", "SUBTOTAL", "importe_neto"),
        "tax_amount": ("TotalTax", "Tax", "TAX", "iva", "impuestos"),
        "discount_amount": ("TotalDiscount", "Discount", "DISCOUNT", "descuento"),
        "total_amount": ("InvoiceTotal", "Total", "TOTAL", "AmountDue", "total", "importe_total"),
        "line_items": ("Items", "LineItems", "conceptos"),
    }

    @field_validator("subtotal", "tax_amount", "discount_amount", "total_amount", mode="before")
    @classmethod
    def _coerce_amount(cls, raw_value: Any) -> Optional[Decimal]:
        """Convert OCR text such as "$ 1,234.56 MXN" into a Decimal."""
        return parse_decimal(raw_value)

    @field_validator("currency", "vendor_tax_id", "customer_tax_id", mode="before")
    @classmethod
    def _normalize_code(cls, raw_value: Optional[str]) -> Optional[str]:
        """Uppercase codes and remove the whitespace OCR sprinkles into them."""
        if raw_value is None:
            return None
        token = "".join(str(raw_value).split()).upper()
        return token or None

    @field_validator("line_items", mode="before")
    @classmethod
    def _coerce_line_items(cls, raw_value: Any) -> list[Any]:
        """Tolerate providers that return None or a single object instead of a list."""
        if raw_value is None:
            return []
        if isinstance(raw_value, (list, tuple)):
            return list(raw_value)
        return [raw_value]
